"""Dense fusion (D-007) and raw-point collection.

* fuse_tsdf: Open3D tensor VoxelBlockGrid TSDF. It averages many noisy LiDAR observations into one surface. Normals
  come from the TSDF gradient and point toward the side the camera observed from, which is "into the room" for walls.
  We use the tensor API because Open3D 0.20's legacy ScalableTSDFVolume silently returns an empty volume
  (found during environment setup, SETUP.md section 6.7).
* collect_raw_points: the filtered LiDAR points themselves (no averaging). Final dimensions are fitted on these, so
  TSDF smoothing at corners cannot bias them.
"""
from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

from floorplan.io.stray import DEPTH_H, DEPTH_W


def fuse_tsdf(cap, frame_ids, T_wc, cfg, with_color: bool = True, device: str = "CPU:0"):
    dev = o3c.Device(device)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=cfg.voxel_m, block_resolution=8, block_count=100000, device=dev)
    cw = cfg.color_width
    ch = int(round(cw * cap.rgb_size[1] / cap.rgb_size[0]))
    frames = {}
    if with_color:
        for i, bgr in cap.iter_rgb(frame_ids):
            frames[i] = np.ascontiguousarray(cv2.resize(bgr, (cw, ch), interpolation=cv2.INTER_AREA)[:, :, ::-1])
    used = 0
    for i in frame_ids:
        d = cap.depth(i, cfg.min_conf, cfg.min_depth_m, cfg.max_depth_m)
        d_mm = np.round(d * 1000).astype(np.uint16)
        if (d_mm > 0).sum() < 500:
            continue
        Kd = cap.K_depth(i)
        Kc = cap.K_rgb[i].copy()
        Kc[:2] *= cw / cap.rgb_size[0]
        E = np.linalg.inv(T_wc[i])                         # world -> camera ("extrinsic" in Open3D)
        dt = o3d.t.geometry.Image(o3c.Tensor(d_mm)).to(dev)
        rgb = frames.get(i)
        if rgb is None:
            rgb, Kc = np.zeros((DEPTH_H, DEPTH_W, 3), np.uint8), Kd
        ct = o3d.t.geometry.Image(o3c.Tensor(rgb)).to(dev)
        Kdt, Kct, Et = (o3c.Tensor(Kd, o3c.float64), o3c.Tensor(Kc, o3c.float64), o3c.Tensor(E, o3c.float64))
        blocks = vbg.compute_unique_block_coordinates(dt, Kdt, Et, 1000.0, cfg.max_depth_m, cfg.trunc_mult)
        vbg.integrate(blocks, dt, ct, Kdt, Kct, Et, 1000.0, cfg.max_depth_m, cfg.trunc_mult)
        used += 1
    tp = vbg.extract_point_cloud(weight_threshold=cfg.weight_threshold)
    pts = tp.point.positions.cpu().numpy().astype(np.float32)
    nrm = tp.point.normals.cpu().numpy().astype(np.float32) if "normals" in tp.point else None
    col = tp.point.colors.cpu().numpy() if "colors" in tp.point else np.zeros_like(pts)
    col = np.clip(col * (255.0 if col.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if nrm is None:  # fallback: PCA normals (unoriented)
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * cfg.voxel_m, max_nn=30))
        nrm = np.asarray(pc.normals, np.float32)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    # Deterministic order: the voxel hash map returns points in a run-dependent order, which made downstream
    # floating-point sums (scene alignment) differ at the 1e-10 level between runs (stability work, outputs/stability).
    order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    pts, nrm, col = pts[order], nrm[order], col[order]
    return dict(points=pts, normals=nrm, colors=col, frames_used=used, vbg=vbg)


def collect_raw_points(cap, frame_ids, T_wc, cfg):
    """High-confidence LiDAR points in world coordinates, de-duplicated on a fine grid.

    Returns points (M,3) float32 and, per point, the range from the camera (used by the noise model), the
    viewing-ray direction (used to tell which side of a wall the camera was on) and the source frame index
    (used to measure a room from a single visit, so two passes' residual drift is not mixed)."""
    s = cfg.raw_stride
    v, u = np.mgrid[0:DEPTH_H:s, 0:DEPTH_W:s]
    P_all, R_all, D_all, F_all = [], [], [], []
    for i in frame_ids:
        d = cap.depth(i, cfg.min_conf, cfg.min_depth_m, cfg.max_depth_m)[v, u]
        ok = d > 0
        if ok.sum() < 50:
            continue
        K = cap.K_depth(i)
        z = d[ok]
        pc = np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], 1)
        T = T_wc[i]
        pw = pc @ T[:3, :3].T + T[:3, 3]
        ray = pw - T[:3, 3]
        P_all.append(pw.astype(np.float32))
        R_all.append(np.linalg.norm(pc, axis=1).astype(np.float32))
        D_all.append((ray / np.linalg.norm(ray, axis=1, keepdims=True)).astype(np.float16))
        F_all.append(np.full(len(pw), i, dtype=np.int32))
    P = np.concatenate(P_all)
    Fr = np.concatenate(F_all)
    R = np.concatenate(R_all)
    D = np.concatenate(D_all)
    # de-duplicate on a fine grid: keep the closest-range observation per cell (least noisy)
    k = np.floor((P - P.min(0)) / cfg.raw_voxel_m).astype(np.int64)        # non-negative cell indices
    key = k[:, 0] + (k[:, 1] << 21) + (k[:, 2] << 42)                      # pack 3 indices into one int64
    order = np.argsort(R, kind="stable")
    _, first = np.unique(key[order], return_index=True)
    sel = order[first]
    return dict(points=P[sel], range=R[sel], ray=D[sel], frame=Fr[sel])
