#!/usr/bin/env python
"""Explore one Stray Scanner capture before designing anything (decision D-001 in docs/DECISIONS.md).

Questions this script answers, with evidence written to outputs/explore/<capture>/:
  1. Pose convention: are the odometry.csv poses camera-to-world in ARKit camera axes (x right, y up, z back)
     or in OpenCV axes (x right, y down, z forward)? We test both by re-projecting one frame's depth into a
     later frame and measuring the depth residual. The right convention gives ~1-2 cm, the wrong one metres.
  2. Gravity axis: ARKit's world frame should be y-up. We check the floor-plane normal of the fused cloud.
  3. Content: trajectory, top-down density and a horizontal slice show rooms, doors and loop closures.
  4. Quality: confidence fractions, timestamp gaps (tracking hiccups), a contact sheet of RGB frames
     (glass, mirrors, low light, damage).

Usage:  python scripts/explore_capture.py <capture_dir> [--every 5] [--voxel 0.03]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pandas as pd
from scipy.spatial.transform import Rotation

DEPTH_W, DEPTH_H = 256, 192
F_ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # ARKit camera (y up, z back) -> OpenCV camera (y down, z fwd)


def load_capture(cap_dir: Path):
    od = pd.read_csv(cap_dir / "odometry.csv", skipinitialspace=True)
    od.columns = [c.strip() for c in od.columns]
    T_wc = np.tile(np.eye(4), (len(od), 1, 1))
    T_wc[:, :3, :3] = Rotation.from_quat(od[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    T_wc[:, :3, 3] = od[["x", "y", "z"]].to_numpy()
    K_rgb = np.loadtxt(cap_dir / "camera_matrix.csv", delimiter=",")
    depth_files = sorted((cap_dir / "depth").glob("*.png"))
    conf_files = sorted((cap_dir / "confidence").glob("*.png"))
    return dict(od=od, T_wc=T_wc, K_rgb=K_rgb, depth_files=depth_files, conf_files=conf_files,
                t=od["timestamp"].to_numpy(), video=cap_dir / "rgb.mp4")


def depth_K(od_row, rgb_w=1920):
    """Depth intrinsics: ARKit's per-frame RGB intrinsics scaled to the 256x192 depth map."""
    s = DEPTH_W / rgb_w
    return np.array([[od_row["fx"] * s, 0, od_row["cx"] * s], [0, od_row["fy"] * s, od_row["cy"] * s], [0, 0, 1.0]])


def backproject(depth_m, K, stride=4):
    v, u = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    z = depth_m[v, u]
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)


def read_depth(cap, i, min_conf=2):
    d = cv2.imread(str(cap["depth_files"][i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    c = cv2.imread(str(cap["conf_files"][i]), cv2.IMREAD_UNCHANGED)
    d[c < min_conf] = 0
    return d


def convention_residual(cap, conv, gaps=(15, 45), n_pairs=40):
    """Median |predicted depth - measured depth| when frame i's points are re-projected into frame i+gap."""
    F = F_ARKIT_TO_CV if conv == "arkit" else np.eye(4)
    N = len(cap["depth_files"])
    res = []
    rng = np.random.default_rng(0)
    for gap in gaps:
        for i in rng.choice(N - gap, size=min(n_pairs, N - gap), replace=False):
            j = i + gap
            Ki, Kj = depth_K(cap["od"].iloc[i]), depth_K(cap["od"].iloc[j])
            P = backproject(read_depth(cap, i), Ki)
            if len(P) < 200:
                continue
            Twi, Twj = cap["T_wc"][i] @ F, cap["T_wc"][j] @ F
            Pw = P @ Twi[:3, :3].T + Twi[:3, 3]
            Tjw = np.linalg.inv(Twj)
            Pj = Pw @ Tjw[:3, :3].T + Tjw[:3, 3]
            Pj = Pj[Pj[:, 2] > 0.1]
            u = np.round(Kj[0, 0] * Pj[:, 0] / Pj[:, 2] + Kj[0, 2]).astype(int)
            v = np.round(Kj[1, 1] * Pj[:, 1] / Pj[:, 2] + Kj[1, 2]).astype(int)
            ok = (u >= 0) & (u < DEPTH_W) & (v >= 0) & (v < DEPTH_H)
            dj = read_depth(cap, j)
            meas = dj[v[ok], u[ok]]
            m = meas > 0
            if m.sum() < 100:
                continue
            res.append(np.median(np.abs(meas[m] - Pj[ok][m][:, 2])))
    return float(np.median(res)) if res else float("nan"), len(res)


def fuse(cap, conv, every=5, voxel=0.03, depth_max=4.0, with_color=True):
    F = F_ARKIT_TO_CV if conv == "arkit" else np.eye(4)
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
                                        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                                        attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=8,
                                        block_count=60000, device=o3c.Device("CPU:0"))
    vid = cv2.VideoCapture(str(cap["video"]))
    N = len(cap["depth_files"])
    used = 0
    for i in range(N):
        if i % every:
            vid.grab()
            continue
        ok, bgr = vid.read()
        d = read_depth(cap, i)
        d_mm = (d * 1000).astype(np.uint16)
        if (d_mm > 0).sum() < 500:
            continue
        K = depth_K(cap["od"].iloc[i])
        rgb = cv2.resize(bgr, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy() if ok \
            else np.zeros((DEPTH_H, DEPTH_W, 3), np.uint8)
        E = np.linalg.inv(cap["T_wc"][i] @ F)  # world -> camera (Open3D "extrinsic")
        Kt, Et = o3c.Tensor(K, o3c.float64), o3c.Tensor(E, o3c.float64)
        dt, ct = o3d.t.geometry.Image(o3c.Tensor(d_mm)), o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb)))
        blocks = vbg.compute_unique_block_coordinates(dt, Kt, Et, 1000.0, depth_max, 4.0)
        vbg.integrate(blocks, dt, ct, Kt, Kt, Et, 1000.0, depth_max, 4.0)
        used += 1
    pcd = vbg.extract_point_cloud(weight_threshold=2.0).to_legacy()
    return pcd, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "explore")
    a = ap.parse_args()
    name = a.capture.parent.name + "__" + a.capture.name
    out = a.out / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cap = load_capture(a.capture)
    N = len(cap["depth_files"])
    stats = dict(capture=str(a.capture), frames=N, duration_s=float(cap["t"][-1] - cap["t"][0]))

    # 1. pose convention
    conv_res = {c: convention_residual(cap, c) for c in ("arkit", "opencv")}
    stats["convention_residual_m"] = {c: r[0] for c, r in conv_res.items()}
    conv = min(conv_res, key=lambda c: conv_res[c][0])
    stats["convention"] = conv

    # timing / tracking
    dt = np.diff(cap["t"])
    stats["fps_median"] = float(1 / np.median(dt))
    stats["timestamp_gaps_over_3_frames"] = int((dt > 3 * np.median(dt)).sum())
    stats["max_gap_s"] = float(dt.max())

    # confidence fractions (sampled)
    fr = []
    for i in np.linspace(0, N - 1, 60).astype(int):
        c = cv2.imread(str(cap["conf_files"][i]), cv2.IMREAD_UNCHANGED)
        fr.append([(c == k).mean() for k in (0, 1, 2)])
    stats["confidence_fraction_low_med_high"] = np.round(np.mean(fr, 0), 3).tolist()

    # 2./3. fuse, gravity, figures
    pcd, used = fuse(cap, conv, every=a.every, voxel=a.voxel)
    stats["frames_fused"] = used
    P = np.asarray(pcd.points)
    C = np.asarray(pcd.colors)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * a.voxel, max_nn=30))
    Nrm = np.asarray(pcd.normals)
    o3d.io.write_point_cloud(str(out / "fused.ply"), pcd)
    stats["points"] = len(P)
    stats["extent_xyz_m"] = np.round(P.max(0) - P.min(0), 2).tolist()

    horiz = np.abs(Nrm[:, 1]) > 0.9  # surfaces whose normal is (anti)parallel to world y: floor / ceiling
    y = P[horiz, 1]
    hist, edges = np.histogram(y, bins=np.arange(y.min() - 0.05, y.max() + 0.05, 0.02))
    centers = (edges[:-1] + edges[1:]) / 2
    floor_y = centers[np.argmax(hist * (centers < np.median(P[:, 1])))]
    above = centers > floor_y + 1.5
    ceil_y = centers[above][np.argmax(hist[above])] if above.any() and hist[above].max() > 0.05 * hist.max() else None
    stats["floor_y"] = float(floor_y)
    stats["ceiling_y"] = float(ceil_y) if ceil_y is not None else None
    stats["ceiling_height_rough_m"] = float(ceil_y - floor_y) if ceil_y is not None else None
    # gravity check: RANSAC plane on near-floor points
    fl = pcd.select_by_index(np.where(horiz & (np.abs(P[:, 1] - floor_y) < 0.05))[0])
    if len(fl.points) > 100:
        plane, _ = fl.segment_plane(0.02, 3, 500)
        n = np.array(plane[:3]) / np.linalg.norm(plane[:3])
        stats["floor_normal"] = np.round(n, 4).tolist()
        stats["floor_tilt_deg"] = float(np.degrees(np.arccos(abs(n[1]))))

    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    tr = cap["T_wc"][:, :3, 3]
    sc = ax[0].scatter(tr[:, 0], tr[:, 2], c=cap["t"] - cap["t"][0], s=2, cmap="viridis")
    ax[0].plot(tr[0, 0], tr[0, 2], "g^", ms=12, label="start")
    ax[0].plot(tr[-1, 0], tr[-1, 2], "rs", ms=12, label="end")
    ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_title("camera path (x-z, top-down); colour = time s")
    plt.colorbar(sc, ax=ax[0])
    wall = (np.abs(Nrm[:, 1]) < 0.3) & (P[:, 1] > floor_y + 0.3) & (P[:, 1] < (ceil_y if ceil_y else floor_y + 2.2) - 0.3)
    ax[1].hist2d(P[wall, 0], P[wall, 2], bins=[np.arange(P[:, 0].min(), P[:, 0].max(), 0.03),
                                              np.arange(P[:, 2].min(), P[:, 2].max(), 0.03)], cmap="gray_r", cmin=1)
    ax[1].plot(tr[:, 0], tr[:, 2], "r-", lw=0.5)
    ax[1].set_aspect("equal"); ax[1].set_title("top-down density of wall-like points (vertical normals) + path")
    sl = (P[:, 1] > floor_y + 1.0) & (P[:, 1] < floor_y + 1.5)
    ax[2].scatter(P[sl, 0], P[sl, 2], s=0.3, c=C[sl])
    ax[2].set_aspect("equal"); ax[2].set_title("slice 1.0-1.5 m above floor (RGB)")
    for a_ in ax:
        a_.set_xlabel("x (m)"); a_.set_ylabel("z (m)"); a_.invert_yaxis()
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centers, hist, width=0.02)
    ax.axvline(floor_y, color="g"); ax.set_title("height histogram of horizontal surfaces (y, m)")
    if ceil_y is not None:
        ax.axvline(ceil_y, color="r")
    fig.tight_layout(); fig.savefig(out / "height_hist.png", dpi=100); plt.close(fig)

    # 4. contact sheet
    vid = cv2.VideoCapture(str(cap["video"]))
    idx = np.linspace(0, N - 1, 16).astype(int)
    tiles = []
    for k in idx:
        vid.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = vid.read()
        if ok:
            f = cv2.resize(f, (480, 360))
            cv2.putText(f, f"#{k} t={cap['t'][k]-cap['t'][0]:.0f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles.append(f)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(len(tiles) // 4)])
    cv2.imwrite(str(out / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])

    stats["runtime_s"] = round(time.time() - t0, 1)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
