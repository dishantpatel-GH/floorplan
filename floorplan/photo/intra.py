"""Extra edges INSIDE a room when feature matching found none: MapAnything proposes, the depth maps verify (P-2c).

Why: the protocol's photos look across the room in different directions, so two photos of the same room often share
too few feature matches (white walls, wide baselines). MapAnything was measured to be unreliable as the SOURCE of
poses here (relative rotation errors of 5-87 deg median per room, docs/modules/photo_tier.md), but it is right often
enough to be a useful PROPOSAL. Each proposed relative pose is:
  1. reduced to the 4 degrees of freedom gravity allows (yaw + shift; tilt disagreement > 10 deg = rejected),
  2. refined by point-to-plane ICP between the two photos' MoGe-2 point clouds (only overlap can do this),
  3. accepted only if the dense free-space check passes AND the photos demonstrably overlap (agreement).
A proposal that cannot be verified (no overlap) is discarded: an unverifiable edge is a guess, and guesses are what
the fallback placement is for, with its widened intervals.
"""
from __future__ import annotations

import numpy as np

from floorplan.photo.link import Edge, _ry, free_space_violation
from floorplan.photo.recon import MapAnythingRunner


def _yaw_of(M: np.ndarray) -> tuple[float, float]:
    """Nearest rotation about +y to M, and M's tilt (deg) away from a pure yaw."""
    th = float(np.arctan2(M[0, 2] - M[2, 0], M[0, 0] + M[2, 2]))
    tilt = float(np.degrees(np.arccos(np.clip(M[1, 1], -1, 1))))
    return th, tilt


def _levelled_cloud(v, stride: int = 3):
    Pc, idx = v.points_cam(stride)
    N = v.normal_cam.reshape(-1, 3)[idx]
    return Pc @ v.R_lev.T, N @ v.R_lev.T


def icp_refine(va, vb, th: float, t: np.ndarray, max_dist: float = 0.2) -> tuple[float, np.ndarray, float]:
    """Point-to-plane ICP of b's levelled cloud onto a's, starting at (th, t); result projected back to yaw + shift."""
    import open3d as o3d
    A, NA = _levelled_cloud(va)
    B, _ = _levelled_cloud(vb)
    pa = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(A))
    pa.normals = o3d.utility.Vector3dVector(NA)
    pb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(B))
    T0 = np.eye(4)
    T0[:3, :3], T0[:3, 3] = _ry(th), t
    reg = o3d.pipelines.registration.registration_icp(
        pb, pa, max_dist, T0, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    th2, _ = _yaw_of(reg.transformation[:3, :3])
    return th2, np.asarray(reg.transformation[:3, 3]).copy(), float(reg.fitness)


def mapanything_edges(views: dict, existing: list[Edge], params, log=print) -> tuple[list[Edge], dict]:
    from floorplan.photo.recon import crop_for_model
    have = {frozenset((e.a, e.b)) for e in existing}
    rooms: dict[str, list[str]] = {}
    for n, v in views.items():
        rooms.setdefault(v.room, []).append(n)
    stats = dict(proposed=0, tilt=0, unverifiable=0, violation=0, accepted=0)
    out = []
    runner = MapAnythingRunner()
    try:
        for room, names in sorted(rooms.items()):
            names = sorted(names)
            if len(names) < 2:
                continue
            geo = runner.reconstruct(names, [views[n].rgb for n in names], [views[n].K[0, 0] for n in names])
            G = {g.name: g for g in geo}
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if frozenset((a, b)) in have:
                        continue
                    stats["proposed"] += 1
                    out_e = _verify(views[a], views[b], G[a], G[b], params, stats)
                    if out_e is not None:
                        out.append(out_e)
    finally:
        runner.close()
    log(f"[photo/intra] MapAnything proposals inside rooms: {stats}")
    return out, stats


def _verify(va, vb, ga, gb, params, stats) -> Edge | None:
    T_ab = np.linalg.inv(ga.T_wc) @ gb.T_wc                     # camera b -> camera a (MapAnything units)
    M = va.R_lev @ T_ab[:3, :3] @ vb.R_lev.T
    th, tilt = _yaw_of(M)
    if tilt > params.intra_max_tilt_deg:
        stats["tilt"] += 1
        return None
    da, db = ga.depth[ga.depth > 0], va.depth[va.depth > 0]
    k = float(np.median(db) / np.median(da)) if len(da) and len(db) else 1.0   # MapAnything units -> MoGe metres
    t = k * va.R_lev @ T_ab[:3, 3]
    th, t, fit = icp_refine(va, vb, th, t)
    v_ab = free_space_violation(va, vb, th, 1.0, t)
    v_ba = free_space_violation(vb, va, -th, 1.0, -_ry(-th) @ t)
    n_cmp = min(v_ab[2], v_ba[2])
    agree = max(v_ab[1], v_ba[1])
    if n_cmp < params.intra_min_overlap_pts or agree < params.intra_min_agreement:
        stats["unverifiable"] += 1
        return None
    viol = max(v_ab[0], v_ba[0])
    if viol > params.edge_max_violation:
        stats["violation"] += 1
        return None
    stats["accepted"] += 1
    return Edge(va.name, vb.name, th, t, 0.0, params.intra_edge_weight, 0, float("nan"), False, viol,
                min(v_ab[1], v_ba[1]))
