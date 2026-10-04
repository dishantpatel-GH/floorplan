"""Video tier v2: join metric scale segments and close loops with a fragment pose graph on predicted depth.

Problem (video_tier.md v1, Issue 13d): on long captures DPVO loses track on blank walls and restarts. Each piece
("segment") is metric on its own after the depth-agreement scale (scale.py), but DPVO's single step across the restart
decides where the next piece is placed, and that step is garbage. Rooms land metres away from where they belong.

Fix: the LiDAR tier's drift machinery (floorplan/recon/drift.py) with the metric MoGe-2 depth maps as "pseudo-LiDAR":
  1. fragments (~1.5 m of travel) are cut INSIDE each segment, never across a restart;
  2. odometry edges only inside a segment (DPVO relative pose, metric after the scale step, looser sigma than ARKit);
  3. across each restart a BRIDGE: point-to-plane ICP between the last fragments before and the first fragments after
     the restart (the camera filmed the same surfaces a second apart), started from "the camera did not jump";
     if no bridge verifies, a weak continuity edge keeps the graph connected and the boundary is reported unbridged;
  4. loop closures (ICP between non-consecutive fragments that overlap under the current estimate) and the plane
     anchors (Manhattan wall direction, one floor level) exactly as in drift.py;
  5. the same robust 4-DoF solver (yaw + translation; tilt comes from gravity, levelled per segment beforehand).
ICP gates are wider than for LiDAR because predicted depth is noisier (MoGe-2 AbsRel 2.8% after per-frame scale,
per-frame scale scatter p10-p90 0.82-1.03, v1 V-5): see video_graph_params.

Output: corrected keyframe poses, a report, and the honest consistency flag: the scene is one rigid map only if every
segment is connected to the others by VERIFIED geometry (an accepted bridge or loop that the solver kept).
"""
from __future__ import annotations

import time
from dataclasses import asdict, replace

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

from floorplan.recon import drift as dr

WEAK_TRANS_SIGMA_M = 0.50       # unbridged restart: "the camera moved less than ~0.5 m in the gap", nothing more
WEAK_YAW_SIGMA_DEG = 20.0


def video_graph_params(**overrides) -> dr.DriftParams:
    """drift.DriftParams with gates for learned depth instead of LiDAR (reasons in video_tier.md v2, V2-4)."""
    base = dict(min_conf=0, min_depth_m=0.2, max_depth_m=3.5,
                frag_travel_m=1.5, frag_max_s=5.0, frag_pixel_stride=4, frag_voxel_m=0.03, normal_radius_m=0.10,
                icp_voxels_m=(0.12, 0.06, 0.04), icp_max_corr_m=(0.40, 0.15, 0.08), icp_iters=(40, 30, 30),
                eval_corr_m=0.06, overlap_voxel_m=0.15, overlap_min=0.30, revisit_min_gap_s=20.0,
                loop_fitness_min=0.30, loop_plane_rmse_max_m=0.030, loop_max_jump_m=0.50, loop_max_jump_deg=10.0,
                min_constraint_frac=0.01, loop_rounds=2,
                odo_trans_sigma_m=0.04, odo_yaw_sigma_deg=1.0, yaw_smooth_sigma_deg=0.0,
                loop_trans_sigma_m=0.04, loop_yaw_sigma_deg=1.0, manhattan_sigma_deg=2.0,
                floor_sigma_m=0.03, floor_gate_m=0.15)
    base.update(overrides)
    return dr.DriftParams(**base)


def _backproject(depth: np.ndarray, K: np.ndarray, p: dr.DriftParams) -> np.ndarray:
    s = p.frag_pixel_stride
    v, u = np.mgrid[0:depth.shape[0]:s, 0:depth.shape[1]:s]
    d = depth[v, u]
    ok = (d > p.min_depth_m) & (d < p.max_depth_m)
    z = d[ok]
    return np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], 1)


def build_fragments(D: np.ndarray, K: np.ndarray, T: np.ndarray, ts: np.ndarray, seg: np.ndarray,
                    p: dr.DriftParams) -> tuple[list[dr.Fragment], np.ndarray]:
    """drift.build_fragments for predicted depth, cut inside each scale segment (never across a VO restart)."""
    frags: list[dr.Fragment] = []
    frag_seg = []
    for g in np.unique(seg):
        ids = np.where(seg == g)[0]
        for k, frames in enumerate(dr.split_fragments(ids, T, ts, p)):
            anchor = int(frames[len(frames) // 2])
            T_aw = np.linalg.inv(T[anchor])
            pts = []
            for i in frames:
                Ti = T_aw @ T[i]
                pts.append(_backproject(D[i], K, p) @ Ti[:3, :3].T + Ti[:3, 3])
            P = np.concatenate(pts) if pts else np.zeros((0, 3))
            if len(P) < 500:                    # blank or failed depth: no geometry to register (keeps its poses)
                continue
            cloud = dr._with_normals(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)), p.frag_voxel_m,
                                     p.normal_radius_m)
            cloud.orient_normals_towards_camera_location(np.zeros(3))
            travel = 0.0 if (not frags or k == 0) else float(np.linalg.norm(T[anchor, :3, 3] - frags[-1].pose[:3, 3]))
            frags.append(dr.Fragment(len(frags), frames, anchor, float(ts[anchor]), travel, T[anchor].copy(), cloud))
            frag_seg.append(int(g))
    return frags, np.asarray(frag_seg)


def _odometry_and_bridges(frags, frag_seg, p, bridge_pairs: int, log) -> list[dr.Edge]:
    edges = []
    for a, b in zip(frags[:-1], frags[1:]):
        if frag_seg[a.index] == frag_seg[b.index]:
            T = np.linalg.inv(b.pose) @ a.pose
            info, n, fit, rmse = dr.point_to_plane_stats(a.cloud, b.cloud, T, p.eval_corr_m)
            edges.append(dr.Edge(a.index, b.index, "odometry", T, info, n, fit, rmse))
            continue
        # restart between a and b: ICP bridges between the last/first `bridge_pairs` fragments of both sides
        before = [f for f in frags[max(0, a.index - bridge_pairs + 1): a.index + 1] if frag_seg[f.index] == frag_seg[a.index]]
        after = [f for f in frags[b.index: b.index + bridge_pairs] if frag_seg[f.index] == frag_seg[b.index]]
        n_ok = 0
        for s in before:
            for t in after:
                T0 = np.linalg.inv(t.pose) @ s.pose
                e = dr.register(s, t, T0, p, "bridge")
                e = dr.verify_loop(e, replace(p, loop_max_jump_m=1.0, loop_max_jump_deg=25.0))
                e.extra["boundary"] = float(frag_seg[b.index])
                edges.append(e)
                n_ok += e.accepted
        # always keep a weak continuity edge (camera did not teleport); verified bridges dominate it when present
        T = np.linalg.inv(b.pose) @ a.pose
        info, n, fit, rmse = dr.point_to_plane_stats(a.cloud, b.cloud, T, p.eval_corr_m)
        w = dr.Edge(a.index, b.index, "odometry", T, info, n, fit, rmse)
        w.extra["weak"] = 1.0
        edges.append(w)
        log(f"[graph] restart before segment {frag_seg[b.index]}: {n_ok}/{len(before) * len(after)} ICP bridges verified")
    return edges


def _solve(frags, edges, man, flo, p):
    used = [e for e in edges if e.kind == "odometry" or e.accepted]
    g = dr.build_graph(frags, used, man, flo, p)
    for k, e in enumerate(used):
        if e.extra.get("weak"):
            g.W_t[k] = np.eye(3) / WEAK_TRANS_SIGMA_M
            g.w_yaw[k] = 1 / np.radians(WEAK_YAW_SIGMA_DEG)
    X, out = dr.solve_4dof(g)
    for e, w in zip(used, out["edge_weights"]):
        e.weight = float(w)
    return X, out


def segment_connectivity(n_seg: int, frag_seg: np.ndarray, edges: list[dr.Edge], min_weight: float = 0.5) -> dict:
    """Segments joined by verified geometry (accepted bridge/loop edges the solver kept with weight >= min_weight)."""
    parent = list(range(n_seg))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    links = 0
    for e in edges:
        if e.kind == "odometry" or not e.accepted or e.weight < min_weight:
            continue
        a, b = int(frag_seg[e.source]), int(frag_seg[e.target])
        if a != b:
            links += 1
            parent[find(a)] = find(b)
    comps = len({find(a) for a in range(n_seg)})
    return dict(segments=n_seg, components=comps, cross_segment_links=links, connected=comps == 1)


def per_keyframe_correction(frags, frag_seg, X_new, seg, ts) -> np.ndarray:
    """World correction C_k (T_new = C_k T_old) for every keyframe, blended in time between anchors of its OWN
    segment (blending across a restart would smear a correct placement into the neighbour segment)."""
    C_all = np.tile(np.eye(4), (len(seg), 1, 1))
    for g in np.unique(seg):
        fs = [f for f in frags if frag_seg[f.index] == g]
        ids = np.where(seg == g)[0]
        if not fs:
            continue
        C = np.stack([X_new[f.index] @ np.linalg.inv(f.pose) for f in fs])
        ta = np.array([f.t_anchor for f in fs])
        if len(fs) == 1:
            C_all[ids] = C[0]
            continue
        tq = np.clip(ts[ids], ta[0], ta[-1])
        C_all[ids, :3, :3] = Slerp(ta, Rotation.from_matrix(C[:, :3, :3]))(tq).as_matrix()
        for a in range(3):
            C_all[ids, a, 3] = np.interp(tq, ta, C[:, a, 3])
    return C_all


def join_segments(D: np.ndarray, K: np.ndarray, T_lev: np.ndarray, ts: np.ndarray, seg: np.ndarray,
                  p: dr.DriftParams | None = None, bridge_pairs: int = 2, log=print) -> tuple[np.ndarray, dict]:
    """Pose graph over predicted-depth fragments. T_lev: metric, gravity-levelled (+y up) keyframe poses.

    Returns corrected keyframe poses and a JSON-serialisable report (fragments, bridges, loops, residuals,
    connectivity of the segments)."""
    p = p or video_graph_params()
    t0 = time.time()
    frags, frag_seg = build_fragments(D, K, T_lev, ts, seg, p)
    n_seg = int(seg.max()) + 1
    X0 = np.stack([f.pose for f in frags])
    if len(frags) < 2:
        return T_lev.copy(), dict(n_fragments=len(frags), skipped="fewer than 2 fragments",
                                  connectivity=segment_connectivity(n_seg, frag_seg, []))
    for f in frags:                             # per fragment: a fragment that sees no floor must not stop the run
        try:
            dr.measure_plane_anchors([f], T_lev, p)
        except ValueError:
            f.floor_y = None
    man, flo, ainfo = dr.anchor_sets(frags, p)
    edges = _odometry_and_bridges(frags, frag_seg, p, bridge_pairs, log)
    X, _ = _solve(frags, edges, man, flo, p)
    log(f"[graph] {len(frags)} fragments in {n_seg} segment(s); anchors {ainfo} ({time.time() - t0:.0f}s)")
    for r in range(p.loop_rounds):
        seen = {(e.source, e.target) for e in edges}
        new = dr.loop_edges(frags, X, p, seen)
        edges += new
        X, _ = _solve(frags, edges, man, flo, p)
        log(f"[graph] round {r + 1}: {len(new)} loop candidates, {sum(e.accepted for e in new)} accepted "
            f"({time.time() - t0:.0f}s)")
    C = per_keyframe_correction(frags, frag_seg, X, seg, ts)
    T_new = C @ T_lev
    loops = [e for e in edges if e.kind not in ("odometry",)]
    used = [e for e in loops if e.accepted]
    conn = segment_connectivity(n_seg, frag_seg, edges)
    res = dr.residual_table([e for e in used if e.kind != "bridge"], dict(before=X0, after=X), frags)
    bridge_res = [dr.edge_residual(e, X, frags) for e in used if e.kind == "bridge"]
    report = dict(
        n_fragments=len(frags), n_segments=n_seg, anchors=dict(ainfo, manhattan_used=len(man), floor_used=len(flo)),
        bridges=dict(tried=sum(e.kind == "bridge" for e in edges), accepted=sum(e.kind == "bridge" and e.accepted
                                                                                   for e in edges),
                     kept=sum(e.kind == "bridge" and e.accepted and e.weight >= 0.5 for e in edges),
                     residual_surface_sep_cm=dr.summarise(r["surface_sep_cm"] for r in bridge_res)),
        loops=dict(candidates=sum(e.kind in ("local", "revisit") for e in loops),
                   accepted=sum(e.kind in ("local", "revisit") and e.accepted for e in loops),
                   revisit_accepted=sum(e.kind == "revisit" and e.accepted for e in loops),
                   down_weighted=sum(e.kind in ("local", "revisit") and e.accepted and e.weight < 0.25 for e in loops)),
        loop_residuals=res, connectivity=conn,
        anchor_spread=dict(before=dr.anchor_spread(frags, X0, man, flo), after=dr.anchor_spread(frags, X, man, flo)),
        correction=dr.frame_correction_stats(T_lev, T_new),
        rejection_reasons=_reasons(loops), runtime_s=round(time.time() - t0, 1),
        params=asdict(p))
    return T_new, report


def _reasons(edges) -> dict:
    out: dict[str, int] = {}
    for e in edges:
        for r in filter(None, e.reason.split("; ")):
            out[r.split(" ")[0]] = out.get(r.split(" ")[0], 0) + 1
    return out
