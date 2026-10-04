"""Drift correction: a 4-DoF fragment pose graph with ICP loop closures and plane anchors.

Why this module exists
  ARKit's visual-inertial odometry (VIO) is very accurate over a few metres, but it is dead reckoning: small errors
  add up along the walk. When the camera returns to a room it has already seen, the second observation of a wall can
  land centimetres (or degrees) away from the first. On a stitched plan that shows up as doubled walls, rooms that do
  not close, and room-to-room offsets. The case study's "Drift accountability" gate requires us to do something about
  it ("poses used as-is" is an automatic fail) and to show an on/off ablation.

The idea in one paragraph
  Trust ARKit locally, correct it globally. We cut the walk into short "fragments" (~1.5 m of travel), each a small
  rigid point cloud made of raw high-confidence LiDAR points (Choi, Zhou & Koltun, CVPR 2015). Each fragment is one
  node of a pose graph. Three kinds of evidence pull on the nodes:
    1. odometry edges: ARKit's relative pose between consecutive fragments (locally accurate, so a stiff spring);
    2. loop closures: fragments that see the same geometry at different times, aligned by point-to-plane ICP;
    3. plane anchors: every fragment's walls should share one Manhattan direction (fixes heading drift) and its floor
       should sit at the one global floor level (fixes vertical drift).
  Only x, y, z and yaw are corrected (4 DoF). Roll and pitch are observed directly by ARKit's accelerometer (gravity),
  so they do not drift; letting a solver change them only lets ICP noise tilt the map (VINS-Mono, Qin et al. 2018,
  uses the same 4-DoF pose graph for the same reason). Outlier loops and anchors are down-weighted by iteratively
  re-weighted least squares with a Cauchy kernel (the role Open3D's "line process" plays in Choi et al.). The
  per-fragment corrections are then blended smoothly over every frame.

Conventions
  * Pose = camera-to-world T_wc, OpenCV camera axes, world +y up (docs/DATA_NOTES.md).
  * Edge (s, t, T_st): p_t = T_st p_s, i.e. T_st = X_t^-1 X_s with X = fragment-anchor camera-to-world.
  * Correction of fragment k: yaw psi_k about the vertical axis through the anchor, then a shift d_k:
    R'_k = Ry(psi_k) R_k, c'_k = c_k + d_k. Ry(psi) turns a wall-normal angle atan2(n_z, n_x) from theta to theta-psi.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from floorplan.io.stray import DEPTH_H, DEPTH_W
from floorplan.plan.align import estimate_floor, manhattan_yaw

REG = o3d.pipelines.registration
RESIDUAL_KEYS = ("surface_sep_cm", "mean_disp_cm", "plan_disp_cm", "vert_disp_cm", "rot_deg")


@dataclass(frozen=True)
class DriftParams:
    """Every tunable of the drift module. The reason for each default is in docs/modules/drift.md (Decisions)."""
    # depth filtering: same as the rest of the pipeline (D-006)
    min_conf: int = 2
    min_depth_m: float = 0.2
    max_depth_m: float = 4.0
    # fragments: short enough that ARKit is locally exact, long enough to hold several walls for a well-posed ICP
    frag_travel_m: float = 1.5
    frag_max_s: float = 5.0
    frag_pixel_stride: int = 4
    frag_voxel_m: float = 0.02
    normal_radius_m: float = 0.06
    # registration: coarse-to-fine point-to-plane ICP from the current pose estimate
    icp_voxels_m: tuple[float, ...] = (0.08, 0.04, 0.02)
    icp_max_corr_m: tuple[float, ...] = (0.20, 0.08, 0.04)
    icp_iters: tuple[int, ...] = (40, 30, 30)
    eval_corr_m: float = 0.03
    # loop-closure candidates and verification
    overlap_voxel_m: float = 0.05
    overlap_min: float = 0.30
    revisit_min_gap_s: float = 20.0
    loop_fitness_min: float = 0.30
    loop_plane_rmse_max_m: float = 0.010
    loop_max_jump_m: float = 0.30
    loop_max_jump_deg: float = 5.0
    min_constraint_frac: float = 0.01
    loop_rounds: int = 2
    icp_workers: int = min(8, os.cpu_count() or 1)   # concurrent single-threaded ICPs (results do not depend on it)
    # what the graph uses (switches exist for the ablation)
    solver: str = "4dof"                   # 4dof (ours) | open3d6 (Open3D 6-DoF LM + line process, Choi et al.)
    use_loops: bool = True
    use_plane_anchors: bool = True
    refine_odometry: bool = False          # open3d6 only: ICP-refine the odometry edges as in Choi et al.
    # noise model (1 sigma). ARKit drift grows like a random walk with distance travelled.
    odo_trans_sigma_m: float = 0.010       # per sqrt(metre) of travel
    odo_yaw_sigma_deg: float = 0.60        # per sqrt(metre) of travel (calibrated by held-out loops, see doc)
    yaw_smooth_sigma_deg: float = 0.03     # change of yaw-drift rate between consecutive fragments (gyro bias is
                                           # slow, so heading drift is a smooth ramp, not jitter); 0 = off
    loop_trans_sigma_m: float = 0.015      # ICP repeatability on LiDAR fragments (measured, see doc)
    loop_yaw_sigma_deg: float = 0.40
    manhattan_sigma_deg: float = 1.0       # one fragment's dominant wall direction
    manhattan_min_score: float = 0.5       # fragment counts as Manhattan if >= 50% of wall normals agree
    manhattan_min_frac: float = 0.5        # anchors on only if >= 50% of fragments are Manhattan
    floor_sigma_m: float = 0.010
    floor_gate_m: float = 0.10             # floor readings further than this from the global floor are ignored
    robust_c: float = 2.0                  # Cauchy kernel scale, in sigmas
    irls_iters: int = 8
    # Open3D line process (open3d6 only)
    prune_corr_m: float = 0.03
    edge_prune_threshold: float = 0.25
    preference_loop_closure: float = 1.0

    @staticmethod
    def from_config(cfg, **overrides) -> "DriftParams":
        """Take depth filtering from the shared Config so all modules see the same points."""
        return replace(DriftParams(min_conf=cfg.min_conf, min_depth_m=cfg.min_depth_m,
                                   max_depth_m=cfg.max_depth_m), **overrides)


@dataclass
class Fragment:
    index: int
    frames: np.ndarray                     # keyframe indices in this fragment
    anchor: int                            # frame whose camera defines the fragment's local frame
    t_anchor: float
    travel_m: float                        # camera travel from the previous anchor (odometry noise grows with it)
    pose: np.ndarray                       # ARKit pose of the anchor (fragment-local -> world)
    cloud: o3d.geometry.PointCloud         # voxelised raw LiDAR points, normals oriented to the anchor camera
    floor_y: float | None = None           # plane anchor: floor level seen by this fragment (world y)
    wall_theta: float | None = None        # plane anchor: dominant wall-normal angle mod 90 deg (rad)
    manhattan_score: float = 0.0
    ceiling_y: float | None = None         # highest ceiling level seen (world y); a CHECK only, never a prior


@dataclass
class Edge:
    source: int
    target: int
    kind: str                              # odometry | local | revisit
    T: np.ndarray                          # measured T_st
    info: np.ndarray                       # 6x6 point-to-plane information (rx, ry, rz, tx, ty, tz)
    n_corr: int = 0
    fitness: float = 1.0
    plane_rmse_m: float = 0.0
    jump_m: float = 0.0                    # how far ICP moved from its initial guess
    jump_deg: float = 0.0
    constraint_frac: float = 1.0           # weakest translation direction / correspondences (1/3 = isotropic)
    accepted: bool = True
    reason: str = ""
    weight: float = 1.0                    # final robust weight in the solve (0..1)
    extra: dict = field(default_factory=dict)


# ============================================================================================ small helpers
def rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R[:3, :3]) - 1) / 2, -1, 1))))


def _ry(psi: np.ndarray) -> np.ndarray:
    """Rotations about +y, vectorised: (n,) -> (n, 3, 3)."""
    c, s = np.cos(psi), np.sin(psi)
    R = np.zeros((len(psi), 3, 3))
    R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s, 1.0, -s, c
    return R


def yaw_angle(R: np.ndarray) -> float:
    """Yaw (about +y) of a near-vertical-axis rotation, consistent with _ry."""
    return float(np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2]))


def _wrap(a, period: float):
    """Wrap angle(s) to (-period/2, period/2]."""
    return (np.asarray(a) + period / 2) % period - period / 2


@contextmanager
def deterministic_open3d():
    """Run Open3D on one thread inside this block, so ICP is bit-reproducible.

    Open3D (>= 0.19) parallelises with oneTBB. Its ICP sums the point-to-plane normal equations (J^T J, J^T r) with a
    parallel reduction whose split/join order depends on thread scheduling, so floating-point rounding differs from
    run to run (measured: edge transforms differ by ~4e-15). That noise is then amplified by the pose-graph solve and
    by the plan extractor (docs/modules/drift.md, Part "Determinism"). One thread makes the summation order fixed."""
    prev = o3d.utility.get_max_threads()
    o3d.utility.set_max_threads(1)
    try:
        yield
    finally:
        o3d.utility.set_max_threads(prev)


def _with_normals(pc: o3d.geometry.PointCloud, voxel: float, radius: float) -> o3d.geometry.PointCloud:
    d = pc.voxel_down_sample(voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return d


# ============================================================================================ 1. fragments
def split_fragments(kf: np.ndarray, T_wc: np.ndarray, timestamps: np.ndarray, p: DriftParams) -> list[np.ndarray]:
    """Cut the keyframe sequence whenever the camera has travelled frag_travel_m or frag_max_s has passed."""
    groups, cur, travel = [], [int(kf[0])], 0.0
    for a, b in zip(kf[:-1], kf[1:]):
        travel += float(np.linalg.norm(T_wc[b, :3, 3] - T_wc[a, :3, 3]))
        if travel >= p.frag_travel_m or timestamps[b] - timestamps[cur[0]] >= p.frag_max_s:
            groups.append(np.asarray(cur))
            cur, travel = [], 0.0
        cur.append(int(b))
    if len(cur) < 3 and groups:              # a tiny tail fragment would give an ill-posed ICP: merge it
        groups[-1] = np.r_[groups[-1], cur]
    else:
        groups.append(np.asarray(cur))
    return groups


def _backproject(cap, i: int, p: DriftParams) -> np.ndarray:
    """Camera-frame 3D points of one depth frame (high confidence only, strided)."""
    s = p.frag_pixel_stride
    v, u = np.mgrid[0:DEPTH_H:s, 0:DEPTH_W:s]
    d = cap.depth(i, p.min_conf, p.min_depth_m, p.max_depth_m)[v, u]
    ok = d > 0
    K = cap.K_depth(i)
    z = d[ok]
    return np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], 1)


def build_fragments(cap, kf: np.ndarray, T_wc: np.ndarray, p: DriftParams) -> list[Fragment]:
    """One voxelised point cloud per fragment, in the frame of its middle keyframe (smallest lever arm)."""
    frags: list[Fragment] = []
    for k, frames in enumerate(split_fragments(kf, T_wc, cap.timestamps, p)):
        anchor = int(frames[len(frames) // 2])
        T_aw = np.linalg.inv(T_wc[anchor])
        pts = []
        for i in frames:
            T = T_aw @ T_wc[i]
            pts.append(_backproject(cap, int(i), p) @ T[:3, :3].T + T[:3, 3])
        cloud = _with_normals(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts))),
                              p.frag_voxel_m, p.normal_radius_m)
        cloud.orient_normals_towards_camera_location(np.zeros(3))   # floors face up, walls face the camera
        travel = 0.0 if not frags else float(np.linalg.norm(T_wc[anchor, :3, 3] - frags[-1].pose[:3, 3]))
        frags.append(Fragment(k, frames, anchor, float(cap.timestamps[anchor]), travel, T_wc[anchor].copy(), cloud))
    return frags


# ============================================================================================ 2. plane anchors
def measure_plane_anchors(frags: list[Fragment], T_wc: np.ndarray, p: DriftParams) -> None:
    """Per fragment, under its ARKit pose: the floor level and the dominant (Manhattan) wall direction.

    These are absolute references that do not depend on loop closures: a heading drift of 1 deg shows up as every
    wall of the fragment turning by 1 deg relative to the rest of the building; a vertical drift of 1 cm shows up as
    the fragment's floor sitting 1 cm off the global floor."""
    for f in frags:
        P = np.asarray(f.cloud.points) @ f.pose[:3, :3].T + f.pose[:3, 3]
        N = np.asarray(f.cloud.normals) @ f.pose[:3, :3].T
        theta, score = manhattan_yaw(N, tol_deg=5.0)
        f.wall_theta, f.manhattan_score = (theta, score) if score > 0 else (None, 0.0)
        floor = estimate_floor(P, N, T_wc[f.frames, 1, 3])
        if floor is not None:
            near = (N[:, 1] > 0.95) & (np.abs(P[:, 1] - floor) < 0.015)   # refine the 1 cm histogram bin
            f.floor_y = float(np.median(P[near, 1])) if near.sum() >= 200 else None
        down = (N[:, 1] < -0.95) & (P[:, 1] > T_wc[f.frames, 1, 3].max() + 0.3)
        f.ceiling_y = _highest_level(P[down, 1])


def _highest_level(y: np.ndarray, min_pts: int = 200) -> float | None:
    """Highest strong horizontal level in a set of heights (1 cm histogram, refined by the median of its points)."""
    if len(y) < min_pts:
        return None
    h, e = np.histogram(y, np.arange(y.min(), y.max() + 0.02, 0.01))
    strong = np.where(h >= max(min_pts, 0.3 * h.max()))[0]
    if not len(strong):
        return None
    top = e[strong.max()] + 0.005
    return float(np.median(y[np.abs(y - top) < 0.015]))


def anchor_sets(frags: list[Fragment], p: DriftParams) -> tuple[list[int], list[int], dict]:
    """Which fragments get a Manhattan prior and which a floor prior, with the gates that decided it."""
    man = [f.index for f in frags if f.wall_theta is not None and f.manhattan_score >= p.manhattan_min_score]
    manhattan_on = len(man) >= p.manhattan_min_frac * len(frags)
    floors = np.array([f.floor_y for f in frags if f.floor_y is not None])
    flo: list[int] = []
    if len(floors) >= 3:
        lowest = np.percentile(floors, 10)      # beds and tables are higher than the floor, never lower
        ref = np.median(floors[np.abs(floors - lowest) < p.floor_gate_m])
        flo = [f.index for f in frags if f.floor_y is not None and abs(f.floor_y - ref) < p.floor_gate_m]
    info = dict(manhattan_fragments=len(man), manhattan_on=bool(manhattan_on), floor_fragments=len(flo))
    return (man if manhattan_on else []), flo, info


# ============================================================================================ 3. registration
def overlap_fraction(a_pts: np.ndarray, b_pts: np.ndarray, a_tree: cKDTree, b_tree: cKDTree,
                     T_ab: np.ndarray, voxel: float) -> float:
    """Share of either fragment that lies within `voxel` of the other under T_ab (a -> b)."""
    d_a, _ = b_tree.query(a_pts @ T_ab[:3, :3].T + T_ab[:3, 3], distance_upper_bound=voxel)
    d_b, _ = a_tree.query((b_pts - T_ab[:3, 3]) @ T_ab[:3, :3], distance_upper_bound=voxel)
    return float(max(np.isfinite(d_a).mean(), np.isfinite(d_b).mean()))


def point_to_plane_stats(src, tgt, T_st: np.ndarray, max_corr: float) -> tuple[np.ndarray, int, float, float]:
    """Information matrix, inlier count, fitness and point-to-plane RMSE of an alignment.

    Information = sum of J^T J with J = [q x n, n] over inlier correspondences (target frame). Unlike Open3D's
    point-to-POINT get_information_matrix_from_point_clouds, this knows that two parallel walls do not constrain
    motion along the walls, so a corridor ICP gets (almost) no say in that direction."""
    P = np.asarray(src.points) @ T_st[:3, :3].T + T_st[:3, 3]
    d, j = cKDTree(np.asarray(tgt.points)).query(P, distance_upper_bound=max_corr)
    ok = np.isfinite(d)
    q, n = np.asarray(tgt.points)[j[ok]], np.asarray(tgt.normals)[j[ok]]
    J = np.hstack([np.cross(q, n), n])
    plane = np.einsum("ij,ij->i", P[ok] - q, n)
    rmse = float(np.sqrt(np.mean(plane ** 2))) if ok.any() else np.inf
    return J.T @ J, int(ok.sum()), float(ok.mean()), rmse


def register(src: Fragment, tgt: Fragment, T_init: np.ndarray, p: DriftParams, kind: str) -> Edge:
    """Coarse-to-fine point-to-plane ICP (Tukey-robust) from T_init, plus the quality numbers used to accept it.

    Call it inside deterministic_open3d() (loop_edges and odometry_edges do): with Open3D's thread pool the result
    is not bit-reproducible."""
    T = T_init.copy()
    for vox, corr, it in zip(p.icp_voxels_m, p.icp_max_corr_m, p.icp_iters):
        s = _with_normals(src.cloud, vox, max(p.normal_radius_m, 2.5 * vox))
        t = _with_normals(tgt.cloud, vox, max(p.normal_radius_m, 2.5 * vox))
        est = REG.TransformationEstimationPointToPlane(REG.TukeyLoss(k=corr))
        T = REG.registration_icp(s, t, corr, T, est, REG.ICPConvergenceCriteria(max_iteration=it)).transformation
    info, n, fit, rmse = point_to_plane_stats(src.cloud, tgt.cloud, T, p.eval_corr_m)
    dT = np.linalg.inv(T_init) @ T
    return Edge(src.index, tgt.index, kind, T, info, n, fit, rmse, float(np.linalg.norm(dT[:3, 3])), rot_deg(dT),
                float(np.linalg.eigvalsh(info[3:, 3:])[0]) / max(n, 1))


def verify_loop(e: Edge, p: DriftParams) -> Edge:
    """Accept a loop only if ICP converged onto overlapping, well-fitting, non-degenerate geometry near the guess."""
    checks = [(e.fitness >= p.loop_fitness_min, f"fitness {e.fitness:.2f} < {p.loop_fitness_min}"),
              (e.plane_rmse_m <= p.loop_plane_rmse_max_m, f"rmse {100 * e.plane_rmse_m:.2f} cm"),
              (e.jump_m <= p.loop_max_jump_m and e.jump_deg <= p.loop_max_jump_deg,
               f"jump {100 * e.jump_m:.0f} cm / {e.jump_deg:.1f} deg"),
              (e.constraint_frac >= p.min_constraint_frac, f"degenerate {e.constraint_frac:.3f}")]
    failed = [msg for ok, msg in checks if not ok]
    e.accepted, e.reason = not failed, "; ".join(failed)
    return e


def odometry_edges(frags: list[Fragment], p: DriftParams) -> list[Edge]:
    """ARKit's relative pose between consecutive fragments (or its ICP refinement for the open3d6 baseline)."""
    edges = []
    for a, b in zip(frags[:-1], frags[1:]):
        T = np.linalg.inv(b.pose) @ a.pose
        if p.refine_odometry:
            with deterministic_open3d():
                e = register(a, b, T, p, "odometry")
            if e.fitness < p.loop_fitness_min or e.jump_m > 0.05 or e.jump_deg > 1.0:
                e.T, e.reason = T, "icp rejected, ARKit kept"
        else:
            info, n, fit, rmse = point_to_plane_stats(a.cloud, b.cloud, T, p.eval_corr_m)
            e = Edge(a.index, b.index, "odometry", T, info, n, fit, rmse)
        edges.append(e)
    return edges


def loop_edges(frags: list[Fragment], X: np.ndarray, p: DriftParams, skip: set[tuple[int, int]]) -> list[Edge]:
    """Find non-consecutive fragment pairs that overlap under poses X, ICP them from X, and verify."""
    coarse = [np.asarray(f.cloud.voxel_down_sample(p.overlap_voxel_m).points) for f in frags]
    trees = [cKDTree(c) for c in coarse]
    centre = np.stack([X[k][:3, :3] @ c.mean(0) + X[k][:3, 3] for k, c in enumerate(coarse)])
    radius = np.array([np.percentile(np.linalg.norm(c - c.mean(0), axis=1), 95) for c in coarse])
    cands = []
    for s in range(len(frags)):
        for t in range(s + 2, len(frags)):
            if (s, t) in skip or np.linalg.norm(centre[s] - centre[t]) > radius[s] + radius[t]:
                continue
            T0 = np.linalg.inv(X[t]) @ X[s]
            ov = overlap_fraction(coarse[s], coarse[t], trees[s], trees[t], T0, p.overlap_voxel_m)
            if ov < p.overlap_min:
                continue
            kind = "revisit" if frags[t].t_anchor - frags[s].t_anchor >= p.revisit_min_gap_s else "local"
            cands.append((s, t, T0, kind, ov))

    def one(c):
        s, t, T0, kind, ov = c
        e = verify_loop(register(frags[s], frags[t], T0, p, kind), p)
        e.extra["overlap"] = ov
        return e
    # Deterministic AND parallel: every ICP runs single-threaded inside Open3D (fixed summation order), and the
    # candidate pairs run concurrently in Python threads (Open3D releases the GIL); map() keeps the input order.
    with deterministic_open3d(), ThreadPoolExecutor(max(1, min(p.icp_workers, len(cands)))) as ex:
        return list(ex.map(one, cands))


# ============================================================================================ 4. 4-DoF solver
@dataclass
class Graph4:
    """Everything the 4-DoF least-squares problem needs, as flat arrays (vectorised residuals)."""
    c: np.ndarray                          # (N,3) ARKit anchor positions
    R: np.ndarray                          # (N,3,3) ARKit anchor rotations
    src: np.ndarray
    tgt: np.ndarray
    p_meas: np.ndarray                     # (E,3) source anchor position in target frame (measured)
    yaw_meas: np.ndarray                   # (E,) measured psi_s - psi_t
    W_t: np.ndarray                        # (E,3,3) translation whitening (information-shaped)
    w_yaw: np.ndarray                      # (E,) 1/sigma yaw
    robust: np.ndarray                     # (E,) bool: loops are robust, odometry is not
    man_idx: np.ndarray
    man_theta: np.ndarray
    floor_idx: np.ndarray
    floor_y: np.ndarray
    p: DriftParams


def _edge_terms(e: Edge, frags: list[Fragment], p: DriftParams) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Measured (position, yaw) of one edge in the 4-DoF parameterisation, and its whitening."""
    Rs, Rt = frags[e.source].pose[:3, :3], frags[e.target].pose[:3, :3]
    yaw = yaw_angle(Rt @ e.T[:3, :3] @ Rs.T)
    if e.kind == "odometry":
        travel = max(frags[e.target].travel_m, 0.1)
        return e.T[:3, 3], yaw, np.eye(3) / (p.odo_trans_sigma_m * np.sqrt(travel)), \
            1 / np.radians(p.odo_yaw_sigma_deg * np.sqrt(travel))
    M = Rt @ (e.info[3:, 3:] / max(e.n_corr, 1)) @ Rt.T          # normalised, world axes; trace = 1
    lam, V = np.linalg.eigh(M)
    W = V @ np.diag(np.sqrt(3 * np.clip(lam, 0, None))) @ V.T / p.loop_trans_sigma_m
    return e.T[:3, 3], yaw, W, 1 / np.radians(p.loop_yaw_sigma_deg)


def build_graph(frags: list[Fragment], edges: list[Edge], man: list[int], flo: list[int], p: DriftParams) -> Graph4:
    terms = [_edge_terms(e, frags, p) for e in edges]
    return Graph4(
        c=np.stack([f.pose[:3, 3] for f in frags]), R=np.stack([f.pose[:3, :3] for f in frags]),
        src=np.array([e.source for e in edges], int), tgt=np.array([e.target for e in edges], int),
        p_meas=np.stack([t[0] for t in terms]), yaw_meas=np.array([t[1] for t in terms]),
        W_t=np.stack([t[2] for t in terms]), w_yaw=np.array([t[3] for t in terms]),
        robust=np.array([e.kind != "odometry" for e in edges]),
        man_idx=np.array(man, int), man_theta=np.array([frags[k].wall_theta for k in man], float),
        floor_idx=np.array(flo, int), floor_y=np.array([frags[k].floor_y for k in flo], float), p=p)


def _unpack(x: np.ndarray, g: Graph4) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Node 0 is the fixed reference (gauge). Extra unknowns: Manhattan direction phi, global floor level."""
    n = len(g.c)
    psi = np.r_[0.0, x[:n - 1]]
    d = np.vstack([np.zeros(3), x[n - 1:4 * (n - 1)].reshape(-1, 3)])
    return psi, d, x[4 * (n - 1)], x[4 * (n - 1) + 1]


def _blocks(x: np.ndarray, g: Graph4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unweighted (but whitened) residual blocks: edges (E,4), Manhattan (M,), floor (F,), yaw smoothness (N-2,).

    Yaw smoothness is the second difference psi[k+1] - 2 psi[k] + psi[k-1]: zero for a steady heading-drift ramp
    (what a residual gyro bias produces), large for fragment-to-fragment jitter (what noisy anchors/loops produce)."""
    psi, d, phi, yf = _unpack(x, g)
    c2 = g.c + d
    Rt = _ry(psi[g.tgt]) @ g.R[g.tgt]
    r_t = c2[g.src] - c2[g.tgt] - np.einsum("nij,nj->ni", Rt, g.p_meas)
    r_t = np.einsum("nij,nj->ni", g.W_t, r_t)
    r_y = _wrap(psi[g.src] - psi[g.tgt] - g.yaw_meas, 2 * np.pi) * g.w_yaw
    r_m = _wrap(g.man_theta - psi[g.man_idx] - phi, np.pi / 2) / np.radians(g.p.manhattan_sigma_deg)
    r_f = (g.floor_y + d[g.floor_idx, 1] - yf) / g.p.floor_sigma_m
    r_s = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / np.radians(g.p.yaw_smooth_sigma_deg) \
        if g.p.yaw_smooth_sigma_deg > 0 else np.zeros(0)
    return np.c_[r_t, r_y], r_m, r_f, r_s


def _sparsity(g: Graph4, n_res: int) -> lil_matrix:
    n = len(g.c)
    S = lil_matrix((n_res, 4 * (n - 1) + 2), dtype=int)

    def node_cols(k):
        return [] if k == 0 else [k - 1] + [n - 1 + 3 * (k - 1) + j for j in range(3)]
    row = 0
    for s, t in zip(g.src, g.tgt):
        for r in range(4):
            for col in node_cols(s) + node_cols(t):
                S[row + r, col] = 1
        row += 4
    for k in g.man_idx:
        S[row, [c for c in node_cols(k)[:1]] + [4 * (n - 1)]] = 1
        row += 1
    for k in g.floor_idx:
        S[row, [c for c in node_cols(k)[2:3]] + [4 * (n - 1) + 1]] = 1
        row += 1
    for k in range(1, n - 1) if g.p.yaw_smooth_sigma_deg > 0 else []:
        S[row, [c for j in (k - 1, k, k + 1) for c in node_cols(j)[:1]]] = 1
        row += 1
    return S


def solve_4dof(g: Graph4) -> tuple[np.ndarray, dict]:
    """Robust 4-DoF pose graph: IRLS with a Cauchy kernel on loops and plane anchors; odometry is never down-weighted.

    Each outer iteration solves an ordinary weighted least-squares problem (scipy TRF with a sparse Jacobian), then
    re-weights every robust term by w = 1 / (1 + (chi2/dof) / c^2). A term that disagrees with the rest of the graph
    by 3.5 sigma ends with w < 0.25 and is reported as pruned."""
    n = len(g.c)
    x = np.zeros(4 * (n - 1) + 2)
    if len(g.man_idx):
        x[-2] = np.angle(np.mean(np.exp(4j * g.man_theta))) / 4
    if len(g.floor_idx):
        x[-1] = np.median(g.floor_y)
    w_e, w_m, w_f = np.ones(len(g.src)), np.ones(len(g.man_idx)), np.ones(len(g.floor_idx))
    c2 = g.p.robust_c ** 2

    def fun(x):
        e, m, f, sm = _blocks(x, g)
        return np.r_[(e * np.sqrt(w_e)[:, None]).ravel(), m * np.sqrt(w_m), f * np.sqrt(w_f), sm]
    n_smooth = max(n - 2, 0) if g.p.yaw_smooth_sigma_deg > 0 else 0
    S = _sparsity(g, 4 * len(g.src) + len(g.man_idx) + len(g.floor_idx) + n_smooth)
    for _ in range(g.p.irls_iters):
        x = least_squares(fun, x, jac_sparsity=S, method="trf", x_scale="jac").x
        e, m, f, _ = _blocks(x, g)
        new_e = np.where(g.robust, 1 / (1 + (e ** 2).sum(1) / 4 / c2), 1.0)
        new_m, new_f = 1 / (1 + m ** 2 / c2), 1 / (1 + f ** 2 / c2)
        change = max(np.abs(new_e - w_e).max(initial=0), np.abs(new_m - w_m).max(initial=0),
                     np.abs(new_f - w_f).max(initial=0))
        w_e, w_m, w_f = new_e, new_m, new_f
        if change < 1e-3:
            break
    psi, d, phi, yf = _unpack(x, g)
    X = np.tile(np.eye(4), (n, 1, 1))
    X[:, :3, :3] = _ry(psi) @ g.R
    X[:, :3, 3] = g.c + d
    return X, dict(edge_weights=w_e, manhattan_weights=w_m, floor_weights=w_f, phi=float(phi),
                   floor_level=float(yf) if len(g.floor_idx) else None)


def solve_open3d6(frags: list[Fragment], edges: list[Edge], p: DriftParams) -> tuple[np.ndarray, dict]:
    """Baseline: Open3D 6-DoF global optimisation (LM + line process), exactly as in Choi et al. / Open3D's system."""
    g = REG.PoseGraph()
    for f in frags:
        g.nodes.append(REG.PoseGraphNode(f.pose.copy()))
    for e in edges:
        g.edges.append(REG.PoseGraphEdge(e.source, e.target, e.T, e.info, e.kind != "odometry"))
    opt = REG.GlobalOptimizationOption(max_correspondence_distance=p.prune_corr_m,
                                       edge_prune_threshold=p.edge_prune_threshold,
                                       preference_loop_closure=p.preference_loop_closure, reference_node=0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        REG.global_optimization(g, REG.GlobalOptimizationLevenbergMarquardt(),
                                REG.GlobalOptimizationConvergenceCriteria(), opt)
    kept = {(e.source_node_id, e.target_node_id) for e in g.edges}
    w = np.array([1.0 if (e.source, e.target) in kept else 0.0 for e in edges])
    return np.stack([np.asarray(n.pose) for n in g.nodes]), dict(edge_weights=w)


def solve(frags: list[Fragment], edges: list[Edge], man: list[int], flo: list[int], p: DriftParams):
    used = [e for e in edges if e.kind == "odometry" or e.accepted]
    if p.solver == "open3d6":
        X, out = solve_open3d6(frags, used, p)
    else:
        X, out = solve_4dof(build_graph(frags, used, man, flo, p))
    for e, w in zip(used, out["edge_weights"]):
        e.weight = float(w)
    return X, out


# ============================================================================================ 5. apply to frames
def interpolate_corrections(frags: list[Fragment], X_new: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """World-frame correction C(t) for every frame time t, so that T_new = C(t) @ T_old.

    At each fragment anchor C = X_new X_old^-1 exactly. Between anchors the rotation is SLERPed and the translation
    linearly blended in time, so no seam appears at fragment boundaries. Frames outside the anchors' time range take
    the nearest anchor's correction."""
    C = np.stack([X_new[f.index] @ np.linalg.inv(f.pose) for f in frags])
    ta = np.array([f.t_anchor for f in frags])
    out = np.tile(np.eye(4), (len(timestamps), 1, 1))
    if len(frags) == 1:
        out[:] = C[0]
        return out
    tq = np.clip(timestamps, ta[0], ta[-1])
    out[:, :3, :3] = Slerp(ta, Rotation.from_matrix(C[:, :3, :3]))(tq).as_matrix()
    for k in range(3):
        out[:, k, 3] = np.interp(tq, ta, C[:, k, 3])
    return out


# ============================================================================================ evaluation
def edge_residual(e: Edge, X: np.ndarray, frags: list[Fragment], n_pts: int = 2000) -> dict:
    """How far apart the two fragments' shared geometry sits under poses X, according to the ICP measurement.

    E = T_st^-1 X_t^-1 X_s is the identity when the poses agree with the measurement. Headline number:
    surface_sep_cm, the mean displacement E causes ALONG each source point's surface normal, i.e. how far apart the
    two observations of the same surfaces sit. It ignores sliding along a surface (e.g. along a corridor), which
    ICP cannot measure, so a degenerate loop is not mistaken for a large residual. Also reported: the full mean
    displacement (lever arm included), split into plan (x, z) and vertical (y), and E's rotation angle."""
    E = np.linalg.inv(e.T) @ np.linalg.inv(X[e.target]) @ X[e.source]
    cloud = frags[e.source].cloud
    sel = np.random.default_rng(0).choice(len(cloud.points), min(n_pts, len(cloud.points)), replace=False)
    P, Nrm = np.asarray(cloud.points)[sel], np.asarray(cloud.normals)[sel]
    disp_local = P @ E[:3, :3].T + E[:3, 3] - P
    disp = disp_local @ X[e.source][:3, :3].T
    return dict(surface_sep_cm=float(100 * np.abs(np.einsum("ij,ij->i", disp_local, Nrm)).mean()),
                mean_disp_cm=float(100 * np.linalg.norm(disp, axis=1).mean()),
                plan_disp_cm=float(100 * np.linalg.norm(disp[:, [0, 2]], axis=1).mean()),
                vert_disp_cm=float(100 * np.abs(disp[:, 1]).mean()), rot_deg=rot_deg(E))


def summarise(values) -> dict:
    v = np.asarray(list(values), float)
    if len(v) == 0:
        return dict(n=0)
    return dict(n=int(len(v)), median=float(np.median(v)), mean=float(v.mean()), rms=float(np.sqrt((v ** 2).mean())),
                p95=float(np.percentile(v, 95)), max=float(v.max()))


def residual_table(edges: list[Edge], poses: dict[str, np.ndarray], frags: list[Fragment]) -> dict:
    """Residual summaries per edge kind (local/revisit loops) for each named pose set."""
    out: dict = {}
    for kind in ("local", "revisit"):
        sel = [e for e in edges if e.kind == kind]
        for name, X in poses.items():
            rs = [edge_residual(e, X, frags) for e in sel]
            out.setdefault(kind, {})[name] = {k: summarise(r[k] for r in rs)
                                              for k in RESIDUAL_KEYS}
    return out


def anchor_spread(frags: list[Fragment], X: np.ndarray, man: list[int], flo: list[int]) -> dict:
    """Spread of the plane anchors under poses X: per-fragment wall direction (deg), floor and ceiling level (cm).

    The ceiling is never used in the solve (rooms may have different ceiling heights), so its spread is an
    independent check of the vertical correction. Only the main ceiling (within 15 cm of the median) is used, and its
    spread is a robust MAD-sigma because a few fragments see a legitimately different ceiling (another room)."""
    psi = np.array([yaw_angle(X[k][:3, :3] @ frags[k].pose[:3, :3].T) for k in range(len(frags))])
    th = np.array([frags[k].wall_theta - psi[k] for k in man])
    yaw_dev = np.degrees(_wrap(th - np.angle(np.mean(np.exp(4j * th))) / 4, np.pi / 2)) if len(th) else np.array([])
    fl = np.array([frags[k].floor_y + X[k][1, 3] - frags[k].pose[1, 3] for k in flo])
    ce = [(k, f.ceiling_y) for k, f in enumerate(frags) if f.ceiling_y is not None]
    ref = np.median([c for _, c in ce]) if ce else 0.0
    cl = np.array([c + X[k][1, 3] - frags[k].pose[1, 3] for k, c in ce if abs(c - ref) < 0.15])
    mad = lambda v: float(100 * 1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 2 else None  # noqa: E731
    return dict(ceiling_fragments=int(len(cl)), ceiling_mad_sigma_cm=mad(cl), floor_mad_sigma_cm=mad(fl),
                wall_yaw_dev_deg=summarise(np.abs(yaw_dev)),
                wall_yaw_std_deg=float(np.std(yaw_dev)) if len(th) else None,
                floor_std_cm=float(100 * np.std(fl)) if len(fl) else None,
                floor_range_cm=float(100 * np.ptp(fl)) if len(fl) else None)


def frame_correction_stats(T_old: np.ndarray, T_new: np.ndarray) -> dict:
    dt = np.linalg.norm(T_new[:, :3, 3] - T_old[:, :3, 3], axis=1)
    dr = np.array([rot_deg(a[:3, :3].T @ b[:3, :3]) for a, b in zip(T_old, T_new)])
    tilt = np.degrees(np.arccos(np.clip([(b[:3, :3] @ a[:3, :3].T)[1, 1] for a, b in zip(T_old, T_new)], -1, 1)))
    return dict(median_trans_cm=float(100 * np.median(dt)), max_trans_cm=float(100 * dt.max()),
                max_vert_cm=float(100 * np.abs(T_new[:, 1, 3] - T_old[:, 1, 3]).max()),
                median_rot_deg=float(np.median(dr)), max_rot_deg=float(dr.max()), max_tilt_deg=float(tilt.max()))


# ============================================================================================ public entry point
@dataclass
class DriftState:
    """Intermediate results, kept so the ablation can re-solve with held-out loops without re-running ICP."""
    frags: list[Fragment]
    edges: list[Edge]
    man: list[int]
    flo: list[int]
    X_arkit: np.ndarray
    X_final: np.ndarray
    params: DriftParams


def run_graph(cap, kf: np.ndarray, T_wc: np.ndarray, p: DriftParams, log=print) -> DriftState:
    """Fragments -> plane anchors -> (solve -> find/verify loops from the current estimate) x rounds -> final solve.

    Loops are searched from the latest estimate, not from raw ARKit: after the plane anchors remove most heading
    drift, revisited fragments start close enough for ICP to converge (its basin is ~20 cm / a few degrees)."""
    t0 = time.time()
    frags = build_fragments(cap, kf, T_wc, p)
    X_arkit = np.stack([f.pose for f in frags])
    if len(frags) < 3:      # a walk shorter than ~3 m has no drift worth correcting and too little for a graph
        log(f"[drift] only {len(frags)} fragment(s): poses kept, nothing to correct")
        return DriftState(frags, [], [], [], X_arkit, X_arkit.copy(), p)
    measure_plane_anchors(frags, T_wc, p)
    man, flo, ainfo = anchor_sets(frags, p) if p.use_plane_anchors else ([], [], {})
    log(f"[drift] {len(frags)} fragments from {len(kf)} keyframes; anchors {ainfo} ({time.time() - t0:.1f}s)")
    edges = odometry_edges(frags, p)
    X, _ = solve(frags, edges, man, flo, p) if p.solver == "4dof" else (X_arkit, None)
    for r in range(p.loop_rounds if p.use_loops else 0):
        seen = {(e.source, e.target) for e in edges}
        new = loop_edges(frags, X, p, seen)
        edges += new
        X, _ = solve(frags, edges, man, flo, p)
        log(f"[drift] round {r + 1}: {len(new)} new loop candidates, {sum(e.accepted for e in new)} accepted "
            f"({time.time() - t0:.1f}s)")
    if not p.use_loops:
        X, _ = solve(frags, edges, man, flo, p)
    return DriftState(frags, edges, man, flo, X_arkit, X, p)


def correct_drift(cap, kf: np.ndarray, T_wc: np.ndarray, cfg_or_params, log=print) -> tuple[np.ndarray, dict]:
    """Return drift-corrected camera-to-world poses for ALL frames, plus a JSON-serialisable report.

    cfg_or_params: a DriftParams, or the shared Config (depth filtering is taken from it, the rest are defaults)."""
    p = cfg_or_params if isinstance(cfg_or_params, DriftParams) else DriftParams.from_config(cfg_or_params)
    t0 = time.time()
    st = run_graph(cap, kf, T_wc, p, log)
    T_new = interpolate_corrections(st.frags, st.X_final, cap.timestamps) @ T_wc
    report = make_report(st, time.time() - t0)
    report["frame_correction"] = frame_correction_stats(T_wc, T_new)
    fc = report["frame_correction"]
    log(f"[drift] loops {report['loops']} | max frame correction {fc['max_trans_cm']:.1f} cm, "
        f"{fc['max_rot_deg']:.2f} deg ({time.time() - t0:.1f}s)")
    report["_state"] = st
    return T_new, report


def make_report(st: DriftState, runtime_s: float) -> dict:
    loops = [e for e in st.edges if e.kind != "odometry"]
    used = [e for e in loops if e.accepted]
    reasons: dict[str, int] = {}
    for e in loops:
        for r in filter(None, e.reason.split("; ")):
            reasons[r.split(" ")[0]] = reasons.get(r.split(" ")[0], 0) + 1
    corr = [st.X_final[k] @ np.linalg.inv(st.X_arkit[k]) for k in range(len(st.frags))]
    poses = dict(arkit=st.X_arkit, corrected=st.X_final)
    return dict(
        params=asdict(st.params), runtime_s=round(runtime_s, 1), n_fragments=len(st.frags),
        anchors=dict(manhattan_fragments=len(st.man), floor_fragments=len(st.flo),
                     **{name: anchor_spread(st.frags, X, st.man, st.flo) for name, X in poses.items()}),
        loops=dict(candidates=len(loops), accepted=len(used), rejected=len(loops) - len(used),
                   down_weighted=sum(e.weight < 0.25 for e in used),
                   revisit_accepted=sum(e.kind == "revisit" for e in used),
                   local_accepted=sum(e.kind == "local" for e in used), rejection_reasons=reasons),
        loop_residuals=residual_table(used, poses, st.frags),
        fragment_correction=dict(
            trans_cm=summarise(100 * np.linalg.norm(st.X_final[:, :3, 3] - st.X_arkit[:, :3, 3], axis=1)),
            rot_deg=summarise(rot_deg(c) for c in corr)),
        edges=[dict(source=e.source, target=e.target, kind=e.kind, fitness=round(e.fitness, 3),
                    plane_rmse_cm=round(100 * e.plane_rmse_m, 2), jump_cm=round(100 * e.jump_m, 2),
                    jump_deg=round(e.jump_deg, 3), constraint_frac=round(e.constraint_frac, 3),
                    accepted=e.accepted, weight=round(e.weight, 3), reason=e.reason,
                    **{k: round(v, 3) for k, v in e.extra.items()}) for e in loops])
