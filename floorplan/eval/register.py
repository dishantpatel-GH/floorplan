"""Rigid 2D registration between two captures (or two plans) of the same space.

Why this exists:
  * Every capture has its own arbitrary heading and origin (DATA_NOTES.md). To compare two captures of the same
    apartment (repeatability gate), or a video/photo-tier plan with the LiDAR reference, we must first put both in
    one frame. Lengths and areas do not depend on the frame, but *matching* ("which wall in capture A is this wall
    in capture B?") does.
  * The registration uses the reconstructed geometry only, never the plans being evaluated. Otherwise a wrong plan
    could pull the registration towards itself and hide its own error.
  * It is rigid on purpose (rotation + translation, no scale). A scale error in a video/photo plan is exactly what
    the tier gates measure; a similarity transform would silently absorb it.

How (each step is a separate function below):
  1. Evidence: wall points (vertical surfaces 0.3-2.0 m above the floor, which removes floor clutter and most
     furniture tops) and floor points, projected to the plan (u, v) = (x, z).
  2. Global search. Both scenes are Manhattan-aligned, so the unknown rotation is k*90 deg plus a small residual.
     For each of the 4 quadrants and a fine yaw grid of +-3 deg we rotate B, rasterise it, and find the best
     translation with one FFT cross-correlation. The rasters are split into 4 channels by wall-normal direction,
     so the two faces of a partition wall (opposite normals, ~10 cm apart) cannot be confused.
  3. Local refinement: 2D point-to-line ICP on wall points with Huber weights (sub-centimetre).
  4. Diagnostics: residual, inlier fraction, overlap of the floor areas, how distinct the winning yaw quadrant is,
     and how well the walls constrain each direction (a corridor constrains only one axis).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

# Normal-direction channels: +u, -u, +v, -v (plan axes after Manhattan alignment).
_AXES = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


@dataclass
class Evidence2D:
    """Plan-view geometry of one capture: what the registration is allowed to look at."""
    name: str
    wall_pts: np.ndarray        # (N, 2) plan metres
    wall_normals: np.ndarray    # (N, 2) unit, pointing to the observed side
    floor_pts: np.ndarray       # (M, 2) plan metres
    floor_y: float = 0.0        # floor height in the capture's aligned world (for the 3D transform)


@dataclass
class RegisterParams:
    wall_band_m: tuple[float, float] = (0.3, 2.0)   # heights above floor that count as wall evidence
    max_wall_ny: float = 0.3                        # |n_y| below this = vertical surface
    min_floor_ny: float = 0.9
    floor_band_m: float = 0.05
    coarse_res_m: float = 0.05
    fine_res_m: float = 0.02
    blur_cells: float = 1.0                         # Gaussian blur of the rasters, in cells (tolerates small drift)
    split_normals: bool = True                      # 4 rasters by normal direction (False only for the ablation)
    yaw_range_deg: float = 5.0                      # residual yaw after both Manhattan alignments (see eval.md I-1)
    coarse_yaw_step_deg: float = 0.5
    fine_yaw_step_deg: float = 0.1
    icp_max_dist_m: tuple[float, ...] = (0.15, 0.08, 0.05, 0.03, 0.03, 0.03)
    icp_normal_cos: float = 0.8                     # correspondences must face the same way (within ~37 deg)
    huber_m: float = 0.01
    inlier_m: float = 0.03                          # residual threshold for the reported inlier statistics
    overlap_res_m: float = 0.05
    seed: int = 0
    max_icp_points: int = 60000


@dataclass
class Registration:
    """Result: T2 maps B's plan coordinates into A's plan; T3 maps B's aligned 3D world into A's."""
    a: str
    b: str
    T2: list                     # 3x3
    T3: list                     # 4x4
    yaw_deg: float
    t: tuple[float, float]
    score: float                 # normalised cross-correlation peak at the coarse level (0..1)
    distinctiveness: float       # best score / best score of any other 90-deg quadrant (>1.2 = unambiguous)
    yaw_at_search_edge: bool     # coarse optimum on the edge of the yaw window: widen the window, do not trust
    residual_rmse_m: float       # RMS point-to-line distance of inlier wall points after ICP
    residual_median_m: float
    inlier_frac_b: float         # fraction of B wall points within inlier_m of an A wall (incl. non-overlap)
    overlap_iou: float           # floor-area IoU after registration
    overlap_of_a: float          # fraction of A's floor covered by B
    overlap_of_b: float          # fraction of B's floor covered by A
    sigma_yaw_deg: float         # Hessian-based 1-sigma (optimistic: neighbouring points are not independent)
    sigma_t_m: tuple[float, float]
    constraint_ratio: float      # smallest / largest eigenvalue of the translation block (near 0 = corridor-like)
    yaw_curve: dict = field(default_factory=dict)   # quadrant -> list of (yaw_deg, score), for the figure

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------------------------- evidence

def scene_evidence(scene: dict, info: dict, name: str, p: RegisterParams = RegisterParams()) -> Evidence2D:
    """Extract wall and floor evidence from a prepared scene (TSDF surface, aligned frame)."""
    P, N = scene["points"].astype(np.float64), scene["normals"].astype(np.float64)
    f = float(info["floor_y"])
    h = P[:, 1] - f
    wall = (np.abs(N[:, 1]) < p.max_wall_ny) & (h > p.wall_band_m[0]) & (h < p.wall_band_m[1])
    floor = (N[:, 1] > p.min_floor_ny) & (np.abs(h) < p.floor_band_m)
    n2 = N[wall][:, [0, 2]]
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12
    return Evidence2D(name, P[wall][:, [0, 2]], n2, P[floor][:, [0, 2]], f)


def plan_evidence(plan: dict, step_m: float = 0.02) -> Evidence2D:
    """Evidence from a plan.json dict: wall segments sampled every step_m, floor = room polygon interiors.

    Used when two plans live in different frames and no scene is available (e.g. a photo-tier plan)."""
    from shapely.geometry import Point, Polygon

    pts, nrm = [], []
    for w in plan.get("walls", []):
        p0, p1 = np.asarray(w["p0"], float), np.asarray(w["p1"], float)
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step_m)) + 1)
        s = np.linspace(0, 1, n)[:, None]
        pts.append(p0 + s * (p1 - p0))
        nrm.append(np.repeat(np.asarray(w["normal"], float)[None], n, 0))
    floor = []
    for r in plan.get("rooms", []):
        poly = Polygon(r["polygon"])
        if poly.is_empty or poly.area <= 0:
            continue
        x0, y0, x1, y1 = poly.bounds
        gx, gy = np.meshgrid(np.arange(x0, x1, step_m * 2.5), np.arange(y0, y1, step_m * 2.5))
        g = np.c_[gx.ravel(), gy.ravel()]
        floor.append(g[[poly.contains(Point(q)) for q in g]])
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros((0, 2))   # noqa: E731
    return Evidence2D(plan.get("capture_id", "plan"), cat(pts), cat(nrm), cat(floor), 0.0)


# ----------------------------------------------------------------------------------------------- geometry helpers

def rot2(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s], [s, c]])


def make_T2(yaw_rad: float, t: np.ndarray) -> np.ndarray:
    T = np.eye(3)
    T[:2, :2] = rot2(yaw_rad)
    T[:2, 2] = t
    return T


def apply_T2(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    return pts @ T[:2, :2].T + T[:2, 2]


def plan_T_to_3d(T2: np.ndarray, dy: float) -> np.ndarray:
    """Lift a plan transform on (u, v) = (x, z) to the 3D aligned world (y up), adding a vertical offset dy."""
    T3 = np.eye(4)
    R = T2[:2, :2]
    T3[np.ix_([0, 2], [0, 2])] = R
    T3[[0, 2], 3] = T2[:2, 2]
    T3[1, 3] = dy
    return T3


# ----------------------------------------------------------------------------------------------- rasters + FFT search

def _channel_rasters(pts: np.ndarray, nrm: np.ndarray, origin: np.ndarray, shape: tuple[int, int],
                     res: float, blur: float, split: bool = True) -> np.ndarray:
    """(4, H, W) wall rasters, one per normal direction. Cell value = sqrt(count): tall walls weigh more than low
    furniture faces, but one very dense wall does not drown everything else."""
    ch = np.argmax(nrm @ _AXES.T, axis=1) if split else np.zeros(len(pts), int)
    ij = np.floor((pts - origin) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
    out = np.zeros((4, *shape))
    np.add.at(out, (ch[ok], ij[ok, 1], ij[ok, 0]), 1.0)
    out = np.sqrt(out)
    if blur > 0:
        out = np.stack([ndimage.gaussian_filter(o, blur) for o in out])
    return out


def _grid(pts: np.ndarray, res: float, margin: float = 0.3) -> tuple[np.ndarray, tuple[int, int]]:
    lo = pts.min(0) - margin
    hi = pts.max(0) + margin
    W, H = np.ceil((hi - lo) / res).astype(int) + 1
    return lo, (int(H), int(W))


def _xcorr_peak(FA: np.ndarray, rb: np.ndarray, pad: tuple[int, int]) -> tuple[float, np.ndarray]:
    """Peak of sum_c xcorr(A_c, B_c) via FFT. Returns (value, shift in cells as (dx, dy)), sub-cell by parabola."""
    FB = np.fft.rfft2(rb, s=pad)
    c = np.fft.irfft2((FA * np.conj(FB)).sum(0), s=pad)
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    sub = []
    for axis, i in ((1, ix), (0, iy)):
        n = c.shape[axis]
        line = c[iy, [(i - 1) % n, i, (i + 1) % n]] if axis == 1 else c[[(i - 1) % n, i, (i + 1) % n], ix]
        den = line[0] - 2 * line[1] + line[2]
        d = 0.5 * (line[0] - line[2]) / den if den < 0 else 0.0
        s = i + d
        sub.append(s - n if s > n / 2 else s)       # circular index -> signed shift
    return float(c[iy, ix]), np.array(sub)


def _search(A: Evidence2D, B: Evidence2D, res: float, yaws_rad: np.ndarray, blur: float, split: bool = True):
    """For each yaw, the translation maximising wall-raster correlation. Returns list of (score, yaw, t)."""
    oA, shA = _grid(A.wall_pts, res)
    rA = _channel_rasters(A.wall_pts, A.wall_normals, oA, shA, res, blur, split)
    normA = np.sqrt((rA ** 2).sum())
    out = []
    for yaw in yaws_rad:
        R = rot2(yaw)
        pb, nb = B.wall_pts @ R.T, B.wall_normals @ R.T
        oB, shB = _grid(pb, res)
        rB = _channel_rasters(pb, nb, oB, shB, res, blur, split)
        pad = (shA[0] + shB[0], shA[1] + shB[1])
        FA = np.fft.rfft2(rA, s=pad)
        val, shift = _xcorr_peak(FA, rB, pad)
        t = oA - oB + shift * res      # B cell x lands on A cell x + shift
        out.append((val / (normA * np.sqrt((rB ** 2).sum()) + 1e-12), float(yaw), t))
    return out


def global_search(A: Evidence2D, B: Evidence2D, p: RegisterParams):
    """Coarse (4 quadrants x fine yaw grid) then fine (around the winner). Returns best (score, yaw, t), the
    per-quadrant curves and the distinctiveness ratio."""
    fine = np.radians(np.arange(-p.yaw_range_deg, p.yaw_range_deg + 1e-9, p.coarse_yaw_step_deg))
    curves, best_per_q = {}, {}
    for q in range(4):
        res = _search(A, B, p.coarse_res_m, q * np.pi / 2 + fine, p.blur_cells, p.split_normals)
        curves[q * 90] = [(round(np.degrees(y), 3), round(s, 5)) for s, y, _ in res]
        best_per_q[q] = max(res, key=lambda r: r[0])
    q_best = max(best_per_q, key=lambda q: best_per_q[q][0])
    others = [best_per_q[q][0] for q in best_per_q if q != q_best]
    distinct = best_per_q[q_best][0] / max(max(others), 1e-12)
    y0 = best_per_q[q_best][1]
    ref = np.radians(np.arange(-p.coarse_yaw_step_deg, p.coarse_yaw_step_deg + 1e-9, p.fine_yaw_step_deg))
    fine_res = _search(A, B, p.fine_res_m, y0 + ref, p.blur_cells, p.split_normals)
    s, yaw, t = max(fine_res, key=lambda r: r[0])
    at_edge = abs(abs(y0 - q_best * np.pi / 2) - np.radians(p.yaw_range_deg)) < 1e-6
    return (best_per_q[q_best][0], yaw, t), curves, float(distinct), bool(at_edge)


# ----------------------------------------------------------------------------------------------- ICP

def _subsample(pts: np.ndarray, nrm: np.ndarray, n: int, seed: int):
    if len(pts) <= n:
        return pts, nrm
    idx = np.random.default_rng(seed).choice(len(pts), n, replace=False)
    return pts[idx], nrm[idx]


def _correspond(tree: cKDTree, A: Evidence2D, pb: np.ndarray, nb: np.ndarray, max_d: float, cos_min: float):
    """Nearest A wall point within max_d whose normal agrees with B's. Returns (idx_b, idx_a, residual)."""
    d, ia = tree.query(pb, k=4, distance_upper_bound=max_d)
    ok = np.isfinite(d)
    ia_safe = np.where(ok, ia, 0)
    agree = ok & ((A.wall_normals[ia_safe] * nb[:, None, :]).sum(-1) > cos_min)
    d = np.where(agree, d, np.inf)
    k = np.argmin(d, axis=1)
    rows = np.arange(len(pb))
    keep = np.isfinite(d[rows, k])
    ib, ia_sel = rows[keep], ia_safe[rows, k][keep]
    r = ((pb[ib] - A.wall_pts[ia_sel]) * A.wall_normals[ia_sel]).sum(1)
    return ib, ia_sel, r


def icp_point_to_line(A: Evidence2D, B: Evidence2D, T0: np.ndarray, p: RegisterParams):
    """Refine T0 by minimising sum huber(n_a . (T p_b - q_a)). Returns (T, residuals, J, w) of the final step."""
    tree = cKDTree(A.wall_pts)
    pb0, nb0 = _subsample(B.wall_pts, B.wall_normals, p.max_icp_points, p.seed)
    T = T0.copy()
    for max_d in p.icp_max_dist_m:
        for _ in range(5):
            pb, nb = apply_T2(T, pb0), nb0 @ T[:2, :2].T
            ib, ia, r = _correspond(tree, A, pb, nb, max_d, p.icp_normal_cos)
            if len(ib) < 50:
                break
            n = A.wall_normals[ia]
            # d/dtheta of R p about the current point: perp(p) = (-p_v, p_u); linearised residual model
            J = np.c_[(n * np.c_[-pb[ib, 1], pb[ib, 0]]).sum(1), n]
            w = np.minimum(1.0, p.huber_m / (np.abs(r) + 1e-12))
            H = J.T @ (J * w[:, None])
            dx = -np.linalg.solve(H + 1e-9 * np.eye(3), J.T @ (w * r))
            T = make_T2(dx[0], dx[1:]) @ T
            if np.abs(dx[0]) < 1e-6 and np.linalg.norm(dx[1:]) < 1e-5:
                break
    pb, nb = apply_T2(T, pb0), nb0 @ T[:2, :2].T
    ib, ia, r = _correspond(tree, A, pb, nb, p.inlier_m, p.icp_normal_cos)
    n = A.wall_normals[ia]
    J = np.c_[(n * np.c_[-pb[ib, 1], pb[ib, 0]]).sum(1), n]
    return T, r, J, len(pb0)


def _uncertainty(r: np.ndarray, J: np.ndarray, centre: np.ndarray) -> tuple[float, tuple[float, float], float]:
    """1-sigma of yaw and translation from the Gauss-Newton Hessian (sigma^2 (J^T J)^-1), plus the translation
    constraint ratio. Rotation is re-expressed about the overlap centre so the translation sigma is not inflated by
    a far-away origin."""
    if len(r) < 10:
        return float("nan"), (float("nan"), float("nan")), 0.0
    Jc = J.copy()
    Jc[:, 0] -= J[:, 1] * -centre[1] + J[:, 2] * centre[0]   # rotation about the centre instead of the origin
    H = Jc.T @ Jc
    cov = np.linalg.pinv(H) * float(np.mean(r ** 2))
    ev = np.linalg.eigvalsh(H[1:, 1:])
    return (float(np.degrees(np.sqrt(cov[0, 0]))), (float(np.sqrt(cov[1, 1])), float(np.sqrt(cov[2, 2]))),
            float(ev[0] / max(ev[1], 1e-12)))


# ----------------------------------------------------------------------------------------------- overlap

def floor_mask(pts: np.ndarray, origin: np.ndarray, shape: tuple[int, int], res: float) -> np.ndarray:
    """Binary floor raster with small holes closed (furniture legs, sparse returns)."""
    m = np.zeros(shape, bool)
    ij = np.floor((pts - origin) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
    m[ij[ok, 1], ij[ok, 0]] = True
    st = ndimage.generate_binary_structure(2, 1)
    return ndimage.binary_fill_holes(ndimage.binary_closing(m, st, iterations=3))


def floor_overlap(A: Evidence2D, B: Evidence2D, T: np.ndarray, res: float) -> tuple[float, float, float]:
    pb = apply_T2(T, B.floor_pts)
    if len(A.floor_pts) == 0 or len(pb) == 0:
        return 0.0, 0.0, 0.0
    origin, shape = _grid(np.vstack([A.floor_pts, pb]), res)
    ma, mb = floor_mask(A.floor_pts, origin, shape, res), floor_mask(pb, origin, shape, res)
    inter = float((ma & mb).sum())
    return inter / max((ma | mb).sum(), 1), inter / max(ma.sum(), 1), inter / max(mb.sum(), 1)


# ----------------------------------------------------------------------------------------------- entry point

def register(A: Evidence2D, B: Evidence2D, p: RegisterParams = RegisterParams()) -> Registration:
    """Register B onto A. Deterministic: no random restarts; the only sampling uses a fixed seed."""
    (score, yaw, t), curves, distinct, at_edge = global_search(A, B, p)
    T, r, J, n_b = icp_point_to_line(A, B, make_T2(yaw, t), p)
    centre = apply_T2(T, B.wall_pts).mean(0) if len(B.wall_pts) else np.zeros(2)
    s_yaw, s_t, ratio = _uncertainty(r, J, centre)
    iou, ov_a, ov_b = floor_overlap(A, B, T, p.overlap_res_m)
    yaw_final = float(np.arctan2(T[1, 0], T[0, 0]))
    return Registration(
        a=A.name, b=B.name, T2=T.tolist(), T3=plan_T_to_3d(T, A.floor_y - B.floor_y).tolist(),
        yaw_deg=float(np.degrees(yaw_final)), t=(float(T[0, 2]), float(T[1, 2])), score=float(score),
        distinctiveness=distinct, yaw_at_search_edge=at_edge,
        residual_rmse_m=float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan"),
        residual_median_m=float(np.median(np.abs(r))) if len(r) else float("nan"),
        inlier_frac_b=len(r) / max(n_b, 1), overlap_iou=iou, overlap_of_a=ov_a, overlap_of_b=ov_b,
        sigma_yaw_deg=s_yaw, sigma_t_m=s_t, constraint_ratio=ratio, yaw_curve={str(k): v for k, v in curves.items()})


def wall_residual_map(A: Evidence2D, B: Evidence2D, T: np.ndarray,
                      max_d: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Per B wall point (after T): unsigned distance to the nearest A wall point (capped at max_d).

    After a good global fit, the spatial pattern of these residuals shows where the two captures disagree locally,
    i.e. where one of them drifted."""
    pb = apply_T2(T, B.wall_pts)
    d, _ = cKDTree(A.wall_pts).query(pb, distance_upper_bound=max_d)
    return pb, np.minimum(d, max_d)


# ----------------------------------------------------------------------------------------------- consistency checks

def _yaw_of(T: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))


def cycle_error(T_ab: np.ndarray, T_bc: np.ndarray, T_ac: np.ndarray, probe_pts: np.ndarray) -> dict:
    """Loop A<-B<-C versus A<-C directly (T_xy maps y's coordinates into x's). For consistent captures and correct
    registrations the composition is the identity. A residual means at least one capture is internally bent
    (drift) or one registration is wrong. probe_pts (in C's frame) measure the effect in metres."""
    D = np.linalg.inv(T_ac) @ T_ab @ T_bc
    disp = np.linalg.norm(apply_T2(D, probe_pts) - probe_pts, axis=1) if len(probe_pts) else np.zeros(1)
    return dict(yaw_deg=_yaw_of(D), max_displacement_m=float(disp.max()), mean_displacement_m=float(disp.mean()))


def local_disagreement(A: Evidence2D, B: Evidence2D, T: np.ndarray, p: RegisterParams = RegisterParams(),
                       tile_m: float = 3.0, min_pts: int = 1500) -> list[dict]:
    """Split B into tiles, re-register each tile alone (starting from the global T), and report how far each tile
    wants to move. One global rigid transform cannot absorb drift, so tiles that move a lot are where the two
    captures' trajectories disagree. A tile whose walls all run one way can slide along them freely, so tiles with
    a translation constraint ratio <= 0.1 are flagged as not well constrained (their displacement is meaningless).
    Returns one dict per tile: centre, points, local yaw and displacement, constraint ratio."""
    local_p = RegisterParams(**{**asdict(p), "icp_max_dist_m": (0.3, 0.15, 0.08, 0.05, 0.03, 0.03)})
    pb = apply_T2(T, B.wall_pts)
    keys = np.floor(pb / tile_m).astype(int)
    out = []
    for key in sorted({tuple(k) for k in keys}):
        sel = np.all(keys == key, axis=1)
        if sel.sum() < min_pts:
            continue
        sub = Evidence2D(B.name, B.wall_pts[sel], B.wall_normals[sel], np.zeros((0, 2)))
        T_loc, r, J, n = icp_point_to_line(A, sub, T, local_p)
        c = B.wall_pts[sel].mean(0)
        D = T_loc @ np.linalg.inv(T)
        ratio = _uncertainty(r, J, apply_T2(T_loc, c)[0])[2]
        out.append(dict(centre=apply_T2(T, c)[0].tolist(), n_points=int(sel.sum()), dyaw_deg=_yaw_of(D),
                        constraint_ratio=ratio, well_constrained=bool(ratio > 0.1),
                        displacement_m=(apply_T2(T_loc, c) - apply_T2(T, c))[0].tolist(),
                        rmse_m=float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan"),
                        inlier_frac=len(r) / max(n, 1)))
    return out
