#!/usr/bin/env python
"""Why are LiDAR wall-to-wall distances short against the laser? Test five hypotheses with evidence.

Finding to explain (docs/modules/arkitscenes_validation.md, Issue 3): on ARKitScenes 47895909 (iPad LiDAR vs Faro
laser) the 43 facing pairs fit error = -1.47 cm - 0.89 cm/m * distance; room box -1.52 cm (depth), -1.92 cm (height).

Hypotheses and the test this script runs for each (all numbers -> <out>/results.json, figures -> <out>/*.png):

  H1 sensor depth bias that grows with range (a depth SCALE error).
     Test that needs no laser and no registration: the same 10 cm patch of wall is seen by many frames from different
     ranges. With per-cell fixed effects, the slope of "depth along the ray minus the patch's mean" against range is
     the depth scale error relative to the poses (s_depth - s_pose). Run per surface, pooled, leave-one-surface-out.
     Absolute version: depth error against the laser plane binned by range and by incidence angle.
  H2 pose / trajectory scale error (ARKit VIO scale) against the laser.
     Similarity (7-DoF) and per-axis-scale (9-DoF) point-to-plane registration of our raw points to the laser, on
     several subsets, with a spatial block bootstrap. A real VIO scale error is isotropic and the same on every subset.
  H3 the registration (ICP on partial overlap) pulls the cloud.
     Re-measure every distance with the laser moved by +-1 cm / +-0.3 deg, and with an independently estimated
     transform. Distances between facing planes are invariant to translation, so H3 can only move per-surface offsets.
  H4 our measurement method biases surfaces inward (one-sided noise, clutter, corner rounding, de-duplication).
     Re-measure each matched surface with 10 estimators (trimmed LS = current, mean, median, KDE mode, RANSAC + LS,
     1.0-1.5 m band, all observations without de-duplication, 1/sigma(r)^2 weighting, near-range only, interior only);
     the laser reference is held fixed. Also: the unit of analysis (43 pairs share ~20 surfaces).
  H5 the laser (or its registration) is wrong.
     Laser plane residuals and estimator spread; pairs on well-seen surfaces that agree to < 1 mm.

The per-surface "inward offset" (ours minus laser, + = into the room) contains a residual registration translation.
We separate them with the model  inward_s = b + sign_s * t_axis(s)  (b = shared bias, t = translation per axis),
which is the registration-free per-surface bias that a correction would remove.

Usage:
  PYTHONPATH=. python scripts/investigate_lidar_bias.py arkit --scene <raw/Training/47895909> --laser <191738.ply>
  PYTHONPATH=. python scripts/investigate_lidar_bias.py stray --capture <Stray capture dir> --scene-dir outputs/<name>
  PYTHONPATH=. python scripts/investigate_lidar_bias.py summary            # pools every room under outputs/lidar_bias
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floorplan.config import Config  # noqa: E402
from floorplan.io.arkitscenes import Z_UP_TO_Y_UP, load_arkitscenes  # noqa: E402
from floorplan.recon.keyframes import select_keyframes  # noqa: E402
from floorplan.uncertainty.noise import sigma_lidar  # noqa: E402

_spec = importlib.util.spec_from_file_location("validate_arkitscenes", ROOT / "scripts" / "validate_arkitscenes.py")
V = importlib.util.module_from_spec(_spec)
sys.modules["validate_arkitscenes"] = V   # dataclasses look their module up in sys.modules
_spec.loader.exec_module(V)            # reuse the validation's stage 1, registration and plane matching unchanged

OUT = ROOT / "outputs" / "lidar_bias"
WINDOW_M = 0.03            # points within 3 cm of the seed plane belong to the surface (same as the validation)
FE_CELL_M = 0.10           # fixed-effect patch size for the near-vs-far test
MIN_OBS_PER_CELL = 20
MIN_RANGE_SPREAD_M = 0.3   # a patch only informs the slope if it was seen from ranges spanning >= 30 cm
INTERIOR_M = 0.30          # "interior" = at least 30 cm (in-plane) from the edge of our surface patch
MAX_INC_DEG = 70.0
BAND_M = (1.0, 1.5)        # laser-measurer band above the floor
NEAR_M = 1.5
RANGE_BINS = (0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
INC_BINS = (0, 15, 30, 45, 60, 75)
EDGE_BINS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
N_BOOT = 200
RNG = np.random.default_rng(0)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"


# ================================================================================ 1-D offset estimators
def est_trimmed(x: np.ndarray, w: np.ndarray | None = None) -> float:
    """Offset-only version of the validation's trimmed LS: (weighted) mean, refitted on points within 3 -> 1 cm."""
    w = np.ones_like(x) if w is None else w
    m = np.average(x, weights=w)
    for thr in V.TRIM_SCHEDULE_M:
        k = np.abs(x - m) < thr
        if k.sum() < 20:
            return float("nan")
        m = np.average(x[k], weights=w[k])
    return float(m)


def est_mode(x: np.ndarray, bw: float = 0.002) -> float:
    """Peak of a Gaussian KDE (2 mm bandwidth) on a 0.5 mm grid: the densest offset, insensitive to one-sided tails."""
    g = np.arange(x.min() - 4 * bw, x.max() + 4 * bw + 5e-4, 5e-4)     # padded: kernel never longer than grid
    h, e = np.histogram(x, np.append(g, g[-1] + 5e-4))
    k = np.exp(-0.5 * (np.arange(-4 * bw, 4 * bw + 5e-4, 5e-4) / bw) ** 2)
    s = np.convolve(h, k, mode="same")
    return float(g[int(np.argmax(s))] + 2.5e-4)


def est_ransac_ls(P: np.ndarray, n_ref: np.ndarray, mid: np.ndarray, axis: int, iters: int = 300) -> float:
    """Plane RANSAC (1 cm threshold, fixed seed) then least squares on its inliers; offset where the plane crosses the
    measuring line through `mid`, expressed along the reference normal."""
    if len(P) < 50:
        return float("nan")
    rng = np.random.default_rng(1)
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ n_ref) < 0.95:
            continue
        inl = np.abs((P - a) @ n) < 0.01
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None:
        return float("nan")
    Q = P[best]
    mu = Q.mean(0)
    n = np.linalg.svd(Q - mu, full_matrices=False)[2][2]
    rest = sum(n[a] * mid[a] for a in range(3) if a != axis)
    q = mid.copy()
    q[axis] = (n @ mu - rest) / n[axis]
    return float(n_ref @ q)


# ================================================================================ observations (no de-duplication)
def collect_observations(cap, frames: np.ndarray, T_align: np.ndarray, cfg: Config, stride: int = 2) -> dict:
    """Every high-confidence depth sample of the keyframes (stride 2), in the aligned frame, with its range, viewing
    ray and frame id. Unlike collect_raw_points we keep ALL observations: the near-vs-far test needs the same patch seen
    from several ranges, and de-duplication (one point per 5 mm voxel) changes how points are weighted."""
    from floorplan.io.stray import DEPTH_H, DEPTH_W
    v, u = np.mgrid[0:DEPTH_H:stride, 0:DEPTH_W:stride]
    R, t = T_align[:3, :3], T_align[:3, 3]
    P, Rg, D, F = [], [], [], []
    for i in frames:
        d = cap.depth(i, cfg.min_conf, cfg.min_depth_m, cfg.max_depth_m)[v, u]
        ok = d > 0
        if ok.sum() < 50:
            continue
        K = cap.K_depth(i)
        z = d[ok]
        pc = np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], 1)
        T = T_align @ cap.T_wc[i]
        pw = pc @ T[:3, :3].T + T[:3, 3]
        r = np.linalg.norm(pc, axis=1)
        P.append(pw.astype(np.float32))
        Rg.append(r.astype(np.float32))
        D.append(((pw - T[:3, 3]) / r[:, None]).astype(np.float32))
        F.append(np.full(len(z), i, np.int32))
    return dict(P=np.concatenate(P).astype(np.float64), r=np.concatenate(Rg).astype(np.float64),
                ray=np.concatenate(D).astype(np.float64), frame=np.concatenate(F))


# ================================================================================ per-surface data
def surface_mid(s) -> np.ndarray:
    return (s.lo + s.hi) / 2


def offset_at(plane, axis: int, mid: np.ndarray, n_ref: np.ndarray) -> float:
    q = mid.copy()
    q[axis] = plane.coord_at(axis, mid)
    return float(n_ref @ q)


def edge_distance(X: np.ndarray, axis: int, Q: np.ndarray, cell: float = 0.05) -> np.ndarray:
    """In-plane distance (m) from each query point Q to the edge of the patch occupied by X (5 cm grid with a
    distance transform). The edge of a wall patch is a corner, an opening or an occluder: where depth is rounded."""
    if len(Q) == 0 or len(X) == 0:
        return np.zeros(len(Q))
    a, b = [k for k in range(3) if k != axis]
    lo = np.minimum(X[:, [a, b]].min(0), Q[:, [a, b]].min(0)) - 2 * cell
    ij = np.floor((X[:, [a, b]] - lo) / cell).astype(int)
    shape = ij.max(0) + 3
    cnt = np.zeros(shape, int)
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
    occ = cnt >= max(3, int(np.percentile(cnt[cnt > 0], 50) * 0.2))
    dt = distance_transform_edt(occ) * cell
    qi = np.clip(np.floor((Q[:, [a, b]] - lo) / cell).astype(int), 0, np.array(shape) - 1)
    return dt[qi[:, 0], qi[:, 1]]


def surface_records(matches: list, ours_fc, laser_fc, obs: dict, obs_tree: cKDTree, floor_y: float) -> list[dict]:
    """For each matched surface: the laser plane (reference), our dedup raw points and all observations on the stretch
    both sensors see, their offsets along the laser normal, range, incidence and distance to the patch edge."""
    recs = []
    for k, m in enumerate(matches):
        s, ls = m["ours"], m["laser"]
        axis, sign = V.FAMILIES[s.family]
        n = np.asarray(ls.plane.n)
        mid = surface_mid(s)
        o_laser_tls = offset_at(ls.plane, axis, mid, n)
        o_ours_tls = offset_at(s.plane, axis, mid, n)
        X = ours_fc.fam[s.family]
        Xs = X[(np.abs(X @ np.asarray(s.plane.n) - s.plane.d) < V.SEED_WINDOW_M)]
        full = Xs                                                # our whole patch (for the edge distance)
        in_c = np.isin(V.cell_keys(Xs, axis), s.cells)
        Xc = Xs[in_c]
        L = laser_fc.fam[s.family]
        Lc = L[(np.abs(L @ n - ls.plane.d) < V.SEED_WINDOW_M) & np.isin(V.cell_keys(L, axis), s.cells)]
        # observations: within 3 cm of OUR plane, inside the common cells, seen from the room side
        cand = obs_tree.query_ball_point(mid, r=float(np.linalg.norm(s.hi - s.lo) / 2 + 0.05))
        cand = np.asarray(cand, int)
        Po = obs["P"][cand]
        keep = (np.abs(Po @ np.asarray(s.plane.n) - s.plane.d) < WINDOW_M) & np.isin(V.cell_keys(Po, axis), s.cells)
        cosi = -(obs["ray"][cand] @ n)
        keep &= cosi > math.cos(math.radians(MAX_INC_DEG))
        idx = cand[keep]
        Po = obs["P"][idx]
        rec = dict(k=k, family=s.family, axis=axis, sign=sign, n=n, mid=mid, area=m["area_m2"],
                   common=m["common_area_m2"], o_laser=o_laser_tls, o_ours_tls=o_ours_tls,
                   X=Xc, xo=Xc @ n, L=Lc, xl=Lc @ n, tall=bool(s.hi[1] > floor_y + V.FULL_HEIGHT_M),
                   obs_idx=idx, obs_x=Po @ n, obs_r=obs["r"][idx], obs_cos=-(obs["ray"][idx] @ n),
                   obs_edge=edge_distance(full, axis, Po), X_edge=edge_distance(full, axis, Xc),
                   obs_y=Po[:, 1] - floor_y, X_y=Xc[:, 1] - floor_y, obs_P=Po)
        # inward offset (+ = our surface is in front of the laser, into the room) = (o_ours - o_laser), because n
        # is the laser normal pointing into the room
        rec["inward_tls_cm"] = 100 * (o_ours_tls - o_laser_tls)
        recs.append(rec)
    return recs


# ================================================================================ H4 estimators
def estimators(rec: dict) -> dict:
    """Our surface position (along the laser normal) by each estimator. The laser reference stays trimmed LS."""
    x, o0 = rec["xo"], rec["o_ours_tls"]
    w = np.abs(x - o0) < WINDOW_M
    xw = x[w]
    ox, orr, oc, oe, oy = rec["obs_x"], rec["obs_r"], rec["obs_cos"], rec["obs_edge"], rec["obs_y"]
    ow = np.abs(ox - o0) < WINDOW_M
    out = dict(trimmed_ls_plane=o0)
    if len(xw) < 50:
        return out
    out["mean_3cm"] = float(xw.mean())
    out["median_3cm"] = float(np.median(xw))
    out["trimmed_offset"] = est_trimmed(xw)
    out["kde_mode"] = est_mode(xw)
    out["ransac_ls"] = est_ransac_ls(rec["X"][w], rec["n"], rec["mid"], rec["axis"])
    if rec["axis"] != 1:
        b = w & (rec["X_y"] > BAND_M[0]) & (rec["X_y"] < BAND_M[1])
        out["band_1.0_1.5m"] = est_trimmed(x[b]) if b.sum() >= 50 else float("nan")
    if ow.sum() >= 50:
        out["all_obs_trimmed"] = est_trimmed(ox[ow])
        out["all_obs_mode"] = est_mode(ox[ow])
        out["all_obs_weighted"] = est_trimmed(ox[ow], 1.0 / sigma_lidar(orr[ow]) ** 2)
        nr = ow & (orr < NEAR_M)
        out["near_obs_r<1.5m"] = est_trimmed(ox[nr]) if nr.sum() >= 50 else float("nan")
        it = ow & (oe >= INTERIOR_M)
        out["interior_obs_0.3m"] = est_trimmed(ox[it]) if it.sum() >= 50 else float("nan")
    return out


def laser_estimators(rec: dict) -> dict:
    """H5: does the laser's own surface position depend on the estimator? (it should not, by more than ~1 mm)."""
    x = rec["xl"]
    w = np.abs(x - rec["o_laser"]) < WINDOW_M
    if w.sum() < 50:
        return {}
    xw = x[w]
    return dict(trimmed_ls_plane=rec["o_laser"], mean_3cm=float(xw.mean()), median_3cm=float(np.median(xw)),
                trimmed_offset=est_trimmed(xw), kde_mode=est_mode(xw),
                rmse_mm=float(np.sqrt(np.mean((xw - rec["o_laser"]) ** 2)) * 1000))


# ================================================================================ bias/translation separation
def bias_translation_fit(recs: list[dict], inward: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """inward_s = b + sign_s * t_axis(s): b = shared inward bias of every surface, t = residual registration
    translation per axis (x, y, z). Registration-free because a translation moves facing surfaces in opposite
    inward directions. Needs both orientations on an axis for that axis' t to be identifiable."""
    ok = np.isfinite(inward)
    A = np.zeros((ok.sum(), 4))
    rows = [r for r, o in zip(recs, ok) if o]
    for i, r in enumerate(rows):
        A[i, 0] = 1.0
        A[i, 1 + r["axis"]] = r["sign"]
    y = inward[ok]
    w = np.ones(len(y)) if weights is None else weights[ok]
    keep = np.ones(4, bool)
    for a in range(3):                       # an axis seen from one side only: t not identifiable -> drop surfaces
        signs = {r["sign"] for r in rows if r["axis"] == a}
        if len(signs) < 2:
            keep[1 + a] = False
    use = np.array([keep[1 + r["axis"]] for r in rows])
    if use.sum() < 3:
        return dict(b_cm=float("nan"), n=int(use.sum()))
    Aw = A[use][:, keep] * np.sqrt(w[use])[:, None]
    sol, *_ = np.linalg.lstsq(Aw, y[use] * np.sqrt(w[use]), rcond=None)
    t = np.full(3, np.nan)
    t[keep[1:]] = sol[1:]
    resid = y[use] - A[use][:, keep] @ sol
    return dict(b_cm=float(sol[0]), t_cm=t.tolist(), n=int(use.sum()),
                resid_mad_cm=float(1.4826 * np.median(np.abs(resid - np.median(resid)))))


# ================================================================================ pairs from offsets
def facing_pairs_idx(recs: list[dict], min_gap: float = V.MIN_FACING_GAP_M) -> list[tuple[int, int, int]]:
    """(low-side surface, high-side surface, axis) for every facing pair, same rule as the validation."""
    pairs, ss = [], recs
    for lo_f, hi_f, axis in (("+x", "-x", 0), ("+z", "-z", 2), ("floor", "ceiling", 1)):
        horiz = [k for k in (0, 2) if k != axis]
        for a in ss:
            if a["family"] != lo_f:
                continue
            for b in ss:
                if b["family"] != hi_f:
                    continue
                A, B = a["_s"], b["_s"]
                if all(min(A.hi[k], B.hi[k]) > max(A.lo[k], B.lo[k]) for k in horiz):
                    if abs(b["o_laser"] + a["o_laser"]) >= min_gap:   # normals opposite: distance = -(oa + ob)
                        pairs.append((a["k"], b["k"], axis))
    return pairs


def pair_errors(recs: list[dict], pairs: list, inward: np.ndarray) -> np.ndarray:
    """Distance error of a facing pair = -(inward_a + inward_b): each surface standing x cm into the room shortens
    the distance by x cm."""
    return np.array([-(inward[a] + inward[b]) for a, b, _ in pairs])


def fit_offset_slope(d: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    A = np.stack([np.ones_like(d), d], 1)
    (c0, c1), *_ = np.linalg.lstsq(A, e, rcond=None)
    return float(c0), float(c1)


def cluster_bootstrap_slope(recs: list[dict], pairs: list, inward: np.ndarray) -> dict:
    """The 43 pairs reuse ~20 surfaces, so they are NOT 43 independent samples. Resample SURFACES (clusters) and refit
    error = offset + slope * distance to get honest intervals."""
    d = np.array([abs(recs[a]["o_laser"] + recs[b]["o_laser"]) for a, b, _ in pairs])     # m
    e = pair_errors(recs, pairs, inward) * 100                                              # cm
    c0, c1 = fit_offset_slope(d, e)
    ks = np.array(sorted({k for a, b, _ in pairs for k in (a, b)}))
    boots = []
    for _ in range(N_BOOT):
        pick = RNG.choice(ks, len(ks), replace=True)
        cnt = {k: int((pick == k).sum()) for k in ks}
        w = np.array([cnt[a] * cnt[b] for a, b, _ in pairs], float)
        if (w > 0).sum() < 4 or len(np.unique(d[w > 0])) < 3:
            continue
        A = np.stack([np.ones_like(d), d], 1) * np.sqrt(w)[:, None]
        sol, *_ = np.linalg.lstsq(A, e * np.sqrt(w), rcond=None)
        boots.append(sol)
    boots = np.array(boots)
    return dict(n_pairs=len(pairs), n_surfaces=int(len(ks)), offset_cm=c0,
                slope_cm_per_m=c1, offset_ci95=np.percentile(boots[:, 0], [2.5, 97.5]).tolist(),
                slope_ci95=np.percentile(boots[:, 1], [2.5, 97.5]).tolist())


# ================================================================================ H1: near vs far
def near_far_slope(x: np.ndarray, r: np.ndarray, cosi: np.ndarray, P: np.ndarray, axis: int,
                   groups: np.ndarray | None = None) -> dict:
    """Slope of depth error vs range within 10 cm patches (fixed effects). depth error along the ray = -(x - patch
    mean)/cos(incidence): a point in front of the surface means the depth was SHORT. Per-patch demeaning removes the
    plane position, any registration error and wall unevenness; what is left is how the measured position depends on
    the range it was seen from: (depth scale error) - (pose scale error)."""
    a, b = [k for k in range(3) if k != axis]
    key = (np.floor(P[:, a] / FE_CELL_M).astype(np.int64) << 21) + np.floor(P[:, b] / FE_CELL_M).astype(np.int64)
    if groups is not None:
        key = key * 64 + groups
    dep = -x / np.clip(cosi, 0.2, 1)
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    rmin = np.full(len(uk), np.inf)
    rmax = np.full(len(uk), -np.inf)
    np.minimum.at(rmin, inv, r)
    np.maximum.at(rmax, inv, r)
    good = (cnt >= MIN_OBS_PER_CELL) & (rmax - rmin >= MIN_RANGE_SPREAD_M)
    m = good[inv]
    if m.sum() < 200:
        return dict(n=int(m.sum()), slope_mm_per_m=float("nan"))
    inv2 = np.unique(inv[m], return_inverse=True)[1]
    dm = dep[m] - (np.bincount(inv2, dep[m]) / np.bincount(inv2))[inv2]
    rm = r[m] - (np.bincount(inv2, r[m]) / np.bincount(inv2))[inv2]
    # robust: drop gross outliers (> 3 cm from the patch mean) and refit once
    k = np.abs(dm) < 0.03
    num, den = float(rm[k] @ dm[k]), float(rm[k] @ rm[k])
    resid = dm[k] - num / den * rm[k]
    se = float(np.sqrt(np.mean(resid ** 2) / den))       # iid standard error (optimistic: obs are correlated)
    return dict(n=int(k.sum()), cells=int(good.sum()), slope_mm_per_m=num / den * 1000, se_mm_per_m=se * 1000,
                num=num, den=den, range_spread_m=float(np.percentile(rm[k], 95) - np.percentile(rm[k], 5)))


def fe_cos_offset(x: np.ndarray, cosi: np.ndarray, P: np.ndarray, axis: int) -> dict:
    """Internal (laser-free) estimate of a CONSTANT depth offset. If every depth reading is short by b_d, a surface
    point seen at incidence theta moves into the room by b_d * cos(theta). Within a 10 cm patch (fixed effect), regress
    the observed offset along the normal on cos(theta): the slope is b_d. A range-proportional error would instead
    show in near_far_slope. num/den returned for pooling across surfaces (pool_slopes)."""
    a, b = [k for k in range(3) if k != axis]
    key = (np.floor(P[:, a] / FE_CELL_M).astype(np.int64) << 21) + np.floor(P[:, b] / FE_CELL_M).astype(np.int64)
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    cmin = np.full(len(uk), np.inf)
    cmax = np.full(len(uk), -np.inf)
    np.minimum.at(cmin, inv, cosi)
    np.maximum.at(cmax, inv, cosi)
    good = (cnt >= MIN_OBS_PER_CELL) & (cmax - cmin >= 0.2)
    m = good[inv]
    if m.sum() < 200:
        return dict(n=int(m.sum()), slope_mm_per_m=float("nan"))
    inv2 = np.unique(inv[m], return_inverse=True)[1]
    xm = x[m] - (np.bincount(inv2, x[m]) / np.bincount(inv2))[inv2]
    cm = cosi[m] - (np.bincount(inv2, cosi[m]) / np.bincount(inv2))[inv2]
    k = np.abs(xm) < 0.03
    num, den = float(cm[k] @ xm[k]), float(cm[k] @ cm[k])
    resid = xm[k] - num / den * cm[k]
    # slope_mm_per_m is reused by pool_slopes; here it means "mm of depth offset" (x in m, cos unitless -> *1000 = mm)
    return dict(n=int(k.sum()), cells=int(good.sum()), slope_mm_per_m=num / den * 1000,
                se_mm_per_m=float(np.sqrt(np.mean(resid ** 2) / den)) * 1000, num=num, den=den,
                range_spread_m=float(np.percentile(cm[k], 95) - np.percentile(cm[k], 5)))


def h1_internal_cos(recs: list[dict]) -> dict:
    per = []
    for r in recs:
        m = np.abs(r["obs_x"] - r["o_ours_tls"]) < WINDOW_M
        if m.sum() < 200:
            continue
        res = fe_cos_offset(r["obs_x"][m] - r["o_ours_tls"], r["obs_cos"][m], r["obs_P"][m], r["axis"])
        if np.isfinite(res["slope_mm_per_m"]):
            per.append(dict(k=r["k"], family=r["family"], area=r["area"], **res))
    out = pool_slopes(per)
    return {k.replace("slope_mm_per_m", "depth_offset_mm"): v for k, v in out.items()}


def pool_slopes(per: list[dict]) -> dict:
    """Pooled fixed-effects slope = sum(num) / sum(den): surfaces seen over a wide range of ranges carry more
    information and get more weight (unlike averaging per-surface slopes). Cluster bootstrap over surfaces."""
    if not per:
        return dict(per_surface=[], n_surfaces=0)
    num = np.array([p["num"] for p in per])
    den = np.array([p["den"] for p in per])
    pooled = float(num.sum() / den.sum() * 1000)
    boots = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(per), len(per))
        boots.append(num[i].sum() / den[i].sum() * 1000)
    loo = [float((num.sum() - num[i]) / (den.sum() - den[i]) * 1000) for i in range(len(per))] if len(per) > 1 else []
    return dict(per_surface=[{k: v for k, v in p.items() if k not in ("num", "den")} for p in per],
                pooled_slope_mm_per_m=pooled, ci95_mm_per_m=np.percentile(boots, [2.5, 97.5]).tolist(),
                loo_range_mm_per_m=[min(loo), max(loo)] if loo else None, n_surfaces=len(per),
                median_slope_mm_per_m=float(np.median([p["slope_mm_per_m"] for p in per])))


def h1_internal(recs: list[dict], interior_only: bool = True) -> dict:
    per = []
    for r in recs:
        m = np.abs(r["obs_x"] - r["o_ours_tls"]) < WINDOW_M
        if interior_only:
            m &= r["obs_edge"] >= INTERIOR_M
        if m.sum() < 200:
            continue
        res = near_far_slope(r["obs_x"][m] - r["o_ours_tls"], r["obs_r"][m], r["obs_cos"][m], r["obs_P"][m],
                             r["axis"])
        if np.isfinite(res["slope_mm_per_m"]):
            per.append(dict(k=r["k"], family=r["family"], area=r["area"], **res))
    return pool_slopes(per)


def balanced_bias(recs: list[dict], value_key: str, bins: tuple, mask_fn=None) -> list[dict]:
    """Translation-free inward bias per bin of `value_key` (range, incidence or edge distance).

    A residual registration translation t moves every +axis surface by +t and every -axis surface by -t (inward).
    So for each axis, b = (median inward of + side + median inward of - side) / 2 cancels t exactly. Pooled over
    axes, weighted by the smaller side's count. Units: cm, + = our surface stands into the room (distances short)."""
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        bs, ws, n = [], [], 0
        for axis in (0, 1, 2):
            meds = {}
            for sgn in (1, -1):
                xs = []
                for r in recs:
                    if r["axis"] != axis or r["sign"] != sgn:
                        continue
                    m = np.abs(r["obs_x"] - r["o_ours_tls"]) < WINDOW_M
                    v = r[value_key]
                    m &= (v >= lo) & (v < hi)
                    if mask_fn is not None:
                        m &= mask_fn(r)
                    xs.append(r["obs_x"][m] - r["o_laser"])
                x = np.concatenate(xs) if xs else np.zeros(0)
                if len(x) >= 100:
                    meds[sgn] = (float(np.median(x)), len(x))
            if len(meds) == 2:
                bs.append((meds[1][0] + meds[-1][0]) / 2 * 100)
                ws.append(min(meds[1][1], meds[-1][1]))
                n += meds[1][1] + meds[-1][1]
        if bs:
            rows.append(dict(bin=[lo, hi], b_cm=float(np.average(bs, weights=ws)), n=int(n), axes=len(bs),
                             per_axis_cm=[round(b, 3) for b in bs]))
    return rows


def h1_absolute(recs: list[dict]) -> dict:
    """Translation-free inward bias against the laser by range, by incidence and by distance to the patch edge
    (corner rounding). A pure depth-scale error s would make b grow like s * range."""
    for r in recs:
        r["obs_inc"] = np.degrees(np.arccos(np.clip(r["obs_cos"], 0, 1)))
    interior = lambda r: r["obs_edge"] >= INTERIOR_M  # noqa: E731
    return dict(by_range=balanced_bias(recs, "obs_r", RANGE_BINS),
                by_range_interior=balanced_bias(recs, "obs_r", RANGE_BINS, interior),
                by_incidence=balanced_bias(recs, "obs_inc", INC_BINS),
                by_edge_distance=balanced_bias(recs, "obs_edge", EDGE_BINS))


# ================================================================================ H2: similarity registration
def similarity_icp(src: np.ndarray, tgt_P: np.ndarray, tgt_N: np.ndarray, mode: str, iters: int = 20,
                   boot: int = 0) -> dict:
    """Point-to-plane registration with a scale: mode 'iso' (7 DoF) or 'axis' (rotation, translation, one scale per
    aligned axis = 9 DoF). Gauss-Newton with trimmed correspondences (3 -> 1.5 cm). Starts at identity (the
    validation's rigid ICP already registered the clouds). Bootstrap: resample 0.5 m spatial blocks."""
    tree = cKDTree(tgt_P)
    c = src.mean(0)
    X = src.copy()
    scale = np.ones(3)
    thr_sched = np.linspace(0.03, 0.015, iters)

    def system(Xc, idx):
        q, n = tgt_P[idx], tgt_N[idx]
        res = np.einsum("ij,ij->i", Xc - q, n)
        J = [np.cross(Xc - c, n), n]
        if mode == "iso":
            J.append(np.einsum("ij,ij->i", Xc - c, n)[:, None])
        else:
            J.append((Xc - c) * n)
        return np.hstack(J), res

    for it in range(iters):
        d, idx = tree.query(X, workers=-1)
        m = d < 0.05
        J, res = system(X[m], idx[m])
        k = np.abs(res) < thr_sched[it]
        sol = np.linalg.lstsq(J[k], -res[k], rcond=None)[0]
        w, t, sc = sol[:3], sol[3:6], sol[6:]
        Rw = o3d.geometry.get_rotation_matrix_from_axis_angle(w)
        S = np.diag(1 + (np.repeat(sc, 3) if mode == "iso" else sc))
        X = (X - c) @ (Rw @ S).T + c + t
        scale *= 1 + (np.repeat(sc, 3) if mode == "iso" else sc)
    d, idx = tree.query(X, workers=-1)
    m = d < 0.05
    J, res = system(X[m], idx[m])
    k = np.abs(res) < 0.015
    out = dict(mode=mode, scale_minus_1_pct=((scale - 1) * 100).tolist(), n=int(k.sum()),
               rmse_mm=float(np.sqrt(np.mean(res[k] ** 2)) * 1000))
    if boot:
        Xm = X[m][k]
        blk = (np.floor(Xm[:, 0] / 0.5).astype(int) * 1000 + np.floor(Xm[:, 1] / 0.5).astype(int)) * 1000 \
            + np.floor(Xm[:, 2] / 0.5).astype(int)
        ub, binv = np.unique(blk, return_inverse=True)
        Jk, rk = J[k], res[k]
        sims = []
        for _ in range(boot):
            cnt = np.bincount(RNG.integers(0, len(ub), len(ub)), minlength=len(ub))
            wts = cnt[binv].astype(float)
            sol = np.linalg.lstsq(Jk * np.sqrt(wts)[:, None], -rk * np.sqrt(wts), rcond=None)[0]
            sims.append(sol[6:] * 100)
        sims = np.array(sims)
        base = (scale - 1) * 100 if mode == "axis" else np.array([(scale[0] - 1) * 100])
        out["ci95_pct"] = [(base[i] + np.percentile(sims[:, i], [2.5, 97.5])).tolist() for i in range(sims.shape[1])]
    return out


# ================================================================================ H3: registration perturbation
def perturb(T: np.ndarray, dt=(0, 0, 0), yaw_deg=0.0, tilt_deg=0.0) -> np.ndarray:
    R = o3d.geometry.get_rotation_matrix_from_xyz((math.radians(tilt_deg), math.radians(yaw_deg), 0.0))
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = dt
    return P @ T


def h3_registration(scene: dict, cfg: Config, laser: dict, T_LA: np.ndarray, centre: np.ndarray, floor_y: float,
                    ref_T: np.ndarray | None) -> list[dict]:
    variants = [("as_registered", T_LA)]
    for ax, name in ((0, "x"), (1, "y"), (2, "z")):
        for s in (-1, 1):
            dt = np.zeros(3)
            dt[ax] = 0.01 * s
            variants.append((f"shift_{name}{'+' if s > 0 else '-'}1cm", perturb(T_LA, dt)))
    variants += [("yaw+0.3deg", perturb(T_LA, yaw_deg=0.3)), ("yaw-0.3deg", perturb(T_LA, yaw_deg=-0.3)),
                 ("tilt+0.2deg", perturb(T_LA, tilt_deg=0.2))]
    if ref_T is not None:
        variants.append(("independent_smoke_test_transform", ref_T))
    EP, EN = laser["eval"]
    out = []
    for name, T in variants:
        LP, LN = V.transform(T, EP), EN @ T[:3, :3].T
        fc = V.family_cloud(LP, LN, V.EVAL_VOXEL_M)
        d = V.measure_against_laser(scene["raw_points"], scene["raw_normals"], cfg.raw_voxel_m, fc, centre, floor_y)
        offs = [m["inward_offset_cm"] for m in d["_matches"]]
        out.append(dict(variant=name, room_box_error_cm={k: v["error_cm"] for k, v in d["room_box"].items()},
                        error_vs_distance=d["error_vs_distance"], n_pairs=len(d["pairs"]),
                        inward_offset_median_cm=float(np.median(offs)) if offs else None))
    return out


# ================================================================================ figures
def fig_h1(abs_: dict, internal: dict, out: Path, title: str):
    """Translation-free inward bias by range / incidence / edge distance, and the internal near-vs-far slopes."""
    fig, ax = plt.subplots(1, 4, figsize=(23, 5))
    panels = ((ax[0], (("by_range", C1, "all points"), ("by_range_interior", C2, f"interior (>= {INTERIOR_M} m from edge)")),
               "range from camera (m)"),
              (ax[1], (("by_incidence", C1, "all points"),), "incidence angle (deg, 0 = head-on)"),
              (ax[2], (("by_edge_distance", C1, "all points"),), "in-plane distance to the patch edge (m)"))
    for a, series, lab in panels:
        for key, col, name in series:
            rows = abs_[key]
            a.plot([np.mean(r["bin"]) for r in rows], [r["b_cm"] for r in rows], "o-", c=col, label=name)
        a.axhline(0, c="#52514e", lw=0.8)
        a.set_xlabel(lab); a.set_ylabel("inward bias b (cm), + = surface into the room")
        a.grid(alpha=0.3); a.legend(frameon=False, fontsize=8)
    ax[0].set_title("bias vs range (a depth-scale error would be a sloped line)", fontsize=10)
    ax[1].set_title("bias vs incidence", fontsize=10)
    ax[2].set_title("bias vs distance to corner / opening / occluder", fontsize=10)
    per = internal.get("per_surface", [])
    if per:
        y = np.arange(len(per))
        ax[3].errorbar([p["slope_mm_per_m"] for p in per], y, xerr=[1.96 * p["se_mm_per_m"] for p in per], fmt="o",
                       c=C3, ms=4)
        ax[3].set_yticks(y, [f"{p['family']} #{p['k']} (range spread {p['range_spread_m']:.2f} m)" for p in per],
                         fontsize=7)
        lo, hi = internal["ci95_mm_per_m"]
        ax[3].axvspan(lo, hi, color="#0b0b0b", alpha=0.12,
                      label=f"pooled {internal['pooled_slope_mm_per_m']:+.1f} mm/m, 95% CI [{lo:+.1f}, {hi:+.1f}]")
        ax[3].axvline(0, c="#52514e", lw=0.8)
        ax[3].set_xlim(-60, 60)
        ax[3].set_xlabel("near-vs-far slope (mm depth error per m range)")
        ax[3].set_title("internal test, no laser: same 10 cm patch from different ranges", fontsize=10)
        ax[3].legend(frameon=False, fontsize=8, loc="lower right"); ax[3].grid(alpha=0.3, axis="x")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out / "h1_depth_vs_range.png", dpi=110); plt.close(fig)


def fig_maps(recs: list[dict], box_ks: list[int], out: Path, title: str):
    """Face-on map of our offset from the laser plane (median per 10 cm cell, all observations) for the room-box
    surfaces: where on each surface the inward bias lives (edges? one corner? everywhere?)."""
    sel = [r for r in recs if r["k"] in box_ks]
    fig, axs = plt.subplots(1, len(sel), figsize=(4.6 * len(sel), 4.6), squeeze=False)
    im = None
    for a, r in zip(axs[0], sel):
        i, j = [k for k in range(3) if k != r["axis"]]
        if r["axis"] != 1:
            i, j = (2 if r["axis"] == 0 else 0), 1
        P = r["obs_P"]
        x = (r["obs_x"] - r["o_laser"]) * 100
        m = np.abs(r["obs_x"] - r["o_ours_tls"]) < WINDOW_M
        gi = np.floor(P[m, i] / 0.1).astype(int)
        gj = np.floor(P[m, j] / 0.1).astype(int)
        uk, inv = np.unique(np.stack([gi, gj], 1), axis=0, return_inverse=True)
        inv = inv.ravel()
        med = np.array([np.median(x[m][inv == q]) for q in range(len(uk))])
        ok = np.bincount(inv) >= 20
        im = a.scatter(uk[ok, 0] * 0.1 + 0.05, uk[ok, 1] * 0.1 + 0.05, c=med[ok], cmap="RdBu_r",
                       vmin=-3, vmax=3, s=60, marker="s")
        a.set_aspect("equal"); a.grid(alpha=0.2)
        a.set_title(f"{r['family']} #{r['k']} (inward {r['inward_tls_cm']:+.2f} cm)", fontsize=9)
        a.set_xlabel(f"{'xyz'[i]} (m)"); a.set_ylabel(f"{'xyz'[j]} (m)")
    if im is not None:
        fig.colorbar(im, ax=axs, shrink=0.8, label="our offset from the laser plane (cm), + = into the room")
    fig.suptitle(title); fig.savefig(out / "surface_maps.png", dpi=100, bbox_inches="tight"); plt.close(fig)


def fig_estimators(table: dict, out: Path, title: str):
    names = list(table.keys())
    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    box_keys = ["width_x", "width_z", "height"]
    xs = np.arange(len(names))
    for i, (k, col) in enumerate(zip(box_keys, (C1, C2, C3))):
        ax[0].bar(xs + (i - 1) * 0.27, [table[n]["room_box_cm"].get(k, np.nan) for n in names], 0.27, color=col,
                  label=k)
    ax[0].axhline(0, c="#52514e", lw=0.8); ax[0].set_xticks(xs, names, rotation=40, ha="right", fontsize=8)
    ax[0].set_ylabel("room-box error vs laser (cm)"); ax[0].legend(frameon=False); ax[0].grid(alpha=0.3, axis="y")
    ax[0].set_title("H4: room box by estimator (laser reference fixed)", fontsize=10)
    ax[1].bar(xs - 0.2, [table[n]["bias"]["b_cm"] for n in names], 0.4, color=C4, label="shared inward bias b (cm)")
    ax[1].bar(xs + 0.2, [table[n]["pairs_full_height_median_abs_cm"] for n in names], 0.4, color="#9ec3ef",
              label="full-height pairs median |error| (cm)")
    ax[1].axhline(0, c="#52514e", lw=0.8); ax[1].set_xticks(xs, names, rotation=40, ha="right", fontsize=8)
    ax[1].legend(frameon=False); ax[1].grid(alpha=0.3, axis="y")
    ax[1].set_title("H4: per-surface inward bias (registration-free) and pair error", fontsize=10)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out / "h4_estimators.png", dpi=110); plt.close(fig)


def fig_profiles(recs: list[dict], box_ks: list[int], out: Path, title: str):
    """Distribution of our points and the laser's along the normal for the room-box surfaces, and inward offset vs
    distance to the patch edge: shows whether the bias is a one-sided tail (method) or a shifted peak (sensor)."""
    sel = [r for r in recs if r["k"] in box_ks]
    fig, axs = plt.subplots(2, max(3, len(sel)), figsize=(4.2 * max(3, len(sel)), 8), squeeze=False)
    bins = np.linspace(-4, 4, 81)
    for j, r in enumerate(sel):
        a = axs[0, j]
        a.hist((r["xo"] - r["o_laser"]) * 100, bins, density=True, histtype="step", lw=1.8, color=C1,
               label="our raw points (dedup)")
        a.hist((r["obs_x"] - r["o_laser"]) * 100, bins, density=True, histtype="step", lw=1.2, color=C3,
               label="all observations")
        a.hist((r["xl"] - r["o_laser"]) * 100, bins, density=True, histtype="step", lw=1.8, color=C2, label="laser")
        a.axvline(0, c="#52514e", lw=0.8)
        a.set_title(f"{r['family']} #{r['k']}: inward {r['inward_tls_cm']:+.2f} cm", fontsize=9)
        a.set_xlabel("offset from laser plane (cm), + = into the room")
        b = axs[1, j]
        e = r["obs_edge"]
        x = (r["obs_x"] - r["o_laser"]) * 100
        rows = []
        for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 2)):
            m = (e >= lo) & (e < hi) & (np.abs(x) < 3)
            if m.sum() > 100:
                rows.append(((lo + hi) / 2 if hi < 2 else 0.6, np.median(x[m]), m.sum()))
        if rows:
            b.plot([q[0] for q in rows], [q[1] for q in rows], "o-", c=C1)
        b.axhline(0, c="#52514e", lw=0.8); b.grid(alpha=0.3)
        b.set_xlabel("distance to patch edge (m)"); b.set_ylabel("median offset (cm)")
    axs[0, 0].legend(frameon=False, fontsize=7)
    for j in range(len(sel), axs.shape[1]):
        axs[0, j].axis("off"); axs[1, j].axis("off")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out / "surface_profiles.png", dpi=100); plt.close(fig)


# ================================================================================ main analyses
def analyse_arkit(scene_dir: Path, laser_paths: list[Path], out: Path, ref_transform: Path | None, log) -> dict:
    t0 = time.time()
    cfg = Config.load()
    laser = V.load_laser(laser_paths)
    cap = load_arkitscenes(scene_dir, "nearest")
    scene, info = V.build_stage1(cap, cfg)
    T_LA, reg = V.register_laser(scene, info, laser, cfg)
    log(f"[{cap.capture_id}] stage1 + registration {time.time()-t0:.0f} s: {reg}")
    EP, EN = laser["eval"]
    LP, LN = V.transform(T_LA, EP), EN @ T_LA[:3, :3].T
    laser_fc = V.family_cloud(LP, LN, V.EVAL_VOXEL_M)
    floor_y = info["floor_y"]
    med = np.median(scene["traj"][:, [0, 2]], axis=0)
    centre = np.array([med[0], floor_y + 1.2, med[1]])
    dims = V.measure_against_laser(scene["raw_points"], scene["raw_normals"], cfg.raw_voxel_m, laser_fc, centre, floor_y)
    matches, box = dims["_matches"], dims["_box"]
    log(f"[{cap.capture_id}] baseline room box {[(k, v['error_cm']) for k, v in dims['room_box'].items()]}, "
        f"fit {dims['error_vs_distance']}, {len(matches)} matched surfaces")

    kf = select_keyframes(cap, cfg)
    obs = collect_observations(cap, kf, scene["T_align"], cfg)
    obs_tree = cKDTree(obs["P"])
    recs = surface_records(matches, dims["_fc"], laser_fc, obs, obs_tree, floor_y)
    for r, m in zip(recs, matches):
        r["_s"] = m["ours"]
    box_ks = [i for i, m in enumerate(matches) for pair in box.values() for mm in pair if mm is m]
    box_axis = {name: [i for i, m in enumerate(matches) for mm in pair if mm is m] for name, pair in box.items()}
    pairs = facing_pairs_idx(recs)
    log(f"[{cap.capture_id}] {len(obs['P']):,} observations; {len(pairs)} facing pairs ({time.time()-t0:.0f} s)")

    # --- unit of analysis: pairs vs surfaces, and the registration-free bias b (current estimator) ---
    inward_tls = np.array([r["inward_tls_cm"] for r in recs]) / 100
    check = pair_errors(recs, pairs, inward_tls) * 100
    val_err = sorted(p["error_cm"] for p in dims["pairs"])
    res = dict(scene=str(scene_dir), capture=cap.capture_id, registration=reg, baseline_room_box=dims["room_box"],
               baseline_error_vs_distance=dims["error_vs_distance"], n_surfaces=len(recs), n_pairs=len(pairs),
               pair_error_reconstruction=dict(ours_sorted_cm=np.round(sorted(check), 2).tolist(),
                                              validation_sorted_cm=val_err),
               cluster_bootstrap_current=cluster_bootstrap_slope(recs, pairs, inward_tls))

    # --- H4: estimator table ---
    est = [estimators(r) for r in recs]
    names = sorted({k for e in est for k in e}, key=lambda n: list(est[0].keys()).index(n) if n in est[0] else 99)
    weights_area = np.array([r["common"] for r in recs])
    table = {}
    for name in names:
        inward = np.array([(e.get(name, np.nan) - r["o_laser"]) if np.isfinite(e.get(name, np.nan)) else np.nan
                           for e, r in zip(est, recs)])
        rb = {}
        for bname, ks in box_axis.items():
            if len(ks) == 2 and np.all(np.isfinite(inward[ks])):
                rb[bname] = float(-(inward[ks[0]] + inward[ks[1]]) * 100)
        pe = pair_errors(recs, pairs, np.nan_to_num(inward, nan=np.nan)) * 100
        full = np.array([p for p, (a, b, ax) in zip(pe, pairs) if ax != 1 and recs[a]["tall"] and recs[b]["tall"]])
        ok = np.isfinite(pe)
        dd = np.array([abs(recs[a]["o_laser"] + recs[b]["o_laser"]) for a, b, _ in pairs])
        c0, c1 = fit_offset_slope(dd[ok], pe[ok]) if ok.sum() >= 5 else (np.nan, np.nan)
        table[name] = dict(room_box_cm=rb, bias=bias_translation_fit(recs, inward * 100),
                           bias_area_weighted=bias_translation_fit(recs, inward * 100, weights_area),
                           inward_median_cm=float(np.nanmedian(inward) * 100),
                           pairs_n=int(ok.sum()), pairs_fit=dict(offset_cm=c0, slope_cm_per_m=c1),
                           pairs_full_height_median_abs_cm=float(np.nanmedian(np.abs(full))) if len(full) else np.nan,
                           pairs_full_height_median_signed_cm=float(np.nanmedian(full)) if len(full) else np.nan,
                           per_surface_inward_cm=np.round(inward * 100, 2).tolist())
    res["h4_estimators"] = table
    res["surfaces"] = [dict(k=r["k"], family=r["family"], area_m2=r["area"], common_m2=r["common"], tall=r["tall"],
                            n_raw=int(len(r["xo"])), n_obs=int(len(r["obs_x"])), n_laser=int(len(r["xl"])),
                            inward_tls_cm=round(r["inward_tls_cm"], 2),
                            median_range_m=float(np.median(r["obs_r"])) if len(r["obs_r"]) else None,
                            median_incidence_deg=float(np.degrees(np.arccos(np.median(r["obs_cos"]))))
                            if len(r["obs_cos"]) else None,
                            interior_frac=float((r["obs_edge"] >= INTERIOR_M).mean()) if len(r["obs_edge"]) else None,
                            normal_angle_deg=float(np.degrees(np.arccos(np.clip(abs(np.asarray(
                                matches[r["k"]]["ours"].plane.n) @ r["n"]), 0, 1)))),
                            in_room_box=r["k"] in box_ks) for r in recs]
    log(f"[{cap.capture_id}] H4 done ({time.time()-t0:.0f} s)")

    # --- H5: laser ---
    lz = [laser_estimators(r) for r in recs]
    spread = [max(abs(v - l["trimmed_ls_plane"]) for k, v in l.items() if k not in ("rmse_mm",)) * 1000
              for l in lz if l]
    res["h5_laser"] = dict(plane_rmse_mm_median=float(np.median([l["rmse_mm"] for l in lz if l])),
                           estimator_spread_mm_median=float(np.median(spread)),
                           estimator_spread_mm_max=float(np.max(spread)),
                           best_pairs_abs_cm=sorted(np.round(np.abs(check), 2).tolist())[:5])

    # --- H1 ---
    b_tls = table["trimmed_ls_plane"]["bias"]
    res["h1_internal_interior"] = h1_internal(recs, True)
    res["h1_internal_all"] = h1_internal(recs, False)
    res["h1_internal_cos_offset"] = h1_internal_cos(recs)
    log(f"[{cap.capture_id}] internal constant-depth-offset estimate {res['h1_internal_cos_offset'].get('pooled_depth_offset_mm')}"
        f" mm, 95% CI {res['h1_internal_cos_offset'].get('ci95_mm_per_m')}")
    ab = h1_absolute(recs)
    res["h1_absolute"] = ab
    fig_h1(ab, res["h1_internal_interior"], out, f"ARKitScenes {cap.capture_id}: H1 depth bias vs range")
    log(f"[{cap.capture_id}] H1 internal pooled slope {res['h1_internal_interior'].get('pooled_slope_mm_per_m')} mm/m; "
        f"{res['h1_internal_interior'].get('ci95_mm_per_m')}; by range {[(r['bin'], round(r['b_cm'], 2)) for r in ab['by_range']]}; "
        f"by edge {[(r['bin'], round(r['b_cm'], 2)) for r in ab['by_edge_distance']]} ({time.time()-t0:.0f} s)")

    # --- H2 ---
    ov = cKDTree(LP).query(scene["raw_points"], workers=-1)[0] < 0.05
    S = scene["raw_points"][ov]
    S = S[V.voxel_subsample(S, 0.01)]
    walls = np.abs(scene["raw_normals"][ov][V.voxel_subsample(scene["raw_points"][ov], 0.01)][:, 1]) < 0.3
    T_s = scene["points"][cKDTree(LP).query(scene["points"], workers=-1)[0] < 0.05]
    res["h2_similarity"] = dict(
        raw_iso=similarity_icp(S, LP, LN, "iso", boot=N_BOOT), raw_axis=similarity_icp(S, LP, LN, "axis", boot=N_BOOT),
        raw_walls_iso=similarity_icp(S[walls], LP, LN, "iso", boot=N_BOOT),
        tsdf_iso=similarity_icp(T_s, LP, LN, "iso", boot=N_BOOT))
    log(f"[{cap.capture_id}] H2 {json.dumps({k: v['scale_minus_1_pct'] for k, v in res['h2_similarity'].items()})} "
        f"({time.time()-t0:.0f} s)")

    # --- H3 ---
    ref_T = None
    if ref_transform is not None and ref_transform.exists():
        ref_T = scene["T_align"] @ Z_UP_TO_Y_UP @ np.loadtxt(ref_transform) @ np.linalg.inv(Z_UP_TO_Y_UP)
    res["h3_registration"] = h3_registration(scene, cfg, laser, T_LA, centre, floor_y, ref_T)
    log(f"[{cap.capture_id}] H3 done ({time.time()-t0:.0f} s)")

    fig_estimators(table, out, f"ARKitScenes {cap.capture_id}: measurement method (H4)")
    fig_maps(recs, box_ks, out, f"ARKitScenes {cap.capture_id}: where our room-box surfaces differ from the laser")
    fig_profiles(recs, box_ks, out, f"ARKitScenes {cap.capture_id}: room-box surfaces, ours vs laser along the normal")
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def analyse_stray(capture: Path, scene_dir: Path, log) -> dict:
    """H1 internal test on a Stray Scanner sample capture (no ground truth): near-vs-far slope on large walls."""
    from floorplan.io.stray import load_stray
    from floorplan.pipeline.scene import load_scene
    cfg = Config.load()
    cap = load_stray(capture)
    scene, info = load_scene(scene_dir)
    T_wc = scene["T_wc"]
    cap.T_wc = T_wc
    P = scene["raw_points"].astype(np.float64)
    N = V.pca_normals(P)
    N[np.einsum("ij,ij->i", N, scene["raw_ray"].astype(np.float64)) > 0] *= -1
    fc = V.family_cloud(P, N, cfg.raw_voxel_m)
    surfaces = [s for s in V.find_surfaces(fc) if s.area_m2 >= 0.3]
    obs = collect_observations(cap, scene["kf"], scene["T_align"], cfg)
    tree = cKDTree(obs["P"])
    per, per_cos = [], []
    for k, s in enumerate(surfaces):
        axis, sign = V.FAMILIES[s.family]
        n = np.asarray(s.plane.n)
        mid = surface_mid(s)
        cand = np.asarray(tree.query_ball_point(mid, r=float(np.linalg.norm(s.hi - s.lo) / 2 + 0.05)), int)
        Po = obs["P"][cand]
        cosi = -(obs["ray"][cand] @ n)
        keep = (np.abs(Po @ n - s.plane.d) < WINDOW_M) & np.isin(V.cell_keys(Po, axis), s.cells)
        keep &= cosi > math.cos(math.radians(MAX_INC_DEG))
        Po, cosi, rr = Po[keep], cosi[keep], obs["r"][cand][keep]
        X = fc.fam[s.family]
        X = X[np.abs(X @ n - s.plane.d) < V.SEED_WINDOW_M]
        e = edge_distance(X, axis, Po)
        m = e >= INTERIOR_M
        if m.sum() < 200:
            continue
        r = near_far_slope(Po[m] @ n - s.plane.d, rr[m], cosi[m], Po[m], axis)
        if np.isfinite(r["slope_mm_per_m"]):
            per.append(dict(k=k, family=s.family, area=round(s.area_m2, 3), **r))
        w = np.abs(Po @ n - s.plane.d) < WINDOW_M
        c = fe_cos_offset(Po[w] @ n - s.plane.d, cosi[w], Po[w], axis) if w.sum() >= 200 else None
        if c is not None and np.isfinite(c["slope_mm_per_m"]):
            per_cos.append(dict(k=k, family=s.family, area=round(s.area_m2, 3), **c))
    res = dict(capture=str(capture), **pool_slopes(per))
    res["cos_offset"] = {k.replace("slope_mm_per_m", "depth_offset_mm"): v for k, v in pool_slopes(per_cos).items()}
    log(f"[{cap.capture_id}] internal constant-depth-offset {res['cos_offset'].get('pooled_depth_offset_mm')} mm, "
        f"95% CI {res['cos_offset'].get('ci95_mm_per_m')}")
    log(f"[{cap.capture_id}] internal near-vs-far slope pooled {res.get('pooled_slope_mm_per_m')} mm/m, "
        f"95% CI {res.get('ci95_mm_per_m')}, over {len(per)} surfaces")
    return res


def summary(log) -> dict:
    """Pool every analysed room. Evaluate the pre-registered correction (floorplan/uncertainty/bias.py) on the
    development rooms (leave-one-room-out) and on the hold-out rooms (value fixed before they were analysed)."""
    from floorplan.uncertainty.bias import LidarBiasConfig, correct_interior_distance
    prereg = json.loads((OUT / "preregistration.json").read_text())
    cfg = LidarBiasConfig.load(inward_bias_m=prereg["b_cm"] / 100)
    rooms = {}
    for p in sorted(OUT.glob("arkit_*/results.json")):
        r = json.loads(p.read_text())
        rooms[r["capture"]] = r
    dev = [c for c in prereg["development_rooms"] if c in rooms]
    table, before, after = {}, {}, {}
    for cid, r in rooms.items():
        box = {k: v for k, v in r["baseline_room_box"].items()}
        if cid in dev:   # leave-one-room-out: b from the OTHER development rooms only
            others = [e["error_cm"] for c in dev if c != cid for e in rooms[c]["baseline_room_box"].values()]
            b_m = -np.mean(others) / 2 / 100
        else:
            b_m = cfg.inward_bias_m
        c = LidarBiasConfig.load(inward_bias_m=b_m)
        corr = {k: (correct_interior_distance(v["ours_m"], c) - v["laser_m"]) * 100 for k, v in box.items()}
        t = r["h4_estimators"]["trimmed_ls_plane"]
        table[cid] = dict(
            role="development" if cid in dev else "hold-out", b_used_cm=b_m * 100,
            room_box_before_cm={k: v["error_cm"] for k, v in box.items()}, room_box_after_cm=corr,
            pairs_fit=r["baseline_error_vs_distance"], cluster_bootstrap=r["cluster_bootstrap_current"],
            per_surface_bias_area_weighted_cm=t["bias_area_weighted"]["b_cm"],
            per_surface_bias_cm=t["bias"]["b_cm"],
            h1_by_range=[(x["bin"], round(x["b_cm"], 2)) for x in r["h1_absolute"]["by_range"]],
            h1_internal_slope_mm_per_m=r["h1_internal_interior"].get("pooled_slope_mm_per_m"),
            h1_internal_slope_ci=r["h1_internal_interior"].get("ci95_mm_per_m"),
            h2_iso_scale_pct=r["h2_similarity"]["raw_iso"]["scale_minus_1_pct"][0],
            h2_axis_scale_pct=r["h2_similarity"]["raw_axis"]["scale_minus_1_pct"],
            h3_room_box_range_cm={k: [min(v["room_box_error_cm"].get(k, np.nan) for v in r["h3_registration"]),
                                      max(v["room_box_error_cm"].get(k, np.nan) for v in r["h3_registration"])]
                                  for k in box},
            estimator_room_box_cm={n: v["room_box_cm"] for n, v in r["h4_estimators"].items()},
            registration=r["registration"], n_pairs=r["n_pairs"])
        before[cid] = [v["error_cm"] for v in box.values()]
        after[cid] = list(corr.values())

    def agg(ids):
        b = np.concatenate([before[i] for i in ids]) if ids else np.zeros(0)
        a = np.concatenate([after[i] for i in ids]) if ids else np.zeros(0)
        if not len(b):
            return {}
        return dict(n=int(len(b)), before_mean_cm=float(b.mean()), before_mean_abs_cm=float(np.abs(b).mean()),
                    before_within_1cm=float((np.abs(b) <= 1).mean()), before_within_1_5cm=float((np.abs(b) <= 1.5).mean()),
                    after_mean_cm=float(a.mean()), after_mean_abs_cm=float(np.abs(a).mean()),
                    after_within_1cm=float((np.abs(a) <= 1).mean()), after_within_1_5cm=float((np.abs(a) <= 1.5).mean()),
                    after_std_cm=float(a.std(ddof=1)) if len(a) > 1 else None)
    hold = [c for c in rooms if c not in dev]
    out = dict(preregistration=prereg, rooms=table, development_loro=agg(dev), holdout=agg(hold),
               all_rooms=agg(list(rooms)))
    all_err = np.concatenate([before[i] for i in rooms])
    out["refit_all_rooms"] = dict(b_cm=float(-all_err.mean() / 2), sigma_distance_cm=float(all_err.std(ddof=1)),
                                  n=int(len(all_err)))
    stray = {}
    for p in sorted(OUT.glob("stray_*/results.json")):
        r = json.loads(p.read_text())
        stray[Path(r["capture"]).name] = dict(near_far_slope_mm_per_m=r.get("pooled_slope_mm_per_m"),
                                              ci95=r.get("ci95_mm_per_m"), n_surfaces=r.get("n_surfaces"),
                                              cos_offset_mm=r.get("cos_offset", {}).get("pooled_depth_offset_mm"),
                                              cos_offset_ci95=r.get("cos_offset", {}).get("ci95_mm_per_m"))
    out["stray_internal"] = stray
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    fig_summary(table, out, OUT)
    log(json.dumps({k: out[k] for k in ("development_loro", "holdout", "all_rooms", "refit_all_rooms")}, indent=1))
    return out


def fig_summary(table: dict, out: dict, path: Path):
    """Room-box errors before / after the pre-registered correction, per room (development rooms leave-one-out)."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    xs, labels, i = [], [], 0
    for cid, t in table.items():
        for k in t["room_box_before_cm"]:
            ax[0].bar(i - 0.2, t["room_box_before_cm"][k], 0.4, color=C2, label="before" if i == 0 else None)
            ax[0].bar(i + 0.2, t["room_box_after_cm"][k], 0.4, color=C1, label="after correction" if i == 0 else None)
            labels.append(f"{cid}\n{t['role'][:4]} {k}")
            xs.append(i)
            i += 1
        i += 0.6
    for g in (1.5, -1.5):
        ax[0].axhline(g, c="#e34948", lw=0.8, ls="--")
    ax[0].axhline(0, c="#52514e", lw=0.8)
    ax[0].set_xticks(xs, labels, rotation=60, ha="right", fontsize=7)
    ax[0].set_ylabel("room-box error vs laser (cm)"); ax[0].legend(frameon=False); ax[0].grid(alpha=0.3, axis="y")
    ax[0].set_title("ARKitScenes rooms: room-box error before / after (dashed = 1.5 cm gate)", fontsize=10)
    ids = list(table)
    y = np.arange(len(ids))
    ax[1].scatter([table[c]["h2_iso_scale_pct"] for c in ids], y, c=C4, label="H2 similarity scale - 1 (%)")
    ax[1].scatter([table[c]["pairs_fit"]["slope_cm_per_m"] for c in ids], y, c=C2, marker="s",
                  label="pair error-vs-distance slope (cm/m = %)")
    ax[1].scatter([-2 * table[c]["per_surface_bias_area_weighted_cm"] for c in ids], y, c=C1, marker="D",
                  label="-2 x per-surface inward bias (cm)")
    ax[1].scatter([table[c]["pairs_fit"]["offset_cm"] for c in ids], y, c=C3, marker="^",
                  label="pair error-vs-distance offset (cm)")
    ax[1].axvline(0, c="#52514e", lw=0.8)
    ax[1].set_yticks(y, ids); ax[1].legend(frameon=False, fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].set_title("scale-type vs offset-type evidence per room", fontsize=10)
    fig.tight_layout(); fig.savefig(path / "summary_before_after.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("arkit")
    a1.add_argument("--scene", type=Path, required=True)
    a1.add_argument("--laser", type=Path, nargs="+", required=True)
    a1.add_argument("--reference-transform", type=Path, default=None)
    a2 = sub.add_parser("stray")
    a2.add_argument("--capture", type=Path, required=True)
    a2.add_argument("--scene-dir", type=Path, required=True)
    sub.add_parser("summary")
    a = ap.parse_args()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    log = lambda m: print(m, flush=True)  # noqa: E731
    if a.cmd == "arkit":
        out = OUT / f"arkit_{a.scene.name}"
        out.mkdir(parents=True, exist_ok=True)
        res = analyse_arkit(a.scene, a.laser, out, a.reference_transform, log)
        (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    elif a.cmd == "stray":
        out = OUT / f"stray_{a.capture.name}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(analyse_stray(a.capture, a.scene_dir, log), indent=2,
                                                     default=float))
    else:
        summary(log)


if __name__ == "__main__":
    main()
