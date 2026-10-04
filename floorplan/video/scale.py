"""Metric scale for an up-to-scale video trajectory, from single-image metric depth.

Monocular odometry (DPVO) knows the camera path only up to an unknown factor, and that factor DRIFTS along the
video (measured: 0.87 -> 0.69 within 37 s on single_room) and can JUMP when tracking restarts (x2.6 and x9 on
floor_only), decision V-6. Wall lengths inherit any scale error, and the video gate is 3%.

Idea (dense, triangulation-free): for two keyframes i and j, back-project i's metric depth, move it with the
relative pose whose translation is multiplied by a candidate scale s, and compare the predicted depth with j's
metric depth at the pixels it lands on. Only the right s makes the two depth maps agree. Each such pair is a scale
measurement. Summing the cost curves of all pairs in a sliding time window gives the local scale (cut at detected
jumps), and re-integrating the trajectory with that local scale removes the drift.
"""
from __future__ import annotations

import numpy as np

TRUNC = 0.1   # truncate the |log depth ratio| at 10%: occlusions and wrong pixels then cost a constant


def _backproject(D: np.ndarray, K: np.ndarray, stride: int) -> np.ndarray:
    v, u = np.mgrid[0:D.shape[0]:stride, 0:D.shape[1]:stride]
    z = D[v, u]
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)


def _cost(X: np.ndarray, D_j: np.ndarray, K: np.ndarray) -> tuple[float, int]:
    """Mean truncated |log(z_pred / z_obs)| of points X (camera j frame) against depth map D_j."""
    z = X[:, 2]
    front = z > 0.05
    u = np.round(K[0, 0] * X[front, 0] / z[front] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * X[front, 1] / z[front] + K[1, 2]).astype(int)
    h, w = D_j.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if inside.sum() < 200:
        return np.inf, 0
    zo = D_j[v[inside], u[inside]]
    ok = zo > 0
    r = np.minimum(np.abs(np.log(z[front][inside][ok] / zo[ok])), TRUNC)
    return float(r.mean()), int(ok.sum())


def pair_cost_curve(D_i, D_j, K, T_i, T_j, grid: np.ndarray, stride: int = 4) -> np.ndarray:
    """Cost of every candidate translation scale in `grid` for one keyframe pair (inf where they do not overlap).

    We keep the whole curve instead of its minimum: one pair is noisy (a pure rotation makes the curve flat,
    a depth error shifts it), but SUMMING the curves of many pairs gives a sharp, robust minimum. This is the
    same reason bundle adjustment sums residuals instead of averaging per-pair solutions."""
    X = _backproject(D_i, K, stride)
    T_ji = np.linalg.inv(T_j) @ T_i
    RX = X @ T_ji[:3, :3].T
    return np.array([_cost(RX + s * T_ji[:3, 3], D_j, K)[0] for s in grid])


def curve_minimum(grid: np.ndarray, cost: np.ndarray) -> float:
    """Arg-min of a cost curve on a log grid, refined by a parabola through the three best points."""
    k = int(np.argmin(cost))
    if 0 < k < len(grid) - 1 and np.all(np.isfinite(cost[k - 1:k + 2])):
        y0, y1, y2 = cost[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den > 0 else 0.0
        step = np.log(grid[1] / grid[0])
        return float(grid[k] * np.exp(np.clip(off, -1, 1) * step))
    return float(grid[k])


def keyframe_pairs(n: int, max_gap: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, min(n, i + 1 + max_gap))]


def _normalised(curves: np.ndarray, grid: np.ndarray, min_contrast: float) -> np.ndarray:
    """Shift every curve to a zero minimum and drop flat curves.

    A scale at which the two views stop overlapping gets the maximum cost (TRUNC), so it can never win. A curve is
    flat when the pair carries no scale information (pure rotation, tiny baseline): its cost changes by less than
    `min_contrast` when the scale moves +-20% from its optimum. Such curves are zeroed (no opinion)."""
    c = np.where(np.isfinite(curves), curves, TRUNC)
    c = c - c.min(axis=1, keepdims=True)
    k = np.argmin(c, axis=1)
    rows = np.arange(len(c))
    lo = np.clip(np.searchsorted(grid, grid[k] / 1.2), 0, len(grid) - 1)
    hi = np.clip(np.searchsorted(grid, grid[k] * 1.2), 0, len(grid) - 1)
    contrast = np.minimum(c[rows, lo], c[rows, hi])
    c[contrast < min_contrast] = 0.0
    return c


def _fill_nan_nearest(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if not ok.any():
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def segment_scale_jumps(mid: np.ndarray, argmins: np.ndarray, n: int, half: int, jump: float,
                        min_len: int) -> np.ndarray:
    """Segment id per keyframe, cutting where the robust local scale jumps by more than `jump` (ratio).

    Monocular VO can silently RESTART its scale after a tracking failure (measured on floor_only: x2.8, then x11
    within one video). A smooth window would average across such a step, so we first find the steps with a running
    MEDIAN of per-pair scale estimates (a median keeps steps sharp) and then smooth only inside each segment."""
    s_m = np.full(n, np.nan)
    for k in range(n):
        v = argmins[np.abs(mid - k) <= half]
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            s_m[k] = np.median(v)
    s_m = _fill_nan_nearest(s_m)
    cut = np.abs(np.diff(np.log(s_m))) > np.log(jump)
    seg = np.concatenate([[0], np.cumsum(cut)])
    # a real VO restart persists; a segment shorter than min_len keyframes is noise and joins its predecessor
    starts = [0] + [k for k in range(1, n) if seg[k] != seg[k - 1]]
    keep = [a for a, b in zip(starts, starts[1:] + [n]) if b - a >= min_len or a == 0]
    out = np.zeros(n, int)
    for g, a in enumerate(keep):
        out[a:] = g
    return out


def _merge_similar_segments(seg, seg_pair, same, C, fine, jump: float, protected=()) -> np.ndarray:
    """Undo cuts whose two sides have nearly the same FINE scale (ratio < sqrt(jump)): the running median of the
    noisy coarse estimates can step across the threshold without any real VO restart (Issue 13b)."""
    seg = seg.copy()
    while True:
        ids = np.unique(seg)
        scales = []
        for g in ids:
            rows = np.where((seg_pair == g) & same)[0]
            scales.append(curve_minimum(fine, C[rows].sum(0)) if len(rows) and C[rows].max() > 0 else np.nan)
        merged = False
        for a in range(1, len(ids)):
            if int(np.argmax(seg == ids[a])) in protected:      # a fresh VO run starts here: never merge
                continue
            if np.isfinite(scales[a]) and np.isfinite(scales[a - 1]) and \
                    abs(np.log(scales[a] / scales[a - 1])) < 0.5 * np.log(jump):
                seg[seg == ids[a]] = ids[a - 1]
                seg_pair = np.where(seg_pair == ids[a], ids[a - 1], seg_pair)
                merged = True
                break
        if not merged:
            return np.unique(seg, return_inverse=True)[1]


def estimate_scales(D: np.ndarray, K: np.ndarray, T: np.ndarray, max_gap: int, sigma_kf: float,
                    min_contrast: float = 0.005, jump: float = 1.5, n_boot: int = 2000, block: int = 10,
                    seed: int = 0, forced_cuts=()) -> dict:
    """Per-keyframe (local) metric scale of an up-to-scale trajectory T (N,4,4) from metric depth D (N,h,w).

    1. coarse cost curve per keyframe pair on a wide grid (10% steps over 6 decades)
    2. scale-jump segmentation from the running median of the per-pair coarse minima
    3. fine curves (1% steps, +-35% around the local coarse value), summed in a Gaussian time window that does not
       cross a segment boundary
    4. block bootstrap over pairs, per segment, for the statistical part of the uncertainty
    """
    pairs = keyframe_pairs(len(T), max_gap)
    mid = np.array([(i + j) / 2 for i, j in pairs])
    coarse = np.exp(np.arange(np.log(1e-3), np.log(1e3), np.log(1.1)))
    Cc = _normalised(np.array([pair_cost_curve(D[i], D[j], K, T[i], T[j], coarse) for i, j in pairs]), coarse,
                     min_contrast)
    informative = Cc.max(axis=1) > 0
    argmins = np.where(informative, coarse[np.argmin(Cc, axis=1)], np.nan)
    seg = segment_scale_jumps(mid, argmins, len(T), 2 * max_gap, jump, min_len=20)
    if len(forced_cuts):                    # v2: keyframes where a separate VO run starts (its own unknown scale)
        cut = np.diff(seg) != 0
        for f in forced_cuts:               # a jump found 1-3 keyframes from a forced cut is the same event
            cut[max(0, f - 4): f + 3] = False
        cut[np.asarray(forced_cuts, int) - 1] = True
        seg = np.concatenate([[0], np.cumsum(cut)])
    seg_pair = seg[np.array([i for i, _ in pairs])]
    same = seg_pair == seg[np.array([j for _, j in pairs])]          # pairs straddling a jump carry no information

    fine = np.exp(np.arange(np.log(1e-3), np.log(1e3), np.log(1.01)))
    C = np.zeros((len(pairs), len(fine)))
    s_coarse_kf = np.empty(len(T))
    for k in range(len(T)):                                          # coarse local value: Gaussian sigma 3
        w = np.exp(-0.5 * ((mid - k) / 3.0) ** 2) * (seg_pair == seg[k]) * same
        s_coarse_kf[k] = curve_minimum(coarse, (w[:, None] * Cc).sum(0))
    for p, (i, j) in enumerate(pairs):
        if not (same[p] and informative[p]):
            continue
        c0 = np.searchsorted(fine, s_coarse_kf[i])
        cols = np.arange(max(0, c0 - 30), min(len(fine), c0 + 31))
        curve = pair_cost_curve(D[i], D[j], K, T[i], T[j], fine[cols])
        curve = np.where(np.isfinite(curve), curve, TRUNC)
        C[p] = TRUNC
        C[p, cols] = curve
        C[p] -= C[p].min()
    seg = _merge_similar_segments(seg, seg_pair, same, C, fine, jump, protected=set(int(k) for k in forced_cuts))
    seg_pair = seg[np.array([i for i, _ in pairs])]
    same = seg_pair == seg[np.array([j for _, j in pairs])]
    s_local = np.full(len(T), np.nan)
    has_info = C.max(axis=1) > 0
    for k in range(len(T)):
        w = np.exp(-0.5 * ((mid - k) / sigma_kf) ** 2) * (seg_pair == seg[k])
        if (w * has_info).sum() > 1e-3:                               # else: no information near k, fill below
            s_local[k] = curve_minimum(fine, (w[:, None] * C).sum(0))
    s_local = np.exp(_fill_nan_nearest(np.log(s_local)))
    for g in np.unique(seg):
        # inside one segment the scale drifts slowly; a local value more than 1.5x away from the segment's value
        # comes from a window with too little information (typically the last few keyframes) and is clamped
        rows = np.where((seg_pair == g) & same)[0]
        if len(rows):
            s_seg = curve_minimum(fine, C[rows].sum(0))
            s_local[seg == g] = np.clip(s_local[seg == g], s_seg / 1.5, s_seg * 1.5)
    rng = np.random.default_rng(seed)
    segments = []
    for g in np.unique(seg):
        rows = np.where((seg_pair == g) & same & informative)[0]
        kfs = np.where(seg == g)[0]
        if len(rows) == 0:
            segments.append(dict(id=int(g), keyframes=[int(kfs[0]), int(kfs[-1])], pairs=0, sigma_rel_stat=None))
            continue
        total = C[rows].sum(0)
        c0 = int(np.argmin(total))
        cols = np.arange(max(0, c0 - 40), min(len(fine), c0 + 41))
        starts = np.arange(0, len(rows), block)
        boots = []
        for _ in range(n_boot):
            pick = np.concatenate([rows[b:b + block] for b in rng.choice(starts, len(starts))])
            boots.append(curve_minimum(fine[cols], C[pick][:, cols].sum(0)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        segments.append(dict(id=int(g), keyframes=[int(kfs[0]), int(kfs[-1])], pairs=int(len(rows)),
                             s=curve_minimum(fine, total), ci95_stat=[float(lo), float(hi)],
                             sigma_rel_stat=float((np.log(hi) - np.log(lo)) / (2 * 1.96))))
    sig = [g["sigma_rel_stat"] for g in segments if g["sigma_rel_stat"] is not None]
    # statistical sigma reported for the scene: keyframe-weighted mean over segments (segments without information
    # inherit the worst one)
    worst = max(sig) if sig else np.nan
    w_sig = [((g["keyframes"][1] - g["keyframes"][0] + 1), g["sigma_rel_stat"] or worst) for g in segments]
    sigma_stat = float(sum(n * s for n, s in w_sig) / sum(n for n, _ in w_sig))
    return dict(s_local=s_local, segment=seg, segments=segments, sigma_rel_stat=sigma_stat, pairs=len(pairs),
                informative_pairs=int((informative & same).sum()))


def rescale_trajectory(T: np.ndarray, s_local: np.ndarray) -> np.ndarray:
    """Re-integrate camera positions with a per-step scale: C'_k = C'_{k-1} + s_k * (C_k - C_{k-1})."""
    out = T.copy()
    C = T[:, :3, 3]
    steps = np.diff(C, axis=0) * (0.5 * (s_local[1:] + s_local[:-1]))[:, None]
    out[:, :3, 3] = np.vstack([C[:1] * s_local[0], C[:1] * s_local[0] + np.cumsum(steps, axis=0)])
    return out
