"""Metric scale for an up-to-scale video trajectory, from single-image metric depth.

Monocular odometry (DPVO) knows the camera path only up to an unknown factor, and that factor DRIFTS along the
video (measured: 0.87 -> 0.69 within 37 s on single_room) and can JUMP when tracking restarts (x2.6 and x9 on
floor_only), decision V-6. Wall lengths inherit any scale error, and the video gate is 3%.

Segments (dense, triangulation-free): for two keyframes i and j, back-project i's metric depth, move it with the
relative pose whose translation is multiplied by a candidate scale s, and compare the predicted depth with j's
metric depth at the pixels it lands on. Only the right s makes the two depth maps agree. A running median of these
per-pair scales finds the jumps, and the path is cut into segments there.

Local scale, two methods (VideoParams.scale_method, D-076):
  * "pnp" (fix loop, docs/FIX_LOOP.md, D-075): votes from PnP on the same metric depth. The depth-agreement votes were
    too few to follow DPVO's scale: on my own video 30 of 902 pairs voted, none between t 12 and 37 s, where DPVO's
    scale dropped about 13x, and the +-1.5x clamp hid the drop. For keyframes i and i+2/4/6, SIFT matches, i's metric
    depth at its keypoints and PnP give the camera step in metres. Where it agrees with DPVO's step, metric step / DPVO
    step is one vote. The running median of the votes within +-8 keyframes is the local scale, and re-integrating the
    trajectory with it removes the drift.
  * "depth_agreement" (before D-075): the fine depth-agreement curves summed in a Gaussian window (sigma 15
    keyframes), clamped to 1.5x of the segment's value.
"""
from __future__ import annotations

import cv2
import numpy as np

TRUNC = 0.1   # truncate the |log depth ratio| at 10%: occlusions and wrong pixels then cost a constant
PNP_GAPS = (2, 4, 6)   # PnP pairs: keyframes 2, 4 and 6 apart


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


def _angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def pnp_steps(images: list, D: np.ndarray, K: np.ndarray, gaps=PNP_GAPS) -> list[dict]:
    """Camera step in metres between keyframes i and i+g, without the VO: SIFT matches, i's metric depth at its
    keypoints, PnP with RANSAC in j, then a Levenberg-Marquardt refine on the inliers.

    `images` are the upright keyframes of the depth maps D (N,h,w), at any size with the same aspect; K holds the
    intrinsics of D. One dict per solved pair: i, j, R and c, with X_j = R (X_i - c), so c is j's centre in i's camera
    frame. Features are kept only for the keyframes still to be paired (long videos have 1000+ keyframes)."""
    h_d, w_d = D.shape[1:]
    sift = cv2.SIFT_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    feats, out = {}, []
    W = H = Ki = None
    for i in range(len(images)):
        for k in range(i, min(len(images), i + max(gaps) + 1)):
            if k not in feats:
                img = cv2.imread(str(images[k]))
                if Ki is None:                                       # same focal as the depth maps, at this size
                    H, W = img.shape[:2]
                    Ki = K.astype(np.float64)
                    Ki[0] *= W / w_d
                    Ki[1] *= H / h_d
                kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
                feats[k] = (np.array([p.pt for p in kp], np.float32).reshape(-1, 2), des)
        for j in (i + g for g in gaps if i + g < len(images)):
            (pi, di), (pj, dj) = feats[i], feats[j]
            if di is None or dj is None or len(di) < 50 or len(dj) < 50:
                continue
            good = [a for a, b in (m for m in bf.knnMatch(di, dj, k=2) if len(m) == 2)
                    if a.distance < 0.75 * b.distance]                  # Lowe's ratio test
            if len(good) < 40:
                continue
            ui = pi[[m.queryIdx for m in good]]
            uj = pj[[m.trainIdx for m in good]].astype(np.float64)
            z = D[i, np.clip((ui[:, 1] * h_d / H).astype(int), 0, h_d - 1),
                  np.clip((ui[:, 0] * w_d / W).astype(int), 0, w_d - 1)].astype(np.float64)
            ok = z > 0.2
            if ok.sum() < 30:
                continue
            X = np.stack([(ui[ok, 0] - Ki[0, 2]) / Ki[0, 0] * z[ok], (ui[ok, 1] - Ki[1, 2]) / Ki[1, 1] * z[ok],
                          z[ok]], 1)                                    # i's keypoints in metres, i's camera frame
            res = cv2.solvePnPRansac(X, uj[ok], Ki, None, iterationsCount=500, reprojectionError=3.0,
                                     confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
            if not res[0] or res[3] is None or len(res[3]) < 25:
                continue
            inl = res[3][:, 0]
            rv, tv = cv2.solvePnPRefineLM(X[inl], uj[ok][inl], Ki, None, res[1], res[2])
            R = cv2.Rodrigues(rv)[0]
            out.append(dict(i=i, j=j, R=R, c=-R.T @ tv[:, 0], inliers=int(len(inl))))
        feats.pop(i)
    return out


def pnp_scale_votes(steps: list[dict], T: np.ndarray, min_step_m: float, max_dir_deg: float,
                    max_rot_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale votes (i, j, metric step / DPVO step) from the PnP steps that agree with the up-to-scale trajectory T.

    A pair votes only if its PnP step is at least min_step_m long (a shorter step is mostly turning, and its length
    is noise) and points the same way as DPVO's step within max_dir_deg, with rotations within max_rot_deg (else one
    of the two is wrong for this pair)."""
    out = []
    for s in steps:
        T_ji = np.linalg.inv(T[s["j"]]) @ T[s["i"]]
        c_v = -T_ji[:3, :3].T @ T_ji[:3, 3]                              # j's centre in i's frame, DPVO units
        b_p, b_v = float(np.linalg.norm(s["c"])), float(np.linalg.norm(c_v))
        ang = np.degrees(np.arccos(np.clip(s["c"] @ c_v / max(b_p * b_v, 1e-12), -1, 1)))
        if b_p >= min_step_m and ang < max_dir_deg and _angle_deg(s["R"] @ T_ji[:3, :3].T) < max_rot_deg:
            out.append((s["i"], s["j"], b_p / max(b_v, 1e-12)))
    I, J, v = np.array(out, float).reshape(-1, 3).T
    return I.astype(int), J.astype(int), v


def running_median_scale(n: int, seg: np.ndarray, mid: np.ndarray, vote: np.ndarray, vote_seg: np.ndarray,
                         half: int, min_votes: int) -> tuple[np.ndarray, np.ndarray]:
    """Local scale of each keyframe k: the median of its segment's votes with |mid - k| <= half, if there are at least
    min_votes. A median keeps a real step of DPVO's scale sharp and ignores the odd wrong vote. Between such keyframes
    the scale is log-interpolated, inside the segment only, and held at the segment's ends. A segment with no such
    keyframe stays NaN. Returns (s_local, measured)."""
    s = np.full(n, np.nan)
    for k in range(n):
        v = vote[(vote_seg == seg[k]) & (np.abs(mid - k) <= half)]
        if len(v) >= min_votes:
            s[k] = np.median(v)
    measured = np.isfinite(s)
    for g in np.unique(seg):
        ids = np.where(seg == g)[0]
        m = ids[measured[ids]]
        if len(m):
            s[ids] = np.exp(np.interp(ids, m, np.log(s[m])))
    return s, measured


def _depth_agreement_local(C, fine, mid, seg, seg_pair, same, informative, sigma_kf, n_boot, block, seed):
    """Local scale from the depth-agreement curves (the step before D-075, kept as scale_method "depth_agreement"):
    the fine curves summed in a Gaussian time window that does not cross a segment boundary, clamped to 1.5x of the
    segment's value, and a block bootstrap over pairs per segment."""
    n = len(seg)
    s_local = np.full(n, np.nan)
    has_info = C.max(axis=1) > 0
    for k in range(n):
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
    return s_local, segments


def estimate_scales(D: np.ndarray, K: np.ndarray, T: np.ndarray, images: list, max_gap: int,
                    method: str = "pnp", sigma_kf: float = 15.0,
                    min_contrast: float = 0.005, jump: float = 1.5, n_boot: int = 2000, block: int = 10,
                    seed: int = 0, forced_cuts=(), min_step_m: float = 0.08, max_dir_deg: float = 35.0,
                    max_rot_deg: float = 4.0, half_kf: int = 8, min_votes: int = 3, exclude=None) -> dict:
    """Per-keyframe (local) metric scale of an up-to-scale trajectory T (N,4,4) from metric depth D (N,h,w) and the
    keyframe images it was predicted from.

    1. coarse cost curve per keyframe pair on a wide grid (10% steps over 6 decades)
    2. scale-jump segmentation from the running median of the per-pair coarse minima
    3. fine curves (1% steps, +-35% around the local coarse value): merge cuts with nearly the same scale both sides
    4. method "pnp": scale votes from PnP (pnp_steps, pnp_scale_votes); their running median inside each segment is
       the local scale. Method "depth_agreement": the fine curves in a Gaussian window, clamped (_depth_agreement_local;
       `images` is not used)
    5. block bootstrap, per segment, for the statistical part of the uncertainty (over the votes, or over the pairs)
    `exclude` (bool per keyframe, D-084): steep keyframes; no pair that contains one votes, in either method.
    """
    if method not in ("pnp", "depth_agreement"):
        raise ValueError(f"scale method {method!r}: 'pnp' or 'depth_agreement'")
    pairs = keyframe_pairs(len(T), max_gap)
    mid = np.array([(i + j) / 2 for i, j in pairs])
    coarse = np.exp(np.arange(np.log(1e-3), np.log(1e3), np.log(1.1)))
    Cc = _normalised(np.array([pair_cost_curve(D[i], D[j], K, T[i], T[j], coarse) for i, j in pairs]), coarse,
                     min_contrast)
    ex = np.zeros(len(T), bool) if exclude is None else np.asarray(exclude, bool)
    if ex.any():                            # D-084: a pair with a steep keyframe has no opinion
        Cc[ex[[i for i, _ in pairs]] | ex[[j for _, j in pairs]]] = 0.0
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
    if method == "depth_agreement":
        s_local, segments = _depth_agreement_local(C, fine, mid, seg, seg_pair, same, informative, sigma_kf, n_boot,
                                                   block, seed)
        return dict(s_local=s_local, segment=seg, segments=segments, sigma_rel_stat=_scene_sigma(segments),
                    pairs=len(pairs), informative_pairs=int((informative & same).sum()), method=method)

    # local scale: running median of the PnP votes. It replaces a Gaussian window over the depth-agreement curves and
    # its +-1.5x clamp around the segment's value, which hid a 13x drop of DPVO's scale on my own video.
    I, J, vote = pnp_scale_votes(pnp_steps(images, D, K), T, min_step_m, max_dir_deg, max_rot_deg)
    keep = (seg[I] == seg[J]) & ~ex[I] & ~ex[J]                      # a pair across a cut has no single DPVO scale
    vote, mid_v, seg_v = vote[keep], 0.5 * (I[keep] + J[keep]), seg[I[keep]]
    s_local, measured = running_median_scale(len(T), seg, mid_v, vote, seg_v, half_kf, min_votes)
    for g in np.unique(seg):
        # no keyframe of the segment has min_votes votes nearby: the median of its few votes, else its depth-agreement
        # scale, else (no information at all) the nearest keyframe's value
        if np.isfinite(s_local[seg == g]).any():
            continue
        rows = np.where((seg_pair == g) & same)[0]
        if (seg_v == g).any():
            s_local[seg == g] = np.median(vote[seg_v == g])
        elif len(rows) and C[rows].max() > 0:
            s_local[seg == g] = curve_minimum(fine, C[rows].sum(0))
    s_local = np.exp(_fill_nan_nearest(np.log(s_local)))
    rng = np.random.default_rng(seed)
    segments = []
    for g in np.unique(seg):
        kfs = np.where(seg == g)[0]
        sel = np.where(seg_v == g)[0]
        s_g = float(np.median(s_local[kfs]))
        # self-check signals of this scale (frontend.segment_quality): the share of the segment's keyframes that have
        # min_votes votes nearby, and how far the votes sit from the local scale (median |log|)
        cov = float(measured[kfs].mean())
        if len(sel) < min_votes:
            segments.append(dict(id=int(g), keyframes=[int(kfs[0]), int(kfs[-1])], votes=int(len(sel)), s=s_g,
                                 sigma_rel_stat=None, vote_coverage=cov, vote_residual=None))
            continue
        # the votes' offset from the local scale, resampled in blocks of consecutive votes (neighbours share frames)
        r = np.log(vote[sel] / s_local[np.round(mid_v[sel]).astype(int)])
        starts = np.arange(0, len(r), block)
        boots = [np.median(np.concatenate([r[b:b + block] for b in rng.choice(starts, len(starts))]))
                 for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        segments.append(dict(id=int(g), keyframes=[int(kfs[0]), int(kfs[-1])], votes=int(len(sel)), s=s_g,
                             ci95_stat=[s_g * float(np.exp(lo)), s_g * float(np.exp(hi))],
                             sigma_rel_stat=float((hi - lo) / (2 * 1.96)), vote_coverage=cov,
                             vote_residual=float(np.median(np.abs(r)))))
    return dict(s_local=s_local, segment=seg, segments=segments, sigma_rel_stat=_scene_sigma(segments),
                pairs=len(pairs), informative_pairs=int((informative & same).sum()), method=method,
                votes=int(len(vote)), measured_keyframes=int(measured.sum()))


def _scene_sigma(segments: list[dict]) -> float:
    """Statistical sigma reported for the scene: keyframe-weighted mean over segments (segments without information
    inherit the worst one)."""
    sig = [g["sigma_rel_stat"] for g in segments if g["sigma_rel_stat"] is not None]
    worst = max(sig) if sig else np.nan
    w_sig = [((g["keyframes"][1] - g["keyframes"][0] + 1), g["sigma_rel_stat"] or worst) for g in segments]
    return float(sum(n * s for n, s in w_sig) / sum(n for n, _ in w_sig))


def rescale_trajectory(T: np.ndarray, s_local: np.ndarray) -> np.ndarray:
    """Re-integrate camera positions with a per-step scale: C'_k = C'_{k-1} + s_k * (C_k - C_{k-1})."""
    out = T.copy()
    C = T[:, :3, 3]
    steps = np.diff(C, axis=0) * (0.5 * (s_local[1:] + s_local[:-1]))[:, None]
    out[:, :3, 3] = np.vstack([C[:1] * s_local[0], C[:1] * s_local[0] + np.cumsum(steps, axis=0)])
    return out
