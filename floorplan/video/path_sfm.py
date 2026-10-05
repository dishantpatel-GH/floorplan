"""Camera path of the video tier from global structure from motion, with the metric scale from MoGe-2 depth.

Why: DPVO tracks frame to frame. On my own video (take1) its scale dropped about 13x at t 34.5 s when the phone
turned close past a blank white pillar, then wandered 2-4x, and tracking collapsed while the ceiling was filmed at the
end. One bad stretch then misplaces every later room, and each DPVO run breaks differently (I-007). Structure from motion does not need continuity: every keyframe is matched to its neighbours AND
to keyframes far apart in time, and bundle adjustment places all of them jointly. A blank stretch then costs the
keyframes that see nothing, not the scale of the rest of the walk.

Steps (path_source = "sfm" in VideoParams):
  1. pairs: each keyframe with its next `path_sfm_window` keyframes, plus long links that do not depend on the walk
     order: the `path_sfm_retrieval_k` most similar keyframes by a VLAD of the ALIKED descriptors (no extra model) and,
     optionally, every keyframe against every `path_sfm_long_every`-th one;
  2. ALIKED + LightGlue matches (hloc, GPU), pycolmap incremental mapping with the intrinsics fixed to the run's
     calibrated focal; every model is kept, not only the largest;
  3. metric scale per model: at each registered keyframe, the median ratio of its MoGe-2 depth to the SfM depth of the
     points it observes; the model's scale is the median over its keyframes (spread and a block bootstrap reported);
  4. keyframes no model registered are filled from DPVO's relative motion between registered neighbours, corrected to
     meet the next registered keyframe; gaps longer than `path_sfm_max_gap` keyframes, or fills that do not meet
     their end, stay in the walk but leave the fusion;
  5. models are joined through a DPVO link only if it is short and consistent (DPVO's metric scale agrees on both
     sides and the DPVO path fits both models' shapes); otherwise each one is its own segment, placed by the same
     link, and the existing segment machinery (levelling, pose graph, self-check) treats the boundary as a restart.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# 1. pairs
# ----------------------------------------------------------------------------------------------------------------------
def window_pairs(n: int, window: int) -> set[tuple[int, int]]:
    return {(i, j) for i in range(n) for j in range(i + 1, min(n, i + 1 + window))}


def stride_pairs(n: int, every: int, window: int) -> set[tuple[int, int]]:
    """Every keyframe against every `every`-th keyframe (beyond the sequential window)."""
    if every <= 0:
        return set()
    return {(min(i, a), max(i, a)) for a in range(0, n, every) for i in range(n) if abs(i - a) > window}


def vlad_descriptors(feats_path: Path, names: list[str], n_clusters: int = 64, per_image: int = 300,
                     seed: int = 0) -> np.ndarray:
    """One global descriptor per keyframe: VLAD (power- and intra-normalised) of its ALIKED descriptors, with a
    vocabulary learned on this video's own descriptors. Good enough to find revisits inside one walk."""
    import h5py
    from scipy.cluster.vq import kmeans2
    descs = []
    with h5py.File(str(feats_path), "r") as f:
        for n in names:
            descs.append(np.asarray(f[n]["descriptors"][()], np.float32).T)   # hloc stores (D, N)
    rng = np.random.default_rng(seed)
    sample = np.concatenate([d[rng.choice(len(d), min(per_image, len(d)), replace=False)] for d in descs if len(d)])
    C, _ = kmeans2(sample.astype(np.float64), n_clusters, iter=20, minit="++", seed=seed)
    C = C.astype(np.float32)
    out = np.zeros((len(descs), n_clusters * C.shape[1]), np.float32)
    for k, d in enumerate(descs):
        if not len(d):
            continue
        a = np.argmin((d * d).sum(1)[:, None] - 2 * d @ C.T + (C * C).sum(1)[None], axis=1)
        V = np.zeros_like(C)
        np.add.at(V, a, d - C[a])
        V = np.sign(V) * np.sqrt(np.abs(V))
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        v = V.ravel()
        out[k] = v / max(float(np.linalg.norm(v)), 1e-12)
    return out


def retrieval_pairs(G: np.ndarray, k: int, window: int) -> set[tuple[int, int]]:
    """Top-k most similar keyframes per keyframe, outside the sequential window."""
    if k <= 0:
        return set()
    S = G @ G.T
    idx = np.arange(len(S))
    S[np.abs(idx[:, None] - idx[None]) <= window] = -np.inf
    out = set()
    for i in range(len(S)):
        for j in np.argsort(-S[i])[:k]:
            if np.isfinite(S[i, j]):
                out.add((min(i, int(j)), max(i, int(j))))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 2. features, matches (GPU), mapping (CPU)
# ----------------------------------------------------------------------------------------------------------------------
def extract(image_dir: Path, names: list[str], work: Path, max_kp: int) -> Path:
    from hloc import extract_features
    conf = copy.deepcopy(extract_features.confs["aliked-n16"])
    conf["model"]["max_num_keypoints"] = max_kp
    return extract_features.main(conf, image_dir, work, image_list=names, as_half=False)   # fp32 keypoints


def match(pairs_file: Path, feats: Path, matches: Path) -> Path:
    from hloc import match_features
    conf = copy.deepcopy(match_features.confs["aliked+lightglue"])
    conf["model"]["mp"] = True
    match_features.main(conf, pairs_file, features=feats, matches=matches)   # pairs already in the file are skipped
    return matches


def write_pairs(path: Path, names: list[str], pairs) -> None:
    path.write_text("".join(f"{names[i]} {names[j]}\n" for i, j in sorted(pairs)))


def map_models(image_dir: Path, names: list[str], pairs_file: Path, feats: Path, matches: Path, out: Path,
               f_px: float, min_model: int = 10, seed: int = 0, threads: int = 1, log=print,
               abs_pose_min_inliers: int = 30, abs_pose_min_inlier_ratio: float = 0.25,
               structure_less: bool = False) -> dict:
    """pycolmap incremental mapping of every model, the focal and principal point fixed (one radial term refined).

    abs_pose_*: how many 2D-3D inliers a keyframe needs to be registered (COLMAP: 30 and 25%); structure_less: a
    keyframe with too few of them may still be registered from its 2D-2D matches to registered ones. Relaxing them
    joins the fragments of close sweeps past blank walls into one model, but a wrong one (single_room against ARKit:
    +299% in scale, 1.5 m in shape), so the defaults stay strict (params.py)."""
    import pycolmap
    from hloc.reconstruction import create_empty_db, get_image_ids, import_images
    from hloc.triangulation import estimation_and_geometric_verification, import_features, import_matches
    out.mkdir(parents=True, exist_ok=True)
    db = out / "database.db"
    create_empty_db(db)
    import cv2
    h, w = cv2.imread(str(image_dir / names[0])).shape[:2]
    import_images(image_dir, db, pycolmap.CameraMode.SINGLE, names,
                  dict(camera_model="SIMPLE_RADIAL", camera_params=f"{f_px},{w / 2},{h / 2},0"))
    ids = get_image_ids(db)
    with pycolmap.Database.open(db) as d:
        import_features(ids, d, feats)
        import_matches(ids, d, pairs_file, matches)
    estimation_and_geometric_verification(db, pairs_file)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = True
    opts.multiple_models = True
    opts.min_model_size = int(min_model)
    opts.random_seed = int(seed)
    opts.num_threads = int(threads)                 # 1 thread: the same matches give the same model (I-007)
    opts.mapper.abs_pose_min_num_inliers = int(abs_pose_min_inliers)
    opts.mapper.abs_pose_min_inlier_ratio = float(abs_pose_min_inlier_ratio)
    opts.structure_less_registration_fallback = bool(structure_less)
    (out / "models").mkdir(exist_ok=True)
    t0 = time.time()
    recs = pycolmap.incremental_mapping(db, image_dir, out / "models", options=opts)
    log(f"[path-sfm] mapping: {len(recs)} model(s) in {time.time() - t0:.0f} s: "
        + ", ".join(f"{r.num_reg_images()} images" for r in recs.values()))
    db.unlink(missing_ok=True)                      # a second copy of features and matches; the disk is small
    return recs


@dataclass
class Model:
    id: int
    kf: np.ndarray            # registered keyframe indices (sorted)
    T: np.ndarray             # (len(kf),4,4) camera-to-world in the model's frame, SfM units
    obs: list                 # per registered keyframe: (xy COLMAP pixels, depth in SfM units, reprojection px, track)
    n_points: int
    focal: float
    extra: list


def model_arrays(rec, names: list[str], mid: int) -> Model:
    index = {n: k for k, n in enumerate(names)}
    kfs, Ts, obs = [], [], []
    for im in rec.images.values():
        if not im.has_pose or im.name not in index:
            continue
        cw = np.asarray(im.cam_from_world().matrix())
        Tcw = np.eye(4)
        Tcw[:3] = cw
        p2d = [(p.point3D_id, p.xy) for p in im.points2D if p.has_point3D()]
        if p2d:
            X = np.array([rec.points3D[pid].xyz for pid, _ in p2d])
            err = np.array([rec.points3D[pid].error for pid, _ in p2d])
            trk = np.array([rec.points3D[pid].track.length() for pid, _ in p2d])
            xy = np.array([q for _, q in p2d], float).reshape(-1, 2)
            z = (X @ cw[:, :3].T + cw[:, 3])[:, 2]
        else:
            xy, z, err, trk = np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0)
        kfs.append(index[im.name])
        Ts.append(np.linalg.inv(Tcw))
        obs.append((xy, z, err, trk))
    order = np.argsort(kfs)
    cam = next(iter(rec.cameras.values()))
    return Model(id=mid, kf=np.asarray(kfs, int)[order], T=np.asarray(Ts)[order].reshape(-1, 4, 4),
                 obs=[obs[k] for k in order], n_points=int(rec.num_points3D()), focal=float(cam.params[0]),
                 extra=[float(x) for x in cam.params[3:]])


# ----------------------------------------------------------------------------------------------------------------------
# 3. metric scale from MoGe-2
# ----------------------------------------------------------------------------------------------------------------------
def keyframe_depth_ratios(m: Model, D: np.ndarray, sfm_width: int, max_err_px: float = 2.0, min_track: int = 3,
                          min_obs: int = 15) -> np.ndarray:
    """Per registered keyframe: median of MoGe-2 depth / SfM depth at its observed points (NaN if < min_obs)."""
    s = D.shape[2] / sfm_width
    out = np.full(len(m.kf), np.nan)
    for r, (k, (xy, z, err, trk)) in enumerate(zip(m.kf, m.obs)):
        if not len(z):
            continue
        u = np.floor(xy[:, 0] * s).astype(int)               # COLMAP corner convention -> depth-map pixel
        v = np.floor(xy[:, 1] * s).astype(int)
        ok = (err < max_err_px) & (trk >= min_track) & (z > 0) & (u >= 0) & (u < D.shape[2]) & (v >= 0) \
            & (v < D.shape[1])
        d = np.zeros(len(z))
        d[ok] = D[k, v[ok], u[ok]]
        ok &= d > 0
        if ok.sum() >= min_obs:
            out[r] = float(np.median(d[ok] / z[ok]))
    return out


def model_scale(ratios: np.ndarray, block: int = 10, n_boot: int = 1000, seed: int = 0) -> dict:
    """Robust scale of one model (metres per SfM unit) from its keyframes' depth ratios, with a block bootstrap."""
    r = ratios[np.isfinite(ratios)]
    if len(r) < 3:
        return dict(scale=float("nan"), keyframes=int(len(r)))
    lr = np.log(r)
    s = float(np.exp(np.median(lr)))
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(lr), block)
    boots = [np.median(np.concatenate([lr[b:b + block] for b in rng.choice(starts, len(starts))])) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p10, p90 = np.percentile(r, [10, 90])
    return dict(scale=s, keyframes=int(len(r)), p10_rel=float(p10 / s), p90_rel=float(p90 / s),
                mad_rel=float(1.4826 * np.median(np.abs(lr - np.log(s)))),
                sigma_rel_stat=float((hi - lo) / (2 * 1.96)))


def local_scale_check(ratios: np.ndarray, scale: float, half: int = 10, max_ratio: float = 1.25,
                      min_n: int = 5) -> tuple[np.ndarray, dict]:
    """Keyframes where the model is bent: the running median (+-half keyframes of the model) of the depth ratios
    departs from the model's scale by more than max_ratio. A wrong long link can fold part of a model onto another place
    at the wrong size (take1 with every keyframe linked to every 4th one: a 20-keyframe stretch at 2.85x the model's
    scale); MoGe-2 sees each keyframe on its own, so it notices. Returns the keep mask and a report."""
    n = len(ratios)
    loc = np.full(n, np.nan)
    for i in range(n):
        w = ratios[max(0, i - half): i + half + 1]
        w = w[np.isfinite(w)]
        if len(w) >= min_n:
            loc[i] = np.median(w) / scale
    bad = np.isfinite(loc) & (np.abs(np.log(np.where(np.isfinite(loc), loc, 1.0))) > np.log(max_ratio))
    dev = np.abs(np.log(loc[np.isfinite(loc)])) if np.isfinite(loc).any() else np.zeros(1)
    return ~bad, dict(local_scale_max_dev=float(np.exp(dev.max())), bent_keyframes=int(bad.sum()))


def drop_keyframes(m: Model, keep: np.ndarray) -> Model:
    return Model(id=m.id, kf=m.kf[keep], T=m.T[keep], obs=[o for o, k in zip(m.obs, keep) if k], n_points=m.n_points,
                 focal=m.focal, extra=m.extra)


# ----------------------------------------------------------------------------------------------------------------------
# 4-5. one path: models placed through DPVO links, gaps filled from DPVO
# ----------------------------------------------------------------------------------------------------------------------
def _chordal_mean(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(Rs.sum(0))
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _rot_deg(R: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def dpvo_metric_scale(T_m: np.ndarray, T_vo: np.ndarray, ids: np.ndarray, min_step_m: float = 0.05,
                      gap: int = 2) -> float:
    """Metres per DPVO unit over the keyframes `ids` (all placed): median of metric step / DPVO step for keyframes
    `gap` apart in the list whose metric step is at least min_step_m. NaN if fewer than 3 such steps."""
    if len(ids) <= gap:
        return float("nan")
    a, b = ids[:-gap], ids[gap:]
    dm = np.linalg.norm(T_m[b, :3, 3] - T_m[a, :3, 3], axis=1)
    dv = np.linalg.norm(T_vo[b, :3, 3] - T_vo[a, :3, 3], axis=1)
    ok = (dm >= min_step_m) & (dv > 1e-9)
    return float(np.median(dm[ok] / dv[ok])) if ok.sum() >= 3 else float("nan")


def dpvo_chain(T_ref: np.ndarray, T_vo: np.ndarray, a: int, ks: np.ndarray, s: float, R_align=None) -> np.ndarray:
    """Poses of keyframes `ks` from keyframe a's placed pose and DPVO's motion since a, DPVO translation times s.
    R_align (DPVO world -> path world) defaults to the one implied by keyframe a alone."""
    if R_align is None:
        R_align = T_ref[a, :3, :3] @ T_vo[a, :3, :3].T
    out = np.tile(np.eye(4), (len(ks), 1, 1))
    out[:, :3, :3] = np.einsum("ij,njk->nik", R_align, T_vo[ks, :3, :3])
    out[:, :3, 3] = T_ref[a, :3, 3] + s * (T_vo[ks, :3, 3] - T_vo[a, :3, 3]) @ R_align.T
    return out


def _end_correct(T_fill: np.ndarray, T_end_pred: np.ndarray, T_end: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Spread the miss at the far end of a fill linearly over it (rotation by slerp, position linearly)."""
    from scipy.spatial.transform import Rotation
    dR = Rotation.from_matrix(T_end[:3, :3] @ T_end_pred[:3, :3].T).as_rotvec()
    e = T_end[:3, 3] - T_end_pred[:3, 3]
    out = T_fill.copy()
    for k, a in enumerate(alpha):
        Rk = Rotation.from_rotvec(a * dR).as_matrix()
        out[k, :3, :3] = Rk @ T_fill[k, :3, :3]
        out[k, :3, 3] = T_fill[k, :3, 3] + a * e
    return out


def assemble_path(models: list[Model], scales: dict, T_vo: np.ndarray, ts: np.ndarray, params, log=print) -> dict:
    """One metric path for every keyframe from the SfM models (metric by `scales`) and DPVO's poses.

    Returns T_wc (N,4,4), segment (N,) (one id per group of models joined through consistent links, numbered in walk
    order), placed (N,) bool (pose from SfM or from a consistent short DPVO fill: may enter the fusion), source (N,)
    in {"sfm", "fill", "dpvo"} and a report."""
    N = len(T_vo)
    window, max_gap, link_gap = params.path_sfm_link_window, params.path_sfm_max_gap, params.path_sfm_link_max_gap
    good = [m for m in models if np.isfinite(scales[m.id]["scale"])]
    owner = np.full(N, -1)
    T_own = np.full((N, 4, 4), np.nan)
    for m in sorted(good, key=lambda m: -len(m.kf)):           # a keyframe in two models belongs to the larger one
        Tm = m.T.copy()
        Tm[:, :3, 3] *= scales[m.id]["scale"]
        free = owner[m.kf] < 0
        owner[m.kf[free]] = m.id
        T_own[m.kf[free]] = Tm[free]
    rep = dict(models=[], links=[])
    if not (owner >= 0).any():
        raise RuntimeError("no SfM model with a metric scale")
    # runs: maximal stretches of keyframes with the same owner (unowned keyframes do not break a run)
    owned = np.where(owner >= 0)[0]
    runs, start = [], owned[0]
    for p, q in zip(owned[:-1], owned[1:]):
        if owner[q] != owner[p]:
            runs.append((owner[p], start, p))
            start = q
    runs.append((owner[owned[-1]], start, owned[-1]))

    def near(mid, k, side):                                   # up to `window` owned keyframes of model mid next to k
        ids = np.where(owner == mid)[0]
        ids = ids[ids <= k][-window:] if side == "before" else ids[ids >= k][:window]
        return ids

    # candidate links between consecutive runs of different models
    cands = []
    for (ma, _, a), (mb, b, _) in zip(runs[:-1], runs[1:]):
        cands.append((int(b - a), int(ma), int(mb), int(a), int(b)))
    # place the largest model's frame first; then attach models through their best link (consistent ones first)
    sizes = {m.id: int((owner == m.id).sum()) for m in good}
    anchor = max(sizes, key=sizes.get)
    M = {anchor: np.eye(4)}                                   # model frame -> path frame (rigid, metric already)
    group = {anchor: 0}
    T = np.full((N, 4, 4), np.nan)

    def to_path(mid):
        ids = np.where(owner == mid)[0]
        T[ids] = np.einsum("ij,njk->nik", M[mid], T_own[ids])

    to_path(anchor)

    def try_link(c):
        gap, ma, mb, a, b = c
        if (ma in M) == (mb in M):
            return None
        src, dst, k_src, k_dst = (ma, mb, a, b) if ma in M else (mb, ma, b, a)
        side_src, side_dst = ("before", "after") if ma in M else ("after", "before")
        ids_s, ids_d = near(src, k_src, side_src), near(dst, k_dst, side_dst)
        s_src = dpvo_metric_scale(T, T_vo, ids_s)
        s_dst = dpvo_metric_scale(T_own, T_vo, ids_d)
        s = s_src if np.isfinite(s_src) else s_dst if np.isfinite(s_dst) else global_s
        R_al = _chordal_mean(np.einsum("nij,nkj->nik", T[ids_s, :3, :3], T_vo[ids_s, :3, :3]))
        chain = dpvo_chain(T, T_vo, k_src, ids_d, s, R_al)
        R_m = _chordal_mean(np.einsum("nij,nkj->nik", chain[:, :3, :3], T_own[ids_d, :3, :3]))
        t_m = np.mean(chain[:, :3, 3] - T_own[ids_d, :3, 3] @ R_m.T, axis=0)
        Mx = np.eye(4)
        Mx[:3, :3], Mx[:3, 3] = R_m, t_m
        placed = np.einsum("ij,njk->nik", Mx, T_own[ids_d])
        rot_res = float(np.median(_rot_deg(np.einsum("nij,nkj->nik", placed[:, :3, :3], chain[:, :3, :3]))))
        pos_res = float(np.median(np.linalg.norm(placed[:, :3, 3] - chain[:, :3, 3], axis=1)))
        span = float(np.linalg.norm(np.diff(chain[:, :3, 3], axis=0), axis=1).sum()) if len(chain) > 1 else 0.0
        ratio = float(s_src / s_dst) if np.isfinite(s_src) and np.isfinite(s_dst) else float("nan")
        reasons = []
        if gap > link_gap:
            reasons.append(f"gap {gap} > {link_gap} keyframes")
        if not (np.isfinite(ratio) and abs(np.log(ratio)) <= np.log(params.path_sfm_link_max_scale_ratio)):
            reasons.append(f"DPVO scale ratio across the link {ratio:.2f}")
        if rot_res > params.path_sfm_link_max_rot_deg:
            reasons.append(f"rotation residual {rot_res:.1f} deg")
        if pos_res > params.path_sfm_link_max_pos_m + 0.1 * span:
            reasons.append(f"position residual {pos_res:.2f} m")
        return dict(src=int(src), dst=int(dst), keyframes=[int(a), int(b)], gap=gap, scale_src=s_src,
                    scale_dst=s_dst, rot_res_deg=rot_res, pos_res_m=pos_res, consistent=not reasons,
                    reasons=reasons, M=Mx)

    global_s = dpvo_metric_scale(T_own, T_vo, np.where(owner == anchor)[0])
    if not np.isfinite(global_s):
        global_s = 1.0
    n_groups = 1
    while len(M) < len(sizes):
        tried = [r for r in (try_link(c) for c in sorted(cands)) if r is not None]
        if not tried:
            break
        ok = [r for r in tried if r["consistent"]]
        best = ok[0] if ok else min(tried, key=lambda r: r["gap"])
        M[best["dst"]] = best["M"]
        if best["consistent"]:
            group[best["dst"]] = group[best["src"]]
        else:
            group[best["dst"]] = n_groups
            n_groups += 1
        to_path(best["dst"])
        rep["links"].append({k: v for k, v in best.items() if k != "M"})
        log(f"[path-sfm] model {best['dst']} placed from model {best['src']} via keyframes {best['keyframes']}: "
            + ("consistent" if best["consistent"] else "NOT joined (" + "; ".join(best["reasons"]) + ")"))
    for mid in sizes:                                         # models without any link (cannot happen in walk order)
        if mid not in M:
            owner[owner == mid] = -1
    # fill unowned keyframes from DPVO between placed neighbours
    source = np.where(owner >= 0, "sfm", "").astype(object)
    seg_g = np.array([group.get(int(o), -1) if o >= 0 else -1 for o in owner])
    fills = []
    k = 0
    while k < N:
        if owner[k] >= 0:
            k += 1
            continue
        b = k
        while b < N and owner[b] < 0:
            b += 1
        a = k - 1                                             # gap keyframes k..b-1; placed neighbours a and b
        ks = np.arange(k, b)
        lo = a if a >= 0 else None
        hi = b if b < N else None
        ref = lo if lo is not None else hi
        ids = np.concatenate([near(owner[lo], lo, "before") if lo is not None else [],
                              near(owner[hi], hi, "after") if hi is not None else []]).astype(int)
        s = dpvo_metric_scale(T, T_vo, np.unique(ids))
        s = s if np.isfinite(s) else global_s
        fill = dpvo_chain(T, T_vo, ref, ks, s)
        info = dict(keyframes=[int(k), int(b - 1)], n=int(len(ks)), scale=s)
        consistent = len(ks) <= max_gap
        if lo is not None and hi is not None:
            end_pred = dpvo_chain(T, T_vo, lo, np.array([hi]), s)[0]
            miss = float(np.linalg.norm(end_pred[:3, 3] - T[hi, :3, 3]))
            miss_deg = float(_rot_deg(end_pred[:3, :3] @ T[hi, :3, :3].T))
            span = float(np.linalg.norm(T[hi, :3, 3] - T[lo, :3, 3]))
            alpha = (ts[ks] - ts[lo]) / max(ts[hi] - ts[lo], 1e-6)
            fill = _end_correct(fill, end_pred, T[hi], alpha)
            info.update(end_miss_m=miss, end_miss_deg=miss_deg, same_group=bool(seg_g[lo] == seg_g[hi]))
            consistent &= (miss <= params.path_sfm_fill_max_miss_m + 0.3 * span
                           and miss_deg <= params.path_sfm_link_max_rot_deg and seg_g[lo] == seg_g[hi])
        else:
            consistent = False                                # head or tail of the walk: no second anchor
        T[ks] = fill
        source[ks] = "fill" if consistent else "dpvo"
        seg_g[ks] = seg_g[lo] if lo is not None else seg_g[hi]
        info["used"] = bool(consistent)
        fills.append(info)
        k = b
    # segments in walk order (ids 0.. by first appearance of each group)
    first = {}
    for g in seg_g:
        first.setdefault(int(g), len(first))
    seg = np.array([first[int(g)] for g in seg_g])
    placed = source != "dpvo"
    for m in good:
        if m.id in M:
            rep["models"].append(dict(id=m.id, keyframes=int((owner == m.id).sum()), first=int(m.kf[0]),
                                      last=int(m.kf[-1]), group=int(first.get(group[m.id], -1))))
    rep.update(anchor=int(anchor), groups=int(len(first)), fills=fills,
               keyframes_sfm=int((source == "sfm").sum()), keyframes_filled=int((source == "fill").sum()),
               keyframes_dropped=int((source == "dpvo").sum()), dpvo_scale_median=global_s)
    return dict(T_wc=T, segment=seg, placed=placed, source=source.astype(str), report=rep)


# ----------------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------------
def build_models(files: list[Path], work: Path, f_px: float, params, log=print, gpu: bool = True) -> tuple[list, dict]:
    """Steps 1-2, cached in `work`: features and matches are reused when present (the GPU part), the mapping when the
    pair list, focal and mapper settings are the same."""
    image_dir = files[0].parent
    names = [f.name for f in files]
    work.mkdir(parents=True, exist_ok=True)
    feats = work / "feats-aliked-n16.h5"
    if not feats.exists():
        if not gpu:
            raise RuntimeError("SfM path: features missing (GPU step)")
        extract(image_dir, names, work, params.max_keypoints)
    n = len(names)
    pairs = window_pairs(n, params.path_sfm_window)
    n_win = len(pairs)
    G = vlad_descriptors(feats, names, seed=params.seed)
    ret = retrieval_pairs(G, params.path_sfm_retrieval_k, params.path_sfm_window)
    lng = stride_pairs(n, params.path_sfm_long_every, params.path_sfm_window)
    pairs |= ret | lng
    pairs_file = work / "pairs.txt"
    write_pairs(pairs_file, names, pairs)
    matches = work / "matches.h5"
    if not _all_matched(matches, names, pairs):
        if not gpu:
            raise RuntimeError("SfM path: matches missing (GPU step)")
        match(pairs_file, feats, matches)
    key = dict(names=names, pairs=len(pairs), f=round(float(f_px), 3), min_model=params.path_sfm_min_model,
               seed=params.seed, window=params.path_sfm_window, k=params.path_sfm_retrieval_k,
               every=params.path_sfm_long_every, abs_pose=[params.path_sfm_abs_pose_min_inliers,
                                                           params.path_sfm_abs_pose_min_inlier_ratio],
               structure_less=params.path_sfm_structure_less)
    meta = work / "models.json"
    import pycolmap
    if meta.exists() and json.loads(meta.read_text()).get("key") == key:
        recs = {int(p.name): pycolmap.Reconstruction(str(p)) for p in sorted((work / "sfm/models").iterdir())
                if p.is_dir()}
        log(f"[path-sfm] reusing {len(recs)} cached model(s)")
    else:
        if (work / "sfm").is_symlink():             # models of other settings: never mixed in
            (work / "sfm").unlink()
        elif (work / "sfm").exists():
            import shutil
            shutil.rmtree(work / "sfm")
        recs = map_models(image_dir, names, pairs_file, feats, matches, work / "sfm", f_px,
                          params.path_sfm_min_model, params.seed, params.path_sfm_threads, log,
                          params.path_sfm_abs_pose_min_inliers, params.path_sfm_abs_pose_min_inlier_ratio,
                          params.path_sfm_structure_less)
        meta.write_text(json.dumps(dict(key=key)))
    models = [model_arrays(r, names, int(i)) for i, r in sorted(recs.items())]
    stats = dict(pairs=len(pairs), window_pairs=n_win, retrieval_pairs=len(ret - window_pairs(n, params.path_sfm_window)),
                 stride_pairs=len(lng), models=len(models), registered=int(sum(len(m.kf) for m in models)))
    return models, stats


def _all_matched(matches: Path, names: list[str], pairs) -> bool:
    if not matches.exists():
        return False
    import h5py
    from hloc.utils.io import find_pair
    with h5py.File(str(matches), "r") as f:
        for i, j in pairs:
            try:
                find_pair(f, names[i], names[j])
            except ValueError:
                return False
    return True


def sfm_camera_path(files: list[Path], D: np.ndarray, T_vo: np.ndarray, ts_kf: np.ndarray, work: Path, f_px: float,
                    params, log=print, gpu: bool = True) -> dict:
    """The whole SfM path for the keyframes `files` (upright, final rotation), D their metric depth maps (cleaned),
    T_vo DPVO's keyframe poses (for gap fills and links only), f_px the focal at the size of `files`."""
    import cv2
    t0 = time.time()
    models, stats = build_models(files, work, f_px, params, log, gpu)
    w_sfm = cv2.imread(str(files[0])).shape[1]
    scales, per_kf = {}, {}
    for k_, m in enumerate(models):
        r = keyframe_depth_ratios(m, D, w_sfm)
        sc = model_scale(r, seed=params.seed)
        if np.isfinite(sc["scale"]):
            keep, chk = local_scale_check(r, sc["scale"], max_ratio=params.path_sfm_max_local_scale)
            sc.update(chk)
            if not keep.all():                     # bent stretches leave the model; the scale is taken again
                models[k_] = m = drop_keyframes(m, keep)
                r = r[keep]
                sc = dict(model_scale(r, seed=params.seed), **chk)
        per_kf[m.id] = r
        scales[m.id] = sc
        log(f"[path-sfm] model {m.id}: {len(m.kf)} keyframes ({m.kf[0] if len(m.kf) else '-'}-"
            f"{m.kf[-1] if len(m.kf) else '-'}), {m.n_points} points, scale {sc['scale']:.4f} m/unit from "
            f"{sc['keyframes']} keyframes (p10-p90 {sc.get('p10_rel', np.nan):.3f}-{sc.get('p90_rel', np.nan):.3f}, "
            f"stat 1-sigma {100 * sc.get('sigma_rel_stat', np.nan):.2f}%; local scale within "
            f"{sc.get('local_scale_max_dev', np.nan):.2f}x, {sc.get('bent_keyframes', 0)} bent keyframes left out)")
    big = [m for m in models if len(m.kf) >= params.path_sfm_min_model and scales[m.id]["keyframes"] >= 3]
    path = assemble_path(big, scales, T_vo, ts_kf, params, log)
    path["models"] = big
    path["scales"] = scales
    path["ratios"] = per_kf
    path["stats"] = dict(stats, runtime_s=round(time.time() - t0, 1), **path["report"])
    log(f"[path-sfm] {path['report']['keyframes_sfm']} keyframes from SfM, {path['report']['keyframes_filled']} "
        f"filled from DPVO, {path['report']['keyframes_dropped']} left out of the fusion; "
        f"{path['report']['groups']} segment group(s)")
    return path


def largest_coverage(models: list[Model], n: int) -> float:
    """Share of the walk's n keyframes in the largest model (after the bent-model check)."""
    return max((len(m.kf) for m in models), default=0) / max(n, 1)


def scale_report(path: dict, n: int, params) -> dict:
    """The front end's scale-step result (estimate_scales' keys) for an SfM path: the path is metric already, so the
    local scale is 1 everywhere; each segment carries the statistical sigma of its models' MoGe-2 scales."""
    seg = path["segment"]
    owner_scale = {}
    for m in path["models"]:
        owner_scale[m.id] = path["scales"][m.id]
    segments = []
    for g in np.unique(seg):
        ids = np.where(seg == g)[0]
        mids = [x["id"] for x in path["report"]["models"] if x["group"] == g]
        sig = [owner_scale[i].get("sigma_rel_stat") for i in mids if owner_scale[i].get("sigma_rel_stat") is not None]
        segments.append(dict(id=int(g), keyframes=[int(ids[0]), int(ids[-1])], s=1.0, models=mids,
                             model_scales_m_per_unit=[owner_scale[i]["scale"] for i in mids],
                             sigma_rel_stat=float(max(sig)) if sig else None,
                             vote_coverage=float(np.mean(path["placed"][ids])), vote_residual=None,
                             pairs=0))
    known = [s for s in segments if s["sigma_rel_stat"] is not None]
    worst = max((s["sigma_rel_stat"] for s in known), default=0.05)
    w = np.array([s["keyframes"][1] - s["keyframes"][0] + 1 for s in segments], float)
    sig_scene = float(np.sum(w * np.array([s["sigma_rel_stat"] if s["sigma_rel_stat"] is not None else worst
                                            for s in segments])) / w.sum())
    return dict(s_local=np.ones(n), segment=seg.astype(int), segments=segments, sigma_rel_stat=sig_scene,
                pairs=int(path["stats"]["pairs"]), informative_pairs=int(path["stats"]["pairs"]), method="sfm",
                votes=None, measured_keyframes=int(path["placed"].sum()))


def frontend_path(files: list[Path], D: np.ndarray, T_kf: np.ndarray, vo: dict, kf: np.ndarray, ts: np.ndarray,
                  work: Path, f_px: float, geo: dict, ok: np.ndarray, ups_kf: np.ndarray, flipped: bool,
                  exclude: np.ndarray, params, log=print) -> dict:
    """What the video front end swaps in for path_source "sfm": metric keyframe poses, a keyframe-only "VO" path
    (frames between keyframes are interpolated), the keyframes left out of the fusion, gravity re-expressed in the
    SfM world, and the scale-step result."""
    from floorplan.video.evaluate import camera_axes_after_rotation
    path = sfm_camera_path(files, D, T_kf, ts[kf], work / "path_sfm", f_px, params, log)
    cov = largest_coverage(path["models"], len(kf))
    path["stats"]["largest_model_coverage"] = cov
    (work / "path_sfm" / "path_report.json").write_text(json.dumps(dict(path["stats"], scales={
        int(k): v for k, v in path["scales"].items()}), indent=1, default=float))
    if cov < params.path_sfm_min_coverage:
        log(f"[path-sfm] the largest model holds {100 * cov:.0f}% of the keyframes (< "
            f"{100 * params.path_sfm_min_coverage:.0f}%): the walk is in pieces, DPVO's path is kept")
        return None
    T = path["T_wc"]
    uc = geo["up_cam"][ok][: len(ups_kf)]
    if flipped:                              # same re-expression as after DPVO's fresh runs (frontend step 8)
        uc = -(uc @ camera_axes_after_rotation(180))
    use = path["placed"][ups_kf]
    ups = np.einsum("nij,nj->ni", T[ups_kf, :3, :3], uc)
    from floorplan.video.frontend import _robust_mean_direction
    up, up_spread = _robust_mean_direction(ups[use] if use.sum() >= 3 else ups)
    pitch = np.degrees(np.arcsin(np.clip(T[:, :3, 2] @ up, -1, 1)))
    log(f"[path-sfm] gravity in the SfM world from {int(use.sum())} GeoCalib keyframes (spread {up_spread:.1f} deg)")
    return dict(T_kf=T, vo=dict(vo, frames=np.asarray(kf), T_wc=T.copy()), exclude=np.asarray(exclude, bool)
                | ~path["placed"], ups=ups, up=up, up_spread=up_spread, pitch=pitch,
                sc=scale_report(path, len(T), params),
                report=dict(path["stats"], scales={int(k): v for k, v in path["scales"].items()}))
