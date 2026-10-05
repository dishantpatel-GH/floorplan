"""Camera path from MapAnything windows (VideoParams.path_source = "mapanything"), instead of DPVO's.

Why try it: on my own video DPVO's frame-to-frame path broke. Its scale dropped about 13x at t 34.5 s past a blank
pillar and wandered 2-4x after it, and no later step recovers a path that integrates such a step. MapAnything
(facebook/map-anything-apache, Apache-2.0) solves a whole set of views at once, in metres. Here it gets the calibrated
focal as intrinsics and, optionally, each keyframe's MoGe-2 depth as metric depth (as in the photo tier, P-2b), so it
only has to register the views. It never integrates a step, so a blank pillar costs one window, not the rest of the
walk.

All keyframes do not fit on the 8 GB GPU at once, so they are solved in windows of `size` keyframes that overlap by
`overlap`, and the windows are chained:
  * "points": the next window is fitted onto the chain by a robust rigid (SE3) or similarity (Sim3) fit of the shared
    keyframes' depth points. The same pixels appear in both windows, so the correspondences are exact.
  * "anchored": the next window gets the chain's poses of its shared keyframes as pose inputs, so MapAnything solves
    it in the chain's frame itself.
A link whose fit is poor starts a new segment, like a VO restart: the front end levels each segment and its pose
graph tries to join them.

Decision V-3 rejected images-only MapAnything windows (focal -36 to +3%, scale -9 to +71% on single_room). This
module adds the focal and the depth and measures the chained path again (scripts/mapanything_path_check.py).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

SUB = 2                     # stored depth maps keep every 2nd model pixel (disk: 228 keyframes ~ 20 MB)


def window_bounds(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """[a, b) keyframe ranges of `size` keyframes, each sharing `overlap` with the one before; the last window is
    moved back so that it is full (it then shares more)."""
    if size <= overlap:
        raise ValueError(f"window size {size} must exceed the overlap {overlap}")
    if n <= size:
        return [(0, n)]
    out, a = [], 0
    while True:
        b = min(a + size, n)
        out.append((max(0, b - size), b))
        if b >= n:
            return out
        a += size - overlap


def depth_on_model_grid(D: np.ndarray, full_size: tuple[int, int], crop: tuple[float, float, float],
                        model_wh: tuple[int, int]) -> np.ndarray:
    """A depth map of the full image (any resolution, same aspect) sampled (nearest) at the model's pixel centres."""
    (W0, H0), (ox, oy, s), (Wm, Hm) = full_size, crop, model_wh
    h, w = D.shape
    u = (np.arange(Wm) + 0.5) / s + ox
    v = (np.arange(Hm) + 0.5) / s + oy
    ud = np.clip((u * w / W0).astype(int), 0, w - 1)
    vd = np.clip((v * h / H0).astype(int), 0, h - 1)
    return D[vd[:, None], ud[None, :]].astype(np.float32)


def backproject(depth: np.ndarray, K: np.ndarray, sub: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points of the valid pixels of a z-depth map stored at 1/sub of the grid K belongs to."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64)
    ok = z > 0
    uu, vv = (u[ok] + 0.5) * sub - 0.5, (v[ok] + 0.5) * sub - 0.5
    X = np.stack([(uu - K[0, 2]) * z[ok] / K[0, 0], (vv - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
    return X, ok


def robust_fit(src: np.ndarray, dst: np.ndarray, w: np.ndarray, with_scale: bool, iters: int = 8,
               scale_m: float = 0.05) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """dst ~ s R src + t, weighted least squares with Cauchy re-weighting of the residuals (scale_m).
    Returns s, R, t and the final residuals (m)."""
    wk = w.astype(np.float64).copy()
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    for _ in range(iters):
        ws = wk / wk.sum()
        mu_s, mu_d = ws @ src, ws @ dst
        A, B = src - mu_s, dst - mu_d
        U, S, Vt = np.linalg.svd((B * ws[:, None]).T @ A)
        Dg = np.eye(3)
        Dg[2, 2] = np.sign(np.linalg.det(U @ Vt))
        R = U @ Dg @ Vt
        s = float(np.trace(np.diag(S) @ Dg) / (ws @ (A ** 2).sum(1))) if with_scale else 1.0
        t = mu_d - s * R @ mu_s
        r = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        wk = w / (1.0 + (r / scale_m) ** 2)
    return s, R, t, r


def _rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def link_windows(T_a: np.ndarray, K_a: np.ndarray, D_a: np.ndarray, C_a: np.ndarray,
                 T_b: np.ndarray, K_b: np.ndarray, D_b: np.ndarray, C_b: np.ndarray, with_scale: bool,
                 max_depth_m: float = 8.0) -> dict:
    """Similarity M taking window b's frame onto the chain, from the shared keyframes (same pixels on both sides).
    T_a: chain poses of the shared keyframes; T_b: their poses in window b; D_*: depth at 1/SUB of the model grid;
    C_*: confidence. Returns M (4x4, scale in its rotation block), s and quality numbers."""
    src, dst, w = [], [], []
    for k in range(len(T_a)):
        both = (D_a[k] > 0) & (D_b[k] > 0) & (D_a[k] < max_depth_m) & (D_b[k] < max_depth_m)
        if both.sum() < 200:
            continue
        Xa, _ = backproject(np.where(both, D_a[k], 0), K_a[k], SUB)
        Xb, _ = backproject(np.where(both, D_b[k], 0), K_b[k], SUB)
        dst.append(Xa @ T_a[k, :3, :3].T + T_a[k, :3, 3])
        src.append(Xb @ T_b[k, :3, :3].T + T_b[k, :3, 3])
        c = np.minimum(C_a[k][both], C_b[k][both]).astype(np.float64)
        w.append(np.clip(c, 1e-3, None) / np.maximum(Xa[:, 2], 0.3))       # near points are better
    poses_only = not src
    if poses_only:                  # no shared depth (blank frames): the shared cameras alone, rigid
        M_k = np.einsum("nij,njk->nik", T_a, np.linalg.inv(T_b))
        U, _, Vt = np.linalg.svd(M_k[:, :3, :3].sum(0))
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        s, t = 1.0, np.mean(T_a[:, :3, 3] - T_b[:, :3, 3] @ R.T, axis=0)
        r = np.array([np.nan])
    else:
        src, dst, w = np.concatenate(src), np.concatenate(dst), np.concatenate(w)
        s, R, t, r = robust_fit(src, dst, w, with_scale)
        s_sim3 = s if with_scale else robust_fit(src, dst, w, True)[0]
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = s * R, t
    # the shared cameras after the fit, against the chain's
    rot = [_rot_deg(T_a[k, :3, :3].T @ R @ T_b[k, :3, :3]) for k in range(len(T_a))]
    pos = [float(np.linalg.norm(T_a[k, :3, 3] - (s * R @ T_b[k, :3, 3] + t))) for k in range(len(T_a))]
    if poses_only:
        return dict(ok=True, M=M, s=s, points=0, median_resid_m=0.0, within_5cm=1.0, cam_rot_deg=float(max(rot)),
                    cam_pos_m=float(max(pos)), poses_only=True)
    return dict(ok=True, M=M, s=s, s_sim3=float(s_sim3), points=int(len(r)), median_resid_m=float(np.median(r)),
                within_5cm=float(np.mean(r < 0.05)), cam_rot_deg=float(max(rot)), cam_pos_m=float(max(pos)))


def apply_sim3(M: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Poses T (n,4,4) moved by the similarity M; rotations stay orthonormal."""
    s = np.cbrt(np.linalg.det(M[:3, :3]))
    out = T.copy()
    out[:, :3, :3] = np.einsum("ij,njk->nik", M[:3, :3] / s, T[:, :3, :3])
    out[:, :3, 3] = T[:, :3, 3] @ M[:3, :3].T + M[:3, 3]
    return out


def average_poses(T: np.ndarray) -> np.ndarray:
    """One pose from several estimates (n,4,4): chordal mean rotation, median position."""
    U, _, Vt = np.linalg.svd(T[:, :3, :3].sum(0))
    out = np.eye(4)
    out[:3, :3] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    out[:3, 3] = np.median(T[:, :3, 3], axis=0)
    return out


def chain_windows(wins: list[dict], with_scale: bool = False, average: bool = False, max_resid_m: float = 0.10,
                  min_within_5cm: float = 0.15, max_cam_rot_deg: float = 10.0, max_cam_pos_m: float = 0.30) -> dict:
    """Keyframe poses on one chain from per-window solutions ("points" chaining).

    wins: [{idx (n,), T (n,4,4), K (n,3,3), depth (n,h,w), conf (n,h,w)}] in keyframe order, consecutive ones sharing
    keyframes. Window b is fitted onto window b-1 as already placed. A link that fails the checks starts a new segment,
    placed so that the first shared keyframe keeps its pose (the front end's restart handling and pose graph take it
    from there). A keyframe then takes the pose of the window it sits most centrally in or, with `average`, the
    average of its poses in every window of that window's segment (independent estimates: the noise drops)."""
    n = int(max(int(w["idx"].max()) for w in wins)) + 1
    placed, seg_w, links = [], [], []
    g = 0
    for wi, w in enumerate(wins):
        idx = np.asarray(w["idx"])
        if wi == 0:
            placed.append(w["T"].copy())
            seg_w.append(0)
            continue
        pw = wins[wi - 1]
        prev = np.asarray(pw["idx"])
        shared = np.intersect1d(prev, idx)
        ia, ib = np.searchsorted(prev, shared), np.searchsorted(idx, shared)
        Ta = placed[wi - 1][ia]
        lk = link_windows(Ta, pw["K"][ia], pw["depth"][ia], pw["conf"][ia], w["T"][ib], w["K"][ib], w["depth"][ib],
                          w["conf"][ib], with_scale)
        good = lk["ok"] and lk["median_resid_m"] <= max_resid_m and lk["within_5cm"] >= min_within_5cm \
            and lk["cam_rot_deg"] <= max_cam_rot_deg and lk["cam_pos_m"] <= max_cam_pos_m
        if good:
            M = lk["M"]
        else:                                   # new segment: continue from the first shared keyframe's pose
            g += 1
            M = Ta[0] @ np.linalg.inv(w["T"][ib[0]])
        placed.append(apply_sim3(M, w["T"]))
        seg_w.append(g)
        links.append(dict(window=wi, keyframes=[int(idx[0]), int(idx[-1])], shared=len(shared), linked=bool(good),
                          **{k: v for k, v in lk.items() if k != "M"}))
    T = np.full((n, 4, 4), np.nan)
    seg = np.full(n, -1, int)
    src = np.full(n, -1, int)
    best = np.full(n, -np.inf)
    for wi, w in enumerate(wins):              # the window a keyframe sits most centrally in (ties: the later one)
        idx = np.asarray(w["idx"])
        centrality = np.minimum(np.arange(len(idx)), np.arange(len(idx))[::-1])
        upd = centrality >= best[idx]
        best[idx[upd]], src[idx[upd]] = centrality[upd], wi
    for k in range(n):
        wi = src[k]
        if wi < 0:
            continue
        seg[k] = seg_w[wi]
        est = [placed[v][np.searchsorted(wins[v]["idx"], k)] for v in range(len(wins))
               if seg_w[v] == seg_w[wi] and k in set(np.asarray(wins[v]["idx"]).tolist())] if average else []
        T[k] = average_poses(np.stack(est)) if len(est) > 1 else placed[wi][np.searchsorted(wins[wi]["idx"], k)]
    return dict(T=T, segment=seg, window=src, links=links)


def smooth_path(T: np.ndarray, seg: np.ndarray, sigma_kf: float) -> np.ndarray:
    """Gaussian smoothing of the keyframe poses along the walk, inside each segment (sigma in keyframes; 0 = off):
    weighted chordal mean of the rotations, weighted mean of the positions. MapAnything's per-view pose noise is
    larger than the motion between neighbouring keyframes (0.5 s apart), so a walk is smoother than its poses."""
    if sigma_kf <= 0:
        return T
    out = T.copy()
    r = int(np.ceil(3 * sigma_kf))
    for g in np.unique(seg[seg >= 0]):
        ids = np.where((seg == g) & np.isfinite(T[:, 0, 0]))[0]
        for a, k in enumerate(ids):
            nb = ids[max(0, a - r): a + r + 1]
            w = np.exp(-0.5 * ((nb - k) / sigma_kf) ** 2)
            U, _, Vt = np.linalg.svd(np.einsum("n,nij->ij", w, T[nb, :3, :3]))
            out[k, :3, :3] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            out[k, :3, 3] = w @ T[nb, :3, 3] / w.sum()
    return out


def fill_from_windows(T_glob: np.ndarray, wins: list[dict], min_anchors: int = 3, near: int = 6) -> tuple:
    """MapAnything as a LOCAL refinement of a global metric path (for example SfM): keyframes the global path lacks
    (NaN) are placed from a window that holds them and at least `min_anchors` placed keyframes. The window's poses are
    moved rigidly onto the global path by its `near` placed keyframes closest in time (chordal mean of the rotation
    offsets, median of the position offsets); both are metric, so no scale is fitted. Returns the filled path and,
    per filled keyframe, the window used and the anchors' misfit (m) after the move."""
    T = T_glob.copy()
    have = np.isfinite(T[:, 0, 0])
    info = {}
    for k in np.where(~have)[0]:
        best = None
        for wi, w in enumerate(wins):
            idx = np.asarray(w["idx"])
            if k not in set(idx.tolist()):
                continue
            anc = idx[have[idx]]
            if len(anc) < min_anchors:
                continue
            anc = anc[np.argsort(np.abs(anc - k))][:near]
            c = min(np.searchsorted(idx, k), len(idx) - 1 - np.searchsorted(idx, k))
            if best is None or c > best[0]:
                best = (c, wi, anc)
        if best is None:
            continue
        _, wi, anc = best
        w = wins[wi]
        idx = np.asarray(w["idx"])
        Tw = w["T"][np.searchsorted(idx, anc)]
        U, _, Vt = np.linalg.svd(np.einsum("nij,nkj->ik", T_glob[anc, :3, :3], Tw[:, :3, :3]))
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        t = np.median(T_glob[anc, :3, 3] - Tw[:, :3, 3] @ R.T, axis=0)
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, t
        T[k] = M @ w["T"][np.searchsorted(idx, k)]
        misfit = np.linalg.norm(T_glob[anc, :3, 3] - (Tw[:, :3, 3] @ R.T + t), axis=1)
        info[int(k)] = dict(window=int(wi), anchors=[int(a) for a in anc], anchor_misfit_m=float(np.median(misfit)))
    return T, info


# ---------------------------------------------------------------------------------------------------------- inference

def _views(rgbs, Ks, depths, poses):
    """MapAnything input views: model-grid images, intrinsics, optional metric depth and optional pose inputs."""
    from mapanything.utils.image import preprocess_inputs
    import torch
    views = []
    for k, rgb in enumerate(rgbs):
        v = {"img": rgb, "intrinsics": Ks[k].astype(np.float32)}
        if depths is not None:
            v["depth_z"] = depths[k].astype(np.float32)
        if poses is not None and poses[k] is not None:
            v["camera_poses"] = poses[k].astype(np.float32)
        views.append(v)
    views = preprocess_inputs(views, resize_mode="fixed_size", size=(int(rgbs[0].shape[1]), int(rgbs[0].shape[0])))
    for v in views:
        v["is_metric_scale"] = torch.tensor([True])
    return views


def _infer(model, views) -> list[dict]:
    import torch
    with torch.inference_mode():
        preds = model.infer(views, memory_efficient_inference=True, minibatch_size=1, use_amp=True, amp_dtype="bf16",
                            apply_mask=True, mask_edges=True)
    out = []
    for p in preds:
        m = p["mask"][0, ..., 0].bool().cpu().numpy()
        d = p["depth_z"][0, ..., 0].float().cpu().numpy()
        d[~m] = 0
        out.append(dict(T=p["camera_poses"][0].float().cpu().numpy().astype(np.float64),
                        K=p["intrinsics"][0].float().cpu().numpy().astype(np.float64),
                        depth=d[::SUB, ::SUB].astype(np.float16),
                        conf=p["conf"][0].float().cpu().numpy().reshape(d.shape)[::SUB, ::SUB].astype(np.float16)))
    torch.cuda.empty_cache()
    return out


def prepare_inputs(files: list[Path], f_px: float, depth: np.ndarray | None, long_side: int = 518):
    """Model-grid images, intrinsics and (optional) depth of the keyframes. files: upright keyframes (any size, one
    aspect); f_px: focal at that size; depth: (N,h,w) metric z-depth of the same keyframes (any size, same aspect)."""
    import cv2
    from floorplan.photo.recon import crop_for_model
    rgbs, Ks, Ds = [], [], []
    for k, f in enumerate(files):
        rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        small, crop, model_wh = crop_for_model(rgb, long_side)
        full = (rgb.shape[1], rgb.shape[0])
        ox, oy, s = crop
        Ks.append(np.array([[f_px * s, 0, (full[0] / 2 - ox) * s], [0, f_px * s, (full[1] / 2 - oy) * s], [0, 0, 1]]))
        rgbs.append(small)
        if depth is not None:
            Ds.append(depth_on_model_grid(np.asarray(depth[k], np.float32), full, crop, model_wh))
    return rgbs, Ks, (Ds if depth is not None else None)


def _stack(idx, preds) -> dict:
    return dict(idx=np.asarray(idx), T=np.stack([p["T"] for p in preds]), K=np.stack([p["K"] for p in preds]),
                depth=np.stack([p["depth"] for p in preds]), conf=np.stack([p["conf"] for p in preds]))


def solve_windows(model, rgbs, Ks, Ds, bounds, log=print) -> list[dict]:
    """Each window solved on its own (for "points" chaining)."""
    wins = []
    for a, b in bounds:
        t0 = time.time()
        p = _infer(model, _views(rgbs[a:b], Ks[a:b], None if Ds is None else Ds[a:b], None))
        wins.append(_stack(np.arange(a, b), p))
        log(f"[ma-path] window {a}-{b - 1}: {time.time() - t0:.1f} s")
    return wins


def solve_anchored(model, rgbs, Ks, Ds, bounds, log=print) -> list[dict]:
    """"anchored" chaining: window b gets the chain's poses of the keyframes it shares with the chain as pose inputs
    (its view 0 is always one of them), so MapAnything solves it in the chain's frame. Each window is stored in the
    frame of its view 0; chain_anchored places them."""
    wins = []
    for a, b in bounds:
        t0 = time.time()
        idx = np.arange(a, b)
        poses = None
        if wins:
            T = chain_anchored(wins)["T"]
            n_have = len(T)
            T0 = T[a]
            poses = [np.linalg.inv(T0) @ T[k] if k < n_have and np.isfinite(T[k, 0, 0]) else None for k in idx]
        p = _infer(model, _views(rgbs[a:b], Ks[a:b], None if Ds is None else Ds[a:b], poses))
        wins.append(_stack(idx, p))
        log(f"[ma-path] anchored window {a}-{b - 1}: {time.time() - t0:.1f} s")
    return wins


def chain_anchored(wins: list[dict], max_rot_deg: float = 10.0, max_pos_m: float = 0.30) -> dict:
    """Chain of an anchored solve: window b's view 0 is a keyframe the chain already holds, so b is placed by that
    pose. The other shared keyframes were pose inputs; if MapAnything moved them by more than 10 deg or 0.3 m it did
    not accept the anchors, and the window starts a new segment (placed by its view 0 only)."""
    n = int(max(int(w["idx"].max()) for w in wins)) + 1
    T = np.full((n, 4, 4), np.nan)
    seg = np.zeros(n, int)
    links, g = [], 0
    for wi, w in enumerate(wins):
        idx = w["idx"]
        if wi == 0:
            T[idx] = w["T"]
            continue
        shared = idx[np.isfinite(T[idx, 0, 0])]
        Tg = np.einsum("ij,njk->nik", T[idx[0]] @ np.linalg.inv(w["T"][0]), w["T"])
        ib = np.searchsorted(idx, shared)
        rot = max(_rot_deg(T[k, :3, :3].T @ Tg[i, :3, :3]) for k, i in zip(shared, ib))
        pos = max(float(np.linalg.norm(T[k, :3, 3] - Tg[i, :3, 3])) for k, i in zip(shared, ib))
        good = rot <= max_rot_deg and pos <= max_pos_m
        if not good:
            g += 1
        new = idx > shared[-1] if good else idx > shared[0]
        T[idx[new]], seg[idx[new]] = Tg[new], g
        links.append(dict(window=wi, keyframes=[int(idx[0]), int(idx[-1])], shared=int(len(shared)),
                          linked=bool(good), anchor_rot_deg=float(rot), anchor_pos_m=float(pos)))
    return dict(T=T, segment=seg, links=links)


def load_model():
    from floorplan.photo.recon import MapAnythingRunner
    return MapAnythingRunner().model


def _key(files, f_px, use_depth, size, overlap, mode) -> str:
    h = hashlib.sha1(json.dumps([[f.name for f in files], round(float(f_px), 2), bool(use_depth), int(size),
                                 int(overlap), mode]).encode()).hexdigest()[:12]
    return h


def save_windows(path: Path, wins: list[dict], extra: dict | None = None) -> None:
    arrs = {}
    for i, w in enumerate(wins):
        for k, v in w.items():
            arrs[f"w{i}_{k}"] = v
    np.savez_compressed(path, n_windows=len(wins), extra=json.dumps(extra or {}), **arrs)


def load_windows(path: Path) -> tuple[list[dict], dict]:
    z = np.load(path)
    wins = [{k: z[f"w{i}_{k}"] for k in ("idx", "T", "K", "depth", "conf")} for i in range(int(z["n_windows"]))]
    return wins, json.loads(str(z["extra"]))


def mapanything_path(files: list[Path], f_px: float, depth: np.ndarray | None, work: Path, size: int = 24,
                     overlap: int = 8, mode: str = "points", with_scale: bool = False, average: bool = False,
                     log=print) -> dict:
    """Keyframe camera-to-world poses (metres, OpenCV axes, world = keyframe 0's camera) from MapAnything windows.

    files: upright keyframes; f_px: focal at their size; depth: metric depth per keyframe (or None: MapAnything's own
    metric scale). The windows are cached in work/ (one GPU pass); the chaining runs on the CPU every time.
    Returns T (N,4,4), segment (N,), links and the cache file."""
    cache = Path(work) / f"ma_windows_{_key(files, f_px, depth is not None, size, overlap, mode)}.npz"
    bounds = window_bounds(len(files), size, overlap)
    if cache.exists():
        wins, extra = load_windows(cache)
        log(f"[ma-path] reusing {len(wins)} cached windows ({cache.name})")
    else:
        rgbs, Ks, Ds = prepare_inputs(files, f_px, depth)
        model = load_model()
        t0 = time.time()
        solve = solve_anchored if mode == "anchored" else solve_windows
        wins = solve(model, rgbs, Ks, Ds, bounds, log)
        extra = dict(seconds=round(time.time() - t0, 1), size=size, overlap=overlap, mode=mode,
                     depth_input=depth is not None)
        del model
        import torch
        torch.cuda.empty_cache()
        save_windows(cache, wins, extra)
    if mode == "anchored":
        out = chain_anchored(wins)
    else:
        out = chain_windows(wins, with_scale=with_scale, average=average)
    out.update(cache=str(cache), windows=len(wins), gpu_s=extra.get("seconds"))
    return out


# ------------------------------------------------------------------------------------------------- video front end

def scale_report(res: dict, n: int) -> dict:
    """The front end's scale-step result (estimate_scales' keys) for a MapAnything path: the path is metric already,
    so the local scale is 1 everywhere. A segment's statistical sigma is the RMS of log(scale) of the similarity fits
    of its links (how much consecutive windows disagree on scale), 3% if it has no link."""
    seg = np.asarray(res["segment"])
    segments = []
    for g in np.unique(seg):
        ids = np.where(seg == g)[0]
        s = [lk["s_sim3"] for lk in res["links"] if lk.get("linked") and lk.get("s_sim3") and
             seg[min(lk["keyframes"][1], n - 1)] == g]
        sig = float(np.sqrt(np.mean(np.log(s) ** 2))) if s else 0.03
        segments.append(dict(id=int(g), keyframes=[int(ids[0]), int(ids[-1])], s=1.0, sigma_rel_stat=sig,
                             vote_coverage=1.0, vote_residual=None, windows=int(len(s)) + 1, pairs=0))
    w = np.array([s["keyframes"][1] - s["keyframes"][0] + 1 for s in segments], float)
    sig_scene = float(np.sum(w * np.array([s["sigma_rel_stat"] for s in segments])) / w.sum())
    return dict(s_local=np.ones(n), segment=seg.astype(int), segments=segments, sigma_rel_stat=sig_scene, pairs=0,
                informative_pairs=0, method="mapanything", votes=None, measured_keyframes=int(n))


def frontend_path(files: list[Path], D: np.ndarray, T_kf: np.ndarray, vo: dict, kf: np.ndarray, ts: np.ndarray,
                  work: Path, f_px: float, geo: dict, ok: np.ndarray, ups_kf: np.ndarray, flipped: bool,
                  exclude: np.ndarray, params, log=print) -> dict:
    """What the video front end swaps in for path_source "mapanything" (same keys as path_sfm.frontend_path): metric
    keyframe poses, a keyframe-only "VO" path (frames between keyframes are interpolated), gravity re-expressed in the
    MapAnything world and the scale-step result. MapAnything gets the raw MoGe-2 depth (not capped at 4 m) when
    params.ma_depth_input is set. DPVO (T_kf, vo) is not used beyond the flip check that ran before."""
    from floorplan.video.evaluate import camera_axes_after_rotation
    from floorplan.video.frontend import _robust_mean_direction
    n = len(kf)
    depth = None
    if params.ma_depth_input:
        z = np.load(Path(work) / f"depth_{params.depth_model}.npz")
        depth = z["depth"][:n].astype(np.float32)
    cache_dir = Path(work) / "path_mapanything"
    cache_dir.mkdir(exist_ok=True)
    res = mapanything_path(list(files[:n]), f_px, depth, cache_dir, params.ma_window, params.ma_overlap,
                           params.ma_chain, params.ma_sim3, params.ma_average, log)
    T = res["T"]
    uc = geo["up_cam"][ok][: len(ups_kf)]
    if flipped:                              # same re-expression as after DPVO's fresh runs (frontend step 8)
        uc = -(uc @ camera_axes_after_rotation(180))
    ups = np.einsum("nij,nj->ni", T[ups_kf, :3, :3], uc)
    up, up_spread = _robust_mean_direction(ups)
    pitch = np.degrees(np.arcsin(np.clip(T[:, :3, 2] @ up, -1, 1)))
    linked = sum(lk["linked"] for lk in res["links"])
    log(f"[ma-path] {res['windows']} windows of {params.ma_window} (overlap {params.ma_overlap}, {params.ma_chain}, "
        f"depth input {params.ma_depth_input}): {linked} of {len(res['links'])} links kept, "
        f"{len(np.unique(res['segment']))} segment(s); gravity from {len(ups)} GeoCalib keyframes "
        f"(spread {up_spread:.1f} deg)")
    report = dict(windows=res["windows"], window=params.ma_window, overlap=params.ma_overlap, chain=params.ma_chain,
                  sim3=params.ma_sim3, average=params.ma_average, depth_input=params.ma_depth_input,
                  gpu_s=res.get("gpu_s"),
                  links=[{k: v for k, v in lk.items() if k != "M"} for lk in res["links"]])
    return dict(T_kf=T, vo=dict(vo, frames=np.asarray(kf), T_wc=T.copy()), exclude=np.asarray(exclude, bool),
                ups=ups, up=up, up_spread=up_spread, pitch=pitch, sc=scale_report(res, n), report=report)
