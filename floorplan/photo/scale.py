"""Metric scale of the photo tier: every available cue, fused by inverse variance (decision P-4, D-013).

Cues:
  * learned metric depth (MoGe-2), always present. After link.py pins the per-photo corrections to a common scale,
    the remaining global error is the model's systematic bias: sigma = scale_model_sigma (4%, measured by the video
    tier on this apartment against LiDAR; it is not removed as a "calibration", because that would tune on the test).
  * camera-height prior: people hold phones at chest height; weak (sigma ~11%), reported but only used when no
    better cue exists... it is kept in the fusion because it is independent of the depth model.
  * the A4 / US-Letter sheet (D-013): OPTIONAL, off by default (PhotoParams.use_sheet, D-067: no reference object in
    the protocol). When switched on, the sheet module (floorplan.scale) is called through a small adapter.

Returned scale k multiplies every length of the linked reconstruction.
"""
from __future__ import annotations

import numpy as np

CAMERA_HEIGHT_PRIOR_M = (1.35, 0.15)   # chest-height photo (protocol); mean and 1-sigma across adults


def camera_height_cue(cam_heights: list[float]) -> dict | None:
    """Scale from the median camera height above the floor vs the chest-height prior."""
    h = np.array([x for x in cam_heights if x is not None and x > 0.3])
    if len(h) < 2:
        return None
    m = float(np.median(h))
    mu, sd = CAMERA_HEIGHT_PRIOR_M
    return dict(cue="camera_height_prior", scale=mu / m, sigma_rel=sd / mu, n=int(len(h)), median_height_m=m)


def _full_res(v, ph):
    """The photo at full resolution and its intrinsics (the view's K scaled from the depth resolution)."""
    if ph is None or getattr(ph, "rgb", None) is None:
        return None, None
    H, W = ph.rgb.shape[:2]
    h, w = v.depth.shape
    if abs(W / w - H / h) > 0.02 * (W / w):          # not the same field of view: use the depth-res image
        return None, None
    s = W / w
    K = np.array(v.K, float).copy()
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return ph.rgb, K


def _observe(args):
    fn, rgb, K, depth, up, rgb_full, K_full = args
    try:
        return fn(rgb, K, depth, up, rgb_full=rgb_full, K_full=K_full), None
    except TypeError:                                   # an observe_image() without the full-resolution arguments
        return fn(rgb, K, depth, up), None
    except Exception as e:                              # a detector failure must never stop the pipeline
        return [], repr(e)


def sheet_cues(views: dict, poses: dict, photos: dict | None = None, max_pitch_deg: float = 10.0,
               workers: int = 6, photo_spread: float = 0.06) -> tuple[list[dict], str]:
    """Paper-sheet observations via floorplan.scale (opt-in: the front-end calls this only if PhotoParams.use_sheet).

    Expected interface (documented in docs/modules/photo_tier.md section 3, P-4):
        floorplan.scale.observe_image(rgb, K, depth, up_cam) -> list of dict(scale, sigma_rel, ...)
    where `scale` multiplies this image's depth to make the sheet its true size. Each observation is converted
    to the common frame by the photo's own scale correction exp(log_s)."""
    try:
        import floorplan.scale as S
    except ImportError:
        return [], "floorplan.scale not installed"
    fn = getattr(S, "observe_image", None)
    if fn is None:
        return [], "floorplan.scale has no observe_image()"
    from concurrent.futures import ProcessPoolExecutor
    todo = []
    for name, v in views.items():
        if name not in poses:
            continue
        fwd = v.R_lev @ np.array([0.0, 0.0, 1.0])
        if np.degrees(np.arcsin(np.clip(fwd[1], -1, 1))) > max_pitch_deg:   # ceiling photos cannot see the floor
            continue
        rgb_f, K_f = _full_res(v, (photos or {}).get(name))
        todo.append((name, (fn, v.rgb, v.K, v.depth, v.R_lev.T @ np.array([0, 1.0, 0]), rgb_f, K_f)))
    obs, errs = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (name, _), (res, err) in zip(todo, ex.map(_observe, [a for _, a in todo])):
            if err:
                errs.append(f"{name}: {err}")
            for o in res or []:
                # D-056: the sheet measures THIS photo's depth scale exactly (sim: 0.966 vs true 0.968, 1.028 vs
                # 1.029). It is used as a cue for the global factor directly, with MoGe-2's photo-to-photo scale
                # spread added to its sigma. Dividing by the pose graph's per-photo correction exp(log_s) instead
                # imported that correction's noise (sim truth: MAD 11% vs the raw spread's 5.6%; k became 1.10 for
                # an ideal 0.975, every room +10%).
                obs.append(dict(cue="paper_sheet", image=name, scale=float(o["scale"]),
                                sigma_rel=float(np.hypot(o.get("sigma_rel", 0.01), photo_spread)),
                                sheet_sigma_rel=float(o.get("sigma_rel", 0.01)),
                                **{k: o[k] for k in ("height_m", "paper", "score") if k in o}))
    status = f"ok: sheet seen in {len(obs)} of {len(todo)} photos searched"
    return obs, status + (f"; {len(errs)} detector errors, first {errs[0]}" if errs else "")


def fuse(cues: list[dict]) -> dict:
    """Inverse-variance fusion in log space. sigma_rel of the result is the 1-sigma relative scale uncertainty."""
    cues = [c for c in cues if c and np.isfinite(c["scale"]) and c["scale"] > 0]
    if not cues:
        return dict(scale=1.0, sigma_rel=0.1, cues=[])
    w = np.array([1 / c["sigma_rel"] ** 2 for c in cues])
    ls = np.array([np.log(c["scale"]) for c in cues])
    m = float((w * ls).sum() / w.sum())
    sig = float(1 / np.sqrt(w.sum()))
    chi2 = float((w * (ls - m) ** 2).sum())
    if len(cues) > 1 and chi2 / (len(cues) - 1) > 1:     # cues disagree more than their sigmas allow: inflate
        sig *= float(np.sqrt(chi2 / (len(cues) - 1)))
    return dict(scale=float(np.exp(m)), sigma_rel=sig, cues=cues, chi2=chi2)
