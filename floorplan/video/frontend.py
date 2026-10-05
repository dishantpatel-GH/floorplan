"""Video tier front-end: one RGB video -> metric, gravity-aligned 3D scene in the same format as the LiDAR tier.

Pipeline (each step is explained in docs/modules/video_tier.md):
  1. scan the video: timestamps, image motion, sharpness            (frames.py)
  2. motion-triggered keyframes, sharpest frame per trigger          (frames.py)
  3. upright rotation: container flag, else GeoCalib axis vote       (orientation.py)
  4. focal length: SfM self-calibration on the first keyframes, GeoCalib as fallback (sfm.py, orientation.py)
  5. camera path for every 2nd frame with DPVO (up to scale)        (vo.py)
  6. gravity from GeoCalib up vectors; flip check from camera pitch  (orientation.py, here)
  7. metric depth per keyframe with MoGe-2                           (depth.py)
  8. metric scale: segment cuts from depth agreement, local scale from PnP votes or depth agreement (scale.py, D-076)
  9. TSDF fusion of the predicted depth with the shared LiDAR-tier code (floorplan.recon.fusion)
 10. floor refinement + Manhattan alignment with the shared code     (floorplan.plan.align)
v2 (docs/modules/video_tier.md, "Video tier v2"):
  * focal = robust fusion of container metadata (ffprobe), SfM and GeoCalib, with a 1-sigma that enters the scale sigma
  * scale segments at VO restarts; a fresh DPVO run per segment; per-segment self-check (untrusted -> left out)
  * per-segment gravity levelling + restart continuity; fragment pose graph on predicted depth (posegraph.py)
  * PnP scale (D-076): the path is moved vertically so that every keyframe sees the floor at one height
  * optional paper-sheet cue (floorplan.scale; off by default, D-067) with flatness and consistency gates
  * info["scale_sigma_rel"] and info["whole_scene_consistent"] are the honest summary for downstream code
The result plugs into the same plan extractors as the LiDAR tier; only the uncertainty is wider.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from floorplan.config import Config
from floorplan.plan.align import align_scene, estimate_floor, gravity_correction
from floorplan.recon.fusion import fuse_tsdf
from floorplan.video import depth as depth_mod
from floorplan.video.evaluate import camera_axes_after_rotation
from floorplan.video.frames import export_keyframes, resolve_video, scan_video, select_keyframes
from floorplan.video.orientation import detect_rotation, gravity_and_focal
from floorplan.video.params import VideoParams
from floorplan.video.scale import estimate_scales, rescale_trajectory
from floorplan.video.sfm import run_sfm
from floorplan.video.vo import interpolate_poses, run_dpvo


@dataclass
class VideoCapture:
    """Capture-like adapter so the LiDAR tier's fuse_tsdf can fuse predicted depth unchanged.

    Frame ids are keyframe ordinals 0..N-1; depth(i) returns the cleaned metric depth of keyframe i."""
    depth_maps: np.ndarray            # (N, h, w) metres, 0 = invalid
    K_d: np.ndarray                   # (3,3) intrinsics of the depth maps
    image_files: list[Path]           # upright colour keyframes (any resolution with the same aspect)
    rgb_size: tuple[int, int] = (0, 0)
    K_rgb: np.ndarray = field(default=None)

    def __post_init__(self):
        h, w = cv2.imread(str(self.image_files[0])).shape[:2]
        self.rgb_size = (w, h)
        s = w / self.depth_maps.shape[2]
        K = self.K_d.copy()
        K[:2] *= s
        self.K_rgb = np.repeat(K[None], len(self.image_files), axis=0)

    @property
    def n(self) -> int:
        return len(self.depth_maps)

    def depth(self, i: int, min_conf: int = 0, min_m: float = 0.0, max_m: float = np.inf) -> np.ndarray:
        d = self.depth_maps[i].copy()
        d[(d < min_m) | (d > max_m)] = 0.0
        return d

    def K_depth(self, i: int) -> np.ndarray:
        return self.K_d

    def iter_rgb(self, indices) -> Iterator[tuple[int, np.ndarray]]:
        for i in sorted(set(int(k) for k in indices)):
            yield i, cv2.imread(str(self.image_files[i]))


def _intrinsics(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def _robust_mean_direction(v: np.ndarray, max_dev_deg: float = 10.0, iters: int = 3) -> tuple[np.ndarray, float]:
    """Mean unit vector after iteratively dropping outliers; returns it and the angular spread (deg)."""
    m = v.mean(0)
    keep = np.ones(len(v), bool)
    for _ in range(iters):
        m = v[keep].mean(0)
        m /= np.linalg.norm(m)
        keep = np.degrees(np.arccos(np.clip(v @ m, -1, 1))) < max_dev_deg
        if keep.sum() < 3:
            break
    dev = np.degrees(np.arccos(np.clip(v[keep] @ m, -1, 1)))
    return m, float(np.median(dev)) if len(dev) else float("nan")


def _rotation_to_y(up: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector `up` onto +y."""
    from floorplan.plan.align import _rot_between
    return _rot_between(up, np.array([0.0, 1.0, 0.0]))


def _focal(files: list[Path], geo_f: float, meta: dict, work: Path, params: VideoParams, log) -> tuple[float, dict]:
    """Focal length (px at the SfM image size) and its 1-sigma relative uncertainty (v2, decision V2-2).

    Sources, fused in log space with the robust group-aware fusion of floorplan.scale (outlier rejection with >= 3
    sources, Birge inflation when they disagree):
      * container metadata (35 mm equivalent; iPhone QuickTime keys), sigma 5% (video crop / stabilisation unknown);
      * SfM self-calibration on the first keyframes (>= calib_min_registered images), sigma 3%;
      * GeoCalib median over keyframes, sigma 3.5%.
    The image-based sources were biased the same way on this data, so the fused sigma is floored at 2%.
    Cached in work/calib.json: SfM is not bit-repeatable (v1 Issue 13c) and every later cache depends on f."""
    import json
    from floorplan.scale.fuse import ScaleCue, fuse_scale
    from floorplan.video.metadata import FOCAL_META_SIGMA, focal_px_from_f35
    names = [f.name for f in files[: params.calib_max_images]]
    cache = work / "calib.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("names") == names:
            log(f"[calib] reusing cached focal {c['f']:.1f} px ({c['source']})")
            return float(c["f"]), c
    w_s, h_s = cv2.imread(str(files[0])).shape[1::-1]
    cues = [ScaleCue("geocalib", "other", geo_f, params.focal_sigma_geocalib)]
    calib = dict(geocalib_f=geo_f)
    try:
        res = run_sfm(files[0].parent, names, work / "sfm_calib", geo_f, params, log=log)
        calib.update(sfm=res.stats)
        if res.stats["registered"] >= params.calib_min_registered:
            cues.append(ScaleCue("sfm_self_calibration", "other", float(res.K[0, 0]), params.focal_sigma_sfm))
            calib["sfm_f"] = float(res.K[0, 0])
    except RuntimeError as e:
        log(f"[calib] SfM failed ({e})")
    if meta.get("f35_mm"):
        f_meta = focal_px_from_f35(meta["f35_mm"], w_s, h_s)
        cues.append(ScaleCue("metadata_35mm", "other", f_meta, FOCAL_META_SIGMA))
        calib["metadata_f"] = f_meta
    f, sig_abs, rep = fuse_scale(cues)
    sig = max(sig_abs / f, params.focal_sigma_floor)
    calib.update(source="+".join(c["name"] for c in rep["cues"] if c["used"]), f=float(f), sigma_rel=float(sig),
                 fusion=rep, names=names)
    cache.write_text(json.dumps(calib, indent=1, default=float))
    return float(f), calib


def _cached_dpvo(video: Path, work: Path, rot: int, f_full: float, up_size, params: VideoParams, log) -> dict:
    """DPVO, reused when the cached run used the same rotation, focal, stride, resolution and loop-closure flag."""
    import json
    key = dict(rot=int(rot), f=round(float(f_full), 3), size=list(up_size), stride=params.dpvo_stride,
               height=params.dpvo_height, loop=bool(params.dpvo_loop_closure))
    meta = work / "dpvo_meta.json"
    if (work / "dpvo.npz").exists() and meta.exists() and json.loads(meta.read_text()) == key:
        from floorplan.video.vo import load_dpvo
        log("[vo] reusing cached DPVO run")
        return load_dpvo(work / "dpvo.npz")
    vo = run_dpvo(video, work / "dpvo.npz", rot, f_full, up_size, params, log)
    meta.write_text(json.dumps(key))
    return vo


def _cached_geocalib(files: list[Path], work: Path, params: VideoParams) -> dict:
    names = np.array([f.name for f in files])
    path = work / "geocalib.npz"
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["names"], names) and int(z["every"]) == params.gravity_every:
            return {k: z[k] for k in ("index", "up_cam", "pitch_deg", "focal_px", "focal_sigma_px")}
    geo = gravity_and_focal(files, params.gravity_every, seed=params.seed)
    np.savez(path, names=names, every=params.gravity_every, **geo)
    return geo


def _raw_points(cap: VideoCapture, T_wc: np.ndarray, stride: int, voxel: float) -> dict:
    """Back-projected predicted depth in world coordinates, de-duplicated on a grid (closest range kept)."""
    v, u = np.mgrid[0:cap.depth_maps.shape[1]:stride, 0:cap.depth_maps.shape[2]:stride]
    K = cap.K_d
    P_all, R_all, D_all = [], [], []
    for i in range(cap.n):
        z = cap.depth_maps[i][v, u]
        ok = z > 0
        pc = np.stack([(u[ok] - K[0, 2]) * z[ok] / K[0, 0], (v[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
        pw = pc @ T_wc[i, :3, :3].T + T_wc[i, :3, 3]
        ray = pw - T_wc[i, :3, 3]
        P_all.append(pw)
        R_all.append(np.linalg.norm(pc, axis=1))
        D_all.append(ray / np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-9))
    P, R, D = np.concatenate(P_all), np.concatenate(R_all), np.concatenate(D_all)
    k = np.floor((P - P.min(0)) / voxel).astype(np.int64)
    key = k[:, 0] + (k[:, 1] << 21) + (k[:, 2] << 42)
    order = np.argsort(R, kind="stable")
    _, first = np.unique(key[order], return_index=True)
    sel = order[first]
    return dict(points=P[sel].astype(np.float32), range=R[sel].astype(np.float32), ray=D[sel].astype(np.float16))


def _cached_depth(path: Path, files: list[Path], K: np.ndarray, model: str) -> np.ndarray:
    """Predict metric depth, or reuse a previous run's prediction for the same frames, focal and model (the
    slowest GPU step; re-running the CPU stages after a parameter change should not repeat it)."""
    names = np.array([f.name for f in files])
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["names"], names) and np.allclose(z["K"], K):
            return z["depth"]
    D = depth_mod.predict_depth(files, K[0, 0], model)
    np.savez_compressed(path, depth=D.astype(np.float16), names=names, K=K)
    return D


def _export_cached(video: Path, kf: np.ndarray, rot: int, out_dir: Path, tag: str, side: int) -> list[Path]:
    """export_keyframes, skipped when every keyframe image already exists (long videos take minutes to decode)."""
    files = [out_dir / tag / f"{i:06d}.jpg" for i in kf]
    if all(f.exists() for f in files):
        return files
    return export_keyframes(video, kf, rot, out_dir, {tag: side})[tag]


def _rerun_vo_per_segment(video, work, rot, f_full, up_size, vo, kf, seg, params, log) -> tuple[dict, list[int]]:
    """v2 (V2-1): a fresh DPVO run from the start of every segment after a VO restart.

    After DPVO loses track its internal state (patch graph, depth estimates) stays corrupted for a while, so the
    rest of that run can be wrong in SHAPE, not only in scale (floor_only v1: last segment Sim3 ATE 1.4 m). A fresh
    run initialises cleanly. Each fresh run has its own scale (estimated afterwards, cut forced at its start) and
    its own world frame: it is rotated onto the original run's orientations (chordal mean over the segment: DPVO's
    rotation survives restarts far better than its scale, v1 median rotation error 3.8 deg on floor_only) and
    translated to start where the original run was at that frame."""
    import json
    from floorplan.video.evaluate import umeyama  # noqa: F401  (kept for symmetry with the evaluation)
    from scipy.spatial.transform import Rotation
    T_out, frames = vo["T_wc"].copy(), vo["frames"]
    starts = [int(kf[np.argmax(seg == g)]) for g in np.unique(seg)][1:]
    bounds = starts + [int(frames[-1]) + 1]
    runs = []
    # D-061 (live-run time): re-run only segments long enough to matter, longest first, at most vo_rerun_max. Each
    # re-run reloads DPVO; on fragmented captures they cost more than everything else (real with_ceiling: 12
    # segments, 715 s of 1735 s; sim 1 BHK: 28 segments, 1570 s of 3435 s).
    n_kf = {a: int(np.sum((kf >= a) & (kf < b))) for a, b in zip(bounds[:-1], bounds[1:])}
    keep = sorted((a for a in n_kf if n_kf[a] >= params.vo_rerun_min_keyframes), key=lambda a: -n_kf[a])
    keep = set(keep[:params.vo_rerun_max])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a not in keep:
            runs.append(dict(start_frame=a, end_frame=b, skipped=f"{n_kf[a]} keyframes (min "
                                                                 f"{params.vo_rerun_min_keyframes}) or over the cap"))
            continue
        a0 = a - a % params.dpvo_stride
        out = work / f"dpvo_from{a0}_to{b}.npz"
        key = dict(rot=int(rot), f=round(float(f_full), 3), start=a0, end=b)
        meta = out.with_suffix(".json")
        if out.exists() and meta.exists() and json.loads(meta.read_text()) == key:
            from floorplan.video.vo import load_dpvo
            r = load_dpvo(out)
        else:
            r = None
            for attempt in range(3):            # the GPU is shared: wait and retry on a failure (OOM)
                try:
                    r = run_dpvo(video, out, rot, f_full, up_size, params, log, start=a0, end=b)
                    break
                except RuntimeError as e:
                    log(f"[vo] fresh DPVO run {a0}-{b} failed (attempt {attempt + 1}): {str(e)[-200:]}")
                    time.sleep(30)
            if r is None:                       # keep the first run's poses for this segment, and say so
                runs.append(dict(start_frame=a0, end_frame=b, failed=True))
                continue
            meta.write_text(json.dumps(key))
        sel = np.isin(frames, r["frames"])
        Tn = r["T_wc"][np.searchsorted(r["frames"], frames[sel])]
        To = vo["T_wc"][sel]
        M = np.einsum("nij,nkj->ik", To[:, :3, :3], Tn[:, :3, :3])      # chordal mean of R_old R_new^T
        U, _, Vt = np.linalg.svd(M)
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        Tn = Tn.copy()
        Tn[:, :3, :3] = np.einsum("ij,njk->nik", R, Tn[:, :3, :3])
        Tn[:, :3, 3] = Tn[:, :3, 3] @ R.T
        Tn[:, :3, 3] += To[0, :3, 3] - Tn[0, :3, 3]
        T_out[sel] = Tn
        rot_dev = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", To[:, :3, :3], Tn[:, :3, :3]) - 1) / 2,
                                               -1, 1)))
        runs.append(dict(start_frame=a0, end_frame=b, frames=int(sel.sum()), runtime_s=r["runtime_s"],
                         median_rot_dev_from_first_run_deg=float(np.median(rot_dev))))
        log(f"[vo] fresh DPVO run frames {a0}-{b}: median rotation deviation from the first run "
            f"{runs[-1]['median_rot_dev_from_first_run_deg']:.1f} deg")
    forced = [int(np.searchsorted(kf, a)) for a in starts]
    return dict(vo, T_wc=T_out, reruns=runs), forced


def segment_quality(D: np.ndarray, K: np.ndarray, T_m: np.ndarray, seg: np.ndarray, s_local: np.ndarray,
                    params: VideoParams, votes: dict | None = None, exclude: np.ndarray | None = None) -> dict:
    """Self-check of each segment's camera path, without ground truth (v2, V2-5; D-076 for the PnP scale).

    Symptoms of a VO run that is wrong in SHAPE (not just in scale), or of a scale that is not measured:
      * residual (both methods): under the metric poses, neighbouring keyframes' depth maps disagree (median truncated
        |log depth ratio| of pairs 1-2 keyframes apart; 0.040-0.041 on good segments, 0.063 on a broken one);
      * scale_method "depth_agreement", spread: the scale has to change a lot inside the segment (local scale
        p90/p10); a healthy DPVO segment drifts slowly (measured 1.2-1.35 on good segments, 2.25 = both clamps hit
        on broken ones);
      * scale_method "pnp" (votes = {segment: (coverage, vote residual)}): the PnP scale follows DPVO's real drift,
        so its spread is reported, not judged. The segment needs votes of its own on at least half of its keyframes
        (coverage); elsewhere its scale is interpolated and DPVO's motion is not measured (D-076: every sample
        segment that ARKit puts more than 100% off had coverage <= 0.44). The vote residual is reported only.
    A segment is trusted only if every check passes and it has enough path to measure them. Steep keyframes
    (`exclude`, D-084) are left out of the residual."""
    from floorplan.video.scale import TRUNC, _backproject, _cost
    ex = np.zeros(len(seg), bool) if exclude is None else np.asarray(exclude, bool)
    out = {}
    for g in np.unique(seg):
        ids = np.where(seg == g)[0]
        costs = []
        for i in ids[:-1]:
            for j in (i + 1, i + 2):
                if j <= ids[-1] and not (ex[i] or ex[j]):
                    Tji = np.linalg.inv(T_m[j]) @ T_m[i]
                    c, _ = _cost(_backproject(D[i], K, 8) @ Tji[:3, :3].T + Tji[:3, 3], D[j], K)
                    costs.append(min(c, TRUNC) if np.isfinite(c) else TRUNC)
        v = s_local[ids]
        spread = float(np.percentile(v, 90) / np.percentile(v, 10))
        res = float(np.median(costs)) if costs else float("nan")
        path = float(np.linalg.norm(np.diff(T_m[ids, :3, 3], axis=0), axis=1).sum()) if len(ids) > 1 else 0.0
        reasons = []
        if params.scale_method == "pnp":
            cov = (votes or {}).get(int(g), (0.0, None))[0]
            if cov < params.trust_min_vote_coverage:
                reasons.append(f"vote coverage {cov:.2f} < {params.trust_min_vote_coverage}")
        elif spread > params.trust_max_scale_spread:
            reasons.append(f"local scale spread {spread:.2f} > {params.trust_max_scale_spread}")
        if not res <= params.trust_max_residual:
            reasons.append(f"depth residual {res:.3f} > {params.trust_max_residual}")
        if len(ids) < params.trust_min_keyframes:
            reasons.append(f"only {len(ids)} keyframes")
        out[int(g)] = dict(scale_spread=spread, depth_residual=res, path_m=path, n_keyframes=int(len(ids)),
                           trusted=not reasons, reasons=reasons)
    return out


def _frame_segments(kf: np.ndarray, seg: np.ndarray, n: int) -> np.ndarray:
    """Scale segment of every video frame (the segment of the last keyframe at or before it)."""
    return seg[np.clip(np.searchsorted(kf, np.arange(n), side="right") - 1, 0, len(kf) - 1)]


def _rigid_about(R: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pivot - R @ pivot
    return M


def _level_segments(T_kf: np.ndarray, T_all: np.ndarray, kf: np.ndarray, seg: np.ndarray, ts: np.ndarray,
                    up: np.ndarray, ups: np.ndarray, ups_kf: np.ndarray, params: VideoParams) -> tuple:
    """v2 (V2-3): gravity-level every scale segment with ITS OWN GeoCalib up vectors, and remove impossible camera
    jumps across VO restarts. Returns levelled (+y up) keyframe and frame poses and a report.

    Inside a segment DPVO's rotation is consistent, but a restart can rotate DPVO's world; one global "up" would then
    tilt every later room. The 4-DoF pose graph cannot fix tilt, so it is fixed here, per segment, from gravity."""
    from floorplan.plan.align import _rot_between
    n = len(T_all)
    seg_f = _frame_segments(kf, seg, n)
    R_up = np.eye(4)
    R_up[:3, :3] = _rotation_to_y(up)
    T_kf, T_all = T_kf.copy(), T_all.copy()
    rep = []
    for g in np.unique(seg):
        sel = seg[ups_kf] == g
        ids = np.where(seg == g)[0]
        info = dict(segment=int(g), gravity_frames=int(sel.sum()))
        if sel.sum() >= 5 and len(np.unique(seg)) > 1:
            up_g, spread = _robust_mean_direction(ups[sel])
            tilt = float(np.degrees(np.arccos(np.clip(up_g @ up, -1, 1))))
            info.update(tilt_vs_global_deg=tilt, spread_deg=spread)
            if tilt > 0.5:
                M = _rigid_about(_rot_between(up_g, up), T_kf[ids[0], :3, 3])
                T_kf[ids] = M @ T_kf[ids]
                T_all[seg_f == g] = M @ T_all[seg_f == g]
        rep.append(info)
    T_kf = np.einsum("ij,njk->nik", R_up, T_kf)
    T_all = np.einsum("ij,njk->nik", R_up, T_all)
    # continuity across restarts: the camera cannot move faster than walking speed between two keyframes
    jumps = []
    for k in np.where(np.diff(seg) != 0)[0]:
        step = T_kf[k + 1, :3, 3] - T_kf[k, :3, 3]
        dt = max(float(ts[kf[k + 1]] - ts[kf[k]]), 1e-3)
        jumps.append(dict(keyframe=int(k + 1), step_m=float(np.linalg.norm(step)), dt_s=dt))
        if np.linalg.norm(step) > params.max_walk_speed_mps * dt:
            T_kf[k + 1:, :3, 3] -= step
            T_all[kf[k + 1]:, :3, 3] -= step
            jumps[-1]["reset"] = True
    return T_kf, T_all, R_up, dict(segments=rep, restart_steps=jumps)


def _apply_kf_corrections(C_kf: np.ndarray, kf: np.ndarray, seg: np.ndarray, ts: np.ndarray,
                          T_all: np.ndarray) -> np.ndarray:
    """Blend per-keyframe world corrections onto every frame (slerp/linear in time, within each segment)."""
    from scipy.spatial.transform import Rotation, Slerp
    seg_f = _frame_segments(kf, seg, len(T_all))
    out = T_all.copy()
    for g in np.unique(seg):
        ks = np.where(seg == g)[0]
        fr = np.where(seg_f == g)[0]
        tk, tq = ts[kf[ks]], np.clip(ts[fr], ts[kf[ks[0]]], ts[kf[ks[-1]]])
        C = np.tile(np.eye(4), (len(fr), 1, 1))
        if len(ks) == 1:
            C[:] = C_kf[ks[0]]
        else:
            C[:, :3, :3] = Slerp(tk, Rotation.from_matrix(C_kf[ks, :3, :3]))(tq).as_matrix()
            for a in range(3):
                C[:, a, 3] = np.interp(tq, tk, C_kf[ks, a, 3])
        out[fr] = C @ T_all[fr]
    return out


def floor_heights(D: np.ndarray, K: np.ndarray, T: np.ndarray, stride: int = 4, min_drop: float = 0.5,
                  max_drop: float = 2.2) -> np.ndarray:
    """Height of each keyframe's camera above the floor it sees in its own metric depth map, gravity from the levelled
    pose T (+y up): the lowest strong horizontal level 0.5-2.2 m below the camera (2 cm bins), refined by the median
    of its points. NaN where no such level is seen (camera looking up or ahead, dropped keyframe)."""
    v, u = np.mgrid[0:D.shape[1]:stride, 0:D.shape[2]:stride]
    out = np.full(len(D), np.nan)
    for k in range(len(D)):
        z = D[k][v, u]
        ok = z > 0
        if ok.sum() < 300:
            continue
        X = np.stack([(u[ok] - K[0, 2]) * z[ok] / K[0, 0], (v[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
        y = X @ T[k, 1, :3]                                       # height relative to the camera
        y = y[(y < -min_drop) & (y > -max_drop)]
        if len(y) < 300:
            continue
        h, e = np.histogram(y, np.arange(y.min() - 0.01, y.max() + 0.03, 0.02))
        h = np.convolve(h, np.ones(3) / 3, mode="same")
        strong = np.where(h >= max(150, 0.3 * h.max()))[0]
        if len(strong):
            near = np.abs(y - (e[strong.min()] + 0.01)) < 0.02
            out[k] = -float(np.median(y[near]))
    return out


def level_to_floor(D: np.ndarray, K: np.ndarray, T_lev: np.ndarray, seg: np.ndarray,
                   params: VideoParams) -> tuple[np.ndarray, dict]:
    """D-076: vertical correction of each keyframe so that the floor it sees sits at one height.

    The PnP scale follows DPVO's real drift, so where DPVO's scale is small a glitch in DPVO's step direction becomes
    metres: on take1 (before r2's DPVO run) one step went 1.1 m down at a local scale of 44, and the floor seen after
    t 80 s sat 1.6-2 m below the floor seen before. Each keyframe measures its camera's height above the floor in its
    own metric depth, which does not depend on the path. The floor's height in the scene, path y minus that height,
    should be one value. Its running median inside each segment (+-floor_level_half_kf, >= 3 keyframes, interpolated
    in between) is the path's vertical error there, and it is removed. A keyframe counts only if its camera height is
    within floor_level_band_m of the run's median: a bed or table top seen as the lowest level is 0.5-0.8 m closer.
    Real camera motion moves the path and the measured height together, so it is kept. Returns dy per keyframe."""
    h = floor_heights(D, K, T_lev)
    n = len(T_lev)
    dy = np.zeros(n)
    ok = np.isfinite(h)
    rep = dict(keyframes_with_floor=int(ok.sum()), applied=False)
    if ok.sum() < 3 * params.floor_level_half_kf:
        return dy, rep
    h_ref = float(np.median(h[ok]))
    for _ in range(2):                                            # median, then the median of the accepted ones
        acc = ok & (np.abs(h - h_ref) <= params.floor_level_band_m)
        h_ref = float(np.median(h[acc]))
    f = T_lev[:, 1, 3] - h                                        # the floor's height as seen by each keyframe
    m = np.full(n, np.nan)
    idx = np.arange(n)
    for g in np.unique(seg):
        ids = idx[seg == g]
        for k in ids:
            sel = acc & (seg == g) & (np.abs(idx - k) <= params.floor_level_half_kf)
            if sel.sum() >= 3:
                m[k] = np.median(f[sel])
        meas = ids[np.isfinite(m[ids])]
        if len(meas):
            m[ids] = np.interp(ids, meas, m[meas])
    F = float(np.median(f[acc]))
    dy = np.where(np.isfinite(m), F - m, 0.0)
    f_after = f[acc] + dy[acc]
    rep.update(applied=True, accepted=int(acc.sum()), camera_height_m=h_ref, floor_y=F,
               dy_min_m=float(dy.min()), dy_max_m=float(dy.max()),
               floor_p10_p90_before_m=[float(np.percentile(f[acc], 10)), float(np.percentile(f[acc], 90))],
               floor_p10_p90_after_m=[float(np.percentile(f_after, 10)), float(np.percentile(f_after, 90))])
    return dy, rep


def _sheet_probe(args):
    path, K, sig_f, n_floor = args
    from floorplan.scale.sheet import Intrinsics, find_sheet
    try:
        # n_floor: gravity "up" in this camera's frame -> only quads lying FLAT on a horizontal plane are accepted
        m = find_sheet(cv2.imread(str(path)), Intrinsics(K, sig_f, 0.01 * max(K[0, 2], K[1, 2]) * 2, "video"),
                       floor_normal_cam=n_floor)
    except Exception as e:                                     # a detector crash must never stop the pipeline
        return None, repr(e)
    return m, None


def _sheet_cues(files: list[Path], pitch: np.ndarray, seg: np.ndarray, f_px: float, sig_f: float,
                T_lev: np.ndarray, params: VideoParams, log) -> dict:
    """Search each segment's most downward-looking keyframes for an optional paper sheet (floorplan.scale)."""
    from concurrent.futures import ProcessPoolExecutor
    w, h = cv2.imread(str(files[0])).shape[1::-1]
    K = _intrinsics(f_px, w, h)
    probe = []
    per_seg = int(np.clip(params.sheet_probe_total // max(len(np.unique(seg)), 1), 5, params.sheet_probe_per_segment))
    for g in np.unique(seg):
        ids = np.where(seg == g)[0]
        ids = ids[np.argsort(pitch[ids])][:per_seg]                    # most downward first (D-061: total capped)
        probe += sorted(ids.tolist())
    found = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        args = [(files[k], K, sig_f, T_lev[k, :3, :3].T @ np.array([0.0, 1.0, 0.0])) for k in probe]
        for k, (m, err) in zip(probe, ex.map(_sheet_probe, args)):
            if m is not None:
                found[k] = m
    log(f"[sheet] searched {len(probe)} keyframes, sheet measured in {len(found)}: "
        + ", ".join(f"kf {k} ({files[k].name}) h={m.height_m:.2f} m" for k, m in found.items()))
    return found


def _fuse_segment_scales(found: dict, seg: np.ndarray, cam_h: np.ndarray, sig_learned: dict,
                         params: VideoParams) -> dict:
    """Per-segment scale correction c_g = metric / current units, fusing the learned-depth scale (c = 1, its sigma)
    with sheet camera-height cues (floorplan.scale.fuse_scale). Segments without a sheet borrow the fusion of all
    sheets, with an extra 3% for the room-to-room variation of the depth model's bias (-4.8% vs -7.1% measured)."""
    from floorplan.scale.fuse import ScaleCue, fuse_scale, learned_depth_cue
    rejected = []

    def cue(k, m):
        return ScaleCue(f"sheet_kf{k}", "sheet", m.height_m / cam_h[k], m.sigma_rand_m / m.height_m,
                        float(np.hypot(m.sigma_sys_m / m.height_m, params.sheet_floor_rel_sigma)), "sheet",
                        f"h_sheet={m.height_m:.3f} m, h_recon={cam_h[k]:.3f}, paper={m.paper}")
    good = {}
    for k, m in found.items():
        if not (np.isfinite(cam_h[k]) and cam_h[k] > 0.3):
            continue
        # a sheet that is not on THE floor (a paper on a table, a white box lid) gives a wildly different scale:
        # it must agree with the learned-depth scale of its segment within 3 sigma (combined)
        z = abs(np.log(m.height_m / cam_h[k])) / np.hypot(sig_learned[int(seg[k])], m.sigma_m / m.height_m)
        if z <= 3.0:
            good[k] = m
        else:
            rejected.append(dict(keyframe=int(k), ratio=float(m.height_m / cam_h[k]), z=float(z)))
    out = {}
    all_cues = [cue(k, m) for k, m in good.items()]
    c_glob, s_glob = (1.0, None)
    if all_cues:
        c_glob, s_abs, _ = fuse_scale([learned_depth_cue(1.0, float(np.median(list(sig_learned.values()))))]
                                      + all_cues)
        s_glob = s_abs / c_glob
    for g in np.unique(seg):
        mine = sorted([(m.detection.score if m.detection else 0, k) for k, m in good.items() if seg[k] == g],
                      reverse=True)[: params.sheet_max_cues_per_segment]
        if mine:
            c, s_abs, rep = fuse_scale([learned_depth_cue(1.0, sig_learned[g])] + [cue(k, good[k]) for _, k in mine])
            out[int(g)] = dict(c=float(c), sigma_rel=float(s_abs / c), source="sheet+learned_depth",
                               cues=[dict(name=r["name"], scale=r["scale"], rel_sigma=r["rel_sigma"], used=r["used"],
                                          note=r["note"]) for r in rep["cues"]])
        elif s_glob is not None:
            out[int(g)] = dict(c=float(c_glob), sigma_rel=float(np.sqrt(s_glob ** 2 + sig_learned[g] ** 2
                                                                         - params.scale_model_sigma ** 2 + 0.03 ** 2)),
                               source="other rooms' sheets + learned_depth")
        else:
            out[int(g)] = dict(c=1.0, sigma_rel=float(sig_learned[g]), source="learned_depth only (no sheet seen)")
        out[int(g)]["sheets_rejected_not_on_floor"] = [r for r in rejected if seg[r["keyframe"]] == g]
    return out


def _geometry(D, K_d, files, T_kf_vo, vo, kf, n, ts, sc_local, seg, up, ups, ups_kf, params, log):
    """Metric scale -> per-segment levelling -> pose graph -> TSDF fusion -> alignment (one pass)."""
    T_kf_m = rescale_trajectory(T_kf_vo, sc_local)
    s_frames = np.interp(vo["frames"], kf, sc_local)
    T_vo_m = rescale_trajectory(vo["T_wc"], s_frames)
    T_all = interpolate_poses(vo["frames"], T_vo_m, n)
    T_all[kf] = T_kf_m
    T_lev, T_all_lev, R_up, lev = _level_segments(T_kf_m, T_all, kf, seg, ts, up, ups, ups_kf, params)
    floor_level = params.floor_level if params.floor_level is not None else params.scale_method == "pnp"
    if floor_level:                                        # D-076: one floor height for every keyframe
        dy, lev["floor_level"] = level_to_floor(D, K_d, T_lev, seg, params)
        if lev["floor_level"]["applied"]:
            C_kf = np.tile(np.eye(4), (len(T_lev), 1, 1))
            C_kf[:, 1, 3] = dy
            T_all_lev = _apply_kf_corrections(C_kf, kf, seg, ts, T_all_lev)
            T_lev = T_lev.copy()
            T_lev[:, 1, 3] += dy
            T_all_lev[kf] = T_lev
            r = lev["floor_level"]
            log(f"[floor] {r['accepted']} of {r['keyframes_with_floor']} keyframes see the floor "
                f"{r['camera_height_m']:.2f} m below the camera; path moved {r['dy_min_m']:+.2f}.."
                f"{r['dy_max_m']:+.2f} m vertically; floor p10-p90 {np.ptp(r['floor_p10_p90_before_m']):.2f} -> "
                f"{np.ptp(r['floor_p10_p90_after_m']):.2f} m")
    if params.debug_dir:                                   # graph inputs, to iterate on posegraph.py offline
        np.savez(Path(params.debug_dir) / "graph_inputs.npz", T_lev=T_lev, seg=seg, ts_kf=ts[kf], K_d=K_d,
                 s_local=sc_local)
    graph = None
    if params.use_pose_graph:
        from floorplan.video.posegraph import join_segments
        T_new, graph = join_segments(D, K_d, T_lev, ts[kf], seg, log=log)
        C_kf = np.einsum("nij,njk->nik", T_new, np.linalg.inv(T_lev))
        T_all_lev = _apply_kf_corrections(C_kf, kf, seg, ts, T_all_lev)
        T_all_lev[kf] = T_new
        T_lev = T_new
    cap = VideoCapture(depth_maps=D, K_d=K_d, image_files=files)
    cfg = Config.load(voxel_m=params.voxel_m, trunc_mult=params.trunc_mult, weight_threshold=params.weight_threshold,
                      color_width=params.color_width, max_depth_m=params.depth_max_m, min_depth_m=0.1)
    fused = fuse_tsdf(cap, np.arange(cap.n), T_lev, cfg, with_color=True)
    raw = _raw_points(cap, T_lev, params.raw_stride, params.raw_voxel_m)
    traj_lev = T_all_lev[:, :3, 3]
    floor0 = estimate_floor(fused["points"], fused["normals"], traj_lev[:, 1])
    if floor0 is None:                                  # no floor seen: keep GeoCalib's gravity as is
        R_g, tilt = np.eye(3), 0.0
    else:
        R_g, tilt = gravity_correction(fused["points"], fused["normals"], floor0, max_tilt_deg=10.0)
    Pg, Ng = fused["points"] @ R_g.T, fused["normals"] @ R_g.T
    try:
        T_al, P, N, ainfo = align_scene(Pg, Ng, traj_lev @ R_g.T, cfg)
    except TypeError:                                   # align_scene found no floor: Manhattan yaw only, no levels
        from floorplan.plan.align import manhattan_yaw
        yaw, score = manhattan_yaw(Ng, cfg.manhattan_tol_deg)
        c_, s_ = np.cos(-yaw), np.sin(-yaw)
        T_al = np.eye(4)
        T_al[:3, :3] = np.array([[c_, 0, -s_], [0, 1, 0], [s_, 0, c_]])
        P, N = Pg @ T_al[:3, :3].T, Ng @ T_al[:3, :3].T
        ainfo = dict(floor_tilt_before_deg=0.0, manhattan_yaw_deg=float(np.degrees(yaw)), manhattan_score=score,
                     floor_y=None, ceiling_y=None, ceiling_height_global=None, floor_not_found=True)
        log("[align] WARNING: no floor found in the fused scene; heights are not available")
    R_lev2al = T_al[:3, :3] @ R_g
    T_align = np.eye(4)
    T_align[:3, :3] = R_lev2al @ R_up[:3, :3]           # (metric, un-levelled) world -> aligned frame
    T_wc = np.einsum("ij,njk->nik", np.linalg.inv(R_up), T_all_lev)   # report poses in that world, as v1 did
    return dict(P=P, N=N, colors=fused["colors"], raw=raw, R_lev2al=R_lev2al, T_align=T_align, T_wc=T_wc,
                traj_al=traj_lev @ R_lev2al.T, ainfo=ainfo, T_lev=T_lev, tilt=tilt, levelling=lev, graph=graph)


def build_scene_from_video(capture_dir_or_mp4, params: VideoParams | None = None, work_dir: Path | None = None,
                           log=print) -> tuple[dict, dict]:
    """Run the video tier and return (scene, info) compatible with floorplan.pipeline.scene.save_scene."""
    from floorplan.video.metadata import video_metadata
    params = params or VideoParams()
    t0 = time.time()
    if work_dir and not params.debug_dir:
        params.debug_dir = str(work_dir)
    timings = {}

    def tick(name):
        timings[name] = round(time.time() - t0 - sum(timings.values()), 1)

    video = resolve_video(capture_dir_or_mp4).resolve()      # DPVO opens it from another working directory
    work = (Path(work_dir) if work_dir else video.parent / "_video_tier_work").resolve()   # DPVO runs in its repo
    work.mkdir(parents=True, exist_ok=True)
    meta = video_metadata(video)
    log(f"[meta] codec {meta['codec']}, size {meta['size']}, rotation {meta['rotation_deg']}, "
        f"35mm focal {meta['f35_mm']}, make/model {meta['make']}/{meta['model']}")

    # 1-2. scan + keyframes (snapped to the frames DPVO will see)
    scan = scan_video(video, params.scan_width)
    n = len(scan.timestamps)
    ts = scan.timestamps
    kf = select_keyframes(scan, params.kf_shift_frac, params.kf_max_dt_s, params.kf_sharp_window)
    if params.dpvo_stride <= 0:
        # v2 (V2-I10): DPVO should see ~30 frames per second whatever the recording rate. The sample (60 fps) used
        # stride 2; a 30 fps phone file with stride 2 gave DPVO only 15 fps and it lost track (phone_like runs)
        fps = 1.0 / max(float(np.median(np.diff(ts))), 1e-3)
        params.dpvo_stride = max(1, int(round(fps / params.dpvo_target_fps)))
        log(f"[video] {fps:.1f} fps -> DPVO stride {params.dpvo_stride}")
    kf = np.unique(np.clip(np.round(kf / params.dpvo_stride) * params.dpvo_stride, 0, n - 1).astype(int))
    log(f"[video] {video}: {n} frames, {len(kf)} keyframes")
    tick("scan_keyframes")

    # 3. upright rotation (axis from GeoCalib; the 180-degree ambiguity is resolved in step 6)
    rot, rot_ev = detect_rotation(video, kf, params.orient_probe_frames, scan.rotation_meta, seed=params.seed)
    tick("rotation")

    # 4. export upright keyframes, focal length (metadata + SfM + GeoCalib, with its uncertainty)
    files = _export_cached(video, kf, rot, work / f"frames_r{rot}", "sfm", params.sfm_long_side)
    geo = _cached_geocalib(files, work, params)
    geo_f = float(np.median(geo["focal_px"]))
    f_sfm_res, calib = _focal(files, geo_f, meta, work, params, log)
    sig_f = float(calib.get("sigma_rel", params.focal_sigma_sfm))
    w_s, h_s = cv2.imread(str(files[0])).shape[1::-1]
    up_size = (scan.size[1], scan.size[0]) if rot in (90, 270) else scan.size
    f_full = f_sfm_res * up_size[0] / w_s
    log(f"[calib] focal {f_full:.1f} px at {up_size} ({calib['source']}, 1-sigma {100 * sig_f:.1f}%; "
        f"GeoCalib {geo_f * up_size[0] / w_s:.1f})")
    tick("calibration")

    # 5. visual odometry
    vo = _cached_dpvo(video, work, rot, f_full, up_size, params, log)
    pos = np.searchsorted(vo["frames"], kf)
    T_kf = vo["T_wc"][pos]
    tick("dpvo")

    # 6. gravity: GeoCalib up vectors (camera frame) rotated into the DPVO world; frames beyond +-45 deg pitch
    #    are outside GeoCalib's training range and are ignored
    ok = np.abs(geo["pitch_deg"]) < params.gravity_max_pitch_deg
    ups_kf = geo["index"][ok]
    ups = np.einsum("nij,nj->ni", T_kf[ups_kf, :3, :3], geo["up_cam"][ok])
    up, up_spread = _robust_mean_direction(ups)
    z_axes = T_kf[:, :3, 2]
    pitch = np.degrees(np.arcsin(np.clip(z_axes @ up, -1, 1)))
    flipped = bool(np.median(pitch) > params.flip_min_pitch_deg)
    if flipped:
        # Phones film walkthroughs looking DOWN (measured median pitch -19 to -31 deg on all three captures).
        # A median pitch clearly above the horizon means the frames are upside down: GeoCalib always calls the top
        # of the picture "up". A margin is required: with a poor gravity estimate the median pitch sits near 0 and
        # its sign is noise (Issue 15). Rotate the frames by 180 deg more, flip "up", re-express the camera axes.
        rot = (rot + 180) % 360
        up, ups = -up, -ups
        R180 = camera_axes_after_rotation(180)
        T_kf = T_kf.copy()
        T_kf[:, :3, :3] = T_kf[:, :3, :3] @ R180
        vo["T_wc"][:, :3, :3] = vo["T_wc"][:, :3, :3] @ R180
        files = _export_cached(video, kf, rot, work / f"frames_r{rot}", "sfm", params.sfm_long_side)
        pitch = -pitch
    rot_ev.update(final_rotation=rot, flipped_by_pitch_check=flipped, median_pitch_deg=float(np.median(pitch)))
    log(f"[gravity] up from {ok.sum()} GeoCalib frames (spread {up_spread:.1f} deg); rotation {rot} "
        f"(flip {flipped}); median camera pitch {np.median(pitch):.1f} deg")

    # 6b. D-084: steep keyframes (ceiling looks, straight down) leave the scale votes and the fusion; a look at the end
    #     ends the walk where it starts, one in the middle cuts a scale segment after it (steep.py)
    kf_all, ex, steep_cuts, steep_rep = kf, np.zeros(len(kf), bool), [], dict(applied=False)
    if params.steep_frames:
        from floorplan.video.steep import local_pitch, steep_keyframes
        p_loc = local_pitch(T_kf[:, :3, :3], ups_kf, ups, up, params.steep_up_half_kf)
        st = steep_keyframes(p_loc, params.steep_up_deg, params.steep_down_deg, params.steep_min_kf,
                             params.steep_tail_kf)
        k_end = st["walk_end"]
        ex, steep_cuts = st["exclude"][:k_end], st["cuts"]
        steep_rep = dict(applied=True, stretches=st["stretches"], walk_end_keyframe=k_end, cuts=steep_cuts,
                         excluded_keyframes=int(ex.sum()), keyframes_cut_at_end=int(len(kf) - k_end),
                         frames_cut_at_end=int(n - kf[k_end]) if k_end < len(kf) else 0)
        if k_end < len(kf):
            kf, files, pos, T_kf, pitch = kf[:k_end], files[:k_end], pos[:k_end], T_kf[:k_end], pitch[:k_end]
            ok = ok & (geo["index"] < k_end)
            keep = ups_kf < k_end
            ups_kf, ups = ups_kf[keep], ups[keep]
            up, up_spread = _robust_mean_direction(ups)
        log(f"[steep] looks (keyframes, up/down): {st['stretches']}; walk ends at keyframe {k_end} of {len(kf_all)}"
            f" (t {ts[kf_all[min(k_end, len(kf_all) - 1)]]:.1f} s); {int(ex.sum())} steep keyframes left out of "
            f"votes and fusion; segment cuts after looks {steep_cuts}")
    tick("gravity")

    # 7. metric depth on upright keyframes at <= 504 px
    dfiles = _export_cached(video, kf_all, rot, work / f"depth_frames_r{rot}", "d", params.depth_long_side)
    h_d, w_d = cv2.imread(str(dfiles[0])).shape[:2]
    K_d = _intrinsics(f_full * w_d / up_size[0], w_d, h_d)
    D = _cached_depth(work / f"depth_{params.depth_model}.npz", dfiles, K_d, params.depth_model)[:len(kf)]
    D = np.stack([depth_mod.clean_depth(d, params.depth_max_m, params.depth_edge_rel) for d in D])
    tick("depth")

    # 7b. path_source "sfm" (path_sfm.py): the keyframes' metric path comes from global SfM scaled by MoGe-2 depth, not
    #     from DPVO; the scale step and DPVO's fresh runs are skipped, DPVO only fills short gaps and places models
    #     path_source "mapanything" (path_mapanything.py): the same swap, the path from chained MapAnything windows
    sfm = None
    if params.path_source in ("sfm", "mapanything"):
        if params.path_source == "sfm":
            from floorplan.video.path_sfm import frontend_path
        else:
            from floorplan.video.path_mapanything import frontend_path
        sfm = frontend_path(files, D, T_kf, vo, kf, ts, work, f_sfm_res, geo, ok, ups_kf, flipped, ex, params, log)
        T_kf, vo, ex, ups, up, up_spread, pitch = (sfm[k] for k in ("T_kf", "vo", "exclude", "ups", "up",
                                                                     "up_spread", "pitch"))
        tick(f"path_{params.path_source}")

    # 8. metric scale with drift correction; segments = stretches between VO scale restarts; the local scale comes
    #    from depth agreement (default) or from PnP votes on the keyframe images and their depth (scale_method, D-076)
    scale_args = dict(method=params.scale_method, sigma_kf=params.scale_sigma_kf, jump=params.scale_jump,
                      min_contrast=params.scale_min_contrast, n_boot=params.bootstrap,
                      block=params.bootstrap_block, seed=params.seed, min_step_m=params.scale_vote_min_step_m,
                      max_dir_deg=params.scale_vote_max_dir_deg, max_rot_deg=params.scale_vote_max_rot_deg,
                      half_kf=params.scale_vote_half_kf)
    sc = sfm["sc"] if sfm else estimate_scales(D, K_d, T_kf, files, params.scale_max_gap, **scale_args)
    seg = np.asarray(sc["segment"])
    vo_reruns = []
    if sfm:
        pass
    elif params.vo_rerun_segments and len(sc["segments"]) > 1:
        vo, forced = _rerun_vo_per_segment(video, work, rot, f_full, up_size, vo, kf, seg, params, log)
        vo_reruns = vo["reruns"]
        T_kf = vo["T_wc"][pos]
        uc = geo["up_cam"][ok]
        if flipped:                     # camera axes were turned by 180 deg and GeoCalib's "up" was upside down
            uc = -(uc @ camera_axes_after_rotation(180))
        ups = np.einsum("nij,nj->ni", T_kf[ups_kf, :3, :3], uc)
        up, up_spread = _robust_mean_direction(ups)
        pitch = np.degrees(np.arcsin(np.clip(T_kf[:, :3, 2] @ up, -1, 1)))
        sc = estimate_scales(D, K_d, T_kf, files, params.scale_max_gap,
                             forced_cuts=sorted(set(forced) | set(steep_cuts)), exclude=ex, **scale_args)
        seg = np.asarray(sc["segment"])
        log(f"[scale] after fresh VO runs: {len(sc['segments'])} segment(s), starts "
            f"{[s['keyframes'][0] for s in sc['segments']]}")
    elif ex.any() or steep_cuts:        # D-084: steep keyframes do not vote; a look in the middle cuts a segment
        sc = estimate_scales(D, K_d, T_kf, files, params.scale_max_gap, forced_cuts=steep_cuts, exclude=ex,
                             **scale_args)
        seg = np.asarray(sc["segment"])
        log(f"[scale] without steep keyframes: {len(sc['segments'])} segment(s), starts "
            f"{[s['keyframes'][0] for s in sc['segments']]}")
    sig_learned = {}
    for s in sc["segments"]:
        st = s["sigma_rel_stat"] if s["sigma_rel_stat"] is not None else sc["sigma_rel_stat"]
        s["sigma_rel_learned"] = float(np.sqrt(st ** 2 + params.scale_model_sigma ** 2
                                               + (params.focal_scale_sensitivity * sig_f) ** 2))
        sig_learned[s["id"]] = s["sigma_rel_learned"]
    votes_txt = (f"; {sc['votes']} PnP votes, {sc['measured_keyframes']}/{len(kf)} keyframes measured"
                 if params.scale_method == "pnp" else f" ({params.scale_method})")
    log(f"[scale] {len(sc['segments'])} scale segment(s); local scale {sc['s_local'].min():.3f}-"
        f"{sc['s_local'].max():.3f}; statistical 1-sigma {100 * sc['sigma_rel_stat']:.1f}%{votes_txt}")
    tick("scale")

    # quality self-check per segment; untrusted segments are reported and (by default) left out of the scene
    s_local = sc["s_local"].copy()
    votes = {s_["id"]: (s_.get("vote_coverage", 0.0), s_.get("vote_residual")) for s_ in sc["segments"]}
    quality = segment_quality(D, K_d, rescale_trajectory(T_kf, s_local), seg, s_local, params, votes=votes,
                              exclude=ex)
    for s_ in sc["segments"]:
        s_.update(quality[s_["id"]])
    trusted = np.array([quality[int(g)]["trusted"] for g in seg])

    def _votes_txt(g):
        cov, vres = votes.get(int(g), (0.0, None))
        return "" if params.scale_method != "pnp" else \
            f"vote coverage {cov:.2f}, vote residual {'-' if vres is None else f'{vres:.3f}'}, "
    log(f"[quality] " + "; ".join(f"seg {g}: spread {q['scale_spread']:.2f}, {_votes_txt(g)}residual "
                                  f"{q['depth_residual']:.3f}, {'trusted' if q['trusted'] else 'UNTRUSTED'}"
                                  for g, q in quality.items()))
    D_fuse = D
    if params.drop_untrusted_segments and trusted.any() and not trusted.all():
        D_fuse = D.copy()
        D_fuse[~trusted] = 0.0                      # their geometry never enters the scene (unobserved, not free)
        log(f"[quality] {int((~trusted).sum())} keyframes of untrusted segments left out of the scene")
    if ex.any():                                    # D-084: steep keyframes never enter the scene
        D_fuse = D_fuse.copy() if D_fuse is D else D_fuse
        D_fuse[ex] = 0.0

    # 9-10. levelling per segment, pose graph (joins segments, closes loops), fusion, alignment. An SfM path keeps its
    #     placements across segment boundaries: the walking-speed reset is for DPVO restarts (it would also move later
    #     keyframes that the SfM model already placed)
    import dataclasses
    geo_params = dataclasses.replace(params, max_walk_speed_mps=float("inf")) if sfm else params
    geo_out = _geometry(D_fuse, K_d, files, T_kf, vo, kf, n, ts, s_local, seg, up, ups, ups_kf, geo_params, log)
    tick("graph_fusion_alignment")

    # 11. paper sheet (optional, off by default: no reference object in the protocol, D-067) -> per-segment scale
    #     correction
    no_sheet = "no sheet seen" if params.use_sheet else "sheet cue off, D-067"
    seg_scale = {g: dict(c=1.0, sigma_rel=sig_learned[g], source=f"learned_depth only ({no_sheet})")
                 for g in sig_learned}
    sheets = {}
    if params.use_sheet:
        sheets = _sheet_cues(files, pitch, seg, f_sfm_res, sig_f, geo_out["T_lev"], params, log)
        if sheets and geo_out["ainfo"].get("floor_y") is not None:
            cam_h = geo_out["traj_al"][kf, 1] - geo_out["ainfo"]["floor_y"]
            seg_scale = _fuse_segment_scales(sheets, seg, cam_h, sig_learned, params)
            c = np.array([seg_scale[int(g)]["c"] for g in seg])
            log(f"[sheet] per-segment corrections {[round(v['c'], 4) for v in seg_scale.values()]}")
            if np.max(np.abs(np.log(c))) > 0.002:
                D_fuse = D_fuse * c[:, None, None].astype(D.dtype)
                s_local = s_local * c
                geo_out = _geometry(D_fuse, K_d, files, T_kf, vo, kf, n, ts, s_local, seg, up, ups, ups_kf, geo_params,
                                    log)
        tick("sheet")

    # 12. honest uncertainty and consistency
    for s in sc["segments"]:
        s.update(scale_correction=seg_scale[s["id"]]["c"], sigma_rel=seg_scale[s["id"]]["sigma_rel"],
                 scale_source=seg_scale[s["id"]]["source"], cues=seg_scale[s["id"]].get("cues"))
    n_kf = len(kf)
    kept = [s for s in sc["segments"] if s["trusted"]] or sc["segments"]
    big = [s for s in kept if s["keyframes"][1] - s["keyframes"][0] + 1 >= 0.1 * n_kf] or kept
    sig_tot = float(max(s["sigma_rel"] for s in big))
    status = "ok"
    if not trusted.any():
        # no segment passed the self-check: the numbers cannot be certified; a wide floor keeps them honest
        sig_tot, status = max(sig_tot, params.untrusted_sigma_floor), "no trusted segment"
    elif not trusted.all():
        status = "untrusted segments dropped"
    graph = geo_out["graph"]
    connected = bool(graph["connectivity"]["connected"]) if graph else len(sc["segments"]) == 1
    consistent = (len(sc["segments"]) == 1 or connected) and bool(trusted.all())
    if not consistent:
        why = [] if trusted.all() else [f"{int((~trusted).sum())} keyframes in segments that failed the self-check"]
        if len(sc["segments"]) > 1 and not connected:
            why.append(f"{len(sc['segments']) - 1} VO restart(s) not joined by verified geometry")
        log(f"[scale] WARNING: whole-scene geometry is NOT reliable: {'; '.join(why)}")
    log(f"[scale] scene 1-sigma {100 * sig_tot:.1f}% (worst large segment); whole_scene_consistent={consistent}")

    P, N, raw, R_lev2al = geo_out["P"], geo_out["N"], geo_out["raw"], geo_out["R_lev2al"]
    n_walk = int(kf_all[len(kf)]) if len(kf) < len(kf_all) else n       # D-084: frames after the walk's end go
    T_all, ts = geo_out["T_wc"][:n_walk], ts[:n_walk]
    R = geo_out["T_align"][:3, :3]
    scene = dict(points=P.astype(np.float32), normals=N.astype(np.float32), colors=geo_out["colors"],
                 raw_points=(raw["points"] @ R_lev2al.T).astype(np.float32), raw_range=raw["range"],
                 raw_ray=(raw["ray"].astype(np.float32) @ R_lev2al.T).astype(np.float16),
                 T_align=geo_out["T_align"], traj=(T_all[:, :3, 3] @ R.T).astype(np.float32), kf=kf, T_wc=T_all,
                 timestamps=ts, kf_segment=seg.astype(np.int16), kf_trusted=trusted & ~ex)
    info = dict(geo_out["ainfo"], tier="video", source=str(video), frames=n, keyframes=int(len(kf)),
                surface_points=int(len(P)), raw_points=int(len(raw["points"])),
                floor_tilt_from_geocalib_deg=float(geo_out["tilt"]), gravity_spread_deg=up_spread,
                rotation=rot_ev, video_metadata={k: v for k, v in meta.items() if k != "keys"},
                calibration=dict({k: v for k, v in calib.items() if k not in ("names", "fusion")},
                                 f_full_px=f_full, upright_size=list(up_size), sigma_rel=sig_f),
                scale=dict(segments=sc["segments"], vo_restarts=len(sc["segments"]) - 1,
                           # one rigid map only if every segment is tied to the others by VERIFIED geometry (ICP
                           # bridge or loop kept by the robust pose graph); otherwise rooms in different segments
                           # may be misplaced and downstream stitching must not treat the scene as rigid
                           whole_scene_consistent=consistent,
                           s_local_min=float(s_local.min()), s_local_max=float(s_local.max()),
                           sigma_rel_stat=float(sc["sigma_rel_stat"]), sigma_rel_model=params.scale_model_sigma,
                           sigma_rel_focal=params.focal_scale_sensitivity * sig_f,
                           sigma_rel_total=sig_tot, sigma_rel=sig_tot, ci95_rel_total=1.96 * sig_tot,
                           sheets_found=len(sheets), status=status,
                           trusted_keyframes=int(trusted.sum()),
                           pairs=sc["pairs"], informative_pairs=sc["informative_pairs"], method=params.scale_method,
                           votes=sc.get("votes"), measured_keyframes=sc.get("measured_keyframes")),
                scale_sigma_rel=sig_tot, whole_scene_consistent=consistent,
                levelling=geo_out["levelling"], pose_graph=graph,
                dpvo=dict(frames=int(len(vo["frames"])), runtime_s=vo["runtime_s"], peak_gb=vo["peak_gb"],
                          fresh_runs=vo_reruns),
                steep=dict(steep_rep, walk_frames=n_walk),
                depth_model=params.depth_model, recommended_measurement_cloud="points", timings_s=timings,
                runtime_s=round(time.time() - t0, 1), params=params.to_dict())
    if sfm:
        info[f"path_{params.path_source}"] = sfm["report"]
    log(f"[video] done in {info['runtime_s']} s: {len(P)} surface points, floor {info['floor_y']}, "
        f"ceiling {info['ceiling_y']}")
    return scene, info
