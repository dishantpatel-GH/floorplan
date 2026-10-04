"""Option comparison for the video tier (evidence for decisions V-3, V-5 and V-6 in docs/modules/video_tier.md).

Run AFTER scripts/run_video_frontend.py on a Stray Scanner capture: it reuses that run's upright keyframes and scores
each alternative against the ARKit poses / LiDAR depth of the same capture.

  sfm         hloc ALIKED+LightGlue + pycolmap on ALL keyframes: how many images register in one model
  da3_windows Depth Anything 3 (DA3-BASE) any-view poses on 30-keyframe windows (up to scale): Sim3 error
  mapanything MapAnything (Apache weights) images-only, 30-keyframe windows: metric scale error and Sim3 error
  dpvo        DPVO trajectory of the run: Sim3 error per window, i.e. how much the monocular scale drifts
  depth       MoGe-2 vs DA3METRIC-LARGE single-image metric depth vs LiDAR: per-frame scale ratio and shape error

Usage: python -m floorplan.video.experiments <capture_dir> <video_tier_out_dir> [--only sfm,depth,...]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from floorplan.pipeline.scene import load_scene
from floorplan.video.depth import predict_depth
from floorplan.video.evaluate import arkit_reference, trajectory_metrics
from floorplan.video.params import VideoParams
from floorplan.video.sfm import run_sfm

WINDOW = 30


def _windows(n: int) -> list[slice]:
    return [slice(a, min(a + WINDOW, n)) for a in range(0, n - WINDOW // 2, WINDOW)]


def _summ(m: dict) -> dict:
    return dict(sim3_ate_m=m["sim3"]["ate_rmse_m"], se3_ate_m=m["se3"]["ate_rmse_m"],
                scale_error_pct=m["scale_error_pct"], rot_err_deg=m["rot_err_median_deg"], path_m=m["path_length_m"])


def exp_sfm(files, ref_T, work: Path, f_prior: float) -> dict:
    res = run_sfm(files[0].parent, [f.name for f in files], work / "exp_sfm", f_prior, VideoParams())
    reg = res.registered
    m = trajectory_metrics(res.T_wc[reg], ref_T[reg]) if reg.sum() >= 3 else None
    return dict(registered=int(reg.sum()), images=len(files), largest_model=_summ(m) if m else None,
                focal_px_sfm_res=float(res.K[0, 0]))


def exp_da3(files, ref_T) -> list[dict]:
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE",
                                           revision="f4a6c9b3c95e41c82048423d3493a81ec3fa810e").to("cuda").eval()
    out = []
    for w in _windows(len(files)):
        p = model.inference([str(f) for f in files[w]], process_res=504)
        E = np.tile(np.eye(4), (len(p.extrinsics), 1, 1))
        E[:, :3] = p.extrinsics
        out.append(dict(start=w.start, **_summ(trajectory_metrics(np.linalg.inv(E), ref_T[w]))))
    del model
    torch.cuda.empty_cache()
    return out


def _to_f32(o):
    if torch.is_tensor(o):
        return o.float() if o.dtype == torch.bfloat16 else o
    if isinstance(o, (list, tuple)):
        return type(o)(_to_f32(x) for x in o)
    if isinstance(o, dict):
        return {k: _to_f32(v) for k, v in o.items()}
    if dataclasses.is_dataclass(o):
        for f in dataclasses.fields(o):
            setattr(o, f.name, _to_f32(getattr(o, f.name)))
    return o


def _bf16_backbone(model) -> None:
    """Encoder + multi-view transformer in bf16 (4.9 -> 2.6 GB of weights), outputs cast back to fp32
    (same recipe as the environment smoke test; casting the whole model fails with a dtype mismatch)."""
    for m in (model.encoder, model.info_sharing):
        m.to(torch.bfloat16)
        m.register_forward_hook(lambda mod, inp, out: _to_f32(out))


def exp_mapanything(files, ref_T, true_fx_over_w: float) -> list[dict]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    model = MapAnything.from_pretrained("facebook/map-anything-apache",
                                        revision="00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a").to("cuda").eval()
    _bf16_backbone(model)
    out = []
    for w in _windows(len(files)):
        views = preprocess_inputs([{"img": cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)} for f in files[w]])
        preds = model.infer(views, memory_efficient_inference=True, minibatch_size=1, use_amp=True,
                            amp_dtype="bf16", apply_mask=True, mask_edges=True)
        T = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds])
        fx_w = preds[0]["intrinsics"][0, 0, 0].item() / views[0]["img"].shape[-1]
        out.append(dict(start=w.start, focal_error_pct=100 * (fx_w / true_fx_over_w - 1),
                        **_summ(trajectory_metrics(T, ref_T[w]))))
    del model
    torch.cuda.empty_cache()
    return out


def exp_dpvo(scene, ref_T) -> dict:
    """Scale of the RAW (uncorrected) DPVO path per window: the spread is the monocular scale drift."""
    z = np.load(Path(scene["_work"]) / "dpvo.npz")
    from scipy.spatial.transform import Rotation
    P = z["poses"]
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(P[:, 3:]).as_matrix()
    T[:, :3, 3] = P[:, :3]
    Tk = T[np.searchsorted(z["frames"], scene["kf"])]
    out = [dict(start=w.start, sim3_scale=trajectory_metrics(Tk[w], ref_T[w])["sim3"]["scale"],
                sim3_ate_m=trajectory_metrics(Tk[w], ref_T[w])["sim3"]["ate_rmse_m"]) for w in _windows(len(Tk))]
    whole = _summ(trajectory_metrics(Tk, ref_T))
    return dict(windows=out, whole_uncorrected=whole)


def exp_depth(dfiles, cap, rows, rot: int, f_full: float, up_w: int) -> dict:
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rot)
    w_d = cv2.imread(str(dfiles[0])).shape[1]
    out = {}
    for model in ("moge2", "da3metric"):
        D = predict_depth(dfiles, f_full * w_d / up_w, model)
        ratios, absrel = [], []
        for k, r in enumerate(rows):
            L = cap.depth(r, 2, 0.2, 4.0)
            L = cv2.rotate(L, code) if code is not None else L
            M = cv2.resize(D[k], L.shape[::-1], interpolation=cv2.INTER_NEAREST)
            ok = (L > 0) & (M > 0)
            if ok.sum() < 500:
                continue
            q = np.median(M[ok] / L[ok])
            ratios.append(q)
            absrel.append(np.mean(np.abs(M[ok] / q - L[ok]) / L[ok]))
        ratios = np.asarray(ratios)
        out[model] = dict(frames=len(ratios), median_scale_ratio=float(np.median(ratios)),
                          ratio_p10=float(np.percentile(ratios, 10)), ratio_p90=float(np.percentile(ratios, 90)),
                          absrel_after_per_frame_scale=float(np.median(absrel)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--only", default="sfm,da3_windows,mapanything,dpvo,depth")
    a = ap.parse_args()
    scene, info = load_scene(a.run_dir)
    work = a.run_dir / "work"
    scene["_work"] = str(work)
    rot = info["rotation"]["final_rotation"]
    sub = "frames_flipped" if info["rotation"]["flipped_by_pitch_check"] else "frames"
    files = sorted((work / sub / "sfm").glob("*.jpg"))
    dfiles = sorted((work / "depth_frames" / "d").glob("*.jpg"))
    ref = arkit_reference(a.capture, scene["timestamps"], scene["kf"], rot)
    up_w = info["calibration"]["upright_size"][0]
    true_fx = float(np.median(ref["cap"].K_rgb[:, 0, 0]))
    res = dict(capture=str(a.capture), keyframes=len(files), true_focal_px=true_fx,
               estimated_focal_px=info["calibration"]["f_full_px"],
               focal_error_pct=100 * (info["calibration"]["f_full_px"] / true_fx - 1))
    if "geocalib_f" in info["calibration"]:
        w_s = cv2.imread(str(files[0])).shape[1]
        res["geocalib_focal_error_pct"] = 100 * (info["calibration"]["geocalib_f"] * up_w / w_s / true_fx - 1)
    todo = a.only.split(",")
    out_file = a.run_dir / "experiments.json"
    done = json.loads(out_file.read_text()) if out_file.exists() else {}

    def save(key, value):
        done.update(res)
        done[key] = value
        out_file.write_text(json.dumps(done, indent=2, default=float))
        print(key, json.dumps(value, default=float)[:2000], flush=True)

    if "sfm" in todo:
        save("sfm_all_keyframes", exp_sfm(files, ref["T_wc"], work, true_fx * cv2.imread(str(files[0])).shape[1] / up_w))
    if "dpvo" in todo:
        save("dpvo_raw", exp_dpvo(scene, ref["T_wc"]))
    if "depth" in todo:
        save("depth_models_vs_lidar", exp_depth(dfiles, ref["cap"], ref["rows"], rot, info["calibration"]["f_full_px"],
                                                up_w))
    if "da3_windows" in todo:
        save("da3_base_windows", exp_da3(files, ref["T_wc"]))
    if "mapanything" in todo:
        save("mapanything_windows", exp_mapanything(files, ref["T_wc"], true_fx / up_w))


if __name__ == "__main__":
    main()
