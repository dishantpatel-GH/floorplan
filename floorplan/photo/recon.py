"""Per-room metric multi-view reconstruction with MapAnything (Apache-2.0 weights), decision P-2.

Why MapAnything and not SfM for the geometry inside a room:
  * The protocol's stills are deliberately WIDE-baseline (corner to opposite corner, 2-8 per room). Measured on the
    sample apartment: joint SfM (ALIKED + LightGlue + COLMAP) registered only 17 of 46 photos, in 5 fragments, with a
    focal drifting 9% from the truth. Classical SfM needs overlapping, many-view input; these photos are not that.
  * MapAnything is a feed-forward multi-view model trained for exactly this: few views, wide baselines, and it
    outputs METRIC depth, poses and a dense point map per view in one shared frame. Feeding it the EXIF focal length
    (as intrinsics) removes its largest ambiguity.

Crop bookkeeping: MapAnything works at 518 px with sides that are multiples of 14. We centre-crop each photo to that
aspect ratio ourselves, so the model's resize is a pure scaling, and record (offset, scale) to map any full-resolution
pixel (for example a feature match) to the model's point map exactly.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import numpy as np
import torch

MAPANYTHING_REPO = "facebook/map-anything-apache"


@dataclass
class ViewGeom:
    name: str
    pts: np.ndarray        # (h,w,3) points in the ROOM frame (metres, model scale)
    depth: np.ndarray      # (h,w) z-depth in the camera (metres); 0 = invalid
    conf: np.ndarray       # (h,w) model confidence
    T_wc: np.ndarray       # (4,4) camera-to-room (OpenCV axes)
    K: np.ndarray          # (3,3) intrinsics at the model resolution (as predicted)
    rgb: np.ndarray        # (h,w,3) uint8 model-resolution image
    crop: tuple[float, float, float]   # (ox, oy, s): model_px = (fullres_px - (ox, oy)) * s


def model_size(w: int, h: int, long_side: int = 518, patch: int = 14) -> tuple[int, int]:
    """(W, H) at the model resolution: long side 518, short side the nearest multiple of 14."""
    if w >= h:
        return long_side, max(patch, int(round(long_side * h / w / patch)) * patch)
    return max(patch, int(round(long_side * w / h / patch)) * patch), long_side


def crop_for_model(rgb: np.ndarray, long_side: int) -> tuple[np.ndarray, tuple[float, float, float], tuple[int, int]]:
    """Centre-crop to the model aspect ratio and resize; return image, (ox, oy, s) and the model size."""
    import cv2
    h, w = rgb.shape[:2]
    W, H = model_size(w, h, long_side)
    a = W / H
    if w / h > a:                      # too wide: crop columns
        cw, ch = h * a, h
    else:
        cw, ch = w, w / a
    ox, oy = (w - cw) / 2, (h - ch) / 2
    crop = rgb[int(round(oy)):int(round(oy + ch)), int(round(ox)):int(round(ox + cw))]
    s = W / crop.shape[1]
    small = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
    return small, (float(round(ox)), float(round(oy)), float(s)), (W, H)


def _bf16_backbone(model):
    """Encoder + multi-view transformer in bf16 (4.9 -> 2.6 GB of weights); heads stay fp32 (SETUP.md 6.1)."""
    def to_f32(o):
        if torch.is_tensor(o):
            return o.float() if o.dtype == torch.bfloat16 else o
        if isinstance(o, (list, tuple)):
            return type(o)(to_f32(x) for x in o)
        if isinstance(o, dict):
            return {k: to_f32(v) for k, v in o.items()}
        if dataclasses.is_dataclass(o):
            for f in dataclasses.fields(o):
                setattr(o, f.name, to_f32(getattr(o, f.name)))
        return o
    for m in (model.encoder, model.info_sharing):
        m.to(torch.bfloat16)
        m.register_forward_hook(lambda mod, inp, out: to_f32(out))
    return model


def free_gpu_gb() -> float:
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def wait_for_gpu(need_gb: float, timeout_s: float = 1800, poll_s: float = 15, log=print) -> None:
    """The GPU is shared with other jobs: wait (bounded) until enough memory is free instead of failing with OOM."""
    import time
    t0 = time.time()
    while free_gpu_gb() < need_gb and time.time() - t0 < timeout_s:
        log(f"[photo] waiting for GPU: {free_gpu_gb():.1f} GB free, need {need_gb} GB")
        time.sleep(poll_s)


class MapAnythingRunner:
    """Loads the model once; reconstructs one group of views (one room) per call."""

    def __init__(self, device: str = "cuda"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from mapanything.models import MapAnything
        wait_for_gpu(need_gb=3.5)
        # cast on the CPU first, then move: the GPU never holds the 4.9 GB fp32 copy (shared 8 GB GPU)
        self.model = _bf16_backbone(MapAnything.from_pretrained(MAPANYTHING_REPO).eval()).to(device)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()

    def reconstruct(self, names: list[str], rgbs: list[np.ndarray], f_full: list[float | None],
                    long_side: int = 518, depths: list[np.ndarray] | None = None) -> list[ViewGeom]:
        """rgbs: upright full-resolution images; f_full: focal (px, full resolution) or None if unknown.

        depths (optional): per-view metric depth AT THE MODEL RESOLUTION (e.g. MoGe-2 run on the cropped image).
        MapAnything then solves only the multi-view registration and inherits the depth's metric scale, instead
        of guessing scale itself (decision P-2b)."""
        from mapanything.utils.image import preprocess_inputs
        views, crops, smalls = [], [], []
        for rgb, f in zip(rgbs, f_full):
            small, crop, (W, H) = crop_for_model(rgb, long_side)
            v = {"img": small}
            if f is not None:
                ox, oy, s = crop
                h, w = rgb.shape[:2]
                v["intrinsics"] = np.array([[f * s, 0, (w / 2 - ox) * s],     # principal point at the centre
                                            [0, f * s, (h / 2 - oy) * s], [0, 0, 1]], np.float32)
            if depths is not None and f is not None:
                v["depth_z"] = depths[len(views)].astype(np.float32)
            views.append(v); crops.append(crop); smalls.append(small)
        views = preprocess_inputs(views, resize_mode="fixed_size",
                                  size=(int(smalls[0].shape[1]), int(smalls[0].shape[0])))
        for v in views:
            if "depth_z" in v:
                v["is_metric_scale"] = torch.tensor([True])
        with torch.inference_mode():
            preds = self.model.infer(views, memory_efficient_inference=True, minibatch_size=1, use_amp=True,
                                     amp_dtype="bf16", apply_mask=True, mask_edges=True)
        out = []
        for n, p, crop, small in zip(names, preds, crops, smalls):
            m = p["mask"][0, ..., 0].bool().cpu().numpy()
            d = p["depth_z"][0, ..., 0].float().cpu().numpy()
            d[~m] = 0
            out.append(ViewGeom(name=n, pts=p["pts3d"][0].float().cpu().numpy(), depth=d,
                                conf=p["conf"][0].float().cpu().numpy().reshape(d.shape),
                                T_wc=p["camera_poses"][0].float().cpu().numpy(),
                                K=p["intrinsics"][0].float().cpu().numpy(), rgb=small, crop=crop))
        torch.cuda.empty_cache()
        return out
