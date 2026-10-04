"""Metric depth for each keyframe from a single image, with the focal length passed in.

Two Apache/MIT-licensed models are supported and compared on this data (decision V-5 in the module doc):
  * MoGe-2 ViT-L (Ruicheng/moge-2-vitl-normal, MIT): metric point map + mask; takes the horizontal FOV.
  * Depth Anything 3 metric (depth-anything/DA3METRIC-LARGE, Apache-2.0): outputs "canonical" depth that becomes
    metres after multiplying by f_px / 300, where f_px is the focal length at the processed resolution.

Why pass the focal length: monocular depth is ambiguous between "small room seen with a wide lens" and "large room
with a narrow lens". Giving the model the focal length removes that ambiguity; the environment smoke tests showed
MoGe-2's error halving with the true FOV.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

# pinned revisions (scripts/fetch_weights.py downloads exactly these)
MOGE_REPO, MOGE_FILE, MOGE_REV = "Ruicheng/moge-2-vitl-normal", "model.pt", "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"
DA3_REPO, DA3_REV = "depth-anything/DA3METRIC-LARGE", "4010e39f3634a45bc60553321fb49fb760bd594e"


def _read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _moge(files: list[Path], fx: float, batch: int = 4) -> np.ndarray:
    from huggingface_hub import hf_hub_download
    from moge.model.v2 import MoGeModel                  # MoGe-2; the repo default is v3, which we do not use
    model = MoGeModel.from_pretrained(hf_hub_download(MOGE_REPO, MOGE_FILE, revision=MOGE_REV)).cuda().eval()
    out = []
    for k in range(0, len(files), batch):
        rgb = np.stack([_read_rgb(f) for f in files[k:k + batch]])
        W = rgb.shape[2]
        fov_x = math.degrees(2 * math.atan(W / (2 * fx)))
        x = torch.from_numpy(rgb).float().div(255).permute(0, 3, 1, 2).cuda()
        with torch.inference_mode():
            r = model.infer(x, fov_x=fov_x, use_fp16=True)
        d = r["depth"].float().cpu().numpy()
        d[~r["mask"].cpu().numpy() | ~np.isfinite(d)] = 0.0
        out.append(d)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def _da3(files: list[Path], fx: float, batch: int = 8) -> np.ndarray:
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(DA3_REPO, revision=DA3_REV).to("cuda").eval()
    W = _read_rgb(files[0]).shape[1]
    out = []
    for k in range(0, len(files), batch):
        imgs = [_read_rgb(f) for f in files[k:k + batch]]
        p = model.inference(imgs, process_res=max(imgs[0].shape[:2]))
        f_proc = fx * p.depth.shape[2] / W                 # focal at the processed resolution
        d = p.depth * (f_proc / 300.0)
        if d.shape[1:] != imgs[0].shape[:2]:
            d = np.stack([cv2.resize(x, imgs[0].shape[1::-1], interpolation=cv2.INTER_NEAREST) for x in d])
        out.append(d)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def predict_depth(files: list[Path], fx: float, model: str) -> np.ndarray:
    """(N, H, W) metric depth in metres for upright images of equal size; fx is in pixels of those images."""
    if model == "moge2":
        return _moge(files, fx)
    if model == "da3metric":
        return _da3(files, fx)
    raise ValueError(f"unknown depth model {model}")


def clean_depth(d: np.ndarray, max_m: float, edge_rel: float) -> np.ndarray:
    """Zero out far pixels and 'flying pixels' on depth discontinuities, where a model blends fore- and
    background into points floating in mid-air. Same spirit as dropping low-confidence LiDAR (D-006)."""
    d = d.copy()
    pad = np.pad(d, 1, mode="edge")
    jump = np.zeros_like(d)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        jump = np.maximum(jump, np.abs(pad[dy:dy + d.shape[0], dx:dx + d.shape[1]] - d))
    d[(d > max_m) | (jump > edge_rel * np.maximum(d, 1e-6))] = 0.0
    return d
