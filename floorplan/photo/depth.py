"""Per-image metric depth with MoGe-2 (MIT weights), decision P-3.

Why MoGe-2: on the sample apartment the video tier measured MoGe-2's metric scale at a median ratio of 0.95 against
LiDAR (p10-p90 0.82-1.03), better than Depth Anything 3 metric (0.92) and much better than MapAnything's own
images-only scale (0.65-0.94 per photo, measured here, docs/modules/photo_tier.md). MoGe-2 takes the field of view as
an input, so the EXIF focal length directly removes the focal/scale ambiguity of a single image.
"""
from __future__ import annotations

import math

import numpy as np
import torch

MOGE_REPO, MOGE_FILE = "Ruicheng/moge-2-vitl-normal", "model.pt"


class MoGeRunner:
    def __init__(self, device: str = "cuda"):
        from huggingface_hub import hf_hub_download
        from moge.model.v2 import MoGeModel          # MoGe-2 explicitly: the repo default is v3 (not installed)
        from floorplan.photo.recon import wait_for_gpu
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"                           # I-012: no CUDA -> slow but working, never a crash
        else:
            wait_for_gpu(need_gb=2.0)
        self.model = MoGeModel.from_pretrained(hf_hub_download(MOGE_REPO, MOGE_FILE)).to(device).eval()
        self.device = device

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(self, rgb: np.ndarray, fx: float | None) -> dict:
        """rgb: (h,w,3) uint8 at the working resolution; fx in pixels at that resolution (None = let MoGe guess).
        Returns depth (h,w) metres (0 = invalid), normals (h,w,3) in the OpenCV camera frame, fx used."""
        h, w = rgb.shape[:2]
        fov_x = math.degrees(2 * math.atan(w / (2 * fx))) if fx else None
        x = torch.from_numpy(rgb).float().div(255).permute(2, 0, 1).to(self.device)
        with torch.inference_mode():
            r = self.model.infer(x, fov_x=fov_x, use_fp16=self.device == "cuda")
        d = r["depth"].float().cpu().numpy()
        m = r["mask"].cpu().numpy() & np.isfinite(d)
        d[~m] = 0
        n = r["normal"].float().cpu().numpy() if "normal" in r else None
        K = r["intrinsics"].float().cpu().numpy() * np.array([[w], [h], [1]])
        return dict(depth=d.astype(np.float32), normal=n, fx=float(K[0, 0]))


def clean_depth(d: np.ndarray, max_m: float, edge_rel: float) -> np.ndarray:
    """Drop far pixels and 'flying pixels' on depth edges (a relative jump to any 4-neighbour above edge_rel)."""
    d = d.copy()
    d[d > max_m] = 0
    pad = np.pad(d, 1, mode="edge")
    jump = np.zeros_like(d)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        nb = pad[dy:dy + d.shape[0], dx:dx + d.shape[1]]
        jump = np.maximum(jump, np.abs(nb - d) / np.maximum(d, 1e-3))
    d[jump > edge_rel] = 0
    return d
