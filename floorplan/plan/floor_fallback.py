"""Floor level when the front-end could not detect one (D-032).

Why: the video and photo front-ends find the floor from up-facing surfaces. Learned depth can leave too few
reliable floor normals (video with_ceiling: "floor_not_found"), and the plan extractor needs a floor level to
place its wall band. A cold run at the walk-in must never crash, so we estimate the floor, say how, and mark
heights as weaker evidence.

Two cues, strongest first:
  1. points: the lowest dense horizontal layer of reconstructed points that lies well below the cameras
     (normals ignored: they are what failed);
  2. camera height: a handheld capture at chest height puts the floor ~1.4 m (+-0.15 m) below the median camera
     (protocol: phone at chest height).
"""
from __future__ import annotations

import numpy as np

CHEST_HEIGHT_M, CHEST_SIGMA_M = 1.40, 0.15


def estimate_floor(scene: dict, info: dict) -> dict:
    cam_y = float(np.median(scene["traj"][:, 1])) if "traj" in scene and len(scene["traj"]) else None
    pts = scene.get("raw_points")
    if pts is None or not len(pts):
        pts = scene.get("points")
    if cam_y is not None and pts is not None and len(pts):
        y = pts[:, 1]
        below = y[(y < cam_y - 0.6) & (y > cam_y - 2.5)]
        if len(below) > 1000:
            edges = np.arange(below.min(), below.max() + 0.02, 0.02)
            h, e = np.histogram(below, bins=edges)
            h = np.convolve(h, np.ones(3) / 3, mode="same")
            strong = np.where(h >= max(0.02 * h.sum(), 0.25 * h.max()))[0]
            if len(strong):
                floor = float((e[strong.min()] + e[strong.min() + 1]) / 2)
                if 0.8 <= cam_y - floor <= 2.0:
                    return dict(floor_y=floor, source="points: lowest dense layer below the cameras",
                                sigma_m=0.03, camera_height_m=round(cam_y - floor, 3))
    if cam_y is not None:
        return dict(floor_y=cam_y - CHEST_HEIGHT_M, source="camera-height prior (chest height 1.40 m)",
                    sigma_m=CHEST_SIGMA_M, camera_height_m=CHEST_HEIGHT_M)
    return dict(floor_y=None, source="none: no camera path and no points", sigma_m=None)
