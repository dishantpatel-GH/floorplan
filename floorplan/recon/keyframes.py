"""Keyframe selection (D-008).

Why: the capture runs at 60 Hz, so consecutive frames are nearly identical. Fusing all of them costs time and adds no
information. A frame becomes a keyframe once the camera has moved or turned enough, or enough time has passed. Frames
whose depth is mostly invalid (glass, pointing at the sky through a window, out of range) are skipped.
"""
from __future__ import annotations

import numpy as np


def relative_motion(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """Translation (m) and rotation (deg) between two camera-to-world poses."""
    dT = np.linalg.inv(Ta) @ Tb
    t = float(np.linalg.norm(dT[:3, 3]))
    c = (np.trace(dT[:3, :3]) - 1) / 2
    return t, float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def select_keyframes(cap, cfg, T_wc: np.ndarray | None = None) -> np.ndarray:
    T_wc = cap.T_wc if T_wc is None else T_wc
    keep = []
    last = None
    for i in range(cap.n):
        if last is not None:
            t, r = relative_motion(T_wc[last], T_wc[i])
            dt = cap.timestamps[i] - cap.timestamps[last]
            if t < cfg.kf_trans_m and r < cfg.kf_rot_deg and dt < cfg.kf_max_dt_s:
                continue
        conf = cap.confidence(i)
        if (conf >= cfg.min_conf).mean() < cfg.kf_min_valid_frac:
            continue
        keep.append(i)
        last = i
    return np.asarray(keep, dtype=int)
