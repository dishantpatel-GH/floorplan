"""Video decoding and keyframe selection using image motion only.

Why: the video tier gets no poses, so the LiDAR keyframe rule ("5 cm or 5 deg or 0.5 s", D-008) cannot be applied.
We replace it with the same idea in image space: a new keyframe whenever the picture has shifted by a fixed fraction
of its width (phase correlation on tiny frames, which is what both rotation and translation produce on screen) or a
time cap is reached. Fixed-rate sampling would either waste frames when the user stands still or leave gaps without
overlap when they turn quickly, and SfM breaks exactly at those gaps.

Within each trigger we keep the sharpest of the last few frames (variance of the Laplacian), because handheld video
has motion blur when turning and blurred frames give few, badly localised features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# cv2 rotate codes, indexed by the clockwise rotation (deg) that makes a stored frame upright
ROTATE_CODES = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
MIN_RESPONSE = 0.1           # phase-correlation peak below this is unreliable (measured against ARKit, video_tier.md)
MAX_UNRELIABLE_SHIFT = 0.05  # ... and its shift is capped at 5% of the width


@dataclass
class VideoScan:
    path: Path
    size: tuple[int, int]        # (width, height) of the stored frames
    timestamps: np.ndarray       # (N,) seconds from the container's presentation timestamps
    shift: np.ndarray            # (N,) image shift to the previous frame, as a fraction of the frame width
    sharpness: np.ndarray        # (N,) variance of the Laplacian (higher = sharper)
    rotation_meta: int           # rotation the container asks for (deg clockwise); 0 if none


def resolve_video(capture_dir_or_mp4: str | Path) -> Path:
    """Accept either an .mp4/.mov file or a capture folder that contains rgb.mp4."""
    p = Path(capture_dir_or_mp4)
    if p.is_dir():
        p = p / "rgb.mp4"
    if not p.exists():
        raise FileNotFoundError(f"no video at {p}")
    return p


def _rotation_meta(cap: cv2.VideoCapture) -> int:
    """Rotation stored in the container (iPhone Camera-app videos carry it; Stray Scanner's rgb.mp4 does not)."""
    deg = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META))) % 360
    return deg if deg in ROTATE_CODES else 0


def scan_video(path: Path, scan_width: int) -> VideoScan:
    """One decoding pass: timestamps, frame-to-frame shift and sharpness for every frame."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)          # we handle orientation ourselves (see orientation.py)
    rot = _rotation_meta(cap)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    h_small = int(round(scan_width * size[1] / size[0]))
    win = cv2.createHanningWindow((scan_width, h_small), cv2.CV_32F)
    ts, shifts, sharp = [], [], []
    prev = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        g = cv2.cvtColor(cv2.resize(bgr, (2 * scan_width, 2 * h_small), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        g = cv2.resize(g, (scan_width, h_small), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is None:
            shifts.append(0.0)
        else:
            (dx, dy), response = cv2.phaseCorrelate(prev, g, win)
            shift = float(np.hypot(dx, dy)) / scan_width
            # a weak correlation peak (blur, blank wall) can report a random shift; cap it so one bad estimate
            # cannot trigger a burst of keyframes
            shifts.append(min(shift, MAX_UNRELIABLE_SHIFT) if response < MIN_RESPONSE else shift)
        prev = g
    cap.release()
    if not ts:
        raise ValueError(f"could not decode any frame from {path}")
    return VideoScan(path=path, size=size, timestamps=np.asarray(ts), shift=np.asarray(shifts),
                     sharpness=np.asarray(sharp), rotation_meta=rot)


def select_keyframes(scan: VideoScan, shift_frac: float, max_dt_s: float, sharp_window: int) -> np.ndarray:
    """Motion-triggered keyframes; the sharpest frame of a short window before each trigger is kept."""
    cum = np.cumsum(scan.shift)
    n = len(cum)
    keep = [int(np.argmax(scan.sharpness[:sharp_window]))]
    while True:
        k = keep[-1]
        moved = (cum - cum[k] > shift_frac) | (scan.timestamps - scan.timestamps[k] > max_dt_s)
        moved[: k + 1] = False
        if not moved.any():
            break
        j = int(np.argmax(moved))
        lo = max(k + 1, j - sharp_window + 1)
        keep.append(lo + int(np.argmax(scan.sharpness[lo: j + 1])))
    if keep[-1] < n - 1 and scan.timestamps[-1] - scan.timestamps[keep[-1]] > 0.5 * max_dt_s:
        keep.append(n - 1)
    return np.asarray(keep, dtype=int)


def iter_frames(path: Path, indices) -> Iterator[tuple[int, np.ndarray]]:
    """Decode the video sequentially and yield (index, stored BGR frame) for the sorted requested indices."""
    wanted = sorted(set(int(i) for i in indices))
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    cur, k = 0, 0
    while k < len(wanted):
        if cur < wanted[k]:
            if not cap.grab():
                break
            cur += 1
            continue
        ok, bgr = cap.read()
        cur += 1
        if not ok:
            break
        yield wanted[k], bgr
        k += 1
    cap.release()


def upright(bgr: np.ndarray, rot_cw: int) -> np.ndarray:
    code = ROTATE_CODES[rot_cw]
    return bgr if code is None else cv2.rotate(bgr, code)


def resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def export_keyframes(path: Path, kf: np.ndarray, rot_cw: int, out_dir: Path, long_sides: dict[str, int]) -> dict:
    """Write each keyframe upright as JPEG at every requested resolution: {tag: [file paths in kf order]}."""
    files: dict[str, list[Path]] = {tag: [] for tag in long_sides}
    for tag in long_sides:
        (out_dir / tag).mkdir(parents=True, exist_ok=True)
    for i, bgr in iter_frames(path, kf):
        up = upright(bgr, rot_cw)
        for tag, side in long_sides.items():
            f = out_dir / tag / f"{i:06d}.jpg"
            cv2.imwrite(str(f), resize_long(up, side), [cv2.IMWRITE_JPEG_QUALITY, 95])
            files[tag].append(f)
    return files
