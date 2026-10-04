"""Loader for ARKitScenes "raw" captures, exposed with the same interface as floorplan.io.stray.StrayCapture.

Why this exists: the sample data has no ground truth, so the absolute accuracy of the LiDAR tier can only be checked
on a public dataset that ships a laser scan of the same room. ARKitScenes (Apple, iPad Pro LiDAR + ARKit poses +
a Faro Focus S70 scan) is that dataset. Rather than writing a second pipeline for it, this adapter makes an
ARKitScenes capture look exactly like a Stray Scanner capture, so keyframes, TSDF fusion, raw-point collection and
alignment run UNCHANGED on it. Whatever accuracy we measure is therefore the accuracy of the real pipeline code.

Conventions (verified on scene 47895909, see data/arkitscenes/README.md and docs/modules/arkitscenes_validation.md):
  * lowres_wide.traj rows are `t rx ry rz tx ty tz` = WORLD->CAMERA (axis-angle, metres), OpenCV camera axes.
    We invert them to camera->world, which is what the pipeline expects (StrayCapture.T_wc).
  * ARKitScenes' world is gravity-aligned with +Z up (not ARKit's native +Y up). The pipeline assumes +Y up
    (floorplan/plan/align.py looks for the floor along y), so every pose is pre-multiplied by Z_UP_TO_Y_UP, a proper
    rotation (det +1, so the plan is not mirrored): (x, y, z)_arkitscenes -> (x, z, -y)_pipeline.
  * Poses are 10 Hz but depth is 60 Hz. Two modes (validated in scripts/validate_arkitscenes.py):
      - "nearest" (default): keep only depth frames with a pose within NEAREST_TOL_S (every pose has one within
        0.5 ms). Every pose used is a real ARKit estimate.
      - "interpolate": keep every depth frame inside the trajectory's time span and interpolate its pose
        (slerp on rotation, linear on translation).
  * lowres_depth is uint16 millimetres, 256x192, registered to lowres_wide; intrinsics change per frame (.pincam
    `w h fx fy cx cy`), and the same K applies to RGB and depth. Depth is dense (0 never appears), so the confidence
    map is the only validity signal.
  * lowres_wide has a couple more frames than lowres_depth: frames are indexed by depth, as in the README.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from floorplan.io.stray import DEPTH_H, DEPTH_W

# (x, y, z)_arkitscenes -> (x, z, -y)_pipeline: a -90 deg rotation about x. ARKitScenes up (+z) becomes +y.
Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, -1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
NEAREST_TOL_S = 0.005          # a depth frame "has a pose" if one lies within 5 ms (observed max: 0.5 ms)
POSE_MODES = ("nearest", "interpolate")


@dataclass
class ArkitScenesCapture:
    """Same fields and methods as StrayCapture, so the stage-1 pipeline can consume it unchanged."""
    root: Path
    timestamps: np.ndarray            # (N,) seconds, depth-frame time stamps
    T_wc: np.ndarray                  # (N, 4, 4) camera-to-world, OpenCV camera axes, +y-up world
    K_rgb: np.ndarray                 # (N, 3, 3) per-frame intrinsics (pixels at rgb_size)
    rgb_size: tuple[int, int]         # (width, height) of lowres_wide = depth size
    depth_files: list[Path]
    conf_files: list[Path]
    rgb_files: list[Path]
    imu: None = None                  # ARKitScenes raw ships no IMU stream we use
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.timestamps)

    @property
    def capture_id(self) -> str:
        return self.root.name

    def K_depth(self, i: int) -> np.ndarray:
        s = DEPTH_W / self.rgb_size[0]
        K = self.K_rgb[i].copy()
        K[:2] *= s
        return K

    def confidence(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.conf_files[i]), cv2.IMREAD_UNCHANGED)

    def depth(self, i: int, min_conf: int = 2, min_m: float = 0.2, max_m: float = 4.0) -> np.ndarray:
        """Metric depth (float32 metres) with invalid pixels set to 0; identical filtering to StrayCapture.depth."""
        d = cv2.imread(str(self.depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        if min_conf > 0:
            d[self.confidence(i) < min_conf] = 0.0
        d[(d < min_m) | (d > max_m)] = 0.0
        return d

    def iter_rgb(self, indices) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, BGR image) for the requested indices in sorted order (same contract as Stray)."""
        for i in sorted(set(int(i) for i in indices)):
            bgr = cv2.imread(str(self.rgb_files[i]), cv2.IMREAD_COLOR)
            if bgr is not None:
                yield i, bgr


def read_traj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """lowres_wide.traj -> (timestamps, camera-to-world 4x4 in the ARKitScenes +z-up world)."""
    A = np.loadtxt(path, ndmin=2)
    T_cw = np.tile(np.eye(4), (len(A), 1, 1))            # world -> camera, as stored
    T_cw[:, :3, :3] = Rotation.from_rotvec(A[:, 1:4]).as_matrix()
    T_cw[:, :3, 3] = A[:, 4:7]
    return A[:, 0], np.linalg.inv(T_cw)


def _frame_stamp(path: Path) -> float:
    """`<video_id>_<seconds>.png` -> seconds."""
    return float(path.stem.split("_", 1)[1])


def _poses_nearest(t_pose: np.ndarray, T_pose: np.ndarray, t_frame: np.ndarray):
    """For each pose, the depth frame closest in time (if within NEAREST_TOL_S). Returns (frame_idx, T_wc)."""
    j = np.clip(np.searchsorted(t_frame, t_pose), 1, len(t_frame) - 1)
    j = np.where(np.abs(t_frame[j - 1] - t_pose) <= np.abs(t_frame[j] - t_pose), j - 1, j)
    ok = np.abs(t_frame[j] - t_pose) <= NEAREST_TOL_S
    j, T = j[ok], T_pose[ok]
    _, first = np.unique(j, return_index=True)          # one pose per frame, keep the order of time
    return j[first], T[first]


def _poses_interpolated(t_pose: np.ndarray, T_pose: np.ndarray, t_frame: np.ndarray):
    """Every depth frame inside the trajectory span, pose by slerp (rotation) + lerp (translation)."""
    idx = np.flatnonzero((t_frame >= t_pose[0]) & (t_frame <= t_pose[-1]))
    t = t_frame[idx]
    R = Slerp(t_pose, Rotation.from_matrix(T_pose[:, :3, :3]))(t).as_matrix()
    p = np.stack([np.interp(t, t_pose, T_pose[:, k, 3]) for k in range(3)], 1)
    T = np.tile(np.eye(4), (len(idx), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = R, p
    return idx, T


def _read_pincam(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    w, h, fx, fy, cx, cy = np.loadtxt(path)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), (int(w), int(h))


def load_arkitscenes(scene_dir: str | Path, pose_mode: str = "nearest") -> ArkitScenesCapture:
    """Load an ARKitScenes raw scene folder (e.g. raw/Training/47895909) as a pipeline capture."""
    root = Path(scene_dir)
    if pose_mode not in POSE_MODES:
        raise ValueError(f"pose_mode must be one of {POSE_MODES}, got {pose_mode!r}")
    for sub in ("lowres_depth", "confidence", "lowres_wide", "lowres_wide_intrinsics", "lowres_wide.traj"):
        if not (root / sub).exists():
            raise FileNotFoundError(f"{root} is not an ARKitScenes raw scene: missing {sub}")
    depth_all = sorted((root / "lowres_depth").glob("*.png"), key=_frame_stamp)
    t_frame = np.array([_frame_stamp(p) for p in depth_all])
    t_pose, T_pose = read_traj(root / "lowres_wide.traj")
    pick = _poses_nearest if pose_mode == "nearest" else _poses_interpolated
    idx, T_wc = pick(t_pose, T_pose, t_frame)
    if len(idx) == 0:
        raise ValueError(f"{root}: no depth frame matches a pose")

    depth_files = [depth_all[i] for i in idx]
    names = [p.name for p in depth_files]
    conf_files = [root / "confidence" / n for n in names]
    rgb_files = [root / "lowres_wide" / n for n in names]
    K_list, sizes = zip(*(_read_pincam(root / "lowres_wide_intrinsics" / f"{Path(n).stem}.pincam") for n in names))
    if set(sizes) != {(DEPTH_W, DEPTH_H)}:
        raise ValueError(f"expected {DEPTH_W}x{DEPTH_H} lowres frames, got sizes {sorted(set(sizes))}")
    missing = [p for p in conf_files + rgb_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} confidence/RGB files missing, e.g. {missing[0]}")

    T_wc = Z_UP_TO_Y_UP[None] @ T_wc                   # into the pipeline's +y-up world
    dt = np.abs(t_frame[idx, None] - t_pose[None, :]).min(1)   # time to the nearest real pose, per frame
    meta = dict(pose_mode=pose_mode, frames=len(idx), depth_frames_total=len(depth_all), poses_total=len(t_pose),
                pose_rate_hz=float(1 / np.median(np.diff(t_pose))),
                depth_rate_hz=float(1 / np.median(np.diff(t_frame))),
                duration_s=float(t_frame[idx[-1]] - t_frame[idx[0]]), max_time_to_real_pose_s=float(dt.max()),
                rgb_size=(DEPTH_W, DEPTH_H), world_conversion="ARKitScenes +z up -> pipeline +y up: (x, y, z) -> (x, z, -y)")
    return ArkitScenesCapture(root=root, timestamps=t_frame[idx], T_wc=T_wc, K_rgb=np.stack(K_list),
                              rgb_size=(DEPTH_W, DEPTH_H), depth_files=depth_files, conf_files=conf_files,
                              rgb_files=rgb_files, meta=meta)
