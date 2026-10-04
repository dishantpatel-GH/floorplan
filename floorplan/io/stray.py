"""Loader for Stray Scanner captures (the format of the provided sample data).

Conventions (measured, see docs/DATA_NOTES.md and DECISIONS D-005):
  * odometry.csv pose = T_world_camera (camera-to-world) with OpenCV camera axes (x right, y down, z forward)
  * world frame: gravity-aligned, +y up, origin at the start pose, heading arbitrary
  * depth/NNNNNN.png: uint16 millimetres, 256x192, registered to the RGB camera
  * depth intrinsics = per-frame RGB intrinsics (odometry.csv fx, fy, cx, cy) scaled by 256/1920
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

DEPTH_W, DEPTH_H = 256, 192


@dataclass
class StrayCapture:
    root: Path
    timestamps: np.ndarray            # (N,) seconds (device uptime)
    T_wc: np.ndarray                  # (N, 4, 4) camera-to-world, OpenCV camera axes
    K_rgb: np.ndarray                 # (N, 3, 3) per-frame RGB intrinsics (pixels at rgb_size)
    rgb_size: tuple[int, int]         # (width, height) of rgb.mp4
    depth_files: list[Path]
    conf_files: list[Path]
    imu: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.timestamps)

    @property
    def capture_id(self) -> str:
        return self.root.name

    def K_depth(self, i: int) -> np.ndarray:
        """Depth intrinsics = RGB intrinsics scaled to 256x192 with pixel-centre-correct principal point (D-011).

        Plain scaling c' = c*s puts the principal point half a (big) pixel off; c' = (c + 0.5)*s - 0.5 is exact for
        pixel-centre coordinates. Measured: lower multi-view residual on all three sample captures."""
        s = DEPTH_W / self.rgb_size[0]
        K = self.K_rgb[i].copy()
        K[0, 0] *= s
        K[1, 1] *= s
        K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
        return K

    def confidence(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.conf_files[i]), cv2.IMREAD_UNCHANGED)

    def depth(self, i: int, min_conf: int = 2, min_m: float = 0.2, max_m: float = 4.0) -> np.ndarray:
        """Metric depth (float32 metres) with invalid pixels set to 0 (D-006: high confidence only, 0.2-4 m)."""
        d = cv2.imread(str(self.depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        if min_conf > 0:
            d[self.confidence(i) < min_conf] = 0.0
        d[(d < min_m) | (d > max_m)] = 0.0
        return d

    def iter_rgb(self, indices) -> Iterator[tuple[int, np.ndarray]]:
        """Decode rgb.mp4 sequentially and yield (frame_index, BGR image) for the requested sorted indices."""
        wanted = sorted(set(int(i) for i in indices))
        if not wanted:
            return
        cap = cv2.VideoCapture(str(self.root / "rgb.mp4"))
        k, cur = 0, 0
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


def load_stray(root: str | Path) -> StrayCapture:
    root = Path(root)
    for f in ("odometry.csv", "camera_matrix.csv", "rgb.mp4"):
        if not (root / f).exists():
            raise FileNotFoundError(f"{root} is not a Stray Scanner capture: missing {f}")
    od = pd.read_csv(root / "odometry.csv", skipinitialspace=True)
    od.columns = [c.strip() for c in od.columns]
    n = len(od)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_quat(od[["qx", "qy", "qz", "qw"]].to_numpy(float)).as_matrix()
    T[:, :3, 3] = od[["x", "y", "z"]].to_numpy(float)

    K0 = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    K = np.tile(K0, (n, 1, 1))
    if {"fx", "fy", "cx", "cy"} <= set(od.columns) and od["fx"].notna().all():
        K[:, 0, 0], K[:, 1, 1] = od["fx"].to_numpy(float), od["fy"].to_numpy(float)
        K[:, 0, 2], K[:, 1, 2] = od["cx"].to_numpy(float), od["cy"].to_numpy(float)

    vid = cv2.VideoCapture(str(root / "rgb.mp4"))
    rgb_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    n_video = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.release()

    depth_files = sorted((root / "depth").glob("*.png"))
    conf_files = sorted((root / "confidence").glob("*.png"))
    if not (len(depth_files) == len(conf_files) == n):
        raise ValueError(f"frame count mismatch: odometry {n}, depth {len(depth_files)}, confidence {len(conf_files)}")
    imu = None
    if (root / "imu.csv").exists():
        imu = pd.read_csv(root / "imu.csv", skipinitialspace=True)
        imu.columns = [c.strip() for c in imu.columns]
    meta = dict(n_video_frames=n_video, rgb_size=rgb_size, frames=n,
                duration_s=float(od["timestamp"].iloc[-1] - od["timestamp"].iloc[0]))
    return StrayCapture(root=root, timestamps=od["timestamp"].to_numpy(float), T_wc=T, K_rgb=K, rgb_size=rgb_size,
                        depth_files=depth_files, conf_files=conf_files, imu=imu, meta=meta)
