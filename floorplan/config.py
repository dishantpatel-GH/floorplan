"""All tunable parameters in one place, each with the decision that justifies it (docs/DECISIONS.md)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    # --- depth filtering (D-006) ---
    min_conf: int = 2                 # ARKit confidence: 2 = high only
    min_depth_m: float = 0.2
    max_depth_m: float = 4.0          # LiDAR noise grows with range; Apple's sensor is specified to ~5 m

    # --- keyframes (D-008) ---
    kf_trans_m: float = 0.05          # new keyframe after 5 cm of motion ...
    kf_rot_deg: float = 5.0           # ... or 5 degrees of rotation ...
    kf_max_dt_s: float = 0.5          # ... or 0.5 s, whichever comes first
    kf_min_valid_frac: float = 0.25   # skip frames where < 25% of depth pixels survive filtering

    # --- TSDF fusion (D-007) ---
    voxel_m: float = 0.02
    trunc_mult: float = 4.0           # truncation = 4 voxels
    weight_threshold: float = 3.0     # a surface voxel must be seen in >= 3 frames
    color_width: int = 512            # colour is fused at reduced resolution (only used for display/damage)

    # --- raw points kept for measurement (D-007: measure on raw points, not on the TSDF surface) ---
    raw_stride: int = 2               # every 2nd depth pixel in u and v
    raw_voxel_m: float = 0.005        # 5 mm de-duplication grid

    # --- alignment (D-009) ---
    max_floor_tilt_deg: float = 2.0   # refine gravity from the floor plane only if it is within 2 deg of ARKit's
    manhattan_tol_deg: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def load(path: str | Path | None = None, **overrides) -> "Config":
        cfg = Config()
        if path:
            for k, v in json.loads(Path(path).read_text()).items():
                setattr(cfg, k, v)
        names = {f.name for f in fields(Config)}
        for k, v in overrides.items():
            if k not in names:
                raise KeyError(f"unknown config key {k}")
            setattr(cfg, k, v)
        return cfg
