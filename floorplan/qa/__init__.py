"""Quality assurance for difficult surfaces and capture conditions (mirrors, glass, glossy floors, low light, blur).

Every detector returns `SurfaceFlag`s: what was found, where, the evidence, and the recommended handling, so the plan
extractor, the damage module and the front-ends can act on them consistently. See docs/FAILURE_MODES.md.
"""
from floorplan.qa.surfaces import (  # noqa: F401
    Handling,
    SurfaceFlag,
    FrameQuality,
    frame_quality,
    flag_frames,
    pick_sharpest,
    depth_confidence_stats,
    lowconf_blob_mask,
    floor_reflection_stats,
    below_floor_points,
    VoxelOccupancy,
    occlusion_violations,
    cluster_plan,
    reflection_consistency,
    is_mirror,
    detect_mirrors,
    points_behind_walls,
    classify_wall_gaps,
)
