"""Video tier: a handheld RGB walkthrough video (no depth, poses, IMU or intrinsics) -> metric 3D scene.

Entry point: build_scene_from_video(capture_dir_or_mp4, params) -> (scene, info), in the same format as the LiDAR
tier's floorplan.pipeline.scene.build_scene, so the plan extractors run on either. See docs/modules/video_tier.md.
"""
from floorplan.video.frontend import build_scene_from_video
from floorplan.video.params import VideoParams

__all__ = ["build_scene_from_video", "VideoParams"]
