"""Photo tier: 2-8 stills per room folder, no depth, no poses -> one metric scene in the LiDAR scene format.

Entry point: build_scene_from_photos(photo_root, params=None) -> (scene, info). See docs/modules/photo_tier.md.
"""
from floorplan.photo.frontend import build_scene_from_photos

__all__ = ["build_scene_from_photos"]
