"""Metric scale for the photo and video tiers: cue fusion (learned depth + priors) and a paper-sheet detector (D-013).

The sheet detector is optional and OFF by default (PhotoParams.use_sheet / VideoParams.use_sheet, D-067): no capture
instruction asks for a sheet. The use below is the opt-in path.

Typical use (photo or video tier):
    from floorplan.scale import load_image_with_intrinsics, find_sheet, sheet_cue, camera_height_prior, fuse_scale
    img, intr = load_image_with_intrinsics(path)            # or Intrinsics(K, focal_rel_sigma)
    meas = find_sheet(img, intr, floor_normal_cam=None)     # None when no trustworthy sheet is visible
    cues = [sheet_cue(meas, h_recon=camera_height_in_recon_units)] if meas else []
    cues.append(camera_height_prior(median_camera_height_in_recon_units))
    scale, sigma, report = fuse_scale(cues)                  # metric = scale * reconstruction units
"""
from floorplan.scale.fuse import (ScaleCue, camera_height_prior, ceiling_height_prior, door_height_prior,
                                  fuse_scale, learned_depth_cue, sheet_cue)
from floorplan.scale.sheet import (PAPERS, SIZE_INVARIANT_M, DetectorParams, Intrinsics, SheetDetection,
                                   SheetMeasurement, default_intrinsics, depth_map_scale, detect_sheets, find_sheet,
                                   load_image_with_intrinsics, measure_sheet, observe_image, plane_depth)

__all__ = ["ScaleCue", "camera_height_prior", "ceiling_height_prior", "door_height_prior", "fuse_scale",
           "learned_depth_cue", "sheet_cue", "PAPERS", "SIZE_INVARIANT_M", "DetectorParams", "Intrinsics",
           "SheetDetection", "SheetMeasurement", "default_intrinsics", "depth_map_scale", "detect_sheets",
           "find_sheet", "load_image_with_intrinsics", "measure_sheet", "observe_image", "plane_depth"]
