"""plan_alpha: boundary-first (cell complex) floor-plan extractor for the LiDAR tier.

Public entry point: extract_plan(scene, info, capture_id) -> floorplan.model.Plan.
See docs/modules/plan_alpha.md for the method, decisions and results.
"""
from floorplan.plan.alpha.extract import extract_plan

__all__ = ["extract_plan"]
