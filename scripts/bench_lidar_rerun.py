#!/usr/bin/env python
"""Benchmark (LiDAR, sample): same capture, same command, run twice -> how different are the plans?

    python scripts/run_capture.py <floor_only> --tier lidar --out outputs/benchmark/lidar/floor_only/drift_on_rerun_nodamage --no-damage
    python scripts/bench_lidar_rerun.py      -> outputs/benchmark/lidar/rerun_floor_only.json

Both runs share the world gauge and T_align agrees to 1e-5, so the plans are compared with the identity transform,
using the judge's compare() unchanged (the repeatability gate applied to a run against itself). It also re-runs the
extractor on each saved scene twice, to tell extractor nondeterminism apart from scene (drift solver) differences.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bench_lidar_eval import plan  # noqa: E402
from judge_common import compare, jsonable  # noqa: E402

from floorplan.pipeline.scene import load_scene  # noqa: E402
from floorplan.plan.beta import extract_plan  # noqa: E402

B = ROOT / "outputs" / "benchmark" / "lidar"
A, Bp = plan("floor_only", "drift_on"), plan("floor_only", "drift_on_rerun_nodamage")
res = compare(A, Bp, np.eye(3), "run1", "run2")
s = res["summary"]
n = s["walls_pass"] + s["walls_fail"] + s["walls_unmatched"]
sa, ia = load_scene(B / "floor_only/drift_on/scene")
sb, ib = load_scene(B / "floor_only/drift_on_rerun_nodamage/scene")
ext = {}
for name, (sc, inf) in {"run1_scene": (sa, ia), "run2_scene": (sb, ib)}.items():
    ext[name] = [round(extract_plan(sc, inf, "x").footprint_area.value, 4) for _ in range(2)]
out = dict(walls_judged=n, walls_pass=s["walls_pass"], walls_fail=s["walls_fail"], walls_unmatched=s["walls_unmatched"],
           pass_rate=s["walls_pass"] / n, wilson=s["wilson_pass"], median_abs_delta_cm=100 * s["wall_delta_median_m"],
           same_topology=s["same_topology"], rooms=[s["rooms_a"], s["rooms_b"], s["rooms_matched"]],
           footprint=[s["footprint_a"], s["footprint_b"]],
           room_areas=[dict(a=r["a"], b=r["b"], area_a=r["area_a"], area_b=r["area_b"]) for r in s["room_bbox"]],
           openings=[(o["a"], o["b"], o["delta"]) for o in s["openings"]],
           scene_diff=dict(surface_points=[len(sa["points"]), len(sb["points"])],
                           raw_points=[len(sa["raw_points"]), len(sb["raw_points"])],
                           traj_max_abs_diff_mm=float(1000 * np.abs(sa["traj"] - sb["traj"]).max()),
                           T_align_max_abs_diff=float(np.abs(sa["T_align"] - sb["T_align"]).max())),
           extractor_footprint_twice_per_scene=ext)
(B / "rerun_floor_only.json").write_text(json.dumps(jsonable(out), indent=1, default=float))
print(json.dumps(jsonable(out), indent=1, default=float))
