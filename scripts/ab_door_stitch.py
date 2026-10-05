"""Paired A/B of door stitching (D-081) on ONE photo front-end run: the plan with the stitch and the plan with the
stitch undone, from the same pose graph and room layouts. Two separate runs of the same photos differ (COLMAP's
match verification is random: rooms moved 0.1-0.4 m between two runs of k38), so only a paired comparison shows what
the stitch itself does.

    python scripts/run_capture.py <photos> --tier photo --out <run> --photo-param door_stitch=true ...
    python scripts/ab_door_stitch.py <run>        # writes <run>/ab_off and <run>/ab_on (plan.json, scene_info.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floorplan.export.json_export import save_plan_json  # noqa: E402
from floorplan.photo.door_stitch import undo_stitch  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402
from floorplan.plan.beta import extract_plan  # noqa: E402


def main():
    run = Path(sys.argv[1])
    scene, info = load_scene(run / "scene")
    if not (info.get("door_stitch") or {}).get("undo"):
        raise SystemExit("no door_stitch undo record in this run (run it with --photo-param door_stitch=true)")
    for name, (sc, inf) in (("ab_on", (scene, info)), ("ab_off", undo_stitch(scene, info))):
        out = run / name
        try:
            plan = extract_plan(sc, inf, run.name, tier="photo")
        except Exception as e:  # noqa: BLE001 - one variant's plan step failing must not hide the other's result
            print(f"{name}: plan step failed: {type(e).__name__}: {e}")
            (out / "FAILED").parent.mkdir(parents=True, exist_ok=True)
            (out / "FAILED").write_text(f"{type(e).__name__}: {e}\n")
            continue
        plan.tier = "photo"
        (out / "scene").mkdir(parents=True, exist_ok=True)
        save_plan_json(plan, out / "plan.json")
        (out / "scene" / "scene_info.json").write_text(json.dumps(inf, indent=1, default=float))
        print(f"{name}: {len(plan.rooms)} rooms -> {out}")


if __name__ == "__main__":
    main()
