"""Paired A/B of door stitching (D-081) on ONE photo front-end run: the plan with the stitch and the plan with the
stitch undone, from the same pose graph and room layouts. Two separate runs of the same photos differ (COLMAP's
match verification is random: rooms moved 0.1-0.4 m between two runs of k38), so only a paired comparison shows what
the stitch itself does.

    python scripts/run_capture.py <photos> --tier photo --out <run> --photo-param door_stitch=true ...
    python scripts/ab_door_stitch.py <run>        # writes <run>/ab_off and <run>/ab_on (plan.json, scene_info.json)
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floorplan.export.json_export import save_plan_json  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402
from floorplan.plan.beta import extract_plan  # noqa: E402


def undo(scene: dict, info: dict) -> tuple[dict, dict]:
    """The scene and info as they were before the stitch moved the rooms (anchors, sides, cameras)."""
    scene, info = dict(scene), copy.deepcopy(info)
    u = (info.get("door_stitch") or {}).get("undo") or {}
    for r, v in (u.get("layouts") or {}).items():
        lay = info["room_layouts"][r]
        for k in ("anchor", "sides_plan"):
            if v.get(k) is None:
                lay.pop(k, None)
            else:
                lay[k] = v[k]
    if u.get("T_wc"):
        names = [str(x) for x in scene["cam_names"]]
        T = np.array(scene["T_wc"], float)
        for n, t in u["T_wc"].items():
            T[names.index(n)] = np.asarray(t, float)
        scene["T_wc"] = T
        scene["traj"] = T[:, :3, 3].astype(np.float32)
    return scene, info


def main():
    run = Path(sys.argv[1])
    scene, info = load_scene(run / "scene")
    if not (info.get("door_stitch") or {}).get("undo"):
        raise SystemExit("no door_stitch undo record in this run (run it with --photo-param door_stitch=true)")
    for name, (sc, inf) in (("ab_on", (scene, info)), ("ab_off", undo(scene, info))):
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
