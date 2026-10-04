"""Damage regions + concealed-damage flags + scope for one capture (any tier).

Examples
  LiDAR (Stray capture + prebuilt scene + plan):
    python scripts/run_damage.py --capture TakeHome/Dataset/single_room/c00a170fe1 \
        --scene outputs/single_room__c00a170fe1 --plan outputs/plan_alpha/single_room__c00a170fe1/plan.json
  Video tier scene (floorplan.video output dir):
    python scripts/run_damage.py --video-scene outputs/video_tier/single_room__c00a170fe1
  Any front-end that wrote views.json + images/ (photo tier, injected test data):
    python scripts/run_damage.py --views <dir> --scene <scene dir> [--plan plan.json]

Without --plan the newest of outputs/plan_alpha|plan_beta/<scene name>/plan.json is used; without any plan the
module fits its own wall planes from the scene (ids P1..Pn).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floorplan.damage import analyse_damage, to_plan_items, views_for_tier  # noqa: E402
from floorplan.damage.detect import DetectParams  # noqa: E402
from floorplan.damage.project import Views, views_from_stray, views_from_video  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def find_plan(scene_dir: Path) -> Path | None:
    cands = [ROOT / "outputs" / p / scene_dir.name / "plan.json" for p in ("plan_alpha", "plan_beta")]
    cands = [c for c in cands if c.exists()]
    return max(cands, key=lambda c: c.stat().st_mtime) if cands else None


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture", type=Path, help="Stray Scanner capture folder (LiDAR tier)")
    src.add_argument("--video-scene", type=Path, help="video-tier output folder (scene.npz + work/)")
    src.add_argument("--views", type=Path, help="folder with views.json + images/")
    src.add_argument("--input", type=Path, help="v2: the run_capture input (Stray folder, video file/folder, photo "
                                                 "root) with --tier and --scene; uses the same views_for_tier() as "
                                                 "the one-command hook")
    ap.add_argument("--scene", type=Path, help="scene folder (scene.npz); default: outputs/<capture name>")
    ap.add_argument("--plan", type=Path, help="plan.json; 'none' to force the scene fallback")
    ap.add_argument("--out", type=Path, help="output folder; default outputs/damage/<name>")
    ap.add_argument("--res", type=float, default=0.005, help="orthophoto pixel size (m)")
    ap.add_argument("--max-views", type=int, default=160)
    ap.add_argument("--save-views", type=Path, help="also write the views (for inject_damage.py)")
    ap.add_argument("--floor", action="store_true", help="also analyse floors")
    ap.add_argument("--tier", choices=["lidar", "video", "photo"], help="with --input")
    ap.add_argument("--min-conf", type=float, help="override DetectParams.min_confidence (0 = keep every candidate "
                                                   "that passes the rules; used for the precision/recall sweep)")
    ap.add_argument("--param", action="append", default=[], help="v2: DetectParams override key=value (ablations)")
    ap.add_argument("--no-stain-relief", action="store_true",
                    help="v3 ablation: skip the stain-on-the-surface (region relief) check")
    ap.add_argument("--fine-res", type=float, help="override the crack refinement resolution (0 = off, v1 behaviour)")
    a = ap.parse_args(argv)

    if a.capture:
        scene_dir = a.scene or ROOT / "outputs" / f"{a.capture.parent.name}__{a.capture.name}"
    elif a.video_scene:
        scene_dir = a.scene or a.video_scene
    elif a.input:
        scene_dir = a.scene
        if scene_dir is None or a.tier is None:
            ap.error("--input needs --tier and --scene")
    else:
        scene_dir = a.scene
        if scene_dir is None:
            ap.error("--views needs --scene")
    scene, info = load_scene(scene_dir)
    if a.capture:
        views = views_from_stray(a.capture, scene, max_views=a.max_views)
    elif a.video_scene:
        views = views_from_video(a.video_scene, scene, info, max_views=a.max_views)
    elif a.input:
        views = views_for_tier(scene, info, a.input, a.tier, max_views=a.max_views)
    else:
        views = Views.load(a.views)
    if a.save_views:
        views.save(a.save_views)
    plan_path = None if (a.plan and str(a.plan) == "none") else (a.plan or find_plan(Path(scene_dir)))
    plan = json.loads(plan_path.read_text()) if plan_path else None
    name = (a.views.name if a.views else Path(scene_dir).name) + ("" if views.tier == "lidar" else f"__{views.tier}")
    out = a.out or ROOT / "outputs" / "damage" / name
    params = DetectParams()
    if a.min_conf is not None:
        params.min_confidence = a.min_conf
    if a.fine_res is not None:
        params.crack_fine_res = a.fine_res
    for kv in a.param:
        k, v = kv.split("=", 1)
        setattr(params, k, type(getattr(params, k))(float(v)) if not isinstance(getattr(params, k), tuple) else
                tuple(float(x) for x in v.split(",")))
    res = analyse_damage(views, scene, info, plan, out_dir=out, res=a.res, params=params,
                         include_floor=a.floor, stain_relief=not a.no_stain_relief)
    res["meta"].update(plan_path=str(plan_path) if plan_path else None, scene=str(scene_dir))
    (out / "damage.json").write_text(json.dumps(res, indent=1, default=float))
    status = {x["id"]: x["status"] for x in to_plan_items(res)[0] if not x["concealed"]}
    for d in res["damage"]:
        ext = d["area"]
        print(f"  {d['id']} {d['surface_id']} {d['cls']}: area {ext['value']:.4f} m2 [{ext['lo']:.4f}-{ext['hi']:.4f}] "
              f"{d['width']['value']:.2f} x {d['height']['value']:.2f} m at t={d['center_t']['value']:.2f} "
              f"conf {d['confidence']:.2f} [{status.get(d['id'], '?')}]")
    for f in res["concealed_flags"]:
        print(f"  FLAG {f['rule_id']} on {f['surface_id']} ({f['region_id']}), conf {f['confidence']:.2f}")
    for s in res["scope_items"]:
        kept = any(status.get(r) == "confirmed" for r in s["region_ids"] if r) or not [r for r in s["region_ids"] if r]
        print("  SCOPE" if kept else "  (review only, not quoted) SCOPE", s["line"])
    print(f"[damage] wrote {out / 'damage.json'}")
    return res


if __name__ == "__main__":
    main()
