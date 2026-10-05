#!/usr/bin/env python
"""Name the rooms of a finished run (Bedroom, Kitchen, Living room, ...) and draw the plan again.

Reads <run>/plan.json, <run>/scene/ and the run's images (photo: the photo work dir's class maps; video: the keyframes
in <run>/work; LiDAR: frames from the capture's rgb.mp4), writes plan.json with a name, type and evidence per room
(schema 1.1) and plan.png / plan.svg / plan.dxf into --out. The rules are in floorplan/plan/room_types.py.

With --truth the names are scored: sim_gt.json (simulator; true type per room) or the tape gt_polygons.json with
--truth-types (hall=living_room,...). Plan rooms are paired with the GT room that covers most of them, under the
scorer's whole-plan fit (benchmark/wall_match.global_alignment, IoU >= 0.3 or no pairs); photo rooms named after a GT
room are paired by name.

Usage: python scripts/name_rooms.py RUN_DIR [--out DIR] [--input CAPTURE] [--gpu-lock outputs/.gpu.lock]
                                   [--truth GT_JSON [--truth-types a=type,...]] [--per-room 60]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorplan.export.json_export import plan_from_json, save_plan_json  # noqa: E402
from floorplan.plan import room_types as RT  # noqa: E402

MIN_FIT_IOU = 0.3          # below this the whole-plan fit says nothing about which GT room a plan room is
SIM_TYPES = {"living room": "living_room", "bedroom": "bedroom", "kitchen": "kitchen", "bathroom": "bathroom",
             "balcony": "balcony", "foyer": "foyer", "passage": "passage", "corridor": "passage"}


def load_scene(run: Path) -> tuple[dict, dict]:
    z = np.load(run / "scene" / "scene.npz")
    scene = {k: z[k] for k in ("traj", "kf", "T_wc", "T_align", "kf_trusted", "kf_segment") if k in z.files}
    info = json.loads((run / "scene" / "scene_info.json").read_text())
    return scene, info


def resolve(p: str | None) -> Path | None:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def truth_types(plan: dict, gt_json: Path, type_map: dict[str, str]) -> dict:
    """{plan room id: dict(gt=GT room, type=true type, cover=share of the plan room inside it)} and the fit."""
    from shapely.geometry import Polygon
    from floorplan.benchmark import wall_match as wm
    from floorplan.benchmark.gt_eval import read_gt_geometry
    doc = json.loads(gt_json.read_text())
    gtype = {r["id"]: (SIM_TYPES.get(str(r.get("type", "")).lower()) or type_map.get(r["id"], "room"))
             for r in doc["rooms"]}
    geom = read_gt_geometry(None, gt_json) or {}
    out, fit = {}, None
    by_name = {r["id"]: r for r in plan["rooms"] if r["label"] in gtype and plan["tier"] == "photo"}   # folder names
    for rid, r in by_name.items():
        out[rid] = dict(gt=r["label"], type=gtype[r["label"]], cover=None, how="label")
    rest = [r for r in plan["rooms"] if r["id"] not in out]
    if rest and geom:
        mirror = wm.plan_mirrored(plan)
        glob = wm.global_alignment(plan, geom, mirror)
        fit = None if glob is None else round(glob["score"], 3)
        if glob is not None and glob["score"] >= MIN_FIT_IOU:
            M = np.diag([1.0, -1.0]) if mirror else np.eye(2)
            gpolys = {g: Polygon(v["poly"]).buffer(0) for g, v in geom.items()}
            for r in rest:
                q = Polygon(np.asarray(r["polygon"], float) @ M.T @ glob["R"].T + glob["t"]).buffer(0)
                if q.area <= 0:
                    continue
                cov = {g: q.intersection(gp).area / q.area for g, gp in gpolys.items()}
                g = max(cov, key=cov.get)
                if cov[g] >= 0.3:
                    out[r["id"]] = dict(gt=g, type=gtype[g], cover=round(cov[g], 2), how=f"overlap (fit IoU {fit})",
                                        also=[h for h, c in cov.items() if h != g and c >= 0.2])
    return dict(pairs=out, fit_iou=fit, gt_types=gtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, help="default: the run folder (plan.json is rewritten)")
    ap.add_argument("--input", type=Path, help="capture (default: run_report.json input)")
    ap.add_argument("--gpu-lock", type=Path, help="run the segmenter under flock on this file")
    ap.add_argument("--per-room", type=int, default=60, help="video/LiDAR: frames with the camera in each room")
    ap.add_argument("--spread", type=int, default=200, help="video/LiDAR: more frames, evenly over the capture")
    ap.add_argument("--vote", choices=["project", "camera"], default="project",
                    help="project: a pixel votes for the room its 3D point is in; camera: a frame for its camera's")
    ap.add_argument("--truth", type=Path, help="sim_gt.json or gt_polygons.json: score the names")
    ap.add_argument("--truth-types", default="", help="GT room -> type for a tape GT, e.g. hall=living_room")
    ap.add_argument("--cache", type=Path, help="video/LiDAR: class maps and frames (default OUT/names)")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    run, out = a.run, a.out or a.run
    out.mkdir(parents=True, exist_ok=True)
    doc = json.loads((run / "plan.json").read_text())
    plan = plan_from_json(doc)
    report = json.loads((run / "run_report.json").read_text()) if (run / "run_report.json").exists() else {}
    capture = a.input or resolve(report.get("input"))
    scene, info = load_scene(run) if plan.tier in ("video", "lidar") else ({}, {})
    if plan.tier == "photo":
        work = capture.parent / f"photo_work__{capture.name}"
        cache = work
    elif plan.tier == "video":
        work, cache = run / "work", a.cache or out / "names"
    else:
        work = cache = a.cache or out / "names"
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)  # noqa: E731
    rep = RT.name_rooms(plan, scene, info, plan.tier, capture, work, cache, a.gpu_lock, a.per_room, a.spread, a.vote,
                        log)
    errors = save_plan_json(plan, out / "plan.json")
    if errors:
        print("INVALID:", *errors[:5], sep="\n  ")
    if not a.no_render:
        from floorplan.export.dxf import export_dxf
        from floorplan.export.render import render_plan
        render_plan(plan, out / "plan")
        export_dxf(plan, out / "plan.dxf")
    result = dict(run=str(run), tier=plan.tier, valid=not errors, names=rep)
    if a.truth:
        tmap = dict(kv.split("=", 1) for kv in a.truth_types.split(",") if "=" in kv)
        named = json.loads((out / "plan.json").read_text())
        tt = truth_types(named, a.truth, tmap)
        rows, ok, ok_img, n = [], 0, 0, 0
        for r in named["rooms"]:
            t = tt["pairs"].get(r["id"])
            ev = r.get("type_evidence") or {}
            said = ev.get("images_say", r["type"])
            row = dict(room=r["id"], label=r["label"], area_m2=round(r["floor_area"]["value"], 2), name=r["name"],
                       type=r["type"], images_say=said, rule=ev.get("rule"), images=ev.get("images"),
                       top=(ev.get("top_classes") or [])[:3],
                       truth=None if t is None else t["type"], gt_room=None if t is None else t["gt"],
                       cover=None if t is None else t.get("cover"), also=None if t is None else t.get("also"))
            if t is not None:
                n += 1
                ok += r["type"] == t["type"]
                ok_img += said == t["type"]
                row["correct"] = r["type"] == t["type"]
            rows.append(row)
        result["truth"] = dict(gt=str(a.truth), fit_iou=tt["fit_iou"], rooms=rows, paired=n, correct=ok,
                               correct_from_images=ok_img, unpaired=len(named["rooms"]) - n)
        print(f"names: {ok}/{n} paired rooms right ({ok_img}/{n} from the images alone); "
              f"{len(named['rooms']) - n} plan room(s) without a GT room; fit IoU {tt['fit_iou']}")
        for row in rows:
            print(f"  {row['room']:4s} {row['area_m2']:6.2f} m2  {row['name']:13s} truth {str(row['truth']):12s} "
                  f"({row['gt_room']}, cover {row['cover']})  images say {row['images_say']:12s} {row['rule']}  "
                  f"{row['top']}")
    (out / "room_names.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
