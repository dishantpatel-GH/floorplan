#!/usr/bin/env python
"""Re-run the plan step, and every step after it, on a finished run's saved scene with the current code.

    python scripts/replan_run.py <run_dir> --out DIR [--sem NPZ] [--no-semantic-openings] [--no-room-names]
                                 [--door-priors RULES] [--rotate DEG]

The front end (poses, depth, scale) is not re-run: <run>/scene is used as saved. So a replanned run differs from the
original only where the plan code, the openings, the room names or the drawings changed since the run was made.

Same steps and order as run_capture.py after the front end: plan extraction, implausible openings dropped, the
segmenter's doors and windows (video/photo), the LiDAR bias correction, the tier widening, room names, the doors from
priors (D-085), then plan.json,
the technical drawing, the presentation drawing and the DXF.

Video: the run's work/ folder is linked into <out>/work one entry at a time, so the caches the later steps write
(semantic_kf.npz, class maps) land in <out> and the source run is not touched. --sem points the openings step at
class maps segmented before (scripts/add_openings.py --sem). Photo: the photo work folder next to the run's input.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_capture as rc  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402
from floorplan.uncertainty.tier_budget import widen_plan  # noqa: E402


def scale_sigma(info: dict, tier: str) -> tuple[float, str]:
    """The relative scale sigma and its reason, as run_capture.front_end gives them for this tier."""
    if tier == "lidar":
        return 0.0, "LiDAR depth is metric; no scale term"
    default = 0.03 if tier == "video" else 0.05
    rel = float(info.get("scale_sigma_rel", info.get("scale", {}).get("sigma_rel", default)) or default)
    reason = f"{tier} scale from learned metric depth + priors; no reference object needed (D-067)"
    if tier == "video" and info.get("whole_scene_consistent") is False:
        rel = max(rel, rc.VIDEO_INCONSISTENT_SIGMA_REL)
        reason = (f"video: segments not joined by verified geometry (whole_scene_consistent=False); relative "
                  f"sigma floored at {100 * rc.VIDEO_INCONSISTENT_SIGMA_REL:.0f}% (D-034)")
    return rel, reason


def link_work(src: Path, dst: Path) -> None:
    """dst/<entry> -> src/<entry> for every entry of src, so new files are written in dst."""
    dst.mkdir(parents=True, exist_ok=True)
    for e in src.iterdir():
        t = dst / e.name
        if not t.exists() and not t.is_symlink():
            t.symlink_to(e.resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sem", type=Path, help="video: class maps of the keyframes (default <out>/work/semantic_kf.npz)")
    ap.add_argument("--no-semantic-openings", action="store_true")
    ap.add_argument("--no-room-names", action="store_true")
    ap.add_argument("--door-priors", default="d", metavar="RULES",
                    help="doors from priors (D-085; run_capture.py --door-priors): rules a-d, default 'd', 'none' off")
    ap.add_argument("--rotate", type=float, default=0.0, help="presentation drawing only: turn it DEG degrees")
    a = ap.parse_args()
    run, out = a.run.resolve(), a.out.resolve()
    if run == out:
        ap.error("--out must differ from the run folder")
    src = json.loads((run / "run_report.json").read_text())
    tier, inp = src["tier"], Path(src["input"])
    if not inp.is_absolute():
        inp = (ROOT / inp).resolve()
    name = inp.stem if inp.is_file() else f"{inp.parent.name}__{inp.name}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)  # noqa: E731
    report = dict(replanned_from=str(run), input=str(inp), tier=tier, extractor="beta")

    scene, info = load_scene(run / "scene")
    if not (out / "scene").exists():
        (out / "scene").symlink_to(run / "scene")
    rel, reason = scale_sigma(info, tier)
    if tier == "video":
        work = out / "work"
        link_work(run / "work", work)
    elif tier == "photo":
        work = inp.parent / f"photo_work__{inp.name}"
    else:
        work = out / "names"

    from floorplan.plan.beta import extract_plan
    plan = extract_plan(scene, info, name, tier=tier)
    plan.tier = tier
    dropped = rc.drop_implausible_openings(plan)
    if dropped:
        log(f"dropped {len(dropped)} implausible opening(s): {[d['id'] for d in dropped]}")
    if info.get("floor_fallback"):
        plan.meta["floor_fallback"] = info["floor_fallback"]
    if tier in ("video", "photo") and not a.no_semantic_openings:
        from floorplan.openings.semantic import add_semantic_openings
        report["semantic_openings"] = add_semantic_openings(plan, scene, info, tier, work, log=log, sem_npz=a.sem)
    if tier == "lidar":
        from floorplan.uncertainty.bias import LidarBiasConfig, apply_to_plan
        apply_to_plan(plan, LidarBiasConfig.load(enabled=True))
    widen_plan(plan, rel, reason)
    rel_note = src.get("front_end", {}).get("reliability")
    if rel_note:
        plan.meta["reliability"] = rel_note
    if src.get("warnings"):
        plan.meta["warnings"] = list(src["warnings"])
    report["intervals_clipped_at_zero"] = rc.clip_nonnegative(plan)
    log(f"plan: {len(plan.rooms)} rooms, {len(plan.walls)} walls, {len(plan.openings)} openings")

    if not a.no_room_names:
        from floorplan.plan.room_types import name_rooms
        cache = {"photo": work, "video": out / "work", "lidar": out / "names"}[tier]
        rep = name_rooms(plan, scene, info, tier, inp, work, cache, log=log)
        report["room_names"] = {k: v for k, v in rep.items() if k != "rooms"}

    if a.door_priors and a.door_priors != "none":
        from floorplan.openings.priors import add_prior_doors
        report["door_priors"] = add_prior_doors(plan, scene, info, tier, log=log, rules=a.door_priors)

    from floorplan.export.dxf import export_dxf
    from floorplan.export.json_export import save_plan_json
    from floorplan.export.presentation import PresentationStyle, render_presentation
    from floorplan.export.render import render_plan
    report["schema_problems"] = save_plan_json(plan, out / "plan.json")
    render_plan(plan, out / "plan")
    st = render_presentation(plan, out / "plan_presentation", PresentationStyle(rotate_deg=a.rotate))
    report["presentation"] = {"doors": st.doors, "windows": st.windows, "passages": st.passages,
                              "labels_forced": st.labels_forced, "warnings": st.warnings}
    export_dxf(plan, out / "plan.dxf")
    report["runtime_s"] = round(time.time() - t0, 1)
    (out / "run_report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
