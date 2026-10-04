"""Run the space-first plan extractor (plan_beta) on a prepared scene.

Usage:
    python scripts/run_plan_beta.py outputs/<capture> [outputs/<capture2> ...] [--out outputs/plan_beta]
    python scripts/run_plan_beta.py outputs/<capA> outputs/<capB> --repeatability

Input: scene directories written by scripts/prepare_scene.py (scene.npz + scene_info.json).
Output per capture: <out>/<capture>/plan.json, debug.png, metrics.json.
With --repeatability (exactly two captures of the same space): <out>/repeatability_<A>_vs_<B>.json, a
self-check that registers B onto A and compares rooms and walls (no ground truth needed).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floorplan.pipeline.scene import load_scene  # noqa: E402
from floorplan.plan.beta import BetaParams, extract_plan_debug  # noqa: E402
from floorplan.plan.beta.render import render_debug, render_openings  # noqa: E402
from floorplan.plan.beta.selfcheck import repeatability  # noqa: E402


def parse_overrides(items: list[str]) -> BetaParams:
    """KEY=VALUE strings -> BetaParams, values parsed with the type of the default."""
    base = BetaParams()
    kw = {}
    for item in items:
        key, _, val = item.partition("=")
        default = getattr(base, key)                       # AttributeError = unknown key: fail loudly
        kw[key] = type(default)(val) if not isinstance(default, tuple) else tuple(float(v) for v in val.split(","))
    return dataclasses.replace(base, **kw)


def max_room_overlap(plan) -> float:
    """Largest pairwise overlap area between room polygons (must be ~0: rooms may not overlap)."""
    polys = [Polygon(r.polygon) for r in plan.rooms]
    return max((a.intersection(b).area for i, a in enumerate(polys) for b in polys[i + 1:]), default=0.0)


def metrics(plan) -> dict:
    measured = [w for w in plan.walls if w.length.status == "measured"]
    return dict(
        capture_id=plan.capture_id, rooms=len(plan.rooms), walls=len(plan.walls),
        walls_measured=len(measured), openings=len(plan.openings),
        openings_by_kind={k: sum(o.kind == k for o in plan.openings) for k in ("door", "passage", "window")},
        mirrors=len(plan.meta["mirrors"]), window_candidates=len(plan.meta["window_candidates"]),
        weak_links=len(plan.meta["weak_links"]), dropped_regions=len(plan.meta["dropped_regions"]),
        max_room_overlap_m2=max_room_overlap(plan), min_walls_per_room=min(len(r.wall_ids) for r in plan.rooms),
        footprint_m2=[plan.footprint_area.value, plan.footprint_area.lo, plan.footprint_area.hi],
        rooms_detail=[dict(id=r.id, label=r.label, area_m2=round(r.floor_area.value, 3),
                           area_ci=[round(r.floor_area.lo, 3), round(r.floor_area.hi, 3)],
                           bbox_m=[round(d.value, 3) for d in r.bbox_dims],
                           ceiling_m=None if r.ceiling_height.value is None else round(r.ceiling_height.value, 3),
                           ceiling_status=r.ceiling_height.status, walls=len(r.wall_ids)) for r in plan.rooms],
        median_wall_ci_half_width_m=float(sorted(w.length.half_width for w in measured)[len(measured) // 2])
        if measured else None,
        runtime_s=plan.meta["runtime_s"], timing_s=plan.meta["timing_s"])


def run_one(scene_dir: Path, out_root: Path, params: BetaParams):
    scene, info = load_scene(scene_dir)
    plan, debug = extract_plan_debug(scene, info, info["capture_id"], params)
    out = out_root / scene_dir.name
    out.mkdir(parents=True, exist_ok=True)
    plan_dict = json.loads(json.dumps(dataclasses.asdict(plan), default=float))
    (out / "plan.json").write_text(json.dumps(plan_dict, indent=1))
    m = metrics(plan)
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    render_debug(plan, debug, out / "debug.png")
    render_openings(plan, scene["raw_points"], float(info["floor_y"]), out / "openings.png")
    print(json.dumps({k: m[k] for k in ("capture_id", "rooms", "walls", "openings", "footprint_m2", "runtime_s")},
                     default=float))
    return scene, info, plan_dict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dirs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("outputs/plan_beta"))
    ap.add_argument("--repeatability", action="store_true", help="compare two captures of the same space")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a BetaParams field (ablations), e.g. --set wall_slope_mode=axis")
    args = ap.parse_args()
    if args.repeatability and len(args.scene_dirs) != 2:
        ap.error("--repeatability needs exactly two scene directories")
    params = parse_overrides(args.set)
    results = [run_one(d, args.out, params) for d in args.scene_dirs]
    if args.repeatability:
        (sa, ia, pa), (sb, ib, pb) = results
        rep = repeatability(sa, ia, pa, sb, ib, pb)
        name = f"repeatability_{args.scene_dirs[0].name}_vs_{args.scene_dirs[1].name}.json"
        (args.out / name).write_text(json.dumps(rep, indent=1, default=float))
        print(json.dumps(dict(registration=rep["registration"], summary=rep["summary"]), default=float))


if __name__ == "__main__":
    main()
