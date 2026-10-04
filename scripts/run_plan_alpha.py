"""Run the boundary-first plan extractor on one prepared scene.

    python scripts/run_plan_alpha.py outputs/<capture> [--out outputs/plan_alpha] [--pair outputs/<other capture>]

Writes <out>/<capture>/plan.json, plan_debug.png and metrics.json.
--pair: repeatability self-check against a second capture of the same rooms. The other capture's plan is taken
from <out>/<other>/plan.json (extracted first if missing). The two scenes are registered with
floorplan/eval/register.py (from the reconstructed geometry, never from the plans), the plans are matched with
floorplan/eval/match.py, and floorplan/eval/repeatability.py scores the gate. Also writes a plain rooms-by-area
table so a reader can eyeball the pairing without trusting the matcher.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floorplan.pipeline.scene import load_scene  # noqa: E402
from floorplan.plan.alpha import extract_plan  # noqa: E402
from floorplan.plan.alpha.params import AlphaParams  # noqa: E402
from floorplan.plan.alpha.render import render_plan  # noqa: E402


def metrics(plan) -> dict:
    kinds = {}
    for o in plan.openings:
        kinds[o.kind] = kinds.get(o.kind, 0) + 1
    return dict(
        capture_id=plan.capture_id, rooms=len(plan.rooms), walls=len(plan.walls), openings=kinds,
        walls_measured=sum(w.length.status == "measured" for w in plan.walls),
        rooms_ceiling_observed=sum(r.ceiling_height.value is not None for r in plan.rooms),
        net_area_m2=[plan.footprint_area.value, plan.footprint_area.lo, plan.footprint_area.hi],
        room_overlap_m2=plan.meta["room_overlap_m2"], min_walls_per_room=min((len(r.wall_ids) for r in plan.rooms), default=0),
        rooms_table=[dict(id=r.id, label=r.label, area=round(r.floor_area.value, 3),
                          bbox=[round(r.bbox_dims[0].value, 3), round(r.bbox_dims[1].value, 3)],
                          ceiling=None if r.ceiling_height.value is None else round(r.ceiling_height.value, 4))
                     for r in sorted(plan.rooms, key=lambda r: -r.floor_area.value)],
        timing_s=plan.meta["timing_s"], runtime_s=plan.meta["runtime_s"], warnings=plan.meta["warnings"])


def rooms_by_area(plan: dict) -> list[str]:
    rows = []
    for r in sorted(plan["rooms"], key=lambda r: -r["floor_area"]["value"]):
        L, W = r["bbox_dims"]
        c = r["ceiling_height"]["value"]
        rows.append(f"| {r['id']} | {r['label']} | {r['floor_area']['value']:.2f} | {L['value']:.3f} x {W['value']:.3f} | "
                    f"{'n/o' if c is None else f'{c:.3f}'} |")
    return rows


def pair_check(scene_a: Path, name_a: str, plan_a: dict, scene_b: Path, name_b: str, out_root: Path,
               params: AlphaParams) -> None:
    """Repeatability self-check between two captures of the same rooms (see module docstring)."""
    from floorplan.eval.match import match_plans
    from floorplan.eval.register import register, scene_evidence
    from floorplan.eval.repeatability import repeatability

    path_b = out_root / name_b / "plan.json"
    if not path_b.exists():
        sb, ib = load_scene(scene_b)
        write_outputs(extract_plan(sb, ib, ib["capture_id"], params), sb, ib, out_root / name_b)
    plan_b = json.loads(path_b.read_text())
    sa, ia = load_scene(scene_a)
    sb, ib = load_scene(scene_b)
    reg = register(scene_evidence(sa, ia, name_a), scene_evidence(sb, ib, name_b))
    m = match_plans(plan_a, plan_b, np.array(reg.T2))
    rep = repeatability(plan_a, plan_b, m, name_a, name_b, "lidar")
    out = out_root / "repeatability"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{name_a}__vs__{name_b}"
    head = ["| room | label | area (m2) | bbox L x W (m) | ceiling (m) |", "|---|---|---|---|---|"]
    md = [f"# plan_alpha repeatability self-check: {name_a} vs {name_b}", "",
          f"drift_sigma_m used for intervals: {params.drift_sigma_m}", "",
          f"Scene registration: yaw {reg.yaw_deg:.2f} deg, residual rmse {100 * reg.residual_rmse_m:.1f} cm, "
          f"distinctiveness {reg.distinctiveness:.2f}, floor IoU {reg.overlap_iou:.2f}", "",
          f"## Rooms of {name_a}, by area", *head, *rooms_by_area(plan_a), "",
          f"## Rooms of {name_b}, by area", *head, *rooms_by_area(plan_b), "", rep.to_markdown()]
    (out / f"{stem}.md").write_text("\n".join(md))
    (out / f"{stem}.json").write_text(json.dumps(dict(registration={k: v for k, v in reg.to_dict().items()
                                                                     if k != "yaw_curve"},
                                                      summary=rep.summary(), match=m.to_dict()), indent=1, default=float))
    print(json.dumps(rep.summary(), default=float))


def write_outputs(plan, scene: dict, info: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(dataclasses.asdict(plan), indent=1, default=float))
    m = metrics(plan)
    (out / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    render_plan(plan, scene, info, out / "plan_debug.png")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/plan_alpha"))
    ap.add_argument("--pair", type=Path, default=None, help="second scene of the same rooms (repeatability)")
    ap.add_argument("--name", default=None, help="output folder name (default: the scene folder name)")
    ap.add_argument("--pair-name", default=None, help="output folder name of the --pair scene")
    ap.add_argument("--drift-sigma", type=float, default=None,
                    help="residual drift budget per wall length in m (default AlphaParams; the drift module "
                         "reports one in outputs/drift/<capture>/drift_report.json)")
    a = ap.parse_args()
    params = AlphaParams() if a.drift_sigma is None else dataclasses.replace(AlphaParams(), drift_sigma_m=a.drift_sigma)
    scene, info = load_scene(a.scene_dir)
    t0 = time.time()
    plan = extract_plan(scene, info, info["capture_id"], params)
    out = a.out / (a.name or a.scene_dir.name)
    m = write_outputs(plan, scene, info, out)
    print(json.dumps({k: m[k] for k in ("rooms", "walls", "openings", "walls_measured", "net_area_m2",
                                         "room_overlap_m2", "runtime_s")}, default=float))
    print(f"[plan_alpha] wrote {out} ({time.time() - t0:.1f}s incl. rendering)")
    if a.pair is not None:
        pair_check(a.scene_dir, out.name, json.loads((out / "plan.json").read_text()), a.pair,
                   a.pair_name or a.pair.name, a.out, params)


if __name__ == "__main__":
    main()
