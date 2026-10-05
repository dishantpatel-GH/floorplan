#!/usr/bin/env python
"""Add the segmenter's doors and windows to the plan of a finished video or photo run, and draw it again.

run_capture.py does this step itself (D-078). This command does it for runs made before it, from the run folder
alone: plan.json, scene/ and the run's work data (video: <run>/work; photo: the photo work folder next to the input,
or --work). The video tier's keyframes are segmented once and cached in the work folder (semantic_kf.npz).

Usage: python scripts/add_openings.py <run_dir> --out DIR [--work DIR] [--sem NPZ]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.export.dxf import export_dxf  # noqa: E402
from floorplan.export.json_export import load_plan_json, save_plan_json  # noqa: E402
from floorplan.export.render import render_plan  # noqa: E402
from floorplan.openings.semantic import add_semantic_openings  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402
from floorplan.uncertainty.tier_budget import _widen  # noqa: E402


def photo_work_dir(inp: Path) -> Path:
    """Where build_scene_from_photos keeps its cache when run_capture gives it no work dir."""
    return inp.parent / f"photo_work__{inp.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, help="default: <run>/work (video) or the photo work folder of the input")
    ap.add_argument("--sem", type=Path, help="video: class maps of the keyframes (default <work>/semantic_kf.npz)")
    a = ap.parse_args()
    plan = load_plan_json(a.run / "plan.json")
    scene, info = load_scene(a.run / "scene")
    tier = plan.tier or info.get("tier")
    report = json.loads((a.run / "run_report.json").read_text()) if (a.run / "run_report.json").exists() else {}
    work = a.work or (a.run / "work" if tier == "video" else photo_work_dir(Path(report.get("input", ""))))
    before = {o.id for o in plan.openings}
    rep = add_semantic_openings(plan, scene, info, tier, work, sem_npz=a.sem)
    # the plan was widened by the tier scale term in its run; the new widths get the same term once
    rel = float((plan.meta.get("uncertainty") or {}).get("tier_scale_sigma_rel") or 0.0)
    for o in plan.openings:
        if o.id not in before and rel > 0:
            for m in (o.width, o.height, o.sill_height):
                if m is not None:
                    _widen(m, rel)
    a.out.mkdir(parents=True, exist_ok=True)
    problems = save_plan_json(plan, a.out / "plan.json")
    render_plan(plan, a.out / "plan")
    export_dxf(plan, a.out / "plan.dxf")
    rep["schema_problems"] = problems
    rep["source_run"] = str(a.run)
    (a.out / "openings_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"{a.run} -> {a.out}: {rep.get('status')}, added {rep.get('added')}, "
          f"on geometry {rep.get('matched_geometry')}, rejected {rep.get('rejected')}; schema problems {len(problems)}")
    for o in rep.get("openings", []):
        print(f"  {o.get('opening_id')} {o['kind']:6s} {o['action']:26s} host {o['host_id']:8s} width "
              f"{o['width']:.3f} views {o['views']} ends {o['ends_seen']} bottom {o.get('bottom')} top {o.get('top')}")
    for o in rep.get("rejected_openings", []):
        print(f"  rejected {o['kind']:6s} host {o['host_id']:8s} width {o['width']:.3f} views {o['views']}: "
              f"{o['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
