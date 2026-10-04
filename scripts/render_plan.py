#!/usr/bin/env python
"""Validate a plan.json and render it to plan.svg, plan.png and plan.dxf (+ a render report).

Why a standalone command: the pipeline writes plan.json, and anyone (a reviewer, the fix loop's before/after runs,
another tier's front-end) can re-draw any plan from that file alone, without re-running reconstruction.
The JSON is validated first; an invalid plan is still drawn (so the problem can be seen) but reported, and
--strict turns that into a non-zero exit code for CI.

Usage: python scripts/render_plan.py <plan.json> [--out DIR] [--strict] [--no-dxf]
       (DIR defaults to the folder that holds plan.json)
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.export.dxf import export_dxf, preview_dxf  # noqa: E402
from floorplan.export.json_export import plan_from_json, validate_plan_json  # noqa: E402
from floorplan.export.render import render_plan  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate and render a floor-plan JSON.")
    ap.add_argument("plan", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--strict", action="store_true", help="exit 1 if the JSON fails validation")
    ap.add_argument("--no-dxf", action="store_true")
    ap.add_argument("--dxf-preview", action="store_true", help="also rasterise the DXF (independent read-back check)")
    a = ap.parse_args()
    out = a.out or a.plan.parent
    out.mkdir(parents=True, exist_ok=True)

    doc = json.loads(a.plan.read_text())
    errors = validate_plan_json(doc)
    for e in errors:
        print(f"INVALID {e}", file=sys.stderr)
    plan = plan_from_json(doc)

    t0 = time.perf_counter()
    stats = render_plan(plan, out / "plan")
    report = {"plan": str(a.plan), "valid": not errors, "validation_errors": errors,
              "render": asdict(stats), "render_s": round(time.perf_counter() - t0, 2)}
    if not a.no_dxf:
        report["dxf"] = export_dxf(plan, out / "plan.dxf")
        if a.dxf_preview:
            preview_dxf(out / "plan.dxf", out / "plan_dxf_preview.png")
    (out / "render_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'valid' if not errors else f'{len(errors)} validation errors'}; wrote {out}/plan.svg, plan.png"
          f"{', plan.dxf' if not a.no_dxf else ''}, render_report.json")
    return 1 if (errors and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
