#!/usr/bin/env python
"""One table of every simulated run's gates: house x person-seed x tier (all results are SIMULATED).

Reads outputs/sim/<exp>/runs_<profile>*/eval/own_eval.json (written by scripts/process_own_capture.py) and writes
outputs/sim/SIM_REPORT.md.

Usage: python sim/report.py [--root outputs/sim]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fmt(x, d=1, pct=False):
    if x is None:
        return "—"
    return f"{100 * x:.{d}f}%" if pct else f"{x:.{d}f}"


def rows_for(exp: Path, runs: Path):
    ev = runs / "eval" / "own_eval.json"
    if not ev.exists():
        return []
    res = json.loads(ev.read_text())
    out = []
    for tier, r in res.items():
        if tier == "head_to_head" or not isinstance(r, dict) or "gates" not in r:
            continue
        g = r["gates"]
        walls = [x for x in r["rows"] if x["kind"] == "wall" and x["rel"] is not None]
        wg = g.get("wall_gate", {})
        og = g.get("opening_gate", {})
        cg = g.get("ceiling_gate", {})
        st = g.get("stitch", {})
        fp = st.get("footprint", {})
        adj = st.get("adjacency", {})
        cal = g.get("calibration", {})
        out.append(dict(
            exp=exp.name, runs=runs.name, tier=tier,
            rooms=f"{st.get('matched_rooms', '?')}/{st.get('gt_rooms', '?')} (plan {st.get('plan_rooms', '?')})",
            wall_med=np.median([x["rel"] for x in walls]) if walls else None,
            wall_abs=np.median([abs(x["err"]) for x in walls]) if walls else None,
            wall_pass=wg.get("pass_frac"), wall_tol=wg.get("tol"),
            open_pass=og.get("pass_frac"), open_missed=og.get("missed"), open_phantom=og.get("phantom"),
            ceil_pass=cg.get("pass_frac"),
            ceil_err=np.median([abs(x["err"]) for x in r["rows"] if x["kind"] == "ceiling_height" and x["err"] is not None]) if cg else None,
            fp_err=fp.get("rel_err"), adj=f"{adj.get('found', '—')}/{adj.get('gt', '—')} (+{adj.get('wrong', 0)} wrong)" if adj else "—",
            overlaps=len(st.get("overlaps") or []), stitch=st.get("passed"), cov=cal.get("coverage_95")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "outputs" / "sim")
    a = ap.parse_args()
    allrows = []
    for exp in sorted(p for p in a.root.iterdir() if p.is_dir()):
        for runs in sorted(exp.glob("runs_*")):
            allrows += rows_for(exp, runs)
    L = ["# Simulated benchmark (Isaac Sim + emulated iPhone 15 files; NOT the real benchmark)", "",
         "Each row: one house x one scripted person (seed) x one tier, scored against exact ground truth "
         "(sim/scene_gt.py). Gates as in the case study: walls photo +-8% / video +-3%; openings <= 2 cm on >= 85%; "
         "ceiling <= 1.5 cm; stitch = all rooms, correct adjacency, no overlaps, footprint +-8%.", "",
         "| Run | Tier | Rooms matched | Wall median err | Wall median abs (m) | Walls in gate | Openings <=2 cm (missed/phantom) | "
         "Ceiling <=1.5 cm (median err m) | Footprint err | Adjacency | Overlaps | Stitch | 95% CI coverage |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in allrows:
        L.append(f"| {r['exp']}/{r['runs'].replace('runs_', '')} | {r['tier']} | {r['rooms']} | {fmt(r['wall_med'], 1, True)} | "
                 f"{fmt(r['wall_abs'], 3)} | {fmt(r['wall_pass'], 0, True)} | {fmt(r['open_pass'], 0, True)} "
                 f"({r['open_missed']}/{r['open_phantom']}) | {fmt(r['ceil_pass'], 0, True)} ({fmt(r['ceil_err'], 3)}) | "
                 f"{fmt(r['fp_err'], 1, True)} | {r['adj']} | {r['overlaps']} | "
                 f"{'PASS' if r['stitch'] else ('FAIL' if r['stitch'] is not None else '—')} | {fmt(r['cov'], 0, True)} |")
    out = a.root / "SIM_REPORT.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
