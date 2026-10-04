#!/usr/bin/env python
"""Score plans of own-home captures against tape ground truth, and build the Part 3 head-to-head table.

Usage:
  python scripts/eval_own_capture.py --gt gt/ground_truth.csv --plan photo=<plan.json> --plan video=<plan.json> \
      [--app app_export/app_dimensions.csv] [--out outputs/own_eval]

app_dimensions.csv (transcribed from the consumer app's export or screenshots): room_id,item_type,item_id,value_m
(same naming as ground_truth.csv). Head-to-head rule (case study Part 3): per shared dimension compare |our error| with
|app error|; we "beat" when ours is smaller, "tie" when they differ by <= 5 mm; pass if beat-or-tie >= 70%.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.benchmark.gt_eval import evaluate, read_gt, read_gt_geometry, to_markdown  # noqa: E402

TIE_M = 0.005


def read_app(app_csv: Path) -> tuple[dict, list[str]]:
    """{(room_id, item_type, item_id): metres} and the rows that could not be read. item_type may be empty."""
    app, bad = {}, []
    with open(app_csv, newline="", encoding="utf-8-sig") as f:                # utf-8-sig: Excel writes a BOM
        for r in csv.DictReader(row for row in f if row.strip() and not row.lstrip().startswith("#")):
            try:
                key = (r["room_id"].strip(), (r.get("item_type") or "").strip(), r["item_id"].strip())
                app[key] = float(r["value_m"])
            except (KeyError, TypeError, ValueError, AttributeError):
                bad.append(",".join(str(v) for v in r.values()))
    return app, bad


def head_to_head(res_ours: dict, app_csv: Path, gt):
    """Dimensions are keyed by (room, item type, item id): one id can carry several measurements (door D1 has a
    door_width and a door_height row), so (room, id) alone picked whichever GT row came last. A dimension the app
    reports and our plan does not is listed under 'not_measured_by_us' (it is not a shared dimension: not gated)."""
    app, bad = read_app(app_csv)
    gtv = {(g.room, g.kind, g.item): g.value for g in gt}
    rows, beat_tie, used, not_ours = [], 0, set(), []
    for r in res_ours["rows"]:
        key = (r["room"], r["kind"], r["item"])
        akey = key if key in app else (r["room"], "", r["item"])               # an app row without item_type
        if akey not in app or key not in gtv:
            continue
        used.add(akey)
        if r["value"] is None:
            not_ours.append(dict(room=r["room"], item=r["item"], kind=r["kind"], gt=gtv[key], app=app[akey],
                                 note=r.get("note") or "not measured"))
            continue
        e_ours, e_app = abs(r["value"] - gtv[key]), abs(app[akey] - gtv[key])
        outcome = "tie" if abs(e_ours - e_app) <= TIE_M else ("beat" if e_ours < e_app else "lose")
        beat_tie += outcome != "lose"
        rows.append(dict(room=r["room"], item=r["item"], kind=r["kind"], gt=gtv[key], ours=r["value"], app=app[akey],
                         err_ours=e_ours, err_app=e_app, outcome=outcome))
    frac = beat_tie / len(rows) if rows else None
    return dict(rows=rows, beat_or_tie_frac=frac, passed=None if frac is None else frac >= 0.70,
                not_measured_by_us=not_ours, app_rows_unused=sorted("/".join(k) for k in set(app) - used),
                app_rows_unreadable=bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--gt-json", type=Path, help="GT room polygons for geometric wall pairing (default: sim_gt.json next "
                                                 "to the CSV if present; else walls are paired by robust order)")
    ap.add_argument("--tape-pairing", choices=("anchored", "order-merged"), default="anchored",
                    help="wall pairing without GT polygons: anchored alignment (W1 at the door wall; default) or the "
                         "older order-merged pairing, for comparison")
    ap.add_argument("--plan", action="append", required=True, help="tier=path/to/plan.json")
    ap.add_argument("--app", type=Path)
    ap.add_argument("--h2h-tier", default="video")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "own_eval")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    gt = read_gt(a.gt)
    gt_geom = read_gt_geometry(a.gt, a.gt_json)
    print(f"wall pairing: {'geometric (GT polygons: ' + str(a.gt_json or a.gt.with_name('sim_gt.json')) + ')' if gt_geom else a.tape_pairing + ' (no GT polygons)'}")
    md, results = ["# Own-capture benchmark against tape ground truth", ""], {}
    for spec in a.plan:
        tier, path = spec.split("=", 1)
        res = evaluate(json.loads(Path(path).read_text()), gt, tier, gt_geom, tape_method=a.tape_pairing)
        results[tier] = res
        for g, p in res["wall_pairing"].items():
            if p.get("ambiguity"):
                print(f"WARNING: {tier} {g}: ambiguous wall pairing: {p['ambiguity']}", file=sys.stderr)
        md += [to_markdown(res), ""]
    if a.app and a.h2h_tier not in results:
        md += [f"## Head-to-head: not computed (no plan was scored as '{a.h2h_tier}'; scored: {sorted(results)})", ""]
    if a.app and a.h2h_tier in results:
        h = head_to_head(results[a.h2h_tier], a.app, gt)
        results["head_to_head"] = h
        md += [f"## Head-to-head (our {a.h2h_tier} tier vs consumer app), tie = within {TIE_M * 1000:.0f} mm", "",
               "| Room | Item | GT | Ours | App | Err ours | Err app | Outcome |", "|---|---|---|---|---|---|---|---|"]
        md += [f"| {r['room']} | {r['item']} | {r['gt']:.3f} | {r['ours']:.3f} | {r['app']:.3f} | {r['err_ours']:.3f} | "
               f"{r['err_app']:.3f} | {r['outcome']} |" for r in h["rows"]]
        verdict = "no shared dimension" if h["passed"] is None else ("PASS" if h["passed"] else "FAIL")
        frac = "n/a" if h["beat_or_tie_frac"] is None else f"{h['beat_or_tie_frac']:.2f}"
        md += ["", f"Beat or tie: {frac} over {len(h['rows'])} shared dimension(s) (gate >= 0.70 -> {verdict})"]
        if h["not_measured_by_us"]:
            md += ["", f"Reported by the app but not by our plan ({len(h['not_measured_by_us'])}, not in the gate): "
                   + ", ".join(f"{r['room']} {r['item']}" for r in h["not_measured_by_us"])]
        if h["app_rows_unused"] or h["app_rows_unreadable"]:
            md += ["", f"App rows left out (id not in the ground truth, or no such row in our evaluation): "
                   f"{h['app_rows_unused']}; unreadable rows: {h['app_rows_unreadable']}"]
    (a.out / "own_eval.json").write_text(json.dumps(results, indent=1, default=float))
    (a.out / "own_eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
