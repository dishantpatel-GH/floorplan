#!/usr/bin/env python
"""Read the user's corrected values back from the editable house-plan PDF (scripts/sketch_plan_pdf.py).

Writes a CSV with the original reading and the value now in each box, and lists every value the user changed.
Run with the small PDF env:
  envs/pdf/bin/python floorplan-capture/scripts/read_plan_pdf.py MyHouse_Dataset/house_plan_editable.pdf \
      --base MyHouse_Dataset/house_measurements.csv --out MyHouse_Dataset/house_measurements_checked.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pypdf import PdfReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--base", type=Path, required=True, help="house_measurements.csv written with the PDF")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fields = {k: (v.get("/V") or "") for k, v in (PdfReader(a.pdf).get_fields() or {}).items()}
    rows = list(csv.DictReader(open(a.base)))
    changed = []
    for r in rows:
        raw = str(fields.get(r["id"], r["value_cm"])).strip().replace(",", ".")
        try:
            v = float(raw)
        except ValueError:
            v = None
        r["value_cm_checked"] = "" if v is None else f"{v:.1f}"
        if v is None or abs(v - float(r["value_cm"])) > 1e-6:
            changed.append(f"{r['id']} ({r['room']}, {r['item']}): {r['value_cm']} -> {raw or 'empty'}")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} values, {len(changed)} changed -> {a.out}")
    for c in changed:
        print("  " + c)
    note = str(fields.get("notes", "")).strip()
    if note:
        print("notes: " + note)


if __name__ == "__main__":
    main()
