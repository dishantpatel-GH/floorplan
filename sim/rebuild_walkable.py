#!/usr/bin/env python
"""Rebuild only <scene>/.simcache/gt/walkable.npz with a different body clearance (D-064, user 4 Oct 18:50: "25 cm is
too much; relax it"). Ground truth, openings and the A4 sheet spots are left untouched (scene_gt.py places the sheets
from the walkable map, so a full rebuild would move them).

Usage: python sim/rebuild_walkable.py <scene.usda> [--body 0.18] [--margin 0.12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import scene_geom as sg  # noqa: E402
from sim.scene_gt import walkable_grid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("usd", type=Path)
ap.add_argument("--body", type=float, default=0.18)
ap.add_argument("--margin", type=float, default=0.12)
a = ap.parse_args()
out = a.usd.resolve().parent / ".simcache" / "gt"
gt = json.loads((out / "sim_gt.json").read_text())
geom = sg.load(a.usd)
boxes = sg.object_boxes(geom)
inner = [o for o in gt["openings"] if o["cls"] == "door" and len(o["room_ids"]) == 2]
near = lambda b, o: all(b["min"][i] - 0.2 < o["center"][i] < b["max"][i] + 0.2 for i in (0, 1))  # noqa: E731
sills = [b for b in boxes.values() if b["cls"] == "doorsill" and any(near(b, o) for o in inner)]
omap_f = out / "isaac_omap.npz"
wk = walkable_grid(geom, gt["rooms"], sills, margin_wall=a.margin, body=a.body,
                   omap=dict(np.load(omap_f)) if omap_f.exists() else None)
old = np.load(out / "walkable.npz")
np.savez_compressed(out / "walkable_body25.npz", **{k: old[k] for k in old.files}) if not (out / "walkable_body25.npz").exists() else None
np.savez_compressed(out / "walkable.npz", origin=wk["origin"], res=wk["res"], soft=wk["soft"], strict=wk["strict"],
                    furniture=wk["furniture"])
print(f"[walkable] {a.usd.name}: body {a.body} m: strict cells {int(old['strict'].sum())} -> {int(wk['strict'].sum())}, "
      f"soft {int(old['soft'].sum())} -> {int(wk['soft'].sum())}")
