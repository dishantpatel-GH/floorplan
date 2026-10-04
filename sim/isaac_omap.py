#!/usr/bin/env python
"""Isaac Sim occupancy map of a scene (isaacsim.asset.gen.omap, PhysX collision geometry), user requirement D-051.

The 2D map covers everything in the body/phone height band (default 0.10-1.90 m): walls, furniture, curtains, range
hoods. It is combined with the mesh-based walkable map of scene_gt.py (union of obstacles), and the scripted walker
plans on the result. Writes <scene dir>/.simcache/gt/isaac_omap.npz (occupied bool grid, origin, cell).

Usage: envs/isaacsim_np126/bin/python floorplan-capture/sim/isaac_omap.py <scene.usda> [--z 0.10 1.90] [--cell 0.05]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("usd")
ap.add_argument("--z", nargs=2, type=float, default=(0.10, 1.90))
ap.add_argument("--cell", type=float, default=0.05)
a = ap.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim import isaac_common as ic  # noqa: E402

app = ic.start_app(headless=True)
import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.gen.omap")
app.update()
from isaacsim.asset.gen.omap.bindings import _omap  # noqa: E402

usd = Path(a.usd).resolve()
gt_dir, gt = ic.load_scene_assets(str(usd))
stage = ic.open_stage(app, str(usd))
allp = np.concatenate([np.asarray(r["polygon"]) for r in gt["rooms"]])
lo, hi = allp.min(0) - 0.5, allp.max(0) + 0.5
start = gt["rooms"][0]["ceiling"]["center_xy"]
tl = omni.timeline.get_timeline_interface()
tl.play()                         # PhysX parses and cooks the colliders on play; one frame, then stop (restores poses)
for _ in range(3):
    app.update()
physx = omni.physx.acquire_physx_interface()
gen = _omap.Generator(physx, omni.usd.get_context().get_stage_id())
gen.update_settings(a.cell, 4, 5, 6)          # occupied / free / unknown values
z0, z1 = a.z
gen.set_transform((start[0], start[1], (z0 + z1) / 2), (lo[0] - start[0], lo[1] - start[1], z0 - (z0 + z1) / 2),
                  (hi[0] - start[0], hi[1] - start[1], z1 - (z0 + z1) / 2))
gen.generate2d()
tl.stop()
app.update()
dims = gen.get_dimensions()
buf = np.asarray(gen.get_buffer(), dtype=np.float32)
mn, mx = np.asarray(gen.get_min_bound()), np.asarray(gen.get_max_bound())
print(f"[omap] dims {dims}, buffer {buf.size}, bounds {mn.round(2)} {mx.round(2)}; values {np.unique(buf, return_counts=True)}",
      flush=True)
grid = buf.reshape(dims[1], dims[0])          # rows = y, cols = x (Isaac's buffer order; checked by the overlay below)
out = gt_dir / "isaac_omap.npz"
np.savez_compressed(out, occupied=(grid == 4), free=(grid == 5), unknown=(grid == 6), min_bound=mn, max_bound=mx,
                    cell=a.cell, z=np.array(a.z))
print(f"[omap] -> {out}", flush=True)
app.close()
