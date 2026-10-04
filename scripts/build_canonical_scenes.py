#!/usr/bin/env python
"""Rebuild the canonical LiDAR scenes with drift correction ON and pixel-centre depth intrinsics (D-011, drift module).

Writes outputs/scenes_v2/<capture>/{scene.npz, scene_info.json, drift_report.json}. These are the scenes the final
pipeline produces (scripts/run_capture.py does the same per capture); stored separately so modules evaluated earlier
on outputs/<capture>/scene.npz (drift OFF, plain intrinsics) keep their reproducible inputs.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.config import Config  # noqa: E402
from floorplan.io.stray import load_stray  # noqa: E402
from floorplan.pipeline.scene import build_scene, save_scene  # noqa: E402
from floorplan.recon.drift import correct_drift  # noqa: E402
from floorplan.recon.keyframes import select_keyframes  # noqa: E402

# sample captures: $DATASET, default TakeHome/Dataset next to the repo
DATA = Path(os.environ.get("DATASET", Path(__file__).resolve().parents[2] / "TakeHome" / "Dataset"))
for rel in sys.argv[1:] or ["single_room/c00a170fe1", "single_scan_floor_only/1a8384c3f6", "single_scan_with_ceiling/c7d28f72c6"]:
    t0 = time.time()
    cdir = DATA / rel
    cfg = Config.load()
    cap = load_stray(cdir)
    T_corr, rep = correct_drift(cap, select_keyframes(cap, cfg), cap.T_wc, cfg)
    scene, info, _ = build_scene(cdir, cfg, T_wc=T_corr)
    info["drift_corrected"] = True
    info["depth_intrinsics"] = "pixel-centre (D-011)"
    out = Path(__file__).resolve().parents[1] / "outputs" / "scenes_v2" / f"{cdir.parent.name}__{cdir.name}"
    save_scene(scene, info, out)
    (out / "drift_report.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "state"}, indent=1, default=lambda o: o if isinstance(o, (int, float)) else str(o)))
    print(f"{rel}: done in {time.time() - t0:.0f}s -> {out}", flush=True)
