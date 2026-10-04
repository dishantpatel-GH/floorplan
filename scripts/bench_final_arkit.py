#!/usr/bin/env python
"""Final benchmark: ARKitScenes absolute accuracy (LiDAR tier vs Faro laser) re-run with the CURRENT code.

Why a re-run and not a reuse: outputs/benchmark/arkitscenes/summary.json was written at 01:55, before D-027
(corner-aware bias correction, bias.py 02:09), D-029 (deterministic drift), D-030 (S-3 floor rule) and the fusion
point sort. Its `plan_corrected.json` files therefore carry the old +2b-everywhere correction, and re-running only
the `measure` stage would re-measure those stale plans. So both stages (build = the run_capture LiDAR path, measure
= laser GT) are re-run, unchanged, from scripts/bench_arkitscenes.py; only the output folder is redirected.

    python scripts/bench_final_arkit.py [--no-drift-too]   -> outputs/benchmark/final/arkitscenes/{<room>/, summary.json}
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_arkitscenes as BA  # noqa: E402

BA.OUT = ROOT / "outputs" / "benchmark" / "final" / "arkitscenes"


def main():
    BA.OUT.mkdir(parents=True, exist_ok=True)
    variants = (True, False) if "--no-drift-too" in sys.argv else (True,)
    for room in BA.ROOMS:
        for drift in variants:
            t0 = time.time()
            try:
                BA.build(room, drift)
                BA.measure(room, drift)
                print(f"== {room} drift={drift} ok {time.time() - t0:.1f}s", flush=True)
            except Exception:  # noqa: BLE001 - a failure is a result
                print(f"== {room} drift={drift} FAILED\n{traceback.format_exc()}", flush=True)
    BA.summary()


if __name__ == "__main__":
    main()
