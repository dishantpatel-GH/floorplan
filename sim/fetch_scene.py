#!/usr/bin/env python
"""Download InteriorAgent scenes (Hugging Face dataset spatialverse/InteriorAgent) for the simulator.

Large binaries are fetched by script, never committed (case study constraint). The dataset has its own terms of use
(https://kloudsim-usa-cos.kujiale.com/InteriorAgent/InteriorAgent_Terms_of_Use.pdf; docs/DISCLOSURES.md).
Scenes used: kujiale_0065 (1 BHK + balcony, 59 m2, the development scene), kujiale_0038 (another 1 BHK, 63 m2),
kujiale_0022 (2 BHK, 61 m2). Each is 0.2-0.4 GB.

Usage: python sim/fetch_scene.py kujiale_0065 [kujiale_0038 ...] [--dest data/sim_scenes]
Then:  python sim/scene_gt.py data/sim_scenes/<scene>/<scene>.usda
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DEST = Path(__file__).resolve().parents[2] / "data" / "sim_scenes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    a.dest.mkdir(parents=True, exist_ok=True)
    for sc in a.scenes:
        snapshot_download("spatialverse/InteriorAgent", repo_type="dataset", allow_patterns=[f"{sc}/*", f"{sc}/**"],
                          local_dir=str(a.dest), max_workers=8)
        usd = a.dest / sc / f"{sc}.usda"
        print(f"[fetch] {usd} {'OK' if usd.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
