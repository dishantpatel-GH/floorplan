#!/usr/bin/env python
"""Photo tier, one command: <photo_root>/<room>/*.jpg -> outputs/photo_tier/<name>/scene.npz + scene_info.json.

If the photo folders were simulated from a Stray Scanner capture (scripts/make_photo_folders.py), a truth file sits
next to them; it is then used AFTER the run, for evaluation only (evaluation.json + figures). It is never an input.

Usage: python scripts/run_photo_frontend.py <photo_root> [--out outputs/photo_tier/<name>] [--no-eval]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floorplan.photo import build_scene_from_photos  # noqa: E402
from floorplan.pipeline.scene import save_scene  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo_root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--params", type=Path, default=None, help="JSON file overriding PhotoParams")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    name = a.photo_root.resolve().parent.name if a.photo_root.name == "photos" else a.photo_root.resolve().name
    out = a.out or ROOT / "outputs" / "photo_tier" / name
    params = json.loads(a.params.read_text()) if a.params else None
    scene, info = build_scene_from_photos(a.photo_root, params, work_dir=out / "work")
    save_scene(scene, info, out)
    print(f"saved {out / 'scene.npz'}")
    if not a.no_eval:
        from floorplan.photo.report import evaluate_run
        ev = evaluate_run(a.photo_root, scene, info, out, ROOT)
        if ev:
            print(json.dumps(ev["summary"], indent=1))


if __name__ == "__main__":
    main()
