"""Run one plan extractor on one prepared scene, optionally perturbed, into the judge's output layout.

    python scripts/judge_run.py --producer alpha --capture <capture> --variant raw
    python scripts/judge_run.py --producer beta  --capture <capture> --variant drift_on
    python scripts/judge_run.py --producer alpha --capture <capture> --variant raw --rot90 1 --tag rot90
    python scripts/judge_run.py --producer alpha --capture <capture> --variant raw --shift 0.013 0.007 --tag shift
    python scripts/judge_run.py --producer alpha --capture <capture> --variant raw --keep-frac 0.9 --tag thin90
    python scripts/judge_run.py --producer beta  --capture <capture> --variant raw --crop-frac 0.5 --tag crop_half
    python scripts/judge_run.py --producer beta  --capture <capture> --variant raw --crop-size 1.5 --tag tiny
    python scripts/judge_run.py --producer beta  --capture <capture> --variant drift_on --set jog_max_m=0.15 --tag jog15

Output: outputs/judge/runs/<variant>[_<tag>]/<producer>/<capture>/{plan.json, run.json, debug.png}.
run.json records the perturbation, the runtime, and the error if the extractor crashed (crashes are data here).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge_common import (OUT, PRODUCERS, crop_scene, load_scene_for, run_and_save,  # noqa: E402
                          thin_scene, transform_scene)


def centre_box(scene: dict, frac: float | None, size: float | None) -> tuple[float, float, float, float]:
    """A crop box around the middle of the camera path: either `frac` of the extent along u, or a size x size square.

    The half crop keeps the low-u half of the scene (a partial capture: rooms at the cut lose walls); the tiny crop
    is a degenerate input (one corner of one room) to see whether the extractor fails loudly or returns garbage."""
    P = scene["points"]
    u0, u1, v0, v1 = P[:, 0].min(), P[:, 0].max(), P[:, 2].min(), P[:, 2].max()
    if frac is not None:
        return float(u0), float(u0 + frac * (u1 - u0)), float(v0), float(v1)
    cu, cv = np.median(scene["traj"][:, 0]), np.median(scene["traj"][:, 2])
    return float(cu - size / 2), float(cu + size / 2), float(cv - size / 2), float(cv + size / 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--producer", choices=PRODUCERS, required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--variant", choices=("raw", "drift_on"), default="raw")
    ap.add_argument("--rot90", type=int, default=0, help="quarter turns about the vertical axis")
    ap.add_argument("--shift", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DZ"))
    ap.add_argument("--keep-frac", type=float, default=None, help="keep a random share of surface and raw points")
    ap.add_argument("--crop-frac", type=float, default=None)
    ap.add_argument("--crop-size", type=float, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a parameter of the extractor (experiments only; defaults are what is judged)")
    ap.add_argument("--tag", default="", help="suffix of the variant folder for perturbed runs")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()

    scene, info = load_scene_for(a.capture, a.variant)
    extra = dict(capture=a.capture, variant=a.variant, rot90=a.rot90, shift=list(a.shift))
    if a.rot90 or any(a.shift):
        scene = transform_scene(scene, a.rot90, tuple(a.shift))
    if a.keep_frac is not None:
        scene = thin_scene(scene, a.keep_frac)
        extra.update(keep_frac=a.keep_frac)
    if a.crop_frac is not None or a.crop_size is not None:
        box = centre_box(scene, a.crop_frac, a.crop_size)
        scene = crop_scene(scene, box)
        extra.update(crop_box=box, raw_points_kept=int(len(scene["raw_points"])))
    folder = a.variant + (f"_{a.tag}" if a.tag else "")
    extra.update(overrides=a.set)
    rec = run_and_save(a.producer, scene, info, OUT / "runs" / folder / a.producer / a.capture, extra,
                       draw=not a.no_render, overrides=dict(kv.split("=", 1) for kv in a.set))
    print({k: v for k, v in rec.items() if k != "traceback"})


if __name__ == "__main__":
    main()
