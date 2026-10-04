#!/usr/bin/env python
"""One command for a simulated experiment: scene GT -> session -> render -> phone files -> pipeline -> scores.

Every stage is skipped when its output already exists (so changing only the noise seed re-runs only emulate + the
pipeline). The pipeline stage is exactly what runs on real captures (scripts/process_own_capture.py).

Usage (main .venv):
  python sim/experiment.py --scene <scene.usda> --name k65_s0 [--seed 0] [--fps 10] [--profile iphone15]
         [--drift arkit] [--tiers photo,video,lidar] [--photo-protocol spin] [--takes take1,take2]
         [--session <dir with session.json from teleop.py>]   (use your own driven session instead of a scripted one)
Outputs under outputs/sim/<name>/ : session.json, render/, capture_<profile>[_<tag>]/, runs_<profile>[_<tag>]/
(plans + logs + eval/own_eval.md).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT.parent
PY = sys.executable
ISAAC_PY = MAPPING / "envs" / "isaacsim_np126" / "bin" / "python"


def sh(cmd, log: Path | None = None):
    t = time.time()
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    if log:
        with open(log, "w") as f:
            rc = subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    else:
        rc = subprocess.call([str(c) for c in cmd], cwd=ROOT)
    print(f"    -> rc={rc} in {time.time() - t:.0f}s", flush=True)
    if rc != 0:
        raise SystemExit(f"stage failed: {cmd[1] if len(cmd) > 1 else cmd[0]} (log: {log})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(MAPPING / "data/sim_scenes/kujiale_0065/kujiale_0065.usda"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--session", type=Path, help="existing session dir (teleop.py); default: scripted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--human", type=float, default=1.0)
    ap.add_argument("--profile", default="iphone15")
    ap.add_argument("--drift", default="arkit")
    ap.add_argument("--tiers", default="photo,video,lidar")
    ap.add_argument("--photo-protocol", default="spin")
    ap.add_argument("--takes", default="take1")
    ap.add_argument("--tag", default="", help="suffix for capture/runs folders (e.g. a noise variant)")
    ap.add_argument("--emu-seed", type=int, default=0)
    a = ap.parse_args()
    scene = Path(a.scene).resolve()
    exp = a.session or (ROOT / "outputs" / "sim" / a.name)
    exp.mkdir(parents=True, exist_ok=True)
    gt_dir = scene.parent / ".simcache" / "gt"
    tag = f"_{a.tag}" if a.tag else ""
    print(f"[exp] {a.name}: scene {scene.name}, session {exp}", flush=True)
    if not (gt_dir / "sim_gt.json").exists():
        sh([PY, "sim/scene_gt.py", scene])
    if not (exp / "session.json").exists():
        sh([PY, "sim/auto_capture.py", "--scene", scene, "--session", exp, "--seed", a.seed, "--takes", a.takes,
            "--photo-protocol", a.photo_protocol, "--human", a.human])
    if not (exp / "render" / "render.json").exists():
        sh([ISAAC_PY, "sim/render.py", exp, "--fps", a.fps, "--human", a.human, "--seed", a.seed], exp / "render.log")
    cap = exp / f"capture_{a.profile}{tag}"
    if not (cap / "capture.json").exists():
        sh([PY, "sim/emulate.py", exp, "--profile", a.profile, "--drift", a.drift, "--seed", a.emu_seed, "--out", cap,
            "--tiers", a.tiers], exp / f"emulate_{a.profile}{tag}.log")
    runs = exp / f"runs_{a.profile}{tag}"
    sh([PY, "scripts/process_own_capture.py", cap, "--out", runs, "--tiers", a.tiers, "--h2h-tier", "video"],
       exp / f"process_{a.profile}{tag}.log")
    rep = runs / "eval" / "own_eval.md"
    print(f"[exp] report: {rep}", flush=True)
    meta = dict(name=a.name, scene=str(scene), args=vars(a) | {"session": str(a.session) if a.session else None},
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (runs / "experiment.json").write_text(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
