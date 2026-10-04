#!/usr/bin/env python
"""Process an own-home capture end to end and score it against tape ground truth (D-015).

Expected layout (docs/HOUSE_CAPTURE_GUIDE.md, step 0; the simulator writes the same, sim/emulate.py):
  <root>/photos/<room>/*.jpg        one folder per room; a repeat take of one room is <room>_take2/
  <root>/video/take1.* take2.* [lowlight.*]
  <root>/lidar/<take>/              Stray Scanner folders (optional; LiDAR tier)
  <root>/gt/ground_truth.csv
  <root>/gt/gt_polygons.json             (optional; room outlines from scripts/house_gt.py, for wall pairing)
  <root>/app_export/app_dimensions.csv   (optional; consumer-app dimensions transcribed, same ids as the GT)

Runs (each through the one command, scripts/run_capture.py):
  photo tier on all room folders except *_take2   -> <out>/photo
  photo tier on the repeat room, take 1 and take 2 -> <out>/photo_repeat_take1|take2   (repeatability, photo tier)
  video tier on every video file                    -> <out>/video_<name>
  LiDAR tier on every Stray Scanner folder          -> <out>/lidar_<name>
Then scores every plan against the tape GT and builds the head-to-head table (scripts/eval_own_capture.py).

Usage: python scripts/process_own_capture.py <root> [--out outputs/own] [--skip-runs] [--h2h-tier video]
                                            [--tiers photo,video,lidar]

Check the folder first with scripts/check_own_capture.py (docs/OWN_CAPTURE_RUNBOOK.md). Exit code: 0 = every run and
the scoring succeeded, 1 = a run failed or nothing was scored.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".3gp", ".mkv"}


def run(cmd: list[str], log: Path) -> bool:
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"  {'OK ' if rc == 0 else 'ERR'} {time.time() - t0:6.0f}s  {' '.join(cmd[2:5])} ...  (log: {log})", flush=True)
    return rc == 0


def take1_folder(take2_name: str, names: list[str]) -> str | None:
    """First-take folder of a repeat folder '<x>_take2': '<x>', else the one folder with the same room name after
    the 'NN_' prefix (the guide's layout example numbers the repeat folder on its own: 02_bedroom / 06_bedroom_take2).
    None when there is no such folder, or more than one."""
    base = take2_name[: -len("_take2")]
    if base in names:
        return base
    strip = lambda n: re.sub(r"^\d+_", "", n)  # noqa: E731
    same = [n for n in names if not n.endswith("_take2") and strip(n) == strip(base)]
    return same[0] if len(same) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "own")
    ap.add_argument("--skip-runs", action="store_true", help="only re-score existing plans")
    ap.add_argument("--h2h-tier", default=None, help="tier compared with the consumer app (default: video if "
                                                     "video_take1 was scored, else photo)")
    ap.add_argument("--tiers", default="photo,video,lidar", help="which tiers to run (scoring uses whatever exists)")
    a = ap.parse_args()
    tiers = set(a.tiers.split(","))
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    photos, videos = a.root / "photos", a.root / "video"
    plans: dict[str, Path] = {}
    failed: list[str] = []
    if not a.root.is_dir():
        raise SystemExit(f"capture folder not found: {a.root}")

    # ---- photo tier: all rooms (take 1), plus the repeat room on its own, twice
    if photos.is_dir() and "photo" in tiers:
        main_dir = out / "_photo_inputs_main"
        if not a.skip_runs:
            shutil.rmtree(main_dir, ignore_errors=True)
            main_dir.mkdir(parents=True)
            for d in sorted(p for p in photos.iterdir() if p.is_dir()):
                if not d.name.endswith("_take2"):
                    shutil.copytree(d, main_dir / d.name)
            if not run([PY, "scripts/run_capture.py", str(main_dir), "--tier", "photo", "--out", str(out / "photo")],
                       out / "photo.log"):
                failed.append("photo")
        plans["photo"] = out / "photo" / "plan.json"
        names = [p.name for p in photos.iterdir() if p.is_dir()]
        for k, d2 in enumerate(sorted(p for p in photos.iterdir() if p.is_dir() and p.name.endswith("_take2"))):
            room = take1_folder(d2.name, names)
            if room is None:
                print(f"  repeat folder {d2.name}: no first-take folder found (expected "
                      f"photos/{d2.name[: -len('_take2')]}/): repeat runs skipped", flush=True)
                failed.append(f"photo repeat {d2.name}")
                continue
            # both takes are run under the first take's folder name, so the two plans carry the same room label;
            # a second repeat room gets its own keys instead of overwriting the first one's runs
            tag = "photo_repeat" if k == 0 else f"photo_repeat_{room}"
            for take, src in (("take1", photos / room), ("take2", d2)):
                tdir = out / f"_{tag}_{take}" / room
                if not a.skip_runs and src.is_dir():
                    shutil.rmtree(tdir.parent, ignore_errors=True)
                    shutil.copytree(src, tdir)
                    if not run([PY, "scripts/run_capture.py", str(tdir.parent), "--tier", "photo",
                                "--out", str(out / f"{tag}_{take}")], out / f"{tag}_{take}.log"):
                        failed.append(f"{tag}_{take}")
                plans[f"{tag}_{take}"] = out / f"{tag}_{take}" / "plan.json"

    # ---- video tier: every clip
    if videos.is_dir() and "video" in tiers:
        for v in sorted(p for p in videos.iterdir() if p.suffix.lower() in VIDEO_EXT):
            key = f"video_{v.stem}"
            if not a.skip_runs and not run([PY, "scripts/run_capture.py", str(v), "--tier", "video",
                                            "--out", str(out / key)], out / f"{key}.log"):
                failed.append(key)
            plans[key] = out / key / "plan.json"

    # ---- LiDAR tier: every Stray Scanner folder
    lidar = a.root / "lidar"
    if lidar.is_dir() and "lidar" in tiers:
        for d in sorted(p for p in lidar.iterdir() if (p / "odometry.csv").exists()):
            key = f"lidar_{d.name}"
            if not a.skip_runs and not run([PY, "scripts/run_capture.py", str(d), "--tier", "lidar",
                                            "--out", str(out / key)], out / f"{key}.log"):
                failed.append(key)
            plans[key] = out / key / "plan.json"

    # ---- also pick up plans already in the output folder (re-scoring with --skip-runs)
    for d in sorted(out.iterdir()):
        if d.is_dir() and d.name.startswith(("photo", "video_", "lidar_")) and (d / "plan.json").exists():
            plans.setdefault(d.name, d / "plan.json")

    # ---- score against tape GT (+ head-to-head)
    gt = a.root / "gt" / "ground_truth.csv"
    have = {k: p for k, p in plans.items() if p.exists()}
    missing = sorted(set(plans) - set(have))
    if missing:
        print(f"  missing plans (failed or skipped runs): {missing}")
    scored = False
    if gt.exists() and have:
        cmd = [PY, "scripts/eval_own_capture.py", "--gt", str(gt), "--out", str(out / "eval")]
        if gt.with_name("gt_polygons.json").exists():          # D-072: pair walls on the outlines, not by order
            cmd += ["--gt-json", str(gt.with_name("gt_polygons.json"))]
        labels = []
        for k, p in have.items():
            tier = "photo" if k.startswith("photo") else ("lidar" if k.startswith("lidar") else "video")
            labels.append(tier if k in ("photo", "video_take1", "lidar_take1") else k)
            cmd += ["--plan", f"{labels[-1]}={p}"]
        app = a.root / "app_export" / "app_dimensions.csv"
        if app.exists():
            h2h = a.h2h_tier or ("video" if "video_take1" in have else "photo")
            if h2h not in labels:
                print(f"  head-to-head tier '{h2h}' has no plan (scored: {labels}): no head-to-head table")
            cmd += ["--app", str(app), "--h2h-tier", h2h]
        else:
            print(f"  no {app}: no head-to-head table")
        scored = run(cmd, out / "eval.log")
        if scored:
            print(f"  report: {out / 'eval' / 'own_eval.md'}")
    else:
        print(f"  nothing scored: {'no plans' if gt.exists() else f'no ground truth at {gt}'}")
    if failed:
        print(f"  FAILED: {failed} (see the logs above)")
    sys.exit(0 if scored and not failed and not missing else 1)


if __name__ == "__main__":
    main()
