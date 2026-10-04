"""Semantic wall masks for the photo tier (D-060): only points on WALLS may define a room's sides.

Why: the room box (layout.py) is fitted to "wall points", i.e. MoGe-2 points with a near-horizontal normal in the 1-2 m
band. Wardrobes, kitchen cabinets, shower partitions, vanities and columns are vertical too. On the simulated 2 BHK
(exact ground truth) they became walls: bathroom 1.49 m deep vs 3.54 m, bedroom 2.69 m wide vs 3.67 m, footprint -45%.
An ADE20K segmenter (SegFormer-B5: wall, cabinet, wardrobe, bed, mirror, column, shower, ...) separates them. Mirrors
are dropped too (the room seen in a mirror lies behind the wall).

The segmenter runs in its own env (envs/seg; transformers must not enter the main .venv, I-002) as a subprocess, once
per capture, cached in the photo work dir. If anything fails, the photo tier runs exactly as before (geometry only) and
says so in its report.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np

WALL_LABELS = ("wall", "painting")              # paintings hang flat on walls; their depth is the wall's
FRONT_LABELS = {"cabinet": 0.60, "wardrobe": 0.60, "refrigerator": 0.65, "chest of drawers": 0.50, "bookcase": 0.35,
                "shelf": 0.35, "counter": 0.60, "countertop": 0.60, "kitchen island": 0.90, "stove": 0.60,
                "wardrobe, closet, press": 0.60}  # furniture usually stands against a wall: its typical depth (m)


def _seg_python() -> Path | None:
    p = os.environ.get("FLOORPLAN_SEG_PYTHON")
    if p and Path(p).exists():
        return Path(p)
    root = Path(__file__).resolve().parents[2]
    for base in (root, root.parent):
        c = base / "envs" / "seg" / "bin" / "python"
        if c.exists():
            return c
    return None


def _resize_nearest(m: np.ndarray, h: int, w: int) -> np.ndarray:
    yi = np.minimum((np.arange(h) * m.shape[0] / h).astype(int), m.shape[0] - 1)
    xi = np.minimum((np.arange(w) * m.shape[1] / w).astype(int), m.shape[1] - 1)
    return m[yi][:, xi]


def attach_semantics(photos, views: dict, work: Path, p, log=print) -> dict:
    """Give every view `sem` (uint8 ADE20K class map at the depth resolution), `sem_labels` and `wall_mask`."""
    if not getattr(p, "semantic_walls", True):
        return dict(status="disabled")
    t0 = time.time()
    raw = Path(work) / "semantic_raw.npz"
    names = [ph.name for ph in photos if ph.name in views]
    z = None
    if raw.exists():
        try:
            zz = np.load(raw)
            if set(names) <= {str(k) for k in zz["keys"]}:
                z = zz
        except Exception:                                  # a broken cache is recomputed
            z = None
    if z is None:
        py = _seg_python()
        script = Path(__file__).resolve().parents[2] / "scripts" / "seg_walls.py"
        if py is None or not script.exists():
            log("[photo/sem] no segmentation env (envs/seg): wall points from geometry only")
            return dict(status="no segmentation env")
        lst = Path(work) / "semantic_list.txt"
        lst.write_text("".join(f"{ph.name}\t{Path(ph.path).resolve()}\n" for ph in photos if ph.name in views))
        env = dict(os.environ)
        env.setdefault("HF_HOME", str(py.parents[3] / "weights" / "hf_seg"))
        try:
            r = subprocess.run([str(py), str(script), "--list", str(lst), "--out", str(raw)], env=env,
                               capture_output=True, text=True, timeout=getattr(p, "semantic_timeout_s", 900))
        except subprocess.TimeoutExpired:
            log("[photo/sem] segmentation timed out: wall points from geometry only")
            return dict(status="timed out")
        if r.returncode != 0 or not raw.exists():
            log(f"[photo/sem] segmentation failed (rc {r.returncode}): wall points from geometry only")
            return dict(status="failed", stderr=r.stderr[-500:])
        z = np.load(raw)
    labels = [str(x).strip() for x in z["names"]]
    wall_ids = np.array([i for i, n in enumerate(labels) if n in WALL_LABELS])
    front = {i: FRONT_LABELS[n] for i, n in enumerate(labels) if n in FRONT_LABELS}
    keys = [str(k) for k in z["keys"]]
    done, wall_share = 0, []
    for i, k in enumerate(keys):
        v = views.get(k)
        if v is None:
            continue
        h, w = v.depth.shape
        v.sem = _resize_nearest(z[f"m{i}"], h, w)
        v.sem_labels = labels
        v.sem_wall_ids = wall_ids
        v.sem_front_depth = front
        v.wall_mask = np.isin(v.sem, wall_ids)
        wall_share.append(float(v.wall_mask.mean()))
        done += 1
    log(f"[photo/sem] semantic wall masks for {done}/{len(names)} photos ({time.time() - t0:.0f}s); "
        f"median wall share {np.median(wall_share) if wall_share else 0:.2f}")
    return dict(status="ok", photos=done, wall_labels=list(WALL_LABELS), seconds=round(time.time() - t0, 1))


def semantic_wall_points(view, stride: int) -> np.ndarray | None:
    """Boolean per point of view.points_cam(stride): True on wall pixels; None when the view has no class map."""
    sem = getattr(view, "sem", None)
    if sem is None:
        return None
    _, idx = view.points_cam(stride)
    return np.isin(sem.ravel()[idx], view.sem_wall_ids)
