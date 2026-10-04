#!/usr/bin/env python
"""Benchmark: absolute accuracy of the LiDAR-tier PLAN (the one-command pipeline) against Faro laser ground truth.

Rooms: the 5 ARKitScenes rooms of the LiDAR-bias work (docs/modules/lidar_bias.md); 3 development rooms (the bias
correction b = 0.68 cm was fitted on them) and 2 hold-out rooms (analysed only after b was pre-registered).

Stage `build` (per room) mirrors scripts/run_capture.py's LiDAR path exactly, but takes the ARKitScenes adapter
(floorplan/io/arkitscenes.py) instead of a Stray folder, because floorplan.pipeline.scene.build_scene only accepts a
Stray folder:
    load capture -> correct_drift (default ON) -> keyframes / TSDF / raw points / align_scene (the body of
    build_scene, unchanged) -> plan_beta v2 extract_plan -> [LiDAR bias correction D-021, apply_to_plan] ->
    tier_budget.widen_plan(rel_sigma=0) -> plan.json.
Both plans are written from the SAME extraction: plan_corrected.json (default pipeline) and plan_raw.json
(= run_capture --no-bias-correction). Damage (D-023) is skipped: it does not change walls, rooms or heights.
Then the Faro scan is registered into the plan's aligned frame with scripts/validate_arkitscenes.py's own
register_laser (levelling + Manhattan + 4 yaws + FFT + point-to-plane ICP); the registered laser is cached.

Stage `measure` (per room) measures ground truth on the laser for every plan item:
  * each plan wall line: the laser wall position, fitted with the validation's robust trimmed plane fit on laser
    points whose normal is parallel to the wall, within +-12 cm of the plan line, inside the wall's span (ends trimmed
    by 15 cm), 0.3-2.0 m above the floor;
  * GT corners = intersections of consecutive GT wall lines, so GT wall length = corner-to-corner length, the same
    definition as the plan's; GT room box = extent of the GT polygon along x and z;
  * GT ceiling height = laser ceiling plane - laser floor plane, both fitted on laser points inside the plan room
    polygon (eroded 15 cm), evaluated at the room centroid.
Stage `summary` pools the rooms: per-item error tables, median |error|, share within 1 / 1.5 / 2 cm, coverage of the
95% intervals, with and without the bias correction, development and hold-out separately; figures.

Usage (regenerate everything):
  python scripts/bench_arkitscenes.py build   --room 47895909 [--no-drift]
  python scripts/bench_arkitscenes.py measure --room 47895909 [--no-drift]
  python scripts/bench_arkitscenes.py summary
  python scripts/bench_arkitscenes.py all            # build + measure every room (sequential), then summary
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorplan.config import Config  # noqa: E402
from floorplan.io.arkitscenes import load_arkitscenes  # noqa: E402
from floorplan.plan.align import align_scene  # noqa: E402
from floorplan.pipeline.scene import save_scene, load_scene  # noqa: E402
from floorplan.recon.fusion import collect_raw_points, fuse_tsdf  # noqa: E402
from floorplan.recon.keyframes import select_keyframes  # noqa: E402
from floorplan.uncertainty.tier_budget import widen_plan  # noqa: E402

_spec = importlib.util.spec_from_file_location("validate_arkitscenes", ROOT / "scripts" / "validate_arkitscenes.py")
V = importlib.util.module_from_spec(_spec)
sys.modules["validate_arkitscenes"] = V
_spec.loader.exec_module(V)

DATA = ROOT.parent / "data" / "arkitscenes"
OUT = ROOT / "outputs" / "benchmark" / "arkitscenes"
ROOMS = {   # video_id: (fold, visit_id, laser scan id, role)
    "47895909": ("Training", "482587", "191738", "dev"),
    "42445884": ("Training", "422009", "172562", "dev"),
    "47331133": ("Training", "470348", "188580", "dev"),
    "47333561": ("Training", "469650", "190248", "holdout"),
    "47430003": ("Validation", "470537", "203985", "holdout"),
}


def room_dir(room: str, drift: bool) -> Path:
    return OUT / (room if drift else f"{room}_nodrift")


# ------------------------------------------------------------------------------------------------ build
def build_scene_from_capture(cap, cfg: Config, T_wc: np.ndarray, log):
    """The body of floorplan.pipeline.scene.build_scene, unchanged, for an already-loaded capture object."""
    t0 = time.time()
    kf = select_keyframes(cap, cfg, T_wc)
    log(f"[scene] {cap.capture_id}: {cap.n} frames, {len(kf)} keyframes ({time.time()-t0:.1f}s)")
    fused = fuse_tsdf(cap, kf, T_wc, cfg, with_color=True)
    log(f"[scene] TSDF: {len(fused['points'])} surface points from {fused['frames_used']} frames ({time.time()-t0:.1f}s)")
    raw = collect_raw_points(cap, kf, T_wc, cfg)
    log(f"[scene] raw LiDAR points kept: {len(raw['points'])} ({time.time()-t0:.1f}s)")
    T_align, P, N, info = align_scene(fused["points"], fused["normals"], T_wc[:, :3, 3], cfg)
    R, t = T_align[:3, :3], T_align[:3, 3]
    scene = dict(
        points=P.astype(np.float32), normals=N.astype(np.float32), colors=fused["colors"],
        raw_points=(raw["points"] @ R.T + t).astype(np.float32), raw_range=raw["range"],
        raw_ray=(raw["ray"].astype(np.float32) @ R.T).astype(np.float16), raw_frame=raw["frame"],
        T_align=T_align, traj=(T_wc[:, :3, 3] @ R.T + t).astype(np.float32), kf=kf,
        T_wc=T_wc, timestamps=cap.timestamps)
    info.update(capture_id=cap.capture_id, frames=cap.n, keyframes=int(len(kf)),
                surface_points=int(len(P)), raw_points=int(len(raw["points"])),
                runtime_s=round(time.time() - t0, 1), config=cfg.to_dict())
    return scene, info


def build(room: str, drift: bool = True):
    from floorplan.export.json_export import save_plan_json
    from floorplan.export.render import render_plan
    from floorplan.plan.beta import extract_plan
    from floorplan.uncertainty.bias import LidarBiasConfig, apply_to_plan
    fold, visit, scan, role = ROOMS[room]
    out = room_dir(room, drift)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(None)
    t0 = time.time()
    log = lambda m: print(f"[{room} {time.time() - t0:6.1f}s] {m}", flush=True)  # noqa: E731
    timing, report = {}, dict(room=room, role=role, drift=drift, config=cfg.to_dict())

    cap = load_arkitscenes(DATA / "raw" / fold / room)
    report["capture_meta"] = cap.meta
    T_wc = cap.T_wc
    if drift:
        from floorplan.recon.drift import correct_drift
        t1 = time.time()
        T_wc, report["drift"] = correct_drift(cap, select_keyframes(cap, cfg), cap.T_wc, cfg, log=log)
        timing["drift_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    scene, info = build_scene_from_capture(cap, cfg, T_wc, log)
    timing["scene_s"] = round(time.time() - t1, 1)
    save_scene(scene, info, out / "scene")

    t1 = time.time()
    plan = extract_plan(scene, info, room)
    plan.tier = "lidar"
    timing["extract_s"] = round(time.time() - t1, 1)
    plans = {"raw": copy.deepcopy(plan), "corrected": plan}
    apply_to_plan(plans["corrected"], LidarBiasConfig.load())          # D-021, exactly as run_capture
    for k, p in plans.items():
        widen_plan(p, 0.0, "LiDAR depth is metric; no scale term")
        report[f"schema_problems_{k}"] = save_plan_json(p, out / f"plan_{k}.json")
    render_plan(plans["corrected"], out / "plan")
    timing["pipeline_total_s"] = round(time.time() - t0, 1)
    log(f"plan: {len(plan.rooms)} rooms, {len(plan.walls)} walls, {len(plan.openings)} openings")

    # ---- laser ground truth, registered into the plan's aligned frame (validation code, unchanged)
    t1 = time.time()
    laser = V.load_laser([DATA / "laser_scanner_point_clouds" / visit / f"{scan}.ply"])
    timing["laser_load_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    sc = dict(points=scene["points"].astype(np.float64), normals=scene["normals"].astype(np.float64))
    T, diag = V.register_laser(sc, info, laser, cfg)
    timing["registration_s"] = round(time.time() - t1, 1)
    LP, LN = laser["eval"]
    LP, LN = V.transform(T, LP), LN @ T[:3, :3].T
    lo, hi = scene["raw_points"].min(0) - 0.5, scene["raw_points"].max(0) + 0.5
    keep = np.all((LP > lo) & (LP < hi), axis=1)
    np.savez_compressed(out / "laser_registered.npz", P=LP[keep].astype(np.float32), N=LN[keep].astype(np.float32),
                        T_laser_to_plan=T)
    report["registration"] = diag
    report["laser"] = dict(n_points_file=laser["n_points_file"], n_points_read=laser["n_points_read"],
                           eval_points_kept=int(keep.sum()), eval_spacing_median_mm=laser["eval_spacing_median_mm"])
    report["timing"] = timing
    (out / "build_report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"registration {diag}; done")


# ------------------------------------------------------------------------------------------------ diagnostic
def diag_relaxed(room: str, min_enclosure: float = 0.4):
    """DIAGNOSTIC ONLY (not the pipeline): re-extract the saved scene with a relaxed 'never-entered room' rule, to see
    what the extractor would measure in a room the camera only looked into. Plans go to <room>_diag_unvisited<x>/."""
    import os
    from floorplan.export.json_export import save_plan_json
    from floorplan.plan.beta import extract_plan
    from floorplan.plan.beta.params import BetaParams
    from floorplan.uncertainty.bias import LidarBiasConfig, apply_to_plan
    src = room_dir(room, True)
    out = OUT / f"{room}_diag_unvisited{min_enclosure}"
    out.mkdir(parents=True, exist_ok=True)
    scene, info = load_scene(src / "scene")
    plan = extract_plan(scene, info, room, BetaParams(unvisited_min_enclosure=min_enclosure))
    plan.tier = "lidar"
    plans = {"raw": copy.deepcopy(plan), "corrected": plan}
    apply_to_plan(plans["corrected"], LidarBiasConfig.load())
    for k, p in plans.items():
        widen_plan(p, 0.0, "LiDAR depth is metric; no scale term")
        save_plan_json(p, out / f"plan_{k}.json")
    if not (out / "laser_registered.npz").exists():
        os.symlink(src / "laser_registered.npz", out / "laser_registered.npz")
    print(f"{room} diag: {len(plan.rooms)} rooms, {len(plan.walls)} walls")
    measure(room, True, out)


# ------------------------------------------------------------------------------------------------ measure
WIN_M = 0.12            # laser wall searched within +-12 cm of the plan wall line
END_TRIM_M = 0.15       # leave the 15 cm next to each corner out (corner rounding, skirting returns)
TAPE_BAND_M = (0.8, 1.6)     # where a tape is held (same band the plan uses for wall positions, V2-D5)
FALLBACK_BAND_M = (0.3, 2.0)
MIN_WALL_PTS = 150
ERODE_M = 0.15          # floor / ceiling GT measured inside the room polygon shrunk by 15 cm
TOLS_CM = (1.0, 1.5, 2.0)


def robust_offset(o: np.ndarray) -> tuple[float, float, int] | None:
    """1-D robust position of a planar surface: the histogram peak (1 cm bins, 3-bin smoothing) nearest to the plan
    line among peaks with >= 30% of the best peak's support, then the validation's trimmed-mean schedule."""
    if len(o) < MIN_WALL_PTS:
        return None
    edges = np.arange(-WIN_M, WIN_M + 1e-9, 0.01)
    h, _ = np.histogram(o, edges)
    hs = np.convolve(h, np.ones(3) / 3, "same")
    centres = 0.5 * (edges[1:] + edges[:-1])
    peaks = [i for i in range(len(hs)) if hs[i] >= 0.3 * hs.max() and hs[i] >= hs[max(i - 1, 0)] and hs[i] >= hs[min(i + 1, len(hs) - 1)]]
    c = centres[min(peaks, key=lambda i: abs(centres[i]))]
    for thr in V.TRIM_SCHEDULE_M:
        sel = o[np.abs(o - c) < thr]
        if len(sel) < MIN_WALL_PTS // 3:
            return None
        c = float(np.mean(sel))
    rmse = float(np.sqrt(np.mean((sel - c) ** 2)))
    return c, rmse, int(len(sel))


def laser_wall(w: dict, LP: np.ndarray, LN: np.ndarray, floor_y: float) -> dict:
    """Laser position of the wall under plan wall w, as an offset (m) along w's inward normal from the plan line."""
    p0, p1, n = np.array(w["p0"]), np.array(w["p1"]), np.array(w["normal"], float)
    L = float(np.linalg.norm(p1 - p0))
    d = (p1 - p0) / max(L, 1e-9)
    trim = min(END_TRIM_M, 0.25 * L)
    Q = LP[:, [0, 2]] - p0
    s, o = Q @ d, Q @ n
    nn = LN[:, [0, 2]] @ n
    base = (s > trim) & (s < L - trim) & (np.abs(o) < WIN_M) & (nn > 0.8)   # laser normals face the scanner (room)
    for band in (TAPE_BAND_M, FALLBACK_BAND_M):
        m = base & (LP[:, 1] > floor_y + band[0]) & (LP[:, 1] < floor_y + band[1])
        r = robust_offset(o[m])
        if r is not None:
            covered = np.unique(np.floor(s[m] / 0.05)).size * 0.05 / max(L - 2 * trim, 0.05)
            return dict(offset_m=r[0], rmse_mm=round(1000 * r[1], 2), n=r[2], band=list(band),
                        span_covered=round(min(covered, 1.0), 2), plan_len_m=L)
    return dict(offset_m=None, n=int(base.sum()), plan_len_m=L)


def intersect(n1, c1, n2, c2):
    A = np.array([n1, n2], float)
    if abs(np.linalg.det(A)) < 0.5:
        return None
    return np.linalg.solve(A, np.array([c1, c2], float))


def point_in_poly(X: np.ndarray, poly: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path as MPath
    return MPath(poly).contains_points(X)


def dist_to_boundary(X: np.ndarray, poly: np.ndarray) -> np.ndarray:
    d = np.full(len(X), np.inf)
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        ab = b - a
        t = np.clip(((X - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(X - (a + t[:, None] * ab), axis=1))
    return d


def trimmed_plane_y(P: np.ndarray, seed_y: float) -> tuple[float, float, int] | None:
    """Horizontal surface near seed_y: plane y = a x + b z + c, trimmed LS (validation schedule)."""
    sel = P[np.abs(P[:, 1] - seed_y) < 0.05]
    for thr in V.TRIM_SCHEDULE_M:
        if len(sel) < 200:
            return None
        A = np.c_[sel[:, 0], sel[:, 2], np.ones(len(sel))]
        coef = np.linalg.lstsq(A, sel[:, 1], rcond=None)[0]
        res = P[:, 1] - np.c_[P[:, 0], P[:, 2], np.ones(len(P))] @ coef
        sel = P[np.abs(res) < thr]
    A = np.c_[sel[:, 0], sel[:, 2], np.ones(len(sel))]
    coef = np.linalg.lstsq(A, sel[:, 1], rcond=None)[0]
    return coef, float(np.degrees(np.arctan(np.hypot(coef[0], coef[1])))), int(len(sel))


def dominant_level(y: np.ndarray, lo: float, hi: float) -> float | None:
    y = y[(y > lo) & (y < hi)]
    if len(y) < 200:
        return None
    h, e = np.histogram(y, np.arange(lo, hi + 0.01, 0.01))
    return float(0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]))


def laser_heights(room: dict, LP: np.ndarray, LN: np.ndarray) -> dict:
    poly = np.array(room["polygon"], float)
    X = LP[:, [0, 2]]
    inside = point_in_poly(X, poly)
    inside[inside] &= dist_to_boundary(X[inside], poly) > ERODE_M
    fl = room["floor_level"]
    F = LP[inside & (LN[:, 1] > 0.9)]
    C = LP[inside & (LN[:, 1] < -0.9)]
    yf = dominant_level(F[:, 1], fl - 0.15, fl + 0.15)
    yc = dominant_level(C[:, 1], fl + 1.8, fl + 3.5)
    if yf is None or yc is None:
        return dict(height_m=None, n_floor=len(F), n_ceiling=len(C))
    lf, lc = level_trimmed(F[:, 1], yf), level_trimmed(C[:, 1], yc)
    if lf is None or lc is None:
        return dict(height_m=None, n_floor=len(F), n_ceiling=len(C))
    pf, pc = trimmed_plane_y(F, yf), trimmed_plane_y(C, yc)      # diagnostics only (tilt of the laser levels)
    return dict(height_m=lc[0] - lf[0], floor_y_laser=round(lf[0], 4), ceiling_y_laser=round(lc[0], 4),
                n_floor=lf[1], n_ceiling=lc[1], ceiling_share_on_level=round(lc[1] / max(len(C), 1), 3),
                floor_tilt_deg=None if pf is None else round(pf[1], 3),
                ceiling_tilt_deg=None if pc is None else round(pc[1], 3))


def level_trimmed(y: np.ndarray, seed: float):
    """Horizontal level (the frame is gravity-levelled): dominant 1 cm peak, then the validation's trimmed-mean schedule.
    Same idea as the plan's own ceiling measure (peak + trimmed mean)."""
    c = seed
    for thr in V.TRIM_SCHEDULE_M:
        sel = y[np.abs(y - c) < thr]
        if len(sel) < 200:
            return None
        c = float(np.mean(sel))
    return c, int(len(sel))


def _m(meas: dict | None):
    if meas is None or meas.get("value") is None:
        return None, None, None
    ci = meas.get("ci95") or [None, None]
    return meas["value"], ci[0], ci[1]


def measure(room: str, drift: bool = True, out: Path | None = None):
    out = out or room_dir(room, drift)
    z = np.load(out / "laser_registered.npz")
    LP, LN = z["P"].astype(np.float64), z["N"].astype(np.float64)
    plans = {k: json.loads((out / f"plan_{k}.json").read_text()) for k in ("raw", "corrected")}
    pr = plans["raw"]
    items, gt_rooms = [], []
    walls_by_id = {w["id"]: w for w in pr["walls"]}
    for r in pr["rooms"]:
        ws = [walls_by_id[i] for i in r["wall_ids"]]
        lw = [laser_wall(w, LP, LN, r["floor_level"]) for w in ws]
        # GT line of wall k: n . q = c with c = n . p0 + laser offset (n = inward normal)
        lines = [(np.array(w["normal"], float), None if g["offset_m"] is None else float(np.dot(w["normal"], w["p0"]) + g["offset_m"]))
                 for w, g in zip(ws, lw)]
        nw = len(ws)
        corners = []
        for k in range(nw):            # corner k = start of wall k = intersection of lines k-1 and k
            (n1, c1), (n2, c2) = lines[k - 1], lines[k]
            corners.append(None if c1 is None or c2 is None else intersect(n1, c1, n2, c2))
        gt_poly = None if any(c is None for c in corners) else np.array(corners)
        for k, (w, g) in enumerate(zip(ws, lw)):
            a, b = corners[k], corners[(k + 1) % nw]
            gt = None if a is None or b is None else float(np.linalg.norm(b - a))
            items.append(dict(room=room, kind="wall_length", item=w["id"], gt_m=gt, status=w["length"].get("status"),
                              laser_offset_cm=None if g["offset_m"] is None else round(100 * g["offset_m"], 2),
                              laser_rmse_mm=g.get("rmse_mm"), laser_span_covered=g.get("span_covered"),
                              laser_band=g.get("band")))
        poly = np.array(r["polygon"], float)
        ext_plan = poly.max(0) - poly.min(0)
        dims = r.get("bbox_dims") or {}
        # GT room box: per axis, the laser positions of the plan walls that form the polygon's two extremes
        lo_xy, hi_xy = poly.min(0), poly.max(0)
        order = np.argsort(-ext_plan)              # plan's length = larger plan extent (measure.py)
        for name, ax in zip(("length", "width"), order):
            side = {}
            for sgn, val in ((+1, lo_xy[ax]), (-1, hi_xy[ax])):   # min side: inward normal +axis; max side: -axis
                cands = [(w, g) for w, g in zip(ws, lw) if abs(w["normal"][ax] - sgn) < 1e-6
                         and abs(w["p0"][ax] - val) < 1e-4 and abs(w["p1"][ax] - val) < 1e-4]
                cands = [c for c in cands if c[1]["offset_m"] is not None]
                if cands:
                    w, g = max(cands, key=lambda c: c[1]["plan_len_m"])
                    side[sgn] = (w["id"], val + sgn * g["offset_m"])
            gt = side[-1][1] - side[+1][1] if len(side) == 2 else None
            items.append(dict(room=room, kind="room_dim", item=f"{r['id']}-bbox_{name}({'xz'[ax]})", gt_m=gt,
                              gt_walls=[side[k][0] for k in side]))
        if gt_poly is not None:
            from shapely.geometry import Polygon
            gp = Polygon(gt_poly)
            items.append(dict(room=room, kind="area", item=f"{r['id']}-floor_area", gt_m=float(gp.area), gt_valid=bool(gp.is_valid)))
            items.append(dict(room=room, kind="perimeter", item=f"{r['id']}-perimeter", gt_m=float(gp.length)))
        h = laser_heights(r, LP, LN)
        items.append(dict(room=room, kind="ceiling_height", item=f"{r['id']}-ceiling", gt_m=h.get("height_m"), **{k: v for k, v in h.items() if k != "height_m"}))
        gt_rooms.append(dict(room_id=r["id"], gt_polygon=None if gt_poly is None else gt_poly.round(4).tolist(),
                             plan_polygon=r["polygon"], laser_walls=lw))
    # attach plan values (raw and corrected) to every item
    for it in items:
        for k, p in plans.items():
            rid = it["item"].split("-")[0]
            rm = next((x for x in p["rooms"] if x["id"] == rid), None)
            if it["kind"] == "wall_length":
                meas = next(w for w in p["walls"] if w["id"] == it["item"])["length"]
            elif it["kind"] == "room_dim":
                meas = rm["bbox_dims"]["length" if "bbox_length" in it["item"] else "width"]
            elif it["kind"] == "area":
                meas = rm["floor_area"]
            elif it["kind"] == "perimeter":
                meas = rm["perimeter"]
            else:
                meas = rm["ceiling_height"]
            v, lo, hi = _m(meas)
            it[f"{k}_m"], it[f"{k}_lo"], it[f"{k}_hi"] = v, lo, hi
            if v is not None and it["gt_m"] is not None:
                it[f"{k}_err_cm"] = round(100 * (v - it["gt_m"]), 2)
                it[f"{k}_covered"] = bool(lo <= it["gt_m"] <= hi) if lo is not None else None
                it[f"{k}_ci_halfwidth_cm"] = round(50 * (hi - lo), 2) if lo is not None else None
    res = dict(room=room, role=ROOMS[room][3], drift=drift, n_rooms=len(pr["rooms"]), n_walls=len(pr["walls"]),
               n_openings=len(pr["openings"]), coverage_warning=pr["meta"].get("coverage_warning"),
               dropped_regions=pr["meta"].get("dropped_regions"), items=items, gt_rooms=gt_rooms,
               plan_footprint_m2=(pr.get("footprint_area") or {}).get("value"))
    res["laser_footprint"] = laser_floor_outline(LP, LN, pr)
    (out / "accuracy.json").write_text(json.dumps(res, indent=1, default=float))
    plot_room(room, out, LP, LN, pr, gt_rooms, items)
    for it in items:
        print(room, it["kind"], it["item"], "gt", None if it["gt_m"] is None else round(it["gt_m"], 4),
              "raw", it.get("raw_err_cm"), "corr", it.get("corrected_err_cm"), "cov", it.get("corrected_covered"))
    return res


def laser_floor_outline(LP, LN, plan) -> dict:
    """Free floor the laser sees around the floor level (5 cm grid): a structural reference for how much of the
    room the plan covers. Floor points within 5 cm of the plan's floor level, up-facing."""
    fl = plan["meta"].get("floor_y")
    if fl is None:
        return {}
    m = (LN[:, 1] > 0.9) & (np.abs(LP[:, 1] - fl) < 0.05)
    cells = np.unique(np.floor(LP[m][:, [0, 2]] / 0.05).astype(int), axis=0)
    covered = 0
    for r in plan["rooms"]:
        covered += int(point_in_poly((cells + 0.5) * 0.05, np.array(r["polygon"])).sum())
    return dict(laser_floor_m2=round(len(cells) * 0.0025, 2), laser_floor_in_plan_m2=round(covered * 0.0025, 2))


def plot_room(room, out, LP, LN, plan, gt_rooms, items):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fl = plan["meta"].get("floor_y") or 0.0
    fig, ax = plt.subplots(figsize=(7, 7))
    m = (np.abs(LN[:, 1]) < 0.3) & (LP[:, 1] > fl + 0.8) & (LP[:, 1] < fl + 1.6)
    S = LP[m][::3]
    ax.scatter(S[:, 0], S[:, 2], s=0.2, c="#9aa0a6", label="laser walls (0.8-1.6 m)", rasterized=True)
    fm = (LN[:, 1] > 0.9) & (np.abs(LP[:, 1] - fl) < 0.05)
    F = LP[fm][::20]
    ax.scatter(F[:, 0], F[:, 2], s=0.2, c="#d9e6f2", label="laser floor", rasterized=True, zorder=0)
    for g in gt_rooms:
        P = np.array(g["plan_polygon"] + [g["plan_polygon"][0]])
        ax.plot(P[:, 0], P[:, 2] if P.shape[1] > 2 else P[:, 1], "-", c="#2a78d6", lw=1.6, label="plan (raw)")
        if g["gt_polygon"]:
            G = np.array(g["gt_polygon"] + [g["gt_polygon"][0]])
            ax.plot(G[:, 0], G[:, 1], "--", c="#eb6834", lw=1.2, label="laser GT on plan's walls")
    for w in plan["walls"]:
        it = next(i for i in items if i["item"] == w["id"])
        c = 0.5 * (np.array(w["p0"]) + np.array(w["p1"])) + 0.12 * np.array(w["normal"])
        e = it.get("corrected_err_cm")
        ax.text(c[0], c[1], f"{w['id'].split('-')[1]}\n{'n/a' if e is None else f'{e:+.1f}'}", fontsize=7,
                ha="center", va="center", color="#333")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    h, l = ax.get_legend_handles_labels()
    u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), loc="upper right", fontsize=7, markerscale=10)
    ax.set_title(f"{room} ({ROOMS[room][3]}): plan vs laser; wall labels = corrected length error (cm)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "overlay.png", dpi=130)
    plt.close(fig)

# ------------------------------------------------------------------------------------------------ summary
KINDS = ("wall_length", "room_dim", "ceiling_height", "area", "perimeter")
C_RAW, C_COR = "#2a78d6", "#eb6834"      # validated categorical slots 1-2 (same as validate_arkitscenes.py)


def stats(rows: list[dict], var: str) -> dict:
    e = np.array([r[f"{var}_err_cm"] for r in rows if r.get(f"{var}_err_cm") is not None], float)
    cov = [r[f"{var}_covered"] for r in rows if r.get(f"{var}_covered") is not None]
    hw = [r[f"{var}_ci_halfwidth_cm"] for r in rows if r.get(f"{var}_ci_halfwidth_cm") is not None]
    if len(e) == 0:
        return dict(n=0)
    out = dict(n=int(len(e)), median_abs_cm=round(float(np.median(np.abs(e))), 2), mean_cm=round(float(e.mean()), 2),
               max_abs_cm=round(float(np.abs(e).max()), 2))
    for t in TOLS_CM:
        out[f"within_{t}cm"] = round(float(np.mean(np.abs(e) <= t)), 2)
    out["coverage95"] = round(float(np.mean(cov)), 2) if cov else None
    out["median_ci_halfwidth_cm"] = round(float(np.median(hw)), 2) if hw else None
    return out


B_M = 0.0068   # the pre-registered inward bias (LidarBiasConfig.inward_bias_m)


def corner_signs(plan_json: Path) -> dict:
    """Per wall: (+1/-1, +1/-1) = (start corner, end corner) convex / reflex. Moving every wall outward by b changes a
    wall's corner-to-corner length by +b at a convex end and -b at a reflex end (so +2b only for convex-convex walls)."""
    p = json.loads(plan_json.read_text())
    out = {}
    for r in p["rooms"]:
        V = np.array(r["polygon"], float)
        area2 = np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1])
        orient = np.sign(area2)
        n = len(V)
        def conv(k):   # corner at vertex k, between edge k-1 and edge k
            a, b = V[k] - V[k - 1], V[(k + 1) % n] - V[k]
            return 1 if np.sign(a[0] * b[1] - a[1] * b[0]) == orient else -1
        for k, wid in enumerate(r["wall_ids"]):
            out[wid] = (conv(k), conv((k + 1) % n))
    return out


def summary():
    import csv
    res = {}
    for drift in (True, False):
        for room in ROOMS:
            f = room_dir(room, drift) / "accuracy.json"
            if f.exists():
                res[(room, drift)] = json.loads(f.read_text())
    rows = []
    for (room, drift), r in res.items():
        conv = corner_signs(room_dir(room, drift) / "plan_raw.json")
        for it in r["items"]:
            row = dict(it, role=ROOMS[room][3], drift=drift)
            if it["kind"] == "wall_length":
                sp, sn = conv[it["item"]]
                row["reflex_ends"] = int((sp < 0) + (sn < 0))
                if it.get("raw_err_cm") is not None:     # PROPOSED fix (not the pipeline): +b per convex end, -b per reflex end
                    row["corner_aware_err_cm"] = round(it["raw_err_cm"] + 100 * B_M * (sp + sn), 2)
            rows.append(row)
    for split, cond in (("wall_length_convex_ends", lambda r: r.get("reflex_ends") == 0),
                        ("wall_length_reflex_end", lambda r: (r.get("reflex_ends") or 0) > 0)):
        for r in rows:
            if r["kind"] == "wall_length" and cond(r):
                r.setdefault("splits", []).append(split)
    cols = ["room", "role", "drift", "kind", "item", "status", "reflex_ends", "corner_aware_err_cm", "gt_m", "raw_m", "raw_lo", "raw_hi", "raw_err_cm", "raw_covered",
            "corrected_m", "corrected_lo", "corrected_hi", "corrected_err_cm", "corrected_covered", "corrected_ci_halfwidth_cm",
            "laser_offset_cm", "laser_span_covered"]
    with open(OUT / "items.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    table = []
    for drift in (True, False):
        for role in ("dev", "holdout", "all"):
            for kind in KINDS + ("lengths_all", "wall_length_convex_ends", "wall_length_reflex_end"):
                sel = [r for r in rows if r["drift"] == drift and (role == "all" or r["role"] == role)
                       and (r["kind"] == kind or kind in r.get("splits", [])
                            or (kind == "lengths_all" and r["kind"] in ("wall_length", "room_dim")))]
                n_plan = len(sel)
                n_nogt = sum(r["gt_m"] is None for r in sel)
                for var in ("raw", "corrected", "corner_aware"):
                    table.append(dict(drift=drift, role=role, kind=kind, correction=var, plan_items=n_plan,
                                      items_without_laser_gt=n_nogt, **stats(sel, var)))
    with open(OUT / "summary.csv", "w", newline="") as fh:
        keys = list(dict.fromkeys(k for t in table for k in t))
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(table)
    rooms = []
    for (room, drift), r in res.items():
        rooms.append(dict(room=room, role=ROOMS[room][3], drift=drift, n_rooms=r["n_rooms"], n_walls=r["n_walls"],
                          n_openings=r["n_openings"], coverage_warning=r["coverage_warning"],
                          walls_without_laser_surface=[g_id for g in r["gt_rooms"] for g_id, lw in
                                                       zip([i["item"] for i in r["items"] if i["kind"] == "wall_length"], g["laser_walls"])
                                                       if lw["offset_m"] is None],
                          ceiling=[{k: i.get(k) for k in ("gt_m", "raw_err_cm", "corrected_err_cm", "corrected_covered",
                                                           "corrected_ci_halfwidth_cm")} for i in r["items"] if i["kind"] == "ceiling_height"]))
    timing = {}
    for room in ROOMS:
        f = room_dir(room, True) / "build_report.json"
        if f.exists():
            timing[room] = json.loads(f.read_text()).get("timing")
    (OUT / "summary.json").write_text(json.dumps(dict(table=table, rooms=rooms, timing=timing), indent=1, default=float))
    plot_errors(rows)
    plot_montage()
    for t in table:
        if t["drift"] and t["role"] == "all" and t["n"] and t["kind"].startswith("wall_length"):
            print({k: v for k, v in t.items() if k != "drift"})


def plot_errors(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = [("wall_length", "wall lengths"), ("room_dim", "room box dims"), ("ceiling_height", "ceiling heights")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (kind, title) in zip(axes, kinds):
        for j, role in enumerate(("dev", "holdout")):
            for k, (var, col) in enumerate((("raw", C_RAW), ("corrected", C_COR))):
                e = [r[f"{var}_err_cm"] for r in rows if r["drift"] and r["kind"] == kind and r["role"] == role
                     and r.get(f"{var}_err_cm") is not None]
                x = j * 2.4 + k + rng.uniform(-0.15, 0.15, len(e))
                ax.scatter(x, e, s=28, color=col, edgecolor="white", linewidth=1, zorder=3,
                           label=("raw (--no-bias-correction)" if var == "raw" else "default (bias-corrected)") if j == 0 else None)
        for t, ls in ((1.0, ":"), (1.5, "--"), (2.0, "-")):
            for sgn in (1, -1):
                ax.axhline(sgn * t, color="#bbb", lw=0.8, ls=ls, zorder=1)
        ax.axhline(0, color="#666", lw=0.8)
        ax.set_xticks([0.5, 2.9], ["dev rooms (3)", "hold-out rooms (2*)"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-12, 12)
    axes[0].set_ylabel("plan - laser (cm); lines at +-1 / 1.5 / 2 cm")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("LiDAR plan vs Faro laser, drift correction ON (* hold-out 47430003 produced no room)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "errors_by_kind.png", dpi=130)
    plt.close(fig)


def plot_montage():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ims = [(room, room_dir(room, True) / "overlay.png") for room in ROOMS]
    ims.append(("47430003 diag", OUT / "47430003_diag_unvisited0.4" / "overlay.png"))
    ims = [(n, p) for n, p in ims if p.exists()]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (n, p) in zip(axes.ravel(), ims):
        ax.imshow(plt.imread(p))
    fig.tight_layout()
    fig.savefig(OUT / "overlays_all_rooms.png", dpi=110)
    plt.close(fig)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "measure", "diag", "summary", "all"])
    ap.add_argument("--room", choices=list(ROOMS))
    ap.add_argument("--no-drift", action="store_true")
    a = ap.parse_args()
    if a.stage == "build":
        build(a.room, not a.no_drift)
    elif a.stage == "measure":
        measure(a.room, not a.no_drift)
    elif a.stage == "diag":
        diag_relaxed(a.room)
    elif a.stage == "summary":
        summary()
    elif a.stage == "all":
        for room in ROOMS:
            for drift in (True, False):
                build(room, drift)
                measure(room, drift)
        diag_relaxed("47430003")
        summary()
