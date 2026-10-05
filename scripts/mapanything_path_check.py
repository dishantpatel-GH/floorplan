#!/usr/bin/env python
"""Measure camera paths of a finished video run without ground truth: MapAnything windows (path_mapanything.py)
against the run's own DPVO path (scaled, before and after the pose graph) and, if given, an SfM path.

  solve   (GPU, one process): MapAnything windows for every variant, cached in <out>/windows/
  measure (CPU): chain each variant and score every path on the same numbers:
      * revisits: places filmed twice, true distance from SIFT + MoGe-2 depth + PnP (--revisits json: i, j, dist_m)
      * local steps: the path's step against the PnP step for keyframe pairs 2-6 apart (--steps json: i, j,
        base_pnp_m, ok), as a ratio per stretch of the walk (1.0 = the path has the depth's scale there)
      * too-fast steps (> 1.5 m/s between keyframes) and path length
      * floor: gravity from the run's GeoCalib up vectors; each keyframe's camera height above the floor it sees in
        its own depth map (no path involved) against its height in the path: the spread of the floor level (p10-p90)
        is the path's vertical drift, the median camera height its metric scale (MoGe frames say 1.37 m on take1)

    env -u PYTHONPATH python scripts/mapanything_path_check.py solve <run dir> <out dir> [--variants a,b]
    env -u PYTHONPATH python scripts/mapanything_path_check.py measure <run dir> <out dir> [--revisits ..] [--steps ..]
        [--sfm path.npz] [--arkit <capture dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.video import depth as depth_mod                    # noqa: E402
from floorplan.video import path_mapanything as PM                 # noqa: E402

# name: (window size, overlap, MoGe-2 depth as input, chaining mode, Sim3, average the windows' estimates)
VARIANTS = {
    "w24o8_depth": (24, 8, True, "points", False, False),
    "w24o8_depth_sim3": (24, 8, True, "points", True, False),
    "w24o8_depth_avg": (24, 8, True, "points", False, True),
    "w24o8_nodepth": (24, 8, False, "points", False, False),
    "w16o4_depth": (16, 4, True, "points", False, False),
    "w32o8_depth": (32, 8, True, "points", False, False),
    "w24o16_depth": (24, 16, True, "points", False, False),
    "w24o16_depth_avg": (24, 16, True, "points", False, True),
    "w24o16_depth_sim3_avg": (24, 16, True, "points", True, True),
    "w24o8_depth_anchored": (24, 8, True, "anchored", False, False),
    "w24o8_depth_smooth1": (24, 8, True, "points", False, False, 1.0),
    "w24o8_depth_smooth2": (24, 8, True, "points", False, False, 2.0),
    "w24o16_depth_avg_smooth1": (24, 16, True, "points", False, True, 1.0),
    "w32o8_nodepth_anchored": (32, 8, False, "anchored", False, False),
}


def run_inputs(run: Path) -> dict:
    info = json.loads((run / "scene/scene_info.json").read_text())
    rot = info["rotation"]["final_rotation"]
    files = sorted((run / f"work/frames_r{rot}/sfm").glob("*.jpg"))
    calib = json.loads((run / "work/calib.json").read_text())
    z = np.load(run / "work/depth_moge2.npz")
    sc = np.load(run / "scene/scene.npz")
    kf = sc["kf"]
    n = min(len(files), len(z["depth"]), len(kf))
    return dict(info=info, files=files[:n], f=float(calib["f"]), depth_raw=z["depth"][:n].astype(np.float32),
                K_d=z["K"], kf=kf[:n], ts=sc["timestamps"], scene=sc, rot=rot)


def solve(run: Path, out: Path, variants: list[str]) -> None:
    import torch
    inp = run_inputs(run)
    work = out / "windows"
    work.mkdir(parents=True, exist_ok=True)
    model = None
    for v in variants:
        size, ov, use_d, mode = VARIANTS[v][:4]
        cache = work / f"ma_windows_{PM._key(inp['files'], inp['f'], use_d, size, ov, mode)}.npz"
        if cache.exists():
            print(f"[{v}] cached", flush=True)
            continue
        if model is None:
            model = PM.load_model()
        rgbs, Ks, Ds = PM.prepare_inputs(inp["files"], inp["f"], inp["depth_raw"] if use_d else None)
        import time
        t0 = time.time()
        solve_fn = PM.solve_anchored if mode == "anchored" else PM.solve_windows
        wins = solve_fn(model, rgbs, Ks, Ds, PM.window_bounds(len(rgbs), size, ov), log=lambda s: print(s, flush=True))
        PM.save_windows(cache, wins, dict(seconds=round(time.time() - t0, 1), size=size, overlap=ov, mode=mode,
                                          depth_input=use_d, peak_gb=torch.cuda.max_memory_allocated() / 1e9))
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{v}] {len(wins)} windows in {time.time() - t0:.0f} s, peak {peak:.2f} GB", flush=True)


def chain(run: Path, out: Path, v: str, inp: dict) -> dict | None:
    size, ov, use_d, mode, sim3, avg = VARIANTS[v][:6]
    smooth = VARIANTS[v][6] if len(VARIANTS[v]) > 6 else 0.0
    cache = out / "windows" / f"ma_windows_{PM._key(inp['files'], inp['f'], use_d, size, ov, mode)}.npz"
    if not cache.exists():
        return None
    wins, extra = PM.load_windows(cache)
    res = PM.chain_anchored(wins) if mode == "anchored" else PM.chain_windows(wins, with_scale=sim3, average=avg)
    res["gpu_s"] = extra.get("seconds")
    res["T"] = PM.smooth_path(res["T"], res["segment"], smooth)
    return res


def up_vector(run: Path, T: np.ndarray, inp: dict) -> np.ndarray:
    from floorplan.video.evaluate import camera_axes_after_rotation
    from floorplan.video.frontend import _robust_mean_direction
    g = np.load(run / "work/geocalib.npz")
    idx = np.minimum(g["index"], len(T) - 1)
    ok = (np.abs(g["pitch_deg"]) < 45) & (g["index"] < len(T)) & np.isfinite(T[idx, 0, 0])
    uc = g["up_cam"][ok]
    if inp["info"]["rotation"].get("flipped_by_pitch_check"):
        uc = -(uc @ camera_axes_after_rotation(180))
    ups = np.einsum("nij,nj->ni", T[g["index"][ok], :3, :3], uc)
    up, _ = _robust_mean_direction(ups)
    return up


def level(T: np.ndarray, up: np.ndarray) -> np.ndarray:
    from floorplan.plan.align import _rot_between
    R = np.eye(4)
    R[:3, :3] = _rot_between(up, np.array([0.0, 1.0, 0.0]))
    return np.einsum("ij,njk->nik", R, T)


STRETCHES = [(0, 8), (8, 34.5), (34.5, 44), (44, 57), (57, 63), (63, 70), (70, 76), (76, 91), (91, 101), (101, 109),
             (109, 118)]


def measure_path(name: str, T: np.ndarray, seg: np.ndarray | None, inp: dict, run: Path, revisits: list,
                 steps: list, D: np.ndarray) -> dict:
    from floorplan.video.frontend import floor_heights
    t = inp["ts"][inp["kf"]][:len(T)]
    have = np.isfinite(T[:, 0, 0])
    C = T[:, :3, 3]
    seg = np.zeros(len(T), int) if seg is None else seg
    out = dict(path=name, keyframes_with_pose=int(have.sum()), segments=int(len(np.unique(seg[have]))))
    rv = []
    for r in revisits:
        i, j = r["i"], r["j"]
        if j >= len(T) or not (have[i] and have[j]):
            rv.append(dict(pair=f"kf{i}/kf{j}", true_m=round(r["dist_m"], 2), path_m=None))
            continue
        rv.append(dict(pair=f"kf{i}/kf{j}", true_m=round(r["dist_m"], 2),
                       path_m=round(float(np.linalg.norm(C[i] - C[j])), 2), same_segment=bool(seg[i] == seg[j])))
    out["revisits"] = rv
    ok = have[:-1] & have[1:]
    st = np.linalg.norm(np.diff(C, axis=0), axis=1)
    dt = np.maximum(np.diff(t), 1e-3)
    out["path_m"] = round(float(st[ok].sum()), 1)
    out["too_fast_steps"] = int(np.sum(ok & (st / dt > 1.5)))
    rat, tm = [], []
    for s in steps:
        i, j = s["i"], s["j"]
        if not s.get("ok") or s["base_pnp_m"] < 0.08 or j >= len(T) or not (have[i] and have[j]) or seg[i] != seg[j]:
            continue
        rat.append(np.linalg.norm(C[j] - C[i]) / s["base_pnp_m"])
        tm.append(s["t"])
    rat, tm = np.array(rat), np.array(tm)
    out["step_ratio"] = dict(pairs=int(len(rat)), median=round(float(np.median(rat)), 3) if len(rat) else None,
                             p10=round(float(np.percentile(rat, 10)), 3) if len(rat) else None,
                             p90=round(float(np.percentile(rat, 90)), 3) if len(rat) else None,
                             within_20pct=round(float(np.mean(np.abs(np.log(rat)) < np.log(1.2))), 3)
                             if len(rat) else None)
    out["step_ratio_by_stretch"] = {f"{a}-{b}s": (round(float(np.median(rat[(tm >= a) & (tm < b)])), 2)
                                                 if np.sum((tm >= a) & (tm < b)) >= 3 else None) for a, b in STRETCHES}
    # the front end's self-check residual (neighbouring keyframes' depth maps under this path; trusted <= 0.058)
    from floorplan.video.frontend import segment_quality
    from floorplan.video.params import VideoParams
    Tq = T.copy()
    Tq[~have] = np.eye(4)
    sq = segment_quality(D[:len(T)], inp["K_d"], Tq, np.where(have, seg, -1), np.ones(len(T)), VideoParams())
    out["self_check_residual"] = {int(g_): round(q["depth_residual"], 3) for g_, q in sq.items() if g_ >= 0}
    # floor and camera height (gravity from GeoCalib in this path's world)
    Tf = T.copy()
    Tf[~have] = np.eye(4)
    up = up_vector(run, Tf, inp)
    Tl = level(Tf, up)
    h = floor_heights(D[:len(T)], inp["K_d"], Tl)
    good = have & np.isfinite(h)
    if good.sum() > 10:
        f = Tl[:, 1, 3] - h
        out["floor"] = {}
        for g_ in np.unique(seg[good]):
            m = good & (seg == g_)
            if m.sum() < 5:
                continue
            F = np.median(f[m])
            out["floor"][int(g_)] = dict(keyframes=int(m.sum()), floor_p10_p90_m=round(float(np.ptp(np.percentile(
                f[m], [10, 90]))), 3), cam_height_median_m=round(float(np.median(Tl[m, 1, 3] - F)), 3),
                cam_height_moge_median_m=round(float(np.median(h[m])), 3))
    return out


def arkit_metrics(T: np.ndarray, inp: dict, capture: Path, seg: np.ndarray | None) -> dict:
    from floorplan.video.evaluate import arkit_reference, trajectory_metrics
    ref = arkit_reference(capture, inp["ts"], inp["kf"][:len(T)], inp["rot"])["T_wc"]
    have = np.isfinite(T[:, 0, 0])
    seg = np.zeros(len(T), int) if seg is None else seg
    out = {}
    m = trajectory_metrics(T[have], ref[have])
    out["all"] = dict(keyframes=int(have.sum()), sim3_ate_m=round(m["sim3"]["ate_rmse_m"], 3),
                      se3_ate_m=round(m["se3"]["ate_rmse_m"], 3), scale_error_pct=round(m["scale_error_pct"], 1),
                      rot_err_deg=round(m["rot_err_median_deg"], 1), path_m=round(m["path_length_m"], 1))
    segs = []
    for g in np.unique(seg[have]):
        k = have & (seg == g)
        if k.sum() < 10:
            continue
        m = trajectory_metrics(T[k], ref[k])
        segs.append(dict(segment=int(g), keyframes=int(k.sum()), sim3_ate_m=round(m["sim3"]["ate_rmse_m"], 3),
                         se3_ate_m=round(m["se3"]["ate_rmse_m"], 3), scale_error_pct=round(m["scale_error_pct"], 1)))
    out["segments"] = segs
    w = []
    for a in range(0, int(have.sum()) - 15, 30):
        ids = np.where(have)[0][a:a + 30]
        if len(ids) < 10:
            continue
        m = trajectory_metrics(T[ids], ref[ids])
        w.append(round(m["scale_error_pct"], 1))
    out["window30_scale_error_pct"] = w
    return out


def measure(run: Path, out: Path, a) -> None:
    inp = run_inputs(run)
    D = np.stack([depth_mod.clean_depth(d, 4.0, 0.05) for d in inp["depth_raw"]])
    revisits = [r for r in json.loads(Path(a.revisits).read_text()) if r["dist_m"] < 5] if a.revisits else []
    steps = json.loads(Path(a.steps).read_text()) if a.steps else []
    paths = {}
    g = np.load(run / "work/graph_inputs.npz")
    sc = inp["scene"]
    paths["dpvo_scaled_before_graph"] = (g["T_lev"], g["seg"])
    paths["dpvo_final_scene"] = (sc["T_wc"][inp["kf"]], sc["kf_segment"])
    if a.sfm:
        z = np.load(a.sfm)
        paths["sfm"] = (z["T_wc"] if "T_wc" in z.files else z["T"], z["segment"] if "segment" in z.files else None)
    for v in VARIANTS:
        r = chain(run, out, v, inp)
        if r is not None:
            paths[f"ma_{v}"] = (r["T"], r["segment"])
            (out / f"links_{v}.json").write_text(json.dumps(r["links"], indent=1, default=float))
            np.savez(out / f"path_{v}.npz", T=r["T"], segment=r["segment"])
    res = []
    for name, (T, seg) in paths.items():
        m = measure_path(name, T, seg, inp, run, revisits, steps, D)
        if a.arkit:
            m["arkit"] = arkit_metrics(T, inp, Path(a.arkit), seg)
        res.append(m)
        print(json.dumps(m, default=float), flush=True)
    (out / "measure.json").write_text(json.dumps(res, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["solve", "measure"])
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--revisits")
    ap.add_argument("--steps")
    ap.add_argument("--sfm")
    ap.add_argument("--arkit")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "solve":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        solve(a.run, a.out, a.variants.split(","))
    else:
        measure(a.run, a.out, a)


if __name__ == "__main__":
    main()
