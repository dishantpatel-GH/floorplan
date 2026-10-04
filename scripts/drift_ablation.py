#!/usr/bin/env python
"""Drift ablation: the stitched scene with drift correction OFF (ARKit poses as-is) and ON, plus variants.

Why: the case study's "Drift accountability" gate asks what we do about accumulated drift on the multi-room capture
and wants an ablation showing the stitched footprint with the correction on and off. There is no ground truth, so
every metric here is an internal-consistency metric that needs no tape measure:
  * loop residuals: how far revisited geometry disagrees (cm, deg), measured by ICP between fragments, in-sample and
    with k-fold held-out loops (the honest number: those loops were not used to compute the correction);
  * plane anchors: spread of each fragment's wall direction and floor level (should be one building, one floor);
  * wall sharpness: spread of raw LiDAR wall points around their local wall line (doubled walls = drift);
  * footprint proxies: observed floor area, filled footprint area, robust bounding dimensions.

Usage: python scripts/drift_ablation.py <capture_dir> [--out outputs/drift] [--folds 5] [--no-variants]
Writes outputs/drift/<capture>/: T_wc_corrected.npy, drift_report.json, ablation.json, scene_on/, fig_*.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floorplan.config import Config  # noqa: E402
from floorplan.io.stray import load_stray  # noqa: E402
from floorplan.pipeline.scene import build_scene, load_scene, save_scene  # noqa: E402
from floorplan.recon.drift import (RESIDUAL_KEYS, DriftParams, DriftState, anchor_spread,  # noqa: E402
                                   correct_drift, edge_residual, interpolate_corrections, register, rot_deg, run_graph,
                                   solve, summarise, yaw_angle)

OFF, ON = "#eb6834", "#2a78d6"          # validated categorical slots 2 and 1 (dataviz reference palette)
RNG_SEED = 0


# ============================================================================================ scene metrics
def wall_band(scene: dict, info: dict) -> np.ndarray:
    """Mask of TSDF wall points between 0.3 m above the floor and 0.3 m below the ceiling (or 2.0 m)."""
    P, N = scene["points"], scene["normals"]
    top = info["ceiling_y"] - 0.3 if info["ceiling_y"] is not None else info["floor_y"] + 2.0
    return (np.abs(N[:, 1]) < 0.3) & (P[:, 1] > info["floor_y"] + 0.3) & (P[:, 1] < top)


def wall_sharpness(scene: dict, info: dict, max_raw: int = 4_000_000, gap_m: float = 0.10,
                   tangent_bin_m: float = 0.5, min_pts: int = 100) -> dict:
    """Spread of raw LiDAR wall points around their local wall line, per oriented Manhattan face.

    1. Label TSDF wall points by oriented face (+x, -x, +z, -z); a raw point inherits the label of the nearest TSDF
       wall point within 3 cm. Oriented faces keep the two sides of a wall apart (a wall's thickness is not blur).
    2. Per face, cut the wall into 0.5 m pieces along its length; within a piece, sort points by their offset across
       the wall and split where the gap exceeds 10 cm. Each group is one local wall line.
    3. Residual = offset - group median. If drift doubled a wall (two passes 3 cm apart) the group is bimodal and the
       spread grows; a crisp wall gives the LiDAR noise floor (a few mm)."""
    P, N = scene["points"], scene["normals"]
    band = wall_band(scene, info)
    raw = scene["raw_points"]
    raw = raw[np.random.default_rng(RNG_SEED).choice(len(raw), min(max_raw, len(raw)), replace=False)]
    top = info["ceiling_y"] - 0.3 if info["ceiling_y"] is not None else info["floor_y"] + 2.0
    raw = raw[(raw[:, 1] > info["floor_y"] + 0.3) & (raw[:, 1] < top)]
    face = np.full(len(P), -1)
    for k, (ax, sign) in enumerate([(0, 1), (0, -1), (2, 1), (2, -1)]):
        face[band & (sign * N[:, ax] > 0.9)] = k
    lab = face >= 0
    d, j = cKDTree(P[lab]).query(raw, distance_upper_bound=0.03)
    ok = np.isfinite(d)
    raw, rface = raw[ok], face[lab][j[ok]]
    res = []
    for k, ax in enumerate([0, 0, 2, 2]):
        sel = raw[rface == k]
        if len(sel) < min_pts:
            continue
        perp, tang = sel[:, ax], sel[:, 2 - ax]
        tb = np.floor(tang / tangent_bin_m).astype(int)
        order = np.lexsort((perp, tb))
        perp, tb = perp[order], tb[order]
        new_group = np.r_[True, (np.diff(tb) != 0) | (np.diff(perp) > gap_m)]
        df = pd.DataFrame(dict(g=np.cumsum(new_group), x=perp))
        n = df.groupby("g")["x"].transform("size")
        r = (df["x"] - df.groupby("g")["x"].transform("median"))[n >= min_pts].to_numpy()
        res.append(r)
    r = np.abs(np.concatenate(res))
    return dict(points=int(len(r)), sigma_mad_mm=float(1000 * 1.4826 * np.median(r)),
                p90_mm=float(1000 * np.percentile(r, 90)), within_1cm=float((r < 0.01).mean()))


def footprint(scene: dict, info: dict, cell: float = 0.05) -> dict:
    """Footprint proxies: observed floor area, filled footprint area, robust bounding dimensions of the walls."""
    P, N = scene["points"], scene["normals"]
    fl = (N[:, 1] > 0.9) & (np.abs(P[:, 1] - info["floor_y"]) < 0.05)
    walls = P[wall_band(scene, info)]
    lo = np.minimum(P[fl][:, [0, 2]].min(0), walls[:, [0, 2]].min(0)) - 1
    ij = np.floor((P[fl][:, [0, 2]] - lo) / cell).astype(int)
    occ = np.zeros(ij.max(0) + 3, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(occ, np.ones((7, 7)), iterations=1))
    ext = np.percentile(walls[:, [0, 2]], [0.5, 99.5], axis=0)
    return dict(floor_observed_m2=float(occ.sum() * cell ** 2), footprint_filled_m2=float(filled.sum() * cell ** 2),
                bbox_x_m=float(ext[1, 0] - ext[0, 0]), bbox_z_m=float(ext[1, 1] - ext[0, 1]))


def scene_metrics(scene: dict, info: dict) -> dict:
    return dict(wall_sharpness=wall_sharpness(scene, info), footprint=footprint(scene, info),
                manhattan_score=info["manhattan_score"], floor_y=info["floor_y"], ceiling_y=info["ceiling_y"],
                ceiling_height_global=info["ceiling_height_global"])


# ============================================================================================ loop evaluation
def eval_loops(st: DriftState, X: np.ndarray, kinds=("revisit", "local")) -> dict:
    """Residuals of the reference loop set (loops accepted by the default run) under fragment poses X."""
    out = {}
    for kind in kinds:
        rs = [edge_residual(e, X, st.frags) for e in st.edges if e.kind == kind and e.accepted]
        out[kind] = {k: summarise(r[k] for r in rs) for k in RESIDUAL_KEYS}
    return out


def cross_validate(st: DriftState, folds: int) -> dict:
    """k-fold over accepted loops: re-solve without a fold, measure that fold's residual (ARKit vs corrected).

    The held-out loops were not used to compute the correction, so this is the honest estimate of the drift that
    remains between two passes over the same room (the drift_sigma other modules should use)."""
    loops = [e for e in st.edges if e.kind != "odometry" and e.accepted]
    if len(loops) < 2:
        return dict(n=0, note="fewer than 2 accepted loops: nothing to hold out")
    fold = np.random.default_rng(RNG_SEED).permutation(len(loops)) % min(folds, len(loops))
    rows = []
    for k in range(fold.max() + 1):
        held = {id(e) for e, f in zip(loops, fold) if f == k}
        X, _ = solve(st.frags, [e for e in st.edges if id(e) not in held], st.man, st.flo, st.params)
        for e in (e for e in loops if id(e) in held):
            rows.append(dict(kind=e.kind, arkit=edge_residual(e, st.X_arkit, st.frags),
                             corrected=edge_residual(e, X, st.frags)))
    out = {}
    for kind in ("revisit", "local", "all"):
        sel = [r for r in rows if kind == "all" or r["kind"] == kind]
        out[kind] = {stage: {k: summarise(r[stage][k] for r in sel) for k in RESIDUAL_KEYS}
                     for stage in ("arkit", "corrected")}
    out["rows"] = rows
    return out


def icp_noise_floor(st: DriftState) -> dict:
    """ICP vs ARKit on CONSECUTIVE fragments, where ARKit is locally near-exact (1.5 m of travel, ~3 s).

    Their disagreement bounds the measurement noise of one ICP loop closure from above: loop residuals at or below
    this level cannot be told apart from ICP noise, so it is the floor of every residual reported here."""
    rows = []
    for a, b in zip(st.frags[:-1], st.frags[1:]):
        T0 = np.linalg.inv(b.pose) @ a.pose
        e = register(a, b, T0, st.params, "local")
        if e.fitness >= st.params.loop_fitness_min:
            rows.append(edge_residual(e, st.X_arkit, st.frags))
    return {k: summarise(r[k] for r in rows) for k in RESIDUAL_KEYS}


def sigma_sweep(st: DriftState, folds: int) -> list[dict]:
    """Held-out loop residual for a grid of noise-model sigmas (ICP measurements reused, only the solve is redone)."""
    grid = dict(odo_trans_sigma_m=[0.01, 0.02], odo_yaw_sigma_deg=[0.1, 0.3, 0.6, 1.2],
                yaw_smooth_sigma_deg=[0.0, 0.03, 0.1], loop_trans_sigma_m=[0.015, 0.03],
                manhattan_sigma_deg=[0.5, 1.0, 2.0])
    rows = []
    for vals in itertools.product(*grid.values()):
        over = dict(zip(grid, vals))
        cv = cross_validate(replace(st, params=replace(st.params, **over)), folds)
        rows.append(dict(**over, heldout_surface_sep_cm=cv["all"]["corrected"]["surface_sep_cm"]))
    return rows


def drift_sigma(cv: dict, report: dict) -> dict:
    """1-sigma drift between two observations of the same surface made at different times, for other modules.

    drift_sigma_m = RMS over loops of the surface separation (displacement along the surface normal), which is
    exactly how far a wall seen on two passes is offset. It applies to distances between surfaces observed at
    different times (a room revisited, room-to-room offsets in the stitched plan), not within one fragment.
    Source: held-out loops if any (honest), else in-sample loops (a lower bound, flagged)."""
    lr = report["loop_residuals"]
    candidates = [("held-out loops (k-fold)", cv.get("all")),
                  ("in-sample loops (lower bound)", lr.get("revisit") or lr.get("local"))]
    for basis, src in candidates:
        if src and src["corrected"]["surface_sep_cm"].get("n"):
            pick = lambda s: dict(drift_sigma_m=s["surface_sep_cm"]["rms"] / 100,  # noqa: E731
                                  vertical_m=s["vert_disp_cm"]["rms"] / 100, rot_deg=s["rot_deg"]["rms"])
            return dict(basis=basis, n_loops=src["corrected"]["surface_sep_cm"]["n"],
                        corrected=pick(src["corrected"]), arkit=pick(src["arkit"]))
    return dict(basis="no loops in this capture: use the value from a multi-room capture", corrected=None)


# ============================================================================================ figures
def _density(P: np.ndarray, lo: np.ndarray, shape: tuple[int, int], cell: float) -> np.ndarray:
    ij = np.floor((P[:, [0, 2]] - lo) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    h = np.zeros(shape)
    np.add.at(h, (ij[ok, 0], ij[ok, 1]), 1)
    return h


def overlay_image(P_off: np.ndarray, P_on: np.ndarray, cell: float = 0.02):
    """RGB top-down image: orange = only OFF has a wall here, blue = only ON, dark = both agree."""
    allp = np.vstack([P_off, P_on])
    lo = allp[:, [0, 2]].min(0) - 0.2
    shape = tuple((np.ceil((allp[:, [0, 2]].max(0) + 0.2 - lo) / cell)).astype(int))
    a, b = (_density(P, lo, shape, cell) >= 2 for P in (P_off, P_on))
    grow = np.ones((3, 3), bool)
    both = (a & ndimage.binary_dilation(b, grow)) | (b & ndimage.binary_dilation(a, grow))
    img = np.ones(shape + (3,))
    for mask, hexc in ((a & ~both, OFF), (b & ~both, ON), (both, "#222222")):
        img[mask] = matplotlib.colors.to_rgb(hexc)
    return img.transpose(1, 0, 2), lo, cell, (a ^ b).astype(float)


def fig_overlay(P_off, P_on, traj_off, traj_on, title: str, path: Path):
    img, lo, cell, diff = overlay_image(P_off, P_on)
    ext = [lo[0], lo[0] + img.shape[1] * cell, lo[1], lo[1] + img.shape[0] * cell]
    sm = ndimage.uniform_filter(diff, 50)                    # 1 m windows: where do OFF and ON disagree most?
    zooms = []
    for _ in range(2):
        i, j = np.unravel_index(np.argmax(sm), sm.shape)
        zooms.append((lo[0] + i * cell, lo[1] + j * cell))
        sm[max(0, i - 100):i + 100, max(0, j - 100):j + 100] = 0
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_axes([0.03, 0.06, 0.56, 0.86])
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    ax.plot(traj_off[:, 0], traj_off[:, 2], color=OFF, lw=0.6, alpha=0.7, label="camera path, ARKit (OFF)")
    ax.plot(traj_on[:, 0], traj_on[:, 2], color=ON, lw=0.6, alpha=0.7, label="camera path, corrected (ON)")
    for k, (u, v) in enumerate(zooms):
        ax.add_patch(plt.Rectangle((u - 1, v - 1), 2, 2, fill=False, ec="#52514e", lw=1))
        ax.text(u - 1, v + 1.05, f"zoom {k + 1}", color="#52514e", fontsize=9)
    ax.set_title(title + "\nwalls: orange = OFF only, blue = ON only, dark = both agree", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("u = x (m)"), ax.set_ylabel("v = z (m)")
    for k, (u, v) in enumerate(zooms):
        z = fig.add_axes([0.62, 0.53 - 0.47 * k, 0.36, 0.40])
        z.imshow(img, origin="lower", extent=ext, interpolation="nearest")
        z.set_xlim(u - 1, u + 1), z.set_ylim(v - 1, v + 1)
        z.set_title(f"zoom {k + 1} (2 x 2 m)", fontsize=10)
        z.tick_params(labelsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_side_by_side(scenes: dict, path: Path):
    """Each variant in its own aligned frame, as the plan extractor will see it, with footprint proxies."""
    fig, axes = plt.subplots(1, len(scenes), figsize=(9 * len(scenes), 9))
    for ax, (name, (scene, info, m)) in zip(np.atleast_1d(axes), scenes.items()):
        P = scene["points"][wall_band(scene, info)]
        ax.hist2d(P[:, 0], P[:, 2], bins=[np.arange(P[:, 0].min(), P[:, 0].max(), 0.02),
                                          np.arange(P[:, 2].min(), P[:, 2].max(), 0.02)], cmap="gray_r", cmin=1)
        f, w = m["footprint"], m["wall_sharpness"]
        ax.set_title(f"{name}: bbox {f['bbox_x_m']:.2f} x {f['bbox_z_m']:.2f} m, footprint "
                     f"{f['footprint_filled_m2']:.1f} m2\nwall spread (MAD sigma) {w['sigma_mad_mm']:.1f} mm, "
                     f"{100 * w['within_1cm']:.0f}% of wall points within 1 cm of their line", fontsize=10)
        ax.set_aspect("equal"), ax.grid(True, lw=0.3), ax.set_xlabel("u = x (m)"), ax.set_ylabel("v = z (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fig_corrections(st: DriftState, t0: float, path: Path):
    """Per-fragment correction over time, and the plane anchors before/after (the drift ARKit accumulated).

    The ceiling panel is the independent check: ceilings are never used in the solve."""
    t = np.array([f.t_anchor for f in st.frags]) - t0
    dxz = 100 * np.linalg.norm((st.X_final[:, :3, 3] - st.X_arkit[:, :3, 3])[:, [0, 2]], axis=1)
    dy = 100 * (st.X_final[:, 1, 3] - st.X_arkit[:, 1, 3])
    psi = np.degrees([yaw_angle(a[:3, :3] @ b[:3, :3].T) for a, b in zip(st.X_final, st.X_arkit)])
    fig, ax = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for a, y, lab in ((ax[0, 0], dxz, "plan position correction (cm)"), (ax[0, 1], psi, "yaw correction (deg)"),
                      (ax[2, 1], dy, "vertical correction (cm)")):
        a.plot(t, y, color=ON, marker="o", ms=3, lw=1.5)
        a.set_ylabel(lab)
    ce = [(k, f.ceiling_y) for k, f in enumerate(st.frags) if f.ceiling_y is not None]
    ref = np.median([c for _, c in ce]) if ce else 0.0
    ce = [(k, c) for k, c in ce if abs(c - ref) < 0.15]
    for X, col, lab in ((st.X_arkit, OFF, "ARKit (OFF)"), (st.X_final, ON, "corrected (ON)")):
        sh = np.array([yaw_angle(X[k][:3, :3] @ st.frags[k].pose[:3, :3].T) for k in st.man])
        th = np.array([st.frags[k].wall_theta for k in st.man]) - sh
        if len(th):
            dev = np.degrees((th - np.angle(np.mean(np.exp(4j * th))) / 4 + np.pi / 4) % (np.pi / 2) - np.pi / 4)
            ax[1, 0].plot(t[st.man], dev, color=col, marker="o", ms=3, lw=1, label=lab)
        for a, items in ((ax[1, 1], [(k, st.frags[k].floor_y) for k in st.flo]), (ax[2, 0], ce)):
            if items:
                lv = np.array([v + X[k][1, 3] - st.frags[k].pose[1, 3] for k, v in items])
                a.plot(t[[k for k, _ in items]], 100 * (lv - np.median(lv)), color=col, marker="o", ms=3, lw=1,
                       label=lab)
    ax[1, 0].set_ylabel("fragment wall direction - building (deg)")
    ax[1, 1].set_ylabel("fragment floor level - median (cm)")
    ax[2, 0].set_ylabel("fragment ceiling level - median (cm)\n(check only, not used in the solve)")
    for a in ax.ravel():
        a.grid(True, lw=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    for a in ax[2]:
        a.set_xlabel("time (s)")
    fig.suptitle("Drift corrections per fragment, and the plane anchors under ARKit vs corrected poses")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_loops(st: DriftState, cv: dict, path: Path):
    """Per-loop residual: ARKit vs corrected, in-sample (all accepted loops) and held-out (k-fold)."""
    loops = [e for e in st.edges if e.kind != "odometry" and e.accepted]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    panels = [("in-sample (loops used in the solve)",
               [(edge_residual(e, st.X_arkit, st.frags), edge_residual(e, st.X_final, st.frags), e.kind)
                for e in loops]),
              ("held-out (k-fold: loop NOT used in the solve)",
               [(r["arkit"], r["corrected"], r["kind"]) for r in cv.get("rows", [])])]
    for a, (title, rows) in zip(ax, panels):
        if not rows:
            a.text(0.5, 0.5, "no loops", ha="center", transform=a.transAxes)
            continue
        order = np.argsort([-r[0]["surface_sep_cm"] for r in rows])
        x = np.arange(len(rows))
        a.bar(x - 0.2, [rows[i][0]["surface_sep_cm"] for i in order], 0.4, color=OFF, label="ARKit (OFF)")
        a.bar(x + 0.2, [rows[i][1]["surface_sep_cm"] for i in order], 0.4, color=ON, label="corrected (ON)")
        a.set_xticks(x, ["R" if rows[i][2] == "revisit" else "L" for i in order], fontsize=7)
        a.set_ylabel("surface separation of revisited geometry (cm)")
        a.set_title(title + "; R = revisit (>= 20 s apart), L = local", fontsize=10)
        a.legend(fontsize=8)
        a.grid(True, axis="y", lw=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ============================================================================================ main
def to_common_frame(P: np.ndarray, info_from: dict, T_align_from: np.ndarray, T_align_to: np.ndarray) -> np.ndarray:
    """Points of one aligned scene -> capture world -> another scene's aligned frame (the gauge is shared: fragment
    0 is fixed by the solver, so corrected and ARKit worlds coincide at the start of the walk)."""
    R, t = T_align_from[:3, :3], T_align_from[:3, 3]
    Pw = (P - t) @ R
    return Pw @ T_align_to[:3, :3].T + T_align_to[:3, 3]


def variants(p: DriftParams) -> dict[str, DriftParams]:
    """What each ingredient contributes, plus the textbook baseline (Open3D 6-DoF, ICP odometry, loops only)."""
    return {"loops_only_4dof": replace(p, use_plane_anchors=False),
            "anchors_only_4dof": replace(p, use_loops=False),
            "open3d_6dof_baseline": replace(p, solver="open3d6", use_plane_anchors=False, refine_odometry=True,
                                            loop_rounds=1)}


def json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "drift")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-variants", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="also run the noise-model sigma sweep (slow, ~5 min)")
    a = ap.parse_args()
    name = f"{a.capture.parent.name}__{a.capture.name}"
    out = a.out / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load()
    off_dir = ROOT / "outputs" / name
    if (off_dir / "scene.npz").exists():
        off, off_info = load_scene(off_dir)
    else:
        off, off_info, _ = build_scene(a.capture, cfg)
        save_scene(off, off_info, off_dir)
    cap = load_stray(a.capture)
    p = DriftParams.from_config(cfg)

    T_on, report = correct_drift(cap, off["kf"], cap.T_wc, p)
    st: DriftState = report.pop("_state")
    np.save(out / "T_wc_corrected.npy", T_on)
    on, on_info, _ = build_scene(a.capture, cfg, T_wc=T_on)
    save_scene(on, on_info, out / "scene_on")

    cv = cross_validate(st, a.folds)
    report["heldout_loop_residuals"] = {k: v for k, v in cv.items() if k != "rows"}
    report["drift_sigma"] = drift_sigma(cv, report)
    report["icp_noise_floor"] = icp_noise_floor(st)
    if a.sweep:
        (out / "sigma_sweep.json").write_text(json.dumps(sigma_sweep(st, a.folds), indent=1, default=json_default))
    (out / "drift_report.json").write_text(json.dumps(report, indent=1, default=json_default))

    rows = {"OFF (ARKit as-is)": dict(scene=scene_metrics(off, off_info), loops=eval_loops(st, st.X_arkit),
                                      anchors=anchor_spread(st.frags, st.X_arkit, st.man, st.flo)),
            "ON (4-DoF + loops + plane anchors)": dict(scene=scene_metrics(on, on_info),
                                                       loops=eval_loops(st, st.X_final),
                                                       anchors=anchor_spread(st.frags, st.X_final, st.man, st.flo),
                                                       frame_correction=report["frame_correction"])}
    if not a.no_variants:
        for vname, vp in variants(p).items():
            vs = run_graph(cap, off["kf"], cap.T_wc, vp)
            T_v = interpolate_corrections(vs.frags, vs.X_final, cap.timestamps) @ cap.T_wc
            sc, si, _ = build_scene(a.capture, cfg, T_wc=T_v)
            tilt = max(np.degrees(np.arccos(np.clip((X[:3, :3] @ Y[:3, :3].T)[1, 1], -1, 1)))
                       for X, Y in zip(vs.X_final, st.X_arkit))
            rows[vname] = dict(scene=scene_metrics(sc, si), loops=eval_loops(st, vs.X_final),
                               anchors=anchor_spread(st.frags, vs.X_final, st.man, st.flo),
                               max_tilt_change_deg=float(tilt),
                               odometry_icp_fallbacks=sum(e.kind == "odometry" and bool(e.reason) for e in vs.edges),
                               loops_accepted=sum(e.kind != "odometry" and e.accepted for e in vs.edges),
                               max_rot_change_deg=float(max(rot_deg(X @ np.linalg.inv(Y))
                                                            for X, Y in zip(vs.X_final, st.X_arkit))))
    (out / "ablation.json").write_text(json.dumps(dict(capture=name, variants=rows, heldout=report[
        "heldout_loop_residuals"], drift_sigma=report["drift_sigma"]), indent=1, default=json_default))

    P_off = off["points"][wall_band(off, off_info)]
    P_on = to_common_frame(on["points"][wall_band(on, on_info)], on_info, on["T_align"], off["T_align"])
    traj_on = to_common_frame(on["traj"], on_info, on["T_align"], off["T_align"])
    fig_overlay(P_off, P_on, off["traj"], traj_on, f"{name}: drift correction OFF vs ON (common frame)",
                out / "fig_overlay.png")
    fig_side_by_side({"OFF (ARKit)": (off, off_info, rows["OFF (ARKit as-is)"]["scene"]),
                      "ON (corrected)": (on, on_info, rows["ON (4-DoF + loops + plane anchors)"]["scene"])},
                     out / "fig_side_by_side.png")
    fig_corrections(st, float(cap.timestamps[0]), out / "fig_corrections.png")
    fig_loops(st, cv, out / "fig_loops.png")
    print_table(rows, report)
    print(f"saved {out}")


def print_table(rows: dict, report: dict):
    cols = ("revisit sep cm", "rev deg", "yaw sd deg", "floor cm", "ceil cm", "wall MAD mm", "<1cm", "footprint m2",
            "bbox m")
    print(f"\n{'variant':36s} " + " ".join(f"{c:>12s}" for c in cols))
    nan = float("nan")
    for k, r in rows.items():
        lp, rd = r["loops"]["revisit"]["surface_sep_cm"], r["loops"]["revisit"]["rot_deg"]
        an, sc = r["anchors"], r["scene"]
        vals = (lp.get("median", nan), rd.get("median", nan), an["wall_yaw_std_deg"] or nan,
                an["floor_mad_sigma_cm"] or nan, an["ceiling_mad_sigma_cm"] or nan,
                sc["wall_sharpness"]["sigma_mad_mm"], 100 * sc["wall_sharpness"]["within_1cm"],
                sc["footprint"]["footprint_filled_m2"])
        bbox = f"{sc['footprint']['bbox_x_m']:.2f}x{sc['footprint']['bbox_z_m']:.2f}"
        print(f"{k:36s} " + " ".join(f"{v:12.2f}" for v in vals) + f" {bbox:>12s}")
    print("drift_sigma:", json.dumps(report["drift_sigma"], default=json_default))


if __name__ == "__main__":
    main()
