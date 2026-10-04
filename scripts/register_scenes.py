"""Register every pair of prepared scenes (outputs/<capture>/scene.npz) and save transforms + figures.

Usage:
  python scripts/register_scenes.py                 # all pairs found under outputs/
  python scripts/register_scenes.py --verify-3d     # also cross-check with Open3D 3D point-to-plane ICP
  python scripts/register_scenes.py --scenes 'outputs/drift/*/scene_on/scene.npz' --out outputs/eval/registration_drift_on
                                                    # same, on drift-corrected scenes (loop check on vs off)

Output: outputs/eval/registration/<A>__<B>.json, <A>__<B>_overlay.png, <A>__<B>_residual.png,
        <A>__<B>_yaw.png, <A>__<B>_tiles.png (local disagreement), summary.json (all pairs) and cycles.json
        (loop A<-B<-C vs A<-C for every triple). B is always mapped into A's frame.
The benchmark reads these transforms to match walls/rooms between captures.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorplan.eval.register import (Evidence2D, Registration, RegisterParams, apply_T2,  # noqa: E402
                                     cycle_error, local_disagreement, register, scene_evidence,
                                     wall_residual_map)
from floorplan.pipeline.scene import load_scene  # noqa: E402


def find_scenes(root: Path, patterns: list[str]) -> dict[str, Path]:
    """Scene name -> folder. A scene in a variant folder (e.g. outputs/drift/<capture>/scene_on) is named
    '<capture>@scene_on' so it is never confused with the original capture."""
    out = {}
    for pat in patterns:
        for f in sorted(root.glob(pat)):
            d = f.parent
            out[d.name if d.parent == root / "outputs" else f"{d.parent.name}@{d.name}"] = d
    return out


def _base(name: str) -> str:
    return name.split("@")[0]


def verify_3d(sa: dict, sb: dict, T3: np.ndarray, voxel: float = 0.04) -> dict:
    """Independent check: Open3D point-to-plane ICP on the full 3D TSDF surfaces (walls AND floor), started from
    our 2D answer. If the 2D result is right, the 3D ICP should barely move it."""
    import open3d as o3d

    def cloud(s):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(s["points"].astype(np.float64)))
        pc.normals = o3d.utility.Vector3dVector(s["normals"].astype(np.float64))
        return pc.voxel_down_sample(voxel)

    A, B = cloud(sa), cloud(sb)
    res = o3d.pipelines.registration.registration_icp(
        B, A, 0.05, T3, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    D = res.transformation @ np.linalg.inv(T3)
    ang = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
    return dict(fitness=float(res.fitness), inlier_rmse_m=float(res.inlier_rmse),
                delta_rotation_deg=float(ang), delta_translation_m=[float(x) for x in D[:3, 3]])


def _plot_overlay(A: Evidence2D, B: Evidence2D, reg: Registration, path: Path) -> None:
    T = np.array(reg.T2)
    pb = apply_T2(T, B.wall_pts)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(*A.wall_pts.T, s=0.2, c="#1f4e9c", alpha=0.5, label=f"A: {A.name}")
    ax.scatter(*pb.T, s=0.2, c="#d1495b", alpha=0.5, label=f"B: {B.name} (registered)")
    ax.set_aspect("equal")
    ax.set_xlabel("u = x (m)")
    ax.set_ylabel("v = z (m)")
    ax.legend(markerscale=20, loc="upper right")
    ax.set_title(f"yaw {reg.yaw_deg:.2f} deg, RMSE {reg.residual_rmse_m * 100:.1f} cm, "
                 f"IoU {reg.overlap_iou:.2f}, distinct {reg.distinctiveness:.2f}")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_residual(A: Evidence2D, B: Evidence2D, reg: Registration, path: Path) -> None:
    pb, d = wall_residual_map(A, B, np.array(reg.T2))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(*A.wall_pts.T, s=0.1, c="0.8")
    sc = ax.scatter(*pb.T, s=0.3, c=d * 100, cmap="viridis", vmin=0, vmax=10)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="distance to nearest A wall point (cm), capped at 15")
    ax.set_aspect("equal")
    ax.set_title(f"Local disagreement of B ({B.name}) with A ({A.name}) after one rigid fit")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_yaw(reg: Registration, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for q, curve in reg.yaw_curve.items():
        y = np.array(curve)
        ax.plot(y[:, 0] - float(q), y[:, 1], marker=".", label=f"{q} deg quadrant")
    ax.set_xlabel("fine yaw offset within quadrant (deg)")
    ax.set_ylabel("normalised correlation")
    ax.set_title(f"{reg.b} -> {reg.a}: global yaw search (distinctiveness {reg.distinctiveness:.2f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_tiles(A: Evidence2D, B: Evidence2D, reg: Registration, tiles: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(*A.wall_pts.T, s=0.1, c="0.75")
    ax.scatter(*apply_T2(np.array(reg.T2), B.wall_pts).T, s=0.1, c="#d1495b", alpha=0.3)
    tiles = [t for t in tiles if t["well_constrained"]]
    if tiles:
        c = np.array([t["centre"] for t in tiles])
        d = np.array([t["displacement_m"] for t in tiles])
        q = ax.quiver(c[:, 0], c[:, 1], d[:, 0], d[:, 1], [t["dyaw_deg"] for t in tiles], cmap="coolwarm",
                      angles="xy", scale_units="xy", scale=0.1, width=0.006, clim=(-4, 4))
        fig.colorbar(q, ax=ax, shrink=0.7, label="local yaw correction (deg)")
        for t in tiles:
            ax.annotate(f"{np.linalg.norm(t['displacement_m']) * 100:.0f} cm", t["centre"], fontsize=8,
                        xytext=(4, -10), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_title(f"Local re-fit of 3 m tiles of B ({B.name}); arrows x10; degenerate tiles hidden")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _cycles(regs: dict, ev: dict) -> list[dict]:
    """Every triple (a, b, c) with all three registrations: compare T_ab @ T_bc with T_ac."""
    def T(x, y):          # maps y into x
        if (x, y) in regs:
            return np.array(regs[(x, y)].T2)
        return np.linalg.inv(np.array(regs[(y, x)].T2))
    out = []
    for a, b, c in itertools.combinations(sorted(ev), 3):
        probe = ev[c].wall_pts[:: max(1, len(ev[c].wall_pts) // 2000)]
        out.append(dict(loop=[a, b, c], **cycle_error(T(a, b), T(b, c), T(a, c), probe)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=["outputs/*/scene.npz"],
                    help="glob(s) relative to the repo root; a variant replaces the original capture of its name")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval" / "registration")
    ap.add_argument("--verify-3d", action="store_true")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    found = find_scenes(ROOT, ["outputs/*/scene.npz"] + [p for p in args.scenes if p != "outputs/*/scene.npz"])
    # a variant (capture@variant) replaces its original capture, so each capture appears once
    variants = {_base(n) for n in found if "@" in n}
    scenes = {n: load_scene(d) for n, d in found.items() if "@" in n or n not in variants}
    ev = {n: scene_evidence(s, i, n) for n, (s, i) in scenes.items()}
    summary, regs = [], {}
    for a, b in itertools.combinations(sorted(ev), 2):
        # the larger capture is the reference frame A
        if len(ev[a].wall_pts) < len(ev[b].wall_pts):
            a, b = b, a
        t0 = time.time()
        reg = register(ev[a], ev[b], RegisterParams())
        d = reg.to_dict()
        d["runtime_s"] = round(time.time() - t0, 2)
        if args.verify_3d:
            d["verify_3d"] = verify_3d(scenes[a][0], scenes[b][0], np.array(reg.T3))
        tiles = local_disagreement(ev[a], ev[b], np.array(reg.T2))
        d["local_tiles"] = tiles
        disp = [float(np.linalg.norm(t["displacement_m"])) for t in tiles if t["well_constrained"]]
        d["tiles_summary"] = dict(n=len(disp), median_disp_m=float(np.median(disp)) if disp else None,
                                  p90_disp_m=float(np.percentile(disp, 90)) if disp else None,
                                  median_abs_dyaw_deg=float(np.median([abs(t["dyaw_deg"]) for t in tiles
                                                                       if t["well_constrained"]])) if disp else None)
        regs[(a, b)] = reg
        stem = f"{a}__{b}"
        (out / f"{stem}.json").write_text(json.dumps(d, indent=2))
        _plot_overlay(ev[a], ev[b], reg, out / f"{stem}_overlay.png")
        _plot_residual(ev[a], ev[b], reg, out / f"{stem}_residual.png")
        _plot_yaw(reg, out / f"{stem}_yaw.png")
        _plot_tiles(ev[a], ev[b], reg, tiles, out / f"{stem}_tiles.png")
        summary.append({k: v for k, v in d.items() if k not in ("yaw_curve", "local_tiles")})
        print(f"[register] {b} -> {a}: yaw {reg.yaw_deg:.3f} deg, t ({reg.t[0]:.3f}, {reg.t[1]:.3f}) m, "
              f"rmse {reg.residual_rmse_m * 100:.2f} cm, inliers {reg.inlier_frac_b:.2f}, IoU {reg.overlap_iou:.2f}, "
              f"distinct {reg.distinctiveness:.2f}, {d['runtime_s']} s")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    cycles = _cycles(regs, ev)
    (out / "cycles.json").write_text(json.dumps(cycles, indent=2))
    for cy in cycles:
        print(f"[cycle] {' <- '.join(cy['loop'])}: yaw {cy['yaw_deg']:.2f} deg, "
              f"max displacement {cy['max_displacement_m'] * 100:.1f} cm")


if __name__ == "__main__":
    main()
