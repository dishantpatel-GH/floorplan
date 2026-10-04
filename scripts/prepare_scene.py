#!/usr/bin/env python
"""Build and save the aligned scene for one capture, plus diagnostic figures.

Usage: python scripts/prepare_scene.py <capture_dir> [--out outputs] [--voxel 0.02]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.config import Config  # noqa: E402
from floorplan.pipeline.scene import build_scene, save_scene  # noqa: E402


def figures(scene, info, out: Path):
    P, N, C = scene["points"], scene["normals"], scene["colors"]
    fy = info["floor_y"]
    cy = info["ceiling_y"] if info["ceiling_y"] is not None else fy + 2.4
    wall = (np.abs(N[:, 1]) < 0.3) & (P[:, 1] > fy + 0.2) & (P[:, 1] < cy - 0.2)
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    ax[0].hist2d(P[wall, 0], P[wall, 2], bins=[np.arange(P[:, 0].min(), P[:, 0].max(), 0.02),
                                              np.arange(P[:, 2].min(), P[:, 2].max(), 0.02)], cmap="gray_r", cmin=1)
    tr = scene["traj"]
    ax[0].plot(tr[:, 0], tr[:, 2], "r-", lw=0.4)
    ax[0].set_title(f"{info['capture_id']}: aligned wall density (2 cm) + path; Manhattan score {info['manhattan_score']:.2f}")
    sl = (P[:, 1] > fy + 1.0) & (P[:, 1] < fy + 1.4)
    ax[1].scatter(P[sl, 0], P[sl, 2], s=0.2, c=C[sl] / 255.0)
    ax[1].set_title("aligned slice 1.0-1.4 m above floor")
    for a in ax:
        a.set_aspect("equal"); a.grid(True, lw=0.3); a.set_xlabel("u = x (m)"); a.set_ylabel("v = z (m)")
    fig.tight_layout(); fig.savefig(out / "aligned_overview.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    ap.add_argument("--voxel", type=float, default=None)
    a = ap.parse_args()
    cfg = Config.load(**({"voxel_m": a.voxel} if a.voxel else {}))
    scene, info, cap = build_scene(a.capture, cfg)
    out = a.out / f"{a.capture.parent.name}__{a.capture.name}"
    save_scene(scene, info, out)
    figures(scene, info, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
