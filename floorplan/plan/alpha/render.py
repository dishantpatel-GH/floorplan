"""Debug figure: room polygons, wall ids and lengths, doors and windows over the wall-density image.

It exists to be LOOKED AT: most failures of a plan extractor (a missed wall, two rooms merged, a phantom window)
are obvious in this picture and invisible in a JSON file."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from floorplan.model import Plan  # noqa: E402
from floorplan.plan.alpha.evidence import make_grid, wall_density, wall_points  # noqa: E402
from floorplan.plan.alpha.params import AlphaParams  # noqa: E402


def _fmt(m) -> str:
    return "n/o" if m.value is None else f"{m.value:.2f}±{m.half_width:.2f}"


def render_plan(plan: Plan, scene: dict, info: dict, out_png: Path, p: AlphaParams | None = None) -> None:
    p = p or AlphaParams()
    grid = make_grid(scene["points"], p)
    dens = wall_density(wall_points(scene["points"], scene["normals"], info["floor_y"], info["ceiling_y"], p), grid)
    W, H = grid.nu * grid.res, grid.nv * grid.res
    fig, ax = plt.subplots(figsize=(min(4 + W * 1.6, 30), min(2 + H * 1.6, 30)))
    ax.imshow(np.log1p(dens).T, origin="lower", extent=grid.extent, cmap="gray_r", alpha=0.6)
    ax.plot(scene["traj"][:, 0], scene["traj"][:, 2], "-", c="0.6", lw=0.5, alpha=0.6)
    cmap = plt.get_cmap("tab10")
    for k, r in enumerate(plan.rooms):
        xy = np.asarray(r.polygon + [r.polygon[0]])
        ax.fill(xy[:, 0], xy[:, 1], color=cmap(k % 10), alpha=0.15)
        ax.plot(xy[:, 0], xy[:, 1], "-", color=cmap(k % 10), lw=1.8)
        c = xy[:-1].mean(0)
        ax.text(c[0], c[1], f"{r.id} {r.label.split(' (')[0]}\nA {_fmt(r.floor_area)} m²\nH {_fmt(r.ceiling_height)}",
                ha="center", va="center", fontsize=8, weight="bold", color=cmap(k % 10),
                bbox=dict(fc="white", alpha=0.7, lw=0))
    for w in plan.walls:
        p0, p1, n = np.asarray(w.p0), np.asarray(w.p1), np.asarray(w.normal)
        m = (p0 + p1) / 2 + 0.18 * n
        rot = 90 if abs(p0[0] - p1[0]) < abs(p0[1] - p1[1]) else 0
        col = "k" if w.length.status == "measured" else "tab:red"
        ax.text(m[0], m[1], f"{w.id.split('_')[1]} {_fmt(w.length)}", fontsize=5.5, rotation=rot, ha="center",
                va="center", color=col)
    for o in plan.openings:
        col = {"door": "tab:green", "passage": "tab:olive", "window": "tab:cyan"}[o.kind]
        ax.plot(*o.center, marker="s" if o.kind == "window" else "o", ms=7, color=col, mec="k")
        ax.text(o.center[0], o.center[1] - 0.15, f"{o.id} {o.kind} {_fmt(o.width)}", fontsize=6, color=col,
                ha="center", weight="bold")
    ax.set_title(f"{plan.capture_id}: {len(plan.rooms)} rooms, {len(plan.walls)} walls, {len(plan.openings)} openings; "
                 f"net area {_fmt(plan.footprint_area)} m²  (red text = inferred, n/o = not observed)")
    ax.set_xlabel("u = x (m)")
    ax.set_ylabel("v = z (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
