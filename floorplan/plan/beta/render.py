"""Debug figure: room polygons, wall ids and lengths, doors, windows and mirrors over the wall-density image.

This figure is how we check the extractor by eye: every number printed here is the number in plan.json.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from floorplan.model import Plan  # noqa: E402

KIND_COLOR = {"door": "#d62728", "passage": "#ff7f0e", "window": "#1f77b4"}


def _fmt(m) -> str:
    return "n/a" if m is None or m.value is None else f"{m.value:.2f}±{m.half_width:.2f}"


def render_debug(plan: Plan, debug: dict, out_png: Path, wall_density: np.ndarray | None = None) -> None:
    grid, labels = debug["grid"], debug["labels"]
    extent = [grid.u0, grid.u0 + grid.cols * grid.cell, grid.v0 + grid.rows * grid.cell, grid.v0]
    fig, ax = plt.subplots(figsize=(16, 16 * grid.rows / grid.cols + 1))
    dens = wall_density if wall_density is not None else debug["maps"]["wall_raw"].astype(float)
    bg = np.where(labels > 0, 0.93, 1.0) - 0.8 * np.clip(dens / max(np.percentile(dens[dens > 0], 90), 1), 0, 1)
    ax.imshow(bg, cmap="gray", extent=extent, vmin=0, vmax=1, interpolation="nearest")
    cmap = plt.get_cmap("tab20")
    walls = {w.id: w for w in plan.walls}
    for i, r in enumerate(plan.rooms):
        P = np.array(r.polygon + r.polygon[:1])
        ax.fill(P[:, 0], P[:, 1], color=cmap(i % 20), alpha=0.25, lw=0)
        ax.plot(P[:, 0], P[:, 1], "-", color=cmap(i % 20), lw=1.5)
        c = np.array(r.polygon).mean(0)
        ax.text(c[0], c[1], f"{r.id} {r.label.split(' (')[0]}\n{_fmt(r.floor_area)} m²\n"
                f"{r.bbox_dims[0].value:.2f}x{r.bbox_dims[1].value:.2f} m\nceil {_fmt(r.ceiling_height)}",
                ha="center", va="center", fontsize=8, weight="bold")
        for wid in r.wall_ids:
            w = walls[wid]
            if w.length.value < 0.25:
                continue
            m = (np.array(w.p0) + np.array(w.p1)) / 2 + 0.12 * np.array(w.normal)
            ls = "-" if w.length.status == "measured" else ":"
            ax.plot(*np.array([w.p0, w.p1]).T, ls, color="k", lw=0.8)
            rot = 0 if abs(w.p0[1] - w.p1[1]) < 1e-6 else 90
            ax.text(m[0], m[1], f"{wid.split('-')[1]} {w.length.value:.2f}"
                    + ("" if w.thickness is None else f" t{w.thickness.value*100:.0f}"),
                    fontsize=5.5, ha="center", va="center", rotation=rot,
                    color="k" if w.length.status == "measured" else "0.45")
    for op in plan.openings:
        _draw_opening(ax, op, walls)
    for mr in plan.meta.get("mirrors", []):
        w = walls[mr["wall_id"]]
        ax.plot(*_span_on_wall(w, mr["span"]).T, "-", color="magenta", lw=4, alpha=0.8)
    traj = debug.get("traj")
    if traj is not None:
        ax.plot(traj[:, 0], traj[:, 1], "-", color="green", lw=0.4, alpha=0.6)
    ax.set_title(f"{plan.capture_id}: {len(plan.rooms)} rooms, {len(plan.walls)} walls, {len(plan.openings)} "
                 f"openings, footprint {_fmt(plan.footprint_area)} m²\nred door, orange passage, blue window "
                 f"(dashed = conf < 0.5), magenta mirror; dotted wall = inferred", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _span_on_wall(w, span) -> np.ndarray:
    horizontal = abs(w.p0[1] - w.p1[1]) < 1e-6
    off = w.p0[1] if horizontal else w.p0[0]
    return np.array([[span[0], off], [span[1], off]]) if horizontal else np.array([[off, span[0]], [off, span[1]]])


def _draw_opening(ax, op, walls) -> None:
    """Draw the opening along the wall it is closest to (its wall list may also hold perpendicular walls)."""
    from shapely.geometry import LineString, Point
    cands = [walls[i] for i in op.wall_ids if i in walls]
    if not cands:
        return
    w = min(cands, key=lambda x: LineString([x.p0, x.p1]).distance(Point(op.center)))
    horizontal = abs(w.p0[1] - w.p1[1]) < 1e-6
    half = op.width.value / 2
    c = np.array(op.center)
    seg = np.array([c - [half, 0], c + [half, 0]]) if horizontal else np.array([c - [0, half], c + [0, half]])
    ax.plot(*seg.T, "--" if op.confidence < 0.5 else "-", color=KIND_COLOR[op.kind], lw=3.5, alpha=0.85)
    ax.text(c[0], c[1], f"{op.id} {op.width.value:.2f}", fontsize=6, color=KIND_COLOR[op.kind],
            ha="left", va="bottom", weight="bold")


def render_openings(plan: Plan, raw_points: np.ndarray, floor_y: float, out_png: Path, max_panels: int = 12) -> None:
    """One panel per door/passage: raw points between 0.3 and 1.8 m around it (plan view) and the measured
    jamb-to-jamb segment in red. This is the evidence behind every opening width in plan.json."""
    from shapely.geometry import LineString, Point
    ops = [o for o in plan.openings if o.kind in ("door", "passage")][:max_panels]
    if not ops:
        return
    walls = {w.id: w for w in plan.walls}
    h = raw_points[:, 1] - floor_y
    band = raw_points[(h > 0.3) & (h < 1.8)]
    cols = min(4, len(ops))
    rows = int(np.ceil(len(ops) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    for ax, op in zip(axs.ravel(), ops):
        c = np.array(op.center)
        m = (np.abs(band[:, 0] - c[0]) < 1.0) & (np.abs(band[:, 2] - c[1]) < 1.0)
        pts = band[m][::max(1, m.sum() // 40000)]
        ax.scatter(pts[:, 0], pts[:, 2], s=0.2, c="0.3")
        if "measured along v" in op.evidence or "measured along u" in op.evidence:
            along_v = "measured along v" in op.evidence
        else:
            cands = [walls[i] for i in op.wall_ids if i in walls]
            w = min(cands, key=lambda x: LineString([x.p0, x.p1]).distance(Point(op.center))) if cands else None
            along_v = w is not None and abs(w.p0[0] - w.p1[0]) < abs(w.p0[1] - w.p1[1])
        half = op.width.value / 2
        seg = np.array([c - [0, half], c + [0, half]]) if along_v else np.array([c - [half, 0], c + [half, 0]])
        ax.plot(*seg.T, "-", color=KIND_COLOR[op.kind], lw=2)
        ax.set_title(f"{op.id} {op.kind} {'/'.join(op.room_ids)}\nwidth {_fmt(op.width)} m ({op.width.status}), "
                     f"conf {op.confidence}", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(c[0] - 1, c[0] + 1)
        ax.set_ylim(c[1] + 1, c[1] - 1)
        ax.tick_params(labelsize=6)
    for ax in axs.ravel()[len(ops):]:
        ax.axis("off")
    fig.suptitle(f"{plan.capture_id}: doors and passages, raw LiDAR points 0.3-1.8 m (plan view, 2 x 2 m crops)",
                 fontsize=10)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
