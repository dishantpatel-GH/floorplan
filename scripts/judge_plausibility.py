"""Judge criteria 2, 3 and 5: plausibility, cold-run robustness, runtime of each plan extractor.

    python scripts/judge_plausibility.py                       (after scripts/judge_run.py)
    python scripts/judge_plausibility.py drift_on_band60       (also an experiment folder, drawn the same way)

Plausibility has no ground truth, so it is measured with checks any floor plan must satisfy, computed the same way
for both extractors from their plan.json and the scene:
  * structure: rooms do not overlap, every room has >= 4 walls, the adjacency graph is connected;
  * evidence: share of the observed floor that ends up inside some room (floor recall), and room area the camera
    never entered (space seen through a door, window or mirror; plausible for a balcony, a phantom for a mirror);
  * slivers: rooms under 1.5 m2 or narrower than 0.7 m (wall cores, shelves, corridor stubs);
  * measurement state: share of wall lengths measured on raw points (vs inferred), openings with a measured width.
Robustness: did the perturbed / cropped / degenerate runs finish, and what did they return?
Runtime: wall clock of extract_plan as recorded by judge_run.py (machine shared with other jobs; load noted).

Writes outputs/judge/plausibility.json and plans_<variant>.png (both extractors drawn with ONE renderer and one
scale, so the comparison is not biased by the authors' different debug figures).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from shapely.geometry import Point, Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from floorplan.eval.gates import _connected, room_overlaps  # noqa: E402
from judge_common import CAPTURES, OUT, PRODUCERS, jsonable, load_run, load_scene_for  # noqa: E402

CELL = 0.05                    # 5 cm raster for floor evidence: coarse enough to bridge LiDAR sample gaps
FLOOR_BAND = 0.05              # up-facing TSDF surface within 5 cm of the floor level counts as observed floor
SLIVER_AREA, SLIVER_WIDTH = 1.5, 0.7
KIND_STYLE = {"door": ("#d62828", "o"), "passage": ("#f77f00", "s"), "window": ("#1d78c1", "D")}


def floor_raster(scene: dict, info: dict) -> tuple[np.ndarray, np.ndarray]:
    """Observed-floor occupancy on a 5 cm grid, plus its origin."""
    P, N = scene["points"], scene["normals"]
    m = (np.abs(P[:, 1] - info["floor_y"]) < FLOOR_BAND) & (N[:, 1] > 0.9)
    uv = P[m][:, [0, 2]]
    o = uv.min(0) - 1.0
    ij = np.floor((uv - o) / CELL).astype(int)
    occ = np.zeros(ij.max(0) + 3, bool)
    occ[ij[:, 0], ij[:, 1]] = True
    return ndimage.binary_closing(occ, iterations=1), o


def floor_recall(plan: dict, occ: np.ndarray, origin: np.ndarray) -> float:
    """Share of observed-floor cells whose centre lies inside the union of room polygons."""
    rooms = unary_union([Polygon(r["polygon"]).buffer(0) for r in plan["rooms"] if len(r["polygon"]) >= 3])
    ij = np.argwhere(occ)
    if rooms.is_empty or not len(ij):
        return 0.0
    from shapely import contains_xy
    c = origin + (ij + 0.5) * CELL
    return float(contains_xy(rooms.buffer(0.05), c[:, 0], c[:, 1]).mean())


def unvisited(plan: dict, traj_uv: np.ndarray) -> tuple[int, float]:
    """Rooms the camera never entered (no trajectory sample inside the polygon): count and total area."""
    n, area = 0, 0.0
    pts = traj_uv[:: max(1, len(traj_uv) // 4000)]
    for r in plan["rooms"]:
        poly = Polygon(r["polygon"]).buffer(0)
        if not any(poly.contains(Point(p)) for p in pts):
            n += 1
            area += poly.area
    return n, area


def min_width(poly: Polygon) -> float:
    """Largest inscribed-circle diameter, approximated by negative buffering (how wide the room really is)."""
    lo, hi = 0.0, 5.0
    for _ in range(14):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if not poly.buffer(-mid / 2).is_empty else (lo, mid)
    return lo


def plan_stats(plan: dict, scene: dict, info: dict, occ, origin) -> dict:
    polys = [Polygon(r["polygon"]).buffer(0) for r in plan["rooms"]]
    walls = plan["walls"]
    kinds: dict[str, int] = {}
    for o in plan["openings"]:
        kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
    n_unv, a_unv = unvisited(plan, scene["traj"][:, [0, 2]])
    slivers = [r["id"] for r, p in zip(plan["rooms"], polys) if p.area < SLIVER_AREA or min_width(p) < SLIVER_WIDTH]
    lengths = np.array([w["length"]["value"] for w in walls])
    return dict(
        rooms=len(plan["rooms"]), walls=len(walls), openings=kinds,
        overlap_m2=round(sum(a for _, _, a in room_overlaps(plan)), 4),
        min_walls_per_room=min((len(r["wall_ids"]) for r in plan["rooms"]), default=0),
        connected=_connected(plan), adjacency=len(plan["adjacency"]),
        footprint_m2=plan["footprint_area"]["value"], room_area_sum_m2=round(sum(p.area for p in polys), 2),
        floor_recall=round(floor_recall(plan, occ, origin), 3),
        unvisited_rooms=n_unv, unvisited_area_m2=round(a_unv, 2), sliver_rooms=slivers,
        walls_shorter_than_15cm=int((lengths < 0.15).sum()),
        wall_length_measured_frac=round(float(np.mean([w["length"]["status"] == "measured" for w in walls])), 3),
        openings_width_measured=sum(o["width"]["status"] == "measured" for o in plan["openings"]),
        ceilings_observed=sum(r["ceiling_height"]["value"] is not None for r in plan["rooms"]),
        median_wall_halfwidth_cm=round(100 * float(np.median([(w["length"]["hi"] - w["length"]["lo"]) / 2
                                                              for w in walls])), 2),
    )


def draw(ax, plan: dict, traj_uv: np.ndarray, title: str) -> None:
    cmap = plt.get_cmap("tab10")
    ax.plot(traj_uv[:, 0], traj_uv[:, 1], color="0.75", lw=0.5)
    for i, r in enumerate(plan["rooms"]):
        P = np.array(r["polygon"] + [r["polygon"][0]])
        ax.fill(P[:, 0], P[:, 1], color=cmap(i % 10), alpha=0.25)
        ax.plot(P[:, 0], P[:, 1], color=cmap(i % 10), lw=1)
        c = Polygon(r["polygon"]).buffer(0).representative_point()
        ax.text(c.x, c.y, f"{r['id']}\n{r['floor_area']['value']:.1f}", fontsize=6, ha="center")
    for o in plan["openings"]:
        col, mk = KIND_STYLE.get(o["kind"], ("k", "x"))
        ax.plot(*o["center"], marker=mk, color=col, ms=5)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def runs_table() -> list[dict]:
    rows = []
    for f in sorted((OUT / "runs").glob("*/*/*/run.json")):
        d = json.loads(f.read_text())
        d.pop("traceback", None)
        d["folder"] = str(f.parent.relative_to(OUT))
        rows.append(d)
    return rows


def main() -> None:
    result = {"stats": {}, "load_average": os.getloadavg()}
    for variant in ("raw", "drift_on", *sys.argv[1:]):
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        for j, capture in enumerate(CAPTURES):
            scene, info = load_scene_for(capture, "drift_on" if variant.startswith("drift_on") else "raw")
            occ, origin = floor_raster(scene, info)
            for i, producer in enumerate(PRODUCERS):
                plan = load_run(variant, producer, capture)
                if plan is None:
                    continue
                s = plan_stats(plan, scene, info, occ, origin)
                result["stats"][f"{variant}/{producer}/{capture}"] = s
                draw(axes[i, j], plan, scene["traj"][:, [0, 2]],
                     f"{producer} / {capture.split('__')[0]} ({variant})\n{s['rooms']} rooms, "
                     f"{s['footprint_m2']:.1f} m2, floor recall {s['floor_recall']:.2f}, unvisited "
                     f"{s['unvisited_rooms']} ({s['unvisited_area_m2']:.1f} m2)")
                print(variant, producer, capture, s)
        for kind, (col, mk) in KIND_STYLE.items():
            axes[0, 0].plot([], [], marker=mk, color=col, ls="", label=kind)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / f"plans_{variant}.png", dpi=80)
        plt.close(fig)
    result["runs"] = runs_table()
    (OUT / "plausibility.json").write_text(json.dumps(jsonable(result), indent=1, default=float))
    for r in result["runs"]:
        print(r["folder"], r["status"], r.get("runtime_s"), r.get("rooms"), r.get("error", ""))


if __name__ == "__main__":
    main()
