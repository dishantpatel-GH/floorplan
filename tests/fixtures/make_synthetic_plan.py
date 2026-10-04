#!/usr/bin/env python
"""Generate tests/fixtures/synthetic_plan.json: a realistic apartment Plan with known geometry.

Why a synthetic plan: the exporters (JSON, SVG/PNG, DXF) are built in parallel with the plan extractors, so no real
Plan exists yet. A hand-designed plan with KNOWN answers lets us test the exporters on every case the real data will
throw at them, before the real data arrives:
  * 6 rooms: an L-shaped corridor (the connector), living room, kitchen, bedroom, bathroom and a tiny 1.1 m WC.
  * a 45-degree wall (non-Manhattan), 0.10 m interior partitions, exterior walls whose thickness is unknown.
  * doors, a doorless passage, windows, a low-confidence window behind glass.
  * a room with no ceiling observation, an inferred (not measured) wall length, damage, a concealed-damage flag
    and scope items.
Geometry is derived, not typed: walls come from polygon edges, wall thickness from the gap to the neighbouring
room's parallel wall, opening-to-wall links from distance. So the fixture is self-consistent by construction.

`stress_plan()` is a second, deliberately awkward plan (many rooms, rotated 30 deg, tiny rooms, missing values)
used only to check the renderer does not crash or overlap badly; it is not a realistic home.

Usage: python tests/fixtures/make_synthetic_plan.py [--out tests/fixtures/synthetic_plan.json]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from floorplan.export.json_export import save_plan_json  # noqa: E402
from floorplan.model import Adjacency, Measurement, Opening, Plan, Room, Wall  # noqa: E402

M = Measurement
SIGMA_WALL_M = 0.004          # 1-sigma of a LiDAR wall-face length (two plane fits, ~3 mm each)
SIGMA_OPENING_M = 0.005
SIGMA_THICKNESS_M = 0.006
MAX_PARTITION_M = 0.35        # a parallel neighbouring wall closer than this is the other face of the same wall

# (id, label, polygon (u, v) metres, ceiling height or None)
APARTMENT = [
    ("corridor", "Corridor", [(0, 0), (1.2, 0), (1.2, 4.0), (5.0, 4.0), (5.0, 5.2), (0, 5.2)], 2.62),
    ("living", "Living room", [(1.3, 0), (4.6, 0), (5.6, 1.0), (5.6, 3.9), (1.3, 3.9)], 2.71),
    ("kitchen", "Kitchen", [(5.1, 4.0), (7.6, 4.0), (7.6, 8.8), (5.1, 8.8)], 2.70),
    ("bedroom", "Bedroom", [(0, 5.3), (3.4, 5.3), (3.4, 8.8), (0, 8.8)], 2.70),
    ("bathroom", "Bathroom", [(3.5, 5.3), (5.0, 5.3), (5.0, 7.6), (3.5, 7.6)], 2.45),
    ("wc", "WC", [(3.5, 7.7), (4.6, 7.7), (4.6, 8.8), (3.5, 8.8)], None),
]
# (id, kind, centre (u, v), width, room ids, height, sill, detection confidence, evidence)
OPENINGS = [
    ("o_entry", "door", (0.6, 0.0), 0.90, ["corridor"], 2.05, None, 0.97, "frame edges + depth gap"),
    ("o_living", "door", (1.25, 2.0), 0.82, ["corridor", "living"], 2.04, None, 0.95, "depth gap through wall"),
    ("o_bed", "door", (2.0, 5.25), 0.80, ["corridor", "bedroom"], 2.03, None, 0.94, "depth gap through wall"),
    ("o_bath", "door", (4.2, 5.25), 0.70, ["corridor", "bathroom"], 2.02, None, 0.93, "depth gap through wall"),
    ("o_wc", "door", (4.05, 7.65), 0.62, ["wc", "bathroom"], 2.00, None, 0.88, "depth gap through wall"),
    ("o_kitchen", "passage", (5.05, 4.6), 1.00, ["corridor", "kitchen"], 2.30, None, 0.96, "full-height gap"),
    ("o_win_living", "window", (2.8, 0.0), 1.60, ["living"], 1.40, 0.90, 0.92, "frame + glass hole"),
    ("o_win_bay", "window", (5.1, 0.5), 0.90, ["living"], 1.40, 0.90, 0.85, "frame + glass hole"),
    ("o_win_bed", "window", (0.0, 7.0), 1.40, ["bedroom"], 1.30, 0.95, 0.91, "frame + glass hole"),
    ("o_win_kitchen", "window", (7.6, 6.4), 1.20, ["kitchen"], 1.10, 1.05, 0.90, "frame + glass hole"),
    ("o_win_wc", "window", (4.05, 8.8), 0.50, ["wc"], 0.60, 1.50, 0.40, "frosted glass: no LiDAR returns, colour edge only"),
]


# ----------------------------------------------------------------------------------------------- geometry
def _ccw(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Force counter-clockwise order in (u, v) (positive shoelace area), the model's convention."""
    a = np.asarray(poly, float)
    area2 = float(np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1]))
    return list(poly) if area2 > 0 else list(reversed(poly))


def _area_perimeter(poly) -> tuple[float, float]:
    a = np.asarray(poly, float)
    area = 0.5 * abs(float(np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1])))
    per = float(np.sum(np.linalg.norm(np.roll(a, -1, axis=0) - a, axis=1)))
    return area, per


def _walls_for_room(room_id: str, poly) -> list[Wall]:
    walls = []
    for i, (p0, p1) in enumerate(zip(poly, poly[1:] + poly[:1])):
        d = np.subtract(p1, p0)
        length = float(np.linalg.norm(d))
        n = (-d[1] / length, d[0] / length)                 # left normal of a CCW polygon = into the room
        walls.append(Wall(f"{room_id}_w{i}", room_id, tuple(p0), tuple(p1),
                          M.from_sigma(length, SIGMA_WALL_M, method="plane-fit corner to corner"),
                          (round(n[0], 6), round(n[1], 6)), support_points=int(4000 * length),
                          fit_rmse_m=0.003))
    return walls


def _partition_gap(w: Wall, other: Wall) -> float | None:
    """Distance between two anti-parallel, overlapping wall faces (the two sides of one partition), else None."""
    n, m = np.asarray(w.normal), np.asarray(other.normal)
    if float(n @ m) > -0.99:
        return None
    gap = float((np.asarray(other.p0) - np.asarray(w.p0)) @ -n)
    if not 0.0 < gap < MAX_PARTITION_M:
        return None
    t = np.array([-n[1], n[0]])
    a = sorted([float(np.asarray(w.p0) @ t), float(np.asarray(w.p1) @ t)])
    b = sorted([float(np.asarray(other.p0) @ t), float(np.asarray(other.p1) @ t)])
    return gap if min(a[1], b[1]) - max(a[0], b[0]) > 0.2 else None


def _assign_thickness(walls: list[Wall]) -> None:
    """Measured thickness only where both faces were seen (a neighbouring room's wall); exterior stays None."""
    for w in walls:
        gaps = [g for o in walls if o.room_id != w.room_id and (g := _partition_gap(w, o)) is not None]
        if gaps:
            w.thickness = M.from_sigma(min(gaps), SIGMA_THICKNESS_M, method="distance between the two wall faces")


def _seg_dist(p, a, b) -> float:
    p, a, b = map(np.asarray, (p, a, b))
    t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def _link_openings(openings: list[Opening], walls: list[Wall], tol: float = 0.15) -> None:
    for o in openings:
        o.wall_ids = [w.id for w in walls if w.room_id in o.room_ids and _seg_dist(o.center, w.p0, w.p1) < tol]
        for w in walls:
            if w.id in o.wall_ids:
                w.opening_ids.append(o.id)


# ----------------------------------------------------------------------------------------------- plans
def _rooms_and_walls(spec) -> tuple[list[Room], list[Wall]]:
    rooms, walls = [], []
    for rid, label, poly, ceiling in spec:
        poly = _ccw([(float(u), float(v)) for u, v in poly])
        rw = _walls_for_room(rid, poly)
        area, per = _area_perimeter(poly)
        a = np.asarray(poly)
        ext = a.max(0) - a.min(0)
        ceil = (M.from_sigma(ceiling, 0.004, method="floor/ceiling plane fit") if ceiling is not None
                else M.missing(method="floor/ceiling plane fit", reason="ceiling not in view"))
        rooms.append(Room(rid, label, poly, [w.id for w in rw],
                          M.from_sigma(area, SIGMA_WALL_M * per / 2, unit="m2", method="polygon area"),
                          M.from_sigma(per, SIGMA_WALL_M * math.sqrt(len(poly)), method="sum of wall lengths"),
                          ceil,
                          (M.from_sigma(float(max(ext)), SIGMA_WALL_M, method="aligned bbox"),
                           M.from_sigma(float(min(ext)), SIGMA_WALL_M, method="aligned bbox")),
                          floor_level=-1.5))
        walls += rw
    _assign_thickness(walls)
    return rooms, walls


def _openings(spec) -> list[Opening]:
    out = []
    for oid, kind, c, width, rids, h, sill, conf, ev in spec:
        out.append(Opening(oid, kind, [], rids, c, M.from_sigma(width, SIGMA_OPENING_M, method="jamb-to-jamb gap"),
                           M.from_sigma(h, 0.006, method="head height above floor"),
                           None if sill is None else M.from_sigma(sill, 0.006, method="sill height above floor"),
                           conf, ev))
    return out


def _adjacency(openings: list[Opening], shared_walls: tuple[tuple[str, str], ...] = ()) -> list[Adjacency]:
    adj = [Adjacency(o.room_ids[0], o.room_ids[1], o.id) for o in openings
           if len(o.room_ids) == 2 and o.kind in ("door", "passage")]
    return adj + [Adjacency(a, b, "shared_wall") for a, b in shared_walls]


def _damage_and_scope() -> tuple[list[dict], list[dict]]:
    damage = [
        {"id": "d1", "class": "water_stain", "surface_id": "bathroom:ceiling", "room_id": "bathroom",
         "extent": {"area": M.from_sigma(0.36, 0.03, unit="m2", method="segmentation mask on ceiling plane")},
         "confidence": 0.86, "concealed": False},
        {"id": "d2", "class": "crack", "surface_id": "living_w4", "room_id": "living",
         "extent": {"length": M.from_sigma(0.85, 0.02, method="skeleton length on wall plane")},
         "confidence": 0.78, "concealed": False},
        {"id": "d3", "class": "floor_water_damage", "surface_id": "kitchen:floor", "room_id": "kitchen",
         "extent": {"area": M.from_sigma(0.84, 0.05, unit="m2", method="segmentation mask on floor plane")},
         "plan_polygon": [[6.4, 7.6], [7.5, 7.6], [7.5, 8.7], [6.6, 8.7]], "confidence": 0.7, "concealed": False},
        {"id": "d4", "class": "concealed_moisture", "surface_id": "bathroom_w1", "room_id": "bathroom",
         "concealed": True, "rule": "R-02: ceiling stain within 1 m of a wet-room wall -> inspect wall cavity",
         "confidence": 0.5},
    ]
    scope = [
        {"id": "s1", "surface_id": "bathroom:ceiling", "damage_ids": ["d1"],
         "description": "Stain-block primer and repaint ceiling (2 coats)",
         "quantity": M.from_sigma(3.45, 0.02, unit="m2", method="ceiling area")},
        {"id": "s2", "surface_id": "living_w4", "damage_ids": ["d2"], "description": "Rake out and fill crack",
         "quantity": M.from_sigma(0.85, 0.02, method="crack length")},
        {"id": "s3", "surface_id": "bathroom_w1", "damage_ids": ["d4"],
         "description": "Moisture-meter inspection of wall cavity", "quantity": M(1.0, 1.0, 1.0, "ea", "count")},
    ]
    return damage, scope


def synthetic_plan() -> Plan:
    rooms, walls = _rooms_and_walls(APARTMENT)
    wc_bottom = next(w for w in walls if w.room_id == "wc" and abs(w.p0[1] - 8.8) < 1e-6 and abs(w.p1[1] - 8.8) < 1e-6)
    wc_bottom.length = M.from_sigma(wc_bottom.length.value, 0.03, method="occluded by toilet; corner to corner "
                                    "from neighbouring walls", status="inferred")
    openings = _openings(OPENINGS)
    _link_openings(openings, walls)
    damage, scope = _damage_and_scope()
    footprint = sum(r.floor_area.value for r in rooms)
    sig = math.sqrt(sum(((r.floor_area.hi - r.floor_area.lo) / 3.92) ** 2 for r in rooms))
    shared = (("bedroom", "bathroom"), ("living", "kitchen"))
    return Plan("synthetic_apartment", "lidar", rooms, walls, openings, _adjacency(openings, shared),
                M.from_sigma(footprint, sig, unit="m2", method="sum of room areas"), damage, scope,
                {"synthetic": True, "generator": "tests/fixtures/make_synthetic_plan.py",
                 "note": "hand-designed fixture with known geometry; not a real capture"})


def _rot(p, deg: float):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def stress_plan(seed: int = 0, n: int = 4, angle_deg: float = 30.0) -> Plan:
    """n x n grid of rooms of random size (0.8 to 4 m), rotated by angle_deg, with missing values sprinkled in."""
    rng = np.random.default_rng(seed)
    widths, heights = rng.uniform(0.8, 4.0, n), rng.uniform(0.8, 4.0, n)
    us, vs = np.concatenate([[0], np.cumsum(widths + 0.1)]), np.concatenate([[0], np.cumsum(heights + 0.1)])
    spec = []
    for i in range(n):
        for j in range(n):
            corners = [(us[i], vs[j]), (us[i] + widths[i], vs[j]), (us[i] + widths[i], vs[j] + heights[j]),
                       (us[i], vs[j] + heights[j])]
            ceiling = None if rng.random() < 0.3 else float(rng.uniform(2.3, 2.8))
            spec.append((f"r{i}{j}", f"Room {i}{j}", [_rot(p, angle_deg) for p in corners], ceiling))
    rooms, walls = _rooms_and_walls(spec)
    for w in walls[::7]:
        w.length = M.missing(method="plane-fit corner to corner", reason="wall end occluded")
    openings = []
    for i in range(n - 1):
        for j in range(n):
            c = _rot((us[i + 1] - 0.05, vs[j] + heights[j] / 2), angle_deg)
            kind = "door" if (i + j) % 2 == 0 else "passage"
            w = min(0.7, heights[j] * 0.6)
            openings.append(Opening(f"o{i}{j}", kind, [], [f"r{i}{j}", f"r{i + 1}{j}"], c,
                                    M.from_sigma(w, SIGMA_OPENING_M, method="jamb-to-jamb gap")))
    _link_openings(openings, walls)
    footprint = sum(r.floor_area.value for r in rooms)
    return Plan(f"stress_{n}x{n}_rot{angle_deg:g}", "photo", rooms, walls, openings, _adjacency(openings),
                M.from_sigma(footprint, 0.05 * footprint / 1.96, unit="m2", method="sum of room areas"),
                meta={"synthetic": True, "stress_test": True})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "synthetic_plan.json")
    a = ap.parse_args()
    errors = save_plan_json(synthetic_plan(), a.out)
    if errors:
        raise SystemExit("fixture is invalid:\n" + "\n".join(errors))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
