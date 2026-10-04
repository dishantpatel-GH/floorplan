"""Space-first plan extraction: aligned LiDAR scene -> floorplan.model.Plan (tier 'lidar').

Pipeline (each step in its own module, see docs/modules/plan_beta.md):
  1. freespace.py  interior free space on a 2 cm raster
  2. segment.py    distance transform + watershed + door-aware merging -> room regions
  3. walls.py      line arrangement polygon per room, overlap resolution, raw-point wall refinement
  4. levels.py     per-room floor level and ceiling height
  5. openings.py   doors / passages / windows (with mirror and glass handling)
  6. measure.py    lengths, areas, footprint with 95% intervals
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import LineString, Point, Polygon

from floorplan.model import Adjacency, Measurement, Opening, Plan, Room, Wall
from floorplan.plan.beta_v1 import openings as O
from floorplan.plan.beta_v1.freespace import interior_free_space
from floorplan.plan.beta_v1.grid import Grid, PointIndex
from floorplan.plan.beta_v1.levels import LevelPoints, room_levels
from floorplan.plan.beta_v1.measure import edge_lengths, footprint, room_measurements
from floorplan.plan.beta_v1.params import BetaParams
from floorplan.plan.beta_v1.segment import Neck, segment_rooms
from floorplan.plan.beta_v1.walls import (Bump, Line, RoomGeometry, clean_polygon, edge_normals, polygon_to_geometry,
                                       refine_lines, resolve_overlaps, room_polygon, vertical_points)


@dataclass
class RoomBuild:
    """Working state of one room while the plan is assembled."""
    label: int
    geom: RoomGeometry
    bumps: list[Bump]
    visited: bool
    enclosure: float
    rid: str = ""
    meas: dict = field(default_factory=dict)       # measure.room_measurements output
    floor_level: float = 0.0
    ceiling: Measurement | None = None
    faces: list[O.WallFace] = field(default_factory=list)

    @property
    def polygon(self) -> Polygon:
        return Polygon(self.geom.vertices())


def extract_plan(scene: dict, info: dict, capture_id: str, params: BetaParams | None = None) -> Plan:
    plan, _ = extract_plan_debug(scene, info, capture_id, params)
    return plan


def extract_plan_debug(scene: dict, info: dict, capture_id: str, params: BetaParams | None = None):
    """Same as extract_plan, also returns the intermediate rasters and timings for the debug figure."""
    p = params or BetaParams()
    rng = np.random.default_rng(p.seed)
    t0, timing = time.time(), {}

    def tick(name):
        timing[name] = round(time.time() - t0 - sum(timing.values()), 2)

    floor_y = float(info["floor_y"])
    grid = Grid.around(scene["points"][:, [0, 2]], p.cell_m)
    maps = interior_free_space(scene, info, grid, p)
    tick("free_space")
    labels, necks, _ = segment_rooms(maps["free"], maps["barrier"], p)
    tick("segmentation")
    index = PointIndex(scene["raw_points"], scene["raw_ray"], scene["raw_range"])
    tick("point_index")
    rooms, dropped = _build_rooms(labels, scene, floor_y, grid, maps, index, p)
    tick("walls")
    level_pts = LevelPoints(scene["raw_points"], scene["raw_ray"], scene["raw_range"], floor_y, p)
    for r in rooms:
        r.floor_level, _, r.ceiling = room_levels(r.polygon, level_pts, r.floor_level, p)
    tick("levels")
    for r in rooms:
        r.meas = room_measurements(r.geom, p.mc_samples, rng)
    walls = _make_walls(rooms)
    partners = _thickness(rooms, walls, p)
    tick("wall_assembly")
    openings, notes = _find_openings(rooms, necks, labels, grid, walls, scene, index, p)
    tick("openings")
    plan_rooms = _make_rooms(rooms)
    fp = footprint([r.geom for r in rooms], max(p.mc_samples // 2, 50), rng)
    adjacency = _adjacency(openings, partners, {w.id: w.room_id for w in walls})
    tick("measurements")
    meta = dict(extractor="plan_beta (space-first: free space -> distance-transform watershed -> line arrangement)",
                params=p.to_dict(), floor_y=floor_y, ceiling_y_global=info.get("ceiling_y"),
                dropped_regions=dropped, mirrors=[n for n in notes if n["kind"] == "mirror"],
                window_candidates=[n for n in notes if n["kind"] == "window_candidate"],
                weak_links=[n for n in notes if n["kind"] == "weak_link"],
                wall_tilt_deg={w.id: round(float(np.degrees(np.arctan(t))), 3) for w, t in _tilts(rooms, walls)},
                timing_s=timing, runtime_s=round(time.time() - t0, 2),
                grid=dict(u0=grid.u0, v0=grid.v0, cell=grid.cell, rows=grid.rows, cols=grid.cols))
    plan = Plan(capture_id, "lidar", plan_rooms, walls, openings, adjacency, fp, meta=meta)
    debug = dict(grid=grid, maps=maps, labels=labels, rooms=rooms, necks=necks, traj=scene["traj"][:, [0, 2]])
    return plan, debug


# ---------------------------------------------------------------- rooms and walls

def _build_rooms(labels, scene, floor_y, grid, maps, index, p: BetaParams):
    """Polygons for every region, overlaps resolved, refined on raw points, then validated."""
    seen_open = maps["floor_obs"] | maps["carved"]
    vertical = vertical_points(scene["points"], scene["normals"], floor_y, p)
    polys, lines, extents, bumps = {}, {}, {}, {}
    for lab in range(1, labels.max() + 1):
        polys[lab], lines[lab], extents[lab], bumps[lab] = room_polygon(labels, lab, vertical, grid, seen_open, p)
    polys = resolve_overlaps(polys, extents, grid)
    all_offsets = [l.offset for ls in lines.values() for l in ls]
    shared = [Line(l.axis, l.offset, 0.0, inferred=True) for ls in lines.values() for l in ls]
    traj = scene["traj"][:, [0, 2]]
    rooms, dropped, raster_lines = [], [], {}
    for lab, poly in polys.items():
        poly = clean_polygon(poly, all_offsets, p) if poly is not None else None
        if poly is None or poly.area < p.min_room_m2 or len(poly.exterior.coords) < 5:
            dropped.append(dict(region=int(lab), reason="no valid polygon after overlap resolution"))
            continue
        geom = polygon_to_geometry(poly, lines[lab] + shared, lab)
        raster = [replace(l) for l in geom.lines]
        refine_lines(geom, index, floor_y, p)
        raster_lines[lab] = raster
        visited = bool(contains_xy(Polygon(geom.vertices()), traj[:, 0], traj[:, 1]).any())
        enclosure = _enclosure(geom)
        if not visited and enclosure < p.unvisited_min_enclosure:
            dropped.append(dict(region=int(lab), reason=f"never entered and only {enclosure:.0%} of its boundary "
                                "is measured wall (space seen through an opening or glass)",
                                area_m2=round(float(Polygon(geom.vertices()).area), 2)))
            continue
        rooms.append(RoomBuild(lab, geom, bumps[lab], visited, enclosure, floor_level=floor_y))
    _undo_overlapping_refinements(rooms, raster_lines, p)
    rooms.sort(key=lambda r: -r.polygon.area)
    for i, r in enumerate(rooms):
        r.rid = f"R{i + 1}"
    return rooms, dropped


def _undo_overlapping_refinements(rooms: list[RoomBuild], raster_lines: dict, p: BetaParams,
                                  max_iter: int = 10) -> None:
    """If two refined rooms overlap, the lines of each that now cut into the other measured a surface that is not
    their own wall (typically something seen through a doorway on a virtual boundary). Put those lines back at
    their free-space position and mark them inferred. Polygons before refinement do not overlap, so this ends."""
    for _ in range(max_iter):
        changed = False
        for i, a in enumerate(rooms):
            for b in rooms[i + 1:]:
                inter = a.polygon.intersection(b.polygon)
                if inter.area < 1e-5:
                    continue
                step = False
                for r in (a, b):
                    step |= _revert_lines_in(r, inter, raster_lines[r.label], p.inferred_sigma_m)
                if not step:                                     # already at free-space positions: share the line
                    step = _snap_shared_boundary(a, b, inter)
                changed |= step
        if not changed:
            return


def _strip_lines(r: RoomBuild, region) -> list[int]:
    """Indices of r's lines that run along the thin overlap strip and touch it."""
    u0, v0, u1, v1 = region.bounds
    strip_axis = 0 if (u1 - u0) <= (v1 - v0) else 1        # thin in u -> the culprits are constant-u lines
    V = r.geom.vertices()
    out = []
    for k, e in enumerate(r.geom.edge_lines):
        edge = LineString([V[k], V[(k + 1) % len(V)]])
        if r.geom.lines[e].axis == strip_axis and edge.intersection(region.buffer(1e-3)).length > 1e-3:
            out.append(e)
    return sorted(set(out))


def _snap_shared_boundary(a: RoomBuild, b: RoomBuild, region) -> bool:
    """Move the less certain of the two overlapping boundary lines onto the other (a shared, inferred boundary)."""
    la, lb = _strip_lines(a, region), _strip_lines(b, region)
    if not la or not lb:
        return False
    ea = min(la, key=lambda e: a.geom.lines[e].sigma)
    eb = min(lb, key=lambda e: b.geom.lines[e].sigma)
    (keep, kr), (move, mr) = sorted([(a.geom.lines[ea], a), (b.geom.lines[eb], b)], key=lambda x: x[0].sigma)
    idx = mr.geom.lines.index(move)
    if abs(move.offset - keep.offset) < 1e-9:
        return False
    mr.geom.lines[idx] = replace(move, offset=keep.offset, slope=keep.slope, mid=keep.mid, inferred=True,
                                 sigma=max(move.sigma, keep.sigma))
    return True


def _revert_lines_in(r: RoomBuild, region, raster: list[Line], sigma: float) -> bool:
    """Revert the lines of r that run ALONG the overlap strip (edges crossing it are not the cause)."""
    changed = False
    for e in _strip_lines(r, region):
        line = r.geom.lines[e]
        if line.offset != raster[e].offset:
            r.geom.lines[e] = replace(raster[e], inferred=True, sigma=sigma, slope=line.slope, mid=line.mid)
            changed = True
    return changed


def _tilts(rooms: list[RoomBuild], walls: list[Wall]):
    """(wall, tilt) for measured walls: how far each wall deviates from the Manhattan axis (pass-1 fit)."""
    by_id = {w.id: w for w in walls}
    for r in rooms:
        for k, e in enumerate(r.geom.edge_lines):
            line = r.geom.lines[e]
            if not line.inferred:
                yield by_id[f"{r.rid}-W{k + 1}"], line.tilt


def _enclosure(geom: RoomGeometry) -> float:
    """Fraction of the perimeter that lies on walls measured on raw points."""
    lengths = edge_lengths(geom.vertices())
    measured = sum(L for L, e in zip(lengths, geom.edge_lines) if not geom.lines[e].inferred)
    return float(measured / max(lengths.sum(), 1e-9))


def _make_walls(rooms: list[RoomBuild]) -> list[Wall]:
    walls = []
    for r in rooms:
        V = r.geom.vertices()
        normals = edge_normals(r.geom)
        for k, length in enumerate(r.meas["walls"]):
            line = r.geom.lines[r.geom.edge_lines[k]]
            wid = f"{r.rid}-W{k + 1}"
            p0, p1 = V[k], V[(k + 1) % len(V)]
            walls.append(Wall(wid, r.rid, (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])), length,
                              (float(normals[k][0]), float(normals[k][1])), None, [], line.n_points, line.rmse))
            axis = line.axis
            r.faces.append(O.WallFace(wid, r.rid, axis, line.offset, float(np.sign(normals[k][axis])),
                                      tuple(sorted((float(p0[1 - axis]), float(p1[1 - axis])))), r.floor_level,
                                      line.inferred, line.slope, line.mid))
    return walls


def _thickness(rooms: list[RoomBuild], walls: list[Wall], p: BetaParams) -> dict[str, str]:
    """Two measured faces of different rooms, parallel, facing away from each other, 5-40 cm apart, overlapping
    along the wall -> one wall; thickness = distance between the faces. Returns {wall id: partner wall id}."""
    faces = [(f, r) for r in rooms for f in r.faces if not f.inferred]
    by_id = {w.id: w for w in walls}
    sig = {f.wall_id: r.geom.lines[r.geom.edge_lines[int(f.wall_id.split("-W")[1]) - 1]].sigma for f, r in faces}
    partners = {}
    for fa, ra in faces:
        best = None
        for fb, rb in faces:
            if rb is ra or fb.axis != fa.axis or fb.inward != -fa.inward:
                continue
            overlap = min(fa.span[1], fb.span[1]) - max(fa.span[0], fb.span[0])
            s_mid = 0.5 * (max(fa.span[0], fb.span[0]) + min(fa.span[1], fb.span[1]))
            gap = (fa.at(s_mid) - fb.at(s_mid)) * fa.inward    # > 0 when b lies behind a's face
            if p.thickness_min_m <= gap <= p.thickness_max_m and overlap >= 0.3:
                if best is None or gap < best[0]:
                    best = (gap, fb)
        if best is not None:
            gap, fb = best
            partners[fa.wall_id] = fb.wall_id
            by_id[fa.wall_id].thickness = Measurement.from_sigma(
                float(gap), float(np.hypot(sig[fa.wall_id], sig[fb.wall_id])), "m",
                f"distance between this face and the parallel face {fb.wall_id} of the neighbouring room")
    return partners


# ---------------------------------------------------------------- openings

def _find_openings(rooms, necks: dict[tuple[int, int], Neck], labels, grid: Grid, walls, scene, index, p):
    counter = iter(range(1, 10_000))

    def next_id() -> str:
        return f"O{next(counter)}"

    by_label = {r.label: r for r in rooms}
    openings: list[Opening] = []
    notes = []
    for (a, b), neck in necks.items():
        if a in by_label and b in by_label:
            op = _door_between(by_label[a], by_label[b], neck, grid, index, p, next_id)
            if op is not None:
                openings.append(op)
            else:
                notes.append(dict(kind="weak_link", rooms=[by_label[a].rid, by_label[b].rid],
                                  width_m=round(neck.width_m, 3),
                                  reason="free space touches through a gap narrower than a door and no gap "
                                         "could be measured on raw points: unobserved wall, not an opening"))
    for r in rooms:
        for bump in r.bumps:
            op = _door_from_bump(r, bump, index, p, next_id)
            if op is not None and not _duplicate(op, openings):
                openings.append(op)
    ctx = O.SeeThroughContext(O.RayIndex(scene, p), index, cKDTree(scene["points"]), scene["normals"],
                              {r.rid: r.polygon for r in rooms})
    for r in rooms:
        ceil_h = r.ceiling.value if r.ceiling is not None else None
        for f in r.faces:
            if f.inferred or f.span[1] - f.span[0] < p.window_min_w_m:
                continue
            wall = next(w for w in walls if w.id == f.wall_id)
            depth = wall.thickness.value if wall.thickness is not None else None
            found, nts = O.wall_openings(f, ctx, depth, ceil_h, p, next_id)
            notes += nts
            openings += [op for op in found if not _duplicate(op, openings)]
    _attach(openings, walls, rooms)
    return openings, notes


def _duplicate(op: Opening, existing: list[Opening], dist: float = 0.4) -> bool:
    return any(np.hypot(op.center[0] - e.center[0], op.center[1] - e.center[1]) < dist for e in existing)


def _door_between(ra: RoomBuild, rb: RoomBuild, neck: Neck, grid: Grid, index, p, next_id) -> Opening | None:
    """Door or passage at the neck that separates two rooms.

    The width is measured ALONG the cut (the line across the passage through the saddle), between the nearest
    raw points on either side, in a thin band around the cut. That works for a door in a wall (the band lies
    inside the wall slab, the gap ends are the jamb faces) and for a constriction across a corridor (the gap
    ends are the corridor's side walls) alike."""
    pt = grid.centers(np.array([neck.rc[0]]), np.array([neck.rc[1]]))[0]
    a = neck.axis                                            # the cut lies on a line of constant pt[a]
    t_rng = (pt[a] - p.cut_band_m, pt[a] + p.cut_band_m)
    floor_level = 0.5 * (ra.floor_level + rb.floor_level)
    gap = O.measure_gap(index, a, t_rng, float(pt[1 - a]), neck.width_m / 2, (p.jamb_h_lo_m, p.jamb_h_hi_m),
                        floor_level, p)
    if gap is not None and not (0.5 * neck.width_m <= gap.width <= 2.0 * neck.width_m + 0.2):
        gap = None
    if gap is None and neck.width_m < p.door_min_m:
        return None              # a sub-door-width leak through unobserved wall, not an opening (see weak_links)
    kind = "door" if neck.jambs and gap is not None else "passage"
    width = O.width_measurement(gap, neck.width_m, f"{kind} between {ra.rid} and {rb.rid}")
    ceil = [c.value for c in (ra.ceiling, rb.ceiling) if c is not None and c.value is not None]
    height = (O.head_height(index, a, t_rng, gap, floor_level, min(ceil) if ceil else None, p)
              if gap is not None else None)
    s_c = (gap.left + gap.right) / 2 if gap else float(pt[1 - a])
    center = (float(pt[0]), s_c) if a == 0 else (s_c, float(pt[1]))
    conf = 0.9 if (neck.jambs and gap is not None) else 0.6 if gap is not None else 0.4
    ev = (f"free-space constriction {neck.width_m:.2f} m wide separating {ra.rid} and {rb.rid}; "
          f"wall evidence on both sides: {neck.jambs}; gap ends measured on raw points: {gap is not None}")
    return Opening(next_id(), kind, [], [ra.rid, rb.rid], (float(center[0]), float(center[1])), width,
                   height, None, conf, f"{ev}; measured along {'v' if a == 0 else 'u'}")


def _door_from_bump(r: RoomBuild, bump: Bump, index, p, next_id) -> Opening | None:
    """Doorway into space we did not map (cut off the room polygon as a wall-deep bump)."""
    s_mid, w = (bump.span[0] + bump.span[1]) / 2, bump.span[1] - bump.span[0]
    face = next((f for f in r.faces if f.axis == bump.axis and abs(f.at(s_mid) - bump.offset) < 0.03
                 and f.span[0] - 0.05 <= bump.span[0] and bump.span[1] <= f.span[1] + 0.05), None)
    if face is None or w < p.door_min_m:
        return None
    t_rng = O.slab(face, bump.depth, s_mid)
    gap = O.measure_gap(index, face.axis, t_rng, s_mid, w / 2, (p.jamb_h_lo_m, p.jamb_h_hi_m), face.floor_level, p)
    if gap is not None and not (0.5 * w <= gap.width <= 1.5 * w + 0.1):
        gap = None
    width = O.width_measurement(gap, w, f"doorway in {face.wall_id} to unmapped space")
    ceil_h = r.ceiling.value if r.ceiling is not None else None
    height = O.head_height(index, face.axis, t_rng, gap, face.floor_level, ceil_h, p) if gap else None
    s_c = (gap.left + gap.right) / 2 if gap else s_mid
    center = (face.at(s_c), s_c) if face.axis == 0 else (s_c, face.at(s_c))
    ev = (f"the room's free space continues {bump.depth*100:.0f} cm into the wall over {w:.2f} m and open space "
          "was seen beyond it (doorway to a space that was not mapped)")
    return Opening(next_id(), "door", [face.wall_id], [r.rid], (float(center[0]), float(center[1])), width,
                   height, None, 0.7 if gap is not None else 0.5, ev)


def _attach(openings: list[Opening], walls: list[Wall], rooms: list[RoomBuild]) -> None:
    """Register each opening id on every wall it lies on (centre within 15 cm of the wall segment)."""
    for op in openings:
        c = Point(op.center)
        for w in walls:
            if w.room_id in op.room_ids and _segment_distance(c, w) < 0.15 + 0.2 * (op.kind != "window"):
                if op.id not in w.opening_ids:
                    w.opening_ids.append(op.id)
                if w.id not in op.wall_ids:
                    op.wall_ids.append(w.id)


def _segment_distance(c: Point, w: Wall) -> float:
    return LineString([w.p0, w.p1]).distance(c)


def _adjacency(openings: list[Opening], partners: dict[str, str], room_of: dict[str, str]) -> list[Adjacency]:
    """Rooms connected through a door/passage (via = opening id); otherwise rooms sharing a wall."""
    out, seen = [], set()
    for op in openings:
        if len(op.room_ids) == 2 and frozenset(op.room_ids) not in seen:
            out.append(Adjacency(op.room_ids[0], op.room_ids[1], op.id))
            seen.add(frozenset(op.room_ids))
    for wa, wb in partners.items():
        key = frozenset((room_of[wa], room_of[wb]))
        if key not in seen:
            seen.add(key)
            out.append(Adjacency(room_of[wa], room_of[wb], "shared_wall"))
    return out


# ---------------------------------------------------------------- rooms

def _make_rooms(rooms: list[RoomBuild]) -> list[Room]:
    out = []
    for r in rooms:
        m = r.meas
        V = r.geom.vertices()
        wall_ids = [f"{r.rid}-W{k + 1}" for k in range(len(V))]
        out.append(Room(r.rid, _label(r, m), [(float(x), float(y)) for x, y in V], wall_ids, m["area"],
                        m["perimeter"], r.ceiling, m["bbox"], float(r.floor_level)))
    return out


def _label(r: RoomBuild, m: dict) -> str:
    length, width = m["bbox"][0].value, m["bbox"][1].value
    kind = "corridor" if (width <= 1.6 and length / max(width, 1e-6) >= 2.0) else "room"
    return kind if r.visited else f"{kind} (not entered; seen through an opening)"
