"""Orchestrates the boundary-first plan extractor: aligned scene -> floorplan.model.Plan (tier 'lidar').

    evidence -> wall lines -> cell complex -> inside/outside (graph cut) -> rooms (watershed, snapped to cells)
             -> doors (gaps + jambs) -> walls refit on raw points -> heights -> windows -> plan

Every step is a pure function of the scene and AlphaParams, with fixed seeds, so the same input always gives the
same plan (the repeatability gate needs that from the software before it can ask it of the sensor).
"""
from __future__ import annotations

import time

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from floorplan.model import Adjacency, Measurement, Plan, Room, Wall
from floorplan.plan.alpha import doors as door_mod
from floorplan.plan.alpha import windows as win_mod
from floorplan.plan.alpha.complex import CellComplex, build_complex
from floorplan.plan.alpha.evidence import height_band, interior_evidence, make_grid, wall_points
from floorplan.plan.alpha.heights import level_data, room_levels
from floorplan.plan.alpha.labeling import RoomCells, barrier_raster, group_rooms, label_inside, regularize_polygon
from floorplan.plan.alpha.lines import detect_lines
from floorplan.plan.alpha.measure import build_raw_index
from floorplan.plan.alpha.params import AlphaParams
from floorplan.plan.alpha.walls import (assign_thickness, fit_room_faces, is_rectilinear, make_walls, measure_room,
                                       polygon_edges, reconcile_faces, remove_jogs, wall_axis)


def _room_label(poly, traj_uv: np.ndarray, bbox) -> str:
    entered = any(poly.contains(Point(*q)) for q in traj_uv[:: max(1, len(traj_uv) // 2000)])
    length, width = bbox[0].value, bbox[1].value
    base = "corridor" if width < 1.6 and length / max(width, 1e-6) >= 2.5 else "room"
    return base if entered else f"{base} (seen through an opening, not entered)"


def _exclusions(poly, cands: list[door_mod.DoorCandidate], p: AlphaParams):
    """Along-intervals of each polygon edge that sit in a doorway: excluded from that wall's fit."""
    out: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for k, e in enumerate(polygon_edges(poly)):
        for c in cands:
            if c.family == e.family and abs(c.line_offset - e.offset) < p.max_thickness_m:
                out.setdefault((e.family, k), []).append((c.g0 - p.opening_trim_m, c.g1 + p.opening_trim_m))
    return out


def _walls_on_line(walls: list[Wall], family: str, offset: float, along: float, tol: float) -> list[Wall]:
    out = []
    for w in walls:
        fam, off, a, b, _ = wall_axis(w)
        if fam == family and abs(off - offset) <= tol and a - 0.05 <= along <= b + 0.05:
            out.append(w)
    return out


def _is_exterior(w: Wall, cx: CellComplex, room_map: np.ndarray, own: int) -> bool:
    """No room within 0.6 m behind the wall face, at three points along it."""
    fam, off, a, b, n = wall_axis(w)
    for t in (0.25, 0.5, 0.75):
        along = a + t * (b - a)
        for k in np.arange(0.05, 0.61, 0.05):
            perp = off - n * k
            u, v = (perp, along) if fam == "x" else (along, perp)
            i, j = cx.cell_of(np.array([u]), np.array([v]))
            r = int(room_map[i[0], j[0]])
            if r >= 0 and r != own:
                return False
    return True


def _footprint(area_samples: list[np.ndarray], areas: list[float], status: str) -> Measurement:
    if not areas:
        return Measurement.missing("m2", "sum of room floor areas", "no rooms")
    tot = np.sum(area_samples, axis=0)
    lo, hi = np.percentile(tot, [2.5, 97.5])
    v = float(sum(areas))
    return Measurement(v, float(min(lo, v)), float(max(hi, v)), "m2",
                       "net floor area: sum of room interior areas (rooms are disjoint); Monte Carlo interval", status)


def _clean_rooms(rooms_c: list[RoomCells], room_map: np.ndarray, p: AlphaParams) -> tuple[list[RoomCells], list[str]]:
    """Remove jogs, make rooms disjoint (largest first: a smaller room never overlaps a larger one) and drop any
    room whose polygon is no longer a valid rectilinear shape. A cold run must never crash on one odd room: it is
    skipped, removed from room_map (in place) so no door or window refers to it, and reported as a warning."""
    kept, warnings, taken = [], [], None
    for rc in rooms_c:
        poly = remove_jogs(rc.polygon, p.min_jog_m)
        if taken is not None and poly.intersection(taken).area > 1e-6:
            poly = regularize_polygon(poly.difference(taken), p.min_sliver_m, close=False)
        if poly.is_empty or poly.geom_type != "Polygon" or not is_rectilinear(polygon_edges(poly)):
            warnings.append(f"room r{rc.index} dropped: polygon not rectilinear after clean-up")
            room_map[room_map == rc.index] = -1
            continue
        rc.polygon = poly
        kept.append(rc)
        taken = poly if taken is None else taken.union(poly)
    return kept, warnings


def extract_plan(scene: dict, info: dict, capture_id: str, params: AlphaParams | None = None) -> Plan:
    p = params or AlphaParams()
    t0 = time.time()
    timing: dict[str, float] = {}

    def tick(name: str) -> None:
        timing[name] = round(time.time() - t0 - sum(timing.values()), 2)

    floor_y, ceiling_y = info["floor_y"], info["ceiling_y"]
    rng = np.random.default_rng(p.seed)
    grid = make_grid(scene["points"], p)
    wp = wall_points(scene["points"], scene["normals"], floor_y, ceiling_y, p)
    lines = detect_lines(wp, grid, p)
    evidence = interior_evidence(scene, info, grid, p)
    tick("evidence_lines")
    cx = build_complex(lines, grid, evidence, p)
    inside = label_inside(cx, p)
    barrier = barrier_raster(scene, info, cx, p)
    rooms_c, room_map, _ = group_rooms(cx, inside, barrier, p)
    rooms_c, warnings = _clean_rooms(rooms_c, room_map, p)
    tick("complex_rooms")

    lo, hi = height_band(floor_y, ceiling_y, p)
    idx = build_raw_index(scene, floor_y, lo, hi)
    levels = level_data(scene, floor_y)
    room_ids = {rc.index: f"r{rc.index}" for rc in rooms_c}
    lv = {rc.index: room_levels(rc.polygon, levels, floor_y, p) for rc in rooms_c}
    tick("raw_index_levels")

    cands, openings = [], []
    floor_by_room = {k: v.floor_y for k, v in lv.items()}
    for c in door_mod.find_door_candidates(rooms_c, lines, cx, room_map, evidence["any"] & ~barrier, p):
        o = door_mod.measure_door(c, idx, floor_by_room, floor_y, p, f"o{len(openings)}")
        if o is not None:
            o.room_ids = [room_ids[r] for r in c.rooms]
            cands.append(c)
            openings.append(o)
    tick("doors")

    rooms: list[Room] = []
    walls: list[Wall] = []
    sigma: dict[str, float] = {}
    diag: dict[str, dict] = {}
    area_samples, areas = [], []
    traj_uv = scene["traj"][:, [0, 2]]
    faces = [fit_room_faces(rc.polygon, idx, _exclusions(rc.polygon, cands, p), p) for rc in rooms_c]
    n_reconciled = reconcile_faces(faces)
    for rc, (edges, fits) in zip(rooms_c, faces):
        rid = room_ids[rc.index]
        geo = measure_room(edges, fits, p, rng)
        ws = make_walls(rid, geo)
        for w, f, s in zip(ws, geo.fits, geo.sigmas):
            sigma[w.id] = float(s)
            diag[w.id] = dict(status=f.status, se_mm=round(1000 * f.se, 2), offset_sigma_mm=round(1000 * s, 2),
                              tilt_deg=round(float(np.degrees(np.arctan(f.slope))), 3),
                              rmse_mm=None if f.rmse is None else round(1000 * f.rmse, 1))
        walls += ws
        area_samples.append(geo.area_samples)
        areas.append(geo.area.value)
        rooms.append(Room(rid, _room_label(rc.polygon, traj_uv, geo.bbox), geo.polygon, [w.id for w in ws],
                          geo.area, geo.perimeter, lv[rc.index].ceiling, geo.bbox, float(lv[rc.index].floor_y)))
    facing = assign_thickness(walls, sigma, p)
    tick("walls")

    for o, c in zip(openings, cands):
        along = o.center[1] if c.family == "x" else o.center[0]
        for w in _walls_on_line(walls, c.family, c.line_offset, along, p.max_thickness_m):
            if w.room_id in o.room_ids:
                o.wall_ids.append(w.id)
                w.opening_ids.append(o.id)
    ctx = win_mod.window_context(scene, floor_y)
    counter = iter(range(len(openings), 10 ** 6))
    for rc in rooms_c:
        for w in (w for w in walls if w.room_id == room_ids[rc.index]):
            if w.length.value < p.window_min_w_m or not _is_exterior(w, cx, room_map, rc.index):
                continue
            ws_, warn = win_mod.find_windows(ctx, w, p, lambda: f"o{next(counter)}")
            for o in ws_:
                w.opening_ids.append(o.id)
            openings += ws_
            warnings += warn
    tick("windows")

    adjacency = [Adjacency(o.room_ids[0], o.room_ids[1], o.id) for o in openings if len(o.room_ids) == 2]
    linked = {tuple(sorted((a.room_a, a.room_b))) for a in adjacency}
    by_id = {w.id: w for w in walls}
    for wid, other_id in facing.items():
        key = tuple(sorted((by_id[wid].room_id, by_id[other_id].room_id)))
        if key not in linked:
            linked.add(key)
            adjacency.append(Adjacency(key[0], key[1], "shared_wall"))

    polys = [Polygon(r.polygon) for r in rooms]          # the OUTPUT polygons (fitted, possibly tilted corners)
    overlap = float(sum(polys[i].intersection(polys[j]).area for i in range(len(polys)) for j in range(i)))
    status = "measured" if all(r.floor_area.status == "measured" for r in rooms) else "inferred"
    meta = dict(extractor="plan_alpha (boundary-first cell complex)", params=p.to_dict(),
                lines={k: len(v) for k, v in lines.items()}, cells=list(cx.shape),
                room_overlap_m2=round(overlap, 6), reconciled_face_pairs=n_reconciled,
                union_area_m2=round(float(unary_union(polys).area), 4) if polys else 0.0,
                warnings=warnings, wall_fit=diag, timing_s=timing,
                runtime_s=round(time.time() - t0, 2), floor_y=floor_y, ceiling_y=ceiling_y)
    return Plan(capture_id, "lidar", rooms, walls, openings, adjacency, _footprint(area_samples, areas, status),
                meta=meta)

