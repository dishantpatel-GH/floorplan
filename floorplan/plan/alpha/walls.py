"""Step 6: turn each room polygon into measured walls, corners, lengths, area and perimeter with 95% intervals.

A room polygon from the cell complex is rectilinear: its edges alternate between u = const ('x' family) and
v = const ('z' family). Each edge is refitted on raw points (measure.fit_face) as a line with an offset and a small
tilt. Then:
  * corner k = intersection of the fitted lines of edge k-1 and edge k (exact for tilted walls too);
  * length of edge k = distance between its two corners. It is set by the edge's own line AND by its two
    NEIGHBOURS (which fix where it starts and ends);
  * every interval (lengths, area, perimeter, bounding box) comes from one Monte Carlo: draw every wall offset
    from N(offset, sigma) mc_samples times, rebuild all corners, take the 2.5 / 97.5 percentiles. For an
    axis-aligned room this reproduces sigma_L^2 = sigma_prev^2 + sigma_next^2, and it stays correct for tilted
    walls and for area, which is a non-linear function of the offsets.
Each offset sigma^2 = SE^2 + lidar_sigma^2 + inconsistency^2 + drift_sigma^2 / 2 (measure.py), so a wall length,
which involves two offsets, carries the full drift_sigma^2 budget once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from floorplan.model import Measurement, Wall
from floorplan.plan.alpha.measure import FaceFit, RawIndex, fit_face
from floorplan.plan.alpha.params import AlphaParams


@dataclass
class EdgeSpec:
    family: str                 # 'x': u = offset ; 'z': v = offset
    offset: float               # from the polygon (TSDF line)
    inward: int                 # +1: room on the + side of the line
    a: float                    # along-extent of the edge (polygon coordinates)
    b: float


def polygon_edges(poly: Polygon) -> list[EdgeSpec]:
    """Edges of a CCW rectilinear polygon with their family and inward side."""
    xy = np.asarray(poly.exterior.coords)[:-1]
    out = []
    for k in range(len(xy)):
        (u0, v0), (u1, v1) = xy[k], xy[(k + 1) % len(xy)]
        if abs(u1 - u0) < 1e-6:                     # vertical in plan: u = const, runs along v
            out.append(EdgeSpec("x", float(u0), 1 if v1 < v0 else -1, float(v0), float(v1)))
        else:                                        # v = const, runs along u
            out.append(EdgeSpec("z", float(v0), 1 if u1 > u0 else -1, float(u0), float(u1)))
    return out


def is_rectilinear(edges: list[EdgeSpec]) -> bool:
    return len(edges) >= 4 and all(edges[k].family != edges[k - 1].family for k in range(len(edges)))


def wall_axis(w: Wall) -> tuple[str, float, float, float, float]:
    """(family, offset, along_min, along_max, inward) of a (possibly slightly tilted) wall."""
    (u0, v0), (u1, v1) = w.p0, w.p1
    if abs(u1 - u0) < abs(v1 - v0):
        return "x", (u0 + u1) / 2, min(v0, v1), max(v0, v1), float(np.sign(w.normal[0]))
    return "z", (v0 + v1) / 2, min(u0, u1), max(u0, u1), float(np.sign(w.normal[1]))


# ---------------------------------------------------------------------------------------------------------------
# corners
# ---------------------------------------------------------------------------------------------------------------

def rect_corners(edges: list[EdgeSpec], offsets: np.ndarray) -> np.ndarray:
    """Corners of an axis-aligned polygon: vertex k = (u of the x-edge, v of the z-edge) meeting there."""
    K = len(edges)
    u = np.stack([offsets[..., k] if edges[k].family == "x" else offsets[..., k - 1] for k in range(K)], -1)
    v = np.stack([offsets[..., k] if edges[k].family == "z" else offsets[..., k - 1] for k in range(K)], -1)
    return np.stack([u, v], -1)


def tilted_corners(edges: list[EdgeSpec], offsets: np.ndarray, slopes: np.ndarray, mids: np.ndarray) -> np.ndarray:
    """Corners as intersections of tilted lines. An x-edge is u = o + s (v - m); a z-edge is v = o + s (u - m).
    offsets may carry leading sample dimensions (..., K); slopes and mids are (K,)."""
    K = len(edges)
    out = []
    for k in range(K):
        j = k - 1                                    # previous edge (perpendicular family)
        if edges[k].family == "x":
            ox, sx, mx, oz, sz, mz = offsets[..., k], slopes[k], mids[k], offsets[..., j], slopes[j], mids[j]
        else:
            ox, sx, mx, oz, sz, mz = offsets[..., j], slopes[j], mids[j], offsets[..., k], slopes[k], mids[k]
        # u = ox + sx (v - mx), v = oz + sz (u - mz)  ->  solve for u, then v
        u = (ox + sx * (oz - sz * mz - mx)) / (1 - sx * sz)
        v = oz + sz * (u - mz)
        out.append(np.stack([u, v], -1))
    return np.stack(out, -2)


def remove_jogs(poly: Polygon, tol: float) -> Polygon:
    """Remove steps shorter than `tol` (two parallel edges a few cm apart: the same wall seen twice with a little
    drift, two nearly coincident lines, a door reveal). The two parallel neighbours of the short edge merge into
    one edge at their length-weighted mean offset; the raw-point refit then measures the merged wall as a whole."""
    edges = polygon_edges(poly)
    while len(edges) > 4:
        lengths = [abs(e.b - e.a) for e in edges]
        k = int(np.argmin(lengths))
        if lengths[k] >= tol:
            break
        rot = edges[k - 1:] + edges[:k - 1] if k > 0 else edges[-1:] + edges[:-1]   # rot[0], rot[1], rot[2] = a, short, b
        a, b = rot[0], rot[2]
        la, lb = abs(a.b - a.a), abs(b.b - b.a)
        merged = EdgeSpec(a.family, (a.offset * la + b.offset * lb) / max(la + lb, 1e-9), a.inward, a.a, b.b)
        edges = _refresh_extents([merged] + rot[3:])
    return orient(Polygon(rect_corners(edges, np.array([e.offset for e in edges]))), sign=1.0)


def _refresh_extents(edges: list[EdgeSpec]) -> list[EdgeSpec]:
    """Recompute along-extents after a merge so the list stays a consistent closed chain."""
    V = rect_corners(edges, np.array([e.offset for e in edges]))
    K = len(edges)
    out = []
    for k, e in enumerate(edges):
        p0, p1 = V[k], V[(k + 1) % K]
        a, b = (p0[1], p1[1]) if e.family == "x" else (p0[0], p1[0])
        out.append(EdgeSpec(e.family, e.offset, e.inward, float(a), float(b)))
    return out


# ---------------------------------------------------------------------------------------------------------------
# room measurement
# ---------------------------------------------------------------------------------------------------------------

def _shoelace(V: np.ndarray) -> np.ndarray:
    x, y = V[..., 0], V[..., 1]
    return 0.5 * (np.sum(x * np.roll(y, -1, -1), -1) - np.sum(y * np.roll(x, -1, -1), -1))


def _interval(samples: np.ndarray, value: float, method: str, status: str, unit: str = "m") -> Measurement:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return Measurement(float(value), float(min(lo, value)), float(max(hi, value)), unit, method, status)


@dataclass
class RoomGeometry:
    edges: list[EdgeSpec]
    fits: list[FaceFit]
    sigmas: np.ndarray
    polygon: list[tuple[float, float]]
    area: Measurement
    perimeter: Measurement
    bbox: tuple[Measurement, Measurement]
    lengths: list[Measurement]
    area_samples: np.ndarray    # kept so the footprint interval can be propagated by summing samples


def fit_room_faces(poly: Polygon, idx: RawIndex, exclude: dict[tuple[str, int], list[tuple[float, float]]],
                   p: AlphaParams) -> tuple[list[EdgeSpec], list[FaceFit]]:
    """Refit every edge of one room. exclude[(family, k)] = along-intervals (openings) left out of edge k's fit."""
    edges = polygon_edges(poly)
    return edges, [fit_face(idx, e.family, e.offset, e.inward, e.a, e.b, exclude.get((e.family, k), []), p)
                   for k, e in enumerate(edges)]


def _at(f: FaceFit, along: float) -> float:
    """Position of a fitted (possibly tilted) face at an along-coordinate."""
    return f.offset + f.slope * (along - f.mid)


def reconcile_faces(rooms: list[tuple[list[EdgeSpec], list[FaceFit]]], max_cross: float = 0.1) -> int:
    """Two rooms that share a boundary with no measurable wall between them (a thin partition, or an opening
    split between two rooms) are fitted separately from each side; within noise their faces can CROSS by a few
    mm, which would make the rooms overlap. If one face is measured and the other only inferred, the inferred one
    moves onto the measured one. Otherwise both snap to their common midpoint and half the crossing is added to
    both sigmas. Tilts are dropped. Returns the number of reconciled pairs."""
    n = 0
    for ia, (ea, fa) in enumerate(rooms):
        for eb, fb in rooms[ia + 1:]:
            for ka, (e1, f1) in enumerate(zip(ea, fa)):
                for kb, (e2, f2) in enumerate(zip(eb, fb)):
                    if e1.family != e2.family or e1.inward != -e2.inward:
                        continue
                    lo = max(min(e1.a, e1.b), min(e2.a, e2.b))
                    hi = min(max(e1.a, e1.b), max(e2.a, e2.b))
                    if hi - lo < 0.02:                       # faces must actually face each other
                        continue
                    # gap between the faces at both ends of the overlap (tilted faces can cross part-way)
                    t = min(e1.inward * (_at(f1, x) - _at(f2, x)) for x in (lo, hi))   # > 0: f2 behind f1; < 0: crossed
                    if -max_cross < t < 0:
                        x0 = (lo + hi) / 2
                        if f1.status != f2.status:              # trust the measured face: the inferred one
                            meas, (ff, lst, k) = (f1, (f2, fb, kb)) if f1.status == "measured" else (f2, (f1, fa, ka))
                            lst[k] = FaceFit(_at(meas, ff.mid), meas.slope, ff.mid, ff.sigma, ff.se, ff.rmse,
                                             ff.n_points, ff.status)    # adopts its whole line, tilt included
                        else:                                   # both measured or both inferred: meet halfway
                            m, half = (_at(f1, x0) + _at(f2, x0)) / 2, abs(t) / 2
                            for ff, lst, k in ((f1, fa, ka), (f2, fb, kb)):
                                lst[k] = FaceFit(m, 0.0, ff.mid, float(np.hypot(ff.sigma, half)), ff.se, ff.rmse,
                                                 ff.n_points, ff.status)
                        f1, f2 = fa[ka], fb[kb]
                        n += 1
    return n


def measure_room(edges: list[EdgeSpec], fits: list[FaceFit], p: AlphaParams,
                 rng: np.random.Generator) -> RoomGeometry:
    """Corners, lengths, area, perimeter and bounding box (with Monte Carlo intervals) from fitted faces."""
    o = np.array([f.offset for f in fits])
    s = np.array([f.sigma for f in fits])
    sl = np.array([f.slope for f in fits])
    mid = np.array([f.mid for f in fits])
    K = len(edges)
    V = tilted_corners(edges, o, sl, mid)
    Vs = tilted_corners(edges, o[None, :] + rng.standard_normal((p.mc_samples, K)) * s[None, :], sl, mid)
    seg, seg_s = np.roll(V, -1, 0) - V, np.roll(Vs, -1, -2) - Vs
    L, L_s = np.linalg.norm(seg, axis=-1), np.linalg.norm(seg_s, axis=-1)
    mc = f"Monte Carlo over wall offsets (n={p.mc_samples})"
    lengths = []
    for k in range(K):
        ok = fits[k - 1].status == fits[(k + 1) % K].status == "measured"
        lengths.append(_interval(L_s[:, k], float(L[k]), "corner-to-corner between fitted walls; " + mc,
                                 "measured" if ok else "inferred"))
    status = "measured" if all(f.status == "measured" for f in fits) else "inferred"
    area = _interval(np.abs(_shoelace(Vs)), float(abs(_shoelace(V))), mc, status, "m2")
    per = _interval(L_s.sum(-1), float(L.sum()), mc, status)
    du, dv = float(np.ptp(V[:, 0])), float(np.ptp(V[:, 1]))
    bu = _interval(np.ptp(Vs[..., 0], -1), du, mc, status)
    bv = _interval(np.ptp(Vs[..., 1], -1), dv, mc, status)
    bbox = (bu, bv) if du >= dv else (bv, bu)
    return RoomGeometry(edges, fits, s, [tuple(map(float, c)) for c in V], area, per, bbox, lengths,
                        np.abs(_shoelace(Vs)))


def make_walls(room_id: str, geo: RoomGeometry) -> list[Wall]:
    walls = []
    K = len(geo.edges)
    V = np.asarray(geo.polygon)
    for k, (e, f) in enumerate(zip(geo.edges, geo.fits)):
        p0, p1 = V[k], V[(k + 1) % K]
        t = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        normal = (float(-t[1]), float(t[0]))         # left of the direction = inside for a CCW polygon
        walls.append(Wall(f"{room_id}_w{k}", room_id, tuple(map(float, p0)), tuple(map(float, p1)), geo.lengths[k],
                          normal, None, [], f.n_points, f.rmse))
    return walls


def assign_thickness(walls: list[Wall], sigma: dict[str, float], p: AlphaParams) -> dict[str, str]:
    """Wall thickness from two parallel faces of DIFFERENT rooms with opposite normals, 4-45 cm apart and
    overlapping by >= 0.3 m. sigma[wall_id] = 1-sigma of that face's offset. Returns {wall id: facing wall id}."""
    G = {w.id: wall_axis(w) for w in walls}
    facing: dict[str, str] = {}
    for w in walls:
        fam, off, a, b, n = G[w.id]
        best = None
        for w2 in walls:
            fam2, off2, a2, b2, n2 = G[w2.id]
            if w2.room_id == w.room_id or fam2 != fam or n2 != -n:
                continue
            t = n * (off - off2)                     # how far w2 lies BEHIND w (opposite to w's inward normal)
            if p.min_thickness_m <= t <= p.max_thickness_m and min(b, b2) - max(a, a2) >= p.min_overlap_m:
                if best is None or t < best[0]:
                    best = (t, w2.id)
        if best is not None:
            sig = float(np.hypot(sigma[w.id], sigma[best[1]]))
            w.thickness = Measurement.from_sigma(best[0], sig, method=f"distance between faces {w.id} and {best[1]}")
            facing[w.id] = best[1]
    return facing
