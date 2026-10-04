"""Turning fitted wall lines into dimensioned measurements with 95% intervals.

Every room is a polygon whose corners are intersections of consecutive wall lines, so every derived quantity
(wall length, area, perimeter, bounding box, footprint) is a function of the line parameters alone: each line's
offset (walls.Line) plus one small rotation per room (RoomGeometry.yaw). Each has a 1-sigma uncertainty.

We propagate them by Monte Carlo: draw all line parameters from their normal distributions, recompute the
corners, and read the 2.5% and 97.5% quantiles of each quantity. Monte Carlo instead of closed-form error
propagation because area, bounding box and the union footprint are non-linear, and because a line shared by
several edges (a wall split by a notch) must move as one piece; drawing per line, not per edge, does that.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from floorplan.model import Measurement
from floorplan.plan.beta.walls import RoomGeometry


def sample_lines(geom: RoomGeometry, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(offsets, slopes), each (n, n_lines): offsets drawn per line, ONE room rotation drawn per sample (all
    walls of a room share it, so the sampled room stays rectangular)."""
    c = np.array([l.offset for l in geom.lines])
    sc = np.array([l.sigma for l in geom.lines])
    yaw = geom.yaw + rng.standard_normal(n) * geom.yaw_sigma
    k = np.array([l.slope for l in geom.lines]) - geom.line_slopes(geom.yaw)       # per-line part (free mode)
    sk = np.array([l.slope_sigma for l in geom.lines])
    slopes = geom.line_slopes(yaw) + k + rng.standard_normal((n, len(c))) * sk
    return c + rng.standard_normal((n, len(c))) * sc, slopes


def shoelace(V: np.ndarray) -> np.ndarray:
    """Polygon area for vertex arrays shaped (..., n, 2)."""
    x, y = V[..., 0], V[..., 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, -1) - np.roll(x, -1, -1) * y, -1))


def edge_lengths(V: np.ndarray) -> np.ndarray:
    """Edge k = vertex k -> vertex k+1; V shaped (..., n, 2)."""
    return np.linalg.norm(np.roll(V, -1, -2) - V, axis=-1)


def interval(samples: np.ndarray, value: float, unit: str, method: str, status: str = "measured") -> Measurement:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return Measurement(float(value), float(min(lo, value)), float(max(hi, value)), unit, method, status)


def room_measurements(geom: RoomGeometry, n: int, rng: np.random.Generator) -> dict:
    """Wall lengths, floor area, perimeter and bbox dims (length >= width), all from one set of draws."""
    V0 = geom.vertices()
    Vs = geom.vertices(*sample_lines(geom, n, rng))
    note = f"Monte Carlo over {n} draws of the wall-line offsets and slopes"
    status = "inferred" if any(geom.lines[e].inferred for e in geom.edge_lines) else "measured"
    L0, Ls = edge_lengths(V0), edge_lengths(Vs)
    ne = len(geom.edge_lines)
    walls = []
    for k in range(ne):
        a, b = geom.lines[geom.edge_lines[k - 1]], geom.lines[geom.edge_lines[(k + 1) % ne]]
        st = "inferred" if (a.inferred or b.inferred) else "measured"
        walls.append(interval(Ls[:, k], float(L0[k]), "m", "corner-to-corner distance; each corner is the "
                              "intersection of two wall lines fitted on raw LiDAR points; " + note, st))
    area = interval(shoelace(Vs), float(shoelace(V0)), "m2", f"shoelace area of the wall polygon; {note}", status)
    perim = interval(Ls.sum(1), float(L0.sum()), "m", f"sum of wall lengths; {note}", status)
    ext_s, ext0 = Vs.max(1) - Vs.min(1), V0.max(0) - V0.min(0)
    order = np.argsort(-ext0)
    dims = tuple(interval(ext_s[:, i], float(ext0[i]), "m",
                          f"aligned bounding box {'length' if j == 0 else 'width'}; {note}", status)
                 for j, i in enumerate(order))
    return dict(walls=walls, area=area, perimeter=perim, bbox=dims)


def footprint(geoms: list[RoomGeometry], n: int, rng: np.random.Generator) -> Measurement:
    """Area of the union of room polygons (interior floor; wall thickness is not included)."""
    if not geoms:
        return Measurement.missing("m2", "union of room polygons", "no rooms")
    value = unary_union([Polygon(g.vertices()) for g in geoms]).area
    draws = [g.vertices(*sample_lines(g, n, rng)) for g in geoms]
    samples = np.array([unary_union([Polygon(d[i]).buffer(0) for d in draws]).area for i in range(n)])
    status = "inferred" if any(l.inferred for g in geoms for l in g.lines) else "measured"
    return interval(samples, float(value), "m2", f"area of the union of all room polygons (interior floor, "
                    f"walls excluded); Monte Carlo over {n} draws of all wall lines", status)
