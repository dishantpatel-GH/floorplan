"""Per-room floor level and ceiling height, measured on raw LiDAR points inside the room polygon.

Why per room and not the global levels from align.py: bathrooms are often 1-3 cm lower than the rest of the
flat, and a dropped ceiling in a corridor is 20-30 cm lower than the living room. The ceiling-height gate is
1.5 cm per room, so each room's own floor and ceiling must be measured.

Which points: the polygon is shrunk by 20 cm so wall points, skirting and cornices are excluded. A floor point
must be seen from above (ray pointing down), a ceiling point from below (ray pointing up). That rejects table
tops seen from the side and, importantly, phantom points that a glossy floor reflects to below floor level.
If too little of the room's ceiling was seen, the result is "not observed": we never invent a ceiling.
"""
from __future__ import annotations

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon

from floorplan.model import Measurement
from floorplan.plan.beta.grid import PointIndex
from floorplan.plan.beta.params import BetaParams


def _points_in(poly: Polygon, index: PointIndex) -> np.ndarray:
    u0, v0, u1, v1 = poly.bounds
    idx = index.box(u0, u1, v0, v1)
    P = index.points[idx]
    return idx[contains_xy(poly, P[:, 0], P[:, 2])]


def candidate_levels(h: np.ndarray, min_frac: float = 0.1) -> list[float]:
    """Peaks of a 1 cm height histogram holding at least min_frac of the highest peak."""
    bins = np.arange(h.min() - 0.01, h.max() + 0.02, 0.01)
    hist, edges = np.histogram(h, bins)
    sm = np.convolve(hist, np.ones(3) / 3, mode="same")
    peaks = np.nonzero((sm >= np.roll(sm, 1)) & (sm > np.roll(sm, -1)) & (sm >= min_frac * sm.max()))[0]
    return [float(edges[i] + 0.005) for i in peaks]


def refine_level(h: np.ndarray, start: float, tol: float) -> tuple[float, float, np.ndarray]:
    """Trimmed mean within +-tol of a starting level (two passes): (level, rmse, inlier mask)."""
    est = start
    for _ in range(2):
        inl = np.abs(h - est) < tol
        est = float(np.mean(h[inl]))
    inl = np.abs(h - est) < tol
    return est, float(np.std(h[inl] - est)), inl


class LevelPoints:
    """The only raw points the level estimates need, selected once per capture (a few % of all points):
    floor candidates (near the global floor, seen from above) and ceiling candidates (high, seen from below)."""

    def __init__(self, raw_points: np.ndarray, raw_ray: np.ndarray, raw_range: np.ndarray, floor_y: float,
                 p: BetaParams):
        h = raw_points[:, 1] - floor_y
        ray_y = raw_ray[:, 1].astype(np.float32)
        keep = np.arange(len(h)) % p.level_subsample == 0       # plenty for a 1 cm histogram; much faster
        f = keep & (np.abs(h) < p.floor_tol_m) & (ray_y < -0.05)
        c = keep & (h > p.ceiling_min_h_m - p.floor_tol_m) & (ray_y > 0.05)
        self.floor = PointIndex(raw_points[f], raw_ray[f], raw_range[f])
        self.ceiling = PointIndex(raw_points[c], raw_ray[c], raw_range[c])


def room_levels(poly: Polygon, pts: LevelPoints, floor_y: float, p: BetaParams):
    """Return (floor_level_y, floor_sigma, ceiling Measurement) for one room."""
    inner = poly.buffer(-p.ceiling_shrink_m, join_style="mitre")
    if inner.is_empty or inner.area < 0.2:
        inner = poly
    F = pts.floor.points[_points_in(inner, pts.floor)]
    if len(F) < 100:
        floor_level, floor_se = floor_y, p.inferred_sigma_m / 2
    else:
        best = max(candidate_levels(F[:, 1]), key=lambda c: _patches(F[np.abs(F[:, 1] - c) < p.level_tol_m], p))
        floor_level, rmse, inl = refine_level(F[:, 1], best, p.level_tol_m)
        floor_se = rmse / np.sqrt(_patches(F[inl], p))
    ceiling = _ceiling(inner, pts.ceiling.points[_points_in(inner, pts.ceiling)], floor_level, floor_se, p)
    return floor_level, floor_se, ceiling


def _patches(P: np.ndarray, p: BetaParams) -> int:
    return max(len(np.unique(np.floor(P[:, [0, 2]] / p.patch_m), axis=0)), 1)


def _ceiling(inner: Polygon, P: np.ndarray, floor_level: float, floor_se: float, p: BetaParams) -> Measurement:
    method = "raw LiDAR points seen from below inside the room (shrunk 20 cm), peak + trimmed mean, minus floor"
    sel = P[:, 1] - floor_level > p.ceiling_min_h_m
    cells_total = max(inner.area / p.patch_m ** 2, 1.0)
    cover = min(_patches(P[sel], p) / cells_total, 1.0) if sel.any() else 0.0
    if sel.sum() < 200 or cover < p.ceiling_min_cover:
        return Measurement.missing("m", method, f"ceiling not observed (coverage {cover:.0%} of the room)")
    C = P[sel]
    # several ceiling levels are common (bulkheads, dropped ceilings over kitchens and bathrooms); report the
    # level covering the largest part of the room, by area (10 cm patches), not by point count, because
    # point density depends on how close the phone was
    levels = sorted(((min(_patches(C[np.abs(C[:, 1] - c) < p.level_tol_m], p) / cells_total, 1.0), c)
                     for c in candidate_levels(C[:, 1])), reverse=True)
    level, rmse, inl = refine_level(C[:, 1], levels[0][1], p.level_tol_m)
    ceil_se = rmse / np.sqrt(_patches(C[inl], p))
    sigma = np.sqrt(ceil_se ** 2 + floor_se ** 2 + 2 * p.lidar_sigma_m ** 2 + p.drift_vertical_sigma_m ** 2)
    others = [f"{c - floor_level:.2f} m ({f:.0%})" for f, c in levels[1:] if f >= 0.1 and abs(c - level) > 0.05]
    note = f"; other ceiling levels seen: {', '.join(others)}" if others else ""
    m = Measurement.from_sigma(float(level - floor_level), float(sigma), "m",
                               f"{method}; this level covers {levels[0][0]:.0%} of the room{note}")
    return _double_layer(m, levels, floor_level, p)


def _double_layer(m: Measurement, levels: list[tuple[float, float]], floor_level: float,
                  p: BetaParams) -> Measurement:
    """v2 B-6 (alpha A-14): a second large ceiling layer 5-20 cm from the chosen one is either a real step or the
    same ceiling seen on two passes at different heights (vertical drift). The points cannot tell which, so the
    value stays the dominant layer, the status becomes `inferred` and the interval is widened to cover both.
    Without it a ceiling 11 cm off was reported as measured +-1.7 cm on ARKit poses (judge J-I-7)."""
    if not p.ceiling_double_layer:
        return m
    top_cover, top = levels[0]
    rivals = [c for f, c in levels[1:] if 0.05 < abs(c - top) <= p.ceiling_ambiguous_m
              and f >= p.ceiling_rival_frac * top_cover]
    if not rivals:
        return m
    lo = min([m.lo] + [c - floor_level for c in rivals])
    hi = max([m.hi] + [c - floor_level for c in rivals])
    return Measurement(m.value, float(lo), float(hi), "m",
                       m.method + "; interval spans a second ceiling layer (a step, or drift between passes)",
                       "inferred")
