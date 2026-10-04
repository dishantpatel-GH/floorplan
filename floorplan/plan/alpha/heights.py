"""Per-room floor level and ceiling height, measured on raw points inside the room polygon.

Why per room and not the global levels from plan/align.py: bathrooms are often a step up or down, and a ceiling
can be lowered (soffits, suspended ceilings). The Part 2 gate is per room (<= 1.5 cm), so the room's OWN floor and
ceiling are measured inside its polygon, shrunk by ceil_shrink_m to stay clear of walls and crown moulding.

  floor level   = robust level of raw points near the floor inside the room;
  ceiling level = the HIGHEST strong horizontal down-facing layer inside the room (lower layers are lamps, cabinet
                  bottoms or a door head), then refined on raw points;
  height        = ceiling - floor, sigma^2 = SE_c^2 + SE_f^2 + I_c^2 + I_f^2 + 2 lidar_sigma^2, where I is the
                  surface inconsistency (scatter beyond sensor noise, e.g. two passes at different heights).
If the ceiling was not seen (too few points, or seen over too little of the room), the height is reported as
not_observed with no value: we never invent a ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon

from floorplan.model import Measurement
from floorplan.plan.alpha.measure import inconsistency, robust_fit
from floorplan.plan.alpha.params import AlphaParams


@dataclass
class LevelData:
    """Raw points near the floor and near the ceiling, plus TSDF down-facing points (they carry normals)."""
    floor_pts: np.ndarray
    high_pts: np.ndarray
    ceil_tsdf: np.ndarray


def level_data(scene: dict, floor_y: float) -> LevelData:
    R = scene["raw_points"]
    h = R[:, 1] - floor_y
    P, N = scene["points"], scene["normals"]
    down = (N[:, 1] < -0.9) & (P[:, 1] - floor_y > 1.9)
    return LevelData(R[np.abs(h) < 0.2].astype(np.float64), R[h > 1.9].astype(np.float64), P[down].astype(np.float64))


def _inside(poly: Polygon, pts: np.ndarray) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    m = (pts[:, 0] > minx) & (pts[:, 0] < maxx) & (pts[:, 2] > miny) & (pts[:, 2] < maxy)
    out = np.zeros(len(pts), bool)
    out[m] = shapely.contains_xy(poly, pts[m, 0], pts[m, 2])
    return out


def _plan_patches(pts: np.ndarray, p: AlphaParams) -> np.ndarray:
    return np.floor(pts[:, 0] / p.patch_m).astype(np.int64) * 100_000 + np.floor(pts[:, 2] / p.patch_m).astype(np.int64)


def _level(pts: np.ndarray, p: AlphaParams):
    """Robust level (absolute y) of a set of points believed to lie on one horizontal surface."""
    y = pts[:, 1]
    centre = float(np.median(y))
    fit = robust_fit(y - centre, _plan_patches(pts, p), band=0.2)
    return None if fit is None else (centre + fit.value, fit)


def _ceiling_layers(y: np.ndarray, bin_m: float = 0.05) -> list[tuple[float, float]]:
    """Horizontal down-facing layers as (level, share of points), highest first. 5 cm bins merge one surface."""
    edges = np.arange(y.min() - bin_m, y.max() + 2 * bin_m, bin_m)
    h, e = np.histogram(y, bins=edges)
    out = []
    for k in np.flatnonzero(h >= 0.05 * h.sum()):
        if out and abs(out[-1][0] - (e[k] + e[k + 1]) / 2) <= bin_m + 1e-9:     # neighbouring bins: one layer
            lvl, share = out[-1]
            out[-1] = ((lvl * share + (e[k] + e[k + 1]) / 2 * h[k] / h.sum()) / (share + h[k] / h.sum()),
                       share + h[k] / h.sum())
        else:
            out.append(((e[k] + e[k + 1]) / 2, h[k] / h.sum()))
    return sorted(out, key=lambda t: -t[0])


def _pick_ceiling(layers: list[tuple[float, float]]) -> float:
    """The highest layer that holds at least 30% as many points as the biggest one: the true ceiling, not a lamp
    or a cabinet bottom, but also not a small skylight-like patch above a mostly lowered ceiling."""
    biggest = max(s for _, s in layers)
    return next(lvl for lvl, s in layers if s >= 0.3 * biggest)


@dataclass
class RoomLevels:
    floor_y: float
    floor_se: float
    ceiling: Measurement


def room_levels(poly: Polygon, data: LevelData, global_floor: float, p: AlphaParams) -> RoomLevels:
    core = poly.buffer(-p.ceil_shrink_m, join_style="mitre")
    if core.is_empty:
        core = poly
    fl = data.floor_pts[_inside(core, data.floor_pts)]
    lvl = _level(fl, p) if len(fl) >= p.ceil_min_points else None
    floor_y, floor_se = ((lvl[0], float(np.hypot(lvl[1].se, inconsistency(lvl[1].rmse, p)))) if lvl
                         else (global_floor, p.inferred_sigma_m))

    method = "ceiling level minus room floor level, raw LiDAR inside room polygon shrunk 0.2 m"
    down = data.ceil_tsdf[_inside(core, data.ceil_tsdf)]
    if len(down) < p.ceil_min_points // 3:
        return RoomLevels(floor_y, floor_se, Measurement.missing("m", method, "ceiling not observed in this room"))
    layers = _ceiling_layers(down[:, 1])
    c0 = _pick_ceiling(layers)
    others = [f"{lvl - floor_y:.2f} m ({100 * sh:.0f}%)" for lvl, sh in layers if abs(lvl - c0) > 0.1]
    if others:
        method += "; other down-facing layers: " + ", ".join(others)
    hp = data.high_pts[(np.abs(data.high_pts[:, 1] - c0) < 0.03)]
    hp = hp[_inside(core, hp)]
    cover = len(np.unique(_plan_patches(hp, p))) * p.patch_m ** 2 / max(core.area, 1e-9)
    clvl = _level(hp, p) if len(hp) >= p.ceil_min_points else None
    if clvl is None or cover < p.ceil_min_cover:
        why = f"ceiling seen over {100 * cover:.0f}% of the room only" if clvl else "too few ceiling points"
        return RoomLevels(floor_y, floor_se, Measurement.missing("m", method, why))
    ceil_se = float(np.hypot(clvl[1].se, inconsistency(clvl[1].rmse, p)))
    sig = float(np.sqrt(ceil_se ** 2 + floor_se ** 2 + 2 * p.lidar_sigma_m ** 2))
    m = Measurement.from_sigma(clvl[0] - floor_y, sig, method=method)
    chosen = next(sh for lvl, sh in layers if lvl == c0)
    rivals = [lvl for lvl, sh in layers if 0.05 < abs(lvl - c0) <= p.ceil_ambiguous_m and sh >= 0.3 * chosen]
    if rivals:      # two nearby ceiling layers: a real step or vertical drift between passes; we cannot tell
        lo = min([m.lo] + [r - floor_y for r in rivals])
        hi = max([m.hi] + [r - floor_y for r in rivals])
        m = Measurement(m.value, lo, hi, "m", method + "; interval spans a second ceiling layer (step or drift)",
                        "inferred")
    return RoomLevels(floor_y, floor_se, m)
