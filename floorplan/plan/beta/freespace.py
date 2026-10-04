"""Step 1: the interior free-space map (where a person could stand), on a 2 cm plan raster.

Space-first means we find the rooms from the empty space inside them, not from their walls. Walls are noisy, broken
by doors, hidden behind furniture; the floor area a person walked over is a much more stable signal of "this is a
room". The map combines three observations:

  1. Floor seen by the LiDAR: raw points at floor level, observed from above.
  2. The operator's own path: the camera was carried there, so a disc around it is free.
  3. Ray carving: a LiDAR ray travelled through empty space before it hit something. Where it was lower than the
     wall band (1.0 m) it passed over furniture or floor, so those plan cells are inside the room. This recovers
     floor hidden under tables and behind sofas and counters, which the floor points alone miss.
  4. Furniture holes filled: a sofa or bed hides the floor, leaving a hole. Holes that are mostly surrounded by
     free floor are filled; holes bounded mostly by wall evidence are not (that is wall thickness or a closed
     cupboard, not a room).
Wall evidence is then subtracted so rooms cannot leak through walls, and free-space islands not connected to the
operator's path are dropped: they are seen through glass, reflected in mirrors, or are LiDAR noise.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from floorplan.plan.beta.grid import Grid
from floorplan.plan.beta.params import BetaParams


def disk(radius_cells: float) -> np.ndarray:
    r = int(np.ceil(radius_cells))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= radius_cells * radius_cells + 1e-9


def wall_evidence(points: np.ndarray, normals: np.ndarray, floor_y: float, grid: Grid, p: BetaParams):
    """Cells holding vertical TSDF surface over a tall part of the 1.0-2.0 m height band.

    The height band is the key choice: below 1 m almost every surface is furniture; above 2 m the floor-only
    capture has little data. Requiring vertical extent inside the band (>= wall_min_extent_m of 10 cm height
    bins) rejects desk monitors, shelves and picture frames that only poke into the band; walls are vertical
    over their whole height. Tall furniture (wardrobes, fridges) still counts as wall, see the limitations."""
    h = points[:, 1] - floor_y
    sel = (np.abs(normals[:, 1]) < 0.3) & (h > p.wall_band_lo_m) & (h < p.wall_band_hi_m)
    r, c, ok = grid.index(points[sel][:, [0, 2]])
    hb = np.floor((h[sel] - p.wall_band_lo_m) / 0.1).astype(np.int64)
    key = np.unique((r[ok] * grid.cols + c[ok]) * 16 + hb[ok])          # distinct (cell, 10 cm height bin)
    bins_per_cell = np.bincount(key // 16, minlength=grid.rows * grid.cols).reshape(grid.shape)
    # a wall seen at a grazing angle spreads over 2-3 cells: pool the extent over a 3x3 neighbourhood
    extent = ndi.maximum_filter(bins_per_cell, size=3)
    raw = (bins_per_cell > 0) & (extent * 0.1 >= p.wall_min_extent_m)
    return raw, ndi.binary_dilation(raw, disk(p.wall_dilate_m / p.cell_m))


def floor_observed(raw_points: np.ndarray, raw_ray: np.ndarray, floor_y: float, grid: Grid, p: BetaParams):
    """Cells where the LiDAR hit the floor while looking down (excludes reflections below the floor plane)."""
    h = raw_points[:, 1] - floor_y
    sel = (np.abs(h) < p.floor_tol_m) & (raw_ray[:, 1].astype(np.float32) < -0.05)
    return grid.count(raw_points[sel][:, [0, 2]]) >= 1


def trajectory_disc(traj: np.ndarray, grid: Grid, p: BetaParams) -> np.ndarray:
    m = grid.count(traj[:, [0, 2]]) > 0
    return ndi.binary_dilation(m, disk(p.traj_radius_m / p.cell_m))


def ray_carved(raw_points: np.ndarray, raw_ray: np.ndarray, raw_range: np.ndarray, floor_y: float, grid: Grid,
               p: BetaParams) -> np.ndarray:
    """Cells crossed by a LiDAR ray at a height below p.carve_hi_m (the ray's empty part, before its hit point)."""
    sub = slice(None, None, p.carve_subsample)
    P = raw_points[sub].astype(np.float64)
    D = raw_ray[sub].astype(np.float64)
    R = raw_range[sub].astype(np.float64)
    O = P - D * R[:, None]
    h_o, h_p = O[:, 1] - floor_y, P[:, 1] - floor_y
    with np.errstate(divide="ignore", invalid="ignore"):
        t_s = np.where(h_o <= p.carve_hi_m, 0.0, (h_o - p.carve_hi_m) / (h_o - h_p))
    t_e = 1.0 - p.carve_stop_m / np.maximum(R, 1e-6)          # stop short of the surface that was hit
    ok = (h_p < p.carve_hi_m) & (t_s < t_e)
    O, P, t_s, t_e = O[ok][:, [0, 2]], P[ok][:, [0, 2]], t_s[ok], t_e[ok]
    seg = np.linalg.norm(P - O, axis=1) * (t_e - t_s)
    n = np.ceil(seg / p.cell_m).astype(np.int64) + 1
    out = np.zeros(grid.shape, np.int32)
    for lo in range(0, len(n), 200_000):                       # chunks keep memory bounded
        sl = slice(lo, lo + 200_000)
        rep = np.repeat(np.arange(len(n[sl])), n[sl])
        k = np.arange(len(rep)) - np.repeat(np.cumsum(n[sl]) - n[sl], n[sl])
        t = t_s[sl][rep] + (t_e[sl] - t_s[sl])[rep] * k / np.maximum(n[sl][rep] - 1, 1)
        uv = O[sl][rep] + (P[sl] - O[sl])[rep] * t[:, None]
        out += grid.count(uv)
    return out >= p.carve_min_hits


def fill_furniture_holes(free: np.ndarray, barrier: np.ndarray, p: BetaParams) -> np.ndarray:
    """Fill enclosed non-free regions that are small and mostly bordered by free floor (furniture footprints)."""
    cell_area = p.cell_m ** 2
    holes, n = ndi.label(~free & ~barrier)
    if n == 0:
        return free
    border_labels = np.unique(np.r_[holes[0], holes[-1], holes[:, 0], holes[:, -1]])
    sizes = ndi.sum_labels(np.ones_like(holes), holes, index=np.arange(1, n + 1))
    # fraction of each hole's 1-cell outer ring that is free space (vs. wall evidence)
    ring_free = ndi.sum_labels(free, ndi.grey_dilation(holes, size=3) * (holes == 0), index=np.arange(1, n + 1))
    ring_all = ndi.sum_labels(np.ones_like(holes), ndi.grey_dilation(holes, size=3) * (holes == 0),
                              index=np.arange(1, n + 1))
    frac = ring_free / np.maximum(ring_all, 1)
    ok = (sizes * cell_area <= p.hole_max_m2) & (frac >= p.hole_min_free_frac)
    ok[border_labels[border_labels > 0] - 1] = False
    return free | ok[np.maximum(holes - 1, 0)] & (holes > 0)


def interior_free_space(scene: dict, info: dict, grid: Grid, p: BetaParams) -> dict:
    """Return the maps used downstream: free (bool), wall_raw / barrier (bool), floor_obs, traj_disc, carved."""
    floor_y = info["floor_y"]
    wall_raw, barrier = wall_evidence(scene["points"], scene["normals"], floor_y, grid, p)
    floor_obs = floor_observed(scene["raw_points"], scene["raw_ray"], floor_y, grid, p)
    traj = trajectory_disc(scene["traj"], grid, p)
    carved = ray_carved(scene["raw_points"], scene["raw_ray"], scene["raw_range"], floor_y, grid, p)
    free = floor_obs | traj | carved
    free = ndi.binary_closing(free, disk(p.close_m / p.cell_m)) & ~barrier
    free = fill_furniture_holes(free, barrier, p) & ~barrier
    free = ndi.binary_opening(free, disk(1))
    free = _keep_reached_components(free, traj, p)
    return dict(free=free, wall_raw=wall_raw, barrier=barrier, floor_obs=floor_obs, traj_disc=traj, carved=carved)


def _keep_reached_components(free: np.ndarray, traj: np.ndarray, p: BetaParams) -> np.ndarray:
    """Keep free-space components that touch the operator's path and are not tiny."""
    lab, n = ndi.label(free)
    if n == 0:
        return free
    idx = np.arange(1, n + 1)
    size = ndi.sum_labels(np.ones_like(lab), lab, idx) * p.cell_m ** 2
    reached = ndi.maximum(traj, lab, idx) > 0
    keep = np.r_[False, (size >= p.min_component_m2) & reached]
    return keep[lab]
