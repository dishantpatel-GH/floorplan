"""Step 1 and the inputs of step 4: turn the 3-D scene into 2-D evidence on a plan raster.

Two kinds of evidence drive everything downstream:
  * WALL evidence: vertical surfaces seen by the LiDAR, split into the two Manhattan families (normals along x or
    along z) and kept with the SIGN of their normal. TSDF normals point toward the side the camera observed, so the
    sign says which side of a wall face is the room. A partition wall therefore shows up as two parallel faces with
    opposite signs, which is what lets us measure wall thickness and room interiors separately.
  * INTERIOR evidence: places we know are inside the dwelling: the visible floor, the 2-D footprint of every LiDAR
    ray (the beam travelled through empty space before it hit something), low horizontal surfaces such as beds and
    tables, and the camera path (the operator walked there). Unobserved space carries no evidence either way; the
    labelling step treats it as "probably outside" with a small, explicit cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from floorplan.plan.alpha.params import AlphaParams


@dataclass(frozen=True)
class Grid:
    """Axis-aligned plan raster. Arrays are indexed [iu, iv] (u = x first, v = z second)."""
    u0: float
    v0: float
    res: float
    nu: int
    nv: int

    def index(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        iu = np.floor((np.asarray(u) - self.u0) / self.res).astype(np.int64)
        iv = np.floor((np.asarray(v) - self.v0) / self.res).astype(np.int64)
        ok = (iu >= 0) & (iu < self.nu) & (iv >= 0) & (iv < self.nv)
        return iu, iv, ok

    def to_px(self, coord: float, axis: int) -> int:
        """Plan coordinate -> nearest pixel boundary index along axis 0 (u) or 1 (v)."""
        origin, n = (self.u0, self.nu) if axis == 0 else (self.v0, self.nv)
        return int(np.clip(round((coord - origin) / self.res), 0, n))

    def along_coords(self, axis: int) -> np.ndarray:
        """Pixel-centre coordinates along axis 0 (u) or 1 (v)."""
        origin, n = (self.u0, self.nu) if axis == 0 else (self.v0, self.nv)
        return origin + (np.arange(n) + 0.5) * self.res

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return self.u0, self.u0 + self.nu * self.res, self.v0, self.v0 + self.nv * self.res


def make_grid(points: np.ndarray, p: AlphaParams) -> Grid:
    """Raster covering the scene (robust 0.1-99.9 percentile extent, so a few far outliers do not blow it up)."""
    lo = np.percentile(points[:, [0, 2]], 0.1, axis=0) - p.margin_m
    hi = np.percentile(points[:, [0, 2]], 99.9, axis=0) + p.margin_m
    n = np.ceil((hi - lo) / p.res_m).astype(int)
    return Grid(float(lo[0]), float(lo[1]), p.res_m, int(n[0]), int(n[1]))


@dataclass
class WallPoints:
    """Wall-like TSDF surface points of one Manhattan family.

    family 'x': normal along +-x, so the wall is a line u = const; offset = u, along = v.
    family 'z': normal along +-z, so the wall is a line v = const; offset = v, along = u."""
    family: str
    offset: np.ndarray
    along: np.ndarray
    sign: np.ndarray      # +1 / -1: direction of the normal along the family axis (= side the camera saw)
    height: np.ndarray    # metres above the global floor


def height_band(floor_y: float, ceiling_y: float | None, p: AlphaParams) -> tuple[float, float]:
    """Heights (above floor) where wall evidence is collected. Below: skirting and clutter; above: ceiling."""
    hi = p.band_hi_m if ceiling_y is None else min(p.band_hi_m, ceiling_y - floor_y - 0.3)
    return p.band_lo_m, hi


def wall_points(points: np.ndarray, normals: np.ndarray, floor_y: float, ceiling_y: float | None,
                p: AlphaParams) -> dict[str, WallPoints]:
    h = points[:, 1] - floor_y
    lo, hi = height_band(floor_y, ceiling_y, p)
    nh = np.hypot(normals[:, 0], normals[:, 2])
    base = (np.abs(normals[:, 1]) < p.wall_max_ny) & (h > lo) & (h < hi) & (nh > 1e-6)
    cos_tol = np.cos(np.radians(p.axis_tol_deg))
    out = {}
    for fam, ax, other in (("x", 0, 2), ("z", 2, 0)):
        m = base & (np.abs(normals[:, ax]) >= cos_tol * nh)
        out[fam] = WallPoints(fam, points[m, ax].astype(np.float64), points[m, other].astype(np.float64),
                              np.sign(normals[m, ax]).astype(np.int8), h[m].astype(np.float64))
    return out


def count_image(i: np.ndarray, j: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """2-D histogram of integer indices (already in range). bincount on the flat index is ~10x faster than
    np.add.at and gives identical counts."""
    flat = np.bincount(i.astype(np.int64) * shape[1] + j, minlength=shape[0] * shape[1])
    return flat.reshape(shape).astype(np.int32)


def _rasterize(grid: Grid, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    iu, iv, ok = grid.index(u, v)
    return count_image(iu[ok], iv[ok], (grid.nu, grid.nv))


def ray_free_space(scene: dict, grid: Grid, p: AlphaParams, chunk: int = 50_000) -> np.ndarray:
    """2-D footprint of LiDAR rays: count of rays that crossed each pixel before hitting a surface.

    A ray from the camera to a hit point proves the space in between is empty (at the ray's height). Projected to
    the plan, it marks interior space even where the floor itself was hidden (under tables, behind the sofa).
    We sample n_rays rays with a fixed seed, so the result is deterministic."""
    P, r = scene["raw_points"], scene["raw_range"]
    rng = np.random.default_rng(p.seed)
    sel = rng.choice(len(P), size=min(p.n_rays, len(P)), replace=False)
    hit = P[sel][:, [0, 2]].astype(np.float64)
    ray = scene["raw_ray"][sel].astype(np.float64)
    cam = hit - (r[sel, None].astype(np.float64) * ray)[:, [0, 2]]
    d = hit - cam
    L = np.linalg.norm(d, axis=1)
    L_free = np.maximum(L - p.ray_stop_short_m, 0.0)
    counts = np.zeros((grid.nu, grid.nv), np.int32)
    n_steps = int(np.ceil(L_free.max() / grid.res)) + 1 if len(L_free) else 0
    t = np.arange(n_steps) * grid.res
    for s in range(0, len(sel), chunk):
        dd = d[s:s + chunk] / np.maximum(L[s:s + chunk, None], 1e-9)
        keep = t[None, :] <= L_free[s:s + chunk, None]
        uu = cam[s:s + chunk, 0, None] + dd[:, 0, None] * t[None, :]
        vv = cam[s:s + chunk, 1, None] + dd[:, 1, None] * t[None, :]
        iu, iv, ok = grid.index(uu[keep], vv[keep])
        counts += count_image(iu[ok], iv[ok], counts.shape)
    return counts


def interior_evidence(scene: dict, info: dict, grid: Grid, p: AlphaParams) -> dict[str, np.ndarray]:
    """Boolean rasters of interior evidence (and the union, 'any')."""
    P, N = scene["points"], scene["normals"]
    h = P[:, 1] - info["floor_y"]
    up = N[:, 1] > 0.9
    floor = _rasterize(grid, *P[up & (np.abs(h) < p.floor_band_m)][:, [0, 2]].T) > 0
    tops = _rasterize(grid, *P[up & (h >= p.floor_band_m) & (h < p.furniture_top_max_m)][:, [0, 2]].T) > 0
    free = ray_free_space(scene, grid, p) >= 2
    traj_px = _rasterize(grid, scene["traj"][:, 0], scene["traj"][:, 2]) > 0
    r_px = max(1, int(round(p.traj_radius_m / grid.res)))
    yy, xx = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
    traj = ndimage.binary_dilation(traj_px, structure=(xx ** 2 + yy ** 2) <= r_px ** 2)
    floor = ndimage.binary_closing(floor, iterations=1)       # fill single-pixel sampling holes
    return dict(floor=floor, tops=tops, free=free, traj=traj, traj_core=traj_px, any=floor | tops | free | traj)


def wall_density(walls: dict[str, WallPoints], grid: Grid) -> np.ndarray:
    """Count of wall points per pixel (both families): the background image of the debug figure."""
    img = np.zeros((grid.nu, grid.nv), np.int32)
    for wp in walls.values():
        u, v = (wp.offset, wp.along) if wp.family == "x" else (wp.along, wp.offset)
        img += _rasterize(grid, u, v)
    return img
