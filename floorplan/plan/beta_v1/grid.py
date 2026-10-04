"""Plan raster and a fast box query on millions of raw LiDAR points.

Why a raster at all: room segmentation (distance transform, watershed) is an image operation. Why a separate point
index: final dimensions are measured on raw points (D-007), and a capture holds up to 16 M of them, so each wall
must touch only the points in its own narrow band. Sorting once by x and once by z turns each band query into a
binary search plus a small mask instead of a 16 M-element comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Plan raster: cell (row, col) covers u in [u0 + col*c, ...) and v in [v0 + row*c, ...); (u, v) = (x, z)."""
    u0: float
    v0: float
    cell: float
    rows: int
    cols: int

    @staticmethod
    def around(uv: np.ndarray, cell: float, margin: float = 0.5) -> "Grid":
        lo = uv.min(0) - margin
        hi = uv.max(0) + margin
        cols, rows = np.ceil((hi - lo) / cell).astype(int)
        return Grid(float(lo[0]), float(lo[1]), cell, int(rows), int(cols))

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def index(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, col, inside) for plan points."""
        c = np.floor((uv[:, 0] - self.u0) / self.cell).astype(np.int64)
        r = np.floor((uv[:, 1] - self.v0) / self.cell).astype(np.int64)
        ok = (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.cols)
        return r, c, ok

    def count(self, uv: np.ndarray) -> np.ndarray:
        """Number of points per cell."""
        r, c, ok = self.index(uv)
        out = np.zeros(self.rows * self.cols, np.int32)
        np.add.at(out, r[ok] * self.cols + c[ok], 1)
        return out.reshape(self.shape)

    def centers(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return np.stack([self.u0 + (cols + 0.5) * self.cell, self.v0 + (rows + 0.5) * self.cell], 1)

    def u_of(self, col: float) -> float:
        return self.u0 + col * self.cell

    def v_of(self, row: float) -> float:
        return self.v0 + row * self.cell

    def col_of(self, u: float) -> float:
        return (u - self.u0) / self.cell

    def row_of(self, v: float) -> float:
        return (v - self.v0) / self.cell


class PointIndex:
    """Raw points sorted by x and by z, for axis-aligned box queries in plan coordinates."""

    def __init__(self, points: np.ndarray, ray: np.ndarray, rng: np.ndarray):
        self._by = {}
        for axis, col in (("u", 0), ("v", 2)):
            order = np.argsort(points[:, col], kind="stable")
            self._by[axis] = (points[order, col].copy(), order)
        self.points, self.ray, self.range = points, ray, rng

    def box(self, u_lo: float, u_hi: float, v_lo: float, v_hi: float) -> np.ndarray:
        """Indices of points with u in [u_lo, u_hi] and v in [v_lo, v_hi]; slices along the narrower side."""
        if (u_hi - u_lo) <= (v_hi - v_lo):
            key, other, lo, hi, olo, ohi = "u", 2, u_lo, u_hi, v_lo, v_hi
        else:
            key, other, lo, hi, olo, ohi = "v", 0, v_lo, v_hi, u_lo, u_hi
        vals, order = self._by[key]
        i0, i1 = np.searchsorted(vals, [lo, hi])
        idx = order[i0:i1]
        o = self.points[idx, other]
        return idx[(o >= olo) & (o <= ohi)]
