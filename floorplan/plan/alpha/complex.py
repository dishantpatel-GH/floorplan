"""Step 3: the cell complex. Extend every wall line across the whole plan; the lines cut the plane into cells.

With Manhattan lines the complex is simply a rectilinear grid with uneven spacing: cell (i, j) is the rectangle
[us[i], us[i+1]] x [vs[j], vs[j+1]]. Every room boundary we can output is a chain of cell edges, so every polygon
vertex lies exactly on two detected wall lines. That is the point of "boundary first": the room shape is assembled
from measured walls instead of traced from a noisy raster mask (which would give staircase edges and rounded corners).

For each cell we store how much of it has interior evidence; for each edge between two cells we store how much of
it is backed by wall surface (coverage). Coverage makes a boundary cheap along walls in the inside/outside labelling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from floorplan.plan.alpha.evidence import Grid
from floorplan.plan.alpha.lines import Line
from floorplan.plan.alpha.params import AlphaParams


@dataclass
class CellComplex:
    grid: Grid
    us: np.ndarray                 # (I+1,) cell boundaries along u
    vs: np.ndarray                 # (J+1,) cell boundaries along v
    u_lines: list[Line | None]     # line on each u boundary (None = scene border)
    v_lines: list[Line | None]
    bu: np.ndarray                 # pixel index of each u boundary
    bv: np.ndarray
    interior: np.ndarray           # (I, J) fraction of the cell's pixels with interior evidence
    traj: np.ndarray               # (I, J) bool: the camera passed through the cell
    cov_u: np.ndarray              # (I+1, J) wall coverage of the edge on u boundary i, row j
    cov_v: np.ndarray              # (I, J+1) wall coverage of the edge on v boundary j, column i

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.us) - 1, len(self.vs) - 1

    def area(self) -> np.ndarray:
        return np.outer(np.diff(self.us), np.diff(self.vs))

    def cell_of(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = np.clip(np.searchsorted(self.us, u) - 1, 0, len(self.us) - 2)
        j = np.clip(np.searchsorted(self.vs, v) - 1, 0, len(self.vs) - 2)
        return i, j

    def label_raster(self, cell_labels: np.ndarray) -> np.ndarray:
        """Paint a per-cell integer label onto the pixel grid."""
        iu = np.clip(np.searchsorted(self.bu, np.arange(self.grid.nu), side="right") - 1, 0, len(self.bu) - 2)
        iv = np.clip(np.searchsorted(self.bv, np.arange(self.grid.nv), side="right") - 1, 0, len(self.bv) - 2)
        return cell_labels[iu[:, None], iv[None, :]]


def _integral(img: np.ndarray) -> np.ndarray:
    S = np.zeros((img.shape[0] + 1, img.shape[1] + 1), np.float64)
    S[1:, 1:] = img.cumsum(0).cumsum(1)
    return S


def _box_sums(S: np.ndarray, bu: np.ndarray, bv: np.ndarray) -> np.ndarray:
    """Sum of the image over every cell [bu[i], bu[i+1]) x [bv[j], bv[j+1]) via the integral image."""
    a, b = bu[:-1, None], bu[1:, None]
    c, d = bv[None, :-1], bv[None, 1:]
    return S[b, d] - S[a, d] - S[b, c] + S[a, c]


def _edge_fraction(profile: np.ndarray | None, b: np.ndarray) -> np.ndarray:
    """Mean of a boolean along-line profile over each interval [b[k], b[k+1])."""
    if profile is None:
        return np.zeros(len(b) - 1)
    c = np.r_[0, np.cumsum(profile.astype(np.float64))]
    n = np.maximum(b[1:] - b[:-1], 1)
    return (c[b[1:]] - c[b[:-1]]) / n


def build_complex(lines: dict[str, list[Line]], grid: Grid, evidence: dict[str, np.ndarray],
                  p: AlphaParams) -> CellComplex:
    u0, u1, v0, v1 = grid.extent
    xl = sorted(lines["x"], key=lambda L: L.offset)
    zl = sorted(lines["z"], key=lambda L: L.offset)
    us = np.array([u0] + [L.offset for L in xl] + [u1])
    vs = np.array([v0] + [L.offset for L in zl] + [v1])
    u_lines: list[Line | None] = [None] + xl + [None]
    v_lines: list[Line | None] = [None] + zl + [None]
    bu = np.array([grid.to_px(u, 0) for u in us])
    bv = np.array([grid.to_px(v, 1) for v in vs])
    npix = np.maximum(np.outer(np.diff(bu), np.diff(bv)), 1)
    interior = _box_sums(_integral(evidence["any"]), bu, bv) / npix
    traj = _box_sums(_integral(evidence["traj_core"]), bu, bv) > 0
    cov_u = np.stack([_edge_fraction(None if L is None else L.occ, bv) for L in u_lines])
    cov_v = np.stack([_edge_fraction(None if L is None else L.occ, bu) for L in v_lines])
    return CellComplex(grid, us, vs, u_lines, v_lines, bu, bv, interior, traj, cov_u, cov_v.T)
