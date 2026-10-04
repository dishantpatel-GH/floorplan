"""Step 2: find axis-aligned wall lines by 1-D peak finding.

Why 1-D: after Manhattan alignment (plan/align.py) every wall face of family 'x' is a line u = const. Projecting all
x-family wall points onto u turns "find the walls" into "find the peaks of a histogram", which is fast, has no
random sampling (unlike RANSAC) and finds short walls next to long ones (unlike a global Hough vote, which is
dominated by the longest walls). We do it separately for each normal sign so the two faces of a 10 cm partition
wall stay two lines instead of blurring into one.

Each line keeps an along-line occupancy profile (2 cm bins): where along its length real wall surface was seen,
and where TALL surface was seen (above furniture height). The cell complex uses the first to decide whether a cell
boundary is backed by a wall; door detection looks for gaps in it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from floorplan.plan.alpha.evidence import Grid, WallPoints
from floorplan.plan.alpha.params import AlphaParams


@dataclass
class Line:
    family: str            # 'x' -> u = offset ; 'z' -> v = offset
    offset: float
    signs: tuple[int, ...]  # normal sign(s) of the faces on this line (+1: room on the + side)
    occ: np.ndarray        # bool per along-pixel: wall surface seen (0.3-2.0 m band)
    occ_tall: np.ndarray   # bool per along-pixel: wall surface seen above furniture height
    n_points: int


def _along_axis(family: str) -> int:
    """Plan axis that runs ALONG a wall of this family (x-family walls run along v)."""
    return 1 if family == "x" else 0


def _occupancy(wp: WallPoints, mask: np.ndarray, grid: Grid, p: AlphaParams) -> tuple[np.ndarray, np.ndarray]:
    axis = _along_axis(wp.family)
    n = grid.nv if axis == 1 else grid.nu
    origin = grid.v0 if axis == 1 else grid.u0
    idx = np.floor((wp.along[mask] - origin) / grid.res).astype(np.int64)
    ok = (idx >= 0) & (idx < n)
    cnt = np.bincount(idx[ok], minlength=n)
    tall = np.bincount(idx[ok & (wp.height[mask] >= p.high_from_m)], minlength=n)
    return cnt >= p.occ_min_points, tall >= p.occ_high_min_points


def _peaks(offsets: np.ndarray, p: AlphaParams) -> np.ndarray:
    if len(offsets) == 0:
        return np.array([])
    edges = np.arange(offsets.min() - 3 * p.peak_bin_m, offsets.max() + 3 * p.peak_bin_m, p.peak_bin_m)
    h, e = np.histogram(offsets, bins=edges)
    h = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
    pk, _ = find_peaks(h, height=p.peak_min_points, distance=max(1, int(round(p.merge_tol_m / p.peak_bin_m))))
    return (e[pk] + e[pk + 1]) / 2


def _lines_one_sign(wp: WallPoints, sign: int, grid: Grid, p: AlphaParams) -> list[Line]:
    s = wp.sign == sign
    out = []
    for c in _peaks(wp.offset[s], p):
        near = s & (np.abs(wp.offset - c) < p.line_tol_m)
        c_ref = float(np.median(wp.offset[near]))          # sub-centimetre line position
        near = s & (np.abs(wp.offset - c_ref) < p.line_tol_m)
        occ, tall = _occupancy(wp, near, grid, p)
        if occ.sum() * grid.res >= p.min_line_support_m:
            out.append(Line(wp.family, c_ref, (sign,), occ, tall, int(near.sum())))
    return out


def _merge_opposite(lines: list[Line], p: AlphaParams) -> list[Line]:
    """Opposite-sign lines closer than merge_tol are one thin surface seen from both sides (e.g. a glass panel)."""
    lines = sorted(lines, key=lambda L: L.offset)
    out: list[Line] = []
    for L in lines:
        if out and abs(L.offset - out[-1].offset) < p.merge_tol_m and L.signs != out[-1].signs:
            a = out[-1]
            w = a.n_points + L.n_points
            out[-1] = Line(a.family, (a.offset * a.n_points + L.offset * L.n_points) / w,
                           tuple(sorted(set(a.signs) | set(L.signs))), a.occ | L.occ, a.occ_tall | L.occ_tall, w)
        else:
            out.append(L)
    return out


def detect_lines(walls: dict[str, WallPoints], grid: Grid, p: AlphaParams) -> dict[str, list[Line]]:
    return {fam: _merge_opposite(_lines_one_sign(wp, +1, grid, p) + _lines_one_sign(wp, -1, grid, p), p)
            for fam, wp in walls.items()}


def gaps_of(occ: np.ndarray) -> list[tuple[int, int]]:
    """Runs of False strictly inside the occupied extent: list of [start, end) pixel ranges."""
    idx = np.flatnonzero(occ)
    if len(idx) < 2:
        return []
    g = np.diff(idx) - 1
    return [(int(idx[k] + 1), int(idx[k + 1])) for k in np.flatnonzero(g > 0)]
