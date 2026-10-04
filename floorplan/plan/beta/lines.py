"""v2 (B-1): long Manhattan wall lines as room-SEPARATION evidence, ported from plan_alpha's lines.py.

Why this exists. v1 only treats a vertical surface as wall when it covers >= 40 cm of the 1.0-2.0 m band (D-B2),
so that desks and sofas are not walls. A capture aimed down (floor_only) sees some partitions only below 1 m; free
space then leaks through them and the watershed merges two rooms (judge J-I-4). Lowering the band fixes the merge
but turns furniture into walls and shrinks rooms the same way in both captures: repeatable and wrong (J-I-5).

What we take from alpha instead: 1-D peak lines of vertical TSDF surface over 0.3-2.0 m, found separately for each
normal sign (so the two faces of a partition stay two lines), with an along-line occupancy profile in 2 cm bins.
A line is used ONLY to cut the free space for the segmentation (segment.py); it never removes free space from the
plan, so it cannot shrink a room the way a lower band does.

Furniture discrimination (the J-I-5 lesson): a partition wall is seen from both rooms, so it shows TWO opposite-
facing faces 5-40 cm apart that overlap along the wall. A bed side, a counter front or a wardrobe door shows one
face (its back is against a wall or is hidden). A partition seen from one side only (the floor_only capture saw
one face, up to ~1.3 m) is still a wall if that face rises from near the floor past furniture height: we require
>= sep_min_extent_m (0.8 m) of the 0.3-2.0 m band, which no bed, sofa, desk or counter (<= ~1 m tall) reaches.
Tall furniture (wardrobes) does reach it, but v1's band rule already treats it as wall, so this adds no new
false-wall class. With `barrier_pair_required` a run is a cut where it is paired OR tall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from floorplan.plan.beta.grid import Grid
from floorplan.plan.beta.params import BetaParams


@dataclass
class SepLine:
    """One face line: axis 0 = constant u (the wall runs along v), axis 1 = constant v; sign = normal direction."""
    axis: int
    offset: float
    sign: int
    occ: np.ndarray            # bool per 2 cm along-bin (grid columns for axis 1, grid rows for axis 0)
    tall: np.ndarray           # bool per bin: the face spans >= sep_min_extent_m of height (pooled over 3 bins)


def face_points(points: np.ndarray, normals: np.ndarray, floor_y: float, p: BetaParams):
    """Per axis: (offset, along, sign, height) of vertical TSDF points in the band whose normal is Manhattan."""
    h = points[:, 1] - floor_y
    nh = np.hypot(normals[:, 0], normals[:, 2])
    base = (np.abs(normals[:, 1]) < 0.3) & (h > p.sep_band_lo_m) & (h < p.sep_band_hi_m) & (nh > 1e-6)
    cos_tol = np.cos(np.radians(15.0))
    out = {}
    for axis, (ax, other) in enumerate(((0, 2), (2, 0))):
        m = base & (np.abs(normals[:, ax]) >= cos_tol * nh)
        out[axis] = (points[m, ax].astype(np.float64), points[m, other].astype(np.float64),
                     np.sign(normals[m, ax]).astype(np.int8), h[m].astype(np.float64))
    return out


def _peaks(offsets: np.ndarray, p: BetaParams) -> np.ndarray:
    """Centres of 1 cm histogram peaks with >= sep_peak_min_points voxels (alpha's _peaks, same constants)."""
    if len(offsets) == 0:
        return np.array([])
    edges = np.arange(offsets.min() - 0.03, offsets.max() + 0.03, 0.01)
    if len(edges) < 3:
        return np.array([])
    h, e = np.histogram(offsets, bins=edges)
    h = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
    pk, _ = find_peaks(h, height=p.sep_peak_min_points, distance=3)
    return (e[pk] + e[pk + 1]) / 2


def detect_sep_lines(points: np.ndarray, normals: np.ndarray, floor_y: float, grid: Grid,
                     p: BetaParams) -> list[SepLine]:
    """All face lines with >= sep_min_support_m of occupied 2 cm bins (>= sep_occ_min_points voxels per bin)."""
    lines = []
    for axis, (off, along, sign, height) in face_points(points, normals, floor_y, p).items():
        n_bins, origin = (grid.rows, grid.v0) if axis == 0 else (grid.cols, grid.u0)
        for sg in (+1, -1):
            s = sign == sg
            for c in _peaks(off[s], p):
                near = s & (np.abs(off - c) < 0.03)
                c_ref = float(np.median(off[near]))
                near = s & (np.abs(off - c_ref) < 0.03)
                idx = np.floor((along[near] - origin) / grid.cell).astype(np.int64)
                ok = (idx >= 0) & (idx < n_bins)
                occ = np.bincount(idx[ok], minlength=n_bins) >= p.sep_occ_min_points
                if occ.sum() * grid.cell >= p.sep_min_support_m:
                    tall = _height_extent(idx[ok], height[near][ok], n_bins, p) >= p.sep_min_extent_m - 1e-9
                    lines.append(SepLine(axis, c_ref, sg, occ, occ & tall))
    return lines


def _height_extent(idx: np.ndarray, h: np.ndarray, n_bins: int, p: BetaParams) -> np.ndarray:
    """Per along-bin: metres of the 0.3-2.0 m band covered by the face (distinct 10 cm height bins), pooled over
    the bin and its two neighbours (a face seen at a grazing angle is patchy). A wall rises from the floor past
    furniture height; a bed side, a sofa back or a counter front stops below ~1 m (<= 0.7 m of extent here)."""
    hb = np.floor((h - p.sep_band_lo_m) / 0.1).astype(np.int64)
    key = np.unique(idx * 64 + np.clip(hb, 0, 63))
    ext = np.zeros(n_bins, bool).astype(np.int64)
    bins = key // 64
    hbin = key % 64
    mask = np.zeros((n_bins, 64), bool)
    mask[bins, hbin] = True
    pooled = mask.copy()
    pooled[1:] |= mask[:-1]
    pooled[:-1] |= mask[1:]
    ext = pooled.sum(1)
    return ext * 0.1


def _close_gaps(occ: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill runs of False no longer than max_gap bins that lie between two occupied bins."""
    out = occ.copy()
    idx = np.flatnonzero(occ)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a:b] = True
    return out


def _long_runs(occ: np.ndarray, min_bins: int) -> np.ndarray:
    d = np.diff(np.r_[0, occ.astype(np.int8), 0])
    out = np.zeros_like(occ)
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s >= min_bins:
            out[s:e] = True
    return out


def paired_occupancy(line: SepLine, lines: list[SepLine], p: BetaParams) -> np.ndarray:
    """Bins where an opposite face lies BEHIND this one at a wall's thickness: the two sides of one partition."""
    out = np.zeros_like(line.occ)
    for o in lines:
        if o.axis != line.axis or o.sign != -line.sign:
            continue
        gap = (line.offset - o.offset) * line.sign          # > 0 when o lies behind this face
        if p.thickness_min_m - 0.02 <= gap <= p.thickness_max_m:
            out |= o.occ
    return line.occ & _close_gaps(out, int(round(0.1 / p.cell_m)))


def separation_mask(lines: list[SepLine], grid: Grid, p: BetaParams) -> np.ndarray:
    """Raster of the cut lines: runs >= sep_min_run_m (2 cm gaps closed) drawn one cell wide on the face."""
    cut = np.zeros(grid.shape, bool)
    min_bins = int(round(p.sep_min_run_m / p.cell_m))
    for L in lines:
        occ = L.tall | paired_occupancy(L, lines, p) if p.barrier_pair_required else L.occ
        run = _long_runs(_close_gaps(occ, int(round(p.sep_max_gap_m / p.cell_m))), min_bins)
        if not run.any():
            continue
        if L.axis == 0:
            c = int(np.floor(grid.col_of(L.offset)))
            if 0 <= c < grid.cols:
                cut[run, c] = True
        else:
            r = int(np.floor(grid.row_of(L.offset)))
            if 0 <= r < grid.rows:
                cut[r, run] = True
    return cut


def header_cut_mask(points: np.ndarray, normals: np.ndarray, floor_y: float, grid: Grid, p: BetaParams,
                    low_lines: list[SepLine]) -> tuple[np.ndarray, list[dict]]:
    """v5 (D-046): a WIDE opening between two rooms (a 2.1 m sliding door, a 1.6 m doorway) is wider than any door
    neck (door_max_m), so the free space of the two rooms merges into one room. What still separates them is the
    wall above the opening, the header: rooms are bounded by walls that reach the ceiling, openings are holes in them
    below ~2.0-2.4 m. A true open-plan space has no header.

    Header evidence = a face line in the 2.05-2.75 m band that lies IN THE PLANE of a low-band (0.3-2.0 m) wall line
    (within header_coplanar_m) and spans a stretch where that wall line is open for >= header_min_open_m. The header's
    far side is rarely seen (it is above and behind the operator who walked in), so faces are not required in pairs.
    Faces that are not in a wall's plane are ignored: a tray-ceiling soffit edge (inside the room), kitchen wall
    cabinets (35 cm out), wardrobe tops (60 cm out). The cut leaves a door-sized gap (header_gap_m) in the middle so the
    existing neck rule separates the rooms there and openings.py measures the jamb-to-jamb width around it."""
    from dataclasses import replace
    q = replace(p, sep_band_lo_m=p.header_band_lo_m, sep_band_hi_m=p.header_band_hi_m,
                sep_occ_min_points=p.header_occ_min_points, sep_min_extent_m=99.0)
    heads = _merge_parallel(detect_sep_lines(points, normals, floor_y, grid, q), p.header_merge_m)
    cut = np.zeros(grid.shape, bool)
    found = []
    gap_bins = int(round(p.sep_max_gap_m / p.cell_m))
    head_gap = int(round(p.header_fill_gap_m / p.cell_m))
    min_open = int(round(p.header_min_open_m / p.cell_m))
    min_wall = int(round(p.header_min_wall_m / p.cell_m))
    half_gap = int(round(p.header_gap_m / 2 / p.cell_m))
    for L in heads:
        low = np.zeros_like(L.occ)
        n_wall = 0
        for o in low_lines:                       # the wall line this header continues (same plane, same facing)
            if o.axis == L.axis and o.sign == L.sign and abs(o.offset - L.offset) <= p.header_coplanar_m:
                low |= o.occ
        n_wall = int(low.sum())
        if n_wall < min_wall:
            continue
        low = _close_gaps(low, gap_bins)
        head = _close_gaps(L.occ, head_gap)
        open_ = head & ~low
        d = np.diff(np.r_[0, open_.astype(np.int8), 0])
        for s0, e0 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if e0 - s0 < min_open:
                continue
            mid = (s0 + e0) // 2
            seg = np.zeros_like(open_)
            seg[s0:e0] = True
            seg[max(mid - half_gap, s0):min(mid + half_gap, e0)] = False
            if L.axis == 0:
                c = int(np.floor(grid.col_of(L.offset)))
                if 0 <= c < grid.cols:
                    cut[seg, c] = True
            else:
                r = int(np.floor(grid.row_of(L.offset)))
                if 0 <= r < grid.rows:
                    cut[r, seg] = True
            origin = grid.v0 if L.axis == 0 else grid.u0           # along-line coordinate of bin 0
            found.append(dict(axis=L.axis, offset=round(float(L.offset), 3), open_m=round((e0 - s0) * p.cell_m, 2),
                              wall_m=round(n_wall * p.cell_m, 2), s0=float(origin + s0 * p.cell_m),
                              s1=float(origin + e0 * p.cell_m), sign=int(L.sign),
                              far=_far_face(L, heads, s0, e0, p)))
    return cut, found


def _far_face(L: SepLine, heads: list[SepLine], s0: int, e0: int, p: BetaParams) -> float | None:
    """v6 (I-011): offset of the header's OTHER face, the one the room beyond the opening sees (opposite facing, behind
    L by a wall thickness), or None. The cut is drawn on L, the face coplanar with the near room's wall, so the far
    room's free space runs through the wall slab to it and that room came out one wall thickness too long (sim
    kitchen +21.5 cm). The far face is the far room's inner face above the opening. It must be seen over or within
    header_far_reach_m of the open span: the same wall's far face elsewhere along a long wall also counts, since the
    header lies in the wall's planes (sim k38: far side seen beside the opening only)."""
    reach = int(round(p.header_far_reach_m / p.cell_m))
    best = None
    for o in heads:
        if o.axis != L.axis or o.sign != -L.sign:
            continue
        gap = (L.offset - o.offset) * L.sign                  # > 0 when o lies behind L
        seen = int(o.occ[max(s0 - reach, 0):e0 + reach].sum()) * p.cell_m
        if p.thickness_min_m <= gap <= p.thickness_max_m and seen >= p.header_far_min_m:
            if best is None or gap < best[0]:
                best = (gap, float(o.offset))
    return None if best is None else round(best[1], 4)


def _merge_parallel(lines: list[SepLine], within: float) -> list[SepLine]:
    """Header faces of one opening often come out as several parallel peaks a few cm apart (a door frame and its
    track, drift doubling). Merge same-axis, same-facing lines within `within` metres: union of the occupancy, offset
    weighted by the number of occupied bins."""
    out: list[SepLine] = []
    for L in sorted(lines, key=lambda l: (l.axis, l.sign, l.offset)):
        if out and out[-1].axis == L.axis and out[-1].sign == L.sign and abs(L.offset - out[-1].offset) <= within:
            M = out[-1]
            wa, wb = max(int(M.occ.sum()), 1), max(int(L.occ.sum()), 1)
            out[-1] = SepLine(M.axis, (M.offset * wa + L.offset * wb) / (wa + wb), M.sign, M.occ | L.occ, M.tall | L.tall)
        else:
            out.append(L)
    return out
