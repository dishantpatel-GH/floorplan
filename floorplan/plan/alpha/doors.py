"""Doors and passages, measured jamb to jamb on raw points.

Two detectors, because there are two situations:

A. Between two rooms (most doors). For every pair of rooms we look for CONTACTS: an edge of room A and an edge of
   room B that face each other (parallel, opposite inward normals) at most one wall thickness apart, overlapping
   by at least a door width. The strip between the two faces is the wall. Along the strip, a run of bins that are
   observed free (floor, rays, camera path) and hold no tall surface is an opening through that wall. This works
   wherever the door is, including right next to a corner, where the wall exists on one side of the door only
   (an earlier version looked for gaps INSIDE a single wall line and missed exactly those doors).

B. From a room to unobserved space (entrance door, a room the operator never entered): a gap inside a wall line
   with a room on one side only. It must be seen free well beyond the wall (0.55 m), otherwise it is a niche.

Width: the clear width between the two JAMB faces. Jamb faces are perpendicular to the wall, so we look only at
raw points inside the wall core (between its two faces, 2 cm in from each): there, the only surfaces the LiDAR
can see are the jambs. Each jamb = the robust position of the outermost dense layer on its side. If the jambs are
not visible, the width falls back to the 2 cm raster run, flagged 'inferred' with a wider interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from floorplan.model import Measurement, Opening
from floorplan.plan.alpha.complex import CellComplex
from floorplan.plan.alpha.evidence import Grid
from floorplan.plan.alpha.labeling import RoomCells
from floorplan.plan.alpha.lines import Line, gaps_of
from floorplan.plan.alpha.measure import RawIndex
from floorplan.plan.alpha.params import AlphaParams
from floorplan.plan.alpha.walls import polygon_edges


@dataclass
class DoorCandidate:
    family: str                 # 'x': the wall is u = const; the door runs along v
    line_offset: float          # wall centre line (perpendicular coordinate)
    core: tuple[float, float]   # perpendicular range strictly inside the wall, where only jambs are visible
    g0: float                   # opening extent along the wall (from the 2 cm raster)
    g1: float
    rooms: tuple[int, ...]      # room indices on the two sides (one if the other side is unobserved)
    free_frac: float


# ---------------------------------------------------------------------------------------------------------------
# rasters helpers
# ---------------------------------------------------------------------------------------------------------------

def _block(img: np.ndarray, grid: Grid, family: str, perp: tuple[float, float], along: tuple[float, float]):
    """Sub-image with rows = along-wall bins and columns = across-wall bins."""
    if family == "x":
        b = img[grid.to_px(perp[0], 0):max(grid.to_px(perp[1], 0), grid.to_px(perp[0], 0) + 1),
                grid.to_px(along[0], 1):grid.to_px(along[1], 1)]
        return b.T
    return img[grid.to_px(along[0], 0):grid.to_px(along[1], 0),
               grid.to_px(perp[0], 1):max(grid.to_px(perp[1], 1), grid.to_px(perp[0], 1) + 1)]


def _free_fraction(free: np.ndarray, grid: Grid, family: str, perp: tuple[float, float], g0: float, g1: float) -> float:
    b = _block(free, grid, family, perp, (g0, g1))
    return float(b.mean()) if b.size else 0.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# ---------------------------------------------------------------------------------------------------------------
# A. doors between two rooms
# ---------------------------------------------------------------------------------------------------------------

def _contact_candidates(rooms: list[RoomCells], free: np.ndarray, grid: Grid, p: AlphaParams) -> list[DoorCandidate]:
    edges = {rc.index: polygon_edges(rc.polygon) for rc in rooms}
    out = []
    for ia, A in enumerate(rooms):
        for B in rooms[ia + 1:]:
            for ea in edges[A.index]:
                for eb in edges[B.index]:
                    if ea.family != eb.family or ea.inward != -eb.inward:
                        continue
                    t = ea.inward * (ea.offset - eb.offset)          # B's face lies t behind A's face
                    lo_al = max(min(ea.a, ea.b), min(eb.a, eb.b))
                    hi_al = min(max(ea.a, ea.b), max(eb.a, eb.b))
                    if not (-0.02 <= t <= p.max_thickness_m) or hi_al - lo_al < p.min_door_m:
                        continue
                    out += _open_runs(ea.family, ea.offset, eb.offset, lo_al, hi_al, (A.index, B.index), free, grid, p)
    return out


def _open_runs(family: str, off_a: float, off_b: float, lo_al: float, hi_al: float, rooms: tuple[int, int],
               free: np.ndarray, grid: Grid, p: AlphaParams) -> list[DoorCandidate]:
    lo, hi = sorted((off_a, off_b))
    if hi - lo < p.min_thickness_m:                     # rooms touch with no wall: look +-5 cm around the line
        c = (lo + hi) / 2
        strip, core = (c - 0.05, c + 0.05), (c - 0.05, c + 0.05)
    else:
        strip, core = (lo, hi), (lo + 0.02, hi - 0.02)
    b = _block(free, grid, family, strip, (lo_al, hi_al))
    if b.size == 0:
        return []
    open_bins = b.mean(axis=1) >= p.door_free_min
    out = []
    for s, e in _runs(open_bins):
        g0, g1 = lo_al + s * grid.res, lo_al + e * grid.res
        if g1 - g0 >= p.min_door_m - 0.1:
            out.append(DoorCandidate(family, (lo + hi) / 2, core, g0, g1, tuple(sorted(rooms)),
                                     float(b[s:e].mean())))
    return out


# ---------------------------------------------------------------------------------------------------------------
# B. doors from a room to unobserved space
# ---------------------------------------------------------------------------------------------------------------

def _room_at(cx: CellComplex, room_map: np.ndarray, family: str, offset: float, along: float, side: int) -> int:
    for k in np.arange(0.05, 0.61, 0.05):
        perp = offset + side * k
        u, v = (perp, along) if family == "x" else (along, perp)
        i, j = cx.cell_of(np.array([u]), np.array([v]))
        if room_map[i[0], j[0]] >= 0:
            return int(room_map[i[0], j[0]])
    return -1


def _single_face_core(L: Line, room_side: int, lines: dict[str, list[Line]], p: AlphaParams) -> tuple[float, float]:
    """Wall core behind a face: up to the nearest opposite face 4-45 cm away, or 2-12 cm behind if none."""
    behind = -room_side
    best = None
    for M in lines[L.family]:
        t = behind * (M.offset - L.offset)
        if p.min_thickness_m <= t <= p.max_thickness_m and (best is None or t < best):
            best = t
    depth = best if best is not None else 0.14
    return tuple(sorted((L.offset + behind * 0.02, L.offset + behind * (depth - 0.02))))


def _one_sided_candidates(lines: dict[str, list[Line]], cx: CellComplex, room_map: np.ndarray, free: np.ndarray,
                          p: AlphaParams) -> list[DoorCandidate]:
    grid = cx.grid
    out = []
    for fam, Ls in lines.items():
        coords = grid.along_coords(1 if fam == "x" else 0)
        for L in Ls:
            for s, e in gaps_of(L.occ):
                g0, g1 = coords[s] - grid.res / 2, coords[e - 1] + grid.res / 2
                if not (p.min_door_m <= g1 - g0 <= p.max_door_m + 0.3):
                    continue
                mid = (g0 + g1) / 2
                ra, rb = (_room_at(cx, room_map, fam, L.offset, mid, side) for side in (+1, -1))
                if (ra >= 0) == (rb >= 0):                    # both sides rooms (handled by A) or none
                    continue
                room_side = +1 if ra >= 0 else -1
                ff = _free_fraction(free, grid, fam, (L.offset - 0.1, L.offset + 0.1), g0, g1)
                beyond = L.offset - room_side * 0.55
                deep = _free_fraction(free, grid, fam, (beyond - 0.1, beyond + 0.1), g0, g1)
                if ff >= p.door_free_min and deep >= p.door_free_min:
                    out.append(DoorCandidate(fam, L.offset, _single_face_core(L, room_side, lines, p), g0, g1,
                                             (max(ra, rb),), ff))
    return out


def _center(c: DoorCandidate) -> np.ndarray:
    mid = (c.g0 + c.g1) / 2
    return np.array([c.line_offset, mid] if c.family == "x" else [mid, c.line_offset])


def _dedupe(cands: list[DoorCandidate]) -> list[DoorCandidate]:
    """Several edge pairs (both faces of a wall, or two perpendicular edges at a room corner) can report the same
    opening: keep the best-supported one."""
    out: list[DoorCandidate] = []
    for c in sorted(cands, key=lambda c: (-len(c.rooms), -(c.g1 - c.g0) * c.free_frac)):
        same_wall = any(o.family == c.family and abs(o.line_offset - c.line_offset) < 0.5
                        and min(o.g1, c.g1) - max(o.g0, c.g0) > 0.3 * (c.g1 - c.g0) for o in out)
        same_spot = any(o.rooms == c.rooms and np.linalg.norm(_center(o) - _center(c)) < 0.5 for o in out)
        if not (same_wall or same_spot):
            out.append(c)
    return out


def find_door_candidates(rooms: list[RoomCells], lines: dict[str, list[Line]], cx: CellComplex,
                         room_map: np.ndarray, free: np.ndarray, p: AlphaParams) -> list[DoorCandidate]:
    """free = observed-free pixels that hold no tall surface."""
    return _dedupe(_contact_candidates(rooms, free, cx.grid, p) + _one_sided_candidates(lines, cx, room_map, free, p))


# ---------------------------------------------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------------------------------------------

def _jamb(along: np.ndarray, height: np.ndarray, side: int, p: AlphaParams):
    """Jamb face position from points just outside one end of the free run.

    A jamb is a TALL face: it runs from the floor to the door head. Every 1 cm layer of points is scored by the
    share of 10 cm height bins (band_lo..door_jamb_top) it covers. The jamb is the layer NEAREST the opening whose
    coverage is at least jamb_min_cover and at least 80% of the best layer's. The relative test matters: short
    things near the opening (the edge of an open door leaf, a switch, a hinge) are rejected however much of the
    jamb happened to be seen; an absolute threshold flipped on a 2.5 cm change of floor reference (issue log).
    The position is then mean-shifted onto the face. Returns (position, standard error, n points) or None."""
    if len(along) < p.jamb_min_points:
        return None
    x = -along if side == +1 else along                  # in x, "towards the opening" is always increasing
    edges = np.arange(x.min(), x.max() + 0.02, 0.01)
    if len(edges) < 2:
        return None
    centres = (edges[:-1] + edges[1:]) / 2
    hbins = np.arange(p.band_lo_m, p.door_jamb_top_m + 1e-9, 0.10)
    cover = np.array([np.mean(np.histogram(height[np.abs(x - c) < 0.015], bins=hbins)[0] >= 3) for c in centres])
    ok = cover >= max(p.jamb_min_cover, 0.8 * cover.max())
    if not ok.any():
        return None
    pos = float(centres[np.flatnonzero(ok).max()])       # nearest the opening
    for _ in range(5):                                   # mean-shift onto the face (the bin may sit at its edge)
        sel = np.abs(x - pos) < 0.015
        pos = float(np.median(x[sel]))
    if sel.sum() < p.jamb_min_points:
        return None
    mad = 1.4826 * float(np.median(np.abs(x[sel] - pos)))
    return (-pos if side == +1 else pos), mad / np.sqrt(sel.sum()), int(sel.sum())


def _blocked_fraction(along: np.ndarray, a: float, b: float, min_pts: int, bin_m: float = 0.05) -> float:
    """Share of 5 cm bins across the opening (inside the wall core, at body height) that hold SOLID surface:
    >= min_pts points, i.e. ~10 cm of continuous surface at raw density. Sparse points (wall-face noise smeared
    into a thin core by drift) do not count."""
    if b - a < bin_m:
        return 0.0
    k = np.floor((along[(along > a) & (along < b)] - a) / bin_m).astype(np.int64)
    n = int(np.ceil((b - a) / bin_m))
    return float(np.mean(np.bincount(k, minlength=n)[:n] >= min_pts))


def measure_door(c: DoorCandidate, idx: RawIndex, floor_levels: dict[int, float], global_floor: float,
                 p: AlphaParams, oid: str) -> Opening | None:
    """Jamb-to-jamb width and lintel height. None if the measured width is below a walkable door."""
    al = 2 if c.family == "x" else 0                     # along-wall axis of the 3-D points
    i0, i1 = np.searchsorted(idx.keys[c.family], list(c.core))
    sel = idx.order[c.family][i0:i1]
    along = idx.P[sel, al]
    floor = float(np.mean([floor_levels.get(r, global_floor) for r in c.rooms]))
    h = idx.P[sel, 1] - floor
    body = (h > p.band_lo_m) & (h < p.door_jamb_top_m)
    if _blocked_fraction(along[body], c.g0 + 0.05, c.g1 - 0.05, p.door_solid_pts) >= p.door_blocked_max:
        return None                     # something ~body-high spans the gap (half wall, closed leaf, counter)
    win = (along > c.g0 - p.jamb_search_m) & (along < c.g1 + p.jamb_search_m)
    lsel = body & (along > c.g0 - p.jamb_search_m) & (along < c.g0 + 0.05)
    rsel = body & (along > c.g1 - 0.05) & (along < c.g1 + p.jamb_search_m)
    left = _jamb(along[lsel], h[lsel], -1, p)
    right = _jamb(along[rsel], h[rsel], +1, p)
    if left and right and right[0] - left[0] >= p.min_door_m:
        sig = float(np.sqrt(left[1] ** 2 + right[1] ** 2 + 2 * p.lidar_sigma_m ** 2))
        width = Measurement.from_sigma(right[0] - left[0], sig, method="jamb face to jamb face, raw LiDAR in wall core")
        jl, jr, jamb_txt = left[0], right[0], f"jambs {left[2]}/{right[2]} raw pts"
    else:
        if c.g1 - c.g0 < p.min_door_m:
            return None
        width = Measurement.from_sigma(c.g1 - c.g0, p.res_m, method="free run in 2 cm raster (jambs not seen)",
                                       status="inferred")
        jl, jr, jamb_txt = c.g0, c.g1, "jambs not measurable"
    head = h[win & (along > jl + 0.05) & (along < jr - 0.05) & (h > p.door_jamb_top_m)]
    if len(head) >= p.jamb_min_points:
        height = Measurement.from_sigma(float(np.percentile(head, 5)), float(np.hypot(p.res_m / 2, p.lidar_sigma_m)),
                                        method="lowest lintel layer in wall core")
    else:
        height = Measurement.missing("m", "lintel above the opening", "head not observed")
    two_sided = len(c.rooms) == 2
    conf = (0.4 + 0.3 * min(1.0, c.free_frac) + 0.3 * (width.status == "measured")) * (1.0 if two_sided else 0.7)
    center = (c.line_offset, (jl + jr) / 2) if c.family == "x" else ((jl + jr) / 2, c.line_offset)
    kind = "door" if width.value <= p.max_leaf_m else "passage"
    ev = (f"opening {c.g1 - c.g0:.2f} m in wall {'u' if c.family == 'x' else 'v'}={c.line_offset:.3f}; "
          f"observed free {100 * c.free_frac:.0f}%; {jamb_txt}; "
          f"{'between two rooms' if two_sided else 'leads to unobserved space'}")
    return Opening(oid, kind, [], [], (float(center[0]), float(center[1])), width, height, None, round(conf, 2), ev)
