"""Steps 4-5: label every cell inside/outside with a graph cut, then split the inside into rooms.

Inside/outside is a binary labelling problem with two kinds of cost (a Markov random field):
  * data cost (per cell): calling a cell OUTSIDE costs its area x fraction with interior evidence (we saw floor,
    rays or the camera there); calling it INSIDE costs its area x fraction WITHOUT evidence x w_unknown (we never
    saw it, so it is more likely behind a wall or outside the dwelling).
  * smoothness cost (per edge between two cells): changing label across an edge costs w_smooth x the edge's length
    x the part of it NOT covered by wall surface. Boundaries are therefore cheap where walls are and expensive in
    open space, which pulls the inside/outside boundary onto real walls and removes isolated speckle.
For two labels this energy has an exact global minimum computable by a single s-t minimum cut (Boykov and Jolly
2001; the same formulation as Ochmann et al. 2019 and Mura et al. 2016). Exact and deterministic.

Rooms: a room is a part of the interior that connects to the rest only through openings that are narrower than
the room itself (doors). We find them with the classic distance-transform watershed (Bormann et al. 2016, "Room
segmentation: survey"): tall wall surface is a barrier, the distance to the nearest barrier peaks in the middle of
each room and dips in each doorway, so flooding the distance map from its peaks puts the boundaries in the doors.
Over-segmentation (a long corridor has several peaks) is undone by merging two basins whenever the neck between
them is almost as wide as BOTH basins (no constriction on either side = same space) or wider than any door (open
plan). Comparing with the wider basin is deliberate: a 1.1 m corridor meeting a 3.5 m room is a constriction for the
room, so the corridor stays its own room even when no door leaf is hung there. The
pixel basins are then snapped to the cell complex, so room polygons still follow the measured wall lines.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from floorplan.plan.alpha.complex import CellComplex
from floorplan.plan.alpha.evidence import count_image
from floorplan.plan.alpha.params import AlphaParams


@dataclass
class RoomCells:
    index: int
    cells: np.ndarray        # (K, 2) cell indices (i, j)
    polygon: Polygon         # union of the cells, CCW, holes filled


def _neighbour_edges(cx: CellComplex):
    """All (cell_a, cell_b, edge_length, coverage) of 4-neighbouring cells."""
    I, J = cx.shape
    for i in range(I - 1):          # neighbours across u boundary i+1
        for j in range(J):
            yield (i, j), (i + 1, j), cx.vs[j + 1] - cx.vs[j], cx.cov_u[i + 1, j]
    for i in range(I):              # neighbours across v boundary j+1
        for j in range(J - 1):
            yield (i, j), (i, j + 1), cx.us[i + 1] - cx.us[i], cx.cov_v[i, j + 1]


def label_inside(cx: CellComplex, p: AlphaParams) -> np.ndarray:
    """Exact minimum of the inside/outside energy by s-t min cut. Returns (I, J) bool, True = inside."""
    A = cx.area()
    f = cx.interior
    cost_out = A * f + p.w_traj * cx.traj
    cost_in = A * (1.0 - f) * p.w_unknown
    G = nx.DiGraph()
    I, J = cx.shape
    for i in range(I):
        for j in range(J):
            G.add_edge("s", (i, j), capacity=float(cost_out[i, j]))
            G.add_edge((i, j), "t", capacity=float(cost_in[i, j]))
    for a, b, length, cov in _neighbour_edges(cx):
        w = float(p.w_smooth_m * length * (1.0 - cov))
        G.add_edge(a, b, capacity=w)
        G.add_edge(b, a, capacity=w)
    _, (S, _) = nx.minimum_cut(G, "s", "t")
    inside = np.zeros((I, J), bool)
    for node in S:
        if node != "s":
            inside[node] = True
    return inside


# ---------------------------------------------------------------------------------------------------------------
# room segmentation
# ---------------------------------------------------------------------------------------------------------------

def _basins(free: np.ndarray, res: float, p: AlphaParams) -> tuple[np.ndarray, np.ndarray]:
    """Watershed of the distance-to-barrier map, one basin per significant peak. Returns (labels, distance)."""
    dist = ndimage.distance_transform_edt(free) * res
    markers, _ = ndimage.label(h_maxima(dist, p.peak_h_m))
    return watershed(-dist, markers, mask=free), dist


def _saddles(lab: np.ndarray, dist: np.ndarray) -> dict[tuple[int, int], float]:
    """For each pair of touching basins, the largest clearance (distance value) along their shared boundary."""
    out: dict[tuple[int, int], float] = {}
    for a, b, da, db in ((lab[:-1], lab[1:], dist[:-1], dist[1:]), (lab[:, :-1], lab[:, 1:], dist[:, :-1], dist[:, 1:])):
        m = (a > 0) & (b > 0) & (a != b)
        for x, y, d in zip(a[m], b[m], np.minimum(da[m], db[m])):
            key = (int(min(x, y)), int(max(x, y)))
            out[key] = max(out.get(key, 0.0), float(d))
    return out


def merge_basins(lab: np.ndarray, dist: np.ndarray, res: float, p: AlphaParams) -> np.ndarray:
    """Merge basins separated by no real constriction, and basins too small to be a room (an alcove, the space
    between two wardrobes) into the neighbour they connect to most widely. Greedy, one merge at a time, best
    candidate first, until nothing qualifies."""
    lab = lab.copy()
    while True:
        ids, peak = _max_by_label(lab, dist)
        width = {int(k): 2 * float(v) for k, v in zip(ids, peak)}
        area = dict(zip(ids.tolist(), (ndimage.sum(np.ones_like(dist), lab, ids) * res * res).tolist()))
        best = None
        for (a, b), saddle in _saddles(lab, dist).items():
            neck = 2 * saddle
            small = min(area[a], area[b]) < p.min_room_area_m2
            if small or neck >= p.max_door_m or neck >= p.neck_ratio * max(width[a], width[b]):
                score = (1.0 if small else 0.0) + neck / max(width[a], width[b])
                if best is None or score > best[0]:
                    keep, drop = (a, b) if area[a] >= area[b] else (b, a)
                    best = (score, keep, drop)
        if best is None:
            return lab
        lab[lab == best[2]] = best[1]


def _max_by_label(lab: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(lab[lab > 0])
    return ids, np.asarray(ndimage.maximum(dist, lab, ids))


def _snap_to_cells(cx: CellComplex, inside: np.ndarray, basins: np.ndarray, p: AlphaParams) -> np.ndarray:
    """Each inside cell takes the basin covering most of its pixels; cells that are mostly barrier (wall cores)
    or split between two basins (door strips) are left out. Returns (I, J) basin id, 0 = none."""
    I, J = cx.shape
    out = np.zeros((I, J), np.int64)
    for i, j in np.argwhere(inside):
        block = basins[cx.bu[i]:cx.bu[i + 1], cx.bv[j]:cx.bv[j + 1]]
        if block.size == 0:
            continue
        ids, cnt = np.unique(block[block > 0], return_counts=True)
        if len(ids) and cnt.max() >= p.snap_frac * block.size:
            out[i, j] = ids[np.argmax(cnt)]
    return _adopt_enclosed(cx, inside, out)


def _adopt_enclosed(cx: CellComplex, inside: np.ndarray, cell_basin: np.ndarray) -> np.ndarray:
    """An inside cell left without a basin (e.g. a thin strip that is all barrier: the side panel of a wardrobe)
    joins its neighbours' basin when ALL its assigned neighbours agree. A wall core between two different rooms
    has disagreeing neighbours and stays out. Repeated until stable so multi-cell gaps fill from the outside in."""
    out = cell_basin.copy()
    I, J = out.shape
    changed = True
    while changed:
        changed = False
        for i, j in np.argwhere(inside & (out == 0)):
            nb = {int(out[a, b]) for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                  if 0 <= a < I and 0 <= b < J and out[a, b] > 0}
            if len(nb) == 1:
                out[i, j] = nb.pop()
                changed = True
    return out


def _largest_connected(cells: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Keep the largest edge-connected group of cells (a basin can snap to two separate cell groups)."""
    m = np.zeros(shape, bool)
    m[cells[:, 0], cells[:, 1]] = True
    lab, n = ndimage.label(m)
    if n <= 1:
        return cells
    keep = np.argmax(np.bincount(lab[m]))
    return np.argwhere(lab == keep)


def _polygon(cx: CellComplex, cells: np.ndarray) -> Polygon:
    """Union of edge-connected rectangles = one rectilinear polygon; interior holes are filled."""
    u = unary_union([box(cx.us[i], cx.vs[j], cx.us[i + 1], cx.vs[j + 1]) for i, j in cells])
    return orient(Polygon(u.exterior).simplify(0.0), sign=1.0)


def regularize_polygon(poly: Polygon, width: float, close: bool = True) -> Polygon:
    """Morphological opening then (optionally) closing with square (mitred) corners: removes slivers and fills
    slits narrower than `width` (wall stubs, shelf lines) while every surviving edge stays exactly on its wall line.
    Opening alone only ever shrinks the polygon. Returns the largest remaining piece (empty if nothing survives)."""
    h = width / 2
    o = poly.buffer(-h, join_style="mitre").buffer(h, join_style="mitre")
    if o.is_empty:
        return o
    if close:
        o = o.buffer(h, join_style="mitre").buffer(-h, join_style="mitre")
    if o.geom_type != "Polygon":
        o = max(o.geoms, key=lambda g: g.area)
    return orient(Polygon(o.exterior).simplify(1e-6), sign=1.0)


def barrier_raster(scene: dict, info: dict, cx: CellComplex, p: AlphaParams) -> np.ndarray:
    """Pixels holding TALL vertical surface (any orientation): walls, wardrobes, door frames. Furniture below
    high_from_m is not a barrier, so it never splits a room."""
    P, N = scene["points"], scene["normals"]
    h = P[:, 1] - info["floor_y"]
    m = (np.abs(N[:, 1]) < p.wall_max_ny) & (h >= p.high_from_m) & (h < p.band_hi_m)
    iu, iv, ok = cx.grid.index(P[m, 0], P[m, 2])
    img = count_image(iu[ok], iv[ok], (cx.grid.nu, cx.grid.nv))
    return ndimage.binary_dilation(img >= p.barrier_min_points, iterations=1)


def group_rooms(cx: CellComplex, inside: np.ndarray, barrier: np.ndarray,
                p: AlphaParams) -> tuple[list[RoomCells], np.ndarray, np.ndarray]:
    """Rooms as groups of cells. Returns rooms, the (I, J) room-index map (-1 = none) and the pixel basins."""
    free = cx.label_raster(inside.astype(np.int8)).astype(bool) & ~barrier
    lab, dist = _basins(free, cx.grid.res, p)
    lab = merge_basins(lab, dist, cx.grid.res, p)
    cell_basin = _snap_to_cells(cx, inside, lab, p)
    candidates = []
    for b in np.unique(cell_basin[cell_basin > 0]):
        cells = _largest_connected(np.argwhere(cell_basin == b), cx.shape)
        poly = regularize_polygon(_polygon(cx, cells), p.min_sliver_m)
        if poly.is_empty:
            continue
        if poly.area < p.min_room_area_m2 or poly.buffer(-p.min_room_width_m / 2).is_empty:
            continue
        candidates.append((poly.area, cells, poly))
    rooms, room_map = [], np.full(cx.shape, -1, np.int64)
    for k, (_, cells, poly) in enumerate(sorted(candidates, key=lambda t: -t[0])):   # largest first
        rooms.append(RoomCells(k, cells, poly))
        room_map[cells[:, 0], cells[:, 1]] = k
    return rooms, room_map, lab
