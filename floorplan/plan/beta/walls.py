"""Step 3: room region -> rectilinear wall polygon -> wall offsets measured on raw LiDAR points.

Three sub-steps, each solving one problem:

  a) Candidate wall lines. Every vertical TSDF surface near the room whose normal points INTO the room votes for
     a wall line at its x (if it faces +-x) or z (if it faces +-z). Peaks of these 1-D histograms with enough
     support along the wall are candidate walls. The room mask's own straight edges add "inferred" lines for
     sides where no wall was seen (open passages, unobserved walls).
  b) Polygon by line arrangement (the cell-complex idea used by many floor-plan reconstruction methods). The
     x-lines and z-lines cut the plane into rectangles. A rectangle belongs to the room when the room mask covers
     at least half of it. The union of the chosen rectangles is the room polygon: rectilinear by construction,
     with corners exactly at line intersections, and robust to spurious lines (a line only becomes an edge if
     coverage changes across it).
  c) Raw-point refinement (D-007). Each wall line is re-measured on raw high-confidence LiDAR points in a +-4 cm
     band, seen from inside the room (ray direction check), between 0.3 and 2.0 m, away from corners. The offset
     is a trimmed mean around the histogram mode. Corners = intersections of refined lines, lengths =
     corner-to-corner distances.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage as ndi
from shapely import contains_xy
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from floorplan.plan.beta.freespace import disk
from floorplan.plan.beta.grid import Grid, PointIndex
from floorplan.plan.beta.params import BetaParams
from floorplan.uncertainty.noise import sigma_lidar


@dataclass
class Line:
    """A wall line, nearly axis-aligned: across = offset + slope * (along - mid).

    axis 0: across = u, along = v (the wall runs along v); axis 1: across = v, along = u.
    The topology (which walls exist, which meet at which corner) is Manhattan; the geometry is not forced to be:
    the slope is fitted on raw points (see _refine_one for why)."""
    axis: int
    offset: float                  # across-coordinate at `mid`
    support_m: float               # length of wall evidence along the line (0 for inferred lines)
    inferred: bool
    sigma: float = 0.0             # 1-sigma of the offset after refinement
    slope: float = 0.0             # d(across)/d(along); 0 = exactly axis-aligned
    slope_sigma: float = 0.0
    mid: float = 0.0               # along-coordinate where `offset` is defined (centre of the measured points)
    tilt: float = 0.0              # free-fit slope from pass 1 (diagnostic; enters sigma when not modelled)
    n_points: int = 0
    rmse: float | None = None
    n_patches: int = 0
    single_visit: bool = False     # v2 X-1: all inliers come from the room's chosen visit (else several passes)
    inconsistency: float = 0.0     # v2 B-4: sqrt(rmse^2 - expected sensor rmse^2), scatter the sensor cannot explain


@dataclass
class RoomGeometry:
    """Rectilinear room polygon as a closed sequence of edges, each on a Line."""
    room_label: int
    lines: list[Line]
    edge_lines: list[int]          # edge k lies on lines[edge_lines[k]]; edge k goes vertex k -> vertex k+1
    yaw: float = 0.0               # small room rotation (tan of the angle) shared by all its walls
    yaw_sigma: float = 0.0

    def line_slopes(self, yaw: float | np.ndarray) -> np.ndarray:
        """Slopes of all lines for a room rotation `yaw` (scalar or (samples,)): a rotation by angle t tilts
        walls running along u by +tan t and walls running along v by -tan t."""
        sign = np.array([1.0 if l.axis == 1 else -1.0 for l in self.lines])
        return np.multiply.outer(yaw, sign)

    def vertices(self, offsets: np.ndarray | None = None, slopes: np.ndarray | None = None) -> np.ndarray:
        """(n, 2) corner positions = intersections of consecutive wall lines.

        offsets / slopes may be (n_lines,) or (samples, n_lines) for Monte Carlo."""
        c = np.array([l.offset for l in self.lines]) if offsets is None else offsets
        k = np.array([l.slope for l in self.lines]) if slopes is None else slopes
        m = np.array([l.mid for l in self.lines])
        e = np.array(self.edge_lines)
        prev = np.roll(e, 1)
        axis = np.array([self.lines[i].axis for i in e])
        # vertex k joins edge k-1 and edge k: one "U-line" u = cU + kU (v - mU), one "V-line" v = cV + kV (u - mV)
        iu, iv = np.where(axis == 0, e, prev), np.where(axis == 0, prev, e)
        cu, ku, mu = c[..., iu], k[..., iu], m[iu]
        cv, kv, mv = c[..., iv], k[..., iv], m[iv]
        u = (cu + ku * (cv - mu - kv * mv)) / (1.0 - ku * kv)
        v = cv + kv * (u - mv)
        return np.stack([u, v], -1)


# ---------------------------------------------------------------- a) candidate lines

def room_extent_mask(labels: np.ndarray, label: int, p: BetaParams) -> np.ndarray:
    """Room mask grown back to the wall faces (free space stops wall_dilate short of them), holes filled."""
    own = labels == label
    grow = ndi.binary_dilation(own, disk(p.wall_dilate_m / p.cell_m + 1)) & ((labels == 0) | own)
    return ndi.binary_fill_holes(grow)


def vertical_points(points: np.ndarray, normals: np.ndarray, floor_y: float, p: BetaParams,
                    band: tuple[float, float] | None = None):
    """Plan positions and plan normals of vertical TSDF surface between 0.3 and 2.0 m (or `band`), computed once."""
    h = points[:, 1] - floor_y
    lo, hi = band or (p.refine_h_lo_m, p.refine_h_hi_m)
    sel = (np.abs(normals[:, 1]) < 0.3) & (h > lo) & (h < hi)
    return points[sel][:, [0, 2]], normals[sel][:, [0, 2]]


def evidence_lines(P: np.ndarray, N: np.ndarray, grid: Grid, extent: np.ndarray, p: BetaParams) -> list[Line]:
    """Wall lines voted by vertical TSDF points near the room whose normals point into it."""
    near = ndi.binary_dilation(extent, disk(p.room_wall_search_m / p.cell_m))
    r, c, ok = grid.index(P)
    ok &= near[np.clip(r, 0, grid.rows - 1), np.clip(c, 0, grid.cols - 1)]
    probe_r, probe_c, ok2 = grid.index(P + 0.15 * N)
    ok &= ok2 & extent[np.clip(probe_r, 0, grid.rows - 1), np.clip(probe_c, 0, grid.cols - 1)]
    P, N = P[ok], N[ok]
    lines = []
    for axis in (0, 1):
        facing = np.abs(N[:, axis]) > np.cos(np.radians(15))
        lines += _histogram_lines(P[facing, axis], P[facing, 1 - axis], axis, p)
    return lines


def _histogram_lines(coord: np.ndarray, along: np.ndarray, axis: int, p: BetaParams) -> list[Line]:
    if len(coord) == 0:
        return []
    bins = np.arange(coord.min() - 0.05, coord.max() + 0.06, 0.01)
    hist, _ = np.histogram(coord, bins)
    sm = np.convolve(hist, np.ones(3) / 3, mode="same")
    peaks = np.nonzero((sm >= np.roll(sm, 1)) & (sm > np.roll(sm, -1)) & (sm > 0))[0]
    out = []
    for i in peaks[np.argsort(-sm[peaks])]:
        c = bins[i] + 0.005
        if any(abs(c - l.offset) < p.line_merge_m for l in out):
            continue
        m = np.abs(coord - c) < 0.02
        support = len(np.unique(np.floor(along[m] / p.cell_m))) * p.cell_m
        if support >= p.line_min_support_m:
            out.append(Line(axis, float(c), float(support), inferred=False))
    return out


def mask_lines(extent: np.ndarray, grid: Grid, existing: list[Line], p: BetaParams) -> list[Line]:
    """Inferred lines along straight runs (>= line_min_support_m, contiguous) of the room mask's border.

    They give the arrangement a line where the room ends but no wall was seen: a watershed cut across an open
    passage, or a wall the LiDAR never hit. Contiguity matters: a ragged leak (rays fanning through a window)
    has no long straight run, so it produces no line."""
    out = []
    for axis in (0, 1):
        ax = 1 if axis == 0 else 0                       # raster axis across the line (columns for constant u)
        run_struct = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]) if axis == 0 else np.array([[0, 0, 0], [1, 1, 1],
                                                                                              [0, 0, 0]])
        for side, shift in ((1, 0.0), (-1, 1.0)):
            border = extent & ~np.roll(extent, side, axis=ax)
            runs, n = ndi.label(border, run_struct)
            if n == 0:
                continue
            sizes = ndi.sum_labels(np.ones_like(runs), runs, np.arange(1, n + 1)) * p.cell_m
            pos = ndi.minimum(np.indices(runs.shape)[ax], runs, np.arange(1, n + 1))
            for k in np.unique(pos[sizes >= p.line_min_support_m]):
                off = grid.u_of(k + shift) if axis == 0 else grid.v_of(k + shift)
                if all(abs(off - l.offset) >= 2 * p.line_merge_m for l in existing + out if l.axis == axis):
                    out.append(Line(axis, float(off), 0.0, inferred=True))
    return out


# ---------------------------------------------------------------- b) arrangement polygon

def arrangement_polygon(extent: np.ndarray, grid: Grid, lines: list[Line], p: BetaParams,
                        vertical: tuple[np.ndarray, np.ndarray] | None = None) -> Polygon | None:
    """Union of arrangement rectangles that the room mask covers by >= cell_cover_min.

    v3 (S-3, docs/modules/plan_beta.md "Part v3"): plus every partly covered cell (>= enclosed_min_cover) that is
    ENCLOSED: each of its sides either opens (no wall evidence) onto an included cell, or is a measured wall facing
    into the cell. Without it, a 2.5 m2 cell behind a bed (floor partly unseen) was in or out of the room on a
    0.535 vs 0.457 coverage, i.e. a 2.5 m2 jump from a 0.06 mm pose change."""
    us = sorted({l.offset for l in lines if l.axis == 0})
    vs = sorted({l.offset for l in lines if l.axis == 1})
    if len(us) < 2 or len(vs) < 2:
        return None
    S = np.pad(extent.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ci = np.clip(np.round([grid.col_of(u) for u in us]).astype(int), 0, grid.cols)
    ri = np.clip(np.round([grid.row_of(v) for v in vs]).astype(int), 0, grid.rows)
    nu, nv = len(us) - 1, len(vs) - 1
    cover = np.full((nu, nv), -1.0)
    for i in range(nu):
        for j in range(nv):
            area = (ci[i + 1] - ci[i]) * (ri[j + 1] - ri[j])
            if area <= 0:
                continue
            s = S[ri[j + 1], ci[i + 1]] - S[ri[j], ci[i + 1]] - S[ri[j + 1], ci[i]] + S[ri[j], ci[i]]
            cover[i, j] = s / area
    inside = cover >= p.cell_cover_min
    if p.enclosed_cells and vertical is not None and inside.any():
        measured = {(l.axis, l.offset) for l in lines if not l.inferred}
        _add_enclosed_cells(inside, cover, us, vs, vertical, measured, p)
    boxes = [box(us[i], vs[j], us[i + 1], vs[j + 1]) for i, j in zip(*np.nonzero(inside))]
    if not boxes:
        return None
    poly = unary_union(boxes)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return Polygon(poly.exterior).simplify(1e-6)


def _side_support(P: np.ndarray, N: np.ndarray, axis: int, off: float, lo: float, hi: float, inward: float | None,
                  p: BetaParams) -> float:
    """Share of a cell side (line axis/off, along [lo, hi]) covered by vertical surface points within 3 cm of it.
    inward = +1/-1: only points whose normal points that way across the line count (a wall facing into the cell);
    None: either way (any wall at all)."""
    m = (np.abs(P[:, axis] - off) < 0.03) & (P[:, 1 - axis] >= lo) & (P[:, 1 - axis] <= hi)
    if inward is not None:
        m &= N[:, axis] * inward > np.cos(np.radians(30))
    if not m.any() or hi - lo <= 0:
        return 0.0
    return float(len(np.unique(np.floor(P[m, 1 - axis] / p.cell_m))) * p.cell_m / (hi - lo))


def _add_enclosed_cells(inside: np.ndarray, cover: np.ndarray, us: list, vs: list, vertical, measured: set,
                        p: BetaParams) -> None:
    """S-3: grow `inside` (in place) by partly covered cells that are enclosed by the room and its measured walls.

    A cell qualifies if cover >= enclosed_min_cover and, on each of its 4 sides, either the neighbour across the
    side is inside and the side carries no wall (< enclosed_open_max of it has wall-band surface), or the side lies
    on one of the room's measured wall lines (an evidence line, not a mask line) and wall-band surface faces into
    the cell over >= enclosed_wall_min of the side. Space-first, but never across a
    wall and never beyond one: such a cell is floor the LiDAR did not see (under a bed, behind a sofa) inside the
    room's walls. Iterated, so a chain of such cells along a wall is added."""
    P, N = vertical
    nu, nv = inside.shape
    sel = (P[:, 0] >= us[0] - 0.05) & (P[:, 0] <= us[-1] + 0.05) & (P[:, 1] >= vs[0] - 0.05) & (P[:, 1] <= vs[-1] + 0.05)
    P, N = P[sel], N[sel]
    changed = True
    while changed:
        changed = False
        for i, j in zip(*np.nonzero(~inside & (cover >= p.enclosed_min_cover))):
            ok = True
            # (neighbour index, line axis, line offset, along range, normal sign pointing into this cell)
            for (ni, nj), axis, off, lo, hi, inward in (
                    ((i - 1, j), 0, us[i], vs[j], vs[j + 1], +1.0), ((i + 1, j), 0, us[i + 1], vs[j], vs[j + 1], -1.0),
                    ((i, j - 1), 1, vs[j], us[i], us[i + 1], +1.0), ((i, j + 1), 1, vs[j + 1], us[i], us[i + 1], -1.0)):
                nb_in = 0 <= ni < nu and 0 <= nj < nv and inside[ni, nj]
                if nb_in and _side_support(P, N, axis, off, lo, hi, None, p) < p.enclosed_open_max:
                    continue
                if (axis, off) in measured and _side_support(P, N, axis, off, lo, hi, inward, p) >= p.enclosed_wall_min:
                    continue
                ok = False
                break
            if ok:
                inside[i, j] = True
                changed = True


@dataclass
class Bump:
    """A wall-deep, door-wide outward bump cut off a room polygon: the room's free space entering a doorway."""
    axis: int                      # axis of the wall line the bump sits on (0: constant u)
    offset: float                  # the wall line (the bump's mouth)
    span: tuple[float, float]      # extent along the wall
    depth: float


def remove_doorway_bumps(poly: Polygon, seen_open: np.ndarray, grid: Grid, p: BetaParams
                         ) -> tuple[Polygon, list[Bump]]:
    """Cut outward bumps (turn pattern right-left-left-right on a CCW polygon) that are at most thickness_max deep
    and door_max wide, whose two sides lie on the same wall line, and beyond which the LiDAR saw open space.
    Without this the room polygon follows its free space into every door reveal and the wall gets a fake
    10-20 cm jog at each door. A closed niche (nothing seen beyond it) is kept: it is floor area."""
    bumps = []
    changed = True
    while changed:
        changed = False
        xy = _ccw_coords(poly)
        n = len(xy)
        if n < 8:
            break
        d = np.round(np.roll(xy, -1, axis=0) - xy, 6)            # edge k: vertex k -> k+1
        turn = np.sign(np.cross(np.roll(d, 1, axis=0), d))       # at vertex k, +1 = left turn
        for k in range(n):
            km1, kp1, kp2 = (k - 1) % n, (k + 1) % n, (k + 2) % n
            if not (turn[km1] < 0 and turn[k] > 0 and turn[kp1] > 0 and turn[kp2] < 0):
                continue
            depth_a, width, depth_b = (np.abs(d[i]).sum() for i in (km1, k, kp1))
            before, after = xy[km1], xy[kp2]                      # the two vertices on the wall line
            axis = 0 if abs(d[k][0]) < 1e-9 else 1
            if (max(depth_a, depth_b) <= p.thickness_max_m and width <= p.door_max_m
                    and abs(depth_a - depth_b) < 1e-6 and abs(before[axis] - after[axis]) < 1e-6
                    and _open_beyond(xy[k], xy[kp1], d[km1], seen_open, grid)):
                bumps.append(Bump(axis, float(before[axis]),
                                  tuple(sorted((float(xy[k][1 - axis]), float(xy[kp1][1 - axis])))), depth_a))
                keep = [i for i in range(n) if i not in (k, kp1)]       # drop the two outer corners
                poly = Polygon(xy[keep]).simplify(1e-6)
                changed = True
                break
    return poly, bumps


def _open_beyond(a: np.ndarray, b: np.ndarray, outward: np.ndarray, seen_open: np.ndarray, grid: Grid,
                 reach_m: float = 0.2) -> bool:
    """Is at least a third of the strip just outside edge a-b seen as open (floor or carved free space)?"""
    step = outward / max(np.abs(outward).sum(), 1e-9)
    t = np.linspace(0.1, 0.9, 9)
    pts = [a + (b - a) * ti + step * r for ti in t for r in (0.06, reach_m / 2, reach_m)]
    r, c, ok = grid.index(np.array(pts))
    return bool(ok.all() and seen_open[r, c].mean() >= 1 / 3)


def clean_polygon(poly: Polygon, offsets: list[float], p: BetaParams) -> Polygon | None:
    """Remove slivers left by polygon differences: drop sub-1.5 cm spikes and slits (mitre open + close keeps
    corners square), snap coordinates back onto the candidate lines, and remove short jogs."""
    r = 0.015
    g = poly.buffer(-r, join_style="mitre").buffer(r, join_style="mitre")
    g = g.buffer(r, join_style="mitre").buffer(-r, join_style="mitre")
    if g.is_empty:
        return None
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda x: x.area)
    xy = np.asarray(g.exterior.coords)[:-1]
    offs = np.array(sorted(offsets))
    for j in range(2):
        near = offs[np.argmin(np.abs(xy[:, j, None] - offs[None]), axis=1)]
        xy[:, j] = np.where(np.abs(near - xy[:, j]) < 2e-3, near, xy[:, j])
    g = Polygon(xy).simplify(1e-6)
    return remove_jogs(g, p) if g.is_valid and g.area > 0 else None


def remove_jogs(poly: Polygon, p: BetaParams) -> Polygon:
    """Snap steps shorter than jog_max_m onto the longer neighbouring wall.

    A 2-4 cm step between two parallel wall lines is either two capture passes that disagree slightly or a
    skirting/panel edge; a plan with a 3 cm jog is harder to read and its tiny "wall" cannot be measured.

    A snap that gives back the same outline ends the loop: the step depends on the outline only, so it would repeat
    forever (an invalid outline with a zero-width slit did this on the door-stitched k38 scenes, D-088). The step cap
    is a backstop; each real snap removes a corner pair."""
    changed = True
    steps_left = 10 * len(poly.exterior.coords) + 100
    while changed and steps_left > 0:
        changed = False
        steps_left -= 1
        xy = _ccw_coords(poly)
        n = len(xy)
        if n <= 4:
            break
        d = np.round(np.roll(xy, -1, axis=0) - xy, 6)            # rounding: sign(1e-17) must be 0, not 1
        length = np.abs(d).sum(1)
        for k in np.argsort(length):
            if length[k] >= p.jog_max_m:
                break
            km1, kp1 = (k - 1) % n, (k + 1) % n
            if np.sign(d[km1]).tolist() != np.sign(d[kp1]).tolist():
                continue                                         # not a step between parallel same-direction walls
            axis = 0 if abs(d[km1][0]) < 1e-9 else 1             # neighbours are constant-u (0) or constant-v (1)
            src, dst = (km1, kp1) if length[km1] < length[kp1] else (kp1, km1)
            target = xy[dst][axis]
            xy = xy.copy()
            for vi in (src, (src + 1) % n):                      # both vertices of the shorter neighbour edge
                xy[vi, axis] = target
            snapped = Polygon(xy).simplify(1e-6)
            changed = not _same_corners(snapped, poly)
            if changed:
                poly = snapped
            break
    return poly


def _ccw_coords(poly: Polygon) -> np.ndarray:
    xy = np.asarray(poly.exterior.coords)[:-1]
    return xy if poly.exterior.is_ccw else xy[::-1]


def _same_corners(a: Polygon, b: Polygon) -> bool:
    """Same outline corners (to 1e-6 m), whatever the start vertex or the order."""
    ka, kb = (sorted(map(tuple, np.round(_ccw_coords(g), 6).tolist())) for g in (a, b))
    return ka == kb


def resolve_overlaps(polys: dict[int, Polygon], extents: dict[int, np.ndarray], grid: Grid) -> dict[int, Polygon]:
    """Give each area claimed by two rooms to the room whose free space covers more of it."""
    ids = sorted(polys)
    for i in ids:
        for j in ids:
            if j <= i or polys[i] is None or polys[j] is None:
                continue
            inter = polys[i].intersection(polys[j])
            if inter.area < 1e-4:
                continue
            ci, cj = (_coverage(extents[k], inter, grid) for k in (i, j))
            loser = j if ci >= cj else i
            winner = i if loser == j else j
            rest = polys[loser].difference(polys[winner])
            if rest.geom_type == "MultiPolygon":
                rest = max(rest.geoms, key=lambda g: g.area)
            polys[loser] = Polygon(rest.exterior).simplify(1e-6) if rest.area > 0 else None
    return polys


def _coverage(mask: np.ndarray, region, grid: Grid) -> float:
    """Fraction of the region's raster cells inside the mask (region is a rectilinear shapely geometry)."""
    u0, v0, u1, v1 = region.bounds
    c0, c1 = int(np.floor(grid.col_of(u0))), int(np.ceil(grid.col_of(u1)))
    r0, r1 = int(np.floor(grid.row_of(v0))), int(np.ceil(grid.row_of(v1)))
    rr, cc = np.mgrid[max(r0, 0):min(r1, grid.rows), max(c0, 0):min(c1, grid.cols)]
    xy = grid.centers(rr.ravel(), cc.ravel())
    inside = contains_xy(region, xy[:, 0], xy[:, 1])
    return float(mask[rr.ravel()[inside], cc.ravel()[inside]].mean()) if inside.any() else 0.0


def polygon_to_geometry(poly: Polygon, lines: list[Line], label: int) -> RoomGeometry:
    """Express the polygon as CCW edges on lines; keep only the lines that are used."""
    if not poly.exterior.is_ccw:
        poly = Polygon(list(poly.exterior.coords)[::-1])
    xy = np.asarray(poly.exterior.coords)[:-1]
    used, edge_lines = {}, []
    for k in range(len(xy)):
        a, b = xy[k], xy[(k + 1) % len(xy)]
        axis = 0 if abs(a[0] - b[0]) < 1e-6 else 1
        off = a[axis]
        cands = [i for i, l in enumerate(lines) if l.axis == axis and abs(l.offset - off) < 1e-6]
        if not cands:                                            # a coordinate created by clipping: new line
            lines = lines + [Line(axis, float(off), 0.0, inferred=True)]
            cands = [len(lines) - 1]
        i = min(cands, key=lambda i: lines[i].inferred)          # prefer an evidence line over an inferred one
        edge_lines.append(used.setdefault(i, len(used)))
    order = sorted(used, key=used.get)
    return RoomGeometry(label, [replace(lines[i]) for i in order], edge_lines)   # copies: refined in place


# ---------------------------------------------------------------- c) raw-point refinement

def edge_normals(geom: RoomGeometry) -> np.ndarray:
    """Inward unit normal per edge (CCW polygon: interior is on the left of each edge)."""
    V = geom.vertices()
    d = np.roll(V, -1, axis=0) - V
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def room_visits(poly: Polygon, traj_uv: np.ndarray, times: np.ndarray, p: BetaParams) -> list[tuple[int, int]]:
    """Temporally contiguous visits of the camera to a room: [first, last] frame index of each stay inside it.

    Why (X-1): ARKit is locally exact but drifts between passes. The drift module's held-out residual between two
    observations of the same surface at different times is 1.9-2.5 cm RMS, while consecutive fragments agree to
    ~0.7 cm (drift.md 5.2). A room measured on one visit has all its walls in one locally consistent frame."""
    inside = contains_xy(poly, traj_uv[:, 0], traj_uv[:, 1])
    idx = np.flatnonzero(inside)
    if len(idx) == 0:
        return []
    split = np.flatnonzero(np.diff(times[idx]) > p.visit_gap_s) + 1
    out = []
    for seg in np.split(idx, split):
        if times[seg[-1]] - times[seg[0]] >= p.visit_min_s:
            out.append((int(seg[0]), int(seg[-1])))
    return out


def choose_visit(geom: RoomGeometry, index: PointIndex, floor_level: float, visits: list[tuple[int, int]],
                 p: BetaParams) -> tuple[int, int] | None:
    """The visit that measures the most walls of the room (>= min_patches seen from inside), ties broken by the
    total number of wall patches, then by the earlier visit (deterministic)."""
    if index.frame is None or len(visits) < 2:
        return visits[0] if visits else None
    segs = _segments(geom)
    best, best_key = None, None
    for v in visits:
        n = [_band_patches(line, sg, index, floor_level, p, v) for line, sg in zip(geom.lines, segs) if sg]
        key = (sum(k >= p.min_patches for k in n), sum(n), -v[0])
        if best_key is None or key > best_key:
            best, best_key = v, key
    return best


def _segments(geom: RoomGeometry):
    V = geom.vertices()
    normals = edge_normals(geom)
    return [[(V[k], V[(k + 1) % len(V)], normals[k]) for k, e in enumerate(geom.edge_lines) if e == li]
            for li in range(len(geom.lines))]


def _band_patches(line: Line, segments, index: PointIndex, floor_level: float, p: BetaParams,
                  visit: tuple[int, int]) -> int:
    """Number of 10 cm patches of this wall's band seen from inside the room during one visit."""
    x, s, h, _ = _band_points(line, segments, index, floor_level, p, visit)
    if len(x) == 0:
        return 0
    return int(len(np.unique(np.stack([np.floor(s / p.patch_m), np.floor(h / p.patch_m)], 1), axis=0)))


def refine_lines(geom: RoomGeometry, index: PointIndex, floor_level: float, p: BetaParams,
                 visit: tuple[int, int] | None = None) -> None:
    """Measure every wall line on raw points (in place).

    Pass 1 fits each line with a free small slope ("tilt"). Walls on these captures deviate from the Manhattan
    axes by a median 0.5-0.9 deg (residual yaw, drift, walls not quite square), and the tilts are NOT
    consistent within a room, so neither a per-wall nor a per-room rotation improved repeatability
    (docs/modules/plan_beta.md, I-9). Pass 2 therefore refits the offset with the slope fixed by the model
    (wall_slope_mode: "axis" = 0, the default; "room" = one rotation per room; "free" = keep pass 1), and the
    tilt the model ignores is put into the uncertainty: at the far end of the wall the real surface can be
    |tilt| x reach away from the fitted line, so |tilt| x reach / 2 is added to the offset sigma."""
    V = geom.vertices()
    normals = edge_normals(geom)
    segments = [[(V[k], V[(k + 1) % len(V)], normals[k]) for k, e in enumerate(geom.edge_lines) if e == li]
                for li in range(len(geom.lines))]
    raster = [l.offset for l in geom.lines]                     # arrangement (free-space) positions
    for line, segs in zip(geom.lines, segments):
        _refine_one(line, segs, index, floor_level, p, fixed_slope=None, visit=visit)
    if p.wall_slope_mode == "free":
        return
    if p.wall_slope_mode == "room":
        geom.yaw, geom.yaw_sigma = room_yaw(geom, segments, p)
    for line, segs, k, r0 in zip(geom.lines, segments, geom.line_slopes(geom.yaw), raster):
        tilt = line.tilt = line.slope
        _refine_one(line, segs, index, floor_level, p, fixed_slope=float(k), visit=visit)
        if abs(line.offset - r0) > p.refine_max_shift_m:
            # the fit locked onto another surface (e.g. one seen through a doorway): keep the free-space position
            line.offset, line.inferred, line.sigma = r0, True, p.inferred_sigma_m
            continue
        line.slope_sigma = 0.0                                   # carried by geom.yaw_sigma (room) or none (axis)
        if not line.inferred and segs:
            reach = max(abs(q[1 - line.axis] - line.mid) for a, b, _ in segs for q in (a, b))
            line.sigma = float(np.hypot(line.sigma, abs(tilt - k) * reach / 2))


def room_yaw(geom: RoomGeometry, segments, p: BetaParams) -> tuple[float, float]:
    """Weighted median of the per-line rotations implied by pass-1 slopes, over lines >= 1 m long."""
    t, w = [], []
    for line, segs in zip(geom.lines, segments):
        length = sum(np.linalg.norm(b - a) for a, b, _ in segs)
        if line.inferred or line.rmse is None or length < p.yaw_min_wall_m:
            continue
        t.append(line.slope if line.axis == 1 else -line.slope)
        w.append(line.n_patches)
    if not t:
        return 0.0, np.tan(np.radians(p.max_slope_deg)) / 2
    t, w = np.array(t), np.array(w, float)
    order = np.argsort(t)
    cw = np.cumsum(w[order])
    med = float(t[order][np.searchsorted(cw, cw[-1] / 2)])
    spread = np.sqrt(np.average((t - med) ** 2, weights=w))
    return med, float(max(spread / np.sqrt(len(t)), 1e-4))


def _band_points(line: Line, segments, index: PointIndex, floor_level: float, p: BetaParams,
                 visit: tuple[int, int] | None):
    """(across, along, height, range) of raw points in the wall's +-refine_band, in the measurement height band,
    away from corners, seen from inside the room, and (if visit is given) observed during that visit."""
    a = line.axis
    across, along, height, rng = [], [], [], []
    for p0, p1, n in segments:
        lo, hi = sorted((p0[1 - a], p1[1 - a]))
        gap = min(p.refine_corner_gap_m, 0.25 * (hi - lo))       # short walls keep their middle half
        lo, hi = lo + gap, hi - gap
        if hi <= lo:
            continue
        box_ = [line.offset - p.refine_band_m, line.offset + p.refine_band_m]
        idx = index.box(*(box_ + [lo, hi])) if a == 0 else index.box(*([lo, hi] + box_))
        if visit is not None and index.frame is not None:
            f = index.frame[idx]
            idx = idx[(f >= visit[0]) & (f <= visit[1])]
        P = index.points[idx]
        h = P[:, 1] - floor_level
        ray = index.ray[idx][:, [0, 2]].astype(np.float32)
        ok = (h > p.refine_h_lo_m) & (h < p.refine_h_hi_m) & (ray @ n.astype(np.float32) < -0.1)
        across.append(P[ok][:, [0, 2]][:, a])
        along.append(P[ok][:, [0, 2]][:, 1 - a])
        height.append(h[ok])
        rng.append(index.range[idx][ok])
    cat = (lambda xs: np.concatenate(xs) if xs else np.empty(0))
    return cat(across), cat(along), cat(height), cat(rng)


def _refine_one(line: Line, segments, index: PointIndex, floor_level: float, p: BetaParams,
                fixed_slope: float | None, visit: tuple[int, int] | None = None) -> None:
    """Fit across = offset + slope * (along - mid) on raw points of this wall, seen from inside the room.
    With fixed_slope, only the offset is fitted.

    v2: with a visit (X-1), only that visit's points are used when they are enough (>= min_refine_points over
    >= min_patches); otherwise all passes are mixed and the line is marked so (B-4 gives it the larger
    between-pass drift sigma). With noise_weights (X-2), each point is weighted by 1 / sigma_lidar(range)^2."""
    x, s, h, r = _band_points(line, segments, index, floor_level, p, visit)
    line.single_visit = visit is not None and index.frame is not None
    if visit is not None and (len(x) < p.min_refine_points or _n_patches(s, h, p) < p.min_patches):
        x, s, h, r = _band_points(line, segments, index, floor_level, p, None)
        line.single_visit = False
    if len(x) < p.min_refine_points:
        line.sigma = p.inferred_sigma_m if line.inferred else max(p.inferred_sigma_m / 2, p.lidar_sigma_m)
        line.n_points = int(len(x))
        line.inferred = True
        if fixed_slope is not None:                               # keep the room rectangular
            line.slope, line.mid = fixed_slope, float(np.mean([0.5 * (a[1 - line.axis] + b[1 - line.axis])
                                                                for a, b, _ in segments])) if segments else 0.0
        return
    w = 1.0 / sigma_lidar(r) ** 2 if p.noise_weights else np.ones(len(x))
    c, k, mid, inl = robust_line(x, s, p, fixed_slope, w)
    c = _band_offset(x, s, h, w, inl, c, k, mid, p)
    patches = np.unique(np.stack([np.floor(s[inl] / p.patch_m), np.floor(h[inl] / p.patch_m)], 1), axis=0)
    resid = x[inl] - (c + k * (s[inl] - mid))
    line.offset, line.slope, line.mid = float(c), float(k), float(mid)
    line.rmse = float(np.sqrt(np.mean(resid ** 2)))
    line.n_points = int(inl.sum())
    line.n_patches = int(len(patches))
    # B-4 (alpha A-8): scatter beyond what the sensor explains at these ranges (two passes, a curtain, a bowed
    # wall). Trimmed at +-1.5 cm, so the expected sensor rmse is computed on the same trimmed distribution.
    expected = _trimmed_rmse(float(np.sqrt(np.mean(sigma_lidar(r[inl]) ** 2))), p.refine_trim_schedule_m[-1])
    line.inconsistency = float(np.sqrt(max(line.rmse ** 2 - expected ** 2, 0.0))) if p.inconsistency_term else 0.0
    line.sigma = offset_sigma(line.rmse, line.n_patches, p, line.single_visit, line.inconsistency)
    spread = float(np.std((patches[:, 0] + 0.5) * p.patch_m))
    line.slope_sigma = float(line.rmse / np.sqrt(max(line.n_patches, 1)) / max(spread, p.patch_m))
    # a line proposed by the room outline alone becomes a measured wall once raw points confirm it over a real
    # area (>= min_patches 10 cm patches seen from inside the room)
    line.inferred = line.n_patches < p.min_patches


def _band_offset(x, s, h, w, inl, c: float, k: float, mid: float, p: BetaParams) -> float:
    """X-3: the offset re-measured only on inliers inside the tape-measure band (measure_h_lo..hi), when set.

    Existence and status decisions keep the full 0.3-2.0 m band (a narrower band drops walls, see the
    ablation); only the position is taken where a person holds the tape (~1 m, above skirting)."""
    if p.measure_h_lo_m is None:
        return c
    sel = inl & (h >= p.measure_h_lo_m) & (h <= p.measure_h_hi_m)
    if sel.sum() < p.min_refine_points:
        return c
    return float(np.average(x[sel] - k * (s[sel] - mid), weights=w[sel]))


def _n_patches(s: np.ndarray, h: np.ndarray, p: BetaParams) -> int:
    if len(s) == 0:
        return 0
    return int(len(np.unique(np.stack([np.floor(s / p.patch_m), np.floor(h / p.patch_m)], 1), axis=0)))


def _trimmed_rmse(sigma: float, trim: float) -> float:
    """RMS of a zero-mean normal(sigma) truncated to +-trim: what pure sensor noise gives after our trimming."""
    from scipy.stats import norm
    a = trim / max(sigma, 1e-9)
    var = sigma ** 2 * (1 - 2 * a * norm.pdf(a) / max(2 * norm.cdf(a) - 1, 1e-12))
    return float(np.sqrt(max(var, 0.0)))


def robust_line(x: np.ndarray, s: np.ndarray, p: BetaParams, fixed_slope: float | None = None,
                w: np.ndarray | None = None) -> tuple[float, float, float, np.ndarray]:
    """(offset at mid, slope, mid, inliers) of the dominant wall surface in the band.

    Start at the mode of a 2 mm histogram of the across-coordinate (the wall face is the densest surface in
    the band; furniture and skirting are sparser), then alternate trimming and least squares with a shrinking
    window, so a sloped wall is followed but nearby parallel surfaces are not captured. The slope is clamped to
    max_slope_deg: a larger tilt means the band holds something other than one straight wall."""
    w = np.ones(len(x)) if w is None else w
    mid = float(np.mean(s))
    k = 0.0 if fixed_slope is None else fixed_slope
    xr = x - k * (s - mid)                                     # across-coordinate relative to the sloped line
    bins = np.arange(xr.min(), xr.max() + 0.002, 0.002)
    if len(bins) < 2:
        return float(np.mean(xr)), k, mid, np.ones(len(x), bool)
    hist, edges = np.histogram(xr, bins)
    hist = np.convolve(hist, np.ones(5) / 5, mode="same")
    c = edges[np.argmax(hist)] + 0.001
    k_max = np.tan(np.radians(p.max_slope_deg))
    for trim in p.refine_trim_schedule_m:
        inl = np.abs(x - (c + k * (s - mid))) < trim
        if inl.sum() < 3:
            break
        wi = w[inl]
        new_mid = float(np.average(s[inl], weights=wi))
        if fixed_slope is None:
            sw = np.sqrt(wi)
            A = np.c_[np.ones(inl.sum()), s[inl] - new_mid] * sw[:, None]
            c, k = np.linalg.lstsq(A, x[inl] * sw, rcond=None)[0]
            k = float(np.clip(k, -k_max, k_max))
        else:
            c = float(np.average(x[inl] - k * (s[inl] - new_mid), weights=wi))
        mid = new_mid
    inl = np.abs(x - (c + k * (s - mid))) < p.refine_trim_schedule_m[-1]
    return float(c), k, mid, inl


def offset_sigma(rmse: float, n_patches: int, p: BetaParams, single_visit: bool = False,
                 inconsistency: float = 0.0) -> float:
    """1-sigma of a wall offset: fit standard error (+) LiDAR bias floor (+) residual drift (+) inconsistency.

    The standard error uses the number of 10 cm patches, not points: neighbouring points share the same
    wall bump and the same pose error, so they are not independent samples.
    v2 (B-4): the drift term is the within-visit value (ICP floor) for a single-visit wall and the drift module's
    held-out between-pass residual otherwise; the inconsistency term (alpha A-8) is NOT divided by sqrt(n): it is
    a systematic smear (two passes offset, a curtain) that more points do not average away."""
    se = rmse / np.sqrt(max(n_patches, 1))
    drift = p.drift_sigma_m if single_visit else p.drift_mixed_sigma_m
    return float(np.sqrt(se ** 2 + p.lidar_sigma_m ** 2 + drift ** 2 + inconsistency ** 2))


def room_polygon(labels, label, vertical, grid, seen_open, p: BetaParams, tall=None):
    """(polygon, candidate lines, extent mask, doorway bumps) for one room region, before overlap resolution.

    vertical = vertical_points(...) of the capture; tall = the same in the wall band (1.0-2.0 m, above most
    furniture), used by the S-3 enclosed-cell rule (None = rule off)."""
    extent = room_extent_mask(labels, label, p)
    lines = evidence_lines(*vertical, grid, extent, p)
    lines += mask_lines(extent, grid, lines, p)
    poly = arrangement_polygon(extent, grid, lines, p, tall)
    if poly is None or len(poly.exterior.coords) < 5:
        return None, lines, extent, []
    poly, bumps = remove_doorway_bumps(remove_jogs(poly, p), seen_open, grid, p)
    return poly, lines, extent, bumps


# ---------------------------------------------------------------- d) v2 B-2: canonical outline

def canonicalize(geom: RoomGeometry, index: PointIndex, floor_level: float, p: BetaParams) -> int:
    """Remove steps between parallel walls that are not real geometry; returns the number of merges.

    Why: most repeat-pair failures are TOPOLOGY failures (judge J-I-4/B-2): a 2-5 cm step splits a long wall in one
    capture and not in the other, so the corners, and every length next to them, come from different walls.
    Bigger raster jog snapping was tried by the judge and made it worse (corners move differently per capture).
    v2 works AFTER the raw-point refinement instead, where offsets are measured to a few mm, and merges a step
    only when (a) the two parallel walls are the same plane within measurement noise (|offset difference| <=
    canon_coplanar_m), or (b) the step's connecting edge is short (< canon_min_edge_m) and has no raw-point
    support, and the step is shallow (<= canon_max_jog_m). The merged wall is re-measured on raw points over its
    whole new extent. A measured short edge (a pillar, a real recess) is kept."""
    if not p.canonical_outline:
        return 0
    merges = 0
    while len(geom.edge_lines) > 4 and _merge_one_step(geom, index, floor_level, p):
        merges += 1
    _drop_unused_lines(geom)
    return merges


def _merge_one_step(geom: RoomGeometry, index: PointIndex, floor_level: float, p: BetaParams) -> bool:
    V = geom.vertices()
    n = len(geom.edge_lines)
    length = np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1)
    if _remove_degenerate(geom, length) or _remove_bump(geom, V, length, p):
        return True
    for k in np.lexsort((np.arange(n), length)):                # shortest first, ties by position (deterministic)
        e_prev, e, e_next = geom.edge_lines[k - 1], geom.edge_lines[k], geom.edge_lines[(k + 1) % n]
        lp, ln = geom.lines[e_prev], geom.lines[e_next]
        if e_prev == e_next or lp.axis != ln.axis:
            continue
        d_prev, d_next = V[k] - V[k - 1], V[(k + 2) % n] - V[(k + 1) % n]
        if float(d_prev @ d_next) <= 0:                           # a notch or bump, not a step
            continue
        jog = abs(lp.offset - ln.offset)
        keep, drop = (e_prev, e_next) if _strength(lp) >= _strength(ln) else (e_next, e_prev)
        coplanar = jog <= p.canon_coplanar_m and not lp.inferred and not ln.inferred
        unsupported = (length[k] < p.canon_min_edge_m and geom.lines[e].inferred and jog <= p.canon_max_jog_m)
        if not (coplanar or unsupported):
            continue
        edges = [keep if x == drop else x for i, x in enumerate(geom.edge_lines) if i != k]
        geom.edge_lines = _collapse(edges)
        if coplanar:
            _refit(geom, keep, index, floor_level, p)
        return True
    return False


def _remove_degenerate(geom: RoomGeometry, length: np.ndarray) -> bool:
    """A zero-length edge left by a merge (its two neighbours now share one line): drop it and join them."""
    n = len(geom.edge_lines)
    for k in range(n):
        if length[k] < 1e-6 and geom.edge_lines[k - 1] == geom.edge_lines[(k + 1) % n]:
            geom.edge_lines = _collapse([x for i, x in enumerate(geom.edge_lines) if i != k])
            return True
    return False


def _remove_bump(geom: RoomGeometry, V: np.ndarray, length: np.ndarray, p: BetaParams) -> bool:
    """A shallow bump or notch (depth <= canon_max_jog_m) whose base wall has no raw-point support and whose two
    sides return to the same wall line: the outline is the free-space edge around furniture or a door reveal,
    not a wall. Remove it, so the long wall stays one wall in every capture."""
    n = len(geom.edge_lines)
    if n < 8:
        return False
    for k in np.lexsort((np.arange(n), length)):
        a, b = geom.edge_lines[(k - 2) % n], geom.edge_lines[(k + 2) % n]
        base = geom.lines[geom.edge_lines[k]]
        if a != b or not base.inferred or base.axis != geom.lines[a].axis:
            continue
        if abs(base.offset - geom.lines[a].offset) > p.canon_max_jog_m:
            continue
        drop = {(k - 1) % n, k, (k + 1) % n}
        geom.edge_lines = _collapse([x for i, x in enumerate(geom.edge_lines) if i not in drop])
        return True
    return False


def _strength(line: Line) -> tuple:
    return (not line.inferred, line.n_patches, -line.sigma)


def _collapse(edges: list[int]) -> list[int]:
    """Consecutive edges on the same line are one edge (cyclic)."""
    out = [x for i, x in enumerate(edges) if x != edges[i - 1]]
    return out if out else edges[:1]


def _drop_unused_lines(geom: RoomGeometry) -> None:
    used = sorted(set(geom.edge_lines), key=geom.edge_lines.index)
    remap = {old: new for new, old in enumerate(used)}
    geom.lines = [geom.lines[i] for i in used]
    geom.edge_lines = [remap[i] for i in geom.edge_lines]


def _refit(geom: RoomGeometry, li: int, index: PointIndex, floor_level: float, p: BetaParams) -> None:
    """Re-measure one line on raw points over all edges it now carries (slope fixed to the room model)."""
    segs = _segments(geom)[li]
    line = geom.lines[li]
    _refine_one(line, segs, index, floor_level, p, fixed_slope=float(line.slope))


# ---------------------------------------------------------------- e) v6 (I-011): outline at openings

@dataclass
class OpeningEvidence:
    """What tells an opening in a wall slab apart from real geometry (a niche, a pilaster, a duct), for one plan."""
    spans: list[tuple[int, float, float, float]]   # (axis, plane offset, s_lo, s_hi): header open spans (D-046)
    headers: list[dict]                            # D-046 header cuts with their far face (lines.header_cut_mask)
    sill: tuple[np.ndarray, np.ndarray]            # vertical surface in opening_sill_band (plan points, normals)
    tall: tuple[np.ndarray, np.ndarray]            # ... in the wall band (wall_band_lo..hi: above furniture)
    head: tuple[np.ndarray, np.ndarray]            # ... in the header band (above doors and windows)


def outline_at_openings(geom: RoomGeometry, ev: OpeningEvidence, index: PointIndex, floor_level: float,
                        p: BetaParams) -> list[dict]:
    """Put the room boundary on the wall's INNER face where it crosses a wall slab at an opening (in place).

    Why (I-011): the outline follows free space, and at an opening free space runs into the wall slab. Two failures
    survived remove_doorway_bumps and canonicalize:
      1. Wide openings (D-046) are cut on the header face coplanar with the NEAR room's wall, so the far room's side
         lies on the far face of the wall: one wall thickness too long (sim kitchen 2.689 m vs 2.474 m).
      2. Reveals of doors and windows stay as 8-34 cm excursions (window recesses have no open floor beyond, wide
         openings exceed door_max_m), splitting one wall side into 3-5 "walls" (sim 1 BHK: 52 walls for 26).
    A laser-measuring assessor measures one wall per wall side, face to face, and the opening is a hole in it.

    Rules (each change only removes area, so rooms cannot start to overlap):
      a) an edge on a header cut, in the far room, moves to the header's far face (lines._far_face), or to this
         room's own line on that face when it has one (its wall beside the opening);
      b) an excursion (bump into the slab between two pieces of one wall plane, or a step at a corner) at most
         thickness_max deep is closed onto the inner wall line when there is opening evidence over it: a header
         open span (wide openings), or a window recess (lintel on the inner plane, window frame seen behind, nothing
         at sill height). A niche, duct recess or pilaster has none of these and stays (_opening_evidence).
         Narrow-door reveals are remove_doorway_bumps' job (raster stage); door necks are NOT used here: on the
         real repeat pair (with_ceiling vs floor_only) they fired in one capture and not the other, and the walls
         passing the repeatability gate fell from 15/87 to 11/83.
    Openings are found afterwards on the merged faces, so the opening lies in the one wall that carries it.
    Returns what was changed (plan.meta.opening_outline)."""
    done: list[dict] = []
    if not p.opening_outline:
        return done
    for _ in range(4 * len(geom.edge_lines)):          # each change removes edges or moves one off its cut for good
        step = _header_far_face(geom, ev, index, floor_level, p) or _opening_excursion(geom, ev, index, floor_level, p)
        if step is None:
            break
        done.append(step)
    _drop_unused_lines(geom)
    return done


def _span_of(V: np.ndarray, k: int, axis: int) -> tuple[float, float]:
    a, b = V[k][1 - axis], V[(k + 1) % len(V)][1 - axis]
    return (float(min(a, b)), float(max(a, b)))


def _header_far_face(geom: RoomGeometry, ev: OpeningEvidence, index: PointIndex, floor_level: float,
                     p: BetaParams) -> dict | None:
    """Rule a): one edge of the far room that lies on a header cut moves to the header's far face."""
    V = geom.vertices()
    nrm = edge_normals(geom)
    for k, e in enumerate(geom.edge_lines):
        line = geom.lines[e]
        a = line.axis
        s = float(np.sign(nrm[k][a]))                    # +1: the room lies at larger across-coordinates
        lo, hi = _span_of(V, k, a)
        for h in ev.headers:
            if (h["axis"] != a or h.get("far") is None or h.get("sign") != -s
                    or abs(line.offset - h["offset"]) > p.opening_on_cut_m):
                continue                                 # not this room's side of a header cut (near room: stays)
            if min(hi, h["s1"]) - max(lo, h["s0"]) < 0.5 * min(hi - lo, h["s1"] - h["s0"]):
                continue
            jog = (h["far"] - line.offset) * s           # how far the far face lies inside the room
            if not p.opening_jog_min_m <= jog <= p.thickness_max_m:
                continue
            before = _copy_geom(geom)
            own = [i for i, l in enumerate(geom.lines) if l.axis == a and abs(l.offset - h["far"]) <= 0.04 and i != e]
            if own:                                      # this room's wall beside the opening, on the far face
                target = min(own, key=lambda i: abs(geom.lines[i].offset - h["far"]))
            else:
                geom.lines.append(Line(a, float(h["far"]), 0.0, inferred=True, slope=line.slope, mid=line.mid))
                target = len(geom.lines) - 1
            geom.edge_lines = [target if j == k else x for j, x in enumerate(geom.edge_lines)]
            _drop_steps(geom)
            if not _shrinks(geom, before):
                _restore(geom, before)
                continue
            _fit_face(geom, target, index, floor_level, p, h["far"])
            if not _shrinks(geom, before):
                _restore(geom, before)
                continue
            return dict(rule="header far face", axis=a, cut=round(float(line.offset), 3),
                        face=round(float(geom.lines[target].offset), 3), span=[round(lo, 3), round(hi, 3)],
                        measured=not geom.lines[target].inferred)
    return None


def _opening_excursion(geom: RoomGeometry, ev: OpeningEvidence, index: PointIndex, floor_level: float,
                       p: BetaParams) -> dict | None:
    """Rule b): close one excursion into a wall slab that has opening evidence over it.

    An excursion starts at an anchor edge on a measured wall line A and walks along the outline (either way) through
    short steps (<= thickness_max) and pieces parallel to A that lie BEHIND it (in the slab, opening_jog_min_m to
    thickness_max deep), until it comes back to A's plane (a bump; staircase reveals included) or runs into a corner
    (a long side wall). Inward notches (pilasters, duct boxes, furniture) are never excursions."""
    n = len(geom.edge_lines)
    if n <= 4:
        return None
    V = geom.vertices()
    length = np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1)
    nrm = edge_normals(geom)
    side = [float(np.sign(nrm[k][geom.lines[e].axis])) for k, e in enumerate(geom.edge_lines)]
    for i in np.lexsort((np.arange(n), -length)):      # longest anchor first, ties by position (deterministic)
        A = geom.lines[geom.edge_lines[i]]
        if A.inferred:
            continue                                     # close an excursion only onto a wall measured on raw points
        a, s = A.axis, side[i]
        for d in (1, -1):
            pieces, end, k = [], None, i
            for _ in range(n - 2):
                step, k = (k + d) % n, (k + 2 * d) % n
                if length[step] > p.thickness_max_m:
                    end = "corner" if pieces else None   # the excursion runs into a corner: a real side wall
                    break
                L = geom.lines[geom.edge_lines[k]]
                if L.axis != a or side[k] != s:
                    break
                jog = (A.offset - L.offset) * s          # > 0: L lies behind A, in the wall slab
                if abs(jog) <= p.canon_coplanar_m:
                    end = "bump" if pieces else None     # back on A's plane
                    break
                if not p.opening_jog_min_m <= jog <= p.thickness_max_m:
                    break
                pieces.append(k)
            if end is None:
                continue
            spans = [_span_of(V, k, a) for k in pieces]
            lo, hi = min(x[0] for x in spans), max(x[1] for x in spans)
            why = _opening_evidence([geom.lines[geom.edge_lines[k]] for k in pieces], spans, A, s, lo, hi, ev, p)
            if why is None:
                continue
            before = _copy_geom(geom)
            keep = geom.edge_lines[i]
            moved = set(pieces) | ({k} if end == "bump" else set())     # pieces and the far anchor go onto A
            geom.edge_lines = [keep if j in moved else x for j, x in enumerate(geom.edge_lines)]
            _drop_steps(geom)
            if not _shrinks(geom, before):
                _restore(geom, before)
                continue
            _remeasure(geom, keep, index, floor_level, p)   # the wall now spans the opening: fit it over all of it
            depth = max(abs(before.lines[before.edge_lines[k]].offset - A.offset) for k in pieces)
            return dict(rule="excursion", evidence=why[0], score=round(why[1], 2), axis=a, shape=end,
                        pieces=len(pieces), depth=round(float(depth), 3), span=[round(lo, 3), round(hi, 3)],
                        wall=round(float(geom.lines[keep].offset), 3))
    return None


def _opening_evidence(pieces: list[Line], spans: list[tuple[float, float]], A: Line, s: float, lo: float,
                      hi: float, ev: OpeningEvidence, p: BetaParams) -> tuple[str, float] | None:
    """Why the excursion (pieces behind wall line A, along [lo, hi]) is an opening, or None (keep it).

    wide opening: header open spans (D-046) cover >= opening_min_cover of it, and where they do not, the wall
        beside the opening is on A's plane, not on the pieces' (wall-band surface). Else a piece is the real wall
        (e.g. a long wall with several doors whose short end piece looked like an anchor).
    window recess: the wall plane A goes on ABOVE the excursion (header band: the lintel), the window (frame,
        glass) is seen in the wall band at the recess back, and the recess has no surface at sill height (it does
        not reach the floor). The sill wall itself is often hidden by a kitchen counter, so it is not required.
        A niche or alcove shows its back wall at sill height; a wall that really runs further out has no lintel on
        A's plane (it rises at the piece); an open doorway shows nothing at the back (sim: window backs 38-42% of
        the wall band, the doorway-like real cases 0%); a pilaster or duct box is an inward notch, never an
        excursion."""
    if hi - lo <= 0 or hi - lo > p.opening_max_m:
        return None
    a = A.axis
    nb = max(int(np.ceil((hi - lo) / p.cell_m)), 1)
    half = min(0.03, 0.45 * min(abs(A.offset - F.offset) for F in pieces))   # plane bands must not overlap

    def on_pieces(pts):                              # per bin: surface on the piece that lies behind that bin
        out = np.zeros(nb, bool)
        for F, (s0, s1) in zip(pieces, spans):
            b0, b1 = int(np.floor((s0 - lo) / p.cell_m)), int(np.ceil((s1 - lo) / p.cell_m))
            out[b0:b1] |= _side_bins(*pts, a, F.offset, lo, hi, s, nb, half)[b0:b1]
        return out

    cover = np.zeros(nb, bool)
    for ax, off, s0, s1 in ev.spans:                 # header open spans in this wall slab
        if ax == a and abs(off - A.offset) <= p.thickness_max_m + 0.05 and s1 > lo and s0 < hi:
            cover[int(np.floor((max(s0, lo) - lo) / p.cell_m)):int(np.ceil((min(s1, hi) - lo) / p.cell_m))] = True
    if cover.mean() >= p.opening_min_cover:
        rest = ~cover
        wall_a = _side_bins(*ev.tall, a, A.offset, lo, hi, s, nb, half)[rest].mean() if rest.any() else 0.0
        wall_f = on_pieces(ev.tall)[rest].mean() if rest.any() else 0.0
        if not rest.any() or wall_a >= wall_f:
            return "wide opening (header open span)", float(cover.mean())
    head_a = _side_bins(*ev.head, a, A.offset, lo, hi, s, nb, half).mean()
    if (head_a >= p.opening_head_min and on_pieces(ev.tall).mean() >= p.opening_recess_wall_min
            and on_pieces(ev.sill).mean() <= p.opening_recess_sill_max):
        return "window recess (lintel on the inner plane, window behind, nothing at sill height)", float(head_a)
    return None


def _side_bins(P: np.ndarray, N: np.ndarray, axis: int, off: float, lo: float, hi: float, inward: float,
               nb: int, half: float = 0.03) -> np.ndarray:
    """Per cell-size bin along [lo, hi]: is there vertical surface within `half` of the line, facing `inward`?"""
    m = ((np.abs(P[:, axis] - off) < half) & (P[:, 1 - axis] >= lo) & (P[:, 1 - axis] < hi)
         & (N[:, axis] * inward > np.cos(np.radians(30))))
    out = np.zeros(nb, bool)
    out[np.clip(((P[m, 1 - axis] - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)] = True
    return out


def _fit_face(geom: RoomGeometry, li: int, index: PointIndex, floor_level: float, p: BetaParams,
              far: float) -> None:
    """Measure the moved line on raw points over all its edges: wall band first (the wall beside the opening); if
    that sees too little (the whole side is an opening), the header band (the header's far face, above the opening);
    else it stays at the header's far face from the TSDF, inferred. A fit that jumps > refine_max_shift_m is another
    surface (as in refine_lines): keep the TSDF position."""
    line = geom.lines[li]
    segs = _segments(geom)[li]
    head = replace(p, refine_h_lo_m=p.header_band_lo_m, refine_h_hi_m=p.header_band_hi_m, measure_h_lo_m=None)
    for q in (p, head):
        trial = replace(line)
        _refine_one(trial, segs, index, floor_level, q, fixed_slope=float(line.slope))
        if not trial.inferred and abs(trial.offset - far) <= p.refine_max_shift_m:
            line.__dict__.update(trial.__dict__)
            return
    line.offset, line.inferred = float(far), True
    line.sigma = max(line.sigma, p.inferred_sigma_m / 2, p.lidar_sigma_m)


def _remeasure(geom: RoomGeometry, li: int, index: PointIndex, floor_level: float, p: BetaParams) -> None:
    """Refit a kept wall line over its merged extent (as canonicalize does after a coplanar merge). Kept only if it
    stays measured and moves less than a reveal is deep (opening_jog_min_m): else the fit reached into the slab."""
    trial = _copy_geom(geom)
    _refit(trial, li, index, floor_level, p)
    new, old = trial.lines[li], geom.lines[li]
    if not new.inferred and abs(new.offset - old.offset) < p.opening_jog_min_m:
        geom.lines[li] = new


def _drop_steps(geom: RoomGeometry) -> None:
    """Remove edges left with zero length (both neighbours now on one line), then join same-line neighbours."""
    changed = True
    while changed and len(geom.edge_lines) > 4:
        changed = False
        n = len(geom.edge_lines)
        for k in range(n):
            if geom.edge_lines[k - 1] == geom.edge_lines[(k + 1) % n] != geom.edge_lines[k]:
                geom.edge_lines = _collapse([x for i, x in enumerate(geom.edge_lines) if i != k])
                changed = True
                break
    geom.edge_lines = _collapse(geom.edge_lines)


def _copy_geom(geom: RoomGeometry) -> RoomGeometry:
    return RoomGeometry(geom.room_label, [replace(l) for l in geom.lines], list(geom.edge_lines), geom.yaw,
                        geom.yaw_sigma)


def _restore(geom: RoomGeometry, before: RoomGeometry) -> None:
    geom.lines, geom.edge_lines = before.lines, before.edge_lines


def _shrinks(geom: RoomGeometry, before: RoomGeometry) -> bool:
    """The new outline is a valid rectilinear polygon inside the old one (only wall-slab area was removed)."""
    if len(geom.edge_lines) < 4 or len(geom.edge_lines) % 2:
        return False
    new, old = Polygon(geom.vertices()), Polygon(before.vertices())
    return bool(new.is_valid and new.area > 0.5 * old.area and new.difference(old).area < 1e-4)
