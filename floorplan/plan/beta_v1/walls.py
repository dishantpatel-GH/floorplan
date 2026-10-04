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

from floorplan.plan.beta_v1.freespace import disk
from floorplan.plan.beta_v1.grid import Grid, PointIndex
from floorplan.plan.beta_v1.params import BetaParams


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


def vertical_points(points: np.ndarray, normals: np.ndarray, floor_y: float, p: BetaParams):
    """Plan positions and plan normals of vertical TSDF surface between 0.3 and 2.0 m (computed once)."""
    h = points[:, 1] - floor_y
    sel = (np.abs(normals[:, 1]) < 0.3) & (h > p.refine_h_lo_m) & (h < p.refine_h_hi_m)
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

def arrangement_polygon(extent: np.ndarray, grid: Grid, lines: list[Line], p: BetaParams) -> Polygon | None:
    """Union of arrangement rectangles that the room mask covers by >= cell_cover_min."""
    us = sorted({l.offset for l in lines if l.axis == 0})
    vs = sorted({l.offset for l in lines if l.axis == 1})
    if len(us) < 2 or len(vs) < 2:
        return None
    S = np.pad(extent.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ci = np.clip(np.round([grid.col_of(u) for u in us]).astype(int), 0, grid.cols)
    ri = np.clip(np.round([grid.row_of(v) for v in vs]).astype(int), 0, grid.rows)
    boxes = []
    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            area = (ci[i + 1] - ci[i]) * (ri[j + 1] - ri[j])
            if area <= 0:
                continue
            s = S[ri[j + 1], ci[i + 1]] - S[ri[j], ci[i + 1]] - S[ri[j + 1], ci[i]] + S[ri[j], ci[i]]
            if s / area >= p.cell_cover_min:
                boxes.append(box(us[i], vs[j], us[i + 1], vs[j + 1]))
    if not boxes:
        return None
    poly = unary_union(boxes)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return Polygon(poly.exterior).simplify(1e-6)


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
    skirting/panel edge; a plan with a 3 cm jog is harder to read and its tiny "wall" cannot be measured."""
    changed = True
    while changed:
        changed = False
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
            poly = Polygon(xy).simplify(1e-6)
            changed = True
            break
    return poly


def _ccw_coords(poly: Polygon) -> np.ndarray:
    xy = np.asarray(poly.exterior.coords)[:-1]
    return xy if poly.exterior.is_ccw else xy[::-1]


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


def refine_lines(geom: RoomGeometry, index: PointIndex, floor_level: float, p: BetaParams) -> None:
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
        _refine_one(line, segs, index, floor_level, p, fixed_slope=None)
    if p.wall_slope_mode == "free":
        return
    if p.wall_slope_mode == "room":
        geom.yaw, geom.yaw_sigma = room_yaw(geom, segments, p)
    for line, segs, k, r0 in zip(geom.lines, segments, geom.line_slopes(geom.yaw), raster):
        tilt = line.tilt = line.slope
        _refine_one(line, segs, index, floor_level, p, fixed_slope=float(k))
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


def _refine_one(line: Line, segments, index: PointIndex, floor_level: float, p: BetaParams,
                fixed_slope: float | None) -> None:
    """Fit across = offset + slope * (along - mid) on raw points of this wall, seen from inside the room.
    With fixed_slope, only the offset is fitted."""
    a = line.axis
    across, along, height = [], [], []
    for p0, p1, n in segments:
        lo, hi = sorted((p0[1 - a], p1[1 - a]))
        gap = min(p.refine_corner_gap_m, 0.25 * (hi - lo))       # short walls keep their middle half
        lo, hi = lo + gap, hi - gap
        if hi <= lo:
            continue
        box_ = [line.offset - p.refine_band_m, line.offset + p.refine_band_m]
        idx = index.box(*(box_ + [lo, hi])) if a == 0 else index.box(*([lo, hi] + box_))
        P = index.points[idx]
        h = P[:, 1] - floor_level
        ray = index.ray[idx][:, [0, 2]].astype(np.float32)
        ok = (h > p.refine_h_lo_m) & (h < p.refine_h_hi_m) & (ray @ n.astype(np.float32) < -0.1)
        across.append(P[ok][:, [0, 2]][:, a])
        along.append(P[ok][:, [0, 2]][:, 1 - a])
        height.append(h[ok])
    x = np.concatenate(across) if across else np.empty(0)
    if len(x) < p.min_refine_points:
        line.sigma = p.inferred_sigma_m if line.inferred else max(p.inferred_sigma_m / 2, p.lidar_sigma_m)
        line.n_points = int(len(x))
        line.inferred = True
        if fixed_slope is not None:                               # keep the room rectangular
            line.slope, line.mid = fixed_slope, float(np.mean([0.5 * (a[1 - line.axis] + b[1 - line.axis])
                                                                for a, b, _ in segments])) if segments else 0.0
        return
    s, h = np.concatenate(along), np.concatenate(height)
    c, k, mid, inl = robust_line(x, s, p, fixed_slope)
    patches = np.unique(np.stack([np.floor(s[inl] / p.patch_m), np.floor(h[inl] / p.patch_m)], 1), axis=0)
    resid = x[inl] - (c + k * (s[inl] - mid))
    line.offset, line.slope, line.mid = float(c), float(k), float(mid)
    line.rmse = float(np.sqrt(np.mean(resid ** 2)))
    line.n_points = int(inl.sum())
    line.n_patches = int(len(patches))
    line.sigma = offset_sigma(line.rmse, line.n_patches, p)
    spread = float(np.std((patches[:, 0] + 0.5) * p.patch_m))
    line.slope_sigma = float(line.rmse / np.sqrt(max(line.n_patches, 1)) / max(spread, p.patch_m))
    # a line proposed by the room outline alone becomes a measured wall once raw points confirm it over a real
    # area (>= min_patches 10 cm patches seen from inside the room)
    line.inferred = line.n_patches < p.min_patches


def robust_line(x: np.ndarray, s: np.ndarray, p: BetaParams, fixed_slope: float | None = None
                ) -> tuple[float, float, float, np.ndarray]:
    """(offset at mid, slope, mid, inliers) of the dominant wall surface in the band.

    Start at the mode of a 2 mm histogram of the across-coordinate (the wall face is the densest surface in
    the band; furniture and skirting are sparser), then alternate trimming and least squares with a shrinking
    window, so a sloped wall is followed but nearby parallel surfaces are not captured. The slope is clamped to
    max_slope_deg: a larger tilt means the band holds something other than one straight wall."""
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
        new_mid = float(np.mean(s[inl]))
        if fixed_slope is None:
            A = np.c_[np.ones(inl.sum()), s[inl] - new_mid]
            c, k = np.linalg.lstsq(A, x[inl], rcond=None)[0]
            k = float(np.clip(k, -k_max, k_max))
        else:
            c = float(np.mean(x[inl] - k * (s[inl] - new_mid)))
        mid = new_mid
    inl = np.abs(x - (c + k * (s - mid))) < p.refine_trim_schedule_m[-1]
    return float(c), k, mid, inl


def offset_sigma(rmse: float, n_patches: int, p: BetaParams) -> float:
    """1-sigma of a wall offset: fit standard error (+) LiDAR bias floor (+) residual drift.

    The standard error uses the number of 10 cm patches, not points: neighbouring points share the same
    wall bump and the same pose error, so they are not independent samples."""
    se = rmse / np.sqrt(max(n_patches, 1))
    return float(np.sqrt(se ** 2 + p.lidar_sigma_m ** 2 + p.drift_sigma_m ** 2))


def room_polygon(labels, label, vertical, grid, seen_open, p: BetaParams):
    """(polygon, candidate lines, extent mask, doorway bumps) for one room region, before overlap resolution.

    vertical = vertical_points(...) of the capture."""
    extent = room_extent_mask(labels, label, p)
    lines = evidence_lines(*vertical, grid, extent, p)
    lines += mask_lines(extent, grid, lines, p)
    poly = arrangement_polygon(extent, grid, lines, p)
    if poly is None or len(poly.exterior.coords) < 5:
        return None, lines, extent, []
    poly, bumps = remove_doorway_bumps(remove_jogs(poly, p), seen_open, grid, p)
    return poly, lines, extent, bumps
