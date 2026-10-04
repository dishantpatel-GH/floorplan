"""Openings: doors/passages between rooms, doorways to unmapped space, windows, plus mirror and glass handling.

Where openings come from:
  * Doors between rooms = the narrow necks that kept two rooms apart in segmentation (segment.py).
  * Doorways to space we did not map = the doorway bumps cut off room polygons (walls.remove_doorway_bumps).
  * Openings in a wall plane = places where LiDAR rays from inside the room CROSSED the wall plane and hit
    something beyond it. Each ray is traced from its camera origin (point - range x direction) to the plane.
    Open from the floor up -> door; open between ~0.3 m and ~2.3 m with wall below -> window.

Width is measured between the jamb faces on raw points, not on the raster: inside the wall slab (between the two
wall faces, so architraves on the faces are excluded) and between 0.3 m and 1.8 m height, each 5 cm height slice
gives the left and right edge of the empty gap; the median over slices is the jamb position.

Mirrors: a mirror makes the LiDAR "see through" the wall, because reflected rays return phantom points behind
the wall plane. Test: reflect those points back across the plane. For a mirror they land on real surfaces of
the room (the reflection of the room is the room); for a window they land in empty space.
Glass: clear glass and dark screens return nothing at high confidence. A window-sized region of the wall with no
returns and no see-through, framed by wall on all sides, is reported as a window CANDIDATE (plan.meta), not as
an opening: a wall-mounted TV produces exactly the same signature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import Polygon

from floorplan.model import Measurement, Opening
from floorplan.plan.beta.grid import PointIndex
from floorplan.plan.beta.params import BetaParams


@dataclass
class WallFace:
    """A measured wall edge of a room, in the terms the opening code needs."""
    wall_id: str
    room_id: str
    axis: int            # 0: constant u; 1: constant v
    offset: float
    inward: float        # +1 if the room lies on the +axis side of the face
    span: tuple[float, float]
    floor_level: float
    inferred: bool
    slope: float = 0.0
    mid: float = 0.0

    def at(self, along):
        """Across-coordinate of the face at an along-coordinate (the wall line may be slightly tilted)."""
        return self.offset + self.slope * (along - self.mid)


@dataclass
class Gap:
    left: float
    right: float
    sigma_left: float
    sigma_right: float
    slices: int

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def sigma(self) -> float:
        return float(np.hypot(self.sigma_left, self.sigma_right))


def measure_gap(index: PointIndex, axis: int, t_range: tuple[float, float], center: float, half_w: float,
                h_range: tuple[float, float], floor_level: float, p: BetaParams) -> Gap | None:
    """Jamb-to-jamb gap around `center` along the wall, from raw points inside the wall slab t_range.

    v2 (B-3): first alpha's tall-face jamb rule (jamb_gap); the v1 per-slice solid-bin rule is the fallback."""
    lo, hi = center - half_w - 0.35, center + half_w + 0.35
    t0, t1 = sorted(t_range)
    idx = index.box(t0, t1, lo, hi) if axis == 0 else index.box(lo, hi, t0, t1)
    P = index.points[idx]
    s = P[:, 2] if axis == 0 else P[:, 0]
    h = P[:, 1] - floor_level
    if p.jamb_mode == "alpha":
        g = jamb_gap(s, h, center, half_w, p)
        if g is not None:
            return g
    lefts, rights = [], []
    for z0 in np.arange(h_range[0], h_range[1], p.jamb_slice_m):
        edges = _solid_edges(s[(h >= z0) & (h < z0 + p.jamb_slice_m)], center, lo, hi)
        if edges is not None:
            lefts.append(edges[0])
            rights.append(edges[1])
    if len(lefts) < 5:
        return None
    L, R = np.median(lefts), np.median(rights)
    # standard error of a median ~ 1.2533 * std / sqrt(n); plus the LiDAR bias on each jamb face
    sl = np.hypot(1.2533 * np.std(lefts) / np.sqrt(len(lefts)), p.lidar_sigma_m)
    sr = np.hypot(1.2533 * np.std(rights) / np.sqrt(len(rights)), p.lidar_sigma_m)
    return Gap(float(L), float(R), float(sl), float(sr), len(lefts))


def jamb_gap(s: np.ndarray, h: np.ndarray, center: float, half_w: float, p: BetaParams) -> Gap | None:
    """Alpha's jamb measurement (plan_alpha doors.py, 9 of 11 widths at +-1.4 cm there), on beta's slab points.

    Inside the wall slab the only surfaces the LiDAR can see across the opening are the two jambs, and a jamb is
    a TALL face: floor to door head. So every 1 cm layer near each end of the opening is scored by the share of
    10 cm height bins (0.3-1.8 m) it covers; the jamb is the layer NEAREST the opening whose cover is >= 40% and
    >= 80% of the best layer's. The relative test rejects short things near the opening (the edge of an open door
    leaf, a handle, a switch) however well they were seen. The position is then mean-shifted onto the face."""
    body = (h > p.jamb_h_lo_m) & (h < p.jamb_h_hi_m)
    g0, g1 = center - half_w, center + half_w
    m = p.jamb_search_alpha_m
    left = _alpha_jamb(s[body & (s > g0 - m) & (s < min(g0 + m, center))], h[body & (s > g0 - m) &
                       (s < min(g0 + m, center))], -1, p)
    right = _alpha_jamb(s[body & (s < g1 + m) & (s > max(g1 - m, center))], h[body & (s < g1 + m) &
                        (s > max(g1 - m, center))], +1, p)
    if left is None or right is None or right[0] - left[0] < p.door_min_m * 0.8:
        return None
    sl, sr = (float(np.hypot(j[1], p.lidar_sigma_m)) for j in (left, right))
    return Gap(left[0], right[0], sl, sr, min(left[2], right[2]))


def _alpha_jamb(along: np.ndarray, height: np.ndarray, side: int, p: BetaParams):
    """(position, standard error, n points) of one jamb face, or None. side -1 = left end, +1 = right end.
    Ported from plan_alpha doors._jamb with alpha's constants (15 points, 40% cover)."""
    if len(along) < p.jamb_min_points:
        return None
    x = -along if side == +1 else along                  # "towards the opening" is always increasing x
    edges = np.arange(x.min(), x.max() + 0.02, 0.01)
    if len(edges) < 2:
        return None
    centres = (edges[:-1] + edges[1:]) / 2
    hbins = np.arange(p.jamb_h_lo_m, p.jamb_h_hi_m + 1e-9, 0.10)
    order = np.argsort(x)
    xs, hs = x[order], height[order]
    cover = np.empty(len(centres))
    for i, c in enumerate(centres):                      # sorted slices instead of a full mask per layer (speed)
        a, b = np.searchsorted(xs, [c - 0.015, c + 0.015])
        cover[i] = np.mean(np.histogram(hs[a:b], bins=hbins)[0] >= 3) if b > a else 0.0
    ok = cover >= max(p.jamb_min_cover, 0.8 * cover.max())
    if not ok.any():
        return None
    pos = float(centres[np.flatnonzero(ok).max()])       # nearest the opening
    sel = np.abs(x - pos) < 0.015
    for _ in range(5):                                   # mean-shift onto the face
        sel = np.abs(x - pos) < 0.015
        if not sel.any():
            return None
        pos = float(np.median(x[sel]))
    if sel.sum() < p.jamb_min_points:
        return None
    mad = 1.4826 * float(np.median(np.abs(x[sel] - pos)))
    return (-pos if side == +1 else pos), mad / np.sqrt(sel.sum()), int(sel.sum())


def _solid_edges(s: np.ndarray, center: float, lo: float, hi: float, bin_m: float = 0.01):
    """Nearest SOLID surface left and right of the centre in one height slice.

    Solid = a 1 cm bin holding >= 20% of the slice's densest bin (and >= 3 points). A jamb face seen by the
    LiDAR fills its bins densely; a door handle, a cable or flying pixels give 1-2 points and are skipped.
    The edge is the inner boundary of that bin, refined to the innermost point inside it."""
    if len(s) < 6:
        return None
    edges = np.arange(lo, hi + bin_m, bin_m)
    cnt, _ = np.histogram(s, edges)
    solid = cnt >= max(3, 0.2 * cnt.max())
    c = int((center - lo) / bin_m)
    left_bins = np.nonzero(solid[:c])[0]
    right_bins = np.nonzero(solid[c + 1:])[0] + c + 1
    if len(left_bins) == 0 or len(right_bins) == 0:
        return None
    lb, rb = left_bins[-1], right_bins[0]
    in_l = s[(s >= edges[lb]) & (s < edges[lb + 1])]
    in_r = s[(s >= edges[rb]) & (s < edges[rb + 1])]
    return float(in_l.max()), float(in_r.min())


def head_height(index: PointIndex, axis: int, t_range, gap: Gap, floor_level: float, ceiling_h: float | None,
                p: BetaParams) -> Measurement | None:
    """Underside of the lintel: the lowest 2 cm height level above 1.5 m whose points span >= 60% of the gap.

    "Spans the gap" is the point: a lintel crosses the whole opening, while an open door leaf, a handle or a
    hanging towel only occupies part of it and would otherwise be read as a very low door head."""
    t0, t1 = sorted(t_range)
    lo, hi = gap.left + 0.03, gap.right - 0.03
    if hi <= lo:
        return None
    idx = index.box(t0, t1, lo, hi) if axis == 0 else index.box(lo, hi, t0, t1)
    P = index.points[idx]
    h = P[:, 1] - floor_level
    s = P[:, 2] if axis == 0 else P[:, 0]
    keep = h > 1.5
    h, s = h[keep], s[keep]
    if len(h) < 30:
        return None
    n_cols = max(int((hi - lo) / 0.02), 1)
    col = np.clip(((s - lo) / 0.02).astype(int), 0, n_cols - 1)
    for z in np.arange(1.5, h.max(), 0.02):
        band = (h >= z) & (h < z + 0.02)
        if len(np.unique(col[band])) >= 0.6 * n_cols:
            if ceiling_h is not None and z > ceiling_h - 0.05:
                return None                             # open to the ceiling: no lintel
            under = h[band]
            return Measurement.from_sigma(float(np.median(under)), float(np.hypot(np.std(under), np.sqrt(2) *
                                          p.lidar_sigma_m)), "m", "lowest 2 cm level above 1.5 m whose raw points "
                                          "span >= 60% of the opening width (lintel underside)")
    return None


def slab(face: WallFace, depth: float | None, along: float) -> tuple[float, float]:
    """The across-wall interval inside the wall at `along`, excluding 1.5 cm at each face (architraves)."""
    out, f = -face.inward, face.at(along)
    d = depth if depth is not None else 0.15
    return f + out * 0.015, f + out * max(d - 0.015, 0.03)


def width_measurement(gap: Gap | None, fallback_w: float, method: str) -> Measurement:
    if gap is None:
        return Measurement.from_sigma(fallback_w, 0.05, "m", f"{method}; jambs not resolved on raw points, "
                                      "width from the free-space raster", status="inferred")
    return Measurement.from_sigma(gap.width, gap.sigma, "m", f"{method}; jamb faces from {gap.slices} "
                                  "5 cm height slices on raw LiDAR points (median), between 0.3 and 1.8 m")


# ---------------------------------------------------------------- through-plane analysis (windows, doors, mirrors)

class RayIndex:
    """Subsampled raw points with their camera origins, for see-through tests."""

    def __init__(self, scene: dict, p: BetaParams):
        sub = slice(None, None, p.ray_subsample)
        P = scene["raw_points"][sub].astype(np.float64)
        D = scene["raw_ray"][sub].astype(np.float64)
        R = scene["raw_range"][sub].astype(np.float64)
        self.index = PointIndex(P.astype(np.float32), scene["raw_ray"][sub], scene["raw_range"][sub])
        self.origin = P - D * R[:, None]


def through_profile(face: WallFace, room_poly: Polygon, rays: RayIndex, index: PointIndex, p: BetaParams):
    """Occupancy of the wall plane on a 5 cm (along x height) grid.

    S = raw points ON the plane (+-3 cm); T = rays whose camera was INSIDE this room's polygon, that crossed the
    plane and ended at least `beyond_m` behind it. ("Inside the room", not just "on the room side of the
    plane": a camera in the next corridor is also on that side and would see past the wall's end.)
    Returns (S, T, along_edges, height_edges, (crossing along, crossing height, endpoints))."""
    a, inw = face.axis, face.inward
    s0, s1 = face.span
    b = p.profile_bin_m
    s_edges = np.arange(s0, s1 + b, b)
    h_edges = np.arange(0.0, 2.4 + b, b)
    col, scol = (0, 2) if a == 0 else (2, 0)                # across / along columns in 3D
    f_lo, f_hi = sorted((face.at(s0), face.at(s1)))
    # surface points on the (slightly tilted) wall line
    idx = index.box(f_lo - 0.03, f_hi + 0.03, s0, s1) if a == 0 else index.box(s0, s1, f_lo - 0.03, f_hi + 0.03)
    P = index.points[idx]
    P = P[np.abs(P[:, col] - face.at(P[:, scol])) < 0.03]
    S, _, _ = np.histogram2d(P[:, scol], P[:, 1] - face.floor_level, [s_edges, h_edges])
    # rays ending behind the wall
    far = (f_lo - inw * 4.0, f_hi - inw * p.beyond_m)
    idx = (rays.index.box(min(far), max(far), s0 - 4, s1 + 4) if a == 0 else
           rays.index.box(s0 - 4, s1 + 4, min(far), max(far)))
    E = rays.index.points[idx].astype(np.float64)
    O = rays.origin[idx]
    behind = (face.at(E[:, scol]) - E[:, col]) * inw >= p.beyond_m
    inside = behind & ((O[:, col] - face.at(O[:, scol])) * inw > 0.05)
    inside[inside] = contains_xy(room_poly, O[inside, 0], O[inside, 2])
    E, O = E[inside], O[inside]
    # crossing of the ray O->E with the line across = c + k (along - m)
    d = E - O
    lam = (face.at(O[:, scol]) - O[:, col]) / (d[:, col] - face.slope * d[:, scol])
    C = O + d * lam[:, None]
    cs = C[:, scol]
    ch = C[:, 1] - face.floor_level
    T, _, _ = np.histogram2d(cs, ch, [s_edges, h_edges])
    return S, T, s_edges, h_edges, (cs, ch, E)


@dataclass
class SeeThroughContext:
    """Everything the see-through tests need, built once per capture."""
    rays: RayIndex
    index: PointIndex
    tree: cKDTree                  # KD-tree on TSDF surface points
    tsdf_normals: np.ndarray
    room_polys: dict[str, Polygon]


def is_mirror(E: np.ndarray, face: WallFace, ctx: SeeThroughContext, p: BetaParams) -> tuple[bool, float]:
    """Is this see-through region a mirror? Three tests, all must pass:

    1. The phantom points do not lie inside another mapped room (if they do, we are looking through a real
       opening into a room we also walked through).
    2. Only surfaces FACING the plane are used, before and after reflection. Walls perpendicular to the plane
       that continue through a doorway reflect onto themselves and would otherwise fake a match.
    3. Reflected back across the plane, most phantoms land on a real surface of this side (within 4 cm)."""
    h = E[:, 1] - face.floor_level
    E = E[(h > 0.3) & (h < 2.0)]
    if len(E) < 20:
        return False, 0.0
    in_other = np.zeros(len(E), bool)
    for rid, poly in ctx.room_polys.items():
        if rid != face.room_id:
            in_other |= contains_xy(poly, E[:, 0], E[:, 2])
    if in_other.mean() >= 0.5:
        return False, 0.0
    col = 0 if face.axis == 0 else 2
    _, nn = ctx.tree.query(E, k=1)
    E = E[np.abs(ctx.tsdf_normals[nn, col]) >= 0.7]
    if len(E) < 20:
        return False, 0.0
    R = E.copy()
    R[:, col] = 2 * face.at(E[:, 2 if col == 0 else 0]) - R[:, col]
    d, nn = ctx.tree.query(R, k=1, distance_upper_bound=p.mirror_match_m)
    hit = np.isfinite(d)
    hit[hit] = np.abs(ctx.tsdf_normals[nn[hit], col]) >= 0.7
    frac = float(hit.mean())
    return frac >= p.mirror_min_frac, frac


def wall_openings(face: WallFace, ctx: SeeThroughContext, depth: float | None, ceiling_h: float | None,
                  p: BetaParams, next_id) -> tuple[list[Opening], list[dict]]:
    """Doors and windows in one wall from the through-plane profile.

    Only well-supported detections become openings (a phantom opening costs as much as a missed one in the
    detection score). Weaker ones (see-through without a wall frame, or framed regions with no returns at all,
    i.e. glass / dark screens) are returned as notes with kind "window_candidate" and the reason."""
    room_poly = ctx.room_polys[face.room_id].buffer(0.05, join_style="mitre")
    S, T, s_edges, h_edges, (cs, ch, E) = through_profile(face, room_poly, ctx.rays, ctx.index, p)
    openings, notes = [], []
    lab, n = ndi.label((T >= 2) & (T >= 3 * S), np.ones((3, 3)))
    for k in range(1, n + 1):
        ii, jj = np.nonzero(lab == k)
        box_ = (s_edges[ii.min()], s_edges[ii.max() + 1], h_edges[jj.min()], h_edges[jj.max() + 1])
        s_lo, s_hi, h_lo, h_hi = box_
        sel = (cs >= s_lo) & (cs <= s_hi) & (ch >= h_lo) & (ch <= h_hi)
        reaches_floor = h_lo <= 0.15
        mirror, frac = (False, 0.0) if reaches_floor else is_mirror(E[sel], face, ctx, p)
        if mirror:
            if s_hi - s_lo >= p.mirror_min_w_m:
                notes.append(_note("mirror", face, box_, f"reflected see-through points match real surfaces "
                                   f"({frac:.0%}); the wall is solid here"))
            continue
        kind, reason = _classify(S, ii, jj, box_, reaches_floor, p)
        if kind is None:
            if reason:
                notes.append(_note("window_candidate", face, box_, reason))
            continue
        conf = float(np.clip(0.5 + 0.1 * np.log10(max(sel.sum(), 1)), 0.5, 0.9))
        ev = (f"{int(sel.sum())} LiDAR rays from inside the room crossed the wall plane and hit surfaces >= "
              f"{p.beyond_m*100:.0f} cm behind it; {reason}"
              + ("" if reaches_floor else f"; mirror test: reflected-point match {frac:.0%}"))
        openings.append(_make_opening(face, kind, (s_lo + s_hi) / 2, s_hi - s_lo, (h_lo, h_hi), depth, ctx.index,
                                      ceiling_h, conf, ev, p, next_id))
    notes += _glass_candidates(face, S, T, s_edges, h_edges, p)
    return openings, notes


def _note(kind: str, face: WallFace, box_, reason: str) -> dict:
    s_lo, s_hi, h_lo, h_hi = (float(x) for x in box_)
    return dict(kind=kind, wall_id=face.wall_id, span=[s_lo, s_hi], height=[h_lo, h_hi], reason=reason)


def _classify(S, ii, jj, box_, reaches_floor: bool, p: BetaParams) -> tuple[str | None, str]:
    """door / window / None for one see-through region, with the reason."""
    s_lo, s_hi, h_lo, h_hi = box_
    w, hgt = s_hi - s_lo, h_hi - h_lo
    b = p.profile_bin_m
    if reaches_floor:
        if h_hi >= p.door_min_open_h_m and w >= p.door_min_m:
            return "door", "open from the floor to above 1.8 m"
        return None, ""
    if not (h_lo >= p.window_sill_min_m and h_hi <= p.window_top_max_m + b and w >= p.window_min_w_m
            and hgt >= p.window_min_h_m):
        return None, ""
    c0, c1, r0, r1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
    below = S[c0:c1, int(0.1 / b):max(r0 - 1, int(0.1 / b) + 1)] > 0
    under_sill = float(below.any(axis=1).mean())
    sides = [S[c, r0:r1] > 0 for c in (c0 - 1, c1) if 0 <= c < S.shape[0]]
    side_frac = float(np.mean([x.mean() for x in sides])) if len(sides) == 2 else 0.0
    if under_sill >= 0.6 and side_frac >= 0.5:
        return "window", f"framed: wall below the sill in {under_sill:.0%} of columns, wall on both sides"
    return None, (f"see-through region not framed by wall (wall under sill {under_sill:.0%}, sides "
                  f"{side_frac:.0%}): could be a window or a gap over furniture")


def _glass_candidates(face, S, T, s_edges, h_edges, p: BetaParams) -> list[dict]:
    """Framed, window-sized regions with no returns at all: glass, a dark screen (TV) or a specular panel.

    Reported as candidates only: a TV and a window look the same to this test."""
    band = (h_edges[:-1] >= p.window_sill_min_m) & (h_edges[1:] <= p.window_top_max_m)
    lab, n = ndi.label((S == 0) & (T == 0) & band[None, :])
    out = []
    for k in range(1, n + 1):
        m = lab == k
        ii, jj = np.nonzero(m)
        if ii.min() == 0 or ii.max() == len(s_edges) - 2:
            continue                                     # touches the wall end: not framed
        ring = ndi.binary_dilation(m, np.ones((3, 3))) & ~m
        box_ = (s_edges[ii.min()], s_edges[ii.max() + 1], h_edges[jj.min()], h_edges[jj.max() + 1])
        if box_[1] - box_[0] < p.window_min_w_m or box_[3] - box_[2] < p.window_min_h_m or (S[ring] > 0).mean() < 0.7:
            continue
        out.append(_note("window_candidate", face, box_, "no high-confidence LiDAR returns inside a region framed "
                         "by wall returns: glass, a dark screen (TV) or a specular panel"))
    return out


def _make_opening(face: WallFace, kind: str, s_mid: float, w: float, h_band, depth, index, ceiling_h, conf,
                  evidence, p: BetaParams, next_id) -> Opening:
    t_rng = slab(face, depth if depth is not None else 0.25 if kind == "window" else None, s_mid)
    lo_h = max(h_band[0] + 0.05, p.jamb_h_lo_m) if kind == "window" else p.jamb_h_lo_m
    hi_h = min(h_band[1] - 0.05, p.jamb_h_hi_m) if kind == "window" else p.jamb_h_hi_m
    gap = measure_gap(index, face.axis, t_rng, s_mid, w / 2, (lo_h, hi_h), face.floor_level, p)
    if gap is not None and not (0.5 * w <= gap.width <= 1.5 * w + 0.1):
        gap = None                                       # the gap found is not this opening
    width = width_measurement(gap, w, f"{kind} in wall {face.wall_id}")
    center_s = (gap.left + gap.right) / 2 if gap else s_mid
    center = (face.at(center_s), center_s) if face.axis == 0 else (center_s, face.at(center_s))
    sill = height = None
    if kind == "window":
        sill = Measurement.from_sigma(float(h_band[0]), p.profile_bin_m / 2, "m",
                                      "lowest see-through bin of the wall-plane profile (5 cm bins)",
                                      status="inferred")
        height = Measurement.from_sigma(float(h_band[1] - h_band[0]), p.profile_bin_m / np.sqrt(2), "m",
                                        "extent of the see-through region in the wall-plane profile (5 cm bins)",
                                        status="inferred")
    elif gap is not None:
        height = head_height(index, face.axis, t_rng, gap, face.floor_level, ceiling_h, p)
    conf = conf if gap is not None else min(conf, 0.5)       # width not measured on jambs: less trustworthy
    return Opening(next_id(), kind, [face.wall_id], [face.room_id], (float(center[0]), float(center[1])),
                   width, height, sill, round(conf, 2), evidence)
