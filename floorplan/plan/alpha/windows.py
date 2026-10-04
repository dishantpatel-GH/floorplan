"""Windows on exterior walls, and the two ways glass and mirrors fool a LiDAR.

We only search walls with no room behind them (exterior walls): interior walls rarely carry windows, and limiting
the search halves the chance of a phantom detection (which the Part 2 gate scores as a miss).

For each exterior wall we build a 5 cm raster on the wall plane (along x height, 0.3-2.2 m):
  * SURFACE: raw points ON the plane (+-3 cm): the wall itself was seen there;
  * SEE-THROUGH: raw points far BEHIND the plane whose viewing ray crossed the plane there: the LiDAR saw through
    the wall, so there is an opening (clear glass, an open window) at that spot;
  * NO-RETURN: bins enclosed by seen wall on all sides that have no surface, no see-through, and nothing in front
    of them that could have hidden them (furniture). Dark or clean glass often returns nothing, or only low-
    confidence depth that we drop (D-006), so a window can look like a hole. This is weaker evidence (a black TV
    looks the same), so it gets a lower confidence.

Mirrors: a mirror makes the LiDAR see a REFLECTED copy of the room behind the wall plane. That looks exactly like
see-through, so every see-through region is tested: reflect its behind-the-wall points back across the plane; if
most land within 3 cm of real points in the room, the region is a mirror, not a window. It is reported as a warning,
not an opening.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from floorplan.model import Measurement, Opening, Wall
from floorplan.plan.alpha.evidence import count_image
from floorplan.plan.alpha.params import AlphaParams
from floorplan.plan.alpha.walls import wall_axis


@dataclass
class WindowContext:
    P: np.ndarray            # raw points (subsampled), aligned frame
    cam: np.ndarray          # camera centre per point
    floor_y: float

    def mirror_fraction(self, Q: np.ndarray, ax: int, off: float, inward: float, match_m: float) -> float:
        """Share of reflected points Q that land within mirror_match of a real point in front of the wall.
        The KD-tree is built only on real points in Q's bounding box (fast, and enough for a 3 cm test)."""
        if len(Q) == 0:
            return 0.0
        lo, hi = Q.min(0) - 0.05, Q.max(0) + 0.05
        m = np.all((self.P >= lo) & (self.P <= hi), axis=1) & (inward * (self.P[:, ax] - off) > 0.02)
        if m.sum() == 0:
            return 0.0
        return float(np.mean(cKDTree(self.P[m]).query(Q, k=1)[0] < match_m))


def window_context(scene: dict, floor_y: float, stride: int = 4) -> WindowContext:
    """Every `stride`-th raw point (deterministic): the raw cloud has a point every 5 mm, so 1 in 4 still puts
    ~25 points in every fully seen 5 cm bin of the wall-plane raster (sparser sampling turns thinly seen wall
    into fake "no-return" holes). float32 is ample here."""
    P = scene["raw_points"][::stride].astype(np.float32)
    cam = P - scene["raw_range"][::stride, None].astype(np.float32) * scene["raw_ray"][::stride].astype(np.float32)
    return WindowContext(P, cam, floor_y)


def _wall_frame(w: Wall):
    fam, off, a, b, inward = wall_axis(w)
    ax, al = (0, 2) if fam == "x" else (2, 0)
    return ax, al, off, a, b, inward


def _rasters(ctx: WindowContext, w: Wall, p: AlphaParams):
    ax, al, off, a, b, n = _wall_frame(w)
    bm = p.window_bin_m
    na, nh = int(np.ceil((b - a) / bm)), int(np.ceil((p.window_hi_m - p.window_lo_m) / bm))
    if na < 3 or nh < 3:
        return None
    P, C = ctx.P, ctx.cam
    d = n * (P[:, ax] - off)                       # + = in the room, - = behind the wall
    dc = n * (C[:, ax] - off)
    h = P[:, 1] - ctx.floor_y

    def bins(al_v, h_v):
        i = np.floor((al_v - a) / bm).astype(int)
        j = np.floor((h_v - p.window_lo_m) / bm).astype(int)
        ok = (i >= 0) & (i < na) & (j >= 0) & (j < nh)
        return count_image(i[ok], j[ok], (na, nh))

    on = np.abs(d) < 0.03
    surface = bins(P[on, al], h[on])
    behind = (d < -p.see_through_min_m) & (dc > 0.1)
    t = dc[behind] / (dc[behind] - d[behind])     # fraction of the ray where it crosses the plane
    X = C[behind] + t[:, None] * (P[behind] - C[behind])
    through = bins(X[:, al], X[:, 1] - ctx.floor_y)
    front = (d > 0.05) & (d < 1.5) & (dc > d)       # anything between the camera and the wall can occlude it
    occluder = bins(P[front, al], h[front])
    return surface, through, occluder, P[behind], (ax, al, off, a, n)


def _components(mask: np.ndarray, p: AlphaParams):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    for k in range(1, n + 1):
        ii, jj = np.nonzero(lab == k)
        w = (ii.max() - ii.min() + 1) * p.window_bin_m
        h = (jj.max() - jj.min() + 1) * p.window_bin_m
        if w >= p.window_min_w_m and h >= p.window_min_h_m:
            yield lab == k, ii.min(), ii.max() + 1, jj.min(), jj.max() + 1


def _enclosed_holes(surface: np.ndarray, through: np.ndarray, occluder: np.ndarray) -> np.ndarray:
    """Empty bins not connected to the raster border = holes enclosed by observed wall, and not hidden."""
    empty = (surface == 0) & (through == 0)
    lab, _ = ndimage.label(empty)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    holes = empty & ~np.isin(lab, border)
    return holes & (ndimage.binary_dilation(occluder > 0, iterations=1) == 0)


def find_windows(ctx: WindowContext, w: Wall, p: AlphaParams, next_id) -> tuple[list[Opening], list[str]]:
    r = _rasters(ctx, w, p)
    if r is None:
        return [], []
    surface, through, occluder, behind_pts, (ax, al, off, a, n) = r
    openings, warnings = [], []
    see = (through >= p.see_through_min_hits) & (through > surface)
    for mask, i0, i1, j0, j1 in _components(see, p):
        lo_al, hi_al = a + i0 * p.window_bin_m, a + i1 * p.window_bin_m
        lo_h, hi_h = p.window_lo_m + j0 * p.window_bin_m, p.window_lo_m + j1 * p.window_bin_m
        sel = (behind_pts[:, al] > lo_al - 0.5) & (behind_pts[:, al] < hi_al + 0.5)
        Q = behind_pts[sel].copy()
        Q[:, ax] = 2 * off - Q[:, ax]                       # reflect across the wall plane
        matched = ctx.mirror_fraction(Q, ax, off, n, p.mirror_match_m)
        if matched >= p.mirror_frac:
            warnings.append(f"{w.id}: mirror suspected at along {lo_al:.2f}-{hi_al:.2f} m, h {lo_h:.2f}-{hi_h:.2f} m "
                            f"({100 * matched:.0f}% of see-through points mirror real geometry); not a window")
            continue
        openings.append(_window(w, ax, off, lo_al, hi_al, lo_h, hi_h, 0.75, p, next_id(),
                                f"LiDAR saw through wall plane ({int(through[mask].sum())} rays); "
                                f"mirror test: {100 * matched:.0f}% reflected matches"))
    for mask, i0, i1, j0, j1 in _components(_enclosed_holes(surface, through, occluder), p):
        if (i1 - i0) * (j1 - j0) * p.window_bin_m ** 2 < p.glass_min_area_m2:
            continue
        lo_al, hi_al = a + i0 * p.window_bin_m, a + i1 * p.window_bin_m
        lo_h, hi_h = p.window_lo_m + j0 * p.window_bin_m, p.window_lo_m + j1 * p.window_bin_m
        openings.append(_window(w, ax, off, lo_al, hi_al, lo_h, hi_h, 0.4, p, next_id(),
                                "no LiDAR return inside an observed, unoccluded wall: glass suspected "
                                "(could also be a dark screen)"))
    return openings, warnings


def _window(w: Wall, ax: int, off: float, lo_al: float, hi_al: float, lo_h: float, hi_h: float, conf: float,
            p: AlphaParams, oid: str, evidence: str) -> Opening:
    sig = p.window_bin_m / np.sqrt(6)            # two edges, each uniform within one 5 cm bin
    mid = (lo_al + hi_al) / 2
    center = (off, mid) if ax == 0 else (mid, off)
    m = "5 cm wall-plane raster"
    kind = "door" if lo_h <= p.window_lo_m + 1e-9 and hi_h - lo_h > 1.7 else "window"
    sill = None if kind == "door" else Measurement.from_sigma(lo_h, sig, method=m)
    return Opening(oid, kind, [w.id], [w.room_id], (float(center[0]), float(center[1])),
                   Measurement.from_sigma(hi_al - lo_al, sig, method=m), Measurement.from_sigma(hi_h - lo_h, sig, method=m),
                   sill, conf, evidence)
