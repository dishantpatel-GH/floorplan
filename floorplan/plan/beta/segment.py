"""Step 2: split the interior free space into rooms (Bormann et al. 2016, distance-transform segmentation).

Idea in one line: rooms are wide, doors are narrow. The distance transform (DT) gives, for every free cell, the
distance to the nearest obstacle; it is large in the middle of a room and small in a doorway. So:

  1. Seeds: every DT peak that stands at least `seed_h_m` above the saddle linking it to a higher peak
     (h-maxima). This deliberately over-segments: an L-shaped room gets two seeds.
  2. Watershed on -DT grows the seeds until they meet. Basins meet at the narrowest point between seeds, i.e. at
     the doorway when there is one.
  3. Merge: two neighbouring regions are one room when the place where they meet is wider than a door
     (2 x DT at the saddle > door_max_m): an L-shaped room, an open-plan kitchen, a wide passage. Anything
     narrower separates them. Whether wall evidence flanks the gap on both sides (jambs) does not decide the
     split; it only sets how confident we are that the gap is a real door (see openings.py).
  4. Regions too small to be a room are merged into their widest neighbour if that connection is at least
     door_min_m wide (a piece of a room cut off by a wide-ish opening), otherwise dropped (slivers of free space
     reached through a narrow gap, e.g. rays through a window).

Over-segment then merge is the robust direction: wrong merges are visible (a missing door), wrong seeds are not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from floorplan.plan.beta.params import BetaParams


@dataclass
class Neck:
    """Where two regions touch: the widest point of their shared boundary (the saddle of the DT)."""
    width_m: float
    rc: tuple[int, int]          # raster cell of the saddle
    axis: int                    # the cut lies on a line of constant u (0) or constant v (1), like Line.axis
    jambs: bool

    def separates(self, p: BetaParams) -> bool:
        return self.width_m <= p.door_max_m


def distance_map(free: np.ndarray, p: BetaParams) -> np.ndarray:
    """Distance (m) from each free cell to the nearest non-free cell, corrected for the wall dilation."""
    return ndi.distance_transform_edt(free) * p.cell_m + np.where(free, p.wall_dilate_m, 0.0)


def oversegment(free: np.ndarray, dt: np.ndarray, p: BetaParams) -> np.ndarray:
    smooth = ndi.gaussian_filter(dt, 1.0) * free
    seeds = h_maxima(smooth, p.seed_h_m)
    markers, _ = ndi.label(seeds)
    return watershed(-smooth, markers, mask=free)


def _boundary_pairs(labels: np.ndarray):
    """All 4-neighbour cell pairs with different non-zero labels: (label_a, label_b, cell_a, cell_b)."""
    out = []
    for da, db in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                   ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        a, b = labels[da], labels[db]
        m = (a != b) & (a > 0) & (b > 0)
        r, c = np.nonzero(m)
        ra, ca = r, c
        rb, cb = (r + 1, c) if da[0].stop == -1 else (r, c + 1)
        out.append((a[m], b[m], np.stack([ra, ca], 1), np.stack([rb, cb], 1)))
    la, lb, ca, cb = (np.concatenate(x) for x in zip(*out))
    return la, lb, ca, cb


def _has_jambs(barrier: np.ndarray, rc: tuple[int, int], radius_cells: float) -> bool:
    """Wall evidence on two opposite sides of the saddle (a doorway is a gap between two wall ends)."""
    r0, c0 = rc
    R = int(np.ceil(radius_cells))
    win = barrier[max(r0 - R, 0):r0 + R + 1, max(c0 - R, 0):c0 + R + 1]
    rr, cc = np.nonzero(win)
    if len(rr) == 0:
        return False
    dy = rr + max(r0 - R, 0) - r0
    dx = cc + max(c0 - R, 0) - c0
    d = np.hypot(dx, dy)
    keep = (d > 0) & (d <= radius_cells)
    ang = np.arctan2(dy[keep], dx[keep])
    hist = np.bincount(((ang + np.pi) / (2 * np.pi) * 24).astype(int) % 24, minlength=24) > 0
    return bool((hist & np.roll(hist, 12)).any())          # occupied 15-degree sectors 180 degrees apart


def _neck(dt, barrier, cells_a, cells_b, p: BetaParams) -> Neck:
    vals = np.minimum(dt[cells_a[:, 0], cells_a[:, 1]], dt[cells_b[:, 0], cells_b[:, 1]])
    i = int(np.argmax(vals))
    rc = (int(cells_a[i, 0]), int(cells_a[i, 1]))
    half = float(vals[i])
    jambs = _has_jambs(barrier, rc, (half + p.jamb_search_m) / p.cell_m)
    # the cut is the shared boundary; it runs ACROSS the passage. Its long direction tells which line it lies on.
    spread = np.ptp(cells_a, axis=0)                        # (rows, cols) extent of the boundary cells
    axis = 1 if spread[1] >= spread[0] else 0               # spread along columns (u) -> line of constant v
    return Neck(2 * half, rc, axis, jambs)


def region_necks(labels, dt, barrier, p: BetaParams) -> dict[tuple[int, int], Neck]:
    la, lb, ca, cb = _boundary_pairs(labels)
    lo, hi = np.minimum(la, lb), np.maximum(la, lb)
    swap = la > lb
    A = np.where(swap[:, None], cb, ca)
    B = np.where(swap[:, None], ca, cb)
    out = {}
    key = lo.astype(np.int64) * (labels.max() + 1) + hi
    for k in np.unique(key):
        m = key == k
        out[(int(lo[m][0]), int(hi[m][0]))] = _neck(dt, barrier, A[m], B[m], p)
    return out


def merge_regions(labels: np.ndarray, dt: np.ndarray, barrier: np.ndarray, p: BetaParams):
    """Greedy merging until every remaining boundary is door-sized and every region is room-sized."""
    labels = labels.copy()
    while True:
        necks = region_necks(labels, dt, barrier, p)
        areas = np.bincount(labels.ravel()) * p.cell_m ** 2
        keep, drop = _next_merge(necks, areas, p)
        if drop is None:
            return labels, necks
        labels[labels == drop] = keep


def _next_merge(necks: dict, areas: np.ndarray, p: BetaParams) -> tuple[int | None, int | None]:
    """(keep, drop): widest wider-than-door connection first; then the smallest undersized region.

    keep = 0 means the small region is dropped (it only hangs on through a narrow gap)."""
    wide = [(n.width_m, k) for k, n in necks.items() if not n.separates(p)]
    if wide:
        return max(wide)[1]
    small = sorted((areas[l], l) for k in necks for l in k if areas[l] < p.min_room_m2)
    if not small:
        return None, None
    l = small[0][1]
    width, (a, b) = max((n.width_m, k) for k, n in necks.items() if l in k)
    other = b if a == l else a
    return (other if width >= p.door_min_m else 0), l


def segment_rooms(free: np.ndarray, barrier: np.ndarray, p: BetaParams, cut: np.ndarray | None = None,
                  visited: np.ndarray | None = None):
    """Return (labels 0..N with 0 = not free, door necks between rooms, DT map).

    v2 (B-1): `cut` holds the long wall lines of lines.py. They act as obstacles for the distance transform and
    the watershed only, so a partition seen below the 1.0 m band still narrows the free space to its doorway and
    the neck rule separates the rooms. The cut cells are then given to the nearest room, so no free space is
    lost from the plan (unlike lowering the band, J-I-5)."""
    seg_free = free & ~cut if cut is not None else free
    dt = distance_map(seg_free, p)
    labels = _fill_cut(oversegment(seg_free, dt, p), free & ~seg_free)
    labels, necks = merge_regions(labels, dt, barrier, p)
    if cut is not None and visited is not None:
        labels, necks = _merge_unentered_across_cut(labels, necks, cut & free, visited, dt, barrier, p)
    labels = _drop_small_isolated(labels, p)
    labels, mapping = _relabel(labels)
    necks = {(mapping[a], mapping[b]): n for (a, b), n in necks.items() if a in mapping and b in mapping}
    return labels, necks, dt


def _merge_unentered_across_cut(labels: np.ndarray, necks: dict, cut: np.ndarray, visited: np.ndarray,
                                dt: np.ndarray, barrier: np.ndarray, p: BetaParams):
    """A cut line may separate two rooms only if the operator entered both. A region the camera never entered
    that a cut split off (the strip behind a wardrobe line, a niche) goes back to the entered neighbour it shares
    the most cut cells with. Without this the strip becomes an "unentered room" and D-B10 drops it: floor area
    lost from the plan (single_room footprint 19.4 -> 17.6 m2 before this rule, plan_v2_ablation.md B-1b)."""
    labels = labels.copy()
    near_cut = ndi.binary_dilation(cut, iterations=2)
    for _ in range(50):
        entered = set(np.unique(labels[visited & (labels > 0)]).tolist())
        best = None
        for (a, b) in sorted(necks):
            for lost, keep in ((a, b), (b, a)):
                if lost in entered or keep not in entered:
                    continue
                shared = _shared_cut_cells(labels, lost, keep, near_cut)
                if shared > 0 and (best is None or shared > best[0]):
                    best = (shared, lost, keep)
        if best is None:
            break
        labels[labels == best[1]] = best[2]
        necks = region_necks(labels, dt, barrier, p)
    return labels, necks


def _shared_cut_cells(labels: np.ndarray, a: int, b: int, near_cut: np.ndarray) -> int:
    """Number of 4-neighbour boundary cells between regions a and b that lie on a cut line."""
    ma, mb = labels == a, labels == b
    touch = (ma & (np.roll(mb, 1, 0) | np.roll(mb, -1, 0) | np.roll(mb, 1, 1) | np.roll(mb, -1, 1)))
    return int((touch & near_cut).sum())


def _fill_cut(labels: np.ndarray, cut_free: np.ndarray) -> np.ndarray:
    """Give each free cell that only the cut removed the label of its nearest labelled cell (exact EDT indices:
    deterministic, no scan-order tie-breaking)."""
    if not cut_free.any() or not (labels > 0).any():
        return labels
    _, (ri, ci) = ndi.distance_transform_edt(labels == 0, return_indices=True)
    out = labels.copy()
    out[cut_free] = labels[ri[cut_free], ci[cut_free]]
    return out


def _drop_small_isolated(labels: np.ndarray, p: BetaParams) -> np.ndarray:
    areas = np.bincount(labels.ravel()) * p.cell_m ** 2
    small = np.nonzero(areas < p.min_room_m2)[0]
    out = labels.copy()
    out[np.isin(out, small[small > 0])] = 0
    return out


def _relabel(labels: np.ndarray):
    """Renumber rooms 1..N in descending area order (stable ids within a run)."""
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    mapping = {int(old): i + 1 for i, old in enumerate(order)}
    lut = np.zeros(labels.max() + 1, np.int32)
    for old, new in mapping.items():
        lut[old] = new
    return lut[labels], mapping
