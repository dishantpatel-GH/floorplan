"""Damage detection on rectified surface textures (orthophotos): water stains and cracks.

Approach (chosen by evidence, see docs/modules/damage.md section 3): classical appearance anomaly on the per-surface
orthophoto, verified across views.

  water stain = a blob that is DARKER and YELLOWER than the surface's own robust colour model.
      * The model is a smooth background (masked Gaussian, iteratively re-weighted so the stain itself does not leak
        into it), so slow lighting gradients across a wall are not anomalies.
      * Requiring a yellow/brown chroma shift rejects shadows, which darken without changing chromaticity
        (the classic shadow-invariance argument), and grey dirt.
  crack = a thin, long, dark ridge (multi-scale Sato ridge filter on lightness) that is NOT a straight axis-aligned
      line (those are panel edges, skirting, tile grout, door frames).
  every candidate must
      * be seen by >= min_views views and appear in >= min_support of them (multi-view consistency: glare and
        reflections move with the camera, paint does not);
      * pass shape rules (rectangles = frames, switches, cabinets; giant regions = material changes).

All outputs are metric: an ortho pixel is `res` metres, so areas and extents are pixel counts times constants, with
95% intervals from boundary localisation, threshold sensitivity, registration and the tier's scale uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from skimage.filters import apply_hysteresis_threshold, sato
from skimage.morphology import skeletonize

from floorplan.damage.project import Ortho


@dataclass
class DetectParams:
    min_views: int = 3                # pixel must be seen by >= 3 views to be analysed
    bg_sigma_m: float = 0.20          # background colour model scale (larger than the stains we want to keep)
    stain_smooth_m: float = 0.01
    stain_z_hi: float = 4.0           # seed: YELLOWER (b*) than the background by > 4 robust sigmas ...
    stain_z_lo: float = 2.0           # ... grown down to 2 sigmas (hysteresis)
    stain_seed_db: float = 3.0        # seed also needs >= 3 b* units (visible to a person)
    stain_min_dL: float = 1.0         # a stain must also be darker on average (>= 1 L*): yellow AND darker
    stain_min_db: float = 2.0         # mean yellow shift (b*) of a stain, L*a*b* units ...
    stain_db_per_dL: float = 0.25     # ... and at least 0.25 x the darkening: brown/yellow, not a grey/black object
    stain_min_area_m2: float = 0.004  # ~6 x 6 cm; smaller blobs were fridge magnets and texture in the clean sample
    stain_min_short_m: float = 0.04   # a stain is a blob: its minimum-area rectangle is at least 4 cm on the short side
    stain_max_aspect: float = 5.0     # ... and not a strip (door frames and corner shading bleed in as thin strips)
    stain_max_area_m2: float = 3.0
    stain_max_frac: float = 0.5       # larger than half the surface = a different material, not a stain
    rect_fill: float = 0.88           # area / min-area-rectangle above this + few polygon vertices = an object
    crack_sigmas_px: tuple = (1.0, 1.5, 2.0)
    crack_z_hi: float = 6.0
    crack_z_lo: float = 3.5
    crack_abs_hi: float = 3.0         # Sato response ~0.35-0.5 x line contrast: 3.0 ~ a line 7-8 L* units darker
    crack_abs_lo: float = 1.5
    crack_min_contrast: float = 1.5   # both sides of the line must be >= 1.5 L* brighter (a step edge has one side)
    crack_min_len_m: float = 0.08
    crack_axis_tol_deg: float = 8.0
    crack_bar_tol_m: float = 0.003   # D-024: skeleton pixels within 3 mm of ONE straight line ...
    crack_bar_frac: float = 0.70     # ... >= 70% of them, and that line within crack_axis_tol_deg of H/V = a bar
                                     # (handle, edge, grout line), even with hooks at its ends; jagged cracks fail it
    crack_straight: float = 0.95      # endpoint distance / length above this + axis-aligned = an edge, not a crack
    crack_axis_run_m: float = 0.05    # skeleton pixels on >= 5 cm horizontal/vertical straight runs ...
    crack_max_axis_frac: float = 0.4  # ... above 40% of the length = outline of a frame/door/furniture
    crack_max_width_m: float = 0.02   # wider dark lines are gaps, shadows lines or edges, not cracks
    crack_max_density: float = 1.0    # m of ridge per m2 above this = textured surface (tiles): crack test unreliable
    min_support: float = 0.7         # 0.6 let through a clean-sample crack FP at support 0.67 (damage.md I-7)
    min_support_views: int = 4        # >= 4 views must agree (2 for the photo tier, which has 2-8 photos per room)
    max_spread_L: float = 8.0         # median cross-view disagreement around a region (L*): above = glass/mirror
    halo_m: float = 0.025             # within 2.5 cm of an UNSEEN part of the surface = occlusion-boundary halo
    max_halo_frac: float = 0.5
    support_ratio: float = 0.4        # a view supports a region if its contrast is >= 40% of the consensus contrast
    # --- v2: crack tracing at fine resolution (damage.md v2, I-6) ---
    crack_link_gap_m: float = 0.05    # bridge gaps up to 5 cm between fragments that point at each other: a
                                      # crack crossing a cut-away step edge loses ~2 cm + its end shortening (v2 I-13)
    crack_link_near_m: float = 0.01   # ... and up to 1 cm in any direction (junctions, where Sato dips)
    crack_link_angle_deg: float = 35.0
    crack_axis_cut_m: float = 0.12   # v2: only straight H/V runs >= 12 cm are cut before grouping (v1: 5 cm cut
                                      # off near-horizontal crack BRANCHES; frames/skirting/door edges are longer)
    crack_min_straight: float = 0.70  # v2: span/length below this = tortuous: object outlines, cables, magnets.
                                      # Clean-wall false cracks 0.49-0.69 vs injected cracks 0.69-0.95 (v2 I-17)
    crack_axis_run_fine_m: float = 0.08  # v2: outline rule on the fine skeleton counts runs >= 8 cm (a hand-drawn
                                      # jagged line has 1-2.5 cm segments; frame/door shadow lines run >> 10 cm)
    crack_link_min_m: float = 0.01    # v2: only fragments with >= 1 cm of line are bridged to (not specks/corners)
    crack_pix_contrast: float = 1.5   # v2: a ridge PIXEL must be >= 1.5 L* darker than BOTH sides 1 cm away
    crack_side_m: float = 0.01
    stain_local_sigma_m: float = 0.0  # v2: 0.05 = second, LOCAL colour model for seeds (stain on paper vs the paper);
                                      # OFF by default: it produced 4 clean-sample false stains (v2 I-18)
    crack_seed_min_len_m: float = 0.04  # coarse fragments shorter than this are not worth refining
    crack_fine_res: float = 0.0025    # 2.5 mm/px orthophoto around each candidate (0 = no refinement)
    crack_fine_sigmas_px: tuple = (1.0, 1.5, 2.0, 3.0)
    crack_fine_z_lo: float = 4.0      # fine growth threshold: 4 robust sigmas of the ROI's ridge noise ...
    crack_fine_abs_lo: float = 1.0    # ... and at least this (Sato units, ~ a line 2.5-3 L* darker at 2.5 mm/px)
    crack_spur_m: float = 0.01        # skeleton spurs shorter than 1 cm are thickness artefacts, not branches
    # --- v2: confidence (logistic terms; centres/widths in physical units, see confidence()) ---
    conf_crack_contrast_L: float = 4.0
    conf_crack_contrast_w: float = 1.5
    conf_crack_len_m: float = 0.10
    conf_crack_len_w: float = 0.025
    conf_stain_db: float = 3.0
    conf_stain_db_w: float = 1.0
    conf_views_scale: float = 3.0     # agreement term 1 - exp(-n_agreeing / 3)
    min_confidence: float = 0.10      # v2: chosen from the precision/recall sweep (damage.md v2 D-dmg-11)


@dataclass
class Region:
    surface_id: str
    cls: str                           # water_stain | crack
    pixels: np.ndarray                 # (H,W) bool on the ortho (not serialised)
    score: float
    support: float
    n_views: int
    stats: dict = field(default_factory=dict)
    rejected: str = ""                 # empty = accepted; otherwise the rule that rejected it


def _lab(bgr01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.nan_to_num(bgr01).astype(np.float32), cv2.COLOR_BGR2Lab)


def _masked_blur(x: np.ndarray, w: np.ndarray, sigma_px: float) -> np.ndarray:
    num = cv2.GaussianBlur(x * w, (0, 0), sigma_px)
    den = cv2.GaussianBlur(w, (0, 0), sigma_px)
    return num / np.maximum(den, 1e-6)


def _robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else 1.0


def background(lab: np.ndarray, valid: np.ndarray, sigma_px: float, iters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Smooth robust colour model of the surface; pixels that deviate (the damage) are down-weighted each pass."""
    w = valid.astype(np.float32)
    bg = np.stack([_masked_blur(lab[..., c], w, sigma_px) for c in range(3)], -1)
    for _ in range(iters):
        r = lab - bg
        sL = max(_robust_sigma(r[..., 0][valid]), 0.5)
        sb = max(_robust_sigma(r[..., 2][valid]), 0.5)
        w = (valid & (np.abs(r[..., 0]) < 2.5 * sL) & (np.abs(r[..., 2]) < 2.5 * sb)).astype(np.float32)
        bg = np.stack([_masked_blur(lab[..., c], w, sigma_px) for c in range(3)], -1)
    return bg, w > 0


def _view_support(o: Ortho, region: np.ndarray, ring: np.ndarray, consensus: float, ratio: float) -> tuple[float, int]:
    """Fraction of the views covering the region whose own lightness contrast (region vs ring) is at least `ratio`
    of the consensus contrast (both negative = darker)."""
    sup, n = 0, 0
    for k in range(o.stack.shape[0]):
        L = o.stack[k].astype(np.float32).mean(-1)
        a, b = L[region], L[ring]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 0.5 * region.sum() or b.size < 20:
            continue
        n += 1
        c = 100.0 * (np.median(a) - np.median(b))       # ~L* units
        sup += c <= ratio * consensus
    return (sup / n if n else 0.0), n


def _shape_is_object(region: np.ndarray, res: float, fill_thr: float) -> bool:
    """Rectangle test: picture frames, switches, sockets, cabinet panels have straight edges and fill their
    minimum-area rectangle; stains are irregular blobs (an ellipse fills only pi/4 = 0.79 of it)."""
    cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    fill = region.sum() / max(w * h, 1.0)
    approx = cv2.approxPolyDP(c, max(2.0, 0.01 / res), True)
    return fill > fill_thr and len(approx) <= 8


def _skeleton_length(sk: np.ndarray, res: float) -> float:
    """Length of a 1-px skeleton: orthogonal neighbour pairs count 1 px, diagonal pairs sqrt(2) px."""
    s = sk.astype(np.int32)
    orth = (s[:, 1:] & s[:, :-1]).sum() + (s[1:, :] & s[:-1, :]).sum()
    diag = (s[1:, 1:] & s[:-1, :-1]).sum() + (s[1:, :-1] & s[:-1, 1:]).sum()
    # a diagonal step creates one diagonal pair and, at corners, extra orthogonal pairs; this simple count is within
    # a few % of the true length for crack-like curves (checked on injected cracks)
    return float((orth + np.sqrt(2) * diag) * res) if (orth + diag) else float(res)


def _neighbours(sk: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    return cv2.filter2D(sk.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)


def _endpoint_dirs(sk: np.ndarray, back_px: float) -> list[tuple[int, int, np.ndarray]]:
    """Skeleton endpoints (row, col) and the unit direction in which the line LEAVES there (row, col order),
    estimated from the skeleton pixels within back_px of the endpoint (0-vector for isolated dots)."""
    nb = _neighbours(sk)
    ys, xs = np.nonzero(sk & (nb <= 1))
    if not len(ys):
        return []
    sy, sx = np.nonzero(sk)
    tree = cKDTree(np.c_[sy, sx])
    out = []
    for r, c in zip(ys, xs):
        idx = tree.query_ball_point([r, c], back_px)
        m = np.array([sy[idx].mean(), sx[idx].mean()]) if idx else np.array([r, c], float)
        d = np.array([r, c], float) - m
        nrm = np.linalg.norm(d)
        out.append((int(r), int(c), d / nrm if nrm > 0.5 else np.zeros(2)))
    return out


def link_fragments(mask: np.ndarray, res: float, gap_m: float, near_m: float,
                   angle_deg: float, min_m: float = 0.01) -> tuple[np.ndarray, np.ndarray, int]:
    """Gap bridging (v2): join ridge fragments that belong to one crack.

    Hysteresis breaks a crack wherever it gets faint, and the Sato filter dips at Y-junctions (the Hessian there is
    not line-like). v1 grouped fragments by a 1 cm dilation, which split an injected crack at its junction into two
    regions and lost the branch (damage.md v2, I-6). Here a fragment's END is linked to another fragment when
      * the other fragment is within `near_m` in any direction (junctions), or
      * within `gap_m` and inside a cone of +-angle_deg around the direction the end points to (collinear gaps).
    The cone keeps parallel texture lines (wood grain, brush marks) from being chained sideways. Bridges are drawn
    into the returned mask, so the skeleton and its length run continuously across the gap (a crack is physically
    continuous; the gap is a faint stretch). Returns (bridged mask, group labels on it, number of groups)."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    out = mask.copy()
    if n <= 1:
        return out, lab, n
    sk = skeletonize(mask)
    # fragments with less than min_m of skeleton are specks (JPEG blocks, tape corners): never bridged to or from
    sk_len = ndi.sum(sk, lab, np.arange(n + 1)) * res
    big = sk_len >= min_m
    big[0] = False
    ys, xs = np.nonzero(mask & big[lab])
    labs = lab[ys, xs]
    tree = cKDTree(np.c_[ys, xs])
    parent = np.arange(n + 1)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    gap_px, near_px, cmin = gap_m / res, max(1.5, near_m / res), np.cos(np.radians(angle_deg))
    canvas = np.zeros(mask.shape, np.uint8)
    for r, c, d in _endpoint_dirs(sk, max(3.0, 0.015 / res)):
        a = lab[r, c]
        if a == 0 or not big[a]:
            continue
        best = None
        for j in tree.query_ball_point([r, c], gap_px):
            if labs[j] == a:
                continue
            v = np.array([ys[j] - r, xs[j] - c], float)
            dist = float(np.linalg.norm(v))
            if dist <= near_px or (d.any() and v @ d >= cmin * dist):
                if best is None or dist < best[0]:
                    best = (dist, j)
        if best is not None:
            j = best[1]
            parent[find(a)] = find(int(labs[j]))
            cv2.line(canvas, (int(c), int(r)), (int(xs[j]), int(ys[j])), 1, 1)
    out |= canvas.astype(bool)
    glab, _ = ndi.label(out, structure=np.ones((3, 3)))
    # groups = connected components of the bridged mask (bridges merged the linked fragments)
    return out, glab, int(glab.max())


def mst_length(sk: np.ndarray, res: float) -> float:
    """Length of a skeleton as the minimum spanning tree of its 8-connected pixel graph (v2).

    Summing all neighbour pairs (v1) double-counts staircase corners (a diagonal step also creates two orthogonal
    pairs); the MST keeps exactly one path through every pixel, so a straight diagonal line of n pixels measures
    (n-1) sqrt(2) px. Branches are included (a branched crack is repaired along every branch)."""
    ys, xs = np.nonzero(sk)
    n = len(ys)
    if n < 2:
        return float(res) if n else 0.0
    idx = -np.ones(sk.shape, np.int64)
    idx[ys, xs] = np.arange(n)
    rows, cols, w = [], [], []
    H, W = sk.shape
    for dy, dx, wt in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        j = np.full(n, -1)
        j[ok] = idx[y2[ok], x2[ok]]
        m = j >= 0
        rows.append(np.arange(n)[m]); cols.append(j[m]); w.append(np.full(m.sum(), wt))
    g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    return float(minimum_spanning_tree(g).sum() * res)


def prune_spurs(sk: np.ndarray, spur_px: int) -> np.ndarray:
    """Remove skeleton branches shorter than spur_px that end at a junction (skeletonising a 2-4 px wide band
    leaves such hairs at every bump; they would add a few % of fake length each)."""
    sk = sk.copy()
    nb = _neighbours(sk)
    for r, c in zip(*np.nonzero(sk & (nb == 1))):
        path, cur, prev = [(r, c)], (r, c), None
        while len(path) <= spur_px:
            y, x = cur
            nxt = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)
                   and 0 <= y + dy < sk.shape[0] and 0 <= x + dx < sk.shape[1] and sk[y + dy, x + dx]
                   and (y + dy, x + dx) != prev and (y + dy, x + dx) not in path]
            if len(nxt) != 1:
                if len(nxt) >= 2 and len(path) < spur_px:     # reached a junction: it was a spur
                    for q in path[:-1] if len(path) > 1 else path:
                        sk[q] = False
                break
            prev, cur = cur, nxt[0]
            if nb[cur] >= 3:                                   # junction pixel
                if len(path) < spur_px:
                    for q in path:
                        sk[q] = False
                break
            path.append(cur)
    return sk


def view_spread(o: Ortho) -> np.ndarray:
    """Per-pixel median absolute deviation across views of lightness (~L* units). Matte paint looks the same from
    every view (after the per-view gain); mirrors, glass and glossy tiles show different things from each view."""
    L = o.stack.astype(np.float32).mean(-1) * 100.0
    with np.errstate(all="ignore"):
        med = np.nanmedian(L, 0)
        return np.nan_to_num(np.nanmedian(np.abs(L - med), 0), nan=0.0)


def _local_spread(spread: np.ndarray, pix: np.ndarray) -> float:
    return float(np.median(spread[pix])) if pix.any() else 0.0


def detect_stains(o: Ortho, lab: np.ndarray, bg: np.ndarray, valid: np.ndarray, p: DetectParams,
                  spread: np.ndarray) -> list[Region]:
    res = o.res
    sm = max(p.stain_smooth_m / res, 0.5)
    r = lab - bg
    dL = cv2.GaussianBlur(np.where(valid, r[..., 0], 0).astype(np.float32), (0, 0), sm)
    db = cv2.GaussianBlur(np.where(valid, r[..., 2], 0).astype(np.float32), (0, 0), sm)
    sL = max(_robust_sigma(dL[valid]), 0.7)
    sb = max(_robust_sigma(db[valid]), 0.5)
    # Chroma-first segmentation (changed after the first injection test, see damage.md I-4): segmenting on darkness
    # merged stains with neighbouring shadows and missed pale stains on noisy surfaces; a yellow/brown shift is the
    # signature of a water stain (tannins/minerals carried to the surface) and shadows never seed it.
    z = db / sb
    hi = valid & (z > p.stain_z_hi) & (db > p.stain_seed_db) & (dL < 0)
    lo = valid & (z > p.stain_z_lo)
    # v2: the same test against a LOCAL colour model (5 cm). A stain painted on a bright, bluish sheet of paper is
    # neither darker nor much yellower than the WALL model, only than the paper around it (staged decals: 0/3
    # paper stains seeded before this). Local seeds must also be darker than their local surroundings.
    if p.stain_local_sigma_m:
        bgl, _ = background(lab, valid, p.stain_local_sigma_m / res)
        rl = lab - bgl
        dLl = cv2.GaussianBlur(np.where(valid, rl[..., 0], 0).astype(np.float32), (0, 0), sm)
        dbl = cv2.GaussianBlur(np.where(valid, rl[..., 2], 0).astype(np.float32), (0, 0), sm)
        sbl = max(_robust_sigma(dbl[valid]), 0.5)
        hi |= valid & (dbl / sbl > p.stain_z_hi) & (dbl > p.stain_seed_db) & (dLl < 0)
        lo |= valid & (dbl / sbl > p.stain_z_lo)
    lab_lo, n = ndi.label(lo)              # hysteresis: weak components that contain a strong seed
    keep_ids = np.unique(lab_lo[hi])
    keep_ids = keep_ids[keep_ids > 0]
    surf_area = valid.sum() * res * res
    out = []
    db_raw = cv2.GaussianBlur(np.where(valid, r[..., 2], 0).astype(np.float32), (0, 0), 0.7)
    for i in keep_ids:
        reg = _half_max_boundary(lab_lo == i, db_raw, valid, res)
        area = reg.sum() * res * res
        if area < p.stain_min_area_m2:
            continue
        ring = (ndi.binary_dilation(reg, iterations=max(2, int(0.03 / res))) & ~reg) & valid
        # v2: colour shift relative to the stain's IMMEDIATE surroundings (a 3 cm ring), not the 0.2 m background
        # model. A stain on a sheet of white paper taped to a coloured wall must be judged against the paper; the
        # smooth background there is a blend of paper and wall (damage.md v2, decals).
        if ring.sum() > 20:
            mdL = float(np.median(dL[reg]) - np.median(dL[ring]))
            mdb = float(np.median(db[reg]) - np.median(db[ring]))
        else:
            mdL, mdb = float(np.mean(dL[reg])), float(np.mean(db[reg]))
        sup, nv = _view_support(o, reg, ring, mdL, p.support_ratio) if ring.sum() > 20 else (0.0, 0)
        rg = Region(o.surface.id, "water_stain", reg, score=float(np.mean(z[reg])), support=sup, n_views=nv,
                    stats=dict(mean_dL=mdL, mean_db=mdb, local_db=mdb, bg_dL=float(np.mean(dL[reg])),
                               bg_db=float(np.mean(db[reg])), sigma_L=sL, sigma_b=sb, area_px=int(reg.sum())))
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.c_[np.nonzero(reg)[1], np.nonzero(reg)[0]].astype(np.float32))
        short, long_ = min(rw, rh) * res + res, max(rw, rh) * res + res
        if area > p.stain_max_area_m2 or area > p.stain_max_frac * surf_area:
            rg.rejected = "too large for a stain: material/colour change of the surface"
        elif mdb < max(p.stain_min_db, 1.5 * sb, p.stain_db_per_dL * abs(mdL)):
            rg.rejected = "darkening without a yellow/brown shift: shadow, grime or dark object"
        elif -mdL < p.stain_min_dL:
            rg.rejected = "yellower but not darker: lighting colour cast or a light wooden object"
        elif short < p.stain_min_short_m or long_ / short > p.stain_max_aspect:
            rg.rejected = f"thin strip ({short*100:.0f} x {long_*100:.0f} cm): frame/edge/corner shading, not a blob"
        elif _shape_is_object(reg, res, p.rect_fill):
            rg.rejected = "rectangular with straight edges: object (frame, switch, panel)"
        elif _halo_frac(reg ^ ndi.binary_erosion(reg), o, valid, p.halo_m) > p.max_halo_frac:
            rg.rejected = "hugs the edge of an occluded/unseen patch: occluder halo (shadow, blurred object edge)"
        elif nv < p.min_views:
            rg.rejected = f"seen in only {nv} views"
        elif sup < p.min_support or round(sup * nv) < p.min_support_views:
            rg.rejected = f"inconsistent across views (support {sup:.2f} of {nv}): glare/reflection"
        elif _local_spread(spread, reg | ring) > p.max_spread_L:
            rg.rejected = "views disagree around it (reflective/transparent surface: mirror, glass, gloss)"
        rg.stats.update(z_area_lo=_threshold_area(z, valid, reg, p.stain_z_lo * 1.4, res),
                        z_area_hi=_threshold_area(z, valid, reg, p.stain_z_lo * 0.7, res))
        out.append(rg)
    return out


def _half_max_boundary(core: np.ndarray, db_raw: np.ndarray, valid: np.ndarray, res: float) -> np.ndarray:
    """Re-draw the stain outline at HALF of its own median colour shift, on the lightly smoothed map.

    The detection masks are found on a 1 cm-smoothed map with a noise-relative threshold, which put the outline about
    1 cm outside the true edge (+2 cm width, +10-28% area in the first injection test). The half-maximum edge is the
    standard, contrast-independent definition of where a soft edge is (as FWHM for a blurred line)."""
    core = ndi.binary_fill_holes(core) & valid
    near = ndi.binary_dilation(core, iterations=max(2, int(0.02 / res))) & valid
    # v2: half-way between the stain's core and its immediate surroundings (local half-maximum), so a stain on paper
    # is outlined relative to the paper, not to the wall's background model
    ring = ndi.binary_dilation(near, iterations=max(2, int(0.02 / res))) & ~near & valid
    base = float(np.median(db_raw[ring])) if ring.sum() > 20 else 0.0
    half = base + 0.5 * (float(np.median(db_raw[core])) - base)
    lab, _ = ndi.label(near & (db_raw > half))
    ids = np.unique(lab[core & (lab > 0)])
    reg = np.isin(lab, ids[ids > 0])
    return ndi.binary_fill_holes(ndi.binary_opening(reg, iterations=1)) & valid if reg.any() else core


def _threshold_area(z: np.ndarray, valid: np.ndarray, reg: np.ndarray, thr: float, res: float) -> float:
    """Area of the component overlapping `reg` when the growth threshold is changed: the threshold sensitivity
    term of the area interval (a stain's edge is a gradient; where exactly it 'ends' is partly a convention)."""
    near = ndi.binary_dilation(reg, iterations=max(3, int(0.05 / res)))
    lab, _ = ndi.label(valid & near & (z > thr))
    ids = np.unique(lab[reg & (lab > 0)])
    m = np.isin(lab, ids[ids > 0])
    return float(ndi.binary_fill_holes(m).sum() * res * res)


def _halo_frac(pix: np.ndarray, o: Ortho, valid: np.ndarray, halo_m: float) -> float:
    """Fraction of `pix` within halo_m of a part of the surface that was NOT seen (occluded by furniture, outside
    every view). The surface's own outline does not count, so a stain at the base of a wall is not penalised.
    Occluder edges are blurred/misregistered in some views and leave dark rims; the clean sample showed them."""
    unseen = o.mask & ~valid
    if not unseen.any() or not pix.any():
        return 0.0
    d = ndi.distance_transform_edt(~unseen) * o.res
    return float((d[pix] < halo_m).mean())


def _axis_run_fraction(sk: np.ndarray, run_px: int) -> float:
    """Fraction of skeleton pixels lying on straight horizontal or vertical runs of >= run_px pixels.

    Man-made outlines (picture frames, door leaves, cabinet fronts, the floor-wall junction) are made of such runs;
    cracks wander. The skeleton is thickened by 1 px across the run direction so a 1-px wobble does not break it."""
    sk8 = sk.astype(np.uint8)
    h = cv2.morphologyEx(cv2.dilate(sk8, np.ones((3, 1), np.uint8)), cv2.MORPH_OPEN, np.ones((1, run_px), np.uint8))
    v = cv2.morphologyEx(cv2.dilate(sk8, np.ones((1, 3), np.uint8)), cv2.MORPH_OPEN, np.ones((run_px, 1), np.uint8))
    on = sk & ((h > 0) | (v > 0))
    return float(on.sum() / max(sk.sum(), 1))


def _two_sided_contrast(L: np.ndarray, band: np.ndarray, ring: np.ndarray, centre: np.ndarray,
                        normal_xy: np.ndarray) -> float:
    """Lightness of the line minus each side's lightness; returns the weaker (less negative) of the two.

    A crack is darker than the wall on both sides; an object boundary (step edge) is darker than one side only.
    Sides are split by the line's principal axis (normal_xy is the minor eigenvector, in (x, y) pixel order)."""
    if band.sum() < 3 or ring.sum() < 10:
        return 0.0
    ys, xs = np.nonzero(ring)
    side = (np.c_[xs, ys] - centre) @ normal_xy > 0
    Lb = float(np.median(L[band]))
    c = [Lb - float(np.median(L[ys[m], xs[m]])) for m in (side, ~side) if m.sum() >= 5]
    return max(c) if len(c) == 2 else 0.0


def _ridge(L: np.ndarray, valid: np.ndarray, sigmas, erode: int) -> tuple[np.ndarray, np.ndarray]:
    rid = sato(L, sigmas=sigmas, black_ridges=True, mode="reflect")
    inner = ndi.binary_erosion(valid, iterations=erode)
    rid[~inner] = 0
    return rid, inner


def two_sided_map(L: np.ndarray, d_px: float) -> np.ndarray:
    """Per-pixel symmetric line contrast: max over 4 normal directions of min(L(p+d n), L(p-d n)) - L(p).

    Positive only where BOTH sides are brighter (a dark line in any orientation). On a step edge (tape border,
    paint/tile boundary, object outline) one side is not brighter, so the value is ~0 even though the Sato filter
    responds there (a step is half a ridge). v1 only applied the two-sided test to whole regions, after grouping,
    so a real line touching a step edge was merged with it and rejected as an outline (staged tape decals, v2)."""
    d = max(1, int(round(d_px)))
    e = max(1, int(round(d_px / np.sqrt(2))))
    P = np.pad(L, d + 1, mode="edge")
    H, W = L.shape

    def sh(dy: int, dx: int) -> np.ndarray:
        return P[d + 1 + dy:d + 1 + dy + H, d + 1 + dx:d + 1 + dx + W]

    best = np.full(L.shape, -np.inf, np.float32)
    for dy, dx in ((0, d), (d, 0), (e, e), (e, -e)):
        best = np.maximum(best, np.minimum(sh(dy, dx), sh(-dy, -dx)) - L)
    return best


def _cut_axis_runs(cand: np.ndarray, run: int) -> np.ndarray:
    """Remove straight horizontal/vertical runs of >= run px BEFORE grouping: a crack that touches a door frame or
    skirting would otherwise be merged with it and rejected as an outline (seen in the v1 injection test)."""
    c8 = cand.astype(np.uint8)
    axis_runs = (cv2.morphologyEx(c8, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)) |
                 cv2.morphologyEx(c8, cv2.MORPH_OPEN, np.ones((run, 1), np.uint8))) > 0
    return cand & ~ndi.binary_dilation(axis_runs, iterations=1)


def _exploded(length: float, straight: float, coarse_len: float, coarse_straight: float) -> bool:
    """Leak test for the fine trace: growing far beyond the coarse candidate AND curling up means the hysteresis
    followed texture (wood grain, fridge magnets), not the crack (seen on 2 injected cracks, damage.md v2 I-12)."""
    return length > 1.6 * coarse_len + 0.10 or straight < 0.6 * coarse_straight


def trace_crack(fo: Ortho, seed: np.ndarray, p: DetectParams, coarse_len: float = 0.0,
                coarse_straight: float = 1.0) -> dict | None:
    """Re-trace one crack candidate on a fine (2.5 mm/px) orthophoto of its neighbourhood (v2, damage.md I-6).

    seed = the coarse candidate's skeleton rasterised onto the fine grid. Steps:
      1. multi-scale Sato ridge response on L*; its noise level is measured in THIS window, away from the seed;
      2. hysteresis: everything above the low threshold that is connected to the seed (with gap bridging) belongs
         to the crack. The low threshold is noise-relative (4 sigma) with an absolute floor, so faint stretches of
         a 1.5-3 mm crack that dropped out at 5 mm/px are recovered without flooding a noisy surface;
      3. skeleton, spurs < 1 cm pruned, MST length (includes branches);
      4. threshold sensitivity: the same at 0.7x and 1.4x the low threshold -> the length interval term."""
    res = fo.res
    valid = fo.mask & (fo.count >= p.min_views) & np.isfinite(fo.median[..., 0])
    if valid.sum() < 200 or not (seed & valid).any():
        return None
    lab = _lab(fo.median)
    Lr = lab[..., 0]
    fill = _masked_blur(np.where(valid, Lr, 0).astype(np.float32), valid.astype(np.float32), 4.0)
    L = np.where(valid, Lr, fill).astype(np.float32)
    rid, inner = _ridge(L, valid, p.crack_fine_sigmas_px, 2)
    rid = np.where(two_sided_map(cv2.GaussianBlur(L, (0, 0), 0.7), p.crack_side_m / res) > p.crack_pix_contrast,
                   rid, 0.0)
    seed_zone = ndi.binary_dilation(seed, iterations=max(1, int(round(0.0075 / res)))) & inner
    away = inner & ~ndi.binary_dilation(seed, iterations=max(2, int(round(0.03 / res))))
    noise = max(_robust_sigma(rid[away]), 0.05) if away.sum() > 100 else max(_robust_sigma(rid[inner]), 0.05)
    lo = max(p.crack_fine_z_lo * noise, p.crack_fine_abs_lo)
    run = max(3, int(round(p.crack_axis_cut_m / res)))

    def grow(thr: float) -> np.ndarray:
        m = _cut_axis_runs((rid > thr) & inner, run)
        m, glab, _ = link_fragments(m, res, p.crack_link_gap_m, p.crack_link_near_m, p.crack_link_angle_deg)
        ids = np.unique(glab[seed_zone & m])
        return np.isin(glab, ids[ids > 0])

    def measure(reg: np.ndarray) -> tuple[np.ndarray, float]:
        if not reg.any():
            return reg, 0.0
        sk = prune_spurs(skeletonize(ndi.binary_closing(reg, iterations=1) | reg),
                         max(2, int(round(p.crack_spur_m / res))))
        return sk, mst_length(sk, res)

    def straightness(sk: np.ndarray, length: float) -> float:
        return _line_geometry(sk, res)["span"] / max(length, res) if sk.any() else 1.0

    def ok(sk: np.ndarray, length: float) -> bool:
        return length > 0 and not (coarse_len and _exploded(length, straightness(sk, length), coarse_len,
                                                              coarse_straight))

    # threshold ladder: the lowest threshold whose trace does not leak into texture
    for k in (1.0, 1.4, 2.0, 2.8):
        reg = grow(k * lo)
        sk, length = measure(reg)
        if ok(sk, length):
            lo = k * lo
            break
    else:
        return None
    # threshold sensitivity: lengths at 1.4x and 0.7x (a leaking variant does not count: it is texture)
    alts = []
    for k in (1.4, 0.7):
        ska, la = measure(grow(k * lo))
        if ok(ska, la):
            alts.append(la)
    return dict(ortho=fo, valid=valid, lab=lab, region=reg, skeleton=sk, length=length,
                length_strict=min(alts + [length]), length_loose=max(alts + [length]), noise=noise, lo=lo,
                touches_border=bool((sk & ~ndi.binary_erosion(fo.mask, iterations=3)).any()))


def _line_geometry(sk: np.ndarray, res: float) -> dict:
    ys, xs = np.nonzero(sk)
    pts = np.c_[xs, ys].astype(float)
    c = pts - pts.mean(0)
    _, evec = np.linalg.eigh(c.T @ c)
    ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180
    span = float(np.ptp(c @ evec[:, 1]) * res + res)
    return dict(pts=pts, evec=evec, ang=float(ang), axis_dev=float(min(ang % 90, 90 - ang % 90)), span=span)


def _dominant_line(sk: np.ndarray, res: float, tol_m: float, iters: int = 300) -> tuple[float, float]:
    """(inlier fraction, axis deviation in deg) of the best single straight line through the skeleton (RANSAC).

    Why (D-024): a door handle or an edge with short perpendicular stand-offs makes an L/U-shaped skeleton whose
    overall straightness (span/length) looks crack-like, but most of its pixels sit on ONE straight axis-aligned
    line. Real cracks zig-zag, so few pixels sit within a few millimetres of any single line."""
    ys, xs = np.nonzero(sk)
    if len(xs) < 4:
        return 1.0, 0.0
    pts = np.c_[xs, ys].astype(float)
    rng = np.random.default_rng(0)
    tol = tol_m / res
    best_frac, best_dir = 0.0, np.array([1.0, 0.0])
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        n = np.linalg.norm(d)
        if n < 3:
            continue
        d /= n
        dist = np.abs((pts - pts[i]) @ np.array([-d[1], d[0]]))
        frac = float((dist <= tol).mean())
        if frac > best_frac:
            best_frac, best_dir = frac, d
    ang = np.degrees(np.arctan2(best_dir[1], best_dir[0])) % 180
    return best_frac, float(min(ang % 90, 90 - ang % 90))


def _measure_crack(o: Ortho, valid: np.ndarray, lab: np.ndarray, reg: np.ndarray, sk: np.ndarray, length: float,
                   spread: np.ndarray, p: DetectParams) -> dict:
    """Shape, two-sided contrast and multi-view support of a crack on one orthophoto (coarse or fine)."""
    res = o.res
    g = _line_geometry(sk, res)
    dt = ndi.distance_transform_edt(reg)
    width = float(2 * np.median(dt[sk]) * res) if (reg & sk).any() else res
    k = max(1, int(round(0.005 / res)))                 # the line itself: +-5 mm around the skeleton
    band = ndi.binary_dilation(sk, iterations=k) & valid
    ring = ndi.binary_dilation(band, iterations=3 * k) & ~ndi.binary_dilation(band, iterations=k) & valid
    contrast = _two_sided_contrast(lab[..., 0], band, ring, g["pts"].mean(0), g["evec"][:, 0])
    sup, nv = _view_support(o, band, ring, contrast, p.support_ratio) if ring.sum() > 10 else (0.0, 0)
    bar_frac, bar_axis_dev = _dominant_line(sk, res, max(p.crack_bar_tol_m, 1.5 * res))
    return dict(length_m=length, width_m=width, straightness=g["span"] / max(length, res), axis_dev_deg=g["axis_dev"],
                bar_frac=bar_frac, bar_axis_dev_deg=bar_axis_dev,
                orientation_deg=g["ang"], contrast_L=contrast, support=sup, n_views=nv,
                axis_frac=_axis_run_fraction(sk, max(3, int(round((p.crack_axis_run_fine_m if res < 0.004 else
                                                                    p.crack_axis_run_m) / res)))),
                halo_frac=_halo_frac(sk, o, valid, p.halo_m), local_spread_L=_local_spread(spread, band | ring),
                band=band, ring=ring)


def _crack_rule(m: dict, p: DetectParams) -> str:
    """The v1 rejection rules, applied to the (refined) measurement. Order = cheapest explanation first."""
    if m["length_m"] < p.crack_min_len_m:
        return f"too short ({100 * m['length_m']:.0f} cm < {100 * p.crack_min_len_m:.0f} cm)"
    if m["straightness"] > p.crack_straight and m["axis_dev_deg"] < p.crack_axis_tol_deg:
        return "straight axis-aligned line: panel/skirting/door-frame edge or tile joint"
    if m.get("bar_frac", 0.0) >= p.crack_bar_frac and m.get("bar_axis_dev_deg", 90.0) < p.crack_axis_tol_deg:
        return (f"axis-aligned bar ({100*m['bar_frac']:.0f}% of the line within {1000*p.crack_bar_tol_m:.0f} mm of one "
                f"straight H/V line): handle, edge or grout line, not a jagged crack")
    if m["axis_frac"] > p.crack_max_axis_frac:
        return f"rectilinear outline ({100*m['axis_frac']:.0f}% on horizontal/vertical runs): frame, door or furniture edge"
    if m["straightness"] < p.crack_min_straight:
        return (f"tortuous (span/length {m['straightness']:.2f}): object outline, cable or magnet, not a crack")
    if m["width_m"] > p.crack_max_width_m:
        return f"too wide for a crack ({100*m['width_m']:.1f} cm): gap or shadow line"
    if m["halo_frac"] > p.max_halo_frac:
        return "hugs the edge of an occluded/unseen patch: occluder halo (shadow, blurred object edge)"
    if m["contrast_L"] > -p.crack_min_contrast:
        return "not darker than BOTH sides: a step edge (object boundary) or too faint"
    if m["n_views"] < p.min_views:
        return f"seen in only {m['n_views']} views"
    if m["support"] < p.min_support or round(m["support"] * m["n_views"]) < p.min_support_views:
        return f"inconsistent across views (support {m['support']:.2f} of {m['n_views']})"
    if m["local_spread_L"] > p.max_spread_L:
        return "views disagree around it (reflective/transparent surface: mirror, glass, gloss)"
    return ""


def _to_coarse(fine: np.ndarray, fo: Ortho, o: Ortho) -> np.ndarray:
    """Rasterise a fine-window mask onto the coarse surface grid (for overlays, stain-zone tests, endpoints)."""
    out = np.zeros(o.mask.shape, bool)
    ys, xs = np.nonzero(fine)
    if not len(ys):
        return out
    s, t = fo.surface.pixel_st(xs, ys, fo.res)
    off_f, off_c = fo.surface.meta.get("tile_offset", (0.0, 0.0)), o.surface.meta.get("tile_offset", (0.0, 0.0))
    c, r = o.surface.st_to_pixel(s + off_f[0] - off_c[0], t + off_f[1] - off_c[1], o.res)
    c, r = np.clip(np.round(c).astype(int), 0, out.shape[1] - 1), np.clip(np.round(r).astype(int), 0, out.shape[0] - 1)
    out[r, c] = True
    return out


def _to_fine(coarse: np.ndarray, o: Ortho, fo: Ortho) -> np.ndarray:
    out = np.zeros(fo.mask.shape, bool)
    ys, xs = np.nonzero(coarse)
    s, t = o.surface.pixel_st(xs, ys, o.res)
    off_f, off_c = fo.surface.meta.get("tile_offset", (0.0, 0.0)), o.surface.meta.get("tile_offset", (0.0, 0.0))
    c, r = fo.surface.st_to_pixel(s + off_c[0] - off_f[0], t + off_c[1] - off_f[1], fo.res)
    c, r = np.round(c).astype(int), np.round(r).astype(int)
    ok = (c >= 0) & (c < out.shape[1]) & (r >= 0) & (r < out.shape[0])
    out[r[ok], c[ok]] = True
    # a coarse pixel spans res/fine_res fine pixels: close the dotted line
    return ndi.binary_dilation(out, iterations=max(1, int(round(o.res / fo.res / 2))))


def detect_cracks(o: Ortho, lab: np.ndarray, bg: np.ndarray, valid: np.ndarray, p: DetectParams,
                  spread: np.ndarray, refine=None) -> tuple[list[Region], dict]:
    """Coarse search on the 5 mm orthophoto, then (v2) fine re-tracing of every plausible candidate.

    `refine(s0, t0, s1, t1) -> Ortho | None` builds a fine orthophoto of a window of this surface (in this ortho's
    surface coordinates); without it, or when it fails, the coarse measurement is used and marked as such."""
    res = o.res
    L = np.where(valid, lab[..., 0], bg[..., 0]).astype(np.float32)
    rid, inner = _ridge(L, valid, p.crack_sigmas_px, 3)
    s = max(_robust_sigma(rid[inner]), 0.05) if inner.any() else 1.0
    z = rid / s
    lo_t, hi_t = max(p.crack_z_lo * s, p.crack_abs_lo), max(p.crack_z_hi * s, p.crack_abs_hi)
    sym = two_sided_map(cv2.GaussianBlur(L, (0, 0), 0.7), p.crack_side_m / res) > p.crack_pix_contrast
    rid = np.where(sym, rid, 0.0)
    cand = apply_hysteresis_threshold(rid, lo_t, hi_t) & inner
    cand = _cut_axis_runs(cand, max(3, int(round(p.crack_axis_cut_m / res))))
    cand, groups, n = link_fragments(cand, res, p.crack_link_gap_m, p.crack_link_near_m, p.crack_link_angle_deg)
    out, claimed = [], np.zeros_like(valid)
    order = sorted(range(1, n + 1), key=lambda i: -int((groups == i).sum()))    # longest first (dedupe below)
    for i in order:
        reg = groups == i
        if (reg & claimed).sum() > 0.5 * reg.sum():
            continue                                     # already part of a refined crack
        sk = skeletonize(ndi.binary_closing(reg, iterations=2) | reg)
        length = mst_length(sk, res)
        if length < p.crack_seed_min_len_m:
            continue
        m = _measure_crack(o, valid, lab, reg, sk, length, spread, p)
        rg = Region(o.surface.id, "crack", reg, score=float(np.mean(z[sk])), support=m["support"],
                    n_views=m["n_views"], stats=dict(m, skeleton=sk, refined=False))
        # coarse pre-filter only for the obvious cases; the outline (axis-run) rule is decided on the FINE skeleton:
        # at 5 mm/px with a 3 px tolerance band, a jagged line along a horizontal strip of tape looked 59-92%
        # 'rectilinear' (staged decals, v2 I-14); at 2.5 mm/px its 1-2.5 cm zigzag segments are not 5 cm runs
        structural = (m["straightness"] > p.crack_straight and m["axis_dev_deg"] < p.crack_axis_tol_deg) or \
            m["axis_frac"] > 0.97
        if refine is not None and p.crack_fine_res and not structural:
            ys, xs = np.nonzero(reg)
            s_px, t_px = o.surface.pixel_st(xs, ys, res)
            mg = min(0.05 + 0.5 * length, 0.30)            # room for the faint stretches the coarse pass missed
            fo = refine(s_px.min() - mg, t_px.min() - mg, s_px.max() + mg, t_px.max() + mg)
            tr = trace_crack(fo, _to_fine(sk, o, fo), p, length, m["straightness"]) if fo is not None else None
            if tr is not None:
                fm = _measure_crack(fo, tr["valid"], tr["lab"], tr["region"], tr["skeleton"], tr["length"],
                                    view_spread(fo), p)
                csk = _to_coarse(tr["skeleton"], fo, o)
                cpx = _to_coarse(tr["region"], fo, o) | csk
                # occluder halo is judged on the SURFACE-wide visibility too: the fine window picks its own best
                # views, which can see around an occluder edge that most views do not (3 clean-wall false cracks
                # hugged unseen patches before this, v2 I-15)
                fm["halo_frac"] = max(fm["halo_frac"], _halo_frac(csk, o, valid, p.halo_m), m["halo_frac"])
                # multi-view support is asked on the COARSE grid (5 mm > registration error): at 2.5 mm/px a 1-2 mm
                # line drifts in and out of its 1-px band from view to view, and real tape-drawn lines scored
                # support 0.4-0.7 there (v2 I-16). Contrast and geometry stay fine-resolution measurements.
                mc = _measure_crack(o, valid, lab, cpx, csk, tr["length"], spread, p) if csk.any() else m
                fm["support"], fm["n_views"] = mc["support"], mc["n_views"]
                rg = Region(o.surface.id, "crack", cpx, score=float(np.mean(z[csk])) if csk.any() else rg.score,
                            support=fm["support"], n_views=fm["n_views"],
                            stats=dict(fm, skeleton=csk, refined=True, fine=tr, coarse_length_m=length,
                                       fine_noise=tr["noise"], fine_lo=tr["lo"]))
                claimed |= ndi.binary_dilation(cpx, iterations=1)
        rg.rejected = _crack_rule(rg.stats, p)
        out.append(rg)
    area = valid.sum() * res * res
    # density counts only lines that passed every other test: veined/marbled or busy textures produce many of them,
    # and there the test cannot be trusted. The surface is then flagged for manual review, not silently cleared.
    passed = [r for r in out if not r.rejected]
    density = sum(r.stats["length_m"] for r in passed) / max(area, 1e-6)
    info = dict(ridge_density_m_per_m2=density, textured=bool(density > p.crack_max_density and len(passed) >= 4))
    if info["textured"]:
        for rg in out:
            if not rg.rejected:
                rg.rejected = f"textured surface ({density:.1f} m of lines per m2, e.g. tiles): crack test unreliable"
    return out, info


def analyse(o: Ortho, p: DetectParams | None = None, refine=None) -> tuple[list[Region], dict]:
    """Run both detectors on one orthophoto. Returns all candidates (accepted ones have rejected == '').
    `refine` (optional) builds fine orthophotos of windows of this surface for crack tracing (v2)."""
    p = p or DetectParams()
    valid = o.mask & (o.count >= p.min_views) & np.isfinite(o.median[..., 0])
    valid = ndi.binary_erosion(valid, iterations=2)
    info = dict(surface_id=o.surface.id, valid_m2=float(valid.sum() * o.res ** 2), n_views=len(o.view_ids))
    if valid.sum() < 2000:
        info["skipped"] = "too little texture seen by >= 3 views"
        return [], info
    lab = _lab(o.median)
    bg, _ = background(lab, valid, p.bg_sigma_m / o.res)
    spread = view_spread(o)
    info["median_view_spread_L"] = float(np.median(spread[valid]))
    regs = detect_stains(o, lab, bg, valid, p, spread)
    cracks, cinfo = detect_cracks(o, lab, bg, valid, p, spread, refine)
    info.update(cinfo)
    # a stain's dark tide line is a thin dark curve: without this, every detected stain also produced 2-6 'cracks'
    # along its outline (seen in the injection figure, damage.md I-8)
    # v2: only the stain's OUTLINE band counts. A crack drawn INSIDE a stained/discoloured area (e.g. a line on a
    # strip of beige masking tape that the stain detector also picked up) is not a tide line.
    stain_zone = np.zeros_like(valid)
    for r in regs:
        if not r.rejected:
            stain_zone |= r.pixels & ~ndi.binary_erosion(r.pixels, iterations=max(2, int(0.02 / o.res)))
    if stain_zone.any():
        stain_zone = ndi.binary_dilation(stain_zone, iterations=max(2, int(0.02 / o.res)))
        for c in cracks:
            sk = c.stats["skeleton"]
            if not c.rejected and stain_zone[sk].mean() > 0.5:
                c.rejected = "on the outline of a detected stain: tide line, not a crack"
    for r in regs:
        r.stats["local_spread_L"] = _local_spread(spread, ndi.binary_dilation(r.pixels, iterations=6) & valid)
    for r in regs + cracks:
        r.stats["confidence"] = confidence(r, p)
        if not r.rejected and r.stats["confidence"] < p.min_confidence:
            r.rejected = f"low confidence ({r.stats['confidence']:.2f} < {p.min_confidence:.2f})"
    return regs + cracks, info


# ----------------------------------------------------------------------------------------------------------------------
# Metric description with 95% intervals
# ----------------------------------------------------------------------------------------------------------------------
def _iv(v: float, sigma: float, unit: str, method: str, floor0: bool = True) -> dict:
    lo = v - 1.96 * sigma
    return dict(value=round(v, 4), lo=round(max(lo, 0.0) if floor0 else lo, 4), hi=round(v + 1.96 * sigma, 4),
                unit=unit, method=method, status="measured")


def describe(rg: Region, o: Ortho, pose_sigma: float, scale_sigma: float) -> dict:
    """Region -> JSON with metric extent in SURFACE coordinates (s along the wall from p0, t = height above the
    floor for walls; (x, z) offsets for ceilings/floors), each with a 95% interval.

    Error budget (1 sigma):
      boundary sigma_b  = sqrt((res/2)^2 + pose_sigma^2)   pixel quantisation + view-to-view registration
      threshold term    = half the area change between a 0.7x and 1.4x growth threshold (stains only)
      scale term        = scale_sigma * size (0 for LiDAR; video/photo carry their metric-scale uncertainty)"""
    fine = rg.stats.get("fine") if rg.cls == "crack" else None
    if fine is not None:            # v2: refined cracks are described on their fine (2.5 mm) orthophoto
        o = fine["ortho"]
        pix = fine["region"] | fine["skeleton"]
    else:
        pix = rg.pixels
    res, srf = o.res, o.surface
    ys, xs = np.nonzero(pix)
    s, t = srf.pixel_st(xs, ys, res)
    off = srf.meta.get("tile_offset", (0.0, 0.0))
    s = s + off[0]
    t = t + off[1]
    s0, s1, t0, t1 = s.min() - res / 2, s.max() + res / 2, t.min() - res / 2, t.max() + res / 2
    sb = float(np.hypot(res / 2, pose_sigma))
    w, h = s1 - s0, t1 - t0
    sw = float(np.sqrt(2 * sb ** 2 + (scale_sigma * w) ** 2))
    sh = float(np.sqrt(2 * sb ** 2 + (scale_sigma * h) ** 2))
    cs, ct = float(s.mean()), float(t.mean())
    pos_s = float(np.sqrt(sb ** 2 + (scale_sigma * cs) ** 2))
    pos_t = float(np.sqrt(sb ** 2 + (scale_sigma * ct) ** 2))
    d = dict(surface_id=srf.id, surface_kind=srf.kind, room_id=srf.room_id, cls=rg.cls,
             confidence=round(rg.stats.get("confidence", confidence(rg)), 3), support=round(rg.support, 3),
             n_views=rg.n_views,
             score=round(rg.score, 2),
             bbox_st=[round(float(v), 4) for v in (s0, t0, s1, t1)],
             width=_iv(w, sw, "m", "extent along s of the detected pixels"),
             height=_iv(h, sh, "m", "extent along t of the detected pixels"),
             center_s=_iv(cs, pos_s, "m", "centroid along the surface s axis", floor0=False),
             center_t=_iv(ct, pos_t, "m", "centroid along t (walls: height above the floor)", floor0=False))
    if rg.cls == "water_stain":
        A = rg.pixels.sum() * res * res
        cnts, _ = cv2.findContours(rg.pixels.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(c, True) for c in cnts) * res
        thr = 0.5 * abs(rg.stats["z_area_hi"] - rg.stats["z_area_lo"])
        sA = float(np.sqrt(thr ** 2 + (perim * sb) ** 2 + (2 * scale_sigma * A) ** 2))
        # the same threshold ambiguity moves every edge by ~thr/perimeter: widen width/height accordingly
        edge = 2 * thr / max(perim, 1e-6)
        sw, sh = float(np.hypot(sw, edge)), float(np.hypot(sh, edge))
        d["width"] = _iv(w, sw, "m", "extent along s of the detected pixels")
        d["height"] = _iv(h, sh, "m", "extent along t of the detected pixels")
        d.update(area=_iv(A, sA, "m2", "pixel count x res^2 (holes filled); interval: boundary, threshold, scale"),
                 bottom_t=round(float(t0), 4), top_t=round(float(t1), 4),
                 colour=dict(dL=round(rg.stats["mean_dL"], 2), db=round(rg.stats["mean_db"], 2)))
    else:
        Ls = rg.stats["length_m"]
        # v2 length budget (1 sigma): both endpoints localised to the boundary sigma; threshold sensitivity (half the
        # length change between a 1.4x and a 0.7x growth threshold, i.e. where exactly a fading crack 'ends');
        # skeleton discretisation ~1 px per 10 px of length; scale. The v1 '2x upper bound' is gone (I-6 fixed).
        if fine is not None:
            thr = 0.5 * abs(fine["length_loose"] - fine["length_strict"])
            method = (f"MST length of the ridge skeleton traced at {res*1000:.1f} mm/px with gap bridging; interval: "
                      "endpoints, threshold sensitivity, discretisation, scale")
        else:
            thr = 0.25 * Ls
            method = "MST skeleton length at the coarse 5 mm/px (not refined); interval widened by 25% for fragmentation"
        # trace-completeness term: 10% of the length (1 sigma). Set from the v2 injection residuals (RMS relative
        # error of matched cracks without leaks ~9-10%): faint ends and branches are found or missed as whole
        # pieces, which the threshold-sensitivity term does not capture. Disclosed as calibrated on injection.
        sl = float(np.sqrt(2 * sb ** 2 + thr ** 2 + (0.10 * Ls) ** 2 + (scale_sigma * Ls) ** 2))
        ang = np.radians(rg.stats.get("orientation_deg", 45.0))
        sw2, sh2 = float(np.hypot(sw, thr * abs(np.cos(ang)))), float(np.hypot(sh, thr * abs(np.sin(ang))))
        d["width"] = _iv(w, sw2, "m", "extent along s of the detected crack pixels")
        d["height"] = _iv(h, sh2, "m", "extent along t of the detected crack pixels")
        wdt = rg.stats["width_m"]
        d.update(length=_iv(Ls, sl, "m", method),
                 crack_width=dict(value=round(wdt, 4), lo=0.0, hi=round(wdt + 2 * res, 4), unit="m",
                                  method=f"2 x median distance to the edge; below the {res*1000:.1f} mm ortho pixel "
                                         "the true width is not resolved, so only an upper bound is meaningful",
                                  status="measured" if wdt > 2 * res else "inferred"),
                 area=_iv(Ls * max(wdt, res), Ls * res, "m2", "length x width (for scope only)"),
                 orientation_deg=round(rg.stats["orientation_deg"], 1),
                 endpoints_st=_crack_endpoints(fine["skeleton"] if fine is not None else rg.stats["skeleton"],
                                               srf, res, off),
                 bottom_t=round(float(t0), 4), top_t=round(float(t1), 4),
                 evidence=dict(contrast_L=round(rg.stats.get("contrast_L", 0.0), 2),
                               straightness=round(rg.stats.get("straightness", 0.0), 3),
                               refined=bool(fine is not None),
                               coarse_length_m=round(rg.stats.get("coarse_length_m", Ls), 4)))
    return d


def _crack_endpoints(sk: np.ndarray, srf, res: float, off) -> list[list[float]]:
    """The two skeleton pixels farthest apart along the crack's main axis, in surface (s, t) metres. Used by the
    'crack radiating from an opening corner' rule."""
    ys, xs = np.nonzero(sk)
    pts = np.c_[xs, ys].astype(float)
    c = pts - pts.mean(0)
    _, evec = np.linalg.eigh(c.T @ c)
    proj = c @ evec[:, 1]
    out = []
    for k in (np.argmin(proj), np.argmax(proj)):
        s, t = srf.pixel_st(xs[k], ys[k], res)
        out.append([round(float(s + off[0]), 4), round(float(t + off[1]), 4)])
    return out


def _sig(x: float, c: float, w: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(x - c) / max(w, 1e-6))))


def confidence(rg: Region, p: DetectParams | None = None) -> float:
    """Detection confidence in 0..1 (v2). A product of terms that each answer one question in PHYSICAL units:

      contrast   how much darker (crack: two-sided L*) or yellower (stain: b*) than its own surroundings;
      size       crack length (a 10 cm line is the length at which hairline texture stops looking like a crack);
      shape      cracks wander: a perfectly straight oblique line is more likely a cable/shadow/scratch;
      agreement  how many views independently show it: support x (1 - exp(-n_agreeing/3)).

    v1 used a logistic of the ridge z-score, which saturated at 1.0 on clean paint (the ridge noise is ~0 there, so
    every edge scored z = 40) and gave a video-tier false crack confidence 1.0 (damage.md I-9)."""
    p = p or DetectParams()
    n_agree = rg.support * rg.n_views
    agree = rg.support * (1.0 - np.exp(-n_agree / p.conf_views_scale))
    if rg.cls == "water_stain":
        db = rg.stats.get("local_db", rg.stats.get("mean_db", 0.0))
        return round(float(_sig(db, p.conf_stain_db, p.conf_stain_db_w) * agree), 3)
    c = -rg.stats.get("contrast_L", 0.0)
    L = rg.stats.get("length_m", 0.0)
    straight = rg.stats.get("straightness", 0.9)
    shape = 1.0 - 0.5 * _sig(straight, 0.97, 0.01)
    return round(float(_sig(c, p.conf_crack_contrast_L, p.conf_crack_contrast_w) *
                       _sig(L, p.conf_crack_len_m, p.conf_crack_len_w) * shape * agree), 3)
