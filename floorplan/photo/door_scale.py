"""Reference-free scale cue of the photo tier: STANDARD DOOR HEIGHTS (D-069).

NOT WIRED YET (D-069): floorplan/photo/frontend.py never calls door_cues(), so PhotoParams.door_scale_cue has no effect.

Why: with no paper sheet in the capture (user decision: no reference object) the photo scale is MoGe-2's own, and
MoGe-2 under-scales the real sample photos by 4-8% (outputs/handoff/no_sheet_baseline.md). Every home has doors, and
an interior door is 2.0-2.13 m tall almost everywhere (India 2.1 m / 7 ft = 2.13 m, US 80 in = 2.03 m, EU 2.0-2.1 m).
The ADE20K class map of every photo (semantic.py, D-060) already marks them.

Per photo: connected components of the class "door". A component is used only when it is a whole room door: its top
edge lies inside the image, it stands on the floor, its depth is valid and it has the size of a room door (a wardrobe
or cabinet door fails the floor or the size test). Its height is the top edge above the photo's floor plane. The top
edge is NOT read from the depth map (depth right at a mask edge is unreliable): a vertical plane is fitted to the
door's points and the rays of the top-edge pixels are intersected with it.

One observation per door: s = door_height_prior_m / measured height, in the photo's RAW MoGe-2 units (like the sheet
cue, D-056). All observations are fused into ONE cue. The prior's sigma is systematic within a home (all its doors
have the same height), so it does not shrink with the number of doors; only the photo-to-photo spread does.
"""
from __future__ import annotations

import numpy as np

DOOR_LABELS = ("door",)            # ADE20K. "screen door" (sliding / mesh doors) is not a standard-height leaf
FLOOR_LABELS = ("floor", "rug")
MIN_HEIGHT_PX = 60                 # door height in the image, at the depth resolution (518 px long side)
MIN_WIDTH_PX = 8
BORDER_PX = 3                      # the top edge must be this far inside the image, else the door is cut
MIN_VALID_DEPTH = 0.6              # share of the door's pixels with MoGe depth
MIN_PLANE_INLIERS = 0.6            # share of the door's points on its vertical plane
MIN_INCIDENCE = 0.15               # |cos| between an edge ray and the plane normal: no grazing intersections
MAX_EDGE_MAD_M = 0.04              # the top edge must be level (median abs deviation of its heights)
FLOOR_TOL_M = 0.12                 # door bottom vs the photo's floor plane ("stands on the floor")
HEIGHT_RANGE_M = (1.6, 2.6)        # room door, in MoGe units (the scale itself may be off by 10-15%)
WIDTH_RANGE_M = (0.25, 1.3)        # one leaf, also seen partly; double / sliding doors (> 1.3 m) are not standard
MAX_VS_MOGE = 0.12                 # safety gate: the fused door scale may differ from MoGe-2's by this much
MIN_DOORS, MIN_PHOTOS = 2, 2       # never let one photo decide


def _components(mask: np.ndarray) -> list[np.ndarray]:
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    return [lab == i for i in range(1, n + 1)]


def _vertical_plane(P: np.ndarray, thr: float, iters: int = 300, seed: int = 0):
    """Robust vertical plane n.x = d through levelled points (n horizontal): RANSAC line in the top view, then a
    least-squares refit on its inliers. Returns (n, d, inlier share) or None."""
    q = P[:, [0, 2]]
    if len(q) < 50:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(q), 2, replace=False)
        t = q[j] - q[i]
        ln = np.linalg.norm(t)
        if ln < 0.05:
            continue
        inl = np.abs((q - q[i]) @ (np.array([-t[1], t[0]]) / ln)) < thr
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None or best_n < 30:
        return None
    for _ in range(2):
        c = q[best].mean(0)
        _, _, vt = np.linalg.svd(q[best] - c, full_matrices=False)
        n2 = vt[1]
        best = np.abs((q - c) @ n2) < thr
        if best.sum() < 30:
            return None
    n = np.array([n2[0], 0.0, n2[1]])
    d = float(c @ n2)
    if d < 0:                                           # normal away from the camera, d = distance to the plane
        n, d = -n, -d
    return n, d, float(best.mean())


def _on_plane(u: np.ndarray, v: np.ndarray, K: np.ndarray, R_lev: np.ndarray, n: np.ndarray, d: float):
    """Levelled-frame points where the rays of pixel positions (u, v) meet the plane; mask of sound intersections."""
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))], 1) @ R_lev.T
    den = ray @ n
    ok = den > MIN_INCIDENCE * np.linalg.norm(ray, axis=1)
    t = d / np.where(ok, den, 1.0)
    return ray * t[:, None], ok


def measure_door(comp: np.ndarray, depth: np.ndarray, K: np.ndarray, R_lev: np.ndarray, floor_h: float | None,
                 floor_mask: np.ndarray | None = None) -> dict:
    """Geometry of one door component in the units of `depth`; `reject` names the first failed test (None = usable).

    The same function measures the truth when it is given a LiDAR / ray-cast depth map with its own levelling."""
    h, w = depth.shape
    ys, xs = np.nonzero(comp)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    m = dict(box=[x0, y0, x1, y1], px_height=y1 - y0 + 1, reject=None)
    if y1 - y0 + 1 < MIN_HEIGHT_PX or x1 - x0 + 1 < MIN_WIDTH_PX:
        return dict(m, reject="too small in the image")
    if y0 < BORDER_PX:
        return dict(m, reject="cut by the top border")
    if floor_h is None:
        return dict(m, reject="no floor plane in this photo")
    ok = comp & (depth > 0)
    m["valid_depth"] = float(ok.sum() / comp.sum())
    if m["valid_depth"] < MIN_VALID_DEPTH:
        return dict(m, reject="no depth on the door")
    v, u = np.nonzero(ok)
    if len(u) > 6000:
        sel = np.random.default_rng(0).choice(len(u), 6000, replace=False)
        u, v = u[sel], v[sel]
    z = depth[v, u].astype(float)
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1) @ R_lev.T
    rng_m = float(np.median(np.linalg.norm(P, axis=1)))
    fit = _vertical_plane(P, thr=max(0.03, 0.02 * rng_m))
    if fit is None:
        return dict(m, reject="no vertical plane")
    n, d, share = fit
    m.update(range_m=rng_m, plane_inliers=share)
    if share < MIN_PLANE_INLIERS:
        return dict(m, reject="not planar")
    # top and bottom edge of every column (central 80% of the columns: mask corners are rounded)
    cols = np.nonzero(comp.any(0))[0]
    k = max(1, int(0.1 * len(cols)))
    cols = cols[k:-k] if len(cols) > 2 * k + 4 else cols
    top = comp[:, cols].argmax(0)
    bot = h - 1 - comp[::-1][:, cols].argmax(0)
    Pt, okt = _on_plane(cols.astype(float), top - 0.5, K, R_lev, n, d)       # the pixel's upper boundary
    Pb, okb = _on_plane(cols.astype(float), bot + 0.5, K, R_lev, n, d)
    if okt.sum() < 0.5 * len(cols) or okb.sum() < 0.5 * len(cols):
        return dict(m, reject="seen at a grazing angle")
    y_top = float(np.median(Pt[okt, 1]))
    y_bot = float(np.median(Pb[okb, 1]))
    m.update(top_y=y_top, bottom_y=y_bot, floor_y=float(floor_h),
             top_mad=float(np.median(np.abs(Pt[okt, 1] - y_top))),
             height=y_top - float(floor_h), own_height=y_top - y_bot, bottom_gap=y_bot - float(floor_h),
             cut_side=bool(x0 < BORDER_PX or x1 > w - 1 - BORDER_PX), cut_bottom=bool(y1 > h - 1 - BORDER_PX))
    # width along the plane, over the middle rows
    rows = np.arange(y0 + (y1 - y0) // 5, y1 - (y1 - y0) // 5)
    rows = rows[comp[rows].any(1)]
    left = comp[rows].argmax(1)
    right = w - 1 - comp[rows][:, ::-1].argmax(1)
    Pl, okl = _on_plane(left - 0.5, rows.astype(float), K, R_lev, n, d)
    Pr, okr = _on_plane(right + 0.5, rows.astype(float), K, R_lev, n, d)
    both = okl & okr
    if both.sum() < 5:
        return dict(m, reject="seen at a grazing angle")
    m["width"] = float(np.median(np.linalg.norm((Pr - Pl)[both][:, [0, 2]], axis=1)))
    if floor_mask is not None:                       # is the floor seen directly below the door?
        below = np.clip(bot[None, :] + np.arange(1, 7)[:, None], 0, h - 1)
        m["floor_below"] = float(floor_mask[below, cols[None, :]].mean())
    if m["top_mad"] > MAX_EDGE_MAD_M:
        return dict(m, reject="top edge not level")
    if abs(m["bottom_gap"]) > FLOOR_TOL_M:
        return dict(m, reject="does not stand on the floor")
    if not HEIGHT_RANGE_M[0] <= m["height"] <= HEIGHT_RANGE_M[1]:
        return dict(m, reject="height not a room door's")
    if not WIDTH_RANGE_M[0] <= m["width"] <= WIDTH_RANGE_M[1]:
        return dict(m, reject="width not a room door's")
    return m


def door_candidates(v) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """(component mask, floor mask) of every door-class component of a view; [] without a class map."""
    sem, labels = getattr(v, "sem", None), getattr(v, "sem_labels", None)
    if sem is None or labels is None:
        return []
    ids = [i for i, n in enumerate(labels) if n in DOOR_LABELS]
    floor = np.isin(sem, [i for i, n in enumerate(labels) if n in FLOOR_LABELS])
    return [(c, floor) for c in _components(np.isin(sem, ids)) if c.sum() >= MIN_HEIGHT_PX * MIN_WIDTH_PX]


def fuse_doors(dets: list[dict], p) -> tuple[dict | None, str]:
    """Accepted door observations -> one cue for scale.fuse(), or None with the reason."""
    acc = [d for d in dets if d.get("accepted")]
    photos = sorted({d["image"] for d in acc})
    if len(acc) < MIN_DOORS or len(photos) < MIN_PHOTOS:
        return None, (f"no cue: {len(acc)} door(s) in {len(photos)} photo(s) accepted, "
                      f"need {MIN_DOORS} in {MIN_PHOTOS}")
    ls = np.log([d["scale"] for d in acc])
    med = float(np.median(ls))
    # spread of one observation: MoGe-2's photo-to-photo scale spread plus the edge noise; never below the known
    # per-photo spread (two doors can agree by chance)
    spread = max(1.4826 * float(np.median(np.abs(ls - med))), p.prior_sigma_log_s)
    sigma = float(np.hypot(p.door_height_sigma_rel, spread / np.sqrt(len(photos))))
    scale = float(np.exp(med))
    if abs(med) > np.log(1 + MAX_VS_MOGE):               # 2.4 m designer doors, or not doors at all
        return None, (f"no cue: door scale x{scale:.3f} from {len(acc)} door(s) is more than "
                      f"{100 * MAX_VS_MOGE:.0f}% from the learned depth scale")
    cue = dict(cue="door_height_prior", scale=scale, sigma_rel=sigma, n=len(acc), photos=len(photos),
               spread=spread, height_prior_m=p.door_height_prior_m,
               median_height_units=float(np.median([d["height"] for d in acc])))
    return cue, f"ok: {len(acc)} door(s) in {len(photos)} photo(s)"


def door_cues(views: dict, poses: dict, p, log=print) -> tuple[dict | None, str, list[dict]]:
    """Door-height scale cue over the placed photos: (cue or None, status, every candidate with its verdict)."""
    dets, seen = [], 0
    for name, v in views.items():
        if name not in poses:
            continue
        cands = door_candidates(v)
        seen += getattr(v, "sem", None) is not None
        for comp, floor in cands:
            try:
                m = measure_door(comp, v.depth, v.K, v.R_lev, v.floor_h, floor)
            except Exception as e:                       # a detector failure must never stop the pipeline
                m = dict(reject=f"error: {e!r}")
            m = {k: (round(float(x), 4) if isinstance(x, float) else x) for k, x in m.items()}
            ok = m.pop("reject", None)
            m.update(image=name, accepted=ok is None)
            if ok is None:
                m["scale"] = float(p.door_height_prior_m / m["height"])
            else:
                m["reason"] = ok
            dets.append(m)
    if not seen:
        return None, "no class maps (semantic segmentation did not run)", dets
    cue, status = fuse_doors(dets, p)
    return cue, f"{status}; {len(dets)} door component(s) in {seen} photos searched", dets
