"""Detect a plain A4 / US-Letter sheet lying on the floor and turn it into a METRIC measurement (D-013).

Optional, OFF by default (PhotoParams.use_sheet / VideoParams.use_sheet, D-067): no capture instruction asks for a
sheet.

Why: photos and plain video have no metric scale. A sheet of paper has a standardised size, so once we find its four
corners in an image and know the camera intrinsics K, its pose is fixed, and the camera's distance to the floor
becomes a metric number. Dividing that by the same distance in an up-to-scale reconstruction gives the scale factor
(see floorplan/scale/fuse.py).

Pipeline (detect_sheets):
  1. Candidates: bright regions from several global thresholds + local (adaptive) thresholds + closed Canny edges,
     each simplified to a convex quadrilateral. Several generators, because white paper on a white floor has very
     little contrast and no single threshold finds it in every image.
  2. Sub-pixel edges: for each side, sample intensity profiles across the edge at full resolution, locate the
     steepest bright->dark step with a parabola, and fit a robust straight line. Corners = intersections of adjacent
     lines. Straight-line fitting along the whole side is far more precise than corner detectors (corners are the
     most blurred/damaged part of a sheet) and the residual measures STRAIGHTNESS, i.e. curl.
  3. Verification (false positives cost more than misses, because a wrong scale poisons every measurement):
     bright, uniform, low-chroma interior; brighter than its surroundings on every side; straight edges with a
     consistent bright-inside polarity; edges that END at the corners (tiles and cabinet doors continue as a grid);
     and, using K, the back-projected shape must be a right-angled rectangle with the aspect ratio of A4 or Letter.
  4. Metric pose + Monte Carlo (measure_sheet): the KNOWN rectangle (A4 and Letter, each) is fitted to the corners
     (IPPE; ~5x less noise-sensitive than an unconstrained fit). Corner noise (line fits + a calibrated edge bias),
     focal-length and principal-point uncertainty are propagated by Monte Carlo. The sheet's own rectangle fit is
     used as evidence about the focal length (importance weighting), so a tilted view also self-calibrates f.

Paper type: A4 (297 x 210) and Letter (279.4 x 215.9) differ by +6% in length and -3% in width. Both hypotheses are
kept; their posterior comes from the same rectangle-fit likelihood and the reported distance is the mixture, so an
ambiguous type widens the interval instead of being guessed. (The size measure L^a * W^(1-a), a ~= 0.31, is equal,
234.0 mm, for both papers; the aspect-free gate in rectangle_metrics uses it.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PAPERS = {"A4": (0.297, 0.210), "Letter": (0.2794, 0.2159)}
_A = np.log(0.2159 / 0.210) / np.log((0.297 / 0.2794) * (0.2159 / 0.210))
SIZE_INVARIANT_M = float(0.297 ** _A * 0.210 ** (1 - _A))      # = 0.2794**_A * 0.2159**(1-_A) ~= 0.2340 m
LIKELIHOOD_TEMPER = 1.5  # rectangle-fit likelihood width multiplier; calibrated (1.0 over-confident, 2.0 too wide)
EDGE_BIAS_PX = 0.1      # unmodelled edge-localisation error per side end (px); calibrated in scripts/test_sheet_scale.py


# ----------------------------------------------------------------------------------------------- data classes
@dataclass
class Intrinsics:
    """Pinhole intrinsics with uncertainty. focal_rel_sigma: 1-sigma relative error of f (EXIF ~3-4%, guess ~10%)."""
    K: np.ndarray
    focal_rel_sigma: float = 0.04
    pp_sigma_px: float = 0.0
    source: str = "given"


@dataclass
class SheetDetection:
    corners: np.ndarray                    # (4, 2) float, full-resolution pixels, cyclic order
    score: float                           # 0..1 detection confidence
    checks: dict = field(default_factory=dict)
    line_sigma_px: np.ndarray | None = None      # per-side edge-point residual std (px)
    line_n: np.ndarray | None = None             # per-side number of edge points
    lines: list | None = None                    # per-side (point, direction) for noise propagation
    line_tspread: np.ndarray | None = None       # per-side rms spread of edge points along the line (px)
    source: str = ""


@dataclass
class SheetMeasurement:
    """Metric result for one detection. All distances in metres, 1-sigma values."""
    height_m: float                        # camera centre -> sheet (floor) plane distance
    sigma_m: float                         # total 1-sigma
    sigma_rand_m: float                    # from corner noise only (independent between photos)
    sigma_sys_m: float                     # from intrinsics (shared by photos from the same camera)
    ci95: tuple[float, float]
    normal_cam: np.ndarray                 # unit plane normal in camera coords, pointing towards the camera
    center_cam: np.ndarray                 # sheet centre in camera coords (m)
    paper_probs: dict                      # {"A4": p, "Letter": p}
    paper: str                             # most likely type, or "A4|Letter" if ambiguous
    aspect: float
    angle_err_deg: float
    focal_post_rel: tuple[float, float]    # posterior focal / prior focal: (mean, sigma)
    detection: SheetDetection | None = None

    def to_dict(self) -> dict:
        return dict(height_m=self.height_m, sigma_m=self.sigma_m, sigma_rand_m=self.sigma_rand_m,
                    sigma_sys_m=self.sigma_sys_m, ci95=list(self.ci95), normal_cam=self.normal_cam.tolist(),
                    center_cam=self.center_cam.tolist(), paper_probs=self.paper_probs, paper=self.paper,
                    aspect=self.aspect, angle_err_deg=self.angle_err_deg, focal_post_rel=list(self.focal_post_rel),
                    corners=None if self.detection is None else self.detection.corners.tolist(),
                    score=None if self.detection is None else self.detection.score)


# ----------------------------------------------------------------------------------------------- intrinsics
def default_intrinsics(width: int, height: int) -> Intrinsics:
    """No EXIF: a phone main camera is ~24-28 mm equivalent -> f ~ 0.75 * long side; 10% sigma covers that range."""
    f = 0.75 * max(width, height)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    return Intrinsics(K, 0.10, 0.02 * max(width, height), "default_phone_guess")


def load_image_with_intrinsics(path: str | Path) -> tuple[np.ndarray, Intrinsics]:
    """Read a JPG/HEIC photo, apply the EXIF orientation, and derive K from FocalLengthIn35mmFilm when present.

    35 mm equivalent focal length is defined on the 43.27 mm full-frame diagonal, so f_px = f35 * diag_px / 43.27.
    The value is rounded to whole mm and vendors differ slightly in how they define it, hence ~4% sigma."""
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:                                   # HEIC then simply fails to open; JPG still works
        pass
    im = Image.open(path)
    exif = im.getexif()
    ex = exif.get_ifd(0x8769) if exif else {}
    f35 = ex.get(0xA405) or exif.get(0xA405)
    im = ImageOps.exif_transpose(im).convert("RGB")
    bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if f35:
        f = float(f35) * np.hypot(w, h) / 43.27
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        return bgr, Intrinsics(K, 0.04, 0.01 * max(w, h), "exif_35mm")
    return bgr, default_intrinsics(w, h)


# ----------------------------------------------------------------------------------------------- geometry core
def backproject_parallelogram(corners: np.ndarray, K: np.ndarray) -> np.ndarray:
    """3D corners (up to one global scale) of the parallelogram that projects to the given quad(s).

    A planar parallelogram X_i = l_i r_i has diagonals that bisect: X0 + X2 = X1 + X3. With l0 = 1 this is a 3x3
    linear system, exact for ANY image quad, so it needs no iterative pose solver and vectorises over Monte Carlo
    samples. corners: (..., 4, 2); K: (..., 3, 3). Returns (..., 4, 3)."""
    c = np.asarray(corners, float)
    hom = np.concatenate([c, np.ones(c.shape[:-1] + (1,))], -1)
    r = np.einsum("...ij,...kj->...ki", np.linalg.inv(K), hom)                 # rays (..., 4, 3)
    A = np.stack([r[..., 2, :], -r[..., 1, :], -r[..., 3, :]], -1)            # unknowns l2, l1, l3
    lam = np.linalg.solve(A, -r[..., 0, :][..., None])[..., 0]
    l = np.stack([np.ones(lam.shape[:-1]), lam[..., 1], lam[..., 0], lam[..., 2]], -1)
    return r * l[..., None]


def rectangle_metrics(X: np.ndarray) -> dict:
    """Shape and metric pose of back-projected parallelogram(s) X (..., 4, 3), scaled to the paper size invariant.

    Returns aspect (>=1), corner angle error |angle - 90| (deg), camera-to-plane distance (m), unit normal pointing
    to the camera, and centre (m)."""
    e1 = X[..., 1, :] - X[..., 0, :]
    e2 = X[..., 3, :] - X[..., 0, :]
    l1, l2 = np.linalg.norm(e1, axis=-1), np.linalg.norm(e2, axis=-1)
    cosang = np.sum(e1 * e2, -1) / (l1 * l2)
    angle_err = np.degrees(np.abs(np.arcsin(np.clip(cosang, -1, 1))))
    L, W = np.maximum(l1, l2), np.minimum(l1, l2)
    # true side lengths of the parallelogram are fine for a near-rectangle; its area-like invariant fixes the scale
    k = SIZE_INVARIANT_M / (L ** _A * W ** (1 - _A))
    n = np.cross(e1, e2)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    centre = X.mean(-2)
    d = np.sum(n * centre, -1)
    n = np.where((d > 0)[..., None], -n, n)                                     # point towards camera (origin)
    angle_dev = np.degrees(np.arcsin(np.clip(cosang, -1, 1)))
    return dict(aspect=L / W, angle_err=angle_err, angle_dev=angle_dev, dist=np.abs(d) * k, normal=n, centre=centre * k[..., None])


# ----------------------------------------------------------------------------------------------- candidates
def _lab(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return lab[..., 0], lab[..., 1:]


def _quad_from_contour(c: np.ndarray, min_area: float) -> np.ndarray | None:
    """Convex quadrilateral approximating a contour, or None. Low solidity = something cuts into the region
    (occluder, or a region that is not a single quadrilateral) -> rejected early."""
    area = cv2.contourArea(c)
    if area < min_area:
        return None
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    if ha <= 0 or area / ha < 0.88:
        return None
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.045):
        q = cv2.approxPolyDP(hull, eps * peri, True)
        if len(q) == 4:
            break
    if len(q) != 4:
        return None
    q = q[:, 0, :].astype(np.float64)
    qa = abs(cv2.contourArea(q.astype(np.float32)))
    if qa <= 0 or not (0.85 < ha / qa < 1.15):
        return None
    return q


def _generate_candidates(L: np.ndarray, min_area: float, max_area: float) -> list[tuple[np.ndarray, str]]:
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    out: list[tuple[np.ndarray, str]] = []
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def add(binary: np.ndarray, tag: str) -> None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3)
        cs, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            a = cv2.contourArea(c)
            if min_area <= a <= max_area:
                q = _quad_from_contour(c, min_area)
                if q is not None:
                    out.append((q, tag))

    lo, hi = np.percentile(L, [40, 99.7])
    for t in np.arange(lo, hi, 2.5):                     # global levels (MSER-like): fine steps for low contrast
        add((L > t).astype(np.uint8) * 255, f"thr{t:.0f}")
    h, w = L.shape
    for frac in (0.06, 0.15, 0.3):                          # local levels: paper brighter than its own surroundings
        b = int(max(h, w) * frac) | 1
        for C in (-2, -5):
            add(cv2.adaptiveThreshold(L8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, b, C), f"ad{frac}{C}")
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for t1, t2 in ((8, 24), (20, 60)):                   # edges: regions enclosed by closed edge chains
        e0 = cv2.Canny(cv2.GaussianBlur(L8, (5, 5), 1.0), t1, t2)
        add(255 - cv2.dilate(e0, k3), f"canny{t1}")
        add(255 - cv2.dilate(e0, k5), f"canny{t1}d5")    # closes small gaps where a side meets another edge
    return out


def _shape_prefilter(cands: list[tuple[np.ndarray, str]], s: float, K: np.ndarray,
                     floor_normal_cam: np.ndarray | None) -> list[tuple[np.ndarray, str]]:
    """Drop candidates that cannot be a sheet BEFORE the 50-cluster cap (D-056): rectified (closed form, any quad)
    aspect far from A4/Letter (1.41/1.29), corners far from square, or (with gravity) not lying flat. The final checks
    are much stricter (aspect +-4.5%, 4 deg, 12 deg from the floor); this only stops large non-sheet regions from
    filling the cap. Found on 12 MP photos (sim and phone): tiles, panels and windows gave > 50 larger candidates, so a
    clearly visible sheet was never evaluated."""
    keep = []
    for q, tag in cands:
        try:
            m = rectangle_metrics(backproject_parallelogram(np.asarray(q, float) / s, K))
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not (np.isfinite(m["aspect"]) and 1.15 <= m["aspect"] <= 1.75 and m["angle_err"] <= 15.0):
            continue
        if floor_normal_cam is not None and abs(float(np.dot(m["normal"], floor_normal_cam))) < np.cos(np.radians(25)):
            continue
        keep.append((q, tag))
    return keep


def _cluster(cands: list[tuple[np.ndarray, str]], max_keep: int, max_variants: int = 3) -> list[list]:
    """Group near-identical quads (several generators find the same sheet). Each cluster keeps a few variants: if
    the first one's outline is a few pixels off and edge refinement fails, the next variant gets a chance."""
    cands = sorted(cands, key=lambda c: -abs(cv2.contourArea(c[0].astype(np.float32))))
    clusters: list[list] = []
    for q, tag in cands:
        size = np.sqrt(abs(cv2.contourArea(q.astype(np.float32))))
        for cl in clusters:
            d = np.linalg.norm(_order(q)[:, None] - _order(cl[0][0])[None], axis=-1).min(1)
            if np.max(d) < 0.08 * size:
                if len(cl) < max_variants and all(np.abs(_order(q) - _order(v)).max() > 1.0 for v, _ in cl):
                    cl.append((q, tag))
                break
        else:
            if len(clusters) < max_keep:
                clusters.append([(q, tag)])
    return clusters


def _order(q: np.ndarray) -> np.ndarray:
    """Cyclic counter-clockwise order (image coords) starting from the corner nearest the top-left."""
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    return np.roll(q, -int(np.argmin(q.sum(1))), 0)


# ----------------------------------------------------------------------------------------------- sub-pixel edges
def _sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    m = pts.reshape(-1, 1, 2).astype(np.float32)
    return cv2.remap(img, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(pts.shape[:-1])


def _fit_line(p: np.ndarray, w0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted total-least-squares line with Tukey re-weighting. Returns point, direction, residuals, weights."""
    w = w0.copy()
    for _ in range(4):
        c = (p * w[:, None]).sum(0) / w.sum()
        C = ((p - c) * w[:, None]).T @ (p - c)
        evals, evecs = np.linalg.eigh(C)
        d = evecs[:, 1]
        n = np.array([-d[1], d[0]])
        r = (p - c) @ n
        s = max(1.4826 * np.median(np.abs(r)), 0.25)                     # >= 0.25 px: JPEG/blur edge jitter
        u = r / (4.685 * s)
        w = w0 * np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if w.sum() < 3:
            break
    return c, d, r, w


def _refine_side(Ls: np.ndarray, p0, p1, out_n, halfwidth: float, m: int):
    """Edge points along one side: steepest bright(inside) -> dark(outside) step across the edge, sub-pixel."""
    t = np.linspace(0.1, 0.9, m)
    base = p0[None] + t[:, None] * (p1 - p0)[None]
    offs = np.arange(-halfwidth, halfwidth + 1e-9, 0.5)
    prof = _sample(Ls, base[:, None, :] + offs[None, :, None] * out_n[None, None, :])
    g = np.diff(prof, axis=1)                                                    # outward derivative
    k = np.argmin(g, axis=1)
    strength = -g[np.arange(m), k]
    kk = np.clip(k, 1, g.shape[1] - 2)
    y0, y1, y2 = g[np.arange(m), kk - 1], g[np.arange(m), kk], g[np.arange(m), kk + 1]
    den = y0 - 2 * y1 + y2
    sub = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, 1), 0.0)
    sub = np.clip(sub, -0.5, 0.5)
    o = offs[0] + (kk + sub + 0.5) * 0.5                                         # diff sits between samples
    pts = base + o[:, None] * out_n[None]
    interior_edge = (k > 0) & (k < g.shape[1] - 1)
    return pts, strength, interior_edge


def _intersect(c1, d1, c2, d2) -> np.ndarray:
    A = np.array([d1, -d2]).T
    s = np.linalg.solve(A, c2 - c1)
    return c1 + s[0] * d1


def refine_quad(Lfull: np.ndarray, quad: np.ndarray, passes: int = 2) -> dict | None:
    """Sub-pixel corners from four robust edge lines (full resolution). None if a side has no consistent edge."""
    Ls = cv2.GaussianBlur(Lfull, (0, 0), 0.8)
    q = _order(quad.copy())
    info = None
    for it in range(passes):
        cen = q.mean(0)
        lines, sig, ns, stats = [], [], [], []
        for k in range(4):
            p0, p1 = q[k], q[(k + 1) % 4]
            side = np.linalg.norm(p1 - p0)
            if side < 6:
                return None
            d = (p1 - p0) / side
            n = np.array([-d[1], d[0]])
            if np.dot((p0 + p1) / 2 - cen, n) < 0:
                n = -n
            hw = float(np.clip(0.04 * side, 4, 10)) if it == 0 else 3.0
            m = int(np.clip(side / 3, 12, 80))
            pts, strength, ok = _refine_side(Ls, p0, p1, n, hw, m)
            med = np.median(strength)
            w0 = (ok & (strength > 0.35 * med) & (strength > 0.4)).astype(float)
            if w0.sum() < 0.6 * m:
                return None
            c, dd, r, w = _fit_line(pts, w0)
            inl = w > 0
            stats.append(dict(inlier_frac=float(inl.mean()), strength=float(np.median(strength[inl])),
                              strength_cv=float(np.std(strength[inl]) / max(np.mean(strength[inl]), 1e-6)),
                              rms_px=float(np.sqrt(np.mean(r[inl] ** 2))), side_px=float(side),
                              t_spread=float(np.sqrt(np.mean(((pts[inl] - c) @ dd) ** 2)))))
            lines.append((c, dd))
            sig.append(stats[-1]["rms_px"])
            ns.append(int(inl.sum()))
        q = np.array([_intersect(*lines[(k - 1) % 4], *lines[k]) for k in range(4)])
        if not np.all(np.isfinite(q)):
            return None
        info = dict(corners=q, lines=lines, sigma=np.array(sig), n=np.array(ns), side_stats=stats,
                    tspread=np.array([st["t_spread"] for st in stats]))
    return info


# ----------------------------------------------------------------------------------------------- verification
def _poly_mask(shape, quad: np.ndarray, scale: float) -> np.ndarray:
    c = quad.mean(0)
    p = (c + (quad - c) * scale).round().astype(np.int32)
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, p, 1)
    return m.astype(bool)


def _photometric_checks(L: np.ndarray, ab: np.ndarray, quad: np.ndarray, white_ab: np.ndarray,
                        gmag: np.ndarray | None = None) -> dict:
    """Interior statistics vs. a strip outside each side, on a (possibly downscaled) image.

    Works on a crop around the quad (cost independent of image size). gmag: precomputed gradient magnitude."""
    c = quad.mean(0)
    outer = c + (quad - c) * 1.3
    x0, y0 = np.maximum(np.floor(outer.min(0)).astype(int) - 2, 0)
    x1, y1 = np.minimum(np.ceil(outer.max(0)).astype(int) + 3, [L.shape[1], L.shape[0]])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return dict(ok=False)
    Lc, abc = L[y0:y1, x0:x1], ab[y0:y1, x0:x1]
    q = quad - [x0, y0]
    c = c - [x0, y0]
    if gmag is None:
        gx = cv2.Sobel(Lc, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(Lc, cv2.CV_32F, 0, 1, ksize=3) / 8
        gm = np.hypot(gx, gy)
    else:
        gm = gmag[y0:y1, x0:x1]
    inner = _poly_mask(Lc.shape, q, 0.8)
    if inner.sum() < 30:
        return dict(ok=False)
    Lin = Lc[inner]
    chroma = np.linalg.norm(abc[inner] - white_ab, axis=1)
    core = _poly_mask(Lc.shape, q, 1.04)
    side_contrast = []
    for k in range(4):
        p0, p1 = q[k], q[(k + 1) % 4]
        strip = np.array([p0, p1, c + (p1 - c) * 1.25, c + (p0 - c) * 1.25])
        ms = np.zeros(Lc.shape, np.uint8)
        cv2.fillConvexPoly(ms, strip.round().astype(np.int32), 1)
        ms = ms.astype(bool) & ~core
        side_contrast.append(float(np.median(Lin) - np.median(Lc[ms])) if ms.sum() > 5 else 0.0)
    return dict(ok=True, L_med=float(np.median(Lin)), L_mad=float(1.4826 * np.median(np.abs(Lin - np.median(Lin)))),
                texture=float(np.mean(gm[inner] > 2.5)), chroma=float(np.median(chroma)),
                side_contrast=side_contrast, min_contrast=float(min(side_contrast)))


def _continuation(Ls: np.ndarray, q: np.ndarray, lines) -> float:
    """Fraction of the 8 side extensions (beyond each corner) that still carry a similar edge.

    Paper edges stop at the corners; tile grout, cabinet doors and window frames continue as a grid."""
    hits = 0
    for k in range(4):
        c, d = lines[k]
        p0, p1 = q[k], q[(k + 1) % 4]
        side = np.linalg.norm(p1 - p0)
        n = np.array([-d[1], d[0]])
        for start, sgn in ((p0, -1.0), (p1, 1.0)):
            dirv = d * np.sign(np.dot(p1 - p0, d)) * sgn
            t = np.linspace(0.06, 0.3, 12) * side
            base = start[None] + t[:, None] * dirv[None]
            a = _sample(Ls, base + 2.0 * n[None])
            b = _sample(Ls, base - 2.0 * n[None])
            tin = np.linspace(0.3, 0.7, 12) * side
            bin_ = p0[None] + (tin / side)[:, None] * (p1 - p0)[None]
            a2 = _sample(Ls, bin_ + 2.0 * n[None])
            b2 = _sample(Ls, bin_ - 2.0 * n[None])
            ext = np.median(np.abs(a - b))
            own = np.median(np.abs(a2 - b2))
            hits += int(ext > 0.5 * own)
    return hits / 8.0


def _geometry_check(corners: np.ndarray, intr: Intrinsics) -> dict:
    """Best agreement with a right-angled A4/Letter rectangle over the plausible focal range (+-2.5 sigma)."""
    best = dict(angle_err=np.inf, aspect_err=np.inf)
    fs = np.exp(np.linspace(-2.5, 2.5, 11) * np.log1p(intr.focal_rel_sigma))
    for s in fs:
        K = intr.K.copy()
        K[0, 0] *= s
        K[1, 1] *= s
        m = rectangle_metrics(backproject_parallelogram(corners, K))
        a_err = min(abs(np.log(m["aspect"] / (a / b))) for a, b in PAPERS.values())
        if m["angle_err"] + 40 * a_err < best["angle_err"] + 40 * best["aspect_err"]:
            best = dict(angle_err=float(m["angle_err"]), aspect_err=float(a_err), dist=float(m["dist"]),
                        focal_scale=float(s), normal=m["normal"])
    return best


# ----------------------------------------------------------------------------------------------- public API
@dataclass
class DetectorParams:
    det_max_side: int = 1600          # candidate search resolution (refinement always uses full resolution)
    min_side_px: float = 18.0         # shortest image side of a usable sheet at full resolution
    min_contrast: float = 1.5         # L* units, every side; white paper on white tiles is ~3-8
    max_texture: float = 0.06         # fraction of strong-gradient interior pixels (text, patterns, wood grain)
    max_chroma: float = 14.0          # ab distance from the scene's white
    max_rms_frac: float = 0.004       # edge straightness: rms residual / side length (curl -> curved edges)
    max_rms_px: float = 1.2
    max_angle_err: float = 4.0        # deg, back-projected corner angle vs 90
    max_aspect_err: float = 0.045     # |log(aspect / paper aspect)|; 0.045 rejects 1.5-aspect envelopes/tiles
    max_continuation: float = 0.38    # fraction of side extensions with a continuing edge (grid structures)
    max_floor_angle: float = 12.0     # deg, when the floor normal is known
    dist_range: tuple = (0.15, 8.0)   # m, camera to sheet plane


def detect_sheets(img_bgr: np.ndarray, intr: Intrinsics, floor_normal_cam: np.ndarray | None = None,
                  params: DetectorParams = DetectorParams(), return_rejected: bool = False):
    """Find A4/Letter sheets in an image. Returns accepted detections sorted by score (and rejected ones on request).

    floor_normal_cam (optional): unit floor normal in camera coordinates (e.g. from gravity or a reconstruction); a
    sheet that is not parallel to the floor is then rejected."""
    H, W = img_bgr.shape[:2]
    s = min(1.0, params.det_max_side / max(H, W))
    small = cv2.resize(img_bgr, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    Lsm, absm = _lab(small)
    Lfull = Lsm if s == 1.0 else _lab(img_bgr)[0]
    Ls_full = cv2.GaussianBlur(Lfull, (0, 0), 0.8)
    bright = Lsm > np.percentile(Lsm, 97)
    white_ab = np.median(absm[bright], axis=0) if bright.sum() > 10 else np.zeros(2, np.float32)
    min_area = (params.min_side_px * s) ** 2 * 1.2
    clusters = _cluster(_shape_prefilter(_generate_candidates(Lsm, min_area, 0.35 * Lsm.size), s, intr.K,
                                         floor_normal_cam), 50)
    gmag = np.hypot(cv2.Sobel(Lsm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(Lsm, cv2.CV_32F, 0, 1, ksize=3)) / 8
    accepted, rejected = [], []
    for cl in clusters:
        det, why = None, None
        for q_small, tag in cl:
            det, why = _evaluate(q_small, tag, s, Lsm, absm, gmag, Lfull, Ls_full, white_ab, intr, floor_normal_cam,
                                 params)
            if det is not None:
                break
        if det is not None:
            accepted.append(det)
        elif why is not None:
            rejected.append(why)
    accepted = _nms(sorted(accepted, key=lambda d: -d.score))
    return (accepted, rejected) if return_rejected else accepted


def _evaluate(q_small, tag, s, Lsm, absm, gmag, Lfull, Ls_full, white_ab, intr, floor_normal_cam, params):
    """Refine + verify one candidate. Returns (detection, None) or (None, (quad, tag, checks-with-reasons))."""
    pre = _photometric_checks(Lsm, absm, _order(q_small), white_ab, gmag)
    if not pre.get("ok") or pre["min_contrast"] < 0 or pre["L_med"] < 15:
        return None, (q_small / s, tag, {"reasons": ["pre-check: darker than surroundings or too dark"]})
    ref = refine_quad(Lfull, q_small / s)
    if ref is None:
        return None, (q_small / s, tag, {"reasons": ["no consistent straight edges"]})
    q = ref["corners"]
    sides = np.linalg.norm(q - np.roll(q, -1, 0), axis=1)
    ph = _photometric_checks(Lsm, absm, q * s, white_ab, gmag)
    geo = _geometry_check(q, intr)
    cont = _continuation(Ls_full, q, ref["lines"])
    rms_frac = max(st["rms_px"] / st["side_px"] for st in ref["side_stats"])
    rms_px = max(st["rms_px"] for st in ref["side_stats"])
    checks = dict(source=tag, min_side_px=float(sides.min()), **{k: v for k, v in ph.items() if k != "ok"},
                  rms_frac=float(rms_frac), rms_px=float(rms_px),
                  inlier_frac=float(min(st["inlier_frac"] for st in ref["side_stats"])),
                  angle_err=geo["angle_err"], aspect_err=geo["aspect_err"], dist_m=geo.get("dist", np.nan),
                  continuation=cont)
    reasons = []
    if not ph.get("ok"):
        reasons.append("degenerate")
    else:
        if sides.min() < params.min_side_px: reasons.append("too small")
        if ph["min_contrast"] < params.min_contrast: reasons.append("low contrast on a side")
        if ph["texture"] > params.max_texture: reasons.append("textured interior")
        if ph["chroma"] > params.max_chroma: reasons.append("coloured")
    if rms_frac > params.max_rms_frac and rms_px > 0.6 or rms_px > params.max_rms_px:
        reasons.append("curved edges (curl/occlusion)")
    if checks["inlier_frac"] < 0.75: reasons.append("broken edge (occlusion)")
    if geo["angle_err"] > params.max_angle_err: reasons.append("not a right-angled rectangle")
    if geo["aspect_err"] > params.max_aspect_err: reasons.append("aspect not A4/Letter")
    if cont > params.max_continuation: reasons.append("edges continue (grid/frame)")
    if not (params.dist_range[0] < checks["dist_m"] < params.dist_range[1]): reasons.append("implausible distance")
    if floor_normal_cam is not None and "normal" in geo:
        ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(geo["normal"], floor_normal_cam)), -1, 1))))
        checks["floor_angle"] = ang
        if ang > params.max_floor_angle: reasons.append("not parallel to floor")
    checks["reasons"] = reasons
    det = SheetDetection(corners=q, score=0.0, checks=checks, line_sigma_px=ref["sigma"], line_n=ref["n"],
                         lines=ref["lines"], line_tspread=ref["tspread"], source=tag)
    if reasons:
        return None, (q, tag, checks)
    det.score = _score(checks, params)
    return det, None


def _score(c: dict, p: DetectorParams) -> float:
    """Soft confidence from the margins of the hard gates (1 = every check passes with a wide margin)."""
    terms = [min(1.0, c["min_contrast"] / (3 * p.min_contrast)),
             1 - c["angle_err"] / p.max_angle_err / 2,
             1 - c["aspect_err"] / p.max_aspect_err / 2,
             1 - c["texture"] / p.max_texture / 2,
             1 - c["continuation"] / 2,
             c["inlier_frac"]]
    return float(np.clip(np.prod(np.clip(terms, 0.05, 1)), 0, 1))


def _nms(dets: list[SheetDetection]) -> list[SheetDetection]:
    out: list[SheetDetection] = []
    for d in dets:
        if all(np.linalg.norm(d.corners.mean(0) - o.corners.mean(0)) > 0.3 * np.ptp(o.corners[:, 0]) for o in out):
            out.append(d)
    return out


def _corner_samples(det: SheetDetection, n: int, rng: np.random.Generator, edge_bias_px: float) -> np.ndarray:
    """Monte Carlo corners: perturb each fitted edge line (offset and angle) by its uncertainty, re-intersect.

    Two parts per line:
      * fit noise: n points with residual std s spread rms t along the line give offset sd s/sqrt(n) and angle sd
        s/(t sqrt(n)). This alone is tiny (~0.02 px) and was measured to be ~10x optimistic;
      * edge bias b (px), independently at each end of the side: what the residual cannot see (blur/JPEG/shading
        shifting the steepest-gradient point, paper edge roughness, lens distortion). Its value is calibrated on the
        synthetic tests (coverage of the 95% intervals), see docs/modules/scale_reference.md."""
    out = np.empty((n, 4, 2))
    pert = []
    for k, (c, d) in enumerate(det.lines):
        s = float(det.line_sigma_px[k])
        m = max(int(det.line_n[k]), 3)
        tspread = max(float(det.line_tspread[k]), 1.0)
        side = float(np.linalg.norm(det.corners[(k + 1) % 4] - det.corners[k]))
        nrm = np.array([-d[1], d[0]])
        sd_off = np.hypot(s / np.sqrt(m), edge_bias_px / np.sqrt(2))
        sd_ang = np.hypot(s / (tspread * np.sqrt(m)), edge_bias_px * np.sqrt(2) / max(side, 1.0))
        off = rng.normal(0, sd_off, n)
        ang = rng.normal(0, sd_ang, n)
        ca, sa = np.cos(ang), np.sin(ang)
        dd = np.stack([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]], 1)
        pert.append((c[None] + off[:, None] * nrm[None], dd))
    for k in range(4):
        c1, d1 = pert[(k - 1) % 4]
        c2, d2 = pert[k]
        A = np.stack([d1, -d2], -1)                                                # (n, 2, 2)
        t = np.linalg.solve(A, (c2 - c1)[..., None])[..., 0]
        out[:, k] = c1 + t[:, :1] * d1
    return out


def _rect_object(paper: str) -> np.ndarray:
    L, W = PAPERS[paper]
    return np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0]], np.float64)


def fit_rectangle(corners: np.ndarray, K: np.ndarray, paper: str, refine: bool = True) -> tuple[float, float, np.ndarray,
                                                                                                 np.ndarray]:
    """Maximum-likelihood pose of a KNOWN-size rectangle (IPPE + optional Levenberg-Marquardt).

    Corners must be ordered so that corners[0]->corners[1] is a LONG side. Enforcing the right angles and the aspect
    ratio is what makes this ~5x less sensitive to corner noise than the unconstrained parallelogram (measured: 0.37%
    vs 2.1% height sd for 0.3 px corner noise on a foreshortened view). Returns (camera-to-plane distance, rms
    reprojection error px, unit normal towards camera, centre) in camera coordinates."""
    obj = _rect_object(paper)
    c = np.ascontiguousarray(corners, np.float64)
    ok, rv, tv = cv2.solvePnP(obj, c, K, None, flags=cv2.SOLVEPNP_IPPE)
    if refine:
        rv, tv = cv2.solvePnPRefineLM(obj, c, K, None, rv, tv)
    R = cv2.Rodrigues(rv)[0]
    n = R[:, 2]
    t = tv.ravel()
    d = float(n @ t)
    n = -n if d > 0 else n
    pr, _ = cv2.projectPoints(obj, rv, tv, K, None)
    rms = float(np.sqrt(np.mean(np.sum((pr[:, 0] - c) ** 2, 1))))
    centre = R @ obj.mean(0) + t
    return abs(d), rms, n, centre


def _long_side_first(corners: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Roll the cyclic corner order so the first side is a long side (judged on the back-projected shape)."""
    X = backproject_parallelogram(corners, K)
    l01 = np.linalg.norm(X[1] - X[0])
    l12 = np.linalg.norm(X[2] - X[1])
    return corners if l01 >= l12 else np.roll(corners, -1, 0)


def measure_sheet(det: SheetDetection, intr: Intrinsics, n_mc: int = 600, seed: int = 0,
                  edge_bias_px: float = EDGE_BIAS_PX, like_temper: float = LIKELIHOOD_TEMPER) -> SheetMeasurement:
    """Metric camera-to-sheet-plane distance with Monte Carlo uncertainty (corners + focal + principal point + type).

    For each paper hypothesis (A4, Letter) and each Monte Carlo draw (perturbed corners, focal, principal point) the
    known rectangle is fitted (IPPE). Each focal draw is weighted by how well the OBSERVED corners fit a rectangle of
    that type under that focal (Gaussian likelihood of the reprojection residual): a strongly tilted view thereby
    tightens an uncertain focal, and the same likelihood gives the posterior of A4 vs Letter. The final interval is
    the mixture over both hypotheses, so an ambiguous paper type widens it instead of picking one blindly."""
    rng = np.random.default_rng(seed)
    corners = _long_side_first(det.corners, intr.K)
    det_o = SheetDetection(**{**det.__dict__, "corners": corners})
    shift = int(np.argmin(np.linalg.norm(det.corners - corners[0], axis=1)))
    det_o.lines = [det.lines[(k + shift) % 4] for k in range(4)]
    det_o.line_sigma_px = np.roll(det.line_sigma_px, -shift)
    det_o.line_n = np.roll(det.line_n, -shift)
    det_o.line_tspread = np.roll(det.line_tspread, -shift)
    C = _corner_samples(det_o, n_mc, rng, edge_bias_px)
    sd_corner = float(np.sqrt(np.mean(np.sum((C - corners[None]) ** 2, -1)) / 2))     # per-axis corner sd (px)
    fs = np.exp(rng.normal(0, np.log1p(intr.focal_rel_sigma), n_mc))
    pp = rng.normal(0, intr.pp_sigma_px, (n_mc, 2)) if intr.pp_sigma_px > 0 else np.zeros((n_mc, 2))

    def K_of(k: int) -> np.ndarray:
        K = intr.K.copy()
        K[0, 0] *= fs[k]
        K[1, 1] *= fs[k]
        K[:2, 2] += pp[k]
        return K

    hyp = {}
    for paper in PAPERS:
        d_s, d_r, d_o, logw = np.empty(n_mc), np.empty(n_mc), np.empty(n_mc), np.empty(n_mc)
        for k in range(n_mc):
            Kk = K_of(k)
            d_s[k] = fit_rectangle(C[k], Kk, paper, refine=False)[0]               # corners + intrinsics
            d_r[k] = fit_rectangle(C[k], intr.K, paper, refine=False)[0]          # corner noise only
            d_o[k], rms = fit_rectangle(corners, Kk, paper, refine=False)[:2]     # observed corners, this focal
            logw[k] = -0.5 * 8 * rms ** 2 / (like_temper * sd_corner) ** 2        # 4 corners x 2 coordinates
        # spread around the observed-corner estimate (the perturbed fits' own mean carries a nonlinearity bias)
        hyp[paper] = (np.log(d_o) + (np.log(d_s) - np.log(d_s).mean()), d_r, logw, np.log(d_o))
    lmax = max(v[2].max() for v in hyp.values())
    evid = {p: float(np.mean(np.exp(v[2] - lmax))) for p, v in hyp.items()}
    z = sum(evid.values())
    probs = {p: e / z for p, e in evid.items()}
    # mixture of both hypotheses, each with its own importance weights
    # Variance = focal/paper posterior part (importance-weighted fits of the OBSERVED corners) + corner-noise part
    # (unweighted spread of the perturbed-corner fits). Splitting them keeps the corner part from collapsing onto a
    # few samples when the focal likelihood is sharp (low effective sample size of the weights).
    lo_all = np.concatenate([hyp[p][3] for p in PAPERS])
    w_all = np.concatenate([probs[p] * np.exp(hyp[p][2] - hyp[p][2].max()) / np.exp(hyp[p][2] - hyp[p][2].max()).sum()
                            for p in PAPERS])
    w_all /= w_all.sum()
    mu = float(np.sum(w_all * lo_all))
    var_post = float(np.sum(w_all * (lo_all - mu) ** 2))
    best = max(probs, key=probs.get)
    sd_rand = float(np.std(np.log(hyp[best][1])))
    sd = float(np.sqrt(var_post + sd_rand ** 2))
    sd_sys = float(np.sqrt(var_post))
    lo, hi = float(np.exp(mu - 1.96 * sd)), float(np.exp(mu + 1.96 * sd))
    d0, rms0, n0, c0 = fit_rectangle(corners, intr.K, best)
    nominal = rectangle_metrics(backproject_parallelogram(corners, intr.K))
    wb = np.exp(hyp[best][2] - hyp[best][2].max())
    wb /= wb.sum()
    fmu = float(np.sum(wb * fs))
    fsd = float(np.sqrt(np.sum(wb * (fs - fmu) ** 2)))
    h = float(np.exp(mu))
    return SheetMeasurement(height_m=h, sigma_m=h * sd, sigma_rand_m=h * sd_rand, sigma_sys_m=h * sd_sys,
                            ci95=(lo, hi), normal_cam=n0, center_cam=c0, paper_probs=probs,
                            paper=best if probs[best] > 0.9 else "A4|Letter", aspect=float(nominal["aspect"]),
                            angle_err_deg=float(nominal["angle_err"]), focal_post_rel=(fmu, fsd), detection=det)


def find_sheet(img_bgr: np.ndarray, intr: Intrinsics, floor_normal_cam: np.ndarray | None = None,
               n_mc: int = 600) -> SheetMeasurement | None:
    """One call for the photo/video tiers: best sheet in the image, measured, or None (no trustworthy sheet)."""
    dets = detect_sheets(img_bgr, intr, floor_normal_cam)
    return measure_sheet(dets[0], intr, n_mc) if dets else None


def plane_depth(meas: SheetMeasurement, K: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Metric depth (z) of image points uv (N,2) lying on the sheet/floor plane."""
    r = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    n = meas.normal_cam
    t = np.dot(n, meas.center_cam) / (r @ n)
    return t * r[:, 2]


def depth_map_scale(meas: SheetMeasurement, depth: np.ndarray, K: np.ndarray,
                    image_size: tuple[int, int]) -> tuple[float, float]:
    """Scale factor (metric / predicted) for a learned depth map, from the pixels inside the sheet.

    Lets a single photo's learned depth be corrected to metric by the sheet. K and image_size (w, h) describe the
    full image the sheet was detected in; depth may have any resolution covering the same field of view.
    Returns (ratio, relative 1-sigma) combining the sheet's own uncertainty with the spread of per-pixel ratios."""
    sx, sy = depth.shape[1] / image_size[0], depth.shape[0] / image_size[1]
    mask = _poly_mask(depth.shape, meas.detection.corners * [sx, sy], 0.8)
    vv, uu = np.nonzero(mask)
    uv = np.c_[(uu + 0.5) / sx - 0.5, (vv + 0.5) / sy - 0.5]
    z_true = plane_depth(meas, K, uv)
    z_pred = depth[vv, uu]
    ok = z_pred > 0
    r = z_true[ok] / z_pred[ok]
    if r.size < 5:
        return float("nan"), float("nan")
    ratio = float(np.median(r))
    spread = 1.4826 * np.median(np.abs(np.log(r / ratio))) / np.sqrt(max(r.size / 50, 1))
    return ratio, float(np.hypot(meas.sigma_m / meas.height_m, spread))


def observe_image(rgb: np.ndarray, K: np.ndarray, depth: np.ndarray, up_cam: np.ndarray | None = None,
                  rgb_full: np.ndarray | None = None, K_full: np.ndarray | None = None,
                  focal_rel_sigma: float = 0.04) -> list[dict]:
    """Photo-tier adapter (floorplan.photo.scale.sheet_cues, D-013/D-056; opt-in since D-067): the paper sheet in one
    photo -> the factor that makes this photo's learned depth metric.

    rgb, K, depth: the photo at the depth resolution (518 px long side); rgb_full, K_full: the same photo at full
    resolution, used for detection when given (an A4 sheet 2.5 m away is ~45 px long at 518 px, ~360 px at 4032 px).
    up_cam: gravity 'up' in camera coordinates, so only quads lying flat on the floor are accepted. Returns [] when no
    trustworthy sheet is seen, else [dict(scale, sigma_rel, height_m, paper, score)]."""
    img, Kd = (rgb_full, K_full) if rgb_full is not None and K_full is not None else (rgb, K)
    intr = Intrinsics(np.asarray(Kd, float), focal_rel_sigma, 0.005 * max(img.shape[:2]), "photo")
    m = find_sheet(np.ascontiguousarray(img[..., ::-1]), intr, floor_normal_cam=up_cam)
    if m is None:
        return []
    ratio, sig = depth_map_scale(m, depth, np.asarray(Kd, float), (img.shape[1], img.shape[0]))
    if not np.isfinite(ratio) or not np.isfinite(sig) or ratio <= 0:
        return []
    return [dict(scale=float(ratio), sigma_rel=float(sig), height_m=float(m.height_m), paper=m.paper,
                 score=float(m.detection.score) if m.detection is not None else None)]
