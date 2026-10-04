"""Paint synthetic but realistic water stains and cracks onto REAL surfaces of a capture, consistently in every view,
then score the damage detector against the known truth.

Why: the sample apartment has no visible damage, so recall, extent error and interval coverage can only be measured
with damage whose metric truth is known. The decal is defined in metres on a wall/ceiling and rendered into each
view by ray-plane intersection with the known pose, with a z-buffer occlusion test, so it is geometrically consistent
across views (a sticker would be).

Against the 'inverse crime' (generating and detecting with the same model):
  * the decal is rendered on a plane fitted LOCALLY to the fused scene points around it, not on the plan wall the
    detector uses; plan-vs-real wall offsets therefore show up as registration error, as they would in reality;
  * it is painted into the full views set at the views' resolution and re-compressed as JPEG; the detector picks its
    own subset of views blind;
  * shapes, sizes, colours and placements are random (seeded) and NOT tuned to the detector thresholds.
Remaining shared element (disclosed): the poses are the same ARKit poses in both steps, so pose error is not
simulated beyond what JPEG + local plane offsets add.

Usage
  python scripts/inject_damage.py make  --views outputs/damage/_views/<name> --scene outputs/<name> \
         --out outputs/damage/_inject/<name>_s0 --seed 0
  python scripts/run_damage.py --views outputs/damage/_inject/<name>_s0 --scene outputs/<name> --out <dir>
  python scripts/inject_damage.py score --truth <inject dir>/truth.json --damage <dir>/damage.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floorplan.damage.project import Surface, Views, surfaces_from_plan, surfaces_from_scene, zbuffer  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DECAL_RES = 0.002   # decal texture pixel (m)


# ----------------------------------------------------------------------------------------------------------------------
# decal textures (alpha + tint), in surface metres
# ----------------------------------------------------------------------------------------------------------------------
def make_stain(rng: np.random.Generator, size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Irregular blob with a darker tide line at its edge, yellow-brown. Returns alpha (h,w) and tint (BGR mult)."""
    n = int(size_m / DECAL_RES * 1.4)
    yy, xx = np.mgrid[0:n, 0:n] - n / 2
    r, th = np.hypot(xx, yy), np.arctan2(yy, xx)
    R = 0.5 * size_m / DECAL_RES
    k = np.arange(2, 7)
    amp = rng.uniform(0.04, 0.12, len(k)) / np.sqrt(k / 2)
    ph = rng.uniform(0, 2 * np.pi, len(k))
    edge = R * (1 + sum(a * np.cos(kk * th + p) for a, kk, p in zip(amp, k, ph)))
    inside = r < edge
    depth_in = np.clip((edge - r) / (0.03 / DECAL_RES), 0, 1)          # 0 at edge -> 1 at 3 cm inside
    noise = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), n / 12)
    noise = noise / (np.abs(noise).max() + 1e-6)
    base = rng.uniform(0.22, 0.38)
    a = base * (1 + 0.35 * noise) * (1 - 0.15 * depth_in) + 0.25 * (1 - depth_in) ** 2   # tide line at the edge
    a = np.where(inside, a, 0).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 1.0)                                # ~2 mm soft edge
    tint = np.array([rng.uniform(0.35, 0.55), rng.uniform(0.62, 0.75), rng.uniform(0.80, 0.90)], np.float32)
    return np.clip(a, 0, 0.75), tint


def make_crack(rng: np.random.Generator, length_m: float, angle_deg: float,
               width_m: float) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Random-walk polyline (main direction angle_deg in (s,t), t up) with one branch; dark grey.
    Returns alpha, tint, the polylines (metres, relative to the decal's lower-left corner) and the decal extent."""
    step = 0.005
    pts = [np.zeros(2)]
    ang = np.radians(angle_deg)
    for _ in range(int(length_m / step)):
        ang += rng.normal(0, 0.18)
        ang = 0.7 * ang + 0.3 * np.radians(angle_deg)
        pts.append(pts[-1] + step * np.array([np.cos(ang), np.sin(ang)]))
    main = np.array(pts)
    lines = [main]
    j = int(len(main) * rng.uniform(0.3, 0.7))
    bl, bang = [main[j]], ang + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
    for _ in range(int(length_m * rng.uniform(0.2, 0.4) / step)):
        bang += rng.normal(0, 0.2)
        bl.append(bl[-1] + step * np.array([np.cos(bang), np.sin(bang)]))
    lines.append(np.array(bl))
    allp = np.vstack(lines)
    lo = allp.min(0) - 0.02
    hi = allp.max(0) + 0.02
    W, H = int((hi[0] - lo[0]) / DECAL_RES) + 1, int((hi[1] - lo[1]) / DECAL_RES) + 1
    a = np.zeros((H, W), np.float32)
    for k, ln in enumerate(lines):
        px = np.c_[(ln[:, 0] - lo[0]) / DECAL_RES, (hi[1] - ln[:, 1]) / DECAL_RES]   # row 0 = top
        wpx = max(1, int(round((width_m if k == 0 else 0.6 * width_m) / DECAL_RES)))
        cv2.polylines(a, [np.round(px * 4).astype(np.int32)], False, 0.85, wpx, cv2.LINE_AA, shift=2)
    a = cv2.GaussianBlur(a, (0, 0), 0.6)
    lines_st = [ln - lo for ln in lines]          # relative to the decal's lower-left corner
    return a, np.array([0.25, 0.25, 0.27], np.float32), lines_st, (hi - lo)


def make_paper_stain(rng: np.random.Generator) -> dict:
    """v2 STAGED decal: a brown/yellow blotch painted on a sheet of white A4 paper taped flat to the wall.

    What the detector must survive: the paper is a bright, slightly BLUER (optical brighteners) rectangle with
    straight edges, hand-taped (+-5 deg tilt), with a faint contact shadow along its lower and right edges. The stain
    must be found and measured relative to the paper; the paper itself must not become a 'stain' or a 'crack'.
    Returns alpha (paper footprint), a per-pixel BGR multiplier map, the stain-only truth mask and the extent."""
    pw, ph = (0.21, 0.297) if rng.random() < 0.5 else (0.297, 0.21)
    tilt = np.radians(rng.uniform(-5, 5))
    bright = rng.uniform(1.02, 1.15)
    paper = np.array([bright * 1.04, bright * 1.01, bright * 0.98], np.float32)    # BGR: bluer than warm paint
    diag = np.hypot(pw, ph) + 0.02
    n = int(diag / DECAL_RES)
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) * DECAL_RES - diag / 2
    xr, yr = np.cos(tilt) * xx + np.sin(tilt) * yy, -np.sin(tilt) * xx + np.cos(tilt) * yy
    inside = (np.abs(xr) <= pw / 2) & (np.abs(yr) <= ph / 2)
    alpha = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 0.5)
    shadow = (~inside) & (((xr > pw / 2) & (xr < pw / 2 + 0.003) & (np.abs(yr) <= ph / 2)) |
                          ((yr > ph / 2) & (yr < ph / 2 + 0.003) & (np.abs(xr) <= pw / 2)))     # row down = lower
    tint = np.ones((n, n, 3), np.float32)
    tint[inside] = paper
    tint[shadow] = rng.uniform(0.82, 0.92)
    alpha = np.maximum(alpha, shadow.astype(np.float32))
    size = float(rng.uniform(0.08, min(pw, ph) - 0.05))
    a_s, st = make_stain(rng, size)
    k = a_s.shape[0]
    cy, cx = n // 2 + int(rng.uniform(-0.2, 0.2) * (ph - size) / DECAL_RES), n // 2 + int(rng.uniform(-0.2, 0.2) * (pw - size) / DECAL_RES)
    y0, x0 = cy - k // 2, cx - k // 2
    blot = np.zeros((n, n), np.float32)
    blot[y0:y0 + k, x0:x0 + k] = a_s
    blot *= inside
    tint *= (1 - blot[..., None] + blot[..., None] * st)
    return dict(cls="water_stain", alpha=alpha, tint=tint, truth_mask=blot > 0.1, lines=None,
                ext=np.array([n, n]) * DECAL_RES, staged="paper_stain", paper_wh=(pw, ph), tilt_deg=float(np.degrees(tilt)))


def make_tape_crack(rng: np.random.Generator) -> dict:
    """v2 STAGED decal: a jagged dark line drawn on a strip of beige masking tape stuck to the wall.

    Tape: 18-50 mm wide, 25-45 cm long, horizontal, vertical or oblique; beige = yellower and a little darker than
    the wall, with straight edges. Line: a zigzag (segments 1-2.5 cm, lateral swings up to 35% of the tape width),
    1-2.5 mm pen/marker stroke. The tape is a 'yellow, darker' strip/rectangle the stain detector will see and must
    reject without suppressing the crack drawn on it."""
    tw, tl = float(rng.uniform(0.018, 0.05)), float(rng.uniform(0.25, 0.45))
    ang = float(rng.choice([0.0, 90.0, rng.uniform(20, 60) * rng.choice([-1, 1])]))
    th = np.radians(ang)
    u, v = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])   # (s, t) with t up
    pts, x = [], -0.43 * tl
    while x < 0.43 * tl:
        pts.append(x * u + rng.uniform(-0.35, 0.35) * tw * v)
        x += rng.uniform(0.01, 0.025)
    line = np.array(pts)
    corners = np.array([sx * tl / 2 * u + sy * tw / 2 * v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    lo, hi = corners.min(0) - 0.01, corners.max(0) + 0.01
    W, H = int((hi[0] - lo[0]) / DECAL_RES) + 1, int((hi[1] - lo[1]) / DECAL_RES) + 1
    to_px = lambda q: np.c_[(q[:, 0] - lo[0]) / DECAL_RES, (hi[1] - q[:, 1]) / DECAL_RES]   # noqa: E731 row 0 = top
    tape = np.zeros((H, W), np.float32)
    cv2.fillPoly(tape, [np.round(to_px(corners) * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
    ink = np.zeros((H, W), np.float32)
    wpx = max(1, int(round(rng.uniform(0.001, 0.0025) / DECAL_RES)))
    cv2.polylines(ink, [np.round(to_px(line) * 4).astype(np.int32)], False, 0.85, wpx, cv2.LINE_AA, shift=2)
    ink = cv2.GaussianBlur(ink, (0, 0), 0.6)
    beige = np.array([rng.uniform(0.70, 0.82), rng.uniform(0.84, 0.92), rng.uniform(0.92, 0.98)], np.float32)
    tint = np.ones((H, W, 3), np.float32)
    tint[tape > 0] = beige
    tint *= (1 - ink[..., None] + ink[..., None] * np.array([0.22, 0.22, 0.26], np.float32))
    return dict(cls="crack", alpha=np.maximum(tape, ink), tint=tint, truth_mask=ink > 0.1, lines=[line - lo],
                ext=hi - lo, staged="tape_crack", tape_w=tw, tape_l=tl, angle_deg=ang)


# ----------------------------------------------------------------------------------------------------------------------
# placement and rendering
# ----------------------------------------------------------------------------------------------------------------------
def local_plane(points: np.ndarray, X: np.ndarray, n0: np.ndarray, radius: float = 0.3) -> tuple[np.ndarray, float, int]:
    """Plane fitted to fused scene points near X (within `radius` along the surface and 5 cm across it)."""
    d = points - X
    off = d @ n0
    lat = np.linalg.norm(d - off[:, None] * n0, axis=1)
    m = (lat < radius) & (np.abs(off) < 0.05)
    if m.sum() < 30:
        return n0, float(n0 @ X), int(m.sum())
    P = points[m]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2] if vt[2] @ n0 > 0 else -vt[2]
    return n, float(n @ c), int(m.sum())


def visible_views(X: np.ndarray, srf: Surface, views: Views, zbufs: dict, points: np.ndarray) -> int:
    k = 0
    for i in range(views.n):
        T, K = views.T_wc[i], views.K[i]
        h, w = views.images[i].shape[:2]
        pc = (X - T[:3, 3]) @ T[:3, :3]
        if pc[2] < 0.3:
            continue
        u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
        if not (0 <= u < w and 0 <= v < h):
            continue
        cam = T[:3, 3] - X
        dist = np.linalg.norm(cam)
        if dist > 3.5 or cam @ srf.normal / dist < 0.4:
            continue
        if i not in zbufs:
            zbufs[i] = zbuffer(points, K, T, (h, w))
        zb = zbufs[i]
        dd = zb.shape[0] / h
        z = zb[min(int(v * dd), zb.shape[0] - 1), min(int(u * dd), zb.shape[1] - 1)]
        k += bool(np.isfinite(z) and z >= pc[2] - 0.05)
    return k


def render(views: Views, srf: Surface, center_st: np.ndarray, alpha: np.ndarray, tint: np.ndarray,
           plane: tuple[np.ndarray, float], zbufs: dict, points: np.ndarray) -> int:
    """Composite the decal into every view where it is visible. Returns the number of views painted."""
    n, dpl = plane
    H, W = alpha.shape
    ext = np.array([W, H]) * DECAL_RES
    st0 = center_st - ext / 2                           # lower-left (s, t) of the decal
    corners = [srf.origin + (st0[0] + a * ext[0]) * srf.ax_s + (st0[1] + b * ext[1]) * srf.ax_t
               for a in (0, 1) for b in (0, 1)]
    painted = 0
    for i in range(views.n):
        T, K = views.T_wc[i], views.K[i]
        img = views.images[i]
        h, w = img.shape[:2]
        pc = (np.array(corners) - T[:3, 3]) @ T[:3, :3]
        if (pc[:, 2] < 0.2).any():
            continue
        u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
        v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
        u0, u1 = int(max(np.floor(u.min()), 0)), int(min(np.ceil(u.max()), w - 1))
        v0, v1 = int(max(np.floor(v.min()), 0)), int(min(np.ceil(v.max()), h - 1))
        if u1 <= u0 or v1 <= v0:
            continue
        uu, vv = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
        rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1) @ T[:3, :3].T
        denom = rays @ n
        lam = (dpl - T[:3, 3] @ n) / np.where(np.abs(denom) > 1e-6, denom, np.nan)
        Xw = T[:3, 3] + lam[..., None] * rays
        rel = Xw - srf.origin
        s, t = rel @ srf.ax_s, rel @ srf.ax_t
        mx = (s - st0[0]) / DECAL_RES
        my = (st0[1] + ext[1] - t) / DECAL_RES          # row 0 = top
        a = cv2.remap(alpha, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a[~np.isfinite(lam) | (lam <= 0)] = 0
        if i not in zbufs:
            zbufs[i] = zbuffer(points, K, T, (h, w))
        zb = zbufs[i]
        dd = zb.shape[0] / h
        zz = zb[np.clip((vv * dd).astype(int), 0, zb.shape[0] - 1), np.clip((uu * dd).astype(int), 0, zb.shape[1] - 1)]
        depth = lam * (rays @ T[:3, 2])                  # z of the plane point in the camera
        a[np.isfinite(zz) & (zz < depth - 0.05)] = 0        # occluded; z-buffer holes = nothing known in front
        if a.max() < 0.05:
            continue
        patch = img[v0:v1 + 1, u0:u1 + 1].astype(np.float32)
        if tint.ndim == 3:            # v2: per-pixel multiplier map (staged decals: paper/tape colour + drawing)
            tm = np.stack([cv2.remap(tint[..., c], mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=1.0) for c in range(3)], -1)
        else:
            tm = tint
        mult = 1 - a[..., None] + a[..., None] * tm
        img[v0:v1 + 1, u0:u1 + 1] = np.clip(patch * mult, 0, 255).astype(np.uint8)
        painted += 1
    return painted


def make(a) -> None:
    rng = np.random.default_rng(a.seed)
    views = Views.load(a.views)
    scene, info = load_scene(a.scene)
    plan_path = a.plan or next((p for p in [ROOT / "outputs" / x / Path(a.scene).name / "plan.json"
                                            for x in ("plan_alpha", "plan_beta")] if p.exists()), None)
    plan = json.loads(Path(plan_path).read_text()) if plan_path else None
    surfs = surfaces_from_plan(plan, scene) if plan else surfaces_from_scene(scene, info)
    surfs = [s for s in surfs if s.size[0] > 0.6]
    pts = scene["points"]
    zbufs: dict = {}
    truth, placed = [], []
    kinds = ["water_stain"] * a.stains + ["crack"] * a.cracks + ["paper_stain"] * a.paper_stains + \
        ["tape_crack"] * a.tape_cracks
    for j, kind in enumerate(kinds):
        cls = {"paper_stain": "water_stain", "tape_crack": "crack"}.get(kind, kind)
        for _attempt in range(300):
            srf = surfs[rng.integers(len(surfs))]
            if (cls == "crack" or kind == "paper_stain") and srf.kind != "wall":
                continue
            staged = None
            if kind in ("paper_stain", "tape_crack"):
                staged = make_paper_stain(rng) if kind == "paper_stain" else make_tape_crack(rng)
                alpha, tint, ext, lines, angle = staged["alpha"], staged["tint"], staged["ext"], staged["lines"], \
                    staged.get("angle_deg")
            elif cls == "water_stain":
                size = float(rng.uniform(0.15, 0.45))
                alpha, tint = make_stain(rng, size)
                ext = np.array(alpha.shape[::-1]) * DECAL_RES
                lines, angle = None, None
            else:
                angle = float(rng.choice([1, -1]) * rng.uniform(25, 65) + (0 if rng.random() < 0.5 else 180))
                alpha, tint, lines, ext = make_crack(rng, float(rng.uniform(0.2, 0.5)), angle,
                                                     float(rng.uniform(0.0015, 0.003)))
            margin = ext / 2 + 0.05
            if (2 * margin > np.array(srf.size)).any():
                continue
            wall_base = kind == "water_stain" and j == 0 and srf.kind == "wall"
            cs = rng.uniform(margin[0], srf.size[0] - margin[0])
            ct = margin[1] - 0.04 if wall_base else rng.uniform(margin[1], min(srf.size[1] - margin[1], 2.2))
            c = np.array([cs, ct])
            if any(p[0] == srf.id and np.linalg.norm(p[1] - c) < 0.6 for p in placed):
                continue
            X = srf.origin + cs * srf.ax_s + ct * srf.ax_t
            if any(s0 - 0.05 <= cs <= s1 + 0.05 and t0 - 0.05 <= ct <= t1 + 0.05 for s0, t0, s1, t1 in srf.holes_st):
                continue
            n, dpl, npts = local_plane(pts, X, srf.normal)
            if npts < 200:
                continue                                           # no real wall there (an opening, glass)
            if kind == "paper_stain" and np.median(scene["colors"][np.linalg.norm(pts - X, axis=1) < 0.15]
                                                   .astype(float)) < 120:
                continue      # paper is rendered as a reflectance ratio: only valid on light (painted) surfaces
            nv = visible_views(X, srf, views, zbufs, pts)
            if nv < a.min_views:
                continue
            k = render(views, srf, c, alpha, tint, (n, dpl), zbufs, pts)
            m = staged["truth_mask"] if staged else alpha > 0.1
            ys, xs = np.nonzero(m)
            st0 = c - ext / 2
            s_px = st0[0] + (xs + 0.5) * DECAL_RES
            t_px = st0[1] + ext[1] - (ys + 0.5) * DECAL_RES
            off_plane = float(dpl - n @ X)
            rec = dict(id=f"T{j + 1}", cls=cls, surface_id=srf.id, surface_kind=srf.kind, center_st=c.round(4).tolist(),
                       bbox_st=[float(s_px.min()), float(t_px.min()), float(s_px.max()), float(t_px.max())],
                       width=float(np.ptp(s_px) + DECAL_RES), height=float(np.ptp(t_px) + DECAL_RES),
                       area=float(m.sum() * DECAL_RES ** 2), views_painted=k, views_visible_at_centre=nv,
                       local_plane_offset_m=round(off_plane, 4),
                       local_plane_angle_deg=round(float(np.degrees(np.arccos(np.clip(n @ srf.normal, -1, 1)))), 2))
            if cls == "crack":
                L = sum(float(np.linalg.norm(np.diff(ln, axis=0), axis=1).sum()) for ln in lines)
                rec.update(length=L, angle_deg=angle)
            if staged:
                rec.update(staged=staged["staged"], **{k: (round(float(v), 4) if np.isscalar(v) else
                                                         [round(float(x), 4) for x in v])
                                                      for k, v in staged.items()
                                                      if k in ("paper_wh", "tilt_deg", "tape_w", "tape_l")})
            truth.append(rec)
            placed.append((srf.id, c))
            print(f"[inject] {rec['id']} {cls} on {srf.id} at s={cs:.2f} t={ct:.2f}: area {rec['area']:.4f} m2, "
                  f"{k} views painted, plane offset {off_plane*1000:.0f} mm")
            break
    out = Path(a.out)
    views.save(out)
    if plan:                     # pin the plan: other modules may rewrite plan.json (and its wall ids) at any time
        (out / "plan.json").write_text(json.dumps(plan))
    (out / "truth.json").write_text(json.dumps(dict(seed=a.seed, source_views=str(a.views), plan=str(plan_path),
                                                    items=truth), indent=1))
    print(f"[inject] wrote {len(truth)} decals to {out}")


# ----------------------------------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------------------------------
def _inside(v: float, m: dict) -> bool:
    return m["lo"] <= v <= m["hi"]


def score(truth: list[dict], dets: list[dict], tol: float = 0.05) -> dict:
    """Match each truth decal to detections on the same surface whose bbox overlaps the truth bbox (+tol)."""
    used, rows = set(), []
    for t in truth:
        s0, t0, s1, t1 = t["bbox_st"]
        cands = [d for d in dets if d["surface_id"] == t["surface_id"]
                 and d["bbox_st"][0] <= s1 + tol and d["bbox_st"][2] >= s0 - tol
                 and d["bbox_st"][1] <= t1 + tol and d["bbox_st"][3] >= t0 - tol]
        row = dict(id=t["id"], cls=t["cls"], surface_id=t["surface_id"], detected=bool(cands))
        if cands:
            same = [d for d in cands if d["cls"] == t["cls"]] or cands
            d = max(same, key=lambda x: x["area"]["value"])
            for x in cands:
                if x["cls"] == t["cls"]:      # a wrong-class detection on the decal stays a false positive
                    used.add(x["id"])
            row.update(det_id=d["id"], det_cls=d["cls"], class_ok=d["cls"] == t["cls"], fragments=len(cands),
                       confidence=d["confidence"])
            if d["cls"] == t["cls"]:
                if t["cls"] == "water_stain":
                    row.update(area_true=t["area"], area_det=d["area"]["value"],
                               area_rel_err=(d["area"]["value"] - t["area"]) / t["area"],
                               area_in_ci=_inside(t["area"], d["area"]))
                else:
                    row.update(length_true=t["length"], length_det=d["length"]["value"],
                               length_rel_err=(d["length"]["value"] - t["length"]) / t["length"],
                               length_in_ci=_inside(t["length"], d["length"]))
                row.update(width_err=d["width"]["value"] - t["width"], height_err=d["height"]["value"] - t["height"],
                           width_in_ci=_inside(t["width"], d["width"]), height_in_ci=_inside(t["height"], d["height"]),
                           centre_err=float(np.hypot(d["center_s"]["value"] - 0.5 * (t["bbox_st"][0] + t["bbox_st"][2]),
                                                     d["center_t"]["value"] - 0.5 * (t["bbox_st"][1] + t["bbox_st"][3]))))
        rows.append(row)
    fps = [d for d in dets if d["id"] not in used]
    summ = {}
    for cls in ("water_stain", "crack"):
        r = [x for x in rows if x["cls"] == cls]
        ok = [x for x in r if x.get("class_ok")]
        key = "area" if cls == "water_stain" else "length"
        summ[cls] = dict(n=len(r), recall=len([x for x in r if x["detected"]]) / max(len(r), 1),
                         class_acc=len(ok) / max(len([x for x in r if x["detected"]]), 1),
                         median_abs_rel_err=float(np.median([abs(x[f"{key}_rel_err"]) for x in ok])) if ok else None,
                         ci_coverage=float(np.mean([x[f"{key}_in_ci"] for x in ok])) if ok else None,
                         width_height_ci_coverage=float(np.mean([x["width_in_ci"] for x in ok] + [x["height_in_ci"] for x in ok])) if ok else None,
                         median_abs_width_err_m=float(np.median([abs(x["width_err"]) for x in ok])) if ok else None,
                         median_centre_err_m=float(np.median([x["centre_err"] for x in ok])) if ok else None,
                         mean_conf_tp=float(np.mean([x["confidence"] for x in ok])) if ok else None)
    return dict(per_item=rows, summary=summ, false_positives=[dict(id=d["id"], surface_id=d["surface_id"], cls=d["cls"],
                                                                   confidence=d["confidence"]) for d in fps])


def summarise(inject_dirs: list[Path], eval_dirs: list[Path], clean: list[Path],
              taus=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> dict:
    """v2: pooled injection metrics + the precision/recall trade-off of the confidence threshold.

    For every threshold tau, detections with confidence < tau are dropped, then
      recall(tau)    = injected decals still detected with the right class;
      fp_inj(tau)    = unmatched detections on the injected sets;
      fp_clean(tau)  = detections on the clean captures (every one is false: the sample has no damage).
    precision is computed over injected + clean detections pooled (clean captures cover far more wall)."""
    rows_all, sweep = [], []
    pairs = [(json.loads((i / "truth.json").read_text())["items"], json.loads((e / "damage.json").read_text())["damage"],
              e.name) for i, e in zip(inject_dirs, eval_dirs)]
    cleans = {c.name: json.loads((c / "damage.json").read_text()) for c in clean}
    for tau in taus:
        tp = {"water_stain": 0, "crack": 0}
        n = {"water_stain": 0, "crack": 0}
        fp_inj = 0
        for truth, dets, name in pairs:
            r = score(truth, [d for d in dets if d["confidence"] >= tau])
            for x in r["per_item"]:
                n[x["cls"]] += 1
                tp[x["cls"]] += bool(x.get("class_ok"))
            fp_inj += len(r["false_positives"])
            if tau == taus[0]:
                rows_all += [dict(x, run=name) for x in r["per_item"]]
        fp_clean = {k: len([d for d in v["damage"] if d["confidence"] >= tau]) for k, v in cleans.items()}
        ntp, nfp = sum(tp.values()), fp_inj + sum(fp_clean.values())
        sweep.append(dict(tau=tau, recall_stain=tp["water_stain"] / max(n["water_stain"], 1),
                          recall_crack=tp["crack"] / max(n["crack"], 1), tp=ntp, fp_injected=fp_inj,
                          fp_clean=sum(fp_clean.values()), fp_clean_by_capture=fp_clean,
                          precision=ntp / max(ntp + nfp, 1)))
    ok = [x for x in rows_all if x.get("class_ok")]
    summ = {}
    for cls, key in (("water_stain", "area"), ("crack", "length")):
        r = [x for x in rows_all if x["cls"] == cls]
        o = [x for x in ok if x["cls"] == cls]
        summ[cls] = dict(n=len(r), detected=len(o), recall=len(o) / max(len(r), 1),
                         median_abs_rel_err=float(np.median([abs(x[f"{key}_rel_err"]) for x in o])) if o else None,
                         median_signed_rel_err=float(np.median([x[f"{key}_rel_err"] for x in o])) if o else None,
                         ci_coverage=float(np.mean([x[f"{key}_in_ci"] for x in o])) if o else None,
                         width_height_ci_coverage=float(np.mean([x["width_in_ci"] for x in o] +
                                                                [x["height_in_ci"] for x in o])) if o else None,
                         median_abs_width_err_m=float(np.median([abs(x["width_err"]) for x in o])) if o else None,
                         median_centre_err_m=float(np.median([x["centre_err"] for x in o])) if o else None,
                         mean_conf_tp=float(np.mean([x["confidence"] for x in o])) if o else None,
                         missed=[f"{x['run']} {x['id']} on {x['surface_id']}" for x in r if not x.get("class_ok")])
    return dict(summary=summ, sweep=sweep, per_item=rows_all)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--views", type=Path, required=True)
    m.add_argument("--scene", type=Path, required=True)
    m.add_argument("--plan", type=Path)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--stains", type=int, default=4)
    m.add_argument("--cracks", type=int, default=3)
    m.add_argument("--min-views", type=int, default=6)
    m.add_argument("--paper-stains", type=int, default=0, help="v2 staged decal: blotch on white paper")
    m.add_argument("--tape-cracks", type=int, default=0, help="v2 staged decal: jagged line on masking tape")
    s = sub.add_parser("score")
    s.add_argument("--truth", type=Path, required=True)
    s.add_argument("--damage", type=Path, required=True)
    s.add_argument("--out", type=Path)
    sm = sub.add_parser("summary", help="v2: pooled metrics + confidence-threshold precision/recall sweep")
    sm.add_argument("--inject", type=Path, nargs="+", required=True, help="injection dirs (truth.json)")
    sm.add_argument("--eval", type=Path, nargs="+", required=True, help="matching detector output dirs")
    sm.add_argument("--clean", type=Path, nargs="*", default=[], help="detector outputs on clean captures")
    sm.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        make(a)
    elif a.cmd == "summary":
        res = summarise(a.inject, a.eval, a.clean)
        a.out.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res["summary"], indent=1, default=float))
        for r in res["sweep"]:
            print(f"tau {r['tau']:.2f}: recall stain {r['recall_stain']:.2f} crack {r['recall_crack']:.2f} "
                  f"FP injected {r['fp_injected']} clean {r['fp_clean']} precision {r['precision']:.2f}")
    else:
        t = json.loads(a.truth.read_text())["items"]
        d = json.loads(a.damage.read_text())["damage"]
        res = score(t, d)
        print(json.dumps(res["summary"], indent=1))
        print("false positives:", res["false_positives"])
        if a.out:
            a.out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
