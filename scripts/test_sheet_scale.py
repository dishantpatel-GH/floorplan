"""Blind test of the paper-sheet scale reference (floorplan/scale) on REAL sample frames with a RENDERED sheet.
The cue is optional and off by default since D-067 (no reference object in the protocol); this test covers the opt-in.

Why synthetic sheets: no real photo with a sheet exists (the own capture has none either, D-067). The sample captures give
real images, real lighting, real floors (wood, white tiles, glossy surfaces) and an exactly known camera pose,
intrinsics and floor plane (LiDAR). We render a physically placed A4/Letter sheet onto the floor through that camera
model, so the true camera-to-floor distance is known exactly, then run the detector blind (it only sees pixels and an
optionally perturbed K) and score:
  * detection rate vs distance/contrast, height (= scale) error, 95%-interval coverage (calibration);
  * false positives on frames WITHOUT a sheet (cabinets, door panels, tiles, light patches) and on rendered
    hard negatives (square tiles/notes, 2:1 rectangles, printed pages with text);
  * curled and partially occluded sheets: must be rejected or still measured correctly (never confidently wrong);
  * fuse_scale on a simulated room: sheet + priors (+ a deliberately wrong cue) -> outlier rejection and coverage.

Usage: python scripts/test_sheet_scale.py [--per-capture 45] [--neg-per-capture 60]
Outputs: outputs/scale_reference/{results.json, cases.csv, *.png}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floorplan.io.stray import load_stray  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402
from floorplan.scale import Intrinsics, detect_sheets, measure_sheet, fuse_scale  # noqa: E402
from floorplan.scale.fuse import camera_height_prior, sheet_cue, ScaleCue  # noqa: E402

DATA = ROOT.parent / "TakeHome" / "Dataset"
CAPTURES = {"single_room__c00a170fe1": "single_room/c00a170fe1",
            "single_scan_floor_only__1a8384c3f6": "single_scan_floor_only/1a8384c3f6",
            "single_scan_with_ceiling__c7d28f72c6": "single_scan_with_ceiling/c7d28f72c6"}
OUT = ROOT / "outputs" / "scale_reference"
SS = 4                                    # supersampling factor for anti-aliased edges


# ----------------------------------------------------------------------------------------------- rendering
def paper_texture(rng, size_px=(420, 594)) -> np.ndarray:
    """Relative paper reflectance: 1 + fibre noise (blurred) + a faint linear shading gradient."""
    h, w = size_px
    fib = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5)
    fib = 0.008 * fib / (fib.std() + 1e-6)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    a, b = rng.normal(0, 0.03, 2)
    return 1.0 + fib + a * (gx / w - 0.5) + b * (gy / h - 0.5)


def sheet_points(L: float, W: float, nx: int, ny: int, curl: float = 0.0, rng=None) -> np.ndarray:
    """Sheet grid in its local frame (x along length, z along width, y up). curl lifts one end (non-planar)."""
    x, z = np.meshgrid(np.linspace(-L / 2, L / 2, nx), np.linspace(-W / 2, W / 2, ny))
    y = np.zeros_like(x)
    if curl > 0:
        t = np.clip((x - L / 6) / (L / 3), 0, None)                       # last third of the sheet curls up
        y = curl * t ** 2
    return np.stack([x, y, z], -1)


def project(P: np.ndarray, T_cw: np.ndarray, K: np.ndarray) -> np.ndarray:
    X = P @ T_cw[:3, :3].T + T_cw[:3, 3]
    return np.stack([K[0, 0] * X[..., 0] / X[..., 2] + K[0, 2], K[1, 1] * X[..., 1] / X[..., 2] + K[1, 2], X[..., 2]], -1)


def render_mesh(img: np.ndarray, uvz: np.ndarray, color_grid: np.ndarray,
                shade: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Composite a coloured grid mesh (ny, nx) into img with 4x supersampled, anti-aliased edges.

    shade (optional, image-sized): per-pixel multiplier applied after rasterisation (smooth illumination), so the
    lighting does not inherit the mesh's cell structure."""
    h, w = img.shape[:2]
    uv = uvz[..., :2]
    x0, y0 = np.floor(uv.reshape(-1, 2).min(0)).astype(int) - 3
    x1, y1 = np.ceil(uv.reshape(-1, 2).max(0)).astype(int) + 3
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    rw, rh = x1 - x0, y1 - y0
    col = np.zeros((rh * SS, rw * SS, 3), np.float32)
    alpha = np.zeros((rh * SS, rw * SS), np.float32)
    # pixel-centre convention: low-res coordinate x (centre of pixel x) sits at high-res x*SS + (SS-1)/2
    p = ((uv - [x0, y0]) * SS + (SS - 1) / 2).astype(np.float32)
    ny, nx = uv.shape[:2]
    # paint far cells first so the near (lifted) part of a curl occludes correctly
    order = np.argsort(-uvz[:-1, :-1, 2].ravel())
    for idx in order:
        j, i = divmod(idx, nx - 1)
        poly = np.array([p[j, i], p[j, i + 1], p[j + 1, i + 1], p[j + 1, i]])
        pi = np.round(poly * 16).astype(np.int32)                         # 4 fractional bits for sub-pixel accuracy
        cv2.fillConvexPoly(col, pi, color_grid[j, i].tolist(), cv2.LINE_8, 4)
        cv2.fillConvexPoly(alpha, pi, 1.0, cv2.LINE_8, 4)
    col = cv2.resize(col, (rw, rh), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (rw, rh), interpolation=cv2.INTER_AREA)[..., None]
    roi = img[y0:y1, x0:x1].astype(np.float32)
    a_safe = np.maximum(a, 1e-6)
    paper = col / a_safe
    if shade is not None:
        paper = paper * shade[y0:y1, x0:x1, None]
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(roi * (1 - a) + np.minimum(paper, 252) * a * (a > 0), 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.float32)
    mask[y0:y1, x0:x1] = a[..., 0]
    return out, mask


def illuminant_white(img: np.ndarray) -> np.ndarray:
    """Colour of white under the scene light: median of the brightest unsaturated pixels, normalised."""
    g = img.astype(np.float32).mean(2)
    sel = (g > np.percentile(g, 97)) & (img.max(2) < 250)
    wcol = np.median(img[sel].astype(np.float32), 0) if sel.sum() > 20 else np.array([1, 1, 1.0])
    return wcol / wcol.mean()


def degrade(img: np.ndarray, rng, blur: float) -> np.ndarray:
    """Optics blur + sensor noise + JPEG re-encode, applied to the whole frame so the sheet is not 'too clean'."""
    out = cv2.GaussianBlur(img, (0, 0), blur) if blur > 0.05 else img
    out = np.clip(out.astype(np.float32) + rng.normal(0, 1.5, out.shape), 0, 255).astype(np.uint8)
    q = int(rng.integers(82, 94))
    return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


# ----------------------------------------------------------------------------------------------- placement
def floor_mask(depth: np.ndarray, Kd: np.ndarray, T_aw: np.ndarray, floor_y: float, tol: float = 0.03) -> np.ndarray:
    """Depth pixels whose LiDAR point lies on the floor plane (within tol): where a sheet could really lie."""
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    z = depth
    X = np.stack([(u - Kd[0, 2]) * z / Kd[0, 0], (v - Kd[1, 2]) * z / Kd[1, 1], z], -1)
    y = X @ T_aw[1, :3] + T_aw[1, 3]
    return (z > 0) & (np.abs(y - floor_y) < tol)


def place_sheet(rng, T_aw: np.ndarray, K: np.ndarray, size, floor_y: float, depth: np.ndarray, Kd: np.ndarray,
                dist_range=(0.8, 4.5), curl=0.0):
    """Random sheet pose on the real floor, fully visible, where LiDAR confirms the floor is not occluded."""
    T_cw = np.linalg.inv(T_aw)
    h, w = 2 * K[1, 2], 2 * K[0, 2]
    C = T_aw[:3, 3]
    # candidate centres: depth pixels where LiDAR confirms the floor plane (so the sheet lies on visible floor)
    fm = floor_mask(depth, Kd, T_aw, floor_y)
    vv, uu = np.nonzero(fm)
    if len(uu) < 30:
        return None
    sel = rng.choice(len(uu), min(600, len(uu)), replace=False)
    uvs = np.c_[(uu[sel] + 0.5) * K[0, 2] * 2 / depth.shape[1] - 0.5, (vv[sel] + 0.5) * K[1, 2] * 2 / depth.shape[0] - 0.5,
                np.ones(len(sel))]
    rays = (uvs @ np.linalg.inv(K).T) @ T_aw[:3, :3].T
    t = (floor_y - C[1]) / rays[:, 1]
    dist = t * np.linalg.norm(rays, axis=1)
    valid = (dist >= dist_range[0]) & (dist <= dist_range[1])
    if not valid.any():
        return None
    # uniform target distance over what this view can see (uniform pixels would cluster around ~2 m)
    r_target = rng.uniform(dist[valid].min(), dist[valid].max())
    order = np.argsort(np.where(valid, np.abs(dist - r_target), np.inf))[:40]
    for j in order[valid[order]]:
        P = C + t[j] * rays[j]
        yaw = rng.uniform(0, np.pi)
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        G = sheet_points(*size, 49, 35, curl) @ Ry.T + P + [0, 0.0, 0]
        G[..., 1] += floor_y - P[1]
        uvz = project(G, T_cw, K)
        if (uvz[..., 2] < 0.2).any():
            continue
        if not ((uvz[..., 0] > 15).all() and (uvz[..., 0] < w - 15).all() and (uvz[..., 1] > 15).all()
                and (uvz[..., 1] < h - 15).all()):
            continue
        # LiDAR check: depth at sheet pixels must agree with the floor plane (floor actually visible there)
        uvd = project(G[::3, ::3].reshape(-1, 3), T_cw, Kd)
        ui, vi = np.round(uvd[:, 0]).astype(int), np.round(uvd[:, 1]).astype(int)
        ok = (ui >= 0) & (ui < depth.shape[1]) & (vi >= 0) & (vi < depth.shape[0])
        if ok.mean() < 1:
            continue
        dz = depth[vi, ui]
        if (dz > 0).mean() < 0.8 or np.percentile(np.abs(dz[dz > 0] - uvd[dz > 0, 2]), 90) > 0.04:
            continue
        return G, uvz
    return None


# ----------------------------------------------------------------------------------------------- experiments
def scene_frames(name: str, n: int, rng) -> tuple:
    scene, info = load_scene(ROOT / "outputs" / name)
    cap = load_stray(DATA / CAPTURES[name])
    kf = scene["kf"]
    idx = np.sort(rng.choice(kf, size=min(n, len(kf)), replace=False))
    return scene, info, cap, idx


def floor_frames(scene, info, cap, n: int, rng, min_frac: float = 0.12) -> np.ndarray:
    """Keyframes that see enough floor for a sheet (LiDAR check), so positives are physically placeable."""
    kf = rng.permutation(scene["kf"])
    out = []
    for i in kf:
        T_aw = scene["T_align"] @ scene["T_wc"][i]
        if floor_mask(cap.depth(int(i)), cap.K_depth(int(i)), T_aw, info["floor_y"]).mean() >= min_frac:
            out.append(int(i))
        if len(out) >= n:
            break
    return np.sort(np.array(out, int))


def run_positive(rng, img, T_aw, K, floor_y, depth, Kd, kind: str):
    """Render one sheet case. kind: flat | curled | occluded | negsquare | neg2to1 | negtext."""
    paper = "A4" if rng.random() < 0.6 else "Letter"
    size = {"A4": (0.297, 0.210), "Letter": (0.2794, 0.2159)}[paper]
    if kind == "negsquare":
        size = (0.30, 0.30)
    elif kind == "neg2to1":
        size = (0.40, 0.20)
    curl = rng.uniform(0.03, 0.07) if kind == "curled" else 0.0
    pl = place_sheet(rng, T_aw, K, size, floor_y, depth, Kd, curl=curl)
    if pl is None:
        return None
    G, uvz = pl
    whitec = illuminant_white(img)
    m0 = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m0, np.round(uvz[[0, 0, -1, -1], [0, -1, -1, 0], :2]).astype(np.int32), 1)
    # Paper brightness follows the ILLUMINATION on the floor (shadows, light patches): paper = floor * albedo ratio.
    # The floor is blurred so its own texture (grain, grout) does not print onto the paper.
    lum8 = np.clip(img.astype(np.float32).mean(2), 0, 255).astype(np.uint8)
    illum = cv2.GaussianBlur(cv2.medianBlur(lum8, 31).astype(np.float32), (0, 0), 4)
    floor_lum = float(np.median(img[m0.astype(bool)].astype(np.float32).mean(1)))
    gain = gain_raw = rng.uniform(1.04, 1.7)
    tex = paper_texture(rng, (G.shape[0] - 1, G.shape[1] - 1))
    cols = tex[..., None] * whitec[None, None]
    paper_lum = np.minimum(illum[m0.astype(bool)] * gain, 252.0)
    gain = float(np.median(paper_lum) / max(floor_lum, 1.0))               # effective contrast after clipping
    if curl > 0:                                                           # lifted part is lit differently
        lift = (G[:-1, :-1, 1] - G[:-1, :-1, 1].min()) / max(curl, 1e-6)
        cols *= (1 - 0.15 * lift)[..., None]
    if kind == "negtext":
        cols *= np.where(rng.random(cols.shape[:2]) < 0.35, 0.35, 1.0)[..., None]   # dense printed text
    out, mask = render_mesh(img, uvz, cols, shade=illum * gain_raw)
    if kind == "occluded":
        k = rng.integers(4)
        corner = uvz[[0, 0, -1, -1][k], [0, -1, -1, 0][k], :2]
        rad = 0.35 * np.linalg.norm(uvz[0, 0, :2] - uvz[-1, -1, :2])
        dark = (rng.uniform(20, 90, 3)).tolist()
        cv2.ellipse(out, tuple(np.round(corner).astype(int)), (int(rad), int(rad * 0.6)), float(rng.uniform(0, 180)),
                    0, 360, dark, -1, cv2.LINE_AA)
    out = degrade(out, rng, float(rng.uniform(0.3, 1.1)))
    T_cw = np.linalg.inv(T_aw)
    h_true = float(T_aw[1, 3] - floor_y)
    floor_n_cam = T_cw[:3, :3] @ np.array([0, 1.0, 0])
    corners_true = uvz[[0, 0, -1, -1], [0, -1, -1, 0], :2]
    return dict(img=out, h_true=h_true, paper=paper, contrast_gain=gain, floor_lum=floor_lum,
                dist=float(np.linalg.norm(G.reshape(-1, 3).mean(0) - T_aw[:3, 3])), corners_true=corners_true,
                floor_n_cam=floor_n_cam, kind=kind)


def evaluate_case(case, K, rng, focal_sigma: float):
    """Run the detector blind with a perturbed focal (the error EXIF focal would have) and score it."""
    eps = rng.normal(0, focal_sigma) if focal_sigma > 0 else 0.0
    Kp = K.copy()
    Kp[0, 0] *= np.exp(eps)
    Kp[1, 1] *= np.exp(eps)
    intr = Intrinsics(Kp, max(focal_sigma, 0.005), 0.0, "test")
    t0 = time.time()
    dets = detect_sheets(case["img"], intr)
    dt = time.time() - t0
    row = dict(kind=case["kind"], paper=case["paper"], dist=case["dist"], gain=case["contrast_gain"],
               h_true=case["h_true"], focal_err=eps, detected=len(dets) > 0, runtime_s=dt)
    if dets:
        # a detection only counts as the sheet if it overlaps the true sheet (else it is a false positive)
        d = dets[0]
        err_px = np.linalg.norm(d.corners[:, None] - case["corners_true"][None], axis=-1).min(1).max()
        size = np.linalg.norm(case["corners_true"][0] - case["corners_true"][2])
        row["on_sheet"] = bool(err_px < 0.25 * size)
        meas = measure_sheet(d, intr)
        row.update(h_est=meas.height_m, sigma=meas.sigma_m, sigma_rand=meas.sigma_rand_m, ci_lo=meas.ci95[0],
                   ci_hi=meas.ci95[1], paper_est=meas.paper, p_true_paper=meas.paper_probs.get(case["paper"], 0),
                   corner_err_px=float(err_px), score=d.score, n_dets=len(dets), meas=meas)
        row["rel_err"] = meas.height_m / case["h_true"] - 1
        row["covered"] = bool(meas.ci95[0] <= case["h_true"] <= meas.ci95[1])
    return row


def run_negatives(img, K):
    dets, rej = detect_sheets(img, Intrinsics(K, 0.03), return_rejected=True)
    return dets, rej


# ----------------------------------------------------------------------------------------------- fusion test
def fusion_test(rows: list[dict], rng) -> dict:
    """Simulated rooms: an up-to-scale reconstruction with unknown scale s_true; cues = 2-4 sheet photos (same
    camera, shared focal error), a camera-height prior, and in half the rooms one wrong cue (e.g. a mis-detection
    or a failed learned-depth estimate, 25% off). Checks coverage of the fused 95% interval and the rejection."""
    good = [r for r in rows if r["kind"] == "flat" and r.get("on_sheet")]
    if len(good) < 4:
        return dict(n=0, note="too few detected sheets")
    res = []
    for k in range(300):
        s_true = float(np.exp(rng.uniform(-1, 1)))
        picks = rng.choice(len(good), size=int(rng.integers(2, 5)), replace=False)
        cues = []
        for i in picks:
            r = good[i]
            h_recon = r["h_true"] / s_true
            cues.append(sheet_cue(r["meas"], h_recon, group="camera"))
        h_cam_true = rng.uniform(1.2, 1.65)                                   # the person's real chest height
        cues.append(camera_height_prior(h_cam_true / s_true))
        bad = k % 2 == 0
        if bad:
            cues.append(ScaleCue("learned_depth_fail", "learned_depth", s_true * 1.25, 0.03))
        s, sig, rep = fuse_scale(cues)
        res.append(dict(rel_err=s / s_true - 1, rel_sigma=sig / s, covered=abs(np.log(s / s_true)) <= 1.96 * sig / s,
                        bad=bad, bad_rejected=bad and any(c["name"] == "learned_depth_fail" and not c["used"]
                                                          for c in rep["cues"])))
    e = np.array([r["rel_err"] for r in res])
    return dict(n=len(res), median_abs_err_pct=float(100 * np.median(np.abs(e))),
                p95_abs_err_pct=float(100 * np.percentile(np.abs(e), 95)),
                median_rel_sigma_pct=float(100 * np.median([r["rel_sigma"] for r in res])),
                coverage95=float(np.mean([r["covered"] for r in res])),
                bad_cue_rejected=float(np.mean([r["bad_rejected"] for r in res if r["bad"]])))


# ----------------------------------------------------------------------------------------------- main
def run_capture(name: str, args, seed: int):
    """All cases for one capture (runs in its own process; the captures are independent)."""
    cv2.setNumThreads(1)
    rng = np.random.default_rng(seed)
    rows, negs, examples = [], [], []
    kinds = ["flat"] * 6 + ["curled", "occluded", "negsquare", "neg2to1", "negtext"]
    scene, info, cap, neg = scene_frames(name, args.neg_per_capture, rng)
    pos = floor_frames(scene, info, cap, args.per_capture, rng)
    floor_y, T_align = info["floor_y"], scene["T_align"]
    neg_idx = set(neg.tolist()) - set(pos.tolist())
    for i, img in cap.iter_rgb(np.union1d(pos, neg)):
        K = cap.K_rgb[i]
        if i in neg_idx:
            dets, _ = run_negatives(img, K)
            negs.append(dict(capture=name, frame=int(i), n_false=len(dets),
                             scores=[round(d.score, 3) for d in dets]))
            if dets:
                examples.append(("false positive", img, dets, None))
            continue
        T_aw = T_align @ scene["T_wc"][i]
        depth, kd = cap.depth(i), cap.K_depth(i)
        for kind in rng.choice(kinds, 2):
            case = run_positive(rng, img, T_aw, K, floor_y, depth, kd, str(kind))
            if case is None:
                continue
            row = evaluate_case(case, K, rng, args.focal_sigma)
            row.update(capture=name, frame=int(i))
            rows.append(row)
            if row["kind"] != "flat" or rng.random() < 0.25 or not row.get("on_sheet", True):
                examples.append((row["kind"], case["img"], None, row))
    print(f"[{name}] cases: {len(rows)}, negative frames: {len(negs)}", flush=True)
    return rows, negs, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-capture", type=int, default=45)
    ap.add_argument("--neg-per-capture", type=int, default=60)
    ap.add_argument("--focal-sigma", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(len(CAPTURES)) as ex:
        futs = [ex.submit(run_capture, name, args, args.seed * 100 + k) for k, name in enumerate(CAPTURES)]
        parts = [f.result() for f in futs]
    rows = [r for p in parts for r in p[0]]
    negs = [n for p in parts for n in p[1]]
    examples = [e for p in parts for e in p[2]]
    rng = np.random.default_rng(args.seed + 7)
    summary = summarise(rows, negs)
    summary["fusion"] = fusion_test(rows, rng)
    summary["focal_sigma_used"] = args.focal_sigma
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    with open(OUT / "cases.csv", "w", newline="") as f:
        keys = ["capture", "frame", "kind", "paper", "dist", "gain", "h_true", "focal_err", "detected", "on_sheet",
                "h_est", "sigma", "sigma_rand", "ci_lo", "ci_hi", "rel_err", "covered", "paper_est", "p_true_paper",
                "corner_err_px", "score", "runtime_s"]
        wr = csv.DictWriter(f, keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    import pickle
    with open(OUT / "rows.pkl", "wb") as f:                                # for re-analysis without re-running
        pickle.dump([{k: v for k, v in r.items()} for r in rows], f)
    figures(rows, examples)
    print(json.dumps({k: v for k, v in summary.items() if k != "negative_frames"}, indent=2))
    print(json.dumps(summary["negative_frames"], indent=1)[:3000])


def summarise(rows, negs) -> dict:
    def block(rs):
        det = [r for r in rs if r["detected"]]
        on = [r for r in det if r.get("on_sheet")]
        e = np.array([r["rel_err"] for r in on]) if on else np.array([np.nan])
        z = np.array([abs(np.log1p(r["rel_err"])) / (r["sigma"] / r["h_est"]) for r in on]) if on else np.array([np.nan])
        return dict(n=len(rs), detection_rate=len(on) / max(len(rs), 1), wrong_location=len(det) - len(on),
                    median_abs_err_pct=float(100 * np.nanmedian(np.abs(e))),
                    p95_abs_err_pct=float(100 * np.nanpercentile(np.abs(e), 95)) if on else None,
                    bias_pct=float(100 * np.nanmedian(e)),
                    median_rel_sigma_pct=float(100 * np.nanmedian([r["sigma"] / r["h_est"] for r in on])) if on else None,
                    coverage95=float(np.mean([r["covered"] for r in on])) if on else None,
                    z_rms=float(np.sqrt(np.nanmean(z ** 2))),
                    paper_type_acc=float(np.mean([r["paper_est"] == r["paper"] for r in on])) if on else None,
                    paper_ambiguous=float(np.mean([r["paper_est"] == "A4|Letter" for r in on])) if on else None)
    out = {k: block([r for r in rows if r["kind"] == k]) for k in sorted({r["kind"] for r in rows})}
    flat = [r for r in rows if r["kind"] == "flat"]
    for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 9)):
        out[f"flat_dist_{lo}-{hi}m"] = block([r for r in flat if lo <= r["dist"] < hi])
    for lo, hi in ((1.0, 1.15), (1.15, 1.35), (1.35, 9)):
        out[f"flat_contrast_{lo}-{hi}"] = block([r for r in flat if lo <= r["gain"] < hi])
    out["negative_frames"] = dict(n=len(negs), frames_with_false_positive=int(sum(n["n_false"] > 0 for n in negs)),
                                  false_positive_rate=float(np.mean([n["n_false"] > 0 for n in negs])),
                                  details=[n for n in negs if n["n_false"]])
    out["runtime_s_median"] = float(np.median([r["runtime_s"] for r in rows]))
    return out


def figures(rows, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    flat = [r for r in rows if r["kind"] == "flat" and r.get("on_sheet")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    d = np.array([r["dist"] for r in flat])
    e = 100 * np.array([r["rel_err"] for r in flat])
    s = 100 * np.array([r["sigma"] / r["h_est"] for r in flat])
    ax[0].errorbar(d, e, yerr=1.96 * s, fmt="o", ms=3, alpha=0.6, elinewidth=0.8)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(xlabel="camera-to-sheet distance (m)", ylabel="height (scale) error %", title="error with 95% intervals")
    z = e / s
    ax[1].hist(z, bins=np.linspace(-5, 5, 41), density=True, alpha=0.7)
    xx = np.linspace(-5, 5, 200)
    ax[1].plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "k")
    ax[1].set(xlabel="error / reported sigma", title=f"calibration (cov95 = {np.mean(np.abs(z) < 1.96):.2f})")
    allf = [r for r in rows if r["kind"] == "flat"]
    bins = np.array([0.5, 1.5, 2.5, 3.5, 5])
    rate = [np.mean([r.get("on_sheet", False) for r in allf if lo <= r["dist"] < hi] or [np.nan])
            for lo, hi in zip(bins[:-1], bins[1:])]
    ax[2].bar((bins[:-1] + bins[1:]) / 2, rate, width=0.8)
    ax[2].set(xlabel="distance (m)", ylabel="detection rate", ylim=(0, 1.05), title="detection rate (flat sheets)")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy.png", dpi=110)
    plt.close(fig)
    # example gallery: crops around the (true or detected) sheet with the detected outline
    tiles = []
    for kind, img, dets, row in examples[:24]:
        vis = img.copy()
        if row is not None and row.get("detected"):
            q = row["meas"].detection.corners
            cv2.polylines(vis, [np.round(q).astype(np.int32)], True, (0, 0, 255), 3)
        if dets:
            for dd in dets:
                cv2.polylines(vis, [np.round(dd.corners).astype(np.int32)], True, (0, 0, 255), 3)
        label = kind if row is None else (f"{kind} d={row['dist']:.1f}m g={row['gain']:.2f} " +
                                          (f"err={100 * row['rel_err']:+.1f}%" if row.get("on_sheet") else
                                           "REJECTED" if not row["detected"] else "WRONG"))
        vis = cv2.resize(vis, (480, 360))
        cv2.putText(vis, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(vis)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(OUT / "examples.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
