#!/usr/bin/env python
"""Evidence for docs/FAILURE_MODES.md: how mirrors, glass, glossy floors and low light corrupt each tier.

Stages (each caches its output under outputs/failure_modes/<capture>/):
  frames    per-frame image quality (sharpness, brightness, noise, highlights), LiDAR confidence, low-confidence blobs,
            glossy-floor reflections (points below the floor), camera angular speed  -> frames.csv, frames_*.png
  geometry  phantom geometry: points seen THROUGH observed surfaces (voxel ray-march), mirror reflection test,
            points behind the plan's wall lines, low-confidence map in plan view -> geometry.json, *_plan.png
  synthetic ground-truth box room with a mirror and a glass window -> synthetic.json
  lowlight  simulated low light (photon + read noise) on sample frames -> lowlight_simulation.csv
  examples  a hall mirror and a stairwell in raw LiDAR frames -> examples.json, lidar_mirror_and_stairwell.png
  learned   photo/video-tier proxy: MoGe-2 monocular metric depth vs LiDAR on hard frames (mirror, glass, glossy
            floor), error inside vs outside low-confidence blobs -> learned.json, hard_frames.png
  summary   one JSON with the headline numbers -> outputs/failure_modes/summary.json

Run:  python scripts/analyze_failure_modes.py [--captures all] [--stages frames,geometry,learned,summary]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floorplan.io.stray import load_stray  # noqa: E402
from floorplan.pipeline.scene import load_scene  # noqa: E402
from floorplan.qa.surfaces import (VoxelOccupancy, classify_wall_gaps, depth_confidence_stats,  # noqa: E402
                                   detect_mirrors, flag_frames, below_floor_points, floor_reflection_stats, frame_quality,
                                   lowconf_blob_mask, FrameQuality, _jsonable)

DATA = ROOT.parent / "TakeHome" / "Dataset"
CAPTURES = {"single_room__c00a170fe1": DATA / "single_room" / "c00a170fe1",
            "single_scan_floor_only__1a8384c3f6": DATA / "single_scan_floor_only" / "1a8384c3f6",
            "single_scan_with_ceiling__c7d28f72c6": DATA / "single_scan_with_ceiling" / "c7d28f72c6"}
OUT = ROOT / "outputs" / "failure_modes"
# Illustration frames chosen by looking at the contact sheets (outputs/explore/*/contact_sheet.jpg); they are used
# only for the figure / learned-depth comparison, never by the detectors.
HARD_FRAMES = [("single_scan_with_ceiling__c7d28f72c6", 4547, "bathroom mirror"),
               ("single_scan_with_ceiling__c7d28f72c6", 2598, "dark glass window + mirror"),
               ("single_room__c00a170fe1", 914, "glass shower screen"),
               ("single_scan_with_ceiling__c7d28f72c6", 3897, "glossy floor + balcony glass")]


def angular_speed(T: np.ndarray, t: np.ndarray) -> np.ndarray:
    """deg/s between consecutive poses (rotation blur is the dominant blur source for hand-held capture)."""
    R = T[:, :3, :3]
    c = (np.einsum("nij,nij->n", R[:-1], R[1:]) - 1) / 2
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    dt = np.maximum(np.diff(t), 1e-3)
    w = np.r_[ang / dt, ang[-1] / dt[-1]]
    return pd.Series(w).rolling(5, center=True, min_periods=1).median().to_numpy()


# ------------------------------------------------------------------------------------------------ stage: frames
def stage_frames(name: str, stride: int) -> pd.DataFrame:
    cap = load_stray(CAPTURES[name])
    scene, info = load_scene(ROOT / "outputs" / name)
    Ta = scene["T_align"]
    T_al = np.einsum("ij,njk->nik", Ta, cap.T_wc)
    floor_y = float(info["floor_y"])
    wspeed = angular_speed(cap.T_wc, cap.timestamps)
    vspeed = np.r_[np.linalg.norm(np.diff(cap.T_wc[:, :3, 3], axis=0), axis=1) / np.maximum(np.diff(cap.timestamps), 1e-3), 0]
    rows = []
    t0 = time.time()
    for i, bgr in cap.iter_rgb(range(0, cap.n, stride)):
        q = frame_quality(bgr)
        conf = cap.confidence(i)
        d = cv2.imread(str(cap.depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        small = cv2.resize(bgr, (conf.shape[1], conf.shape[0]), interpolation=cv2.INTER_AREA)
        fr = floor_reflection_stats(d, conf, cap.K_depth(i), T_al[i], floor_y, small)
        cs = depth_confidence_stats(conf)
        rows.append(dict(frame=i, t=cap.timestamps[i] - cap.timestamps[0], **q.__dict__,
                         conf_low=cs["low"], conf_med=cs["medium"], conf_high=cs["high"],
                         lowconf_blob=float(lowconf_blob_mask(conf).mean()), median_depth=float(np.median(d)),
                         ang_speed_dps=float(wspeed[i]), lin_speed_mps=float(vspeed[i]), **fr))
    df = pd.DataFrame(rows)
    qs = [FrameQuality(**{k: r[k] for k in FrameQuality.__dataclass_fields__}) for _, r in df.iterrows()]
    w, blurry, flags = flag_frames(qs, df["t"].to_numpy())
    sh = df["sharpness"].to_numpy()
    lo = np.searchsorted(df["t"].to_numpy(), df["t"].to_numpy() - 1.0)
    hi = np.searchsorted(df["t"].to_numpy(), df["t"].to_numpy() + 1.0, side="right")
    df["sharp_ratio"] = sh / np.array([np.median(sh[a:b]) for a, b in zip(lo, hi)])
    df["weight"], df["blurry"] = w, blurry
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "frames.csv", index=False)
    (out / "frame_flags.json").write_text(json.dumps([f.to_dict() for f in flags], indent=2))
    print(f"[frames] {name}: {len(df)} frames in {time.time()-t0:.0f}s")
    return df


def fig_frames(name: str, df: pd.DataFrame):
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(df.t, df.sharp_ratio, lw=0.6, label="sharpness / median of +-1 s")
    ax[0].scatter(df.t[df.blurry], df.sharp_ratio[df.blurry], s=6, c="r", label="flagged blurry (< 0.5)")
    ax[0].set_ylabel("relative sharpness"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 2.5)
    ax[1].plot(df.t, df.ang_speed_dps, lw=0.6, c="k"); ax[1].set_ylabel("camera rotation (deg/s)")
    ax[2].plot(df.t, df.brightness, lw=0.6, label="median luma"); ax[2].plot(df.t, 10 * df.noise_sigma, lw=0.6,
                                                                          label="10 x noise sigma")
    ax[2].plot(df.t, 1000 * df.clipped_frac, lw=0.6, label="1000 x clipped fraction")
    ax[2].legend(fontsize=8); ax[2].set_ylabel("light")
    ax[3].plot(df.t, df.conf_low + df.conf_med, lw=0.6, label="LiDAR low+medium confidence")
    ax[3].plot(df.t, df.lowconf_blob, lw=0.6, label="... in blobs (not edges)")
    ax[3].plot(df.t, df.below_floor_frac, lw=0.6, label="floor rays hitting BELOW the floor")
    ax[3].legend(fontsize=8); ax[3].set_xlabel("time (s)"); ax[3].set_ylabel("fraction")
    fig.suptitle(f"{name}: per-frame quality (every {int(df.frame.diff().median())}th frame)")
    fig.tight_layout(); fig.savefig(OUT / name / "frames_timeseries.png", dpi=90); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    bins = np.array([0, 15, 30, 60, 90, 150, 1000])
    cat = pd.cut(df.ang_speed_dps, bins)
    g = df.groupby(cat, observed=True)
    ax[0].bar(range(len(g)), g.blurry.mean().values, tick_label=[f"{int(b.left)}-{int(b.right)}" for b in g.blurry.mean().index])
    ax[0].set_xlabel("camera rotation speed (deg/s)"); ax[0].set_ylabel("fraction of frames flagged blurry")
    for k, (n, v) in enumerate(zip(g.size().values, g.blurry.mean().values)):
        ax[0].text(k, v, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax[1].hist(df.brightness, bins=40); ax[1].set_xlabel("median luma (0-255)"); ax[1].set_ylabel("frames")
    ax[1].axvline(60, c="r", ls="--", label="low-light threshold"); ax[1].legend()
    ax[2].scatter(df.clipped_frac + 1e-5, df.lowconf_on_floor_frac, s=4, alpha=0.5)
    ax[2].set_xscale("log"); ax[2].set_xlabel("clipped (blown) pixel fraction"); ax[2].set_ylabel("low-conf fraction on floor rays")
    fig.suptitle(name); fig.tight_layout(); fig.savefig(OUT / name / "frames_summary.png", dpi=90); plt.close(fig)


# ------------------------------------------------------------------------------------------------ stage: geometry
def _plan_walls(name: str):
    for v in ("plan_alpha", "plan_beta"):
        p = ROOT / "outputs" / v / name / "plan.json"
        if p.exists():
            j = json.loads(p.read_text())
            walls = [dict(id=w["id"], p0=w["p0"], p1=w["p1"], normal=w["normal"]) for w in j["walls"]]
            return v, walls
    return None, []


def lowconf_plan_map(name: str, kf_step: int = 3, cell: float = 0.1):
    """Back-project ALL pixels of every kf_step-th keyframe and histogram, in plan view, how many are low-confidence
    blobs. Where the fraction is high, the sensor struggles: glass, mirrors, dark glossy panels, far see-through."""
    cap = load_stray(CAPTURES[name])
    scene, info = load_scene(ROOT / "outputs" / name)
    Ta = scene["T_align"]
    s = 4
    v, u = np.mgrid[0:192:s, 0:256:s]
    allp, flag = [], []
    for i in scene["kf"][::kf_step]:
        conf = cap.confidence(i)
        d = cv2.imread(str(cap.depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32)[v, u] / 1000
        blob = lowconf_blob_mask(conf)[v, u]
        ok = (d > 0.2) & (d < 6)
        K = cap.K_depth(i)
        z = d[ok]
        pc = np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], 1)
        T = Ta @ cap.T_wc[i]
        allp.append(pc @ T[:3, :3].T + T[:3, 3]); flag.append(blob[ok])
    P, F = np.concatenate(allp), np.concatenate(flag)
    fy = info["floor_y"]
    band = (P[:, 1] > fy + 0.1) & (P[:, 1] < fy + 2.4)          # walls/windows band, not floor/ceiling
    P, F = P[band], F[band]
    xmin, zmin = P[:, 0].min(), P[:, 2].min()
    ij = np.floor((P[:, [0, 2]] - [xmin, zmin]) / cell).astype(int)
    shape = ij.max(0) + 1
    tot = np.zeros(shape); low = np.zeros(shape)
    np.add.at(tot, (ij[:, 0], ij[:, 1]), 1); np.add.at(low, (ij[:, 0], ij[:, 1]), F)
    return tot, low, (xmin, zmin, cell), float(F.mean())


def stage_geometry(name: str) -> dict:
    t0 = time.time()
    scene, info = load_scene(ROOT / "outputs" / name)
    P = scene["raw_points"].astype(np.float64)
    O = P - scene["raw_ray"].astype(np.float64) * scene["raw_range"][:, None].astype(np.float64)
    occ = VoxelOccupancy(scene["points"], scene["normals"], voxel=0.05)
    flags, mask = detect_mirrors(P, O, occ, reference=scene["points"], max_query=400000)
    n_query = min(len(P), 400000)
    res = {"capture": name, "raw_points": int(len(P)), "queried": n_query,
           "phantom_points": int(mask.sum()), "phantom_frac": float(mask.sum() / n_query),
           "clusters": [f.to_dict() for f in flags]}
    bf_mask, bf_flags = below_floor_points(P, info["floor_y"])
    res["below_floor"] = {"n_points": int(bf_mask.sum()), "frac": float(bf_mask.mean()),
                          "flags": [f.to_dict() for f in bf_flags]}
    src, walls = _plan_walls(name)
    res["plan_source"] = src
    wall_flags = []
    if walls:
        rng = np.random.default_rng(1)
        q = np.sort(rng.choice(len(P), size=min(len(P), 600000), replace=False))
        wall_flags = classify_wall_gaps(P[q], O[q], walls, scene["points"], floor_y=info["floor_y"])
        res["wall_gaps"] = [f.to_dict() for f in wall_flags]
    tot, low, (x0, z0, cell), lowfrac = lowconf_plan_map(name)
    res["lowconf_blob_frac_wall_band"] = lowfrac
    np.savez_compressed(OUT / name / "geometry_masks.npz", phantom=mask, lc_tot=tot, lc_low=low,
                        lc_origin=np.array([x0, z0, cell]))
    (OUT / name / "geometry.json").write_text(json.dumps(_jsonable(res), indent=2))
    fig_geometry(name, scene, P, mask, flags, wall_flags, walls, tot, low, (x0, z0, cell))
    res["cluster_frames"] = fig_cluster_frames(name, scene, P, mask, flags)
    (OUT / name / "geometry.json").write_text(json.dumps(_jsonable(res), indent=2))
    print(f"[geometry] {name}: phantom {mask.sum()} / {n_query} queried, {len(flags)} clusters, "
          f"{len(wall_flags)} wall-gap flags ({time.time()-t0:.0f}s)")
    return res


KIND_COLOR = {"mirror": "#d62728", "see_through": "#1f77b4", "floor_reflection": "#9467bd",
              "glazing": "#2ca02c", "opening_or_glass_door": "#ff7f0e"}


def fig_geometry(name, scene, P, mask, flags, wall_flags, walls, tot, low, lc):
    S = scene["points"]
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    for a in ax:
        a.scatter(S[::4, 0], S[::4, 2], s=0.2, c="0.75")
        for w in walls:
            a.plot([w["p0"][0], w["p1"][0]], [w["p0"][1], w["p1"][1]], c="k", lw=1)
        a.set_aspect("equal"); a.invert_yaxis()       # I-001: v = z points DOWN for a correct top view
        a.set_xlabel("x (m)"); a.set_ylabel("z (m), drawn downward")
    ph = P[mask]
    ax[0].scatter(ph[:, 0], ph[:, 2], s=1, c="#d62728", alpha=0.5, label=f"seen through a surface ({len(ph)} pts)")
    for f in flags:
        c = f.where["crossing_uv"]
        ax[0].scatter([c[0]], [c[1]], marker="x", s=120, c=KIND_COLOR.get(f.kind, "k"))
        ax[0].annotate(f"{f.kind} {f.evidence['n_points']}p\nrefl.match {f.evidence.get('match_frac_reflected', 0):.2f}",
                       c, fontsize=7, color=KIND_COLOR.get(f.kind, "k"))
    for f in wall_flags:
        s0, s1 = f.where["segment_uv"]
        ax[0].plot([s0[0], s1[0]], [s0[1], s1[1]], lw=5, alpha=0.6, c=KIND_COLOR.get(f.kind, "k"))
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("phantom points (red) + wall-gap classes\n(green glazing, orange opening/glass door, red mirror)")
    x0, z0, cell = lc
    frac = np.where(tot >= 20, low / np.maximum(tot, 1), np.nan)
    im = ax[1].imshow(frac.T, origin="upper", extent=[x0, x0 + tot.shape[0] * cell, z0 + tot.shape[1] * cell, z0],
                      cmap="magma_r", vmin=0, vmax=0.5, alpha=0.9)
    ax[1].invert_yaxis(); ax[1].invert_yaxis()
    plt.colorbar(im, ax=ax[1], fraction=0.04, label="low-confidence blob fraction (wall band)")
    ax[1].set_title("where the LiDAR struggles (glass, mirrors, dark glossy, see-through)")
    fig.suptitle(name); fig.tight_layout(); fig.savefig(OUT / name / "geometry_plan.png", dpi=80); plt.close(fig)


def fig_cluster_frames(name, scene, P, mask, flags, top: int = 6):
    """Verification: for each phantom cluster, find the keyframe whose LiDAR depth produced it and overlay the
    phantom points on that RGB frame, so a human can see WHAT the cluster is (mirror, glass, door, noise)."""
    cap = load_stray(CAPTURES[name])
    Ta = scene["T_align"]
    ph = P[mask]
    flags = sorted(flags, key=lambda f: -f.evidence["n_points"])[:top]
    if not flags:
        return []
    picks = []
    for f in flags:
        lo, hi = np.asarray(f.where["plan_bbox"][0]), np.asarray(f.where["plan_bbox"][1])
        sel = ph[(ph[:, 0] >= lo[0]) & (ph[:, 0] <= hi[0]) & (ph[:, 2] >= lo[1]) & (ph[:, 2] <= hi[1])]
        best = (0, None, None)
        for i in scene["kf"][::2]:
            T = np.linalg.inv(Ta @ cap.T_wc[i])
            pc = sel @ T[:3, :3].T + T[:3, 3]
            K = cap.K_depth(i)
            z = pc[:, 2]
            ok = z > 0.2
            uu = np.round(K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]).astype(int)
            vv = np.round(K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]).astype(int)
            inn = (uu >= 0) & (uu < 256) & (vv >= 0) & (vv < 192)
            if inn.sum() < 20:
                continue
            d = cv2.imread(str(cap.depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            agree = np.abs(d[vv[inn], uu[inn]] - z[ok][inn]) < 0.05 * z[ok][inn]
            if agree.sum() > best[0]:
                best = (int(agree.sum()), int(i), (uu[inn][agree], vv[inn][agree]))
        picks.append((f, best))
    fig, ax = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 3.8), squeeze=False)
    out = []
    frames = {b[1]: None for _, b in picks if b[1] is not None}
    for i, bgr in cap.iter_rgb(list(frames)):
        frames[i] = cv2.resize(bgr, (256, 192), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    for a, (f, (n, i, uv)) in zip(ax[0], picks):
        a.axis("off")
        out.append({"kind": f.kind, "n_points": f.evidence["n_points"], "frame": i, "agreeing_pixels": n})
        if i is None:
            continue
        a.imshow(frames[i]); a.scatter(uv[0], uv[1], s=2, c="r", alpha=0.6)
        a.set_title(f"{f.kind} ({f.evidence['n_points']} pts)\nframe #{i}, refl {f.evidence.get('match_frac_reflected', 0):.2f}"
                    f" / ctrl {f.evidence.get('match_frac_control', 0) or 0:.2f}", fontsize=8)
    fig.suptitle(f"{name}: phantom clusters (red) on the frame that produced them")
    fig.tight_layout(); fig.savefig(OUT / name / "phantom_clusters_frames.png", dpi=90); plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ stage: learned
def stage_learned() -> dict:
    """MoGe-2 single-image metric depth on hard frames vs LiDAR (the photo/video tiers rely on learned depth)."""
    import torch
    sys.path.insert(0, str(ROOT.parent / "third_party" / "moge"))
    from huggingface_hub import hf_hub_download
    from moge.model.v2 import MoGeModel
    ckpt = hf_hub_download("Ruicheng/moge-2-vitl-normal", "model.pt", revision="cb0e8bbd6b1e243589717c78e750b1ba4c093acf")
    # the GPU is shared; 4 frames run fine on CPU, so fall back instead of waiting when < 3 GB is free
    use_gpu = torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3e9
    dev = torch.device("cuda" if use_gpu else "cpu")
    print(f"[learned] device {dev}")
    model = MoGeModel.from_pretrained(ckpt).to(dev).eval()
    caps = {}
    rows, panels = [], []
    for name, fid, label in HARD_FRAMES:
        cap = caps.setdefault(name, load_stray(CAPTURES[name]))
        bgr = next(b for _, b in cap.iter_rgb([fid]))
        rgb = cv2.resize(bgr, (518, round(518 * bgr.shape[0] / bgr.shape[1])), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        K = cap.K_rgb[fid]
        fov_x = float(np.degrees(2 * np.arctan(bgr.shape[1] / 2 / K[0, 0])))
        x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(x, num_tokens=1025, fov_x=fov_x, use_fp16=dev.type == "cuda")
        dm = out["depth"].float().cpu().numpy()
        conf = cap.confidence(fid)
        dl = cv2.imread(str(cap.depth_files[fid]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        dm_s = cv2.resize(np.nan_to_num(dm, nan=0, posinf=0), (256, 192), interpolation=cv2.INTER_AREA)
        hi = (conf == 2) & (dm_s > 0)
        blob = lowconf_blob_mask(conf)
        rel = (dm_s - dl) / np.maximum(dl, 1e-3)
        scale = float(np.median(dl[hi] / dm_s[hi]))
        rel_s = (dm_s * scale - dl) / np.maximum(dl, 1e-3)
        r = dict(capture=name, frame=fid, label=label, fov_x_deg=fov_x, lidar_over_moge_scale=scale,
                 absrel_highconf=float(np.median(np.abs(rel[hi]))),
                 absrel_highconf_after_scale=float(np.median(np.abs(rel_s[hi]))),
                 blob_frac=float(blob.mean()),
                 absrel_in_blobs=float(np.median(np.abs(rel[blob]))) if blob.any() else None,
                 frac_pixels_err_gt_10pct=float((np.abs(rel_s) > 0.10).mean()))
        rows.append(r)
        panels.append((label, fid, bgr, dl, conf, dm_s, rel_s))
        print(f"[learned] {label}: {r}")
    del model
    fig, ax = plt.subplots(len(panels), 5, figsize=(20, 3.6 * len(panels)))
    for k, (label, fid, bgr, dl, conf, dm, rel) in enumerate(panels):
        ax[k, 0].imshow(cv2.resize(bgr, (512, 384))[:, :, ::-1]); ax[k, 0].set_title(f"#{fid} {label}")
        vmax = float(np.percentile(dl, 98))
        ax[k, 1].imshow(dl, cmap="turbo", vmin=0, vmax=vmax); ax[k, 1].set_title("LiDAR depth (ARKit)")
        ax[k, 2].imshow(conf, cmap="RdYlGn", vmin=0, vmax=2); ax[k, 2].set_title("ARKit confidence (red=low)")
        ax[k, 3].imshow(dm, cmap="turbo", vmin=0, vmax=vmax); ax[k, 3].set_title("MoGe-2 metric depth (photo/video tier)")
        im = ax[k, 4].imshow(rel, cmap="coolwarm", vmin=-0.3, vmax=0.3)
        ax[k, 4].set_title("MoGe (median-scaled) - LiDAR, relative"); plt.colorbar(im, ax=ax[k, 4], fraction=0.04)
        for a in ax[k]:
            a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "hard_frames.png", dpi=70); plt.close(fig)
    (OUT / "learned.json").write_text(json.dumps(_jsonable(rows), indent=2))
    return {"rows": rows}


def stage_lowlight(name: str = "single_room__c00a170fe1", n_frames: int = 40, seed: int = 0) -> dict:
    """The sample was shot under good artificial light (no dark frames), so low light is SIMULATED to check that the
    detector responds: scale the linear-light exposure by k and add photon (Poisson) + read noise, as a phone sensor
    would at k-times less light before its auto-ISO gain brings the image back up."""
    rng = np.random.default_rng(seed)
    cap = load_stray(CAPTURES[name])
    ids = np.linspace(0, cap.n - 1, n_frames).astype(int)
    full_well = 4000.0       # photo-electrons at white for a small phone pixel at base ISO (order of magnitude)
    rows = []
    for i, bgr in cap.iter_rgb(ids):
        lin = (bgr.astype(np.float32) / 255.0) ** 2.2
        lin = cv2.resize(lin, (960, 720), interpolation=cv2.INTER_AREA)
        for k in (1, 4, 16, 64):
            for gain_back in (False, True):
                e = rng.poisson(lin * full_well / k).astype(np.float32) + rng.normal(0, 3.0, lin.shape)
                v = np.clip(e / (full_well / k if gain_back else full_well), 0, 1) ** (1 / 2.2)
                q = frame_quality((v * 255).astype(np.uint8))
                rows.append(dict(frame=int(i), light_divisor=k, auto_iso=gain_back, **q.__dict__))
    df = pd.DataFrame(rows)
    qs = [FrameQuality(**{f: r[f] for f in FrameQuality.__dataclass_fields__}) for _, r in df.iterrows()]
    _, _, _ = flag_frames(qs)
    df["low_light"] = (df.brightness < 60) | (df.noise_sigma > 3.0)
    g = df.groupby(["light_divisor", "auto_iso"]).agg(brightness=("brightness", "median"), noise=("noise_sigma", "median"),
                                                       sharpness=("sharpness", "median"),
                                                       flagged=("low_light", "mean")).reset_index()
    g.to_csv(OUT / "lowlight_simulation.csv", index=False)
    print(g.to_string())
    return g.to_dict(orient="records")


def _box_room(rng, L=4.0, W=3.0, H=2.6, n=60000):
    """Points on the 4 walls + floor of an L x W room (x in [0, L], z in [0, W]) plus a cabinet, y up, floor y=0."""
    pts = []
    for _ in range(n):
        f = rng.integers(5)
        a, b = rng.random(), rng.random()
        pts.append([(0, a * H, b * W), (L, a * H, b * W), (a * L, b * H, 0), (a * L, b * H, W), (a * L, 0, b * W)][f])
    pts = np.array(pts)
    door = (np.abs(pts[:, 2] - W) < 1e-9) & (pts[:, 0] > 2.5) & (pts[:, 0] < 3.4) & (pts[:, 1] < 2.0)
    cab = rng.random((8000, 3)) * [0.6, 0.9, 0.5] + [0.5, 0, 0.3]
    return np.vstack([pts[~door], cab])


def stage_synthetic(seed: int = 0) -> dict:
    """Ground-truth check of the wall-gap classifier, because the sample has no mirror that survives the LiDAR
    confidence filter (D-006). A box room with (a) a mirror on wall x=L and (b) a glass window with a 0.9 m sill on
    wall z=0 looking at an outside scene. Phantom points are generated physically: the reflection of real room points
    across the mirror plane, kept only if the ray from a camera to the virtual point passes through the mirror pane."""
    rng = np.random.default_rng(seed)
    L, W = 4.0, 3.0
    room = _box_room(rng, L, W)
    cams = np.c_[rng.uniform(0.8, 3.0, 400), rng.uniform(1.2, 1.6, 400), rng.uniform(0.8, 2.2, 400)]
    # (a) mirror pane on x = L: z in [1.0, 2.0], y in [0.8, 1.9]
    src = room[rng.choice(len(room), 40000)]
    virt = src.copy(); virt[:, 0] = 2 * L - src[:, 0]
    c = cams[rng.integers(len(cams), size=len(virt))]
    t = (L - c[:, 0]) / (virt[:, 0] - c[:, 0])
    hit = c + t[:, None] * (virt - c)
    ok = (t > 0) & (t < 1) & (hit[:, 2] > 1.0) & (hit[:, 2] < 2.0) & (hit[:, 1] > 0.8) & (hit[:, 1] < 1.9)
    mirror_pts, mirror_cam = virt[ok], c[ok]
    # (b) glass window on z = 0: x in [1.0, 2.5], y in [0.9, 2.1]; outside = a facade 6 m away and trees
    outside = np.c_[rng.uniform(-3, 7, 30000), rng.uniform(-2, 6, 30000), rng.uniform(-8, -2, 30000)]
    c = cams[rng.integers(len(cams), size=len(outside))]
    t = (0 - c[:, 2]) / (outside[:, 2] - c[:, 2])
    hit = c + t[:, None] * (outside - c)
    ok = (hit[:, 0] > 1.0) & (hit[:, 0] < 2.5) & (hit[:, 1] > 0.9) & (hit[:, 1] < 2.1) & (np.linalg.norm(outside - c, axis=1) < 6)
    glass_pts, glass_cam = outside[ok], c[ok]
    # (c) doorway on z = W: x in [2.5, 3.4], floor to 2.0 m, next room (floor + far wall) behind it
    nxt = np.vstack([np.c_[rng.uniform(1.5, 4.5, 20000), np.zeros(20000), rng.uniform(W + 0.15, W + 3, 20000)],
                     np.c_[rng.uniform(1.5, 4.5, 20000), rng.uniform(0, 2.6, 20000), np.full(20000, W + 3)]])
    c = cams[rng.integers(len(cams), size=len(nxt))]
    t = (W - c[:, 2]) / (nxt[:, 2] - c[:, 2])
    hit = c + t[:, None] * (nxt - c)
    ok = (hit[:, 0] > 2.5) & (hit[:, 0] < 3.4) & (hit[:, 1] < 2.0) & (np.linalg.norm(nxt - c, axis=1) < 4)
    door_pts, door_cam = nxt[ok], c[ok]
    walls = [dict(id="x0", p0=(0, W), p1=(0, 0), normal=(1, 0)), dict(id="xL", p0=(L, 0), p1=(L, W), normal=(-1, 0)),
             dict(id="z0", p0=(0, 0), p1=(L, 0), normal=(0, 1)), dict(id="zW", p0=(L, W), p1=(0, W), normal=(0, -1))]
    P = np.vstack([room, mirror_pts, glass_pts, door_pts])
    O = np.vstack([cams[rng.integers(len(cams), size=len(room))], mirror_cam, glass_cam, door_cam])
    flags = classify_wall_gaps(P, O, walls, reference=room, floor_y=0.0)
    res = {"n_mirror_phantom": int(len(mirror_pts)), "n_glass_seen_through": int(len(glass_pts)),
           "n_door_seen_through": int(len(door_pts)),
           "truth": {"mirror": "wall xL, along 1.0-2.0 m", "glazing": "wall z0, along 1.0-2.5 m, sill 0.9 m",
                     "opening": "wall zW (p0 = (4, 3)), along 0.6-1.5 m, floor to 2.0 m"},
           "flags": [{"kind": f.kind, "wall": f.evidence["wall_id"], "refl": f.evidence["match_frac_reflected"],
                      "control": f.evidence["match_frac_control"], "sill_coverage": f.evidence["sill_coverage"],
                      "along_wall_m": f.evidence["along_wall_m"]} for f in flags]}
    (OUT / "synthetic.json").write_text(json.dumps(_jsonable(res), indent=2))
    print(json.dumps(_jsonable(res), indent=1))
    return res


# illustration: a hall mirror in the LiDAR stream and a stairwell below the floor (frames picked by inspection)
EXAMPLES = [("single_scan_floor_only__1a8384c3f6", 2016, (slice(30, 110), slice(5, 70)), "hall mirror (person reflected)"),
            ("single_scan_with_ceiling__c7d28f72c6", 3936, None, "stairwell")]


def stage_examples() -> dict:
    """Mirror and below-floor evidence in the raw LiDAR frames: depth of the mirror region vs its confidence, and
    which pixels land below the floor plane."""
    from floorplan.qa.surfaces import pixel_rays_world
    out, fig = {}, plt.figure(figsize=(16, 7.5))
    for r, (name, i, roi, label) in enumerate(EXAMPLES):
        cap = load_stray(CAPTURES[name])
        sc, info = load_scene(ROOT / "outputs" / name)
        T = sc["T_align"] @ cap.T_wc[i]
        d = cv2.imread(str(cap.depth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        conf = cap.confidence(i)
        bgr = next(b for _, b in cap.iter_rgb([i]))
        im = cv2.resize(bgr, (256, 192), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
        y = T[1, 3] + d * pixel_rays_world(cap.K_depth(i), T, 192, 256)[..., 1]
        below = (y < info["floor_y"] - 0.04) & (conf == 2)
        e = {"label": label, "frame": i, "below_floor_highconf_px_frac": float(below.mean())}
        if roi is not None:
            dr, cr = d[roi], conf[roi]
            e.update(roi_median_depth_m=float(np.median(dr)), roi_conf_low=float((cr == 0).mean()),
                     roi_conf_med=float((cr == 1).mean()), roi_conf_high=float((cr == 2).mean()),
                     roi_frac_kept_by_D006=float(((cr == 2) & (dr < 4.0)).mean()),
                     frame_median_depth_m=float(np.median(d)))
        out[f"{name}#{i}"] = e
        ax = fig.add_subplot(2, 3, 3 * r + 1); ax.imshow(im); ax.set_title(f"{name[:26]} #{i}: {label}", fontsize=9)
        if roi is not None:
            ax.add_patch(plt.Rectangle((roi[1].start, roi[0].start), roi[1].stop - roi[1].start,
                                       roi[0].stop - roi[0].start, fill=False, ec="yellow", lw=2))
        ax.axis("off")
        ax = fig.add_subplot(2, 3, 3 * r + 2); h = ax.imshow(d, cmap="turbo", vmin=0, vmax=5)
        plt.colorbar(h, ax=ax, fraction=0.04, label="LiDAR depth (m)"); ax.axis("off")
        ax = fig.add_subplot(2, 3, 3 * r + 3)
        ov = np.dstack([conf == 0, conf == 2, below]).astype(float) * [0.9, 0.6, 1.0]
        ax.imshow(ov); ax.set_title("red = low confidence, green = high, blue/cyan = high-conf BELOW floor", fontsize=8)
        ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "lidar_mirror_and_stairwell.png", dpi=80); plt.close(fig)
    (OUT / "examples.json").write_text(json.dumps(_jsonable(out), indent=2))
    print(json.dumps(_jsonable(out), indent=1))
    return out


# ------------------------------------------------------------------------------------------------ stage: summary
def stage_summary(names):
    s = {}
    for n in names:
        d = OUT / n
        e = {}
        if (d / "frames.csv").exists():
            df = pd.read_csv(d / "frames.csv")
            e["frames"] = {
                "n": len(df), "blurry_frac": float(df.blurry.mean()),
                "blurry_frac_rot_lt30": float(df.blurry[df.ang_speed_dps < 30].mean()),
                "blurry_frac_rot_gt90": float(df.blurry[df.ang_speed_dps > 90].mean()) if (df.ang_speed_dps > 90).any() else None,
                "spearman_sharp_ratio_vs_rot": float(df.sharp_ratio.corr(df.ang_speed_dps, method="spearman")),
                "median_brightness": float(df.brightness.median()), "p10_brightness": float(df.brightness.quantile(0.1)),
                "median_noise_sigma": float(df.noise_sigma.median()),
                "frames_with_clipped_gt_1pct": float((df.clipped_frac > 0.01).mean()),
                "lidar_low_med_conf_frac_mean": float((df.conf_low + df.conf_med).mean()),
                "lidar_lowconf_blob_frac_mean": float(df.lowconf_blob.mean()),
                "lidar_lowconf_blob_frac_p95": float(df.lowconf_blob.quantile(0.95)),
                "floor_rays_below_floor_mean": float(df.below_floor_frac.mean()),
                "floor_rays_below_floor_p95": float(df.below_floor_frac.quantile(0.95)),
                "floor_rays_below_floor_highconf_mean": float(df.below_floor_highconf_frac.mean()),
                "lowconf_on_floor_mean": float(df.lowconf_on_floor_frac.mean()),
                "lowconf_on_floor_when_clipped": float(df.lowconf_on_floor_frac[df.clipped_on_floor_frac > 0.005].mean())
                if (df.clipped_on_floor_frac > 0.005).any() else None,
                "lowconf_on_floor_when_not_clipped": float(df.lowconf_on_floor_frac[df.clipped_on_floor_frac <= 0.005].mean()),
            }
        if (d / "geometry.json").exists():
            g = json.loads((d / "geometry.json").read_text())
            e["geometry"] = {k: g[k] for k in ("raw_points", "queried", "phantom_points", "phantom_frac",
                                                "plan_source", "lowconf_blob_frac_wall_band", "below_floor")}
            e["geometry"]["cluster_frames"] = g.get("cluster_frames")
            e["geometry"]["clusters"] = [{"kind": c["kind"], "n": c["evidence"]["n_points"],
                                          "refl": c["evidence"].get("match_frac_reflected"),
                                          "control": c["evidence"].get("match_frac_control"),
                                          "plane": c["evidence"].get("crossing_plane"),
                                          "uv": c["where"]["crossing_uv"]} for c in g["clusters"]]
            e["geometry"]["wall_gaps"] = [{"kind": c["kind"], "wall": c["evidence"]["wall_id"],
                                           "n": c["evidence"]["n_points"], "refl": c["evidence"]["match_frac_reflected"],
                                           "control": c["evidence"].get("match_frac_control"),
                                           "sill_coverage": c["evidence"]["sill_coverage"]}
                                          for c in g.get("wall_gaps", [])]
        s[n] = e
    if (OUT / "synthetic.json").exists():
        s["synthetic_mirror_glass"] = json.loads((OUT / "synthetic.json").read_text())
    if (OUT / "lowlight_simulation.csv").exists():
        s["lowlight_simulation"] = pd.read_csv(OUT / "lowlight_simulation.csv").to_dict(orient="records")
    if (OUT / "examples.json").exists():
        s["lidar_examples"] = json.loads((OUT / "examples.json").read_text())
    if (OUT / "learned.json").exists():
        s["learned_depth_on_hard_frames"] = json.loads((OUT / "learned.json").read_text())
    (OUT / "summary.json").write_text(json.dumps(_jsonable(s), indent=2))
    print(json.dumps(_jsonable(s), indent=1)[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="all")
    ap.add_argument("--stages", default="frames,geometry,synthetic,lowlight,examples,learned,summary")
    ap.add_argument("--stride", type=int, default=4, help="analyse every n-th frame in the frames stage")
    a = ap.parse_args()
    names = list(CAPTURES) if a.captures == "all" else [n for n in CAPTURES if any(c in n for c in a.captures.split(","))]
    stages = a.stages.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    for n in names:
        (OUT / n).mkdir(parents=True, exist_ok=True)
        if "frames" in stages:
            fig_frames(n, stage_frames(n, a.stride))
        if "geometry" in stages:
            stage_geometry(n)
    if "synthetic" in stages:
        stage_synthetic()
    if "lowlight" in stages:
        stage_lowlight()
    if "examples" in stages:
        stage_examples()
    if "learned" in stages:
        stage_learned()
    if "summary" in stages:
        stage_summary(names)


if __name__ == "__main__":
    main()
