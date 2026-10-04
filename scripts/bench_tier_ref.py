#!/usr/bin/env python
"""Reference-based scoring of a video- or photo-tier plan against the LiDAR plan of the SAME capture.

The sample captures have no tape or laser ground truth. The LiDAR plan of the same capture is the best available
reference (ARKitScenes puts LiDAR room dimensions within ~1 cm of a Faro laser, docs/modules/
benchmark_arkitscenes_accuracy.md), so every number here is labelled REFERENCE-BASED, not ground truth.

    python scripts/bench_tier_ref.py <tier plan.json> <reference lidar plan.json> --tier video|photo \
        [--ref-scene <lidar run>/scene] [--photo-truth <photo_inputs/...>/truth/photo_truth.json] [--out x.json]

Room matching (frame independent where possible):
  * photo with a photo_truth.json (simulated photo folders): each folder's room is the reference room that contains
    the folder's spin-camera centre (camera positions from the truth file, moved into the reference plan frame with
    the reference scene's T_align). The photo plan names each room after its folder (plan_beta T-3), so the match is
    by name, never by the photo geometry being scored.
  * otherwise (video, or photo without truth): rigid 2-D registration of the two footprints, 4 Manhattan yaws x
    FFT cross-correlation on a 5 cm raster, then Hungarian matching of rooms on polygon IoU (IoU >= 0.2).
Metrics: per matched room the two bbox dimensions (sorted; the case study's wall-length gate is scored on these
room dimensions, a proxy that is exact for rectangular rooms) and floor area: relative error, within tolerance
(video 3%, photo 8%), and whether the tier's 95% interval covers the reference value; footprint error and coverage;
adjacency precision / recall against the reference adjacency (mapped through the room match); pairwise overlap area
of the tier's room polygons; connectivity of the tier plan's adjacency graph (one stitched plan).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOL = {"video": 0.03, "photo": 0.08}
RES = 0.05            # raster for the footprint registration (m)
MIN_IOU = 0.2         # room match after registration
OVERLAP_EPS_M2 = 0.01  # polygon overlaps below this are numerical


def _poly(p):
    from shapely.geometry import Polygon
    g = Polygon(p)
    return g if g.is_valid else g.buffer(0)


def _m(meas):
    if not meas:
        return None, None, None
    ci = meas.get("ci95") or [meas.get("lo"), meas.get("hi")]
    return meas.get("value"), (ci[0] if ci else None), (ci[1] if ci else None)


def _dims(room):
    b = room.get("bbox_dims") or {}
    if isinstance(b, dict):
        ds = [_m(b.get("length")), _m(b.get("width"))]
    else:
        ds = [_m(x) for x in b]
    ds = [d for d in ds if d[0] is not None]
    return sorted(ds, key=lambda d: -d[0])


def load_plan(path: Path) -> dict:
    p = json.loads(Path(path).read_text())
    for r in p["rooms"]:
        if isinstance(r["polygon"], str):
            r["polygon"] = json.loads(r["polygon"])
    return p


# ------------------------------------------------------------------------------------------------ registration
def _raster(polys, lo, shape):
    from matplotlib.path import Path as MPath
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    pts = np.c_[(xs.ravel() + 0.5) * RES + lo[0], (ys.ravel() + 0.5) * RES + lo[1]]
    m = np.zeros(len(pts), bool)
    for P in polys:
        m |= MPath(np.asarray(P)).contains_points(pts)
    return m.reshape(shape).astype(np.float32)


def _rot(k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s], [s, c]], float)


def register_footprints(tier: dict, ref: dict) -> dict:
    """T (3x3) mapping tier plan (u, v) into reference plan (u, v): the yaw (multiple of 90 deg, both plans are
    Manhattan-aligned) and translation that maximise footprint overlap. Scale is NOT fitted (a scale error is what
    the tier is scored on)."""
    from scipy.signal import fftconvolve
    R_polys = [np.asarray(r["polygon"], float) for r in ref["rooms"]]
    rlo = np.min([P.min(0) for P in R_polys], 0) - 1.0
    rhi = np.max([P.max(0) for P in R_polys], 0) + 1.0
    best = None
    for k in range(4):
        Rk = _rot(k)
        T_polys = [np.asarray(r["polygon"], float) @ Rk.T for r in tier["rooms"]]
        tlo = np.min([P.min(0) for P in T_polys], 0) - 1.0
        thi = np.max([P.max(0) for P in T_polys], 0) + 1.0
        A = _raster(R_polys, rlo, tuple(np.ceil((rhi - rlo)[::-1] / RES).astype(int)))
        Bm = _raster(T_polys, tlo, tuple(np.ceil((thi - tlo)[::-1] / RES).astype(int)))
        C = fftconvolve(A, Bm[::-1, ::-1], mode="full")
        iy, ix = np.unravel_index(int(np.argmax(C)), C.shape)
        ov = float(C[iy, ix]) * RES ** 2
        # shift (cells) of B's origin in A's raster: full-mode index - (B.shape - 1)
        sy, sx = iy - (Bm.shape[0] - 1), ix - (Bm.shape[1] - 1)
        t = rlo + np.array([sx, sy]) * RES - tlo
        iou = ov / (A.sum() * RES ** 2 + Bm.sum() * RES ** 2 - ov)
        if best is None or iou > best["iou"]:
            T = np.eye(3)
            T[:2, :2] = Rk
            T[:2, 2] = t
            best = dict(yaw_deg=90 * k, iou=round(float(iou), 3), overlap_m2=round(ov, 2), T=T.tolist())
    return best


def _apply(T, P):
    T = np.asarray(T)
    return np.asarray(P, float) @ T[:2, :2].T + T[:2, 2]


# ------------------------------------------------------------------------------------------------ matching
def match_by_iou(tier: dict, ref: dict, T) -> dict[str, str]:
    from scipy.optimize import linear_sum_assignment
    tp = [_poly(_apply(T, r["polygon"])) for r in tier["rooms"]]
    rp = [_poly(r["polygon"]) for r in ref["rooms"]]
    M = np.array([[a.intersection(b).area / max(a.union(b).area, 1e-9) for b in rp] for a in tp])
    ri, ci = linear_sum_assignment(-M)
    return {tier["rooms"][i]["id"]: ref["rooms"][j]["id"] for i, j in zip(ri, ci) if M[i, j] >= MIN_IOU}


def match_by_folder_truth(tier: dict, ref: dict, ref_scene: Path, truth: Path) -> tuple[dict, dict]:
    """photo room (named after its folder) -> reference room containing the folder's spin-camera centre."""
    from shapely.geometry import Point
    from floorplan.pipeline.scene import load_scene
    s, _ = load_scene(ref_scene)
    Ta = np.asarray(s["T_align"])
    tr = json.loads(Path(truth).read_text())
    cams: dict[str, list] = {}
    for name, ph in tr["photos"].items():
        folder = name.split("/")[0]
        c = np.asarray(ph["T_wc"], float)[:3, 3]
        cams.setdefault(folder, []).append((ph.get("role") == "spin", c))
    rp = {r["id"]: _poly(r["polygon"]) for r in ref["rooms"]}
    folder_to_ref, diag = {}, {}
    for folder, cs in cams.items():
        pts = np.array([c for spin, c in cs if spin] or [c for _, c in cs])
        X = Ta[:3, :3] @ np.median(pts, 0) + Ta[:3, 3]
        q = Point(X[0], X[2])
        dist = {rid: g.distance(q) for rid, g in rp.items()}
        rid = min(dist, key=dist.get)
        if dist[rid] <= 0.5:
            folder_to_ref[folder] = rid
        diag[folder] = dict(ref_room=rid if dist[rid] <= 0.5 else None, dist_m=round(dist[rid], 3),
                            centre_uv=[round(X[0], 3), round(X[2], 3)], n_spin=sum(sp for sp, _ in cs))
    m = {}
    for r in tier["rooms"]:
        f = r.get("label") or r["id"]
        if f in folder_to_ref:
            m[r["id"]] = folder_to_ref[f]
    return m, dict(folder_to_ref=folder_to_ref, folders=diag)


# ------------------------------------------------------------------------------------------------ scoring
def _pairs(plan):
    return {frozenset((a["room_a"], a["room_b"])) for a in plan.get("adjacency") or []}


def _components(ids, pairs):
    parent = {i: i for i in ids}

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p in pairs:
        a, b = tuple(p)
        if a in parent and b in parent:
            parent[f(a)] = f(b)
    return len({f(i) for i in ids})


def _row(kind, room, ref_room, v, lo, hi, rv, tol):
    rel = None if v is None or not rv else (v - rv) / rv
    return dict(kind=kind, room=room, ref_room=ref_room, value=v, lo=lo, hi=hi, ref=rv,
                rel_err=None if rel is None else round(rel, 4),
                within_tol=None if rel is None else bool(abs(rel) <= tol),
                covered=None if lo is None or hi is None or rv is None else bool(lo <= rv <= hi),
                rel_halfwidth=None if lo is None or hi is None or not rv else round((hi - lo) / 2 / rv, 4))


def score(tier_plan: Path, ref_plan: Path, tier: str, ref_scene: Path | None = None,
          photo_truth: Path | None = None) -> dict:
    tp, rp = load_plan(tier_plan), load_plan(ref_plan)
    tol = TOL[tier]
    out = dict(label="REFERENCE-BASED (LiDAR plan of the same capture), not ground truth", tier=tier,
               tier_plan=str(tier_plan), ref_plan=str(ref_plan), tolerance_rel=tol)
    if not tp["rooms"]:
        out.update(status="no_rooms")
        return out
    reg = register_footprints(tp, rp)
    out["registration"] = {k: v for k, v in reg.items() if k != "T"}
    if tier == "photo" and photo_truth and ref_scene and Path(photo_truth).exists():
        match, out["folder_matching"] = match_by_folder_truth(tp, rp, ref_scene, photo_truth)
        out["matching"] = "photo folder -> reference room containing the folder's spin-camera centre (truth poses)"
    else:
        match = match_by_iou(tp, rp, reg["T"])
        out["matching"] = f"footprint registration (4 yaws x FFT, {RES} m raster) + Hungarian on room IoU >= {MIN_IOU}"
    out["room_match"] = match
    tr = {r["id"]: r for r in tp["rooms"]}
    rr = {r["id"]: r for r in rp["rooms"]}
    rows = []
    for tid, rid in match.items():
        a, b = tr[tid], rr[rid]
        da, db = _dims(a), _dims(b)
        for k, name in enumerate(("dim_long", "dim_short")):
            if k < len(da) and k < len(db):
                rows.append(_row(name, tid, rid, *da[k], db[k][0], tol))
        rows.append(_row("area", tid, rid, *_m(a["floor_area"]), _m(b["floor_area"])[0], tol))
    out["rows"] = rows
    dims = [r for r in rows if r["kind"].startswith("dim")]
    areas = [r for r in rows if r["kind"] == "area"]
    ref_ids_matched = set(match.values())
    # rooms
    out["rooms"] = dict(tier=len(tp["rooms"]), reference=len(rp["rooms"]), matched=len(match),
                        tier_unmatched=sorted(set(tr) - set(match)),
                        reference_unmatched=sorted(set(rr) - ref_ids_matched),
                        many_to_one=sorted({v for v in match.values() if list(match.values()).count(v) > 1}))
    # dims / areas
    def agg(rs):
        if not rs:
            return dict(n=0)
        e = np.array([abs(r["rel_err"]) for r in rs])
        cov = [r["covered"] for r in rs if r["covered"] is not None]
        hw = [r["rel_halfwidth"] for r in rs if r["rel_halfwidth"] is not None]
        return dict(n=len(rs), within_tol=int(sum(r["within_tol"] for r in rs)), median_abs_rel_err=round(float(np.median(e)), 4),
                    max_abs_rel_err=round(float(e.max()), 4), coverage95=round(float(np.mean(cov)), 3) if cov else None,
                    median_rel_halfwidth=round(float(np.median(hw)), 4) if hw else None)
    out["dims"], out["areas"] = agg(dims), agg(areas)
    # footprint
    fv, flo, fhi = _m(tp["footprint_area"])
    out["footprint"] = _row("footprint", "all", "all", fv, flo, fhi, _m(rp["footprint_area"])[0], tol)
    # photo: the simulated folders may not cover every reference room (the simulator only makes a folder for rooms
    # with a usable spin). The fair whole-property comparison is then against the reference rooms that HAVE a folder.
    covered_ref = set(out.get("folder_matching", {}).get("folder_to_ref", {}).values()) or set(rr)
    out["rooms"]["reference_with_input"] = len(covered_ref)
    if covered_ref != set(rr):
        from shapely.ops import unary_union
        a = unary_union([_poly(rr[i]["polygon"]) for i in covered_ref]).area
        out["footprint_input_rooms"] = _row("footprint_input_rooms", "all", sorted(covered_ref), fv, flo, fhi, a, tol)
    # adjacency, mapped through the room match
    ref_pairs = _pairs(rp)
    tier_pairs = _pairs(tp)
    mapped = []
    for p in tier_pairs:
        a, b = tuple(p)
        ma, mb = match.get(a), match.get(b)
        ok = ma is not None and mb is not None and ma != mb and frozenset((ma, mb)) in ref_pairs
        mapped.append(dict(tier=sorted(p), ref=None if ma is None or mb is None else sorted((ma, mb)), correct=ok))
    found = {frozenset(m["ref"]) for m in mapped if m["correct"]}
    ref_pairs_matched = {p for p in ref_pairs if p <= ref_ids_matched}
    ref_pairs_input = {p for p in ref_pairs if p <= covered_ref}
    out["adjacency"] = dict(
        reported=len(tier_pairs), correct=sum(m["correct"] for m in mapped),
        reference_pairs=len(ref_pairs), reference_pairs_between_matched_rooms=len(ref_pairs_matched),
        recall_all=round(len(found) / len(ref_pairs), 3) if ref_pairs else None,
        recall_matched=round(len(found & ref_pairs_matched) / len(ref_pairs_matched), 3) if ref_pairs_matched else None,
        reference_pairs_between_input_rooms=len(ref_pairs_input),
        recall_input=round(len(found & ref_pairs_input) / len(ref_pairs_input), 3) if ref_pairs_input else None,
        precision=round(sum(m["correct"] for m in mapped) / len(mapped), 3) if mapped else None,
        missing=[sorted(p) for p in ref_pairs - found], pairs=mapped)
    # overlaps between the tier's own rooms
    polys = {r["id"]: _poly(r["polygon"]) for r in tp["rooms"]}
    ids = list(polys)
    ov = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = polys[ids[i]].intersection(polys[ids[j]]).area
            if a > OVERLAP_EPS_M2:
                ov.append(dict(rooms=[ids[i], ids[j]], area_m2=round(a, 3)))
    out["overlaps"] = dict(n=len(ov), total_m2=round(sum(o["area_m2"] for o in ov), 3), pairs=ov)
    out["stitch"] = dict(rooms=len(ids), components=_components(ids, tier_pairs),
                         one_stitched_plan=bool(_components(ids, tier_pairs) == 1))
    # calibration (all interval rows)
    cal = [r for r in rows + [out["footprint"]] if r["covered"] is not None]
    out["calibration"] = dict(n=len(cal), coverage95=round(float(np.mean([r["covered"] for r in cal])), 3) if cal else None,
                              median_rel_halfwidth=round(float(np.median([r["rel_halfwidth"] for r in cal])), 4) if cal else None)
    return out


def plot_overlay(tier_plan: Path, ref_plan: Path, res: dict, path: Path) -> None:
    """Reference rooms (grey, ids) and the tier's rooms placed by the footprint registration (colour = matched)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tp, rp = load_plan(tier_plan), load_plan(ref_plan)
    T = register_footprints(tp, rp)["T"] if tp["rooms"] else np.eye(3).tolist()
    fig, ax = plt.subplots(figsize=(9, 9))
    for r in rp["rooms"]:
        P = np.asarray(r["polygon"], float)
        ax.fill(P[:, 0], P[:, 1], color="0.85", ec="0.4", lw=1.0)
        c = P.mean(0)
        ax.text(c[0], c[1], f"ref {r['id']}", fontsize=8, color="0.3", ha="center")
    m = res.get("room_match") or {}
    for r in tp["rooms"]:
        P = _apply(T, r["polygon"])
        P = np.vstack([P, P[:1]])
        col = "#2a78d6" if r["id"] in m else "#eb6834"
        ax.plot(P[:, 0], P[:, 1], color=col, lw=1.8)
        c = P[:-1].mean(0)
        ax.text(c[0], c[1] - 0.25, f"{res['tier']} {r['id']}" + (f"->{m[r['id']]}" if r["id"] in m else " (unmatched)"),
                fontsize=8, color=col, ha="center")
    fp = res.get("footprint") or {}
    ax.set_title(f"{res['tier']} plan (blue = matched, orange = unmatched) on the LiDAR reference (grey)\n"
                 f"registration IoU {res.get('registration', {}).get('iou')}, footprint {fp.get('value')} vs ref "
                 f"{fp.get('ref')} m2 | REFERENCE-BASED, not ground truth", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tier_plan", type=Path)
    ap.add_argument("ref_plan", type=Path)
    ap.add_argument("--tier", required=True, choices=list(TOL))
    ap.add_argument("--ref-scene", type=Path)
    ap.add_argument("--photo-truth", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = score(a.tier_plan, a.ref_plan, a.tier, a.ref_scene, a.photo_truth)
    s = json.dumps(res, indent=1, default=float)
    if a.out:
        a.out.write_text(s)
    print(s)


if __name__ == "__main__":
    main()
