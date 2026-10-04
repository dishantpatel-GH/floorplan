"""Damage module: per-surface damage regions, concealed-damage flags and scope line items (Part 2 output contract).

Public interface (what other modules call):

    from floorplan.damage import analyse_damage
    result = analyse_damage(views, scene, info, plan=plan_dict_or_None, out_dir=Path(...))
    plan_dict["damage"], plan_dict["scope_items"] = result["damage"], result["scope_items"]

`views` is a floorplan.damage.project.Views (posed images in the aligned frame); build it with
views_from_stray (LiDAR tier), views_from_video (video tier) or Views.load(dir) (any front-end that writes
views.json + images/, e.g. the photo tier or scripts/inject_damage.py).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from floorplan.damage.detect import DetectParams, analyse, describe
from floorplan.damage.project import (Surface, Views, build_ortho, sub_surface, surfaces_from_plan,
                                      surfaces_from_scene, tile_surface, views_from_photos, views_from_stray,
                                      views_from_video, views_from_video_file)
from floorplan.damage.rules import apply_rules
from floorplan.damage.scope import build_scope

__all__ = ["analyse_damage", "run_damage_on_plan", "tier_counts", "views_for_tier", "to_plan_items", "Views", "views_from_stray",
           "views_from_video", "views_from_video_file", "views_from_photos", "DetectParams"]


def _iv(v: float, sigma: float, unit: str, method: str, status: str = "measured") -> dict:
    return dict(value=v, lo=max(v - 1.96 * sigma, 0.0), hi=v + 1.96 * sigma, unit=unit, method=method, status=status)


def _lohi(m: dict) -> dict:
    """v3: accept the published plan.json form ({value, ci95: [lo, hi]}) as well as the in-memory {value, lo, hi}, so
    scripts/run_damage.py can re-run damage on a one-command run's own plan.json."""
    m = dict(m)
    if "lo" not in m and m.get("ci95"):
        m["lo"], m["hi"] = m["ci95"]
    return m


def surface_measurements(surfaces: list[Surface], plan: dict | None) -> dict[str, dict]:
    """Surface areas/lengths with intervals for scope quantities: from the plan where it exists (its intervals are
    the measured ones), else from the surface rectangle with a conservative +-2 cm / +-5 cm."""
    out: dict[str, dict] = {}
    walls = {w["id"]: w for w in (plan or {}).get("walls", [])}
    rooms = {r["id"]: r for r in (plan or {}).get("rooms", [])}
    for s in surfaces:
        meta = dict(kind=s.kind, room_id=s.room_id, room_label=s.meta.get("room_label", ""),
                    openings=s.meta.get("openings", []))
        if s.kind == "wall":
            w = walls.get(s.id)
            L = _lohi(w["length"]) if w else _iv(s.size[0], 0.01, "m", "fallback plane extent", "inferred")
            r = rooms.get(s.room_id)
            ch = r["ceiling_height"] if r else None
            H = _lohi(ch) if ch and ch.get("value") else _iv(s.size[1], 0.08, "m",
                                                           "highest observed wall texture (ceiling not observed)",
                                                           "inferred")
            v = L["value"] * H["value"]
            rel = np.hypot((L["hi"] - L["lo"]) / (2 * 1.96 * L["value"]), (H["hi"] - H["lo"]) / (2 * 1.96 * H["value"]))
            meta.update(length=L, height=H,
                        area=_iv(v, v * rel, "m2", "wall length x room height (gross, openings not deducted)",
                                 "inferred"))
        else:
            r = rooms.get(s.room_id)
            meta.update(area=_lohi(r["floor_area"]) if r else _iv(s.area(), 0.02 * s.area(), "m2", "fallback extent",
                                                                 "inferred"))
        out[s.id] = meta
    return out


def _overlay(o, regs, accepted_ids) -> np.ndarray:
    img = (np.nan_to_num(o.median) * 255).astype(np.uint8).copy()
    img[~(o.mask & (o.count >= 1))] = (40, 40, 40)
    for r, rid in zip(regs, accepted_ids):
        cnts, _ = cv2.findContours(r.pixels.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if rid:
            col = (0, 0, 255) if r.cls == "water_stain" else (0, 200, 0)
            cv2.drawContours(img, cnts, -1, col, 2)
            x, y, _, _ = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            cv2.putText(img, rid, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        else:
            cv2.drawContours(img, cnts, -1, (255, 160, 0) if r.cls == "water_stain" else (200, 0, 200), 1)
    return img


def _bbox_st(r, o) -> list[float]:
    """Parent-surface (s, t) bbox of a candidate (also for rejected ones: needed for precision/recall analysis)."""
    fine = r.stats.get("fine")
    oo, pix = (fine["ortho"], fine["region"] | fine["skeleton"]) if fine is not None else (o, r.pixels)
    ys, xs = np.nonzero(pix)
    if not len(ys):
        return []
    s, t = oo.surface.pixel_st(xs, ys, oo.res)
    off = oo.surface.meta.get("tile_offset", (0.0, 0.0))
    return [round(float(v), 4) for v in (s.min() + off[0], t.min() + off[1], s.max() + off[0], t.max() + off[1])]


def _features(r) -> dict:
    keys = ("length_m", "contrast_L", "straightness", "width_m", "mean_dL", "mean_db", "refined")
    return {k: (round(float(r.stats[k]), 4) if not isinstance(r.stats[k], bool) else r.stats[k])
            for k in keys if k in r.stats}


# D-025: protruding objects (door/wardrobe handles, switches, knobs) are not surface damage. A crack lies flat on the
# surface; a handle stands 2-4 cm proud of it. v2 of this check is LOCAL: it compares the scene points' height above the
# plane ON the detected line (within 6 mm of its skeleton) with the height right BESIDE it (1.5-4 cm away). A crack next
# to a skirting board or furniture is still flat relative to its own sides, so it is kept; a handle is the line itself,
# so it stands out. (v1 used the whole bounding box and wrongly rejected 4/9 injected cracks near furniture/skirting.)
RELIEF_MAX_M = {"lidar": 0.015, "video": 0.04, "photo": 0.04}


def _skeleton_st(r, o) -> np.ndarray | None:
    """Parent-surface (s, t) coordinates of the candidate's centre line (fine skeleton if refined, else its pixels)."""
    fine = r.stats.get("fine")
    if fine is not None and fine.get("skeleton") is not None and fine["skeleton"].any():
        oo, pix = fine["ortho"], fine["skeleton"]
    else:
        oo, pix = o, r.pixels
    ys, xs = np.nonzero(pix)
    if not len(ys):
        return None
    s, t = oo.surface.pixel_st(xs, ys, oo.res)
    off = oo.surface.meta.get("tile_offset", (0.0, 0.0))
    return np.c_[np.asarray(s) + off[0], np.asarray(t) + off[1]]


def _line_relief_m(srf, line_st: np.ndarray, raw_points: np.ndarray, band_m: float = 0.006,
                   ring_lo_m: float = 0.015, ring_hi_m: float = 0.04) -> float | None:
    """Median height of points ON the line minus median height of points BESIDE it (m), or None to abstain."""
    from scipy.spatial import cKDTree
    lo, hi = line_st.min(0) - ring_hi_m, line_st.max(0) + ring_hi_m
    rel = raw_points - srf.origin
    d = rel @ srf.normal
    s, t = rel @ srf.ax_s, rel @ srf.ax_t
    sel = (np.abs(d) < 0.15) & (s >= lo[0]) & (s <= hi[0]) & (t >= lo[1]) & (t <= hi[1])
    if sel.sum() < 80:
        return None
    dist, _ = cKDTree(line_st).query(np.c_[s[sel], t[sel]])
    band, ring = dist <= band_m, (dist >= ring_lo_m) & (dist <= ring_hi_m)
    if band.sum() < 20 or ring.sum() < 50:
        return None
    dd = d[sel]
    return float(np.median(dd[band]) - np.median(dd[ring]))


# v3 (damage_precision): a stain is a discolouration OF the surface, so it must lie ON the surface. The benchmark found
# a confirmed "water stain" (0.73) on floor_only that was the gold lid of a black jar standing on a bathroom shelf: the
# orthophoto only treats things more than 5 cm + 3% of the range in front of the wall as occluders, so a shelf object
# 3-9 cm proud is painted onto the wall texture, and the colour-only stain rules cannot tell a yellow lid from a yellow
# stain. The test is REGION-local, in the spirit of D-025, but ABSOLUTE rather than relative to a ring: an object is
# usually larger than its yellow part, so the ring around the lid is the jar itself and "inside minus ring" is ~0. We
# compare the LiDAR / learned-depth points that fall INSIDE the candidate with the LOCAL wall level (points within
# +-3 cm of the wall plane, 2-30 cm around the candidate). Thresholds per tier as D-025 (sensor noise): LiDAR 1.5 cm,
# learned depth 4 cm. Two-sided for stains: |relief| above the threshold rejects (in front = object; behind = the
# camera saw through the plane: glass, mirror, opening). Only points from 5 cm behind to 25 cm in front of the plane
# are used, so the far face of a wall (>= 7 cm thick) never enters. Abstains (keeps the stain, records why) with
# fewer than STAIN_RELIEF_MIN_IN points inside.
STAIN_RELIEF_MIN_IN = 30
STAIN_RELIEF_MIN_WALL = 50


def _region_relief(o, region: np.ndarray, raw_points: np.ndarray, wall_win_m: float = 0.30,
                   wall_band_m: float = 0.03, gap_m: float = 0.02, front_m: float = 0.25) -> dict:
    """Height of the candidate's own surface above the local wall, from the scene points (m).

    Returns dict(relief_m | None, n_in, n_wall, frac_proud_1p5cm, wall_level_m, note). relief_m is None = abstain.
    o: the tile Ortho (its surface defines the plane and the pixel grid); region: (H, W) bool on that grid."""
    from scipy import ndimage as ndi
    srf, res = o.surface, o.res
    ys, xs = np.nonzero(region)
    out = dict(relief_m=None, n_in=0, n_wall=0, frac_proud_1p5cm=None, wall_level_m=None, note="")
    if not len(ys):
        out["note"] = "empty region"
        return out
    s, t = srf.pixel_st(xs, ys, res)
    lo = np.array([s.min(), t.min()]) - wall_win_m
    hi = np.array([s.max(), t.max()]) + wall_win_m
    rel = raw_points - srf.origin
    ps, pt = rel @ srf.ax_s, rel @ srf.ax_t
    sel = (ps >= lo[0]) & (ps <= hi[0]) & (pt >= lo[1]) & (pt <= hi[1])
    if not sel.any():
        out["note"] = "no scene points near the candidate"
        return out
    d = rel[sel] @ srf.normal                      # + = in front of the wall (into the room)
    ps, pt = ps[sel], pt[sel]
    keep = (d > -0.05) & (d < front_m)
    d, ps, pt = d[keep], ps[keep], pt[keep]
    col, row = srf.st_to_pixel(ps, pt, res)
    col, row = np.round(col).astype(int), np.round(row).astype(int)
    H, W = region.shape
    inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    dist = np.full(len(d), np.inf)
    dist[inb] = ndi.distance_transform_edt(~region)[row[inb], col[inb]] * res
    inside = dist == 0
    # wall reference: outside the candidate (by >= 2 cm) and near the plane; points beyond the ortho grid count too
    far = ~inb | (dist >= gap_m)
    wall = far & (np.abs(d) < wall_band_m)
    out.update(n_in=int(inside.sum()), n_wall=int(wall.sum()))
    if inside.sum() < STAIN_RELIEF_MIN_IN:
        out["note"] = f"abstain: {int(inside.sum())} scene points inside the candidate (< {STAIN_RELIEF_MIN_IN})"
        return out
    level = float(np.median(d[wall])) if wall.sum() >= STAIN_RELIEF_MIN_WALL else 0.0
    if wall.sum() < STAIN_RELIEF_MIN_WALL:
        out["note"] = "few local wall points: wall level taken from the fitted plane"
    h = d[inside] - level
    out.update(relief_m=float(np.median(h)), frac_proud_1p5cm=float((h > 0.015).mean()), wall_level_m=level)
    return out


def analyse_damage(views: Views, scene: dict, info: dict, plan: dict | None = None, out_dir: Path | None = None,
                   res: float = 0.005, params: DetectParams | None = None, include_floor: bool = False,
                   max_views_per_surface: int = 16, log=print, stain_relief: bool = True) -> dict:
    """Detect damage on every surface, apply the concealed-damage rules and build the scope.

    Returns a JSON-ready dict: damage (accepted regions), concealed_flags, scope_items, rejected (counts by rule),
    surfaces (coverage per surface), meta."""
    t0 = time.time()
    params = params or DetectParams()
    if views.tier == "photo":
        # 6-8 photos per room (D-017), and most wall patches are seen by only 2 of them: demanding 3 views / 4
        # agreeing views would make most of a room undetectable. Pixels need 2 views, regions 2 agreeing views; the
        # confidence (agreement term 1 - exp(-n/3)) carries the weaker evidence instead.
        params.min_views = min(params.min_views, 2)
        params.min_support_views = min(params.min_support_views, 2)
    surfaces = surfaces_from_plan(plan, scene, include_floor) if plan else surfaces_from_scene(scene, info)
    log(f"[damage] {len(surfaces)} surfaces ({'plan' if plan else 'scene fallback'}), {views.n} views, tier {views.tier}")
    smeta = surface_measurements(surfaces, plan)
    if out_dir:
        shutil.rmtree(Path(out_dir) / "surfaces", ignore_errors=True)     # no stale figures from earlier runs
        (Path(out_dir) / "surfaces").mkdir(parents=True, exist_ok=True)
    regions, rejected, rejected_detail, coverage, zc = [], {}, [], [], {}
    for srf in surfaces:
        for tl in tile_surface(srf):
            o = build_ortho(tl, views, scene["points"], res=res, max_views=max_views_per_surface, zcache=zc)
            if o is None:
                coverage.append(dict(surface_id=srf.id, tile=tl.meta.get("tile"), analysed_m2=0.0,
                                     note="not seen by >= 2 usable views"))
                continue
            def refine(s0, t0, s1, t1, _tl=tl):
                """Fine orthophoto of a window of this tile (v2 crack tracing)."""
                return build_ortho(sub_surface(_tl, s0, t0, s1, t1), views, scene["points"], res=params.crack_fine_res,
                                   max_views=max_views_per_surface, zcache=zc)
            regs, sinfo = analyse(o, params, refine=refine if params.crack_fine_res else None)
            coverage.append(dict(surface_id=srf.id, tile=tl.meta.get("tile"), analysed_m2=round(sinfo["valid_m2"], 3),
                                 surface_m2=round(tl.area(), 3), n_views=sinfo["n_views"],
                                 textured=sinfo.get("textured", False), note=sinfo.get("skipped", "")))
            ids = []
            for r in regs:
                if r.rejected:
                    key = r.rejected.split(" (")[0].split(":")[0]
                    rejected[f"{r.cls}: {key}"] = rejected.get(f"{r.cls}: {key}", 0) + 1
                    rejected_detail.append(dict(surface_id=srf.id, cls=r.cls, rule=r.rejected,
                                                area_m2=round(float(r.pixels.sum() * res * res), 4),
                                                score=round(r.score, 2), support=round(r.support, 2), n_views=r.n_views,
                                                local_spread_L=round(r.stats.get("local_spread_L", 0.0), 2),
                                                confidence=r.stats.get("confidence"),
                                                bbox_st=_bbox_st(r, o),
                                                features=_features(r)))
                    ids.append("")
                    continue
                d = describe(r, o, views.pose_sigma_m, views.scale_sigma_rel)
                line_st = _skeleton_st(r, o) if r.cls == "crack" and "raw_points" in scene else None
                relief = _line_relief_m(srf, line_st, scene["raw_points"]) if line_st is not None else None
                if r.cls == "water_stain" and "raw_points" in scene and stain_relief:
                    rr = _region_relief(o, r.pixels, scene["raw_points"])
                    relief = rr["relief_m"]
                    d["relief"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rr.items()}
                thr = RELIEF_MAX_M.get(views.tier, 0.04)
                # stains are tested two-sided: a stain is paint/plaster discolouration, so its own points must lie ON
                # the wall; points BEHIND the plane mean the camera looked through it (glass, mirror, opening)
                behind = r.cls == "water_stain" and relief is not None and relief < -thr
                if (r.cls == "water_stain" and views.tier != "lidar" and relief is not None
                        and (relief > thr or behind)):
                    # learned depth (video/photo) is too noisy for the relief to be decisive (median |relief| on
                    # random clean-wall patches 1.4 cm, 90th pct 3-12 cm: damage.md v3.5), so a stain that seems
                    # off the surface is kept but can only be REVIEW: never quoted, never a concealed flag
                    d["relief_review"] = (f"{'behind' if behind else 'in front of'} the surface by "
                                          f"{100 * abs(relief):.1f} cm (learned depth): possible object or "
                                          f"see-through, for a human to check")
                    relief = None
                if relief is not None and (relief > thr or behind):
                    if behind:
                        rule = (f"lies {100 * -relief:.1f} cm behind the surface: see-through (glass, mirror, "
                                f"opening), not a stain on the wall")
                    else:
                        what = ("object (handle, switch, knob), not surface damage" if r.cls == "crack" else
                                "object in front of the wall (on a shelf/sill), not a stain on it")
                        rule = f"protrudes {100 * relief:.1f} cm from the surface: {what}"
                    key = f"{r.cls}: {'behind the surface' if behind else 'protrudes from the surface'}"
                    rejected[key] = rejected.get(key, 0) + 1
                    rejected_detail.append(dict(surface_id=srf.id, cls=r.cls, rule=rule, confidence=d.get("confidence"),
                                                bbox_st=d.get("bbox_st"), relief_m=round(relief, 4),
                                                relief=d.get("relief")))
                    ids.append("")
                    continue
                if relief is not None:
                    d["relief_m"] = round(relief, 4)
                d["id"] = f"D{len(regions) + 1}"
                d["local_view_spread_L"] = round(r.stats.get("local_spread_L", 0.0), 2)
                regions.append(d)
                ids.append(d["id"])
            if out_dir and regs:
                tag = srf.id + ("" if tl.meta.get("tile") is None else f"_t{tl.meta['tile'][0]}{tl.meta['tile'][1]}")
                cv2.imwrite(str(Path(out_dir) / "surfaces" / f"{tag}.jpg"), _overlay(o, regs, ids))
    flags = apply_rules(regions, smeta)
    scope = build_scope(regions, flags, smeta)
    analysed = sum(c["analysed_m2"] for c in coverage)
    out = dict(damage=regions, concealed_flags=flags, scope_items=scope, rejected_candidates=rejected,
               rejected_detail=rejected_detail,
               surfaces=coverage,
               meta=dict(tier=views.tier, n_views=views.n, res_m=res, surfaces=len(surfaces),
                         analysed_m2=round(analysed, 2), plan_used=bool(plan), runtime_s=round(time.time() - t0, 1),
                         pose_sigma_m=views.pose_sigma_m, scale_sigma_rel=views.scale_sigma_rel,
                         params=params.__dict__ | {"crack_sigmas_px": list(params.crack_sigmas_px)}))
    # v3: what is REPORTED is after the D-023/D-026 two-tier filtering (review-tier detections create no flags and no
    # scope); the raw detector counts are kept for diagnosis but labelled as such (the old line mixed the two).
    out["reported"] = tier_counts(out)
    rp = out["reported"]
    log(f"[damage] detector: {len(regions)} candidate regions, {len(flags)} raw flags, {len(scope)} raw scope items "
        f"(before two-tier filtering); analysed {analysed:.1f} m2 in {out['meta']['runtime_s']} s")
    log(f"[damage] reported: {rp['confirmed']} confirmed {rp['confirmed_by_class']}, {rp['review']} review, "
        f"{rp['concealed_flags']} concealed flags, {rp['scope_items']} scope items")
    if out_dir:
        (Path(out_dir) / "damage.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# v2: hook for the one-command CLI (scripts/run_capture.py)
# ----------------------------------------------------------------------------------------------------------------------
def views_for_tier(scene: dict, info: dict, input_path: Path, tier: str, max_views: int = 160) -> Views:
    """Posed images for any tier, from the run_capture inputs alone (no front-end work directories needed).

    lidar: the Stray folder's rgb.mp4 + the scene's (drift-corrected) poses; video: the keyframes decoded again from
    the video file; photo: the original photos + the photo scene's poses. Raises with a readable reason."""
    input_path = Path(input_path)
    if tier == "lidar":
        return views_from_stray(input_path, scene, max_views=max_views)
    if tier == "video":
        return views_from_video_file(input_path, scene, info, max_views=max_views)
    if tier == "photo":
        return views_from_photos(input_path, scene, info)
    raise ValueError(f"unknown tier {tier!r}")


def _measurement(m: dict) -> dict:
    return dict(value=m["value"], lo=m["lo"], hi=m["hi"], unit=m.get("unit", "m"), method=m.get("method", ""),
                status=m.get("status", "measured"))


# D-023: two-tier reporting. Detections at or above CONFIRM_CONFIDENCE are "confirmed" and drive scope line items;
# lower ones are reported as "review" (visible in the JSON and the render, flagged for a human) but generate NO scope
# items and NO concealed-damage flags. Measured basis (docs/modules/damage.md v2): true stains score 0.37-0.98, true
# cracks 0.18-0.82; the clean-apartment false crack found at integration scored 0.17. Over-claiming damage costs a
# homeowner or insurer money, and the brief penalises confident garbage, so low-confidence findings go to review.
CONFIRM_CONFIDENCE = {"water_stain": 0.35, "crack": 0.65}   # D-026: per class; unknown classes use 0.35


def to_plan_items(res: dict) -> tuple[list[dict], list[dict]]:
    """analyse_damage output -> (plan.damage, plan.scope_items) in the plan schema's shape.

    Damage regions: {id, class, surface_id, room_id, extent{area, width, height, length?, center_s, center_t},
    confidence, concealed: false, ...detail}. Concealed-damage flags are damage entries with concealed: true and
    the rule that fired (schema: 'or a concealed-damage flag'). Scope: {id, surface_id, damage_ids, description,
    quantity}."""
    damage = []
    for d in res["damage"]:
        ext = {k: _measurement(d[k]) for k in ("area", "width", "height", "length", "center_s", "center_t") if k in d}
        extra = {k: v for k, v in d.items() if k not in ext and k not in ("id", "cls", "surface_id", "room_id",
                                                                          "confidence")}
        thr = CONFIRM_CONFIDENCE.get(d["cls"], 0.35)
        status = "confirmed" if (d.get("confidence") or 0.0) >= thr and not d.get("relief_review") else "review"
        damage.append(dict(id=d["id"], **{"class": d["cls"]}, surface_id=d["surface_id"], room_id=d.get("room_id"),
                           extent=ext, confidence=d["confidence"], status=status, concealed=False, detail=extra))
    confirmed = {x["id"] for x in damage if x["status"] == "confirmed"}
    for i, f in enumerate(res["concealed_flags"]):
        if f.get("region_id") and f["region_id"] not in confirmed:
            continue                              # rule fired on a review-tier detection: no concealed flag
        damage.append(dict(id=f"F{i + 1}", **{"class": f"concealed_{f['flag']}" if not str(f["flag"]).startswith(
            "concealed") else f["flag"]}, surface_id=f["surface_id"], concealed=True, rule=f["rule_id"],
            confidence=f["confidence"], source_region=f["region_id"], p_condition=f["p_condition"],
            evidence=f["evidence"], severity=f["severity"], actions=f["actions"], rationale=f["rationale"],
            reference=f["reference"]))
    kept = [x for x in res["scope_items"] if not [r for r in x["region_ids"] if r]
            or any(r in confirmed for r in x["region_ids"] if r)]
    scope = [dict(id=f"S{i + 1}", surface_id=x["surface_id"], damage_ids=[r for r in x["region_ids"] if r],
                  description=x["text"], quantity=_measurement(x["quantity"]), item_code=x["item_id"],
                  why=x["why"], confidence=x["confidence"]) for i, x in enumerate(kept)]
    return damage, scope


def tier_counts(res: dict) -> dict:
    """Counts AFTER the two-tier filtering, i.e. what the plan will contain (v3: used by the logs and damage.json)."""
    damage, scope = to_plan_items(res)
    det = [x for x in damage if not x["concealed"]]
    conf = [x for x in det if x["status"] == "confirmed"]
    by = {}
    for x in conf:
        by[x["class"]] = by.get(x["class"], 0) + 1
    return dict(confirmed=len(conf), confirmed_by_class=by, review=len(det) - len(conf),
                concealed_flags=len([x for x in damage if x["concealed"]]), scope_items=len(scope))


def run_damage_on_plan(plan, scene: dict, info: dict, input_path, tier: str, log=print,
                       out_dir: Path | None = None, max_views: int = 160):
    """One-command hook: fill plan.damage and plan.scope_items for any tier. NEVER raises.

    If the views cannot be built (missing video/photos, a front-end that placed no photo) or the analysis fails,
    plan.meta['damage'] records why and plan.damage stays empty, so the plan is still written. When it succeeds,
    plan.meta['damage'] holds the coverage summary: analysed m2 (no damage reported means 'none on the X m2 we
    actually saw'), rejected-candidate counts by rule, and the tier's view/pose/scale figures."""
    status = dict(tier=tier, ok=False)
    try:
        t0 = time.time()
        views = views_for_tier(scene, info, Path(input_path), tier, max_views=max_views)
        if views.n < 2:
            raise ValueError(f"only {views.n} posed image(s) for tier {tier}: no multi-view damage evidence")
        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else plan
        res = analyse_damage(views, scene, info, plan=plan_dict, out_dir=out_dir, log=log)
        damage, scope = to_plan_items(res)
        if hasattr(plan, "damage"):
            plan.damage, plan.scope_items = damage, scope
        else:
            plan["damage"], plan["scope_items"] = damage, scope
        conf = [x for x in damage if not x["concealed"] and x["status"] == "confirmed"]
        status.update(ok=True, n_views=views.n, analysed_m2=res["meta"]["analysed_m2"], regions=len(res["damage"]),
                      confirmed=len(conf), review=len([x for x in damage if not x["concealed"]]) - len(conf),
                      flags=len([x for x in damage if x["concealed"]]), scope_items=len(scope),
                      rejected_candidates=res["rejected_candidates"],
                      surfaces_not_analysed=[c["surface_id"] for c in res["surfaces"] if not c["analysed_m2"]],
                      runtime_s=round(time.time() - t0, 1))
    except Exception as e:                       # noqa: BLE001 - the plan must still be written
        status.update(error=f"{type(e).__name__}: {e}")
        log(f"[damage] skipped: {status['error']}")
    meta = plan.meta if hasattr(plan, "meta") else plan.setdefault("meta", {})
    meta["damage"] = status
    return plan
