"""v4: tier-aware plan extraction (docs/modules/plan_beta.md "Part v4: tiers").

plan_beta was built and tuned on LiDAR scenes. Video and photo scenes have the same keys (D-003), but their surfaces
come from learned metric depth, which is noisier, and their wall surfaces are sparser after TSDF fusion. This module
holds everything that differs per tier, so the LiDAR path stays bit-identical (its overrides are empty):

  T-1  surface_from_raw  add a planar surface built from raw points (PCA normals) to the TSDF surface (video, photo)
  T-2  geom_sigma_m      surface-noise term on every wall offset (video, photo)
  T-3  folder seeds      photo: one room per folder, named after it, widened per folder, adjacency from links
  T-5  inferred_sigma_m  photo: a wall not measured on raw points is only the edge of a few view wedges (0.5 m)
  T-4  folder_hull_fill  photo: convex-hull filling per folder (OFF: it created an overlap, see Part v4)
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import watershed

from floorplan.plan.beta.params import BetaParams

# Every value is justified with its evidence in plan_beta.md Part v4 (v4.3 decisions, v4.4 results).
TIER_OVERRIDES: dict[str, dict] = {
    "lidar": {},
    "video": dict(surface_from_raw=True, geom_sigma_m=0.05, opening_outline=False),
    "photo": dict(geom_sigma_m=0.05, inferred_sigma_m=0.5, use_folder_seeds=True, use_folder_widen=True,
                  use_room_hints=False, min_room_m2=0.8, unvisited_min_enclosure=0.0, opening_outline=False),
}


def tier_of(info: dict | None, tier: str | None = None) -> str:
    t = (tier or (info or {}).get("tier") or "lidar").lower()
    return t if t in TIER_OVERRIDES else "lidar"


def params_for_tier(tier: str, base: BetaParams | None = None) -> BetaParams:
    """BetaParams for a tier: the LiDAR defaults plus that tier's overrides."""
    return replace(base or BetaParams(), tier=tier, **TIER_OVERRIDES.get(tier, {}))


# ---------------------------------------------------------------- T-1: surface from raw points

def raw_surface(raw_points: np.ndarray, raw_ray: np.ndarray, p: BetaParams):
    """Voxel-mean raw points with PCA normals, oriented against the viewing ray; planar voxels only.

    Why: the video TSDF keeps a voxel only after >= 5 integrations (V-9), so walls seen in a few keyframes vanish
    from `points` while they are present in `raw_points` (single_room: the partition between R1 and R2 and most outer
    walls were missing from the 1-2 m band). The raw points have no normals; a 3 cm voxel mean and a PCA over 24
    neighbours gives one, and only planar neighbourhoods are kept (a vertical wall, not clutter)."""
    P = raw_points.astype(np.float64)
    D = raw_ray.astype(np.float64)
    key = np.floor(P / p.raw_surface_voxel_m).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    S = np.zeros((cnt.size, 3))
    R = np.zeros((cnt.size, 3))
    np.add.at(S, inv, P)
    np.add.at(R, inv, D)
    C = S / cnt[:, None]
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
    k = min(p.raw_surface_k, len(C))
    if k < 4:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    _, idx = cKDTree(C).query(C, k=k)
    Q = C[idx] - C[idx].mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q) / k)
    N = v[:, :, 0]
    N[np.einsum("ij,ij->i", N, R) > 0] *= -1                  # face the camera, like TSDF normals
    ok = (w[:, 0] / np.maximum(w.sum(1), 1e-12) < p.raw_surface_max_curv) & (cnt >= p.raw_surface_min_count)
    return C[ok].astype(np.float32), N[ok].astype(np.float32)


def augment_surface(scene: dict, p: BetaParams) -> tuple[dict, dict | None]:
    """Shallow copy of the scene whose surface = TSDF points + raw-point surface (T-1). Returns (scene, report)."""
    if not p.surface_from_raw or "raw_points" not in scene or len(scene["raw_points"]) == 0:
        return scene, None
    C, N = raw_surface(scene["raw_points"], scene["raw_ray"], p)
    out = dict(scene)
    out["points"] = np.concatenate([scene["points"], C]).astype(np.float32)
    out["normals"] = np.concatenate([scene["normals"], N]).astype(np.float32)
    if "colors" in scene:
        out["colors"] = np.concatenate([scene["colors"], np.zeros((len(C), 3), scene["colors"].dtype)])
    for key in ("point_room",):                                 # per-point labels: unknown for the added points
        if key in scene:
            out[key] = np.concatenate([np.asarray(scene[key]), -np.ones(len(C), np.asarray(scene[key]).dtype)])
    return out, dict(tsdf_points=int(len(scene["points"])), raw_surface_points=int(len(C)))


# ---------------------------------------------------------------- T-2: surface-noise term

def add_geom_sigma(geom, p: BetaParams) -> None:
    """Every wall offset of the room gets sqrt(sigma^2 + geom_sigma^2) (in place)."""
    if p.geom_sigma_m <= 0:
        return
    geom.lines[:] = [replace(l, sigma=float(np.hypot(l.sigma, p.geom_sigma_m))) for l in geom.lines]


# ---------------------------------------------------------------- T-3: photo folders

def folder_cameras(scene: dict) -> dict[str, np.ndarray] | None:
    """{folder: (n, 2) plan positions of its placed photos}, from scene cam_room + traj (photo tier only)."""
    if "cam_room" not in scene or "traj" not in scene:
        return None
    rooms = np.asarray(scene["cam_room"]).astype(str)
    traj = np.asarray(scene["traj"])[:, [0, 2]]
    if len(rooms) != len(traj):
        return None
    return {f: traj[rooms == f] for f in sorted(set(rooms.tolist()))}


def folder_segmentation(free: np.ndarray, cams: dict[str, np.ndarray], grid, p: BetaParams):
    """One watershed marker per folder; returns (labels 0..N, {label: folder}, report).

    Seeds: the folder's cameras whose distance to the nearest obstacle is >= folder_seed_dt_frac of its best camera
    (spin-centre photos; a doorway photo stands in the door and would grow into the wrong room), each a disc of the
    operator radius inside free space. A camera outside free space snaps to the nearest free cell within
    folder_snap_m. Basins grow on -DT, so two folders meet at the narrowest place between them (the door). Folders
    are never merged; free space that no folder reaches is left unlabelled (it is not one of the user's rooms)."""
    dt = ndi.distance_transform_edt(free) * p.cell_m
    _, (ri, ci) = ndi.distance_transform_edt(~free, return_indices=True)
    snap_cells = p.folder_snap_m / p.cell_m
    markers = np.zeros(free.shape, np.int32)
    names, report = {}, {}
    rad = int(np.ceil(p.traj_radius_m / p.cell_m))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    disc = xx * xx + yy * yy <= rad * rad
    for k, (folder, uv) in enumerate(sorted(cams.items()), start=1):
        r, c, ok = grid.index(uv)
        r, c = np.clip(r, 0, grid.rows - 1), np.clip(c, 0, grid.cols - 1)
        cells = []
        for rr, cc, inside in zip(r, c, ok):
            if not inside:
                continue
            if not free[rr, cc]:
                sr, sc = ri[rr, cc], ci[rr, cc]
                if np.hypot(sr - rr, sc - cc) > snap_cells:
                    continue
                rr, cc = sr, sc
            cells.append((int(rr), int(cc), float(dt[rr, cc])))
        report[folder] = dict(cameras=int(len(uv)), seeds=0)
        if not cells:
            continue
        best = max(d for _, _, d in cells)
        seeds = [(rr, cc) for rr, cc, d in cells if d >= p.folder_seed_dt_frac * best]
        m = np.zeros(free.shape, bool)
        for rr, cc in seeds:
            r0, r1 = max(rr - rad, 0), min(rr + rad + 1, free.shape[0])
            c0, c1 = max(cc - rad, 0), min(cc + rad + 1, free.shape[1])
            m[r0:r1, c0:c1] |= disc[r0 - rr + rad:r1 - rr + rad, c0 - cc + rad:c1 - cc + rad]
        m &= free & (markers == 0)                    # first folder (sorted) keeps a contested cell: deterministic
        if m.any():
            markers[m] = k
            names[k] = folder
            report[folder]["seeds"] = len(seeds)
    smooth = ndi.gaussian_filter(dt, 1.0) * free
    labels = watershed(-smooth, markers, mask=free) if names else np.zeros(free.shape, np.int32)
    return labels.astype(np.int32), names, report


def folder_hull_fill(scene: dict, info: dict, grid, maps: dict, p: BetaParams) -> tuple[np.ndarray, dict]:
    """Photo-tier hole filling: each folder's observed free space (floor seen + rays carved by ITS photos + its
    camera discs) is filled to its convex hull, minus wall evidence.

    Why: 2-8 photos per room see the floor as separate view wedges with unseen gaps between them; the LiDAR hole
    filler (holes <= 6 m2 mostly bordered by free space) cannot close a gap that opens onto unseen space. A room is
    the space enclosed by the walls its photos saw, and rooms are close to convex; the hull of what one folder saw
    is bounded by those walls (barrier cells are removed again), so it does not cross into a neighbour. L-shaped
    rooms can be over-filled where the inner corner was not seen: a limitation (plan_beta.md Part v4)."""
    from skimage.morphology import convex_hull_image
    from floorplan.plan.beta.freespace import floor_observed, ray_carved
    names = [str(x) for x in np.asarray(scene.get("room_names", []))]
    if "raw_point_room" not in scene or not names:
        return maps["free"], {}
    rp = np.asarray(scene["raw_point_room"])
    cams = folder_cameras(scene) or {}
    floor_y = float(info["floor_y"])
    free = maps["free"].copy()
    rep = {}
    for k, f in enumerate(names):
        sel = rp == k
        if sel.sum() < 100:
            continue
        P, D, R = scene["raw_points"][sel], scene["raw_ray"][sel], scene["raw_range"][sel]
        seen = floor_observed(P, D, floor_y, grid, p) | ray_carved(P, D, R, floor_y, grid, p)
        if f in cams and len(cams[f]):
            seen |= grid.count(cams[f]) > 0
        if seen.sum() < 10:
            continue
        hull = convex_hull_image(seen) & ~maps["barrier"]
        add = hull & ~free
        free |= hull
        rep[f] = dict(seen_m2=round(float(seen.sum()) * p.cell_m ** 2, 2),
                      added_m2=round(float(add.sum()) * p.cell_m ** 2, 2))
    free = ndi.binary_opening(free, np.ones((3, 3), bool))
    return free, rep


# ---------------------------------------------------------------- T-6: photo rooms from per-room layouts (v3)

_SIDE_AX = {"+x": (0, 1.0), "-x": (0, -1.0), "+z": (1, 1.0), "-z": (1, -1.0)}


def _q(a: np.ndarray, value: float, unit: str, method: str, status: str):
    from floorplan.model import Measurement
    lo, hi = (float(x) for x in np.percentile(a, [2.5, 97.5]))
    return Measurement(float(value), min(lo, value), max(hi, value), unit, method, status)


def layout_plan(info: dict, fallback_rooms: list, cams: dict | None, scene: dict, seed: int = 0,
                use_polygon: bool | None = None):
    """Photo tier v3 (docs/modules/plan_beta.md Part v4, "v4.10"): one Manhattan rectangle per room folder from the
    folder's own spin layout (floorplan/photo/layout.py, info.room_layouts), placed at its spin centre, pushed apart
    where two rooms overlap, walls with Monte Carlo intervals per side. Rooms without a layout keep the free-space
    room of v4 (fallback_rooms, already named after their folder).

    v3-poly (use_polygon, default from params `layout_polygon`, on): a room whose layout carries an evidence-backed
    rectilinear polygon (floorplan/photo/polygon.py, lay["polygon_local"]) is built from it instead of the rectangle:
    one wall per polygon edge, Monte Carlo over every wall line, push-apart on the polygon's parts.

    Returns (rooms, walls, openings, footprint, report, layout_folders)."""
    from floorplan.model import Measurement, Opening, Room, Wall
    from floorplan.photo import polygon as PG                     # v3-poly polygon rooms
    pp = info.get("params") or {}
    use_poly = bool(pp.get("layout_polygon", True)) if use_polygon is None else bool(use_polygon)
    rel = float(pp.get("layout_side_sigma_rel", 0.0))
    S = int(pp.get("layout_mc", 400))
    gap = float(pp.get("layout_overlap_gap_m", 0.10))
    max_spread = float(pp.get("layout_max_spin_spread_m", 1.0))
    rng = np.random.default_rng(seed)
    lays = info.get("room_layouts") or {}
    rooms_info = info.get("rooms") or {}
    fb_of = {r.label: r for r in fallback_rooms}
    boxes, report = {}, {}
    poly_err = {}                                    # v3-poly: polygon failures per folder (rectangle kept)
    for folder, lay in sorted(lays.items()):
        if not lay.get("ok"):
            report[folder] = dict(used=False, reason=lay.get("reason"))
            continue
        sides = lay["sides_plan"]
        if lay.get("anchor"):
            spread = _spin_spread(scene, lay["anchor"].get("placed_spin_photos") or [])
            if spread is not None and spread > max_spread:
                # the placed 'spin' photos stand metres apart: not taken from one spot (e.g. corner shots), so the
                # common-centre layout does not hold (photo_tier.md v3, PT-17)
                report[folder] = dict(used=False, reason=f"placed spin photos {spread:.2f} m apart (> {max_spread} "
                                                         "m): not a spin from one spot; free-space room kept")
                continue
            c, how = np.array(lay["anchor"]["centre_uv"], float), "spin centre (placed spin photos)"
        elif folder in fb_of:
            # sizes are still the spin's own; only the POSITION is inferred (centre of the free-space fragment)
            c, how = np.array(fb_of[folder].polygon).mean(0), "inferred: free-space room centroid (spin not placed)"
        elif cams and folder in cams and len(cams[folder]):
            c, how = cams[folder].mean(0), "inferred: mean of the folder's placed photos (spin not placed)"
        else:
            report[folder] = dict(used=False, reason="no position: neither spin photos nor any photo placed")
            continue
        draws, val = {}, {}
        for s, (ax, sg) in _SIDE_AX.items():
            d = sides[s]
            sig = float(np.hypot(d["sigma"], rel * d["offset"]))
            x = rng.normal(d["offset"], sig, S)
            if d.get("lower_bound") is not None:
                x = np.maximum(x, d["lower_bound"])
            draws[s], val[s] = np.maximum(x, 0.1), float(d["offset"])
        boxes[folder] = dict(c=c, val=val, draws=draws, how=how, status={s: sides[s]["status"] for s in _SIDE_AX},
                             widen=float((rooms_info.get(folder) or {}).get("widen", 1.0) or 1.0),
                             anchored=bool(lay.get("anchor")), height=lay.get("height"))
        # ---- v3-poly polygon rooms (photo/polygon.py) -------------------------------------------------------------
        try:
            pg = PG.plan_polygon(lay, rel, S, rng) if use_poly else None
        except Exception as e:                       # never lose the room over the polygon: rectangle stands
            pg = None
            poly_err[folder] = f"{type(e).__name__}: {e}"
        if pg is not None:
            for s, (kind, a) in pg["side_line"].items():
                pos = float(abs(pg["u"][a] if kind == "u" else pg["v"][a]))
                if abs(pos - val[s]) > 0.01:         # a side the polygon moved (doorway photo on an INFERRED side)
                    val[s] = pos
                    draws[s] = np.maximum(np.abs(pg["DU"][:, a] if kind == "u" else pg["DV"][:, a]), 0.1)
        boxes[folder]["poly"] = pg
        # --------------------------------------------------------------------------------------------------------
    clipped = _clip_by_neighbours(boxes, gap, rel, S, rng)
    for f, b in boxes.items():                       # v3-poly: a side cut back by a neighbour -> the plain rectangle
        if b.get("poly") is not None:
            if f in clipped:
                b["poly"] = None
            else:
                PG.sync_sides(b["poly"], b["val"], b["draws"])
    fixed = [tuple(np.r_[np.min(r.polygon, 0)[0], np.max(r.polygon, 0)[0], np.min(r.polygon, 0)[1],
                         np.max(r.polygon, 0)[1]]) for r in fallback_rooms if r.label not in boxes]
    shifts = None
    if any(b.get("poly") is not None for b in boxes.values()):           # v3-poly: rooms made of several parts
        c0 = {f: b["c"].copy() for f, b in boxes.items()}
        try:
            shifts = PG.push_apart_parts(boxes, gap, fixed=fixed)
        except Exception as e:
            for f, b in boxes.items():
                if b.get("poly") is not None:
                    poly_err[f] = f"push-apart: {type(e).__name__}: {e}"
                b["c"], b["poly"] = c0[f], None
    if shifts is None:
        shifts = _push_apart(boxes, gap, fixed=fixed)
    out_rooms, out_walls, used = [], [], {}
    n_id = 0
    for folder in sorted(set(boxes) | set(fb_of)):
        if folder not in boxes:
            continue
        b = boxes[folder]
        rid = fb_of[folder].id if folder in fb_of else f"L{(n_id := n_id + 1)}"
        used[folder] = rid
        u0, u1 = b["c"][0] - b["val"]["-x"], b["c"][0] + b["val"]["+x"]
        v0, v1 = b["c"][1] - b["val"]["-z"], b["c"][1] + b["val"]["+z"]
        dx, dz = b["draws"]["+x"] + b["draws"]["-x"], b["draws"]["+z"] + b["draws"]["-z"]
        st = b["status"]
        all_meas = all(v == "measured" for v in st.values())
        meth = "photo v3: Manhattan rectangle from the folder's own spin photos; Monte Carlo over the 4 wall offsets"
        stat = "measured" if all_meas else "inferred"
        poly = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]                       # counter-clockwise in (u, v)
        # wall k goes from poly[k] to poly[k+1]: -z side, +x side, +z side, -x side
        spec = [("-z", dx, (0.0, 1.0)), ("+x", dz, (-1.0, 0.0)), ("+z", dx, (0.0, -1.0)), ("-x", dz, (1.0, 0.0))]
        wids = []
        pg = b.get("poly")
        if pg is not None:                           # v3-poly: polygon geometry first; any failure -> the rectangle
            try:
                geo = PG.plan_geometry(pg, b["c"])
            except Exception as e:
                pg = None
                poly_err[folder] = f"{type(e).__name__}: {e}"
        for k, (side, L, nrm) in enumerate(spec if pg is None else []):
            wid = f"{rid}-W{k + 1}"
            wids.append(wid)
            perp = ("-x", "+x") if side[1] == "z" else ("-z", "+z")
            ws = "measured" if all(st[q] == "measured" for q in perp) else "inferred"
            Lv = (u1 - u0) if side[1] == "z" else (v1 - v0)
            out_walls.append(Wall(wid, rid, poly[k], poly[(k + 1) % 4],
                                  _q(L, Lv, "m", meth + f"; wall offset {side}: {st[side]}", ws), nrm))
        A = dx * dz
        A_val, P_draws, P_val = (u1 - u0) * (v1 - v0), 2 * (dx + dz), 2 * ((u1 - u0) + (v1 - v0))
        if pg is not None:                           # ---- v3-poly polygon room: one wall per polygon edge ----
            P_xy, edges, A, P_draws, dx, dz = geo
            poly = [tuple(q) for q in P_xy]
            u0, u1, v0, v1 = P_xy[:, 0].min(), P_xy[:, 0].max(), P_xy[:, 1].min(), P_xy[:, 1].max()
            A_val = float(abs(np.dot(P_xy[:, 0], np.roll(P_xy[:, 1], -1))
                              - np.dot(P_xy[:, 1], np.roll(P_xy[:, 0], -1))) / 2)
            P_val = float(sum(e["value"] for e in edges))
            meth = ("photo v3-poly: rectilinear polygon from the folder's own photos (rectangle + "
                    + ", ".join(sorted({c["kind"] for c in pg["changes"]})) + "); Monte Carlo over every wall line")
            stat = "measured" if all(e["status"] == "measured" for e in edges) else "inferred"
            for k, e in enumerate(edges):
                wid = f"{rid}-W{k + 1}"
                wids.append(wid)
                out_walls.append(Wall(wid, rid, e["p0"], e["p1"], _q(e["draws"], e["value"], "m", meth
                                      + f"; wall line ({e['kind']}): {e['line_status']}", e["status"]), e["normal"]))
        b["area_draws"] = A                          # ------------------------------------------------------------
        h = b["height"]
        fb = fb_of.get(folder)
        if h and str(h.get("source", "")).startswith("ceiling photo"):
            # D-057: the room's own ceiling photo (scaled by matches, or fused with the residential prior) beats the
            # free-space room's ceiling points, which mix in every photo's depth scale (sim: living room 3.56 m vs
            # 2.80 m true after one badly scaled ceiling photo entered the cloud)
            ceil = Measurement.from_sigma(h["value"], h["sigma"], "m", "photo v3: " + str(h.get("source")),
                                          "measured" if h.get("linked") else "inferred")
        elif fb is not None and fb.ceiling_height is not None and fb.ceiling_height.value is not None:
            ceil = fb.ceiling_height
        elif h:
            ceil = Measurement.from_sigma(h["value"], h["sigma"], "m", "photo v3: ceiling minus floor points of the "
                                          "spin photos", "measured")
        elif _global_ceiling(info) is not None:
            # D-049: the spin photos (aimed slightly down) did not see this room's ceiling; use the whole scene's
            # ceiling-minus-floor (all photos), inferred, +-10 cm. Sim: global 2.61 m vs rooms 2.56-2.80 m.
            ceil = Measurement.from_sigma(_global_ceiling(info), 0.10, "m", "photo v3: whole-scene ceiling minus floor "
                                          "(this room's spin photos did not see its ceiling)", "inferred")
        else:
            ceil = Measurement.missing("m", "photo v3", "ceiling not seen by the spin photos")
        Lb, Wb = (dx, dz) if (u1 - u0) >= (v1 - v0) else (dz, dx)
        out_rooms.append(Room(rid, folder, poly, wids, _q(A, A_val, "m2", meth, stat),
                              _q(P_draws, P_val, "m", meth, stat), ceil,
                              (_q(Lb, max(u1 - u0, v1 - v0), "m", meth + "; length", stat),
                               _q(Wb, min(u1 - u0, v1 - v0), "m", meth + "; width", stat)),
                              float(fb.floor_level) if fb is not None else 0.0))
        report[folder] = dict(used=True, room=rid, centre=b["how"], sides=st, shift_m=shifts.get(folder),
                              clipped_by_neighbour=clipped.get(folder),
                              anchored=b["anchored"])
        if pg is not None:                           # v3-poly
            report[folder]["polygon"] = dict(vertices=len(poly), changes=pg["changes"])
        if folder in poly_err:
            report[folder]["polygon_error"] = poly_err[folder]
    # rooms with no layout: keep the v4 free-space room as it is (its walls come from the caller)
    tot = np.zeros(S)
    for folder in used:
        b = boxes[folder]
        tot += b["area_draws"]                       # v3-poly: rectangle or polygon area draws
    kept = [r for r in fallback_rooms if r.label not in used]
    fixed = sum((r.floor_area.value or 0.0) for r in kept)
    for r in kept:                               # kept free-space rooms: their own (widened) interval as a Gaussian
        m = r.floor_area
        if m.value is not None and m.lo is not None and m.hi is not None:
            tot += rng.normal(0.0, (m.hi - m.lo) / 3.92, S)
    fp_val = sum(r.floor_area.value for r in out_rooms) + fixed
    fp = _q(tot + fixed, fp_val, "m2", "photo v3: sum of the room rectangles (pushed apart: no overlap) plus the "
            "free-space rooms of folders without a layout", "inferred")
    openings = _pair_openings(info, scene, used, out_walls)
    return out_rooms, out_walls, openings, fp, report, used


def _global_ceiling(info: dict) -> float | None:
    """Whole-scene ceiling height from the photo front-end, if plausible (2.0-4.0 m)."""
    g = info.get("ceiling_height_global")
    if g is None and info.get("ceiling_y") is not None and info.get("floor_y") is not None:
        g = float(info["ceiling_y"]) - float(info["floor_y"])
    if isinstance(g, dict):
        g = g.get("value")
    return float(g) if g is not None and 2.0 <= float(g) <= 4.0 else None


def _clip_by_neighbours(boxes: dict, gap: float, rel: float, S: int, rng) -> dict:
    """A room cannot contain another room's spin centre (the photographer stood inside THAT room). When a wall
    offset of room A reaches past the spin centre of an anchored room B that lies within A's span, A's wall was
    fitted to B's wall seen through a door: A's side is cut back to B's facing wall (minus a wall-face gap) and
    becomes 'inferred' with a wide sigma. Only rooms whose positions come from placed spin photos take part."""
    out = {}
    anch = [f for f in boxes if boxes[f]["anchored"]]
    for a in anch:
        A = boxes[a]
        for s_, (ax, sg) in _SIDE_AX.items():
            o_ax = 1 - ax
            lo_t = A["c"][o_ax] - A["val"]["-z" if o_ax else "-x"]
            hi_t = A["c"][o_ax] + A["val"]["+z" if o_ax else "+x"]
            for b in anch:
                if b == a:
                    continue
                B = boxes[b]
                d = (B["c"] - A["c"]) * sg
                if not (0 < d[ax] < A["val"][s_] and lo_t < B["c"][o_ax] < hi_t):
                    continue
                opp = ("-" if sg > 0 else "+") + ("x" if ax == 0 else "z")
                face = d[ax] - B["val"][opp] - gap            # B's wall facing A, seen from A's centre
                new = face if face > 0.3 else d[ax] / 2
                if new < A["val"][s_]:
                    A["val"][s_] = float(new)
                    sig = max(rel * new, 0.2)
                    A["draws"][s_] = np.maximum(rng.normal(new, sig, S), 0.1)
                    A["status"][s_] = "inferred"
                    out.setdefault(a, {})[s_] = dict(by=b, offset_m=round(float(new), 3))
    return out


def _spin_spread(scene: dict, names: list[str]) -> float | None:
    """Largest distance (m) of a placed spin photo from their mean position (plan view)."""
    cn = [str(x) for x in np.asarray(scene.get("cam_names", []))]
    P = np.array([np.asarray(scene["traj"])[cn.index(n), [0, 2]] for n in names if n in cn])
    if len(P) < 2:
        return None
    return float(np.max(np.linalg.norm(P - P.mean(0), axis=1)))


def _push_apart(boxes: dict, gap: float, iters: int = 200, fixed: list | None = None) -> dict:
    """Move rectangles until no two overlap (a wall-face gap apart). The room whose position is least certain moves:
    not anchored by its spin photos first, then the larger placement widening, then the smaller room. Each move is
    along the axis of least penetration. Returns {folder: total shift (m)}."""
    def rect(b):
        return (b["c"][0] - b["val"]["-x"], b["c"][0] + b["val"]["+x"], b["c"][1] - b["val"]["-z"],
                b["c"][1] + b["val"]["+z"])
    shift = {f: np.zeros(2) for f in boxes}
    names = sorted(boxes)
    for _ in range(iters):
        moved = False
        for i, a in enumerate(names):
            for bname in names[i + 1:]:
                A, B = rect(boxes[a]), rect(boxes[bname])
                ou, ov = min(A[1], B[1]) - max(A[0], B[0]), min(A[3], B[3]) - max(A[2], B[2])
                if ou <= 1e-3 or ov <= 1e-3:
                    continue                                     # only real overlaps; the gap is the push target
                pu, pv = ou + gap, ov + gap
                key = lambda f: (boxes[f]["anchored"], -boxes[f]["widen"], (rect(boxes[f])[1] - rect(boxes[f])[0])
                                 * (rect(boxes[f])[3] - rect(boxes[f])[2]))
                mover, other = (a, bname) if key(a) < key(bname) else (bname, a)
                M, O = rect(boxes[mover]), rect(boxes[other])
                d = np.zeros(2)
                if pu <= pv:
                    d[0] = pu if (M[0] + M[1]) >= (O[0] + O[1]) else -pu
                else:
                    d[1] = pv if (M[2] + M[3]) >= (O[2] + O[3]) else -pv
                boxes[mover]["c"] = boxes[mover]["c"] + d
                shift[mover] += d
                moved = True
        for f in names:                          # rooms kept from the free-space extraction do not move
            for O in fixed or []:
                M = rect(boxes[f])
                ou, ov = min(M[1], O[1]) - max(M[0], O[0]), min(M[3], O[3]) - max(M[2], O[2])
                if ou <= 1e-3 or ov <= 1e-3:
                    continue
                d = np.zeros(2)
                if ou <= ov:
                    d[0] = (ou + gap) if (M[0] + M[1]) >= (O[0] + O[1]) else -(ou + gap)
                else:
                    d[1] = (ov + gap) if (M[2] + M[3]) >= (O[2] + O[3]) else -(ov + gap)
                boxes[f]["c"] = boxes[f]["c"] + d
                shift[f] += d
                moved = True
        if not moved:
            break
    return {f: round(float(np.hypot(*s)), 3) for f, s in shift.items()}


def _pair_openings(info: dict, scene: dict, room_of: dict, walls: list) -> list:
    """A doorway pair whose two photos are both placed is a door: both photos were taken standing in it. Its centre
    is the mean of the two camera positions; its width is not measured (doorway photos look along the room)."""
    from floorplan.model import Measurement, Opening
    names = [str(x) for x in np.asarray(scene.get("cam_names", []))]
    if not names:
        return []
    pos = {n: np.asarray(scene["traj"])[i, [0, 2]] for i, n in enumerate(names)}
    links = set()
    for key in (info.get("room_links_v2") or {}):
        a, _, b = key.partition("|")
        links.add(frozenset((a, b)))
    out = []
    for pr in ((info.get("protocol_v2") or {}).get("doorway_pairs") or []):
        a, b = pr["a"], pr["b"]
        ra, rb = pr["rooms"]
        if a not in pos or b not in pos or ra not in room_of or rb not in room_of or frozenset((ra, rb)) not in links:
            continue
        c = (pos[a] + pos[b]) / 2
        wid = []
        for r in (room_of[ra], room_of[rb]):
            ws = [w for w in walls if w.room_id == r]
            if ws:
                dist = [_seg_dist(c, np.array(w.p0), np.array(w.p1)) for w in ws]
                wid.append(ws[int(np.argmin(dist))].id)
        oid = f"PO{len(out) + 1}"
        for w in walls:
            if w.id in wid:
                w.opening_ids.append(oid)
        out.append(Opening(oid, "door", wid, [room_of[ra], room_of[rb]], (float(c[0]), float(c[1])),
                           Measurement.missing("m", "photo v3 doorway pair", "door width not measured from photos"),
                           confidence=0.6, evidence=f"doorway pair {a} ~ {b} (dt {pr.get('dt_s')} s)"))
    return out


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))
