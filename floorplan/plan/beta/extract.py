"""Space-first plan extraction: aligned LiDAR scene -> floorplan.model.Plan (tier 'lidar').

Pipeline (each step in its own module, see docs/modules/plan_beta.md):
  1. freespace.py  interior free space on a 2 cm raster
  2. segment.py    distance transform + watershed + door-aware merging -> room regions
  3. walls.py      line arrangement polygon per room, overlap resolution, raw-point wall refinement
  4. levels.py     per-room floor level and ceiling height
  5. openings.py   doors / passages / windows (with mirror and glass handling)
  6. measure.py    lengths, areas, footprint with 95% intervals

v2 (docs/modules/plan_beta.md "Part v2", ablation in docs/modules/plan_v2_ablation.md):
  1b. lines.py     alpha's long wall lines as segmentation cuts (paired or tall faces only)
  3c. walls.py     noise-weighted fits, tape-height position band, drift/inconsistency terms in sigma
  3d. walls.py     canonical outline after refinement (unsupported shallow steps and bumps removed)
  4.  levels.py    ceiling double-layer rule; 5. openings.py alpha's tall-face jamb rule
  -   photo-folder room hints (X-4) and meta.coverage_warning (B-8)

v4 (docs/modules/plan_beta.md "Part v4: tiers", tiers.py): per-tier parameters; LiDAR unchanged.
  0.  tiers.py     video/photo: raw-point surface added to the TSDF surface (T-1)
  2'. tiers.py     photo: one seed per room folder, no merging across folders (T-3)
  3e. tiers.py     video/photo: surface-noise term on every wall offset (T-2)
  -   photo: rooms named after folders, per-folder widening, adjacency from the photo links (T-3)

v6 (I-011, docs/ISSUES.md), LiDAR only (video/photo: params.opening_outline off in tiers.TIER_OVERRIDES):
  3f. walls.py     room boundary on the wall's inner face at openings: header far face (wide openings), window
                   recesses and wide-opening reveals closed, so one wall side is one wall carrying its opening
"""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import LineString, Point, Polygon

from floorplan.model import Adjacency, Measurement, Opening, Plan, Room, Wall
from floorplan.plan.beta import openings as O
from floorplan.plan.beta.freespace import interior_free_space
from floorplan.plan.beta.grid import Grid, PointIndex
from floorplan.plan.beta.lines import detect_sep_lines, header_cut_mask, separation_mask
from floorplan.plan.beta.levels import LevelPoints, room_levels
from floorplan.plan.beta.measure import edge_lengths, footprint, room_measurements
from floorplan.plan.beta.params import BetaParams
from floorplan.plan.beta.segment import Neck, region_necks, segment_rooms, split_by_stays
from floorplan.plan.beta import tiers as T
from floorplan.plan.beta.walls import (Bump, Line, OpeningEvidence, RoomGeometry, choose_visit, clean_polygon,
                                       edge_normals, outline_at_openings, polygon_to_geometry, refine_lines,
                                       resolve_overlaps, room_polygon, room_visits, vertical_points, canonicalize)


@dataclass
class RoomBuild:
    """Working state of one room while the plan is assembled."""
    label: int
    geom: RoomGeometry
    bumps: list[Bump]
    visited: bool
    enclosure: float
    rid: str = ""
    meas: dict = field(default_factory=dict)       # measure.room_measurements output
    floor_level: float = 0.0
    ceiling: Measurement | None = None
    faces: list[O.WallFace] = field(default_factory=list)
    outline_notes: list[dict] = field(default_factory=list)   # v6 I-011: changes made by walls.outline_at_openings

    @property
    def polygon(self) -> Polygon:
        return Polygon(self.geom.vertices())


def extract_plan(scene: dict, info: dict, capture_id: str, params: BetaParams | None = None,
                 tier: str | None = None) -> Plan:
    plan, _ = extract_plan_debug(scene, info, capture_id, params, tier)
    return plan


def extract_plan_debug(scene: dict, info: dict, capture_id: str, params: BetaParams | None = None,
                       tier: str | None = None):
    """Same as extract_plan, also returns the intermediate rasters and timings for the debug figure.

    v3 (S-2, docs/modules/plan_beta.md "Part v3"): with phase_votes > 1 the plan is built at several sub-cell raster
    phases and the medoid plan (the one whose walls agree with the most walls of the other phases) is returned.

    v4: `tier` (else info["tier"], else "lidar") selects the tier's parameters (tiers.TIER_OVERRIDES) when `params`
    is None. Explicit `params` are used as given (ablations); LiDAR has no overrides, so its plans are unchanged."""
    p = params or T.params_for_tier(T.tier_of(info, tier))
    if p.phase_votes <= 1:
        return _extract_single(scene, info, capture_id, p)
    return _phase_vote(scene, info, capture_id, p)


def _extract_single(scene: dict, info: dict, capture_id: str, p: BetaParams):
    rng = np.random.default_rng(p.seed)
    t0, timing = time.time(), {}

    def tick(name):
        timing[name] = round(time.time() - t0 - sum(timing.values()), 2)

    floor_y = float(info["floor_y"])
    scene, surface_report = T.augment_surface(scene, p)                    # v4 T-1 (no-op for LiDAR)
    grid = Grid.around(scene["points"][:, [0, 2]], p.cell_m, lattice=p.grid_lattice, phase=tuple(p.grid_phase))
    maps = interior_free_space(scene, info, grid, p)
    tick("free_space")
    cut = None
    if p.use_sep_lines:
        low_lines = detect_sep_lines(scene["points"], scene["normals"], floor_y, grid, p)
        cut = separation_mask(low_lines, grid, p)
        if p.use_header_cuts:                                              # v5 D-046: wide openings
            hcut, headers = header_cut_mask(scene["points"], scene["normals"], floor_y, grid, p, low_lines)
            cut = cut | hcut
            maps["header_cut"] = hcut
            maps["headers"] = headers
        maps["cut"] = cut
    cams = T.folder_cameras(scene) if p.use_folder_seeds else None
    folder_of, folder_report, hull_report = {}, None, None
    if cams:                                                               # v4 T-3: one room per photo folder
        if p.folder_hull_fill:
            maps["free"], hull_report = T.folder_hull_fill(scene, info, grid, maps, p)
        labels, folder_of, folder_report = T.folder_segmentation(maps["free"], cams, grid, p)
        necks = (region_necks(labels, ndi_distance(maps["free"], p), maps["barrier"], p)
                 if len(folder_of) > 1 and len(np.unique(labels[labels > 0])) > 1 else {})
        hints = None
    else:
        labels, necks, seg_dt = segment_rooms(maps["free"], maps["barrier"], p, cut, maps["traj_disc"])
        if p.stay_split and "timestamps" in scene:                         # D-079 (video): split between stays
            labels, stay_report = split_by_stays(labels, seg_dt, maps["barrier"], scene["traj"][:, [0, 2]],
                                                 np.asarray(scene["timestamps"], float), grid, p)
            if stay_report["split"]:
                necks = region_necks(labels, seg_dt, maps["barrier"], p)
            maps["stay_split"] = stay_report
        hints = merge_by_room_hints(labels, necks, scene, grid, floor_y, p)
    tick("segmentation")
    index = PointIndex(scene["raw_points"], scene["raw_ray"], scene["raw_range"], scene.get("raw_frame"))
    tick("point_index")
    rooms, dropped = _build_rooms(labels, scene, floor_y, grid, maps, index, p)
    for r in rooms:
        T.add_geom_sigma(r.geom, p)                                        # v4 T-2 (no-op for LiDAR)
    tick("walls")
    level_pts = LevelPoints(scene["raw_points"], scene["raw_ray"], scene["raw_range"], floor_y, p)
    for r in rooms:
        r.floor_level, _, r.ceiling = room_levels(r.polygon, level_pts, r.floor_level, p)
    tick("levels")
    for r in rooms:
        r.meas = room_measurements(r.geom, p.mc_samples, rng)
    walls = _make_walls(rooms)
    partners = _thickness(rooms, walls, p)
    tick("wall_assembly")
    necks = _necks_on_headers(necks, maps.get("headers") or [], grid)       # v5 D-046: wide openings
    openings, notes = _find_openings(rooms, necks, labels, grid, walls, scene, index, p)
    tick("openings")
    plan_rooms = _make_rooms(rooms, folder_of)
    fp = footprint([r.geom for r in rooms], max(p.mc_samples // 2, 50), rng)
    adjacency = _adjacency(openings, partners, {w.id: w.room_id for w in walls})
    tier_meta = dict(tier=p.tier, surface=surface_report)
    if folder_of:
        rid_of = {folder_of[r.label]: r.rid for r in rooms if r.label in folder_of}
        adjacency, tier_meta["adjacency_evidence"] = _folder_adjacency(adjacency, info, rid_of)
        tier_meta["folders"] = dict(seeds=folder_report, hull_fill=hull_report, room_of_folder=rid_of,
                                    folders_without_room=sorted(set(cams) - set(rid_of)))
        if p.use_folder_widen:
            tier_meta["widen"] = _folder_widen(plan_rooms, walls, fp, info, rid_of)
        if info.get("room_layouts"):                                       # photo v3: rooms from spin layouts
            plan_rooms, walls, openings, fp, adj_l, tier_meta["layouts"] = _layout_rooms(
                scene, info, cams, plan_rooms, walls, openings, fp, p)
            adjacency = adj_l if adj_l is not None else adjacency
    tick("measurements")
    meta = dict(extractor="plan_beta v2 (space-first: free space + wall-line cuts -> distance-transform watershed -> "
                          "line arrangement -> canonical outline)",
                tier_v4=tier_meta, room_hints=hints, stay_split=maps.get("stay_split"), coverage_warning=_coverage_warning(fp, maps, plan_rooms_area(rooms), p),
                opening_outline={r.rid: r.outline_notes for r in rooms if r.outline_notes},      # v6 I-011
                params=p.to_dict(), floor_y=floor_y, ceiling_y_global=info.get("ceiling_y"),
                dropped_regions=dropped, mirrors=[n for n in notes if n["kind"] == "mirror"],
                window_candidates=[n for n in notes if n["kind"] == "window_candidate"],
                weak_links=[n for n in notes if n["kind"] == "weak_link"],
                wall_tilt_deg={w.id: round(float(np.degrees(np.arctan(t))), 3) for w, t in _tilts(rooms, walls)},
                timing_s=timing, runtime_s=round(time.time() - t0, 2),
                grid=dict(u0=grid.u0, v0=grid.v0, cell=grid.cell, rows=grid.rows, cols=grid.cols))
    pillars = {v["room"]: v["pillars"] for v in ((tier_meta.get("layouts") or {}).get("per_folder") or {}).values()
               if isinstance(v, dict) and v.get("pillars")}
    if pillars:                                  # D-083 photo tier: pillars / wall steps per room id (plan frame)
        meta["pillars"] = pillars
    plan = Plan(capture_id, p.tier, plan_rooms, walls, openings, adjacency, fp, meta=meta)
    debug = dict(grid=grid, maps=maps, labels=labels, rooms=rooms, necks=necks, traj=scene["traj"][:, [0, 2]])
    return plan, debug


# ---------------------------------------------------------------- v3: raster-phase vote (S-2)

def vote_phases(n: int) -> list[tuple[float, float]]:
    """n sub-cell phases (in cells), always starting with (0, 0): a k x k lattice for n = k^2, else a Halton-like
    low-discrepancy set. 4 -> the 2 x 2 half-cell lattice, 9 -> thirds."""
    k = int(round(np.sqrt(n)))
    if k * k == n:
        return [(i / k, j / k) for j in range(k) for i in range(k)]
    def vdc(i, b):
        f, r = 1.0, 0.0
        while i:
            f /= b
            r += f * (i % b)
            i //= b
        return r
    return [(vdc(i, 2), vdc(i, 3)) for i in range(n)]


_VOTE_JOB: tuple | None = None          # (scene, info, capture_id): inherited by forked workers, never pickled


def _vote_worker(p: BetaParams):
    scene, info, capture_id = _VOTE_JOB
    return _extract_single(scene, info, capture_id, p)


def wall_agreement(a: Plan, b: Plan, tol_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Per wall of a and of b: True if the other plan has a wall facing the same way whose two ends both lie within
    max(tol_m, 0.5% L) of this wall's ends (the repeatability gate, applied to plan vs plan in one frame)."""
    def arr(p):
        P0 = np.array([w.p0 for w in p.walls], float).reshape(-1, 2)
        P1 = np.array([w.p1 for w in p.walls], float).reshape(-1, 2)
        N = np.array([w.normal for w in p.walls], float).reshape(-1, 2)
        return P0, P1, N, np.linalg.norm(P1 - P0, axis=1)
    A0, A1, An, AL = arr(a)
    B0, B1, Bn, BL = arr(b)
    if not len(AL) or not len(BL):
        return np.zeros(len(AL), bool), np.zeros(len(BL), bool)
    same_dir = (An @ Bn.T) > 0.99
    d_fw = np.maximum(np.linalg.norm(A0[:, None] - B0[None], axis=2), np.linalg.norm(A1[:, None] - B1[None], axis=2))
    d_bw = np.maximum(np.linalg.norm(A0[:, None] - B1[None], axis=2), np.linalg.norm(A1[:, None] - B0[None], axis=2))
    tol = np.maximum(tol_m, 0.005 * np.maximum(AL[:, None], BL[None]))
    ok = same_dir & (np.minimum(d_fw, d_bw) <= tol)
    return ok.any(1), ok.any(0)


def _phase_vote(scene: dict, info: dict, capture_id: str, p: BetaParams):
    """S-2: build the plan at several sub-cell raster phases, return the medoid.

    Why: every raster stage (free space, watershed, arrangement at 50% coverage, mask lines) quantises wall positions
    to the 2 cm cell, so where a wall falls inside a cell decides some outlines (B-5: a 1.3 cm shift changed 41-48%
    of the walls when the phase was pinned). A second capture always has a different phase. The phase is a nuisance
    variable; the medoid over phases is the plan least affected by it (a robust, majority-style choice that keeps a
    real, internally consistent plan: rooms, walls and openings all come from one build, so nothing can overlap).
    The share of phases that reproduce each wall is reported in meta.phase_vote.wall_support (a topology-stability
    flag the intervals cannot express)."""
    global _VOTE_JOB
    t0 = time.time()
    phases = vote_phases(p.phase_votes)
    ps = [replace(p, phase_votes=1, grid_phase=(p.grid_phase[0] + du, p.grid_phase[1] + dv)) for du, dv in phases]
    results = None
    if p.phase_workers > 1:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            _VOTE_JOB = (scene, info, capture_id)
            with ProcessPoolExecutor(min(p.phase_workers, len(ps)), mp_context=mp.get_context("fork")) as ex:
                results = list(ex.map(_vote_worker, ps))
        except Exception:  # noqa: BLE001 - fall back to sequential; the result is the same
            results = None
        finally:
            _VOTE_JOB = None
    if results is None:
        results = [_extract_single(scene, info, capture_id, q) for q in ps]
    n = len(results)
    agree = np.zeros((n, n))
    support = [np.zeros(len(r[0].walls)) for r in results]
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = wall_agreement(results[i][0], results[j][0], p.phase_agree_tol_m)
            agree[i, j] = ai.sum()
            agree[j, i] = aj.sum()
            support[i] += ai
            support[j] += aj
    # score = mean over j of the Dice agreement (walls of i reproduced by j + walls of j reproduced by i) / (n_i + n_j).
    # Normalised: a raw count would favour builds with MORE walls (measured: it chose a fragmented 80-wall build).
    # The first phase wins ties.
    nw = np.array([max(len(r[0].walls), 1) for r in results], float)
    dice = (agree + agree.T) / (nw[:, None] + nw[None, :])
    np.fill_diagonal(dice, 0.0)
    score = dice.sum(1) / max(n - 1, 1)
    best = int(np.argmax(score))
    plan, debug = results[best]
    sup = (support[best] + 1) / n                          # the chosen build itself counts as one vote
    plan.meta["phase_vote"] = dict(
        phases=[list(map(float, ph)) for ph in phases], chosen=best, chosen_phase=list(map(float, phases[best])),
        score=[round(float(x), 4) for x in score], walls=[len(r[0].walls) for r in results], rooms=[len(r[0].rooms) for r in results],
        footprint_m2=[round(float(r[0].footprint_area.value), 3) for r in results],
        wall_support={w.id: round(float(s), 3) for w, s in zip(plan.walls, sup)},
        walls_unstable=[w.id for w, s in zip(plan.walls, sup) if s < 0.5],
        runtime_s=round(time.time() - t0, 2))
    plan.meta["runtime_s"] = round(time.time() - t0, 2)
    return plan, debug


# ---------------------------------------------------------------- v2: room hints (X-4) and coverage warning (B-8)

def merge_by_room_hints(labels: np.ndarray, necks: dict, scene: dict, grid: Grid, floor_y: float,
                        p: BetaParams) -> dict | None:
    """X-4: photo-tier scenes carry the user's room folder of every raw point (raw_point_room). The folder is the
    user's own statement of "this is one room", so two touching regions whose floor points vote for the same
    folder are merged (in place). Only merging: a folder never splits a region, because a photo of the next room
    taken through a doorway would otherwise cut a room in two. Returns what was done, for plan.meta."""
    if not p.use_room_hints or "raw_point_room" not in scene:
        return None
    P = scene["raw_points"]
    near_floor = np.abs(P[:, 1] - floor_y) < 0.3
    r, c, ok = grid.index(P[near_floor][:, [0, 2]])
    lab = labels[r[ok], c[ok]]
    room = np.asarray(scene["raw_point_room"])[near_floor][ok].astype(np.int64)
    major = {}
    for l in np.unique(lab[lab > 0]):
        votes = np.bincount(room[lab == l])
        if votes.sum() >= p.hint_min_points and votes.max() >= p.hint_min_share * votes.sum():
            major[int(l)] = int(np.argmax(votes))
    merged = []
    for (a, b) in sorted(necks):
        if a in major and b in major and major[a] == major[b] and (labels == a).any() and (labels == b).any():
            labels[labels == b] = a
            merged.append([a, b])
    for a, b in merged:                                       # necks between merged regions are inside a room now
        necks.pop((a, b), None)
        for key in [k for k in necks if b in k]:
            other = key[0] if key[1] == b else key[1]
            new = tuple(sorted((a, other)))
            if new not in necks and other != a:
                necks[new] = necks[key]
            necks.pop(key)
    return dict(source="raw_point_room (photo folders)", region_folder=major, merged=merged)


def plan_rooms_area(rooms: list[RoomBuild]) -> float:
    return float(sum(r.polygon.area for r in rooms))


def _coverage_warning(fp: Measurement, maps: dict, rooms_area: float, p: BetaParams) -> str | None:
    """B-8: an empty or tiny plan must not look like a normal result (judge J-I-6: a degenerate crop returned an
    empty plan with status OK). The CLI turns a warning into a non-zero exit code."""
    seen = float(maps["floor_obs"].sum()) * p.cell_m ** 2
    if rooms_area < p.coverage_min_m2:
        return (f"plan covers {rooms_area:.1f} m2 (< {p.coverage_min_m2} m2): capture too partial or degenerate; "
                f"{seen:.1f} m2 of floor was observed")
    return None


# ---------------------------------------------------------------- rooms and walls

def _build_rooms(labels, scene, floor_y, grid, maps, index, p: BetaParams):
    """Polygons for every region, overlaps resolved, refined on raw points, then validated."""
    seen_open = maps["floor_obs"] | maps["carved"]
    vertical = vertical_points(scene["points"], scene["normals"], floor_y, p)
    tall = (vertical_points(scene["points"], scene["normals"], floor_y, p, (p.wall_band_lo_m, p.wall_band_hi_m))
            if p.enclosed_cells else None)
    polys, lines, extents, bumps = {}, {}, {}, {}
    for lab in range(1, labels.max() + 1):
        polys[lab], lines[lab], extents[lab], bumps[lab] = room_polygon(labels, lab, vertical, grid, seen_open, p,
                                                                        tall)
    polys = resolve_overlaps(polys, extents, grid)
    all_offsets = [l.offset for ls in lines.values() for l in ls]
    shared = [Line(l.axis, l.offset, 0.0, inferred=True) for ls in lines.values() for l in ls]
    traj = scene["traj"][:, [0, 2]]
    rooms, dropped, raster_lines = [], [], {}
    for lab, poly in polys.items():
        poly = clean_polygon(poly, all_offsets, p) if poly is not None else None
        if poly is None or poly.area < p.min_room_m2 or len(poly.exterior.coords) < 5:
            dropped.append(dict(region=int(lab), reason="no valid polygon after overlap resolution"))
            continue
        geom = polygon_to_geometry(poly, lines[lab] + shared, lab)
        raster = [replace(l) for l in geom.lines]
        visit = _visit_for(geom, poly, scene, index, floor_y, p)
        refine_lines(geom, index, floor_y, p, visit)
        raster_lines[lab] = raster
        visited = bool(contains_xy(Polygon(geom.vertices()), traj[:, 0], traj[:, 1]).any())
        enclosure = _enclosure(geom)
        dwell = dwell_s(Polygon(geom.vertices()), scene) if p.brief_entry_s > 0 else None
        if visited and dwell is not None and dwell < p.brief_entry_s:          # D-079 (video): a step into a space
            if not p.brief_keep_enclosed or enclosure < p.unvisited_min_enclosure:   # is not a visit of it
                P = Polygon(geom.vertices())
                dropped.append(dict(region=int(lab), reason=f"partially observed: the camera was inside for only "
                                    f"{dwell:.1f} s (< {p.brief_entry_s} s); not a room, not in the footprint",
                                    area_m2=round(float(P.area), 2), dwell_s=round(dwell, 2), partial=True,
                                    polygon=[[round(float(x), 3), round(float(y), 3)] for x, y in P.exterior.coords]))
                continue
        if not visited and enclosure < p.unvisited_min_enclosure:
            dropped.append(dict(region=int(lab), reason=f"never entered and only {enclosure:.0%} of its boundary "
                                "is measured wall (space seen through an opening or glass)",
                                area_m2=round(float(Polygon(geom.vertices()).area), 2)))
            continue
        rooms.append(RoomBuild(lab, geom, bumps[lab], visited, enclosure, floor_level=floor_y))
    refined = {r.label: deepcopy(r.geom) for r in rooms}
    try:
        _undo_overlapping_refinements(rooms, raster_lines, p)
    except Exception as e:                      # D-058: a GEOS topology error here must never stop a live run;
        import logging                          # the rooms keep their refined lines (overlaps are pushed apart later)
        logging.getLogger(__name__).warning("undo_overlapping_refinements skipped: %s", e)
        for r in rooms:                         # D-084: all of them, not a half-done undo (it left a room on take1
            r.geom = refined[r.label]           # before r2 crossing itself, and GEOS stopped the plan step)
    _canonical_outlines(rooms, index, floor_y, p)
    _opening_outlines(rooms, scene, floor_y, maps, index, p)                   # v6 I-011
    rooms.sort(key=lambda r: -r.polygon.area)
    for i, r in enumerate(rooms):
        r.rid = f"R{i + 1}"
    return rooms, dropped


def dwell_s(poly: Polygon, scene: dict, max_step_s: float = 0.5) -> float:
    """D-079: seconds the camera spent inside `poly` (sum of the pose intervals, each capped at 0.5 s so a gap in
    the track does not count as time spent). Without timestamps every pose counts 1/30 s."""
    traj = scene["traj"][:, [0, 2]]
    if "timestamps" in scene and len(scene["timestamps"]) == len(traj):
        dt = np.diff(np.asarray(scene["timestamps"], float), append=np.nan)
        dt = np.clip(np.nan_to_num(dt, nan=np.nanmedian(dt[:-1]) if len(dt) > 1 else 1 / 30), 0.0, max_step_s)
    else:
        dt = np.full(len(traj), 1 / 30)
    return float(dt[contains_xy(poly, traj[:, 0], traj[:, 1])].sum())


def _visit_for(geom: RoomGeometry, poly: Polygon, scene: dict, index: PointIndex, floor_y: float,
               p: BetaParams) -> tuple[int, int] | None:
    """X-1: the camera visit used to measure this room's walls (None = mix all passes, the v1 behaviour)."""
    if not p.single_visit or index.frame is None or "timestamps" not in scene:
        return None
    visits = room_visits(poly, scene["traj"][:, [0, 2]], np.asarray(scene["timestamps"], float), p)
    return choose_visit(geom, index, floor_y, visits, p)


def _canonical_outlines(rooms: list[RoomBuild], index: PointIndex, floor_y: float, p: BetaParams) -> None:
    """B-2 per room, undone for any room whose new outline cuts into a neighbour (removing a bump can push a
    wall across a shared boundary; plans must never overlap, D-B4)."""
    before = {r.label: deepcopy(r.geom) for r in rooms}
    for r in rooms:
        canonicalize(r.geom, index, floor_y, p)
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            if a.polygon.intersection(b.polygon).area > 1e-4:
                for r in (a, b):
                    if r.polygon.intersection((b if r is a else a).polygon).area > 1e-4:
                        r.geom = deepcopy(before[r.label])


def _opening_outlines(rooms: list[RoomBuild], scene: dict, floor_y: float, maps: dict, index: PointIndex,
                      p: BetaParams) -> None:
    """v6 (I-011): each room's boundary on the wall's inner face at openings (walls.outline_at_openings). The
    opening evidence is positional: header open spans and far faces (D-046), and the vertical surface at sill height,
    in the wall band and in the header band. Each change only removes wall-slab area from its own room, so no
    overlap check is needed (unlike B-2)."""
    if not p.opening_outline:
        return
    headers = maps.get("headers") or []
    spans = [(h["axis"], float(h["offset"]), float(h["s0"]), float(h["s1"])) for h in headers]
    band = lambda lo, hi: vertical_points(scene["points"], scene["normals"], floor_y, p, (lo, hi))
    ev = OpeningEvidence(spans, headers, band(*p.opening_sill_band), band(p.wall_band_lo_m, p.wall_band_hi_m),
                         band(p.header_band_lo_m, p.header_band_hi_m))
    for r in rooms:
        r.outline_notes = outline_at_openings(r.geom, ev, index, r.floor_level, p)


def _undo_overlapping_refinements(rooms: list[RoomBuild], raster_lines: dict, p: BetaParams,
                                  max_iter: int = 10) -> None:
    """If two refined rooms overlap, the lines of each that now cut into the other measured a surface that is not
    their own wall (typically something seen through a doorway on a virtual boundary). Put those lines back at
    their free-space position and mark them inferred. Polygons before refinement do not overlap, so this ends."""
    for _ in range(max_iter):
        changed = False
        for i, a in enumerate(rooms):
            for b in rooms[i + 1:]:
                inter = a.polygon.intersection(b.polygon)
                if inter.area < 1e-5:
                    continue
                step = False
                for r in (a, b):
                    step |= _revert_lines_in(r, inter, raster_lines[r.label], p.inferred_sigma_m)
                if not step:                                     # already at free-space positions: share the line
                    step = _snap_shared_boundary(a, b, inter)
                changed |= step
        if not changed:
            return


def _strip_lines(r: RoomBuild, region) -> list[int]:
    """Indices of r's lines that run along the thin overlap strip and touch it."""
    u0, v0, u1, v1 = region.bounds
    strip_axis = 0 if (u1 - u0) <= (v1 - v0) else 1        # thin in u -> the culprits are constant-u lines
    V = r.geom.vertices()
    out = []
    for k, e in enumerate(r.geom.edge_lines):
        edge = LineString([V[k], V[(k + 1) % len(V)]])
        if r.geom.lines[e].axis == strip_axis and edge.intersection(region.buffer(1e-3)).length > 1e-3:
            out.append(e)
    return sorted(set(out))


def _snap_shared_boundary(a: RoomBuild, b: RoomBuild, region) -> bool:
    """Move the less certain of the two overlapping boundary lines onto the other (a shared, inferred boundary)."""
    la, lb = _strip_lines(a, region), _strip_lines(b, region)
    if not la or not lb:
        return False
    ea = min(la, key=lambda e: a.geom.lines[e].sigma)
    eb = min(lb, key=lambda e: b.geom.lines[e].sigma)
    (keep, kr), (move, mr) = sorted([(a.geom.lines[ea], a), (b.geom.lines[eb], b)], key=lambda x: x[0].sigma)
    idx = mr.geom.lines.index(move)
    if abs(move.offset - keep.offset) < 1e-9:
        return False
    mr.geom.lines[idx] = replace(move, offset=keep.offset, slope=keep.slope, mid=keep.mid, inferred=True,
                                 sigma=max(move.sigma, keep.sigma))
    return True


def _revert_lines_in(r: RoomBuild, region, raster: list[Line], sigma: float) -> bool:
    """Revert the lines of r that run ALONG the overlap strip (edges crossing it are not the cause)."""
    changed = False
    for e in _strip_lines(r, region):
        line = r.geom.lines[e]
        if line.offset != raster[e].offset:
            r.geom.lines[e] = replace(raster[e], inferred=True, sigma=sigma, slope=line.slope, mid=line.mid)
            changed = True
    return changed


def _tilts(rooms: list[RoomBuild], walls: list[Wall]):
    """(wall, tilt) for measured walls: how far each wall deviates from the Manhattan axis (pass-1 fit)."""
    by_id = {w.id: w for w in walls}
    for r in rooms:
        for k, e in enumerate(r.geom.edge_lines):
            line = r.geom.lines[e]
            if not line.inferred and f"{r.rid}-W{k + 1}" in by_id:      # photo v3 may replace a room's walls
                yield by_id[f"{r.rid}-W{k + 1}"], line.tilt


def _enclosure(geom: RoomGeometry) -> float:
    """Fraction of the perimeter that lies on walls measured on raw points."""
    lengths = edge_lengths(geom.vertices())
    measured = sum(L for L, e in zip(lengths, geom.edge_lines) if not geom.lines[e].inferred)
    return float(measured / max(lengths.sum(), 1e-9))


def _make_walls(rooms: list[RoomBuild]) -> list[Wall]:
    walls = []
    for r in rooms:
        V = r.geom.vertices()
        normals = edge_normals(r.geom)
        for k, length in enumerate(r.meas["walls"]):
            line = r.geom.lines[r.geom.edge_lines[k]]
            wid = f"{r.rid}-W{k + 1}"
            p0, p1 = V[k], V[(k + 1) % len(V)]
            walls.append(Wall(wid, r.rid, (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])), length,
                              (float(normals[k][0]), float(normals[k][1])), None, [], line.n_points, line.rmse))
            axis = line.axis
            r.faces.append(O.WallFace(wid, r.rid, axis, line.offset, float(np.sign(normals[k][axis])),
                                      tuple(sorted((float(p0[1 - axis]), float(p1[1 - axis])))), r.floor_level,
                                      line.inferred, line.slope, line.mid))
    return walls


def _thickness(rooms: list[RoomBuild], walls: list[Wall], p: BetaParams) -> dict[str, str]:
    """Two measured faces of different rooms, parallel, facing away from each other, 5-40 cm apart, overlapping
    along the wall -> one wall; thickness = distance between the faces. Returns {wall id: partner wall id}."""
    faces = [(f, r) for r in rooms for f in r.faces if not f.inferred]
    by_id = {w.id: w for w in walls}
    sig = {f.wall_id: r.geom.lines[r.geom.edge_lines[int(f.wall_id.split("-W")[1]) - 1]].sigma for f, r in faces}
    partners = {}
    for fa, ra in faces:
        best = None
        for fb, rb in faces:
            if rb is ra or fb.axis != fa.axis or fb.inward != -fa.inward:
                continue
            overlap = min(fa.span[1], fb.span[1]) - max(fa.span[0], fb.span[0])
            s_mid = 0.5 * (max(fa.span[0], fb.span[0]) + min(fa.span[1], fb.span[1]))
            gap = (fa.at(s_mid) - fb.at(s_mid)) * fa.inward    # > 0 when b lies behind a's face
            if p.thickness_min_m <= gap <= p.thickness_max_m and overlap >= 0.3:
                if best is None or gap < best[0]:
                    best = (gap, fb)
        if best is not None:
            gap, fb = best
            partners[fa.wall_id] = fb.wall_id
            by_id[fa.wall_id].thickness = Measurement.from_sigma(
                float(gap), float(np.hypot(sig[fa.wall_id], sig[fb.wall_id])), "m",
                f"distance between this face and the parallel face {fb.wall_id} of the neighbouring room")
    return partners


# ---------------------------------------------------------------- openings

def _necks_on_headers(necks: dict, headers: list[dict], grid: Grid) -> dict:
    """A neck that lies on a header cut (D-046) is a door-sized gap left in a WIDE opening: give it the header's open
    span as its width, so openings.measure_gap searches far enough for the real jambs."""
    if not headers:
        return necks
    from dataclasses import replace as _replace
    out = {}
    for key, n in necks.items():
        pt = grid.centers(np.array([n.rc[0]]), np.array([n.rc[1]]))[0]
        a = n.axis
        for h in headers:
            if h["axis"] == a and abs(pt[a] - h["offset"]) < 0.15 and h["s0"] - 0.1 <= pt[1 - a] <= h["s1"] + 0.1:
                n = _replace(n, width_m=max(n.width_m, h["open_m"]))
                break
        out[key] = n
    return out


def _find_openings(rooms, necks: dict[tuple[int, int], Neck], labels, grid: Grid, walls, scene, index, p):
    counter = iter(range(1, 10_000))

    def next_id() -> str:
        return f"O{next(counter)}"

    by_label = {r.label: r for r in rooms}
    openings: list[Opening] = []
    notes = []
    for (a, b), neck in necks.items():
        if a in by_label and b in by_label:
            op = _door_between(by_label[a], by_label[b], neck, grid, index, p, next_id)
            if op is not None:
                openings.append(op)
            else:
                notes.append(dict(kind="weak_link", rooms=[by_label[a].rid, by_label[b].rid],
                                  width_m=round(neck.width_m, 3),
                                  reason="free space touches through a gap narrower than a door and no gap "
                                         "could be measured on raw points: unobserved wall, not an opening"))
    for r in rooms:
        for bump in r.bumps:
            op = _door_from_bump(r, bump, index, p, next_id)
            if op is not None and not _duplicate(op, openings):
                openings.append(op)
    ctx = O.SeeThroughContext(O.RayIndex(scene, p), index, cKDTree(scene["points"]), scene["normals"],
                              {r.rid: r.polygon for r in rooms})
    for r in rooms:
        ceil_h = r.ceiling.value if r.ceiling is not None else None
        for f in r.faces:
            if f.inferred or f.span[1] - f.span[0] < p.window_min_w_m:
                continue
            wall = next(w for w in walls if w.id == f.wall_id)
            depth = wall.thickness.value if wall.thickness is not None else None
            found, nts = O.wall_openings(f, ctx, depth, ceil_h, p, next_id)
            notes += nts
            openings += [op for op in found if not _duplicate(op, openings)]
    _attach(openings, walls, rooms)
    return openings, notes


def _duplicate(op: Opening, existing: list[Opening], dist: float = 0.4) -> bool:
    return any(np.hypot(op.center[0] - e.center[0], op.center[1] - e.center[1]) < dist for e in existing)


def _door_between(ra: RoomBuild, rb: RoomBuild, neck: Neck, grid: Grid, index, p, next_id) -> Opening | None:
    """Door or passage at the neck that separates two rooms.

    The width is measured ALONG the cut (the line across the passage through the saddle), between the nearest
    raw points on either side, in a thin band around the cut. That works for a door in a wall (the band lies
    inside the wall slab, the gap ends are the jamb faces) and for a constriction across a corridor (the gap
    ends are the corridor's side walls) alike."""
    pt = grid.centers(np.array([neck.rc[0]]), np.array([neck.rc[1]]))[0]
    a = neck.axis                                            # the cut lies on a line of constant pt[a]
    t_rng = (pt[a] - p.cut_band_m, pt[a] + p.cut_band_m)
    floor_level = 0.5 * (ra.floor_level + rb.floor_level)
    gap = O.measure_gap(index, a, t_rng, float(pt[1 - a]), neck.width_m / 2, (p.jamb_h_lo_m, p.jamb_h_hi_m),
                        floor_level, p)
    if gap is not None and not (0.5 * neck.width_m <= gap.width <= 2.0 * neck.width_m + 0.2):
        gap = None
    if gap is None and neck.width_m < p.door_min_m:
        return None              # a sub-door-width leak through unobserved wall, not an opening (see weak_links)
    kind = "door" if neck.jambs and gap is not None else "passage"
    width = O.width_measurement(gap, neck.width_m, f"{kind} between {ra.rid} and {rb.rid}")
    ceil = [c.value for c in (ra.ceiling, rb.ceiling) if c is not None and c.value is not None]
    height = (O.head_height(index, a, t_rng, gap, floor_level, min(ceil) if ceil else None, p)
              if gap is not None else None)
    s_c = (gap.left + gap.right) / 2 if gap else float(pt[1 - a])
    center = (float(pt[0]), s_c) if a == 0 else (s_c, float(pt[1]))
    conf = 0.9 if (neck.jambs and gap is not None) else 0.6 if gap is not None else 0.4
    ev = (f"free-space constriction {neck.width_m:.2f} m wide separating {ra.rid} and {rb.rid}; "
          f"wall evidence on both sides: {neck.jambs}; gap ends measured on raw points: {gap is not None}")
    return Opening(next_id(), kind, [], [ra.rid, rb.rid], (float(center[0]), float(center[1])), width,
                   height, None, conf, f"{ev}; measured along {'v' if a == 0 else 'u'}")


def _door_from_bump(r: RoomBuild, bump: Bump, index, p, next_id) -> Opening | None:
    """Doorway into space we did not map (cut off the room polygon as a wall-deep bump)."""
    s_mid, w = (bump.span[0] + bump.span[1]) / 2, bump.span[1] - bump.span[0]
    face = next((f for f in r.faces if f.axis == bump.axis and abs(f.at(s_mid) - bump.offset) < 0.03
                 and f.span[0] - 0.05 <= bump.span[0] and bump.span[1] <= f.span[1] + 0.05), None)
    if face is None or w < p.door_min_m:
        return None
    t_rng = O.slab(face, bump.depth, s_mid)
    gap = O.measure_gap(index, face.axis, t_rng, s_mid, w / 2, (p.jamb_h_lo_m, p.jamb_h_hi_m), face.floor_level, p)
    if gap is not None and not (0.5 * w <= gap.width <= 1.5 * w + 0.1):
        gap = None
    width = O.width_measurement(gap, w, f"doorway in {face.wall_id} to unmapped space")
    ceil_h = r.ceiling.value if r.ceiling is not None else None
    height = O.head_height(index, face.axis, t_rng, gap, face.floor_level, ceil_h, p) if gap else None
    s_c = (gap.left + gap.right) / 2 if gap else s_mid
    center = (face.at(s_c), s_c) if face.axis == 0 else (s_c, face.at(s_c))
    ev = (f"the room's free space continues {bump.depth*100:.0f} cm into the wall over {w:.2f} m and open space "
          "was seen beyond it (doorway to a space that was not mapped)")
    return Opening(next_id(), "door", [face.wall_id], [r.rid], (float(center[0]), float(center[1])), width,
                   height, None, 0.7 if gap is not None else 0.5, ev)


def _attach(openings: list[Opening], walls: list[Wall], rooms: list[RoomBuild]) -> None:
    """Register each opening id on every wall it lies on (centre within 15 cm of the wall segment)."""
    for op in openings:
        c = Point(op.center)
        for w in walls:
            if w.room_id in op.room_ids and _segment_distance(c, w) < 0.15 + 0.2 * (op.kind != "window"):
                if op.id not in w.opening_ids:
                    w.opening_ids.append(op.id)
                if w.id not in op.wall_ids:
                    op.wall_ids.append(w.id)


def _segment_distance(c: Point, w: Wall) -> float:
    return LineString([w.p0, w.p1]).distance(c)


def _adjacency(openings: list[Opening], partners: dict[str, str], room_of: dict[str, str]) -> list[Adjacency]:
    """Rooms connected through a door/passage (via = opening id); otherwise rooms sharing a wall."""
    out, seen = [], set()
    for op in openings:
        if len(op.room_ids) == 2 and frozenset(op.room_ids) not in seen:
            out.append(Adjacency(op.room_ids[0], op.room_ids[1], op.id))
            seen.add(frozenset(op.room_ids))
    for wa, wb in partners.items():
        key = frozenset((room_of[wa], room_of[wb]))
        if key not in seen:
            seen.add(key)
            out.append(Adjacency(room_of[wa], room_of[wb], "shared_wall"))
    return out


# ---------------------------------------------------------------- rooms

def _make_rooms(rooms: list[RoomBuild], folder_of: dict[int, str] | None = None) -> list[Room]:
    out = []
    for r in rooms:
        m = r.meas
        V = r.geom.vertices()
        wall_ids = [f"{r.rid}-W{k + 1}" for k in range(len(V))]
        label = folder_of[r.label] if folder_of and r.label in folder_of else _label(r, m)   # v4 T-3: folder name
        out.append(Room(r.rid, label, [(float(x), float(y)) for x, y in V], wall_ids, m["area"],
                        m["perimeter"], r.ceiling, m["bbox"], float(r.floor_level)))
    return out


def _label(r: RoomBuild, m: dict) -> str:
    length, width = m["bbox"][0].value, m["bbox"][1].value
    kind = "corridor" if (width <= 1.6 and length / max(width, 1e-6) >= 2.0) else "room"
    return kind if r.visited else f"{kind} (not entered; seen through an opening)"


# ---------------------------------------------------------------- v4 T-3: photo folders (adjacency, widening)

def ndi_distance(free: np.ndarray, p: BetaParams) -> np.ndarray:
    from floorplan.plan.beta.segment import distance_map
    return distance_map(free, p)


def _folder_links(info: dict) -> dict[frozenset, dict]:
    """Room-folder pairs the photo front-end linked: the final pose-graph links (room_links_v2) when present,
    else every link it recorded (v1 info.links)."""
    src = info.get("room_links_v2")
    if src is None:
        src = info.get("links") or {}
    out = {}
    for key, ev in src.items():
        a, _, b = key.partition("|")
        if a and b and a != b:
            out[frozenset((a, b))] = ev
    return out


def _folder_adjacency(computed: list[Adjacency], info: dict, rid_of: dict[str, str]):
    """Photo tier: adjacency ONLY where the photo front-end linked two folders. Rooms that merely touch in the plan
    may do so because one was placed beside the block by the fallback (a guess), so touching is not evidence.
    A link is 'measured' when both rooms are placed by links, 'inferred' when either was a fallback placement.
    `via` is the opening found between the two rooms when there is one, else 'shared_wall' (schema)."""
    rooms_info = info.get("rooms") or {}
    by_pair = {frozenset((a.room_a, a.room_b)): a for a in computed}
    folder_of = {v: k for k, v in rid_of.items()}
    out, evidence = [], []
    for pair, ev in sorted(_folder_links(info).items(), key=lambda kv: sorted(kv[0])):
        fa, fb = sorted(pair)
        if fa not in rid_of or fb not in rid_of:
            evidence.append(dict(folders=[fa, fb], status="not_in_plan", link=ev))
            continue
        ra, rb = rid_of[fa], rid_of[fb]
        prev = by_pair.get(frozenset((ra, rb)))
        via = prev.via if prev is not None and prev.via != "shared_wall" else "shared_wall"
        fallback = any(str((rooms_info.get(f) or {}).get("placement", "linked")).startswith("fallback")
                       for f in (fa, fb))
        out.append(Adjacency(ra, rb, via))
        evidence.append(dict(rooms=[ra, rb], folders=[fa, fb], via=via, link=ev,
                             status="inferred" if fallback else "measured",
                             touching_in_plan=prev is not None))
    linked = {frozenset((a.room_a, a.room_b)) for a in out}
    for a in computed:
        if frozenset((a.room_a, a.room_b)) not in linked:
            evidence.append(dict(rooms=[a.room_a, a.room_b], folders=[folder_of.get(a.room_a), folder_of.get(a.room_b)],
                                 via=a.via, status="dropped: rooms touch in the plan but no photo link joins them"))
    return out, evidence


def _layout_rooms(scene, info, cams, plan_rooms, walls, openings, fp, p):
    """Photo tier v3 (plan_beta.md Part v4, v4.10): replace each folder's free-space room by the Manhattan rectangle
    of its own spin layout (tiers.layout_plan). Folders without a layout keep their (already widened) v4 room, walls
    and openings. Adjacency again only from the photo links."""
    try:
        rooms_l, walls_l, open_l, fp_l, rep, used = T.layout_plan(info, plan_rooms, cams, scene, p.seed)
    except Exception as e:                      # never lose the plan over the layout step: v4 plan stands
        return plan_rooms, walls, openings, fp, None, dict(used=False, error=f"{type(e).__name__}: {e}")
    if not used:                                # no usable layout: the v4 plan stands as it is
        return plan_rooms, walls, openings, fp, None, dict(per_folder=rep, used=False)
    keep = [r for r in plan_rooms if r.label not in used]
    keep_ids = {r.id for r in keep}
    rooms = rooms_l + keep
    walls = walls_l + [w for w in walls if w.room_id in keep_ids]
    openings = open_l + [o for o in openings if set(o.room_ids) <= keep_ids]
    if keep:                                    # footprint: layout rectangles + kept rooms (v4 widened values)
        fp_l.method += f"; {len(keep)} folder(s) without a layout kept their free-space room"
    rid_of = {r.label: r.id for r in rooms}
    adjacency, evidence = _folder_adjacency([], info, rid_of)
    for a in adjacency:                         # via the doorway-pair opening when there is one
        for o in open_l:
            if set(o.room_ids) == {a.room_a, a.room_b}:
                a.via = o.id
    overlaps = []
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            ar = Polygon(a.polygon).buffer(0).intersection(Polygon(b.polygon).buffer(0)).area
            if ar > 0.01:
                overlaps.append(dict(rooms=[a.id, b.id], area_m2=round(float(ar), 3)))
    return rooms, walls, openings, fp_l, adjacency, dict(per_folder=rep, kept_free_space_rooms=sorted(keep_ids),
                                                         adjacency_evidence=evidence, overlaps=overlaps)


def _scale_interval(m: Measurement | None, k: float) -> None:
    if m is None or m.value is None or m.lo is None or m.hi is None or k == 1.0:
        return
    m.lo, m.hi = max(m.value - k * (m.value - m.lo), 0.0), m.value + k * (m.hi - m.value)   # sizes are >= 0
    m.method = f"{m.method}; x{k:g} photo placement widening".strip("; ")


def _folder_widen(rooms: list[Room], walls: list[Wall], fp: Measurement, info: dict, rid_of: dict[str, str]) -> dict:
    """Multiply each room's interval half-widths by its folder's widen factor (photo front-end: linked 2,
    door_match 3.5, fallback 4). The footprint is widened by the area-weighted mean factor."""
    rooms_info = info.get("rooms") or {}
    k_of = {rid: float((rooms_info.get(f) or {}).get("widen", 1.0) or 1.0) for f, rid in rid_of.items()}
    for r in rooms:
        k = k_of.get(r.id, 1.0)
        for m in (r.floor_area, r.perimeter, r.ceiling_height, *(r.bbox_dims or ())):
            _scale_interval(m, k)
    for w in walls:
        k = k_of.get(w.room_id, 1.0)
        _scale_interval(w.length, k)
        _scale_interval(w.thickness, k)
    area = {r.id: (r.floor_area.value or 0.0) for r in rooms}
    tot = sum(area.values())
    k_fp = sum(k_of.get(i, 1.0) * a for i, a in area.items()) / tot if tot > 0 else 1.0
    _scale_interval(fp, k_fp)
    return dict(per_room=k_of, footprint=round(k_fp, 3))
