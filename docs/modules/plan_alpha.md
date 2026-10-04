# Module `plan_alpha`: boundary-first floor-plan extraction (LiDAR tier)

Files:

| File | What it holds |
|---|---|
| `floorplan/plan/alpha/__init__.py` | public entry point `extract_plan(scene, info, capture_id, params=None) -> Plan` |
| `floorplan/plan/alpha/params.py` | every threshold, with the reason for its value (`AlphaParams`) |
| `floorplan/plan/alpha/evidence.py` | step 1: plan raster, wall points, interior (free-space) evidence |
| `floorplan/plan/alpha/lines.py` | step 2: Manhattan wall lines by 1-D peak finding |
| `floorplan/plan/alpha/complex.py` | step 3: the cell complex (cells, edge wall coverage) |
| `floorplan/plan/alpha/labeling.py` | steps 4-5: inside/outside graph cut, rooms by distance-transform watershed |
| `floorplan/plan/alpha/measure.py` | robust fits on raw LiDAR points (walls, floors, ceilings) |
| `floorplan/plan/alpha/walls.py` | step 6: corners, lengths, area, perimeter, thickness with Monte Carlo intervals |
| `floorplan/plan/alpha/heights.py` | per-room floor level and ceiling height |
| `floorplan/plan/alpha/doors.py` | doors and passages, jamb-to-jamb widths |
| `floorplan/plan/alpha/windows.py` | windows, glass with no return, mirror test |
| `floorplan/plan/alpha/extract.py` | orchestration, adjacency, footprint |
| `floorplan/plan/alpha/render.py` | the debug figure |
| `scripts/run_plan_alpha.py` | command line: one capture in, `plan.json` + `metrics.json` + `plan_debug.png` out; `--pair` repeatability self-check |

Run:

```bash
cd floorplan-capture
PY=../.venv/bin/python
# one capture (the scene must exist: scripts/prepare_scene.py)
PYTHONPATH=. $PY scripts/run_plan_alpha.py outputs/single_scan_with_ceiling__c7d28f72c6
# on the drift-corrected scene, with the repeatability self-check against the other whole-apartment scan
PYTHONPATH=. $PY scripts/run_plan_alpha.py outputs/drift/single_scan_with_ceiling__c7d28f72c6/scene_on \
    --name drift_on__single_scan_with_ceiling__c7d28f72c6 \
    --pair outputs/drift/single_scan_floor_only__1a8384c3f6/scene_on --pair-name drift_on__single_scan_floor_only__1a8384c3f6
# optional: --drift-sigma 0.02 (use the drift module's measured residual drift instead of the 1 cm default)
```

Evidence lives in `outputs/plan_alpha/`:

| Path | Content |
|---|---|
| `<capture>/plan.json` | the `Plan` (schema in `floorplan/model.py`); `meta.wall_fit` has per-face diagnostics, `meta.warnings` has mirror warnings |
| `<capture>/metrics.json` | counts, room table, runtime per stage |
| `<capture>/plan_debug.png` | the debug figure |
| `drift_on__<capture>/` | the same, on the drift module's corrected scene (`outputs/drift/<capture>/scene_on`) |
| `repeatability/*.md`, `*.json` | the pair self-check (register + match + repeatability from `floorplan/eval/`) |
| `repeatability/experiment_min_jog.json` | the jog-tolerance experiment (Decision A-12) |
| `summary.json` | the numbers quoted in section 5, regenerated from the plan files |

---

## 1. Purpose and where it sits in the pipeline

```
capture ─► prepare_scene (TSDF + raw points + gravity/Manhattan alignment) ─► [drift correction] ─► scene.npz
                                                                                                  │
                                                       plan_alpha.extract_plan(scene, info) ◄─────┘
                                                                  │
                         Plan: rooms, walls, openings, adjacency, footprint (every number with a 95% interval)
                                                                  │
                                      export / rendering, evaluation (floorplan/eval), tiers share this back-end
```

`plan_alpha` turns an aligned 3-D scene into the **output contract**: one room per real room or corridor, each with
a polygon, walls with lengths, floor area, perimeter, ceiling height, bounding-box dimensions, plus doors, windows,
adjacency and the total footprint. Every number carries a 95% interval and a status (`measured`, `inferred`,
`not_observed`).

It is one of two plan extractors built in parallel (`plan_beta` is the other); both produce the same `Plan` type
so the evaluation harness can compare them. Alpha is the **boundary-first** approach (Mura et al. 2016, Ochmann et
al. 2019, Floor-SP 2019): find the walls first, then decide which regions between walls are rooms. The room shape is
therefore assembled from measured wall lines, never traced from a pixel mask.

Inputs used from the scene (see `floorplan/pipeline/scene.py`): the TSDF surface with normals (`points`,
`normals`) for structure, the raw high-confidence LiDAR points with range and viewing ray (`raw_points`,
`raw_range`, `raw_ray`) for every reported number (D-007), the camera path (`traj`), and the floor/ceiling levels
from `info`.

---

## 2. How it works, step by step

Plan coordinates are `(u, v) = (x, z)` of the aligned world, in metres. After Manhattan alignment, walls run along
u or v.

### Step 1. Evidence rasters (`evidence.py`)

- **What.** A 2 cm plan grid (same as the TSDF voxel). Two kinds of evidence go on it.
  - *Wall evidence*: TSDF points whose normal is within ~17° of horizontal (|n_y| < 0.3), between 0.3 m and 2.0 m
    above the floor. They are split into two families: normals along ±x (walls `u = const`) and normals along ±z
    (walls `v = const`). The **sign** of the normal is kept. TSDF normals point to the side the camera saw, so the
    sign says which side of a face the room is on.
  - *Interior evidence*: places we know are inside the dwelling. These are visible floor (up-facing points within
    ±15 cm of the floor), low horizontal tops (beds, tables), the camera path, and the 2-D footprint of 400k LiDAR
    rays (a ray from the camera to its hit point proves the space in between is empty).
- **Why.** Walls tell us where boundaries can be. Interior evidence tells us which side of a boundary is a room.
  Unobserved space carries no evidence.
- **Without it.** Without the band, floor clutter and ceiling soffits become "walls". Without the sign, the two
  faces of a 10 cm partition blur into one line and wall thickness cannot be measured. Without rays, rooms with a
  lot of furniture (floor hidden) would be called outside.

### Step 2. Wall lines (`lines.py`)

- **What.** For each family and each normal sign, project the wall points onto the perpendicular axis, build a 1 cm
  histogram, and find its peaks. Each peak is a candidate line, refined to the median of its points (sub-cm).
  Along each line we keep an **occupancy profile** in 2 cm bins (where wall surface was seen), plus a "tall"
  profile (surface above 1.2 m, i.e. above most furniture). A line is kept if it is occupied over ≥ 0.4 m.
  Opposite-sign lines closer than 3 cm merge (a thin panel seen from both sides).
- **Why.** After Manhattan alignment, finding walls is a 1-D problem.
- **Without it.** Lines from a Hough transform are dominated by long walls, and RANSAC is random (non-deterministic
  unless seeded) and slow on 1M points.

### Step 3. Cell complex (`complex.py`)

- **What.** Every line is extended across the whole plan. Manhattan lines cut the plane into a rectilinear grid of
  uneven cells. For every cell we store the fraction of its pixels with interior evidence. For every edge between
  two cells we store its **wall coverage** (share of the edge where the line's occupancy profile is on).
- **Why.** Every room polygon we will output is a union of cells. Every polygon vertex therefore lies on two
  detected wall lines, so corners are intersections of walls, not pixels.
- **Without it.** Tracing a raster mask gives staircase edges and rounded corners, and a dimension taken from such a
  polygon is only as good as the 2 cm pixel.

### Step 4. Inside / outside (`labeling.py: label_inside`)

- **What.** Binary labelling with an energy that has two parts. It is solved exactly by one s-t minimum cut
  (`networkx.minimum_cut`).
  - *Data cost* per cell:
    - labelling a cell OUTSIDE costs `area x interior_fraction` (+100 if the camera passed through it);
    - labelling it INSIDE costs `0.6 x area x (1 - interior_fraction)`.
  - *Smoothness cost* per edge: changing label across an edge costs `0.25 m x edge_length x (1 - wall_coverage)`.
    A boundary is therefore cheap where there is a wall and expensive in open space.
- **Why.** Each cell's own evidence is noisy (rays leak through doors, the floor is hidden under beds). The
  smoothness term makes the boundary follow walls and removes speckle. For two labels this energy has an exact,
  deterministic global optimum (Boykov and Jolly 2001), which is the same formulation as Ochmann 2019.
- **Without it.** Thresholding each cell alone gives holes under furniture and islands outside the dwelling.

### Step 5. Rooms (`labeling.py: group_rooms`)

- **What.**
  1. Inside pixels minus "barrier" pixels (TALL vertical surface, ≥ 1.2 m) form the free mask.
  2. The distance transform of the free mask peaks in the middle of rooms and dips in doorways.
  3. A **watershed** from the significant peaks (h-maxima, 10 cm) splits the free space into basins.
  4. Basins merge when the neck between them is ≥ 1.3 m (wider than any door: open plan), or ≥ 85% of the width
     of the **wider** basin (no constriction: the same space split by noise), or when one of them is smaller than
     1 m². This merge rule is Decision A-6.
  5. Basins are snapped to cells: a cell joins the basin covering ≥ 50% of its pixels. Barrier-only cells (a
     wardrobe's side panel) adopt their neighbours' room when all neighbours agree.
  6. Polygons are the union of each room's cells, then cleaned:
     - morphological opening/closing with square corners (removes < 30 cm slivers and slits);
     - removal of < 12 cm jogs;
     - smaller rooms clipped by larger ones, so rooms never overlap.
- **Why.** Rooms are separated by walls with door-sized gaps. A watershed on the distance transform puts boundaries
  exactly at those constrictions. This is the classic morphological room segmentation (Bormann et al. 2016, survey).
- **Without it.** See issues I-2 and I-3: the first two versions merged most of the apartment into one room.

### Step 6. Walls, corners, lengths, areas (`measure.py`, `walls.py`)

- **What.** Each polygon edge is re-measured on **raw** LiDAR points:
  1. Take points within ±6 cm of the edge's line, 15 cm away from both corners (or the middle half of a short
     wall), outside doorways, at wall heights, and **seen from the room side** (viewing ray pointing into the wall).
  2. Start at the mode of the signed distances (the biggest flat surface), then refine with a Tukey biweight
     M-estimator (IRLS). By default the fit is axis-aligned (one offset). A tilted line is used only if it turns the
     face into a clean plane (Decision A-9).
  3. Faces of two rooms that face each other and cross by a few mm are reconciled, so rooms never overlap
     (Decision A-10).
  4. Corners are intersections of adjacent fitted lines. A wall's length is the distance between its two corners,
     so it depends on its two neighbours.
  5. Intervals come from one Monte Carlo (2000 draws) over all wall offsets of the room. It gives lengths, area,
     perimeter and bounding box together.
- **The offset sigma** (1-σ, per wall face):
  ```
  sigma_face^2 = SE^2 + lidar_sigma^2 + inconsistency^2 + drift_sigma^2 / 2
     SE            = robust sigma / sqrt(n_eff), with n_eff = number of distinct 10 cm x 10 cm patches
     lidar_sigma   = 5 mm  (ranging bias that averaging cannot remove)
     inconsistency = sqrt(max(rmse^2 - point_noise^2, 0)), point_noise = 6 mm
                     (scatter the sensor cannot explain: two passes that disagree, a curtain)
     drift_sigma   = 1 cm by default; --drift-sigma to use the drift module's value
  ```
  - For an axis-aligned room, a length involves two faces, so
    `sigma_L^2 = SE_a^2 + SE_b^2 + 2 lidar^2 + I_a^2 + I_b^2 + drift^2`.
  - The drift budget is counted once per length.
  - There is no discretisation term: offsets are continuous fits. Only an `inferred` face (no raw support) uses the
    2 cm TSDF line, with σ = 2 cm.
- **Why each term.**
  - SE alone would claim 0.3 mm on a well-seen wall. That is impossible, and the case study penalises "confident
    garbage".
  - `n_eff` by patches: neighbouring points share their frame's error, so they are not independent.
  - The inconsistency term widens the interval exactly where the scan disagrees with itself (issues I-9 and I-11).
- **Thickness.** Two faces of different rooms, parallel with opposite normals, 4–45 cm apart and overlapping
  ≥ 30 cm. The thickness is the distance between the faces, and its σ is the faces' σ in quadrature.

### Step 7. Floor and ceiling (`heights.py`)

- **What.** Inside the room polygon shrunk by 20 cm:
  - *Floor level*: robust level of raw points near the floor.
  - *Ceiling*: the down-facing TSDF layers above 1.9 m, merged in 5 cm bins.
    - We take the highest layer holding ≥ 30% as many points as the biggest one, then refine it on raw points.
    - If the ceiling is seen over < 15% of the room, or with too few points, the result is
      `Measurement.missing(...)` (`not_observed`). No number is invented.
    - If a second big layer lies within 20 cm, the interval is widened to cover both and the status becomes
      `inferred`: it is a step or vertical drift, and we cannot tell which (issue I-10).
- **Formula.** `height = ceiling - floor`, with
  `sigma^2 = SE_c^2 + SE_f^2 + I_c^2 + I_f^2 + 2 lidar_sigma^2`.
- **Why per room.** Bathrooms and corridors here have lowered ceilings (2.28–2.47 m vs 3.08 m), visible as AC
  bulkheads in the RGB.

### Step 8. Doors and passages (`doors.py`)

- **Detection A, between two rooms.** For every pair of rooms, find *contacts*: an edge of room A and an edge of
  room B, parallel with opposite inward normals, at most one wall thickness apart, overlapping by ≥ a door width.
  Along the strip between the two faces, a run of bins that is observed free and holds no tall surface is an
  opening.
- **Detection B, from a room to unobserved space.** A gap inside a wall line with a room on one side only, and
  free space seen 55 cm beyond (otherwise it is a niche).
- **Rejection.** If ≥ 30% of the opening is *solid* at body height inside the wall core (≥ 50 raw points per
  5 cm bin), it is a half-wall or the wall below a window, not a door. The rays that made it look free passed over
  the sill.
- **Width.** Measured jamb face to jamb face, on raw points inside the wall core (between the two faces, 2 cm in
  from each). There the only visible surfaces are the jambs.
  - Each jamb is the layer nearest the opening that is a tall face: it must cover ≥ 40% and ≥ 80% of the best
    layer's share of the 10 cm height bins.
  - It is then mean-shifted onto the face.
  - `sigma^2 = SE_left^2 + SE_right^2 + 2 lidar_sigma^2`, which gives ±1.4 cm (95%) when both jambs are seen.
  - If a jamb is not visible, the width is the free run on the 2 cm raster, `inferred`, ±3.9 cm.
- **Height.** The lowest dense layer in the wall core above 1.8 m (the lintel), if seen.
- **Kind.** `door` if the width is ≤ 1.0 m, else `passage`.
- **Confidence.** `0.4 + 0.3·free + 0.3·(jambs measured)`, ×0.7 for a one-sided opening.

### Step 9. Windows, glass and mirrors (`windows.py`)

- **Where.** Exterior walls only (no room within 0.6 m behind). On each, a 5 cm raster on the wall plane
  (0.3–2.2 m).
- **Signals.**
  - *SEE-THROUGH*: raw points ≥ 15 cm behind the plane whose viewing ray crossed it. The LiDAR saw through the
    wall: open window or clear glass.
  - *NO-RETURN*: bins enclosed by seen wall that have no return, no see-through and nothing in front that could
    hide them. Dark glass often returns nothing, or only low-confidence depth that we drop (D-006).
    Confidence 0.4, because a black TV looks the same.
- **Mirror test.** A mirror shows a *reflected room behind the wall*, which looks exactly like see-through. Every
  see-through region is reflected back across the plane. If ≥ 50% of its points land within 3 cm of real points,
  it is a mirror: reported in `meta.warnings`, not as a window.
- **Door-like openings.** A see-through region that starts at the floor and is > 1.7 m tall is reported as a
  `door` (balcony / glass door).

### Step 10. Plan assembly (`extract.py`)

- Rooms are labelled `corridor` if they are narrow (< 1.6 m) and elongated (≥ 2.5:1), else `room`. The suffix
  "(seen through an opening, not entered)" is added if the camera path never entered the room.
- Adjacency: every two-room opening, plus `shared_wall` for rooms whose faces define a wall thickness.
- `footprint_area` = sum of room net areas (rooms are disjoint), with a Monte Carlo interval.
- `meta` records the parameters, timing, overlap check (must be 0), reconciled face pairs, per-face fit
  diagnostics and warnings.

---

## 3. Decisions

Each decision gives the options considered, the choice, why, and the evidence. Numbers come from runs of this
code: the final outputs, or the development run named in the issue log.

| # | Decision | Options considered | Choice | Why | Evidence |
|---|---|---|---|---|---|
| A-1 | Approach | (a) mask-first: label pixels, trace polygons; (b) learned polygons (RoomFormer etc., D-004); (c) **boundary-first cell complex** | (c) | Vertices are wall-line intersections, so dimensions come from fitted planes. It is explainable and deterministic. | Polygon vertices lie on fitted lines; overlap 0 on all captures (`metrics.json`). |
| A-2 | Line finding | Hough; RANSAC; **1-D histogram peaks per family and sign** | 1-D peaks | Exact after Manhattan alignment, deterministic, finds short walls next to long ones, keeps the two faces of a partition apart | 9–54 lines per family, < 0.2 s (development log). |
| A-3 | Evidence source for structure vs numbers | raw points for everything; **TSDF for structure, raw for numbers** (D-007) | split | TSDF is uniform and has normals; raw points are unsmoothed. | Wall fit SE 0.3–2 mm on raw points (`meta.wall_fit`). |
| A-4 | Inside/outside | threshold per cell; region growing from the camera; **graph cut** | graph cut | Exact global optimum of a clear energy; boundaries snap to walls | First run already gave a correct apartment outline on with_ceiling (issue I-2 figure). |
| A-5 | Room separation | (a) union-find across edges without "gap-closed" tall wall; (b) (a) + T-junction anchors; (c) **distance-transform watershed + neck merge, snapped to cells** | (c) | (a) leaks through doors next to corners; (b) over-bridges; (c) is the standard morphological method and needs no door model | Rooms on with_ceiling: (a) 3 rooms, 59.5 m² merged; (b) 6 rooms, 38.8 m² still merged; (c) 9–10 rooms (issues I-2, I-3). |
| A-6 | Basin merge rule | neck vs narrower basin; **neck vs wider basin**; neck ≥ 1.3 m always merges | wider + 1.3 m | A 1.1 m corridor meeting a 3.5 m room is a constriction for the room. With "narrower", the corridor merged into the bedroom. | Issue I-4. |
| A-7 | Polygon clean-up | none; **opening + closing 30 cm, jogs < 12 cm, clip by larger rooms** | as bold | Removes wall-core slivers, door-reveal bumps and drift-duplicated steps without moving surviving edges | Walls 148 → 88 on with_ceiling; overlaps 0. Jog tolerance experiment (A-12). |
| A-8 | Wall uncertainty | SE only; SE + fixed floor; **SE + LiDAR bias + inconsistency + drift** | full | SE alone claims sub-mm. The inconsistency term widens exactly where captures disagree (curtains, double passes). | Curtain wall: rmse 23.6 / 39.4 mm, σ 24 / 40 mm. The 7.5 cm cross-capture difference has z = −1.37 (inside the interval). Issue I-11. |
| A-9 | Wall tilt | always axis-aligned; always tilted (≤ 5°); **tilted only if it makes a clean plane** (rmse −30% and ≤ 12 mm) | gated | A tilt fitted to smeared walls skewed rooms into each other; real off-square walls are still handled | Unconditional tilt: a face hit the 5° clip, overlap 0.137 m². Gated: 8–14 tilted faces per capture, overlap → 0 after A-10. Clean-tilt example: rmse 18.5 → 2.8 mm. Issue I-8. |
| A-10 | Faces of two rooms that cross | leave the overlap; clip polygons afterwards; **reconcile faces before corners** | reconcile | Keeps every polygon consistent with its walls; measured faces win over inferred ones | Overlap 0.0085 / 0.0218 / 0.1 m² → 0 on all captures; 1–6 pairs reconciled per capture. Issue I-12. |
| A-11 | Door detector | gaps inside a single wall line; **contacts between two rooms' facing edges** (+ line gaps for one-sided doors) | contacts | Doors next to corners have wall on one side only, so a line has no internal gap | Doors/passages found: line gaps 1 (floor_only) / 3 (with_ceiling) → contacts 10 / 8 (issue I-5). |
| A-12 | Jog tolerance | 0.12 / 0.2 / 0.3 m | **0.12 m** | Larger values remove real geometry and do not improve repeatability | Drift-corrected pair: wall passes 8 / 3 / 4, median Δ 6.4 / 6.4 / 9.1 cm (`repeatability/experiment_min_jog.json`). |
| A-13 | Jamb choice | nearest dense layer; tall-face test with absolute threshold; **tall-face test relative to the best layer + mean shift** | relative | The nearest-layer rule picked a door-leaf edge; the absolute threshold flipped on a 2.5 cm floor shift | Same door in both captures: 0.805 vs 0.909 → **0.911 vs 0.917** m (issue I-7). |
| A-14 | Ceiling layer | highest layer; biggest layer; **highest layer with ≥ 30% of the biggest; widen if two layers < 20 cm apart** | as bold | Handles lamps and cabinet bottoms (lower) and drift doubles (close) honestly | with_ceiling r2: 2.95 / 3.06 m layers under ARKit poses → `inferred` [2.947, 3.079]. On drift-corrected poses: one layer, 3.066 [3.050, 3.081]. Issue I-10. |
| A-15 | Windows only on exterior walls | all walls; **exterior only** | exterior | Phantom openings are scored as misses; interior walls rarely have windows | – |
| A-16 | Mirror handling | ignore; drop see-through; **reflection test** | test | A mirror looks like a window; the test is cheap and explainable | 2–3 mirror warnings on the drift-corrected scans, in bathroom-like rooms (2.38 m ceiling), matching the vanity mirror in frame #4547 (`meta.warnings`). |
| A-17 | Speed | as written; **bincount instead of `np.add.at`, local KD-trees, float32 sort, 1-in-4 raw subsample for window rasters** | optimised | Budget 90 s per capture on a shared machine | with_ceiling 72 s → 15 s (profile: `add.at` 14 s, window rasters 15 s, argsort 12.5 s). Issue I-13. |
| A-18 | Where `drift_sigma` comes from | constant; **parameter, default 1 cm, CLI `--drift-sigma`** | parameter | The drift module measures it (held-out loops). The extractor must not depend on that module at run time. | Drift module: corrected 1.9–2.5 cm, ARKit 3.4–10.9 cm (`outputs/drift/*/drift_report.json`). |

---

## 4. Issues log

Each issue: symptom → root cause (with evidence) → possible fixes → fix chosen and why.

**I-1. Many visible walls seemed to have no line.**
- Symptom: the overlay plot showed walls in the density image with no coloured line.
- Root cause: the plot itself. Single-pixel markers were invisible at that dpi. Printing the lines showed them
  present (e.g. `v = 6.796`, 3.14 m support).
- Side finding: that wall's points spread over 6.70–6.90 m. floor_only is smeared and bent by drift (a 4° slope on
  the bottom wall). We recorded this for the drift module and for A-8.
- Fix: none needed.

**I-2. One room swallowed most of the apartment.**
- Symptom: with_ceiling gave 3 rooms, one of 59.5 m².
- Root cause: rooms were joined across any edge without "gap-closed tall wall". A door **next to a corner** has
  wall on one side only, so the 1-D closing along the line had nothing to close against. The separation-edge
  figure showed closed loops broken exactly at such doors.
- Possible fixes:
  - (a) add the perpendicular wall as an anchor;
  - (b) model doors explicitly;
  - (c) segment rooms by constrictions (watershed).
- Tried (a). Result: 6 rooms, but separation lines now ran 1.3 m into open rooms and the apartment still leaked.
- Chose (c), which needs no door model and is the standard method (Decision A-5).

**I-3. Watershed basins snapped to cells left holes and dropped room parts.**
- Symptom: a room polygon missing a large part (with_ceiling top room).
- Root cause:
  - Thin strip cells that are all barrier (a wardrobe side panel) had no basin and cut the room in two.
  - The polygon code then kept only the largest piece, while the dropped cells stayed in the room map.
  - Tiny basins (< 1 m²) became separate, dropped "rooms", leaving holes.
- Fixes, all three:
  - unassigned inside cells adopt the neighbours' basin when all neighbours agree;
  - keep only the largest connected group of cells;
  - merge tiny basins into their best-connected neighbour.

**I-4. A corridor merged into a bedroom.**
- Root cause: the merge rule compared the neck to the narrower basin. Corridor width ≈ neck width, so the ratio
  was ≈ 1 and they merged.
- Fix: compare with the wider basin (A-6).

**I-5. Door recall was very low.**
- Symptom: 1 door on floor_only, 3 on with_ceiling.
- Root cause: the same corner-door problem as I-2. Gap search inside a single line misses doors that end at a
  perpendicular wall.
- Fix: contacts between the facing edges of two rooms (A-11). Result: 10 and 8 doors/passages.

**I-6. Doors with "jambs not measurable", and false doors.**
- Symptom: most widths fell back to the 2 cm raster.
- Root cause 1: the free run is clipped to the overlap of the two polygon edges, so the real jambs lay outside the
  search window. In the corridor example, the jambs were at 1.289 / 2.122 m while the run was 1.43–2.05 m.
- Root cause 2: some "openings" had solid surface inside the run: a wall below a window (solid at 0.33–0.55 m),
  and a ledge at 1.04 m. Rays passing over the sill made them look free.
- Fixes:
  - search each jamb up to 30 cm beyond its end of the run;
  - reject openings whose core is ≥ 30% solid at body height.
- First version of the solid test: 5 points per bin. That killed good doors, because drift-smeared face points
  leak into thin cores (sparse, ~40 points per bin over 1 m of height). Fixed by a density criterion of
  50 points per 5 cm bin.

**I-7. The same door measured 0.805 m in one capture and 0.909 m in the other.**
- Root cause (figure inspected): the nearest dense layer to the opening was a short vertical element (door-leaf
  edge or fitting) seen only at 1.35–1.95 m, 13 cm inside the real jamb.
- First fix: require a tall face (absolute ≥ 40% height coverage). Results 0.894 / 0.857. The window edge biased
  the median, and the absolute threshold flipped when the floor reference moved by 2.5 cm (0.913 vs 0.805 for the
  same door).
- Final fix: relative coverage (≥ 80% of the best layer) plus mean shift onto the face. Result: **0.911 vs
  0.917 m**.
- Why relative: it does not depend on how much of the jamb happened to be visible, which varies with how the phone
  was held (floor_only was aimed down).

**I-8. Free wall tilt skewed rooms.**
- Symptom: after allowing each wall a fitted tilt (≤ 5°), r3's left wall hit the 5° clip and rooms overlapped by
  0.137 m².
- Root cause: on smeared or cluttered walls the tilt fits the clutter.
- Evidence for keeping some tilt: on several walls a tilted line removes most scatter (rmse 18.5 → 2.8 mm,
  16.8 → 4.3 mm). Median rmse: axis-aligned 12.0 vs tilted 8.3 mm on floor_only, 19.1 vs 15.9 mm on with_ceiling.
- Fix: accept a tilt only when it makes a clean plane (A-9).

**I-9. Wall RMSE of 11–17 mm (median), against ~5 mm expected from the sensor.**
- Root causes:
  1. drift: two passes see the wall at different positions; floor_only is bent;
  2. curtains in front of glass (RGB frames #5846, #6496 show sheer curtains on full-height windows);
  3. walls slightly off square.
- Fix:
  - (3) by A-9;
  - (1) and (2) cannot be fixed here, so they enter the interval as the inconsistency term (A-8);
  - (1) is only partly helped by the drift module: median wall-face rmse goes 12.1 → 10.9 (single_room),
    12.3 → 10.7 (with_ceiling) and 8.2 → 9.3 mm (floor_only); the smear tail stays (p90 24–29 mm).

**I-10. Ceiling with two layers 11 cm apart (with_ceiling r2).**
- Root cause: vertical drift between the two passes. Evidence: on the drift-corrected scene the second layer
  disappears and the height is 3.066 m [3.050, 3.081].
- Fix: when two big layers lie within 20 cm, widen the interval to span both and mark the value `inferred` (A-14).
  A ±1 cm interval would have been confident garbage.

**I-11. Bounding boxes of the same room differ by 7.5 cm between captures.**
- Root cause: the wall that sets the length is a curtain (rmse 23.6 / 39.4 mm). Curtains move between captures.
- Status: the interval already covers it (z = −1.37). A product fix needs a curtain detector (a wavy surface plus
  see-through behind it) that measures the glass or wall behind instead (section 6).

**I-12. Small overlaps (mm × m strips) between rooms after the raw-point refit.**
- Root cause: rooms that share a boundary with no measurable wall thickness (a thin partition, or a passage split
  between two rooms) are fitted from each side independently, and the faces cross within noise.
- Fix: reconcile before corners (A-10). The measured face wins over an inferred one; if both are measured, they
  meet halfway and half the crossing goes into σ.
- A first version moved only the offset, not the tilt, so tilted faces still crossed at their ends. It now adopts
  the whole line.

**I-13. Runtime 72 s on with_ceiling under load, close to the 90 s budget.**
- Profile: `np.add.at` 14 s, window rasters over 4M points 15 s, stable argsort of 12M points 12.5 s, one 4M-point
  KD-tree.
- Fixes: `bincount`; local KD-trees per candidate; default (still deterministic) sort on float32 keys. Result:
  15 s on an idle machine, 25–34 s while other agents loaded all 22 cores.
- A 1-in-8 window subsample was tried and reverted: windows rose 6 → 8 because thinly seen wall turned into
  "no-return holes".

**I-14. Duplicate doors at room corners.**
- Root cause: two perpendicular edge pairs at a corner both reported the same opening.
- Fix: dedupe also by room pair and centre distance < 0.5 m.

**I-15. A non-rectilinear polygon raised an exception.**
- Fix: a cold run must not crash. The room is dropped with a warning and removed from the room map, so no door
  refers to it. It never triggered on the three captures.

---

## 5. Results on the captures

All numbers come from `outputs/plan_alpha/summary.json` and `*/metrics.json`.

- "ARKit" means the scene built with ARKit poses as-is.
- "drift-on" means the drift module's corrected scene. That is the one the pipeline should report, because
  "poses used as-is" fails the drift gate.
- `drift_sigma` = 1 cm (default) in all runs below.

| Capture | Poses | Rooms | Walls (length measured) | Doors (jambs measured) | Windows | Net area m² [95%] | Max overlap | Runtime s |
|---|---|---|---|---|---|---|---|---|
| single_room | ARKit | 3 | 20 (11) | 2 (2) | 1 | 25.13 [24.76, 25.47] | 0 | 3.5 |
| single_room | drift-on | 3 | 26 (12) | 2 (2) | 0 | 24.62 [24.30, 24.94] | 0 | 3.9 |
| floor_only | ARKit | 11 | 94 (32) | 7 (6) | 3 | 66.37 [65.91, 66.84] | 0 | 13.1 |
| floor_only | drift-on | 10 | 78 (27) | 6 (4) | 5 | 65.75 [65.23, 66.28] | 0 | 17.1 |
| with_ceiling | ARKit | 10 | 88 (35) | 5 (2) | 6 | 69.35 [68.69, 70.01] | 0 | 32.5* |
| with_ceiling | drift-on | 9 | 86 (35) | 7 (5) | 4 | 67.91 [67.36, 68.43] | 0 | 34.2* |

\* Measured while the other agents loaded all cores; 15 s on an idle machine (I-13). Every room has ≥ 4 walls.

**Ceiling heights, with_ceiling, drift-on** (m, [95%]):

| Room | Height | Room | Height |
|---|---|---|---|
| r0 | 2.469 [2.451, 2.488] | r4 | 2.364 [2.350, 2.378] |
| r1 | 3.082 [3.061, 3.104] | r5 | 3.055 [3.027, 3.084] |
| r2 | 3.066 [3.050, 3.081] | r6 | 2.380 [2.363, 2.396] |
| r3 | 3.087 [3.073, 3.101] | r8 | 2.479 [2.465, 2.493] |

r7 (seen through a door, not entered) is `not_observed`. floor_only and single_room report `not_observed` for every
room, as they must: the ceiling was never in view.

**Door widths.** Jamb-measured doors have a 95% half-width of 1.39–1.48 cm. Inferred widths have ±3.9 cm.

**Wall faces.**
- Measured faces: 14–55 per capture; rmse p10 / p50 / p90 = 4–6 / 8–12 / 24–32 mm.
- Median length half-width: 4.7–5.8 cm, dominated by the inconsistency term on smeared or curtained walls.
- Length status: 27–35% of lengths are `measured`. The rest have a virtual boundary or a short unmeasured
  neighbour at one end (section 6).

**Wall thickness** (with_ceiling drift-on): pairs from 4.4 to 39 cm. 10–20 cm values are typical partitions.
Values > 30 cm are double walls or shafts.

**Mirrors.**
- with_ceiling drift-on: r0_w3 (two regions) and r6_w9.
- floor_only drift-on: r2_w4 and r7_w4.
- All are reported as warnings, not windows.

**Repeatability self-check** (floor_only vs with_ceiling, same apartment; registration from geometry: yaw 89.05°,
residual 1.2 cm, floor IoU 0.79):

| Poses | Rooms matched | Walls pass / fail / unmatched | Median wall Δ | Interval coverage | Openings matched |
|---|---|---|---|---|---|
| ARKit | 10 | 0 / 53 / 72 | 23.1 cm | 0.25 | 3 |
| drift-on | 8 | 8 / 42 / 44 | 6.4 cm | 0.54 | 6 |

Rooms by area (drift-on; full tables in `repeatability/*.md`). Several rooms agree to a few cm or a few %:

| Room | Area (m²), with_ceiling vs floor_only | bbox (m), with_ceiling vs floor_only |
|---|---|---|
| r1 | 12.47 vs 12.40 | 4.369×3.205 vs 4.606×3.167 |
| r3 | 9.66 vs 9.42 | 3.503×3.015 vs 3.428×2.979 |
| r4 | 6.52 vs 6.27 | 3.138×2.630 vs 3.065×2.631 |
| r2 | 11.13 vs 10.02 | the room is split differently |

Matched door widths, drift-on:
- both sides jamb-measured: Δ 0.56 cm, 1.27 cm and 9.2 cm;
- at least one side inferred: Δ 4.0 cm and 10 cm.

**Gate status, honestly.**
- Repeatability (1 cm / 0.5% per wall): **fail**. The causes are segmentation differences (the same space split or
  notched differently), curtains, and residual drift.
- Opening widths ≤ 2 cm: when both jambs are measured, 2 of 3 matched pairs agree within 2 cm. There is no ground
  truth, so absolute error cannot be scored.
- Ceiling ≤ 1.5 cm: intervals are 1.4–2.9 cm (95% half-width). The spread across captures cannot be evaluated,
  because only one capture sees the ceiling.

Figures (all inspected): `outputs/plan_alpha/<capture>/plan_debug.png` and
`outputs/plan_alpha/drift_on__<capture>/plan_debug.png`.

---

## 6. Limitations, failure modes and what to do next

**Glass.**
- Windows: clear glass gives see-through (detected). Dark glass gives no return, which is weaker evidence
  (confidence 0.4; a black TV looks the same).
- The glass shower screen in single_room is a thin vertical surface with partial returns. It can create a line,
  but it is never "tall barrier everywhere", so it does not split the bathroom.
- Glass partitions between rooms are not modelled as walls with openings.

**Mirrors.**
- Detected by the reflection test, reported as a warning.
- Not yet handled: phantom free space behind a large mirror can make the inside/outside labelling extend a room
  behind the wall. Next step: drop raw points and rays that the mirror test explains before computing free space.

**Curtains and wet-look surfaces.**
- Curtains are measured as the wall (I-11). The interval widens, but the value is biased toward the room.
  Next step: detect wavy, high-rmse faces with see-through behind them and measure the plane behind.
- Wet-look or glossy floors drop to low confidence. The floor evidence then comes from rays, so labelling still
  works; the per-room floor level falls back to the global floor (wider σ) if too few points remain.

**Low light.** The LiDAR does not need light, so geometry is unaffected (these captures are at night). ARKit
tracking gets worse in low light, which shows up as drift and is the drift module's job.

**Clutter.**
- Furniture below 1.2 m never splits rooms (it is not a barrier).
- Tall furniture (wardrobes) acts as a wall, so a wardrobe front can become a room boundary. Its area is then
  excluded from the room.
- Faces behind furniture with no raw support are `inferred`.

**Segmentation instability is the main open problem, and the proposed Part 4 fix-loop target.**
- The same space is split differently in two captures: corridor pieces, notches at doorways, a wardrobe region in
  or out.
- This is the main reason the repeatability gate fails even on drift-corrected scenes (unmatched walls,
  p90 Δ 95 cm).
- Candidate fixes:
  1. Regularise room polygons toward rectangles or larger convex parts (a minimum description length penalty on
     vertices).
  2. Make notch decisions from wall evidence, not from basin snapping: a notch survives only if its boundary is
     ≥ 50% covered by wall.
  3. Decide room splits by door detection (a jamb pair present) instead of by basin geometry.
- Predicted effect of (2) + (3): matched walls rise and the p90 Δ falls below 10 cm. To be verified with
  `--pair`.

**Length status.**
- About 65% of wall lengths are `inferred`, because one end is a virtual boundary (an opening between rooms) or a
  short unmeasured edge. This is honest but unattractive.
- Next step: measure lengths between the nearest *measured* perpendicular walls (bounding dimensions), and report
  the room's length × width from its two pairs of longest faces.

**Non-Manhattan walls.**
- Walls more than 5° off the axes are not represented (they become a staircase of short edges or are ignored).
- These captures have none apart from open door leaves.
- Next step: add a third and fourth line family from the residual normal histogram.

**Single-floor only.** Each room records `floor_level`, but multi-storey stitching is out of scope.

**Ground truth.** None exists for this data. Every accuracy claim above is repeatability or internal consistency,
not accuracy against a tape or laser.

---

## Requested changes to shared code

None required. One suggestion:

- `floorplan/model.py`: consider an optional `Wall.tilt_deg` field, so a near-Manhattan wall's tilt is in the schema
  rather than only in `meta.wall_fit`.
