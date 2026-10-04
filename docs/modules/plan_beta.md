# plan_beta: space-first floor-plan extraction (LiDAR tier)

Code: `floorplan/plan/beta/` (package). Entry point: `extract_plan(scene, info, capture_id) -> floorplan.model.Plan`.
Run: `scripts/run_plan_beta.py`. Evidence: `outputs/plan_beta/`.

```bash
cd floorplan-capture
PY=../.venv/bin/python
# one capture -> outputs/plan_beta/<capture>/{plan.json, metrics.json, debug.png, openings.png}
$PY scripts/run_plan_beta.py outputs/single_room__c00a170fe1
# the repeatability pair, plus the self-check -> outputs/plan_beta/repeatability_<A>_vs_<B>.json
$PY scripts/run_plan_beta.py outputs/single_scan_with_ceiling__c7d28f72c6 \
    outputs/single_scan_floor_only__1a8384c3f6 --repeatability
# ablation: any BetaParams field can be overridden (used for the wall-slope ablation, I-9)
$PY scripts/run_plan_beta.py <A> <B> --repeatability --set wall_slope_mode=room --out outputs/plan_beta/ablation_slope_room
```

All numbers in this document come from those commands. They are collected in `outputs/plan_beta/summary.txt`.

---

## 1. Purpose and where it sits in the pipeline

```
capture (Stray Scanner) -> prepare_scene.py (TSDF + raw points, gravity/Manhattan aligned)   [shared, done]
                        -> plan_beta  (THIS MODULE: rooms, walls, openings, ceiling, intervals)
                        -> export / stitching / evaluation                                    [other modules]
```

The input is the aligned scene. It contains a TSDF surface with normals for finding structure, about 2.5-16 M raw
high-confidence LiDAR points with their viewing rays for measuring, and the camera path. The output is a `Plan`
(tier `lidar`):

- one `Room` per room or corridor, with a counter-clockwise polygon, area, perimeter, bbox and ceiling height;
- `Wall` segments with lengths, thickness, support and fit error;
- `Opening`s (doors, passages, windows) with widths measured on the jamb faces;
- `Adjacency` between rooms and a footprint area;
- a 95% interval on every number, with status `measured`, `inferred` or `not_observed`.

**"Space-first"** means the rooms are found from the **empty space inside them**, not from their walls. Two plan
extractors are being built in parallel. This one follows the room-segmentation survey of Bormann et al. 2016
(morphological, distance-transform and Voronoi segmentation of free space, from robotics). The other one
(`plan_alpha`) starts from walls. Space-first is robust where wall evidence is broken (doors, furniture in front of
walls, the floor-only capture that barely sees walls above 1 m). Its weakness is that a room's outline must still
be snapped to walls, which step 3 does.

---

## 2. How it works, step by step

| Step | Module | What it does | Why it is needed | What goes wrong without it |
|---|---|---|---|---|
| 1 | `freespace.py` | Builds a 2 cm plan raster of **interior free space**. Free space is floor seen by the LiDAR from above, plus a 30 cm disc around the camera path, plus **ray carving**. Ray carving marks every cell a LiDAR ray crossed below 1.0 m; that part of the ray flew over floor or furniture. Small, mostly free-bordered holes (furniture footprints) are then filled. **Wall evidence** is subtracted: vertical TSDF surface over at least 40 cm of the 1.0-2.0 m height band, dilated 4 cm. Islands not touching the camera path are dropped. | Rooms are defined by where you can stand. The floor alone is full of holes: sofas, beds and dark floors give no returns. | Without carving, rooms are 60-80% covered and fall apart into fragments (I-1). Without wall subtraction, rooms leak into each other through walls. |
| 2 | `segment.py` | Computes the **distance transform** (DT) of free space, which is the distance from each cell to the nearest obstacle. Seeds are DT peaks (h-maxima, 10 cm). A **watershed** grows the seeds until they meet; basins meet at the narrowest point between seeds, which is the doorway when there is one. Regions are merged while the place where they meet is wider than a door (> 1.3 m). Narrower connections separate rooms. Regions under 1.2 m² merge into their widest neighbour, or are dropped if they only connect through a gap narrower than 0.5 m. | Rooms are wide and doors are narrow. The DT turns "narrow" into a number. | Too few seeds: an L-shaped room or a corridor is swallowed. No merging: one room per furniture-free blob. |
| 3a | `walls.py` | Generates **candidate wall lines**. Vertical TSDF points within 25 cm of the room, whose normals point into it, vote in 1 cm histograms of x (walls facing ±x) and z (walls facing ±z). Peaks with at least 30 cm of support become lines. Straight runs of at least 30 cm in the room mask's border add "inferred" lines where no wall was seen (an open passage, an unobserved wall). | Walls are axis-aligned after the Manhattan alignment (D-009), so each wall is a 1-D position. | |
| 3b | `walls.py` | **Line arrangement.** The x-lines and z-lines cut the plane into rectangles. A rectangle belongs to the room when the room mask covers at least 50% of it. The union of chosen rectangles is the polygon. It is rectilinear, its corners sit exactly at line intersections, and a spurious line only becomes an edge if coverage changes across it. Cleanup steps follow: jogs (steps under 6 cm) are snapped away, wall-deep and door-wide bumps that open onto seen space are cut as doorway recesses, overlaps between rooms go to the room whose free space covers more of the overlap, and slivers under 1.5 cm are removed. | Turns a ragged raster blob into a clean polygon a homeowner recognises. | Tracing the raster boundary gives staircase outlines with hundreds of 2 cm edges and no notion of "wall". |
| 3c | `walls.py` | **Raw-point refinement** (D-007). Each wall line is re-measured on raw LiDAR points within ±4 cm, between 0.3 and 2.0 m height, away from corners, and seen from inside the room (the viewing ray must point toward the wall). The estimator starts at the mode of a 2 mm histogram, then does trimmed least squares with a window shrinking from ±5 cm to ±1.5 cm. Corners are line intersections and lengths are corner-to-corner. A wall that moves more than 6 cm from its free-space position has locked onto another surface, so it is reset and marked inferred. If two rooms overlap after refinement, the lines running along the overlap are reset; if they still overlap, the less certain line snaps onto the other. | The raster is 2 cm and blurred by the dilations; dimensions must come from the sensor's own points. The ray-direction test keeps the other face of a thin wall, and things seen through a doorway, out of the fit. | Dimensions would be quantised to 2 cm and biased by the 4 cm wall dilation. |
| 4 | `levels.py` | Per room: **floor level** from raw points within 5 cm of the floor that were seen from above, and **ceiling height** from raw points at least 1.9 m above the room's floor that were seen from below, inside the polygon shrunk by 20 cm. Several ceiling levels are common (bulkheads, dropped ceilings), so the level covering the largest area (10 cm patches, not point count) is reported and the others are listed. If less than 10% of the room's ceiling was seen, the result is `not_observed`. | The ceiling-height gate is 1.5 cm per room, and bathroom floors are often 1-3 cm lower. | Point counts favour whatever surface the phone was closest to (I-6). Without the "seen from below" rule, phantom points from glossy floors and wardrobe tops pollute the levels. |
| 5 | `openings.py`, `extract.py` | **Openings.** (a) Doors and passages come from the necks that kept two rooms apart. Width is measured along the cut line on raw points: in each 5 cm height slice (0.3-1.8 m) the nearest solid 1 cm bin on each side of the centre marks the jamb face, and the median over slices is taken. (b) Doorways to unmapped space come from the bumps cut in 3b. (c) **Wall-plane see-through test**: rays whose camera was inside the room, that crossed the wall line and hit something at least 10 cm behind it. Open from the floor to above 1.8 m means a door. Framed (wall below the sill and on both sides) with a sill of at least 0.3 m means a window. A **mirror test** reflects the see-through points back across the wall; a mirror's phantoms land on real surfaces of the room. Window-sized regions with no returns at all (glass, or a TV) become candidates in `plan.meta`, not openings. | Opening width is a headline gate (≤ 2 cm on ≥ 85% of openings, phantom = miss). | Measuring "along the nearest wall" gave wrong numbers for corridor constrictions (I-10). Without the mirror and framing rules, mirrors and furniture gaps become phantom windows (I-11, I-12). |
| 6 | `measure.py` | **Intervals.** Every quantity is a function of the wall-line offsets. Monte Carlo draws all offsets (400 draws per room) and recomputes corners, wall lengths, area, perimeter and bbox; the footprint uses the union of all rooms (200 draws). The 2.5% and 97.5% quantiles form the 95% interval. | "A confidence interval on every measurement." | |
| - | `selfcheck.py` | **Repeatability self-check** for two captures of the same flat. It registers B onto A with the shared `floorplan/eval/register.py` (from geometry, never from the plans), matches rooms by IoU, then compares walls and opposite-wall distances. | There is no ground truth; repeatability is the measurable gate. | |

### The uncertainty model (what the ± means)

For each wall line, 1-sigma:

```
sigma_offset^2 = (rmse / sqrt(n_patches))^2     fit standard error; n_patches = distinct 10x10 cm patches,
                                                not points, because neighbouring points share the same bump
                                                and the same pose error and are not independent
               + lidar_sigma^2                  5 mm per surface: systematic LiDAR bias (iPhone/iPad LiDAR is
                                                quoted at ~1 cm absolute)
               + drift_sigma^2                  1 cm per wall: residual pose drift (placeholder until the drift
                                                module supplies a measured value)
               + (|tilt| * reach / 2)^2         the wall's measured deviation from the Manhattan axis (I-9) times
                                                the distance from the fitted centre to the wall's far end, halved
inferred wall (no raw-point support): sigma_offset = 5 cm
```

- **Wall length:** Monte Carlo over the two neighbouring wall lines (a corner moves when its wall moves).
- **Ceiling height:** sigma² = se_ceiling² + se_floor² + 2·lidar_sigma² + vertical_drift² (5 mm; vertical drift is
  smaller because the IMU observes gravity continuously).
- **Door width:** each jamb has sigma² = (1.2533·std/√n_slices)² + lidar_sigma²; the width adds the two jambs in
  quadrature. The factor 1.2533 is the standard error of a median.
- **Unmeasured widths** (jambs not resolved) get sigma = 5 cm and status `inferred`.

---

## 3. Decisions

Each decision lists the options, the choice, why, and the evidence.

**D-B1 Free space = floor ∪ camera path ∪ low ray carving, then bounded hole filling.**
- *Options:* (a) observed floor only; (b) floor plus a big morphological closing; (c) floor plus ray carving.
- *Choice:* (c), plus filling holes of at most 6 m² that are at least 50% bordered by free space.
- *Why:* the floor alone misses everything under furniture. A closing large enough to bridge a bed (≥ 0.8 m radius)
  also bridges walls. Carving uses information we already have, because every LiDAR ray proves the space it
  crossed was empty, and the 1.0 m height cap keeps it below window sills on most rays.
- *Evidence:* `diagnostics/I01_*.png`: rooms go from patchy to fully covered. Runtime is 1.9-10 s per capture.

**D-B2 Wall evidence = vertical TSDF surface over ≥ 40 cm of the 1.0-2.0 m band.**
- *Options:* any vertical surface; any surface in the band; vertical extent in the band; raw points instead of
  the TSDF.
- *Choice:* vertical extent in the band, measured on the TSDF.
- *Why:* below 1 m almost everything is furniture. A desk with a monitor reaches into the band but covers only
  about 20 cm of it. Raw points add noise; the TSDF already averages.
- *Evidence:* `diagnostics/I08_wall_profile_desk_not_wall.png`. The "wall" between a bedroom and a corridor had
  surface only at 0.75-1.2 m, with the LiDAR seeing through above it. The fix merged the two into one room.
  `I08_wall_evidence_tsdf_vs_raw.png` shows raw points thickening walls and adding clutter.

**D-B3 Segmentation: over-segment (h-maxima 10 cm), then merge only wider-than-door connections.**
- *Options:* threshold the DT at one level; Voronoi graph; watershed with merging.
- *Choice:* watershed with merging.
- *Why:* a single DT threshold cannot handle a 1.0 m corridor next to a 0.8 m door. Wrong merges are visible (a
  missing door) while wrong seeds are not, so over-segmenting first is the safe direction.
- *Evidence:* I-2. With the inverted rule, floor_only produced 1 region for 3 rooms; after the fix it produces 7
  rooms that match with_ceiling's 7 rooms (IoU 0.67-0.94).

**D-B4 Polygon by line arrangement and coverage, not by tracing the raster.**
- *Options:* trace contours and simplify (Douglas-Peucker); fit a minimum rectilinear polygon; use an arrangement.
- *Choice:* arrangement.
- *Why:* corners come out exactly at wall-line intersections, which is what "corner = intersection of adjacent
  fitted walls" requires. The result is rectilinear by construction and robust to spurious lines.
- *Evidence:* `debug.png` for each capture. Every room has at least 4 walls (min 4, 6 and 12 per capture), and
  rooms do not overlap (max overlap 0.0 m² on all three captures).

**D-B5 Dimensions from raw points: mode start, trimmed least squares, ±4 cm band, viewed from inside.**
- *Options:* the TSDF surface (rounded by 2 cm voxels); RANSAC planes; a mode plus trimmed mean.
- *Choice:* mode plus trimmed mean.
- *Why:* the wall face is the densest surface in the band, so the mode finds it. The trimmed mean then averages
  thousands of points without being pulled by skirting or furniture. The ray-direction test is free, because every
  raw point stores its ray.
- *Evidence:* median wall-fit RMSE is 7.3-7.7 mm, which is the LiDAR noise level. Opposite-wall distances in the
  same room repeat to 0.0, 0.4, 1.1 and 4.5 cm between the two captures (section 5).

**D-B6 Walls stay axis-aligned; the measured tilt goes into the interval** (`wall_slope_mode=axis`).
- *Options:* axis-aligned; a free slope per wall; one rotation per room.
- *Choice:* axis-aligned, with the tilt added to sigma.
- *Why:* see I-9. Neither slope model improved repeatability. Per-wall tilts disagree within a room, so the
  distortion is not a rigid rotation and modelling it adds noise. The tilt is real, so it widens the interval
  instead.
- *Evidence:* `outputs/plan_beta/ablation_slope_summary.json`. For same-surface opposite distances the median |Δ|
  is 1.05 cm (axis), 1.38 cm (room) and 2.12 cm (free). Comparable walls within the 95% interval: 5/5 (axis),
  4/4 (room), 4/6 (free).

**D-B7 Doors and passages are measured along the cut, with density-based jamb edges.**
- *Options:* along the nearest room face; along the cut with the nearest point; along the cut with the nearest
  solid bin.
- *Choice:* along the cut, nearest solid bin.
- *Why:* see I-10. Many necks cross a corridor rather than a wall, and handles or a partly open door leaf leave
  1-2 stray points.
- *Evidence:* `openings.png`. with_ceiling doors measure 0.68, 0.83, 0.73 and 0.68 m with 95% half-widths of
  1.4-5.3 cm.

**D-B8 Only well-supported openings are reported; weak detections go to `plan.meta`.**
- *Why:* the gate scores a phantom opening as a miss, so a weak detection costs as much as a missed one, and a
  reviewer can still see the candidates.
- *Rules:* an opening whose width could not be measured on jambs gets confidence ≤ 0.5. Connections narrower
  than 0.5 m with no measurable gap are `weak_links` (an unobserved wall), not doors. Glass or no-return regions
  are `window_candidates`.

**D-B9 Ceiling = the level covering the largest area of the room.**
- *Options:* the highest strong peak; the most points; the most area.
- *Choice:* the most area.
- *Why:* see I-6. Dropped ceilings are real. Points over-represent surfaces close to the phone.

**D-B10 Rooms never entered are kept only if at least 60% of their outline is measured wall.**
- *Why:* free space seen through glass or a door (a balcony, a corridor seen from a doorway) is not a mapped
  room unless its walls were actually measured.
- *Evidence:* with_ceiling's balcony fan and single_room's corridor fan are dropped
  (`plan.meta.dropped_regions`).

---

## 4. Issues log

Each entry gives the symptom, the root cause and its evidence, the possible fixes, and the fix chosen and why.

- **I-1 Free space full of holes, rooms fragmented.**
  - *Symptom:* on the floor-plus-path map, bedrooms were 60-80% covered.
  - *Root cause:* furniture hides the floor. Hole filling alone failed because holes touch walls with gaps and so
    "leak" to the outside.
  - *Possible fixes:* larger closing (bridges walls); more hole filling (same leak); ray carving.
  - *Chosen:* ray carving below 1.0 m, because it is evidence-based.
  - *Side effect:* fan-shaped regions seen through openings appear. These are handled by D-B10 and by keeping only
    components that touch the camera path.

- **I-2 Three rooms merged into one (floor_only).**
  - *Symptom:* the merge log showed regions merging over necks of 0.34-0.44 m.
  - *Root cause:* my rule "keep separate only if the neck is a door (0.5-1.3 m, with jambs)" merged everything
    narrower than a door. A narrow neck is the strongest evidence of separate spaces (a partly closed door).
  - *Chosen:* separate whenever the neck is ≤ 1.3 m. Jambs only set the opening's confidence.

- **I-3 Jogs and doorway bumps in polygons.**
  - *Symptom:* 2-4 cm steps, and 11 cm recesses at doors.
  - *Root cause 1:* two parallel evidence lines (two capture passes, or skirting).
  - *Root cause 2:* a floating-point bug, where `sign(-1e-17)` is not 0, so the jog test silently failed. Fixed by
    rounding the direction vectors.
  - *Root cause 3:* free space continues into the door slab.
  - *Chosen:* snap jogs; cut wall-deep, door-wide bumps only if open space is seen beyond them (a closed niche is
    floor area and is kept).

- **I-4 Rooms overlapped.**
  - Before refinement, each room's own arrangement claimed shared areas. Fix: give the area to the room whose free
    space covers more of it.
  - After refinement, two rooms' lines on a virtual boundary each "measured" surfaces seen through the doorway,
    and crossed by 15 cm (traced shifts of 7-13 cm). Fixes:
    - a ±4 cm band;
    - a 6 cm maximum shift from the free-space position;
    - reverting the lines along the overlap strip;
    - finally, snapping a shared boundary.
  - Reverting every edge touching the overlap was too aggressive: measured walls fell from 48 to 30 on
    with_ceiling. Restricting the revert to edges running along the strip fixed that.
  - *Result:* max overlap is 0.0 m² on all captures.

- **I-5 Refinement changed other rooms' geometry.**
  - *Root cause:* `Line` objects were shared between rooms and refined in place.
  - *Fix:* copy them in `polygon_to_geometry`.

- **I-6 Ceiling heights 2.3-2.5 m against 3.10 m globally.**
  - *Checked:* the per-room histograms (`diagnostics/I06_*`). Most of these rooms show one clean level at
    2.28-2.47 m: real dropped ceilings (bathroom, kitchen, hall). R1 and the corridor show two levels.
  - *Fix:* pick the level with the largest patch coverage and list the others in the method string.
  - *Second bug:* coverage above 100% (patches straddling the polygon edge), now clipped.

- **I-7 Phantom 2-3 m "doors".**
  - *Root cause:* the see-through test accepted any camera on the room side of the wall's infinite line, so
    cameras in the corridor beside the room counted.
  - *Fix:* the camera origin must be inside the room polygon.

- **I-8 A desk counted as a wall.**
  - *Fix:* the vertical-extent criterion (D-B2).
  - *Side effect, accepted:* a wall made of glass above 1.3 m (the bathroom shower screen in with_ceiling has
    returns only at 0.3-1.3 m) no longer separates the bathroom from the bedroom. That follows the evidence. See
    limitations.

- **I-9 Walls are tilted 0.4-1.1° (median) from the Manhattan axes.**
  - *Symptom:* the p90 tilt is 1.9-3.0° (`plan.meta.wall_tilt_deg`). Over 4 m, 1° is 7 cm.
  - *Root cause:* the global Manhattan yaw cannot fit every wall when the reconstruction has residual yaw or drift,
    or the walls are not perfectly square. The two captures also need an 89.05° (not 90°) rotation to register.
  - *Fixes tried:* (1) a free slope per wall; (2) one rotation per room.
  - *Evidence:* D-B6. Neither improved repeatability. Per-wall tilts disagree inside one room (one axis +1°, the
    other mixed), so the distortion is not rigid.
  - *Chosen:* keep walls axis-aligned and widen sigma by |tilt|·reach/2. This is honest and repeatable. The real fix
    is upstream: drift-corrected poses from the drift module.

- **I-10 Door widths wrong (0.26-0.49 m).**
  - *Evidence:* `diagnostics/I10_*`. Many necks cut across a corridor between two parallel walls, and I was
    measuring along the corridor wall.
  - *Fix 1:* measure along the cut line, whose direction is the long axis of the boundary between the two
    regions.
  - *Fix 2:* stray points (a handle, a door leaf) stopped the gap at about 0.43 m, so the jamb edge became the
    nearest solid 1 cm bin.
  - *Result:* doors read 0.58-0.89 m.

- **I-11 Six "mirrors" on floor_only.**
  - *Root cause:* walls perpendicular to the plane continue through doorways and reflect onto themselves, and
    symmetric neighbouring rooms match their reflections.
  - *Fix:* the three-part test. Phantoms must not lie inside another mapped room, only plane-facing surfaces count,
    and the reflected match must be ≥ 50%.
  - *Result:* 0 mirrors on all captures. I cannot confirm there is no mirror in the flat (no ground truth). The
    earlier bathroom-mirror detection (match 82%, 1.0-1.6 m high) disappeared when that bathroom merged into the
    bedroom (I-8), and full-height wardrobe mirrors are a known blind spot.

- **I-12 Phantom windows** (sill about 0.25 m, 0.4-0.6 m wide: gaps over furniture).
  - *Fix:* require framing (wall below the sill in ≥ 60% of columns and on both sides) and a sill of ≥ 0.3 m.
  - *Result:* windows dropped from 12-17 to 1-2 per capture; the rest went to `window_candidates`.

- **I-13 Sub-door "doors" (0.16-0.40 m).**
  - Narrow necks with no measurable gap become `weak_links`; bump doors must be ≥ 0.5 m wide.

- **I-14 Door heads at 1.5-1.6 m.**
  - *Root cause:* the lowest points above 1.5 m inside the gap were a door leaf or frame.
  - *Fix:* the head is the lowest level whose points span ≥ 60% of the gap.
  - *Result:* 2.02, 2.10 and 2.09 m. Unobserved heads are now reported as missing.

- **I-15 Runtime 118 s under machine load** (other agents share the CPU).
  - *Fix:* prefilter the floor and ceiling candidate points once (every 3rd point), prefilter vertical TSDF points
    once, and halve the see-through rays.
  - *Result:* 3-11 s for single_room, 16-36 s for floor_only, 30-52 s for with_ceiling (varies with machine
    load).

---

## 5. Results on the captures

There is no ground truth, so accuracy cannot be scored. What is reported:

| Capture | Rooms | Walls (measured) | Openings | Windows | Window candidates | Footprint m² (95%) | Max overlap | Runtime |
|---|---|---|---|---|---|---|---|---|
| single_room | 2 | 28 (11) | 1 passage | 0 | 3 | 17.52 [17.13, 17.79] | 0.0 | 10.5 s |
| floor_only | 7 | 70 (39) | 3 doors, 2 passages | 1 | 7 | 54.96 [54.19, 55.42] | 0.0 | 36 s |
| with_ceiling | 7 | 84 (43) | 8 doors, 2 passages | 2 | 8 | 62.39 [61.57, 62.91] | 0.0 | 52 s |

Figures: `outputs/plan_beta/<capture>/debug.png` (plan) and `openings.png` (raw-point evidence for every
door/passage width).

**Ceiling heights (with_ceiling; the other two captures correctly report `not_observed` for every room):**

| Room | R1 | R2 | R3 | R4 | R5 | R6 | R7 |
|---|---|---|---|---|---|---|---|
| Height (m) | 2.417 | 3.084 | 2.958 | 3.082 | 2.343 | 2.279 | 2.469 |

- Every value is ±0.017 m (95%).
- R1 also sees 2.37 m (28% of the room) and R3 sees 3.07 m (32%). These are bulkheads, listed in the method
  string.
- The global estimate from align.py was 3.10 m. It is close to the high-ceiling rooms and misses the dropped
  ones.

**Doors (with_ceiling):** 0.682 ±0.053, 0.831 ±0.014, 0.730 ±0.031 and 0.680 ±0.029 m between rooms, with heads at
2.02, 2.10 and 2.09 m. Floor_only gives 0.581, 0.865 and 0.890 m. Matching doors one-to-one across the two captures
needs the register transform per opening; not done yet. The doors into unmapped space found by the see-through
test (with_ceiling O12, O13, O15, 1.2-1.6 m wide) have no second room to confirm them and are the least reliable
detections.

**Repeatability pair** (`repeatability_*.json`; with_ceiling = A, floor_only = B; registration: yaw 89.05°,
residual 1.15 cm, floor IoU 0.79, distinctiveness 2.59):

| Room (A/B) | IoU | Area A / B (m²) | bbox A (m) | bbox B (m) |
|---|---|---|---|---|
| R3/R3 bedroom | 0.94 | 10.90 / 11.22 | 4.826 x 2.980 | 4.815 x 2.984 |
| R2/R2 living | 0.86 | 12.06 / 11.57 | 4.428 x 3.036 | 4.388 x 2.849 |
| R6/R6 | 0.83 | 3.21 / 3.50 | 2.102 x 1.812 | 2.343 x 1.759 |
| R7/R7 | 0.82 | 1.88 / 2.20 | 1.966 x 1.007 | 1.986 x 1.107 |
| R1/R1 | 0.74 | 17.73 / 13.84 | 5.458 x 4.005 | 5.409 x 3.948 |
| R4/R4 | 0.71 | 9.26 / 7.07 | 3.151 x 3.001 | 2.940 x 2.848 |
| R5/R5 | 0.67 | 7.35 / 5.56 | 3.220 x 2.520 | 2.712 x 2.249 |

- **Opposite-wall distances** (what a laser measurer reads), 10 pairs:
  - On the **same surfaces** in both captures: Δ = 0.0, 0.4, -1.1, 4.5 and -5.7 cm.
  - The other 5 pairs differ by 10-28 cm, because one capture measured a furniture front or a different wall.
- **Walls with the same corners in both plans:** 5. Three pass the 1 cm / 0.5% gate (|Δ| 9.9, 9.7, 4.6 mm),
  two fail (68.9 and 10.5 mm), and **5/5 lie within their combined 95% interval**.
- **Verdict on the repeatability gate:** it **fails** at plan level. When both captures measure the same surface
  the numbers agree to about 1 cm. The failures are topological: which surfaces become walls (wardrobe fronts, a
  bulkhead), how much of a room each capture saw (floor_only aimed the phone down), and where an open junction is
  cut. This is "unrepeatable", not "repeatable-but-biased".

**Interval widths:** the median 95% half-width of a measured wall length is 3.8-4.2 cm. About 2.8 cm of that
comes from the 1 cm drift placeholder alone (√2 × 1.96 × 1 cm). The 2 cm opening gate and the 1 cm repeatability
gate are therefore below what this module can currently claim. A smaller measured drift sigma from the drift
module would shrink the intervals directly.

---

## 6. Limitations and failure modes

- **Glass** (shower screens, glass doors, windows).
  - The LiDAR passes through glass or returns at low confidence, which we drop (D-006). A glass partition
    therefore creates no wall, and the spaces on both sides merge. This is observed: the with_ceiling bathroom
    merged into the bedroom (vertical returns only at 0.3-1.3 m).
  - Windows usually show as "no returns" or "see-through to far away"; we report framed no-return regions as
    candidates only.
  - *Next:* use the RGB frames to detect glass and frames (the video-tier models), and project keyframe frustums
    to tell "viewed but no return" (glass) from "never viewed".
- **Mirrors.**
  - A mirror creates phantom space behind the wall. The three-part reflection test catches mounted mirrors.
  - A full-height mirror (wardrobe door) reaches the floor and is treated as a door candidate, so its phantom room
    can survive if it is enclosed. Mitigation: phantom rooms are never entered and are dropped unless ≥ 60%
    enclosed.
  - *Next:* apply the reflection test to whole regions, not only to wall openings.
- **Wet-look and glossy floors.**
  - Reflections appear below the floor (the raw height histograms show peaks at -0.3 to -0.5 m). Floor levels use
    only points within 5 cm of the floor that were seen from above, so phantoms are excluded.
  - A glossy floor can also lose returns. Carving and path discs fill the free space.
- **Low light.**
  - The LiDAR itself is unaffected, but ARKit tracking degrades, which means more drift: tilted walls (I-9),
    double walls and wider fits.
  - Intervals widen automatically through the fit RMSE and the tilt term, but the drift placeholder is constant.
- **Clutter and tall furniture.**
  - Wardrobes, fridges and bookshelves are tall vertical surfaces, so they count as walls and the room polygon
    stops at their front. Area is under-reported. This is the main cause of repeatability failures where one
    capture saw behind the furniture and the other did not.
  - *Next:* the ceiling-contact test, where a real wall reaches the ceiling and furniture does not (needs ceiling
    coverage).
- **Open plan and junctions.** Where a hall meets a living room with no wall, the boundary is where the watershed
  cut it, snapped to the nearest line, and marked `inferred`. Areas of such sub-rooms are ambiguous.
- **Non-Manhattan walls** (angled walls, bay windows, curves) are not modelled; they are approximated by staircase
  edges. *Next:* add angled candidate lines from TSDF normal clusters outside ±5°.
- **Single storey** only: one floor level drives the masks, and stairs are not handled.
- **Rooms never entered** are kept only when well enclosed. Their walls are seen at a distance, so they have wider
  intervals.

---

## Requested changes to shared code

1. **`floorplan/pipeline/scene.py`:** store a frame index (or timestamp) per raw point. This would allow
   splitting with_ceiling into its two passes for a within-capture repeatability and ceiling-spread check, and
   per-pass drift diagnostics.
2. **`floorplan/pipeline/scene.py`:** store depth intrinsics per keyframe. Projecting wall cells into keyframe
   frustums would distinguish "viewed but no return" (glass, dark surface) from "never viewed".
3. **`floorplan/model.py`:**
   - add `Wall.kind` (`solid` | `virtual`), so open-plan boundaries are not drawn as walls;
   - add an orientation or axis to `Opening`, so renderers do not have to guess it;
   - add a `meta` dict to `Room` (for secondary ceiling levels and the visited flag).
4. **Drift module:** expose the measured residual-drift sigma so `BetaParams.drift_sigma_m` (now a 1 cm
   placeholder) can use it.

---
---

# Part v2: the judge's hybrid, plus accuracy work (plan extractor v2)

Everything above this line is v1, kept unchanged because the evolution is part of the story. v1 is frozen in code
as `floorplan/plan/beta_v1/` (an exact copy, imports pointed at itself, never edited after the copy), so every
before/after number can be regenerated. v2 lives in `floorplan/plan/beta/` with the same entry point
`extract_plan(scene, info, capture_id, params=None) -> Plan`.

- One row per change, with before → after on the repeat pair: `docs/modules/plan_v2_ablation.md`.
- Harness: `scripts/plan_v2_eval.py` (imports the judge's `judge_common.compare`, perturbations and floor recall
  unchanged). Evidence: `outputs/plan_v2/`. Final plans: `outputs/plan_v2/<capture>/{plan.json, debug.png}`.

## v2.0 In one paragraph

The judge chose beta and asked for four alpha parts. v2 has them, measured one by one on the drift-corrected
canonical scenes (`outputs/scenes_v2`). The big one is B-1: alpha's long wall lines now cut the free space for the
segmentation, with a furniture test of my own (a cut must be a two-sided partition or rise ≥ 0.8 m), so the
floor_only room merge disappears without shrinking any room. Rooms matched 6 → 8 (all), walls passing 7 / 99 →
13 / 87, median |Δ| 15.0 → 8.2 cm, footprint and floor recall unchanged. My single-visit idea (X-1)
was implemented and **rejected by its own evidence** (same-topology median 1.9 → 3.2 cm). The gate is still
failed; what remains is mostly the extractor's own raster-phase sensitivity (found in this round, B-5) and the
2 cm residual misalignment between the captures.

## v2.1 How the pipeline changed (step table, v2 additions only)

| Step | Module | v2 change | Why | Switch (ablation) |
|---|---|---|---|---|
| 1b | `lines.py` (new) | Alpha's 1-D peak lines of vertical TSDF surface (0.3–2.0 m, per normal sign, 2 cm occupancy, ≥ 40 cm support). A run becomes a **cut** where an opposite face lies 5–40 cm behind it (both sides of a wall) **or** the face covers ≥ 0.8 m of the band. | v1's wall evidence starts at 1.0 m (D-B2); floor_only saw a partition only up to 1.3 m and two rooms merged (J-I-4). Lowering the band cut rooms at furniture (J-I-5). | `use_sep_lines`, `barrier_pair_required`, `sep_min_extent_m` |
| 2 | `segment.py` | The cut is an obstacle for the distance transform and watershed only; cut cells are then given to the nearest room (exact EDT indices). An unentered region that a cut split off goes back to its entered neighbour. | Separate rooms, never lose floor. | (same) |
| 2b | `extract.py` | Photo-tier folders (`raw_point_room`): touching regions whose floor points vote for the same folder are merged. | X-4: one room per folder; only merging, never splitting (a photo through a doorway must not cut a room). | `use_room_hints` |
| 3c | `walls.py` | Weighted fit, w = 1/σ_lidar(range)² (D-011 table). | X-2: near points are less noisy. | `noise_weights` |
| 3c | `walls.py` | The wall's position is re-measured on the inliers between 0.8 and 1.6 m; existence and status still use 0.3–2.0 m. | X-3: the ground-truth tape is held at ~1 m, above skirting and below coving. | `measure_h_lo_m` (None = off) |
| 3c | `walls.py` | Optional single-visit measurement: per room, the contiguous camera visit that measures most walls (`room_visits`, `choose_visit`, `raw_frame`). **Off.** | X-1, rejected by evidence (V2-I-2). | `single_visit` |
| 3d | `walls.py` | `canonicalize`: after refinement, remove shallow (≤ 10 cm) steps and bumps whose connecting or base edge has no raw-point support; merge measured parallel walls ≤ 2 cm apart; re-measure the merged wall. Undone for a room whose new outline would overlap a neighbour. | B-2: one jog in one capture changes the corners of a long wall (topology error). | `canonical_outline`, `canon_*` |
| 4 | `levels.py` | Two large ceiling layers 5–20 cm apart: value = dominant layer, status `inferred`, interval spans both. | B-6 (alpha A-14): the drift double layer was reported ±1.7 cm while 11 cm off (J-I-7). | `ceiling_double_layer` |
| 5 | `openings.py` | Jamb = the 1 cm layer nearest the opening that covers ≥ 40% and ≥ 80% of the best layer's 10 cm height bins (alpha's tall-face rule), mean-shifted onto the face; v1's slice rule is the fallback. | B-3: a door leaf edge or a handle is short, a jamb is tall. | `jamb_mode` |
| 6 | `walls.py`, `params.py` | σ_offset² = SE² + lidar² + drift² + inconsistency², drift = 1 cm within one visit / **2 cm across passes** (drift.md 5.5), inconsistency = √(rmse² − expected sensor rmse²) with the sensor rmse computed for our ±1.5 cm trimming. | B-4: the drift term was a placeholder; a smeared wall must get a wider interval. | `inconsistency_term`, `drift_mixed_sigma_m` |
| – | `extract.py` | `meta.coverage_warning` when the rooms cover < 4 m². | B-8: a degenerate input must not look like a normal result. | `coverage_min_m2` |
| – | `grid.py` | Optional world-lattice raster anchor. **Off.** | B-5 experiment; it exposed the raster-phase sensitivity (V2-I-5). | `grid_lattice` |

### The uncertainty model, v2

```
sigma_offset^2 = (rmse / sqrt(n_patches))^2       fit standard error (unchanged)
               + lidar_sigma^2                    5 mm per surface (unchanged)
               + drift^2                          1.0 cm if all points come from one visit (ICP floor 0.7-1.0 cm)
                                                  2.0 cm otherwise (held-out between-pass residual 1.9-2.5 cm RMS)
               + inconsistency^2                  max(rmse^2 - sensor_rmse^2, 0): scatter the sensor cannot explain
                                                  (two passes, a curtain); NOT divided by sqrt(n), it is systematic
               + (|tilt| * reach / 2)^2           unchanged (I-9)
ceiling: unchanged, plus the double-layer widening (B-6)
door width: alpha's jamb SE = 1.4826 * MAD / sqrt(n) per jamb, plus lidar_sigma, in quadrature
```

With the default `single_visit = False` every wall carries the 2 cm between-pass drift term: the honest value for
a wall built from points of several passes.

## v2.2 Decisions

| # | Decision | Options considered | Choice | Why | Evidence |
|---|---|---|---|---|---|
| V2-D1 | How long wall lines separate rooms | lower the free-space band (judge experiment); every long line as a barrier; paired faces only; **paired OR tall (≥ 0.8 m of 0.3–2.0 m)** | paired OR tall | Any-line cut at beds and counters (footprint −14%); paired-only missed the one-sided partition; the tall rule rejects every furniture class under ~1 m and only accepts tall furniture that v1 already treats as wall | ablation rows 1a, 1b, 1; `outputs/plan_v2/sep_lines_floor_only.txt` |
| V2-D2 | Where the cut acts | in the free-space map; **segmentation only** | segmentation only | A cut that removes free space shrinks rooms (J-I-5). Used only for the DT/watershed and then handed back, it can only separate | footprint 62.6 / 61.9 → 62.7 / 61.9 |
| V2-D3 | Single-visit measurement (X-1) | mix all passes (v1); per-room visit; per-wall visit | **mix (off)** | Implemented per room; the evidence went the other way (V2-I-2). Per-wall would lose the one argument for it (locally consistent corners) | rows 2, 2b, 3 |
| V2-D4 | Noise weights (X-2) | unweighted; **1/σ(range)²** | weighted | Correct estimator under range-dependent noise; small, not significant gain | row 3 |
| V2-D5 | Tape-height band (X-3) | full band; band for everything; **band for the position only** | position only | The band for everything drops walls and a room (recall −4%); position-only keeps the plan and measures where the tape does | rows 5a, 5b, 5 |
| V2-D6 | Drift term in the interval (B-4) | 1 cm placeholder; per-capture value read from the scene's drift report; **1 / 2 cm from the drift module's measured held-out residuals, by single / mixed visit** | 1 / 2 cm | The extractor's input is `(scene, info)`; the scenes_v2 drift report carries in-sample loop residuals, not the held-out `drift_sigma`, so the drift module's recommended values are parameters with their source cited | drift.md 5.2, 5.5 |
| V2-D7 | Canonical outline (B-2) | bigger raster jog snap (judge: worse); **after refinement, only unsupported shallow steps/bumps + coplanar merge** | after refinement | Positions are measured to mm after refinement; unsupported edges are not walls. Bigger snapping moved corners differently per capture | row 8 (neutral on the gate, better median and coverage) |
| V2-D8 | Door jambs (B-3) | v1 per-slice nearest solid bin; **alpha's tall-face rule, v1 fallback** | alpha first | Short objects near the opening (leaf edge, handle) are rejected by the relative coverage rule | row 6: within 2 cm 1/7 → 2/7, median 5.0 → 2.4 cm |
| V2-D9 | Ceiling double layer (B-6) | dominant layer only (v1); **alpha's A-14** | A-14 | Drift between passes makes two layers; we cannot tell a drift double from a real step, so we say so | row 7: drift-off bedroom interval now covers 3.065 m |
| V2-D10 | Raster anchor (B-5) | data minimum (v1); **world lattice: tested, off** | v1 anchor | The lattice made rot90 exact but the repeat pair worse (8/89) and floor_only footprint −7%: phase sensitivity is the real problem, the anchor only moves it | row 10, V2-I-5 |

## v2.3 Issues log

**V2-I-1. The first B-1 versions either cut rooms at furniture or cut nothing.**
- Symptom: any long line as a cut: 9/10 rooms, with_ceiling footprint 62.6 → 54.1 m²; paired faces only: no
  change at all (7/6 rooms).
- Root cause: the partition that floor_only missed was seen from **one side only** (2.30 m line, heights
  0.17–1.29 m); the furniture lines (bed sides, sofa backs, counters) are also one-sided but stop below ~1 m
  (`outputs/plan_v2/sep_lines_floor_only.txt`: e.g. a 0.82 m line at u = 2.99 with heights 0.49–0.75 m).
- Fix options: a higher support threshold (beds are 2 m long: no); require an anchor to wall evidence at both ends
  (a partition ends at a door: no); **vertical extent**.
- Chosen: paired OR ≥ 0.8 m of vertical extent in 0.3–2.0 m. Result: 8 / 8 rooms, all matched, footprint
  unchanged.

**V2-I-2. The single-visit idea (X-1) made repeatability worse.**
- Symptom: same-topology median 1.9 → 3.2 cm, pass 15 → 9 when on (with or without noise weights).
- Root cause: (1) two passes with independent drift errors average to an error about √2 smaller; one visit keeps
  its full drift error; (2) one visit sees fewer 10 cm patches, so the fit is noisier; (3) the chosen visit can
  differ between captures.
- Chosen: off by default; the tools stay for a per-pass diagnostic (splitting with_ceiling into its passes is a
  within-capture repeatability check that needs no second capture).

**V2-I-3. A narrower measurement band dropped a room.**
- Symptom: band 0.8–1.6 m: 7/7 rooms instead of 8/8, recall 0.953 → 0.911; the median |Δ| fell to 4.4 cm, but
  only because 61 walls were judged instead of 95.
- Root cause: the band also decides whether a wall is measured (≥ 200 points over ≥ 10 patches). Fewer points →
  inferred walls → an unentered room fails D-B10's 60% enclosure and is dropped.
- Chosen: use the band for the position only (V2-D5).

**V2-I-4. Removing a bump made two rooms overlap (0.13 m²).**
- Root cause: canonicalization runs after the overlap resolution; removing a bump moved a wall across a shared
  boundary.
- Fix: per room, undo the canonicalization if the new outline overlaps a neighbour (`_canonical_outlines`).
  Result: 0.0 m² on all captures.

**V2-I-5. The extractor is raster-phase sensitive; v1's anchor hid it.**
- Symptom: with the raster on a world lattice, rot90 becomes almost exact (60/70 → 69/70 walls) but a 1.3 cm
  shift keeps only 46/78 and 43/82 walls (59%, 52%).
- Root cause: v1 anchors the raster at the data's minimum point, which moves with a rigid shift, so the shift
  test was only a floating-point test (judge J-I-2). Pinned to a lattice, the shift really changes where each wall
  falls inside its 2 cm cell, and the outline steps (arrangement coverage at 50%, mask lines from raster borders,
  jog and bump rules) follow it.
- Consequence: two captures always differ in phase, so part of the cross-capture topology disagreement is the
  extractor's own noise. Not fixed in this round; it is the Part 4 candidate (`plan_v2_ablation.md`).

**V2-I-6. single_room lost 1.8 m² of footprint.**
- Symptom: 19.41 → 17.61 m², floor recall 0.603 → 0.591, a new 1.9 m² room (an entered alcove).
- Root cause: B-1's cuts change the DT near walls, so the watershed splits the alcove off R1; R1's arrangement no
  longer includes a half-observed strip behind a wardrobe line. The observed floor lost is 1.2 points of recall:
  most of the 1.8 m² is unobserved floor v1 included by rectangle filling.
- Not fixed: without ground truth I cannot say which outline is right. Open problem.

**V2-I-7. thin_scene predates raw_frame.**
- `judge_common.thin_scene` thins raw points but not `raw_frame`. `plan_v2_eval.thin` replays the same random
  draws and applies them to `raw_frame`, so a thinned scene stays consistent. The judge's file is unchanged.

## v2.4 Results (final v2, scenes_v2, drift ON)

| | v1 (frozen, same scenes) | v2 |
|---|---|---|
| Rooms with_ceiling / floor_only / matched | 7 / 6 / 6 | 8 / 8 / **8** |
| Walls passing (Wilson 95%) | 7 / 99 = 7.1% [3.5, 13.9] | **13 / 87 = 14.9%** [8.9, 23.9] |
| Unmatched / different-topology walls | 54 / 28 (median 51 cm) | 36 / 25 (median 32 cm) |
| Same-topology n / pass / median | 17 / 7 / 1.9 cm | 26 / 12 / **1.7 cm** |
| Median \|Δ\| | 15.0 cm | **8.2 cm** |
| Interval coverage, all / same-topology | 0.42 / 0.94 | 0.63 / **0.96** |
| Footprint with_ceiling / floor_only | 62.64 / 61.93 m² | 62.65 / 61.90 m² |
| Floor recall with_ceiling / floor_only | 0.953 / 0.939 | 0.952 / 0.940 |
| Matched doors within 2 cm (cross-capture width) | 0 / 5 (deltas 3.1–12.8 cm) | **2 / 7** (+2 at 2.2, 2.4 cm) |
| Door widths measured on jambs (with_ceiling / floor_only) | 4 / 2 | **8 / 5** |
| Ceilings, with_ceiling, drift ON | all `measured` | all `measured`, unchanged (≤ 0.1 cm) |
| Bedroom ceiling, drift OFF | 2.958 m `measured` [2.941, 2.975] | 2.958 m `inferred` [2.941, **3.068**] |
| Runtime single_room / floor_only / with_ceiling | 3.9 / 18.4 / 29.7 s | 3.9 / 19.6 / 34.9 s |

Ceiling heights (with_ceiling, drift ON, v2): 2.467, 3.085, 3.065, 3.078, 2.359, 2.380, 2.466, 2.471 m. floor_only
saw no ceiling, so the cross-capture ceiling spread cannot be measured on this pair (status `not_observed`, never
invented).

**What the numbers say, plainly.** The room merge is fixed and the gain is not bought by shrinking the plan. The
gate (every wall within 1 cm or 0.5%) is still far away. Even if every wall had the same corners in both captures,
only about half would pass (12 / 26 same-topology walls), because those walls differ by 1.7 cm median and the two
drift-corrected captures are themselves misaligned by 2.0 cm median (registration tiles). The extractor cannot
remove that; the drift module can.

## v2.5 Limitations and open problems

- **Raster-phase sensitivity** (V2-I-5): the biggest extractor-side limiter now. Next fix: phase-averaged outlines.
- **Drift floor:** same-topology walls agree to 1.7 cm, the same as the captures' residual misalignment.
- **Doors:** 3 of 7 matched doors still differ by 9.6–15 cm: two are corridor constrictions with no jambs (width
  falls back to the 2 cm DT neck, status `inferred`), one is a door whose two captures picked different faces.
- **single_room** footprint −9% (V2-I-6), cause understood, correctness unknown without ground truth.
- **Not done:** B-7 glass flag, B-10 partial rooms, the CLI exit code for B-8, ARKitScenes check of X-3 (the laser
  validation measures planes, not this extractor's output).
- **X-4** has no measured effect: on the photo scene available every region voted a different folder.

## Requested changes to shared code (v2)

1. **Drift module:** write `drift_sigma` (held-out between-pass residual, and the within-visit ICP floor) into the
   scene's `scene_info.json`, so `BetaParams.drift_sigma_m` / `drift_mixed_sigma_m` can be read per capture
   instead of being the module's recommended constants.
2. **`scripts/judge_common.py`:** `thin_scene` should also thin `raw_frame` (and any per-raw-point array).
3. **`scripts/run_plan_beta.py`:** exit non-zero when `plan.meta.coverage_warning` is set (B-8).

---

# Part v3: stability (run-to-run flips and raster phase)

Added 2026-10-04 by the stability task. Parts 1–v2 are unchanged. Evidence: `outputs/stability/`
(`plan_eval/` = the plan_v2 harness re-pointed with `PLAN_V2_OUT`, `flip/` = the run-to-run test,
`debug_cells*.log` = the root-cause trace). The drift half of the same problem is in `docs/modules/drift.md`, Part S.

## v3.0 Problem

- The same one-command run on floor_only, three times: footprint 61.90 / **59.37** / 61.90 m², the largest room
  13.99 vs 11.46 m², while the drift-corrected poses differed by ~0.06 mm (`benchmark_lidar_sample.md` §4.1).
  with_ceiling twice: 70 vs 74 walls.
- `plan_v2_ablation.md` B-5: pinning the raster phase and shifting the data by 1.3 cm changed 41–48% of walls.
  Its proposed fix: build each outline at several sub-cell raster offsets and keep the majority.

## v3.1 Root cause of the 2.5 m² flip (with evidence)

`outputs/stability/debug_flip.py` and `debug_cells.py` re-extract the benchmark's two saved floor_only scenes
(`drift_on`, `drift_on_rerun_nodamage`) and log every arrangement cell whose room-mask coverage is near the 50%
inclusion rule (`cell_cover_min`):

| Scene | Cell u 4.97–7.25, v 6.28–7.38 (2.51 m²) coverage | Included (≥ 0.5)? | Room R1 | Footprint |
|---|---|---|---|---|
| drift_on | **0.535** | yes | 13.89 m² | 61.90 m² |
| drift_on_rerun_nodamage | **0.457** | no | 11.35 m² | 59.37 m² |

- The point (6.0, 6.8) is not free space in either scene (label 0): the cell is floor the LiDAR did not see
  (under a bed or similar), with the bedroom's own wall on two sides and the room on the other two.
- A whole 2.5 m² cell is in or out on a single threshold, so the output jumps by 2.5 m² when the coverage moves
  by 0.08. The plan is not a continuous function of its input at that point. with_ceiling, which saw more of that
  floor, includes the cell (R1 14.21–14.31 m²).
- The same cell sits at 0.37–0.66 coverage across four sub-cell raster phases of one scene, which is the B-5
  phase sensitivity seen from a different angle.

## v3.2 Options

| # | Option | Result | Verdict |
|---|---|---|---|
| S-1 | Make the drift solve deterministic (drift.md Part S) | identical scenes from identical input | **kept**; it hides the symptom only, since a real second capture always differs by more than 1e-15 |
| S-2 | **Phase vote** (B-5's proposal): build the plan at k×k sub-cell raster phases (`phase_votes = 4`: 0 / ½ cell in u and v) in forked workers and return the medoid build (highest mean wall-level Dice agreement with the other builds; first version used raw counts) | see v3.4: the medoid is the **shrunken** build (floor_only footprint 61.9 → 59.6 m², R1 13.9 → 12.5 m²), the repeat-pair gate falls 13/87 → 11/94, and the real run-to-run flip is still there (59.61 vs 57.56 m², now in R5) | **rejected**; kept as the switch `phase_votes` (default 1) and as a diagnostic (`meta.phase_vote`) |
| S-3 | **Enclosed-cell rule**: also include a partly covered cell (≥ 25%) if every side either opens onto an included cell (< 50% of the side has wall-band surface, 1.0–2.0 m) or lies on one of the room's measured wall lines with wall-band surface facing into the cell over ≥ 30% of it | the flip disappears: all three floor_only scenes give the same plan; repeat pair unchanged; no footprint change elsewhere | **kept** |
| – | Lower `cell_cover_min` globally | moves the knife edge somewhere else, and pulls in space beyond walls | not tried (no principle) |
| – | Subdivide ambiguous cells along the mask edge | turns the bed edge into fake walls | not tried |

Why S-3 is principled and not a tuned threshold: it is the space-first rule stated explicitly. Floor hidden under
furniture inside a room's walls belongs to the room. The rule never crosses a wall: inner sides must be open. It
never goes beyond a wall: outer sides must be the room's own measured lines. And it never adds a cell the room's
free space does not reach: coverage must be ≥ 25%. Honest caveat: the two side thresholds (30% / 50%) were set
while looking at the one cell that caused the flip. The first version, at 50% / 30%, did not include it, because
its outer wall is only 40% visible in the wall band and its inner side is 40% occupied by a tall object. The
regression checks below show they change nothing else on the three captures.

## v3.3 What changed in code

- `walls.py`: `arrangement_polygon(..., vertical)` keeps the per-cell coverage matrix and calls
  `_add_enclosed_cells` (S-3); `_side_support` measures how much of a cell side has wall-band surface;
  `vertical_points(..., band)`; `room_polygon(..., tall)`.
- `extract.py`: `_build_rooms` computes the wall-band points once (`tall`); `extract_plan_debug` dispatches to the
  phase vote when `phase_votes > 1` (`_phase_vote`, `vote_phases`, `wall_agreement`, forked workers); the old body
  is `_extract_single`.
- `grid.py`: `Grid.around(..., phase=(du, dv))` (sub-cell offset in cells).
- `params.py`: `enclosed_cells = True`, `enclosed_min_cover = 0.25`, `enclosed_wall_min = 0.3`,
  `enclosed_open_max = 0.5`; `grid_phase = (0, 0)`, `phase_votes = 1`, `phase_workers = 4`,
  `phase_agree_tol_m = 0.01`.
- `scripts/plan_v2_eval.py`: `PLAN_V2_OUT` / `PLAN_V2_SCENES` environment overrides, and a fourth invariance test,
  **phase** (`<tag>_phase` run with `--set grid_phase=0.25,0.25`). The data-anchored `shift` test cannot see the
  phase, because the raster corner moves with the data.

## v3.4 Results (before → after); every number from runs in `outputs/stability/`

**Run-to-run flip test** (`outputs/stability/flip_test.py`): the benchmark's saved scenes of the same command,
re-extracted, scored pairwise with the judge's `compare` in their shared frame (`flip/scores.json`).

| | v2 (before) | S-2 vote4 (v2 + vote) | S-3 + vote4 | **S-3 (after)** |
|---|---|---|---|---|
| floor_only footprints, 3 runs (m²) | 61.90 / **59.37** / 61.90 | 59.58 / **57.67** / 59.58 | 59.61 / **57.56** / 59.61 | **61.90 / 61.90 / 61.90** |
| floor_only walls, 3 runs | 70 / 72 / 70 | 80 / 80 / 80 | 78 / 78 / 78 | **70 / 70 / 70** |
| floor_only run 1 vs 2, walls passing | 68 / 72 | 71 / 83 | – | **70 / 70** |
| with_ceiling walls, 2 runs | 70 / 74 | 70 / 74 | 70 / – | 70 / 74 (69 / 74 pass; R5 7.59 vs 7.66 m², a line-position change, not a cell) |

**Repeat pair and invariance** (`plan_eval/scores.json`; A = with_ceiling, B = floor_only, scenes_v2, drift ON):

| Tag | Pass [Wilson 95%] | Rooms m. | Median \|Δ\| | ST n / pass / med | Footprint A / B m² | Recall A / B | Cov all / ST | shift A / B | rot90 A / B | thin90 A / B | phase A / B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `s0` v2 as found | 13/87 = 14.9% [8.9, 23.9] | 8 | 8.2 cm | 26 / 12 / 1.7 cm | 62.7 / 61.9 | 0.952 / 0.940 | 0.63 / 0.96 | 61/71, 70/70 | 60/70, 37/76 (rooms 8 → 7) | 51/71, 61/74 | 52/72, 44/83 (`s3x`) |
| `s4` vote4 (count score) | 11/95 = 11.6% [6.6, 19.6] | 8 | 6.1 cm | 29 / 11 / 1.8 cm | 62.7 / **59.6** | 0.952 / 0.944 | 0.64 / 0.97 | 63/73, 77/81 | 64/73, 55/83 | 54/74, 65/87 | – |
| `s5` S-3 + vote4 (Dice score) | 11/94 = 11.7% [6.7, 19.8] | 8 | 7.4 cm | 27 / 11 / 1.9 cm | 62.7 / **59.6** | 0.952 / 0.944 | 0.59 / 0.96 | 63/73, 78/78 | 66/70, 58/79 (8 rooms) | 54/74, 60/80 | 57/72, 46/87 |
| **`s3` S-3 (final)** | **13/87 = 14.9% [8.9, 23.9]** | 8 | 8.2 cm | 26 / 12 / 1.7 cm | 62.7 / 61.9 | 0.952 / 0.940 | 0.63 / 0.96 | 61/71, 70/70 | 60/70, 37/76 | 51/71, **63/72** | 52/72, **46/81** |

single_room (`plan_eval/runs/{s3x,s3}/single_room__*`): 17.61 m², rooms 8.23 / 7.49 / 1.89 m², identical before
and after.

**Reading.**

- S-3 removes the observed run-to-run flip: 3 of 3 identical plans, against 2 of 3 before. On scenes_v2 it is
  neutral on the gate (it fires on no cell there), slightly better on thin90 and phase for floor_only, and it never
  shrinks a plan.
- The vote **improves the self-invariance tests**: rot90 B 37/76 → 58/79 with no room lost, phase A 52 → 57. But
  it **fails the decisive evidence**: the repeat pair drops by 2 walls, floor_only's footprint shrinks by 2.3 m²
  (with_ceiling: 62.7 m²), and the real flip moves to another room. A repeatability gain that shrinks the plan is
  the J-I-5 red flag, so the vote stays off.
- Why the vote cannot work here: the four phase builds agree with each other on only **0.59–0.73** of their walls
  (mean Dice, `meta.phase_vote.score`). With four noisy samples and several independent near-threshold decisions,
  the medoid build is itself a coin toss per room. The phase sensitivity is real and large (phase test 44–52 of
  72–83 walls), and it is not fixed by this round.

**End-to-end** (`scripts/run_capture.py <capture> --tier lidar --no-damage`, twice per capture, final code, the
four runs in parallel; `outputs/stability/e2e/`, comparison `e2e_compare.py` → `e2e/compare.json`):

| | Before (benchmark runs) | After |
|---|---|---|
| floor_only: footprint / rooms / walls | 61.90 / **59.37** / 61.90 m²; 70 / 72 / 70 walls | **61.9009 / 61.9009 m²; 8 / 8 rooms; 70 / 70 walls; plan.json identical** (timing fields aside) |
| with_ceiling: footprint / walls | 62.50 / 62.56 m²; 70 / **74** walls | **62.4985 / 62.4985 m²; 70 / 70 walls; plan.json identical** |
| Drift poses `T_wc` | differ by ~0.06 mm | **bit-identical** (also every loop edge in the report) |

Residual nondeterminism, outside this task's files: the TSDF surface points (`floorplan/recon/fusion.py`, Open3D
`VoxelBlockGrid`, multi-threaded) come out in a different **order** each run. On floor_only the sorted sets are
identical. On with_ceiling, `T_align` then differs by 6e-10, because the alignment sums over the points in that
order, and the raw points by one float32 ulp (1e-6 m). The plan is unaffected. Proposed fix: sort the extracted
TSDF points lexicographically (or by voxel key) in `fuse_tsdf` before returning them.

## v3.5 Limitations and next step

- Raster-phase sensitivity remains: a quarter-cell move changes 20–40% of walls (phase test). On floor_only, most
  of the change is in segmentation, with 22 walls unmatched. The remaining sources are line positions from 1 cm
  histogram peaks (the with_ceiling R5 change), mask lines at cell boundaries, and watershed boundaries between
  rooms (R2/R3, R7/R8 change with phase). Each yes/no decision needs the same treatment S-3 gave the cell rule: a
  rule whose output moves continuously with the evidence, or a decision taken on physical evidence (walls)
  instead of a coverage fraction.
- Openings are sensitive too. On with_ceiling, the deterministic scene from the e2e runs gives 12 openings, and the
  benchmark's scene of the same command gave 13. S-3 does not change openings: 13 = 13 on the benchmark scene
  (`flip/{v1phase,s3}`). This is the window/door detector reacting to the sub-mm scene difference, and it is not
  investigated here.
- The per-wall phase support (`meta.phase_vote.wall_support` with `phase_votes = 4`) is a measured
  topology-uncertainty flag. Intervals cannot express a topology error. The next step is to use it to mark
  unstable walls in the output, not to pick a build.

---

# Part v4: tiers (video and photo scenes)

Added 2026-10-04 by the tier-aware extraction task. Parts 1–v3 are unchanged, and the LiDAR path is bit-identical
(v4.5). Evidence: `outputs/runs/plan_v4_video_single_room/` and `outputs/runs/plan_v4_photo_with_ceiling_rotate/`
(end-to-end `run_capture.py` runs: plan.json, run_report.json, stdout.log, time.log). The experiment and scoring
scripts were scratch scripts (not committed). They load a saved scene, call `extract_plan(..., tier=...)` and compare
with the LiDAR plan of the same capture.

## v4.0 Problem (plain language)

plan_beta was tuned on clean LiDAR scenes. The video and photo tiers write the same scene format (D-003), but their
surfaces come from learned depth.

- **Video, single_room:** 1 room of 23.3 m² against 17.6 m² and 3 rooms from LiDAR. The 95% interval
  [18.3, 28.4] did not contain the LiDAR value (video_tier.md V2-I9). The scale error was only +2.8%, so this is the
  extractor, not scale.
- **Photo:** the extractor ignored the room folders. Its rooms could not be matched to the operator's rooms (one folder per room). It merged or
  split them freely, and it reported adjacency from rooms that touch only because the photo front-end placed an
  unlinked room "beside the block" (a guess, photo_tier.md P-6).

## v4.1 Root cause (with evidence)

**Video: the TSDF surface lost most of the walls.** In the cached single_room video scene, the 1.0–2.0 m wall band
of `scene["points"]` (the TSDF surface, weight ≥ 5, video_tier.md V-9) shows only fragments of the walls. The R1/R2
partition and most of the outer walls are missing, while `raw_points` show every wall. These are the same depth maps
before fusion, about 1.3 M points. A wall seen by only a few keyframes never reaches the TSDF weight threshold. With
no wall evidence:

- free space leaked through every partition,
- the watershed found one basin,
- the room polygon used whatever lines were left.

Per-point noise is not the cause. The heights of vertical points match LiDAR's (same band histogram), and the
C2C of raw points vs TSDF is about the same, 8.1 vs 7.6 cm (video_tier.md §2).

**Photo:** the segmentation had no input for "which room is this". Folders were used only to *merge* touching
regions (X-4). The adjacency rule (door neck or shared wall face) cannot tell a measured neighbour from a fallback
placement.

## v4.2 How it works now (step table, v4 additions only)

| Step | What | Tier | Why |
|---|---|---|---|
| 0 (T-1) | `tiers.augment_surface`: raw points averaged into 3 cm voxels (voxels with ≥ 2 points), a PCA normal from 24 neighbours, oriented to face the camera along the stored ray, planar voxels only (smallest-eigenvalue share < 5%), added to `points/normals` | video | gives back the wall evidence and wall lines the TSDF dropped |
| 2' (T-3) | `tiers.folder_segmentation`: one watershed marker per folder. The marker is a disc of 30 cm (operator radius) around the folder's cameras whose distance to the nearest obstacle is ≥ 50% of the folder's best camera. Cameras outside free space snap ≤ 0.5 m. No merging across folders; free space no folder reaches stays out | photo | the folder is the operator's own statement of a room; spin photos stand mid-room, doorway photos stand in the door and would grow into the wrong room |
| 3e (T-2) | `tiers.add_geom_sigma`: every wall offset σ ← √(σ² + 5 cm²) | video, photo | surface noise of learned depth that the fit terms do not see |
| 3e (T-5) | `inferred_sigma_m` 5 cm → 50 cm | photo | an unmeasured photo wall is only the edge of a few view wedges, not a free-space edge next to a wall |
| 6 (T-3) | room `label` = folder name. Each room's (and its walls') interval half-widths × `info.rooms[folder].widen` (linked 2, door-match 3.5, fallback 4). Footprint × the area-weighted mean factor. Lower bounds clipped at 0 | photo | the front-end's placement confidence carried into the plan |
| 6 (T-3) | adjacency only for folder pairs the photo front-end linked (`info.room_links_v2`, else `info.links`). `via` = the opening found between them, else `shared_wall`. `meta.tier_v4.adjacency_evidence` records `measured` (both rooms placed by links) or `inferred` (either room a fallback), plus every touching-but-unlinked pair as dropped | photo | touching after a fallback placement is not evidence of a door |
| CLI | `run_capture.py` passes `tier=--tier` to plan_beta. After the tier scale term it clips every size interval's lower bound at 0 (`intervals_clipped_at_zero` in run_report) | all | sizes are ≥ 0; clipping never removes the true value |

The parameters are chosen by `tiers.params_for_tier(tier)`:

| Field (BetaParams) | LiDAR | video | photo |
|---|---|---|---|
| `surface_from_raw` | False | **True** | False (no measurable effect, v4.4) |
| `geom_sigma_m` | 0 | **0.05** | **0.05** |
| `inferred_sigma_m` | 0.05 | 0.05 | **0.5** |
| `use_folder_seeds`, `use_folder_widen` | False | False | **True** |
| `use_room_hints` (X-4 merge) | True | True | False (replaced by seeds) |
| `min_room_m2` | 1.2 | 1.2 | **0.8** |
| `unvisited_min_enclosure` | 0.6 | 0.6 | **0** (a folder room is never "seen through a door") |
| `folder_hull_fill` | False | False | False (tested, rejected, v4.3) |

`extract_plan(scene, info, id, params=None, tier=None)`: the tier comes from the argument, else `info["tier"]`, else
"lidar". Explicit `params` are used as given, so ablation scripts are unaffected. `Plan.tier` and
`meta.tier_v4` record the tier.

## v4.3 Decisions (options, evidence)

Video. All rows are on the cached scene from the earlier integration run (`outputs/runs/video_v2_single_room/scene`,
"run A"). The LiDAR reference is `outputs/benchmark/lidar/single_room/drift_on/plan.json`: 3 rooms (R1 3.04 × 2.76,
R2 3.03 × 2.53, R3 1.90 × 1.06), footprint 17.61 m², adjacency R1-R2, R1-R3, R2-R3.

| Option | Rooms | Footprint m² | R1 / R2 (L × W) | Verdict |
|---|---|---|---|---|
| v3 (LiDAR params) | 1 | 23.34 | 6.74 × 6.17 | baseline |
| **T-1 raw surface** (3 cm, k 24, curvature < 0.05, ≥ 2 pts) | **3** | **17.54** | 3.40 × 3.06 / 2.95 × 2.47 | **kept** |
| T-1 with curvature < 0.08 | 3 | 15.88 | 3.39 × 3.08 / **2.53 × 2.25** | rejected (R2 −16%) |
| T-1 with 2 cm voxels, k 30 | 3 | 18.67 | 3.44 × 3.12 / 2.97 × 2.50 | rejected (more walls, larger R1) |
| T-1 with curvature < 0.08, ≥ 1 pt | 3 | 22.46 | R3 grows to 6.1 m² | rejected (noise lets the corridor leak) |
| + `carve_stop_m` 0.15 | 3 | 17.24 | 3.35 × 3.08 / 2.95 × 2.47 | not adopted (−5 cm on one wall, no principle-level gain) |
| + `carve_min_hits` 4, `line_merge_m` 0.10, `wall_dilate_m` 0.08 | 3 | 17.42 / 17.32 / 17.68 | within ±0.06 m of T-1 | not adopted (no effect) |

On the second video scene (`outputs/video_tier/single_room__c00a170fe1__v2`, "run B", not used for tuning) the
same choice gives 3 rooms and 16.85 m², against 2 rooms and 24.35 m² before. The 3 cm / curvature 0.05 setting is
the only one that is good on both.

T-2 (5 cm surface-noise σ). After removing run B's measured scale error (+2.2%), its four room dimensions differ
from LiDAR by −10.7, −2.0, 0.0 and −9.6 cm. That is an RMS of 7.3 cm per dimension, about 5.2 cm per wall
(÷√2). The fit σ alone was 1–2 cm, which would be confident garbage.

Photo. Four cached photo scenes, scored against the LiDAR plan the photos were simulated from
(`outputs/plan_beta/<capture>/plan.json`; the folders R1..R7 are its room ids):

| Option | Effect | Verdict |
|---|---|---|
| T-3 folder seeds (vs v3) | rooms named and matchable, adjacency 0 of 1 correct → 4 of 4 correct over the 4 scenes, no overlaps | **kept** |
| seed only from cameras with DT ≥ 50% of the folder's best (vs all cameras) | floor_only/rotate 5 rooms and 2/2 correct links vs 4 rooms and 1/1 | **kept** |
| `min_room_m2` 0.8 and no unvisited drop (vs LiDAR 1.2 / 0.6) | keeps 5 vs 4 rooms (with_ceiling/rotate), 2 vs 1 (with_ceiling/corners) | **kept** (a folder is a room the operator declared) |
| T-4 convex-hull fill per folder | with_ceiling/rotate footprint 16.5 → 44.9 m² (LiDAR 62.4), median room-area error 80% → 47%; but floor_only/rotate 28.8% → 68% **and a 2.1 m² overlap** between R2 and R3 | **rejected**: an overlap breaks a gate; kept as `folder_hull_fill` (off) |
| T-1 raw surface for photo | median area error 77.9 → 79.7% / 28.8 → 28.8%, no topology change | not adopted (no evidence of gain) |
| T-5 `inferred_sigma_m` 0.5 m | room-area interval coverage 4/13 → 9/13 rooms over the 4 scenes; values unchanged (same scene, same geometry) | **kept** (honesty; see limitations) |

## v4.4 Results (every number from runs in this session)

**Video, single_room, three front-end runs** (sorted dimensions vs LiDAR R1 3.04 × 2.76, R2 3.03 × 2.53; LiDAR
footprint 17.61 m² [17.07, 17.91]):

| Front-end run | Extractor | Rooms | Adjacency (LiDAR 3 pairs) | Footprint m² | R1 L / W error | R2 L / W error |
|---|---|---|---|---|---|---|
| A (cached) | v3 | 1 | 0 | 23.34 | – (one merged room) | – |
| A (cached) | **v4** | 3 | 3 (R1-R2, R1-R3, R2-R3) | 17.54 (−0.4%) | +11.8% / +10.9% | −2.6% / −2.5% |
| B (cached `__v2`) | v3 | 2 | 1 | 24.35 | – (R1 6.86 × 3.34 merged) | +3.2% / +10.9% |
| B (cached `__v2`) | **v4** | 3 | 3 | 16.85 (−4.3%) | −1.3% / +1.6% | +2.2% / −1.6% |
| C (end-to-end, this session) | v3 (same scene) | 2 | 1 | 24.04 | – (R1 6.84 × 3.23 merged) | −1.9% / +8.5% |
| C (end-to-end) | **v4** | 2 (R3 dropped: no valid polygon) | 1 of 3 | **15.83 [11.81, 19.86]** (−10.1%, covered) | +1.3% / +9.4% | +0.2% / −2.1% |

- v4 room dimensions within the 3% video gate: **9 of 12**. v3: 1 of the 4 comparable dimensions, since R1 is
  merged in every v3 run. Raw error, the scale error not removed.
- End-to-end run C, with `run_capture`'s tier scale term (6.4%): all 4 dimensions, both areas and the footprint lie
  inside their 95% intervals (R1 W 3.02 [2.60, 3.43] vs 2.76).
- Before v4, the area interval [18.3, 28.4] missed 17.4 m².
- Runtime: 270 s end to end (4:32 wall, 5.9 GB RSS; front-end ~240 s). Plan extraction 3.4 s, of which 2.7 s is
  free space including the raw surface.
- Other video scenes. floor_only v2: 8 rooms and 49.0 m² (v3: 4 rooms, 50.6 m²; LiDAR: 8 rooms, 61.9 m²; the
  video drops one untrusted segment, so part of the flat is missing). with_ceiling v2 has no floor (`floor_y`
  None), so both v3 and v4 stop with a TypeError. That is front-end Issue 15, and no plan can be made.

**Photo, four cached scenes** (v4 vs LiDAR, run with `extract_plan(..., tier="photo")`; v3 rooms cannot be matched to
folders):

| Scene | v3 rooms / correct adjacency | v4 rooms (folders) | v4 adjacency correct / reported (LiDAR pairs) | Overlaps | Footprint v4 vs LiDAR | Median room-area / dimension error | Area interval covers |
|---|---|---|---|---|---|---|---|
| with_ceiling / rotate | 3 / 0 of 0 | 5 of 7 | **2 / 2** (9) | 0 | 17.9 vs 62.4 | 77.9% / 53.8% | 1/5 |
| floor_only / rotate | 4 / 0 of 1 | 5 of 6 | **2 / 2** (7) | 0 | 37.5 vs 55.0 | 28.8% / 33.3% | 5/5 |
| with_ceiling / corners | 2 / – | 2 of 7 | 0 / 0 | 0 | 17.3 vs 62.4 | 35.4% / 33.0% | 2/2 |
| floor_only / corners | 1 / – | 1 of 7 | 0 / 0 | 0 | 0.9 vs 55.0 | 74.9% / 48.0% | 1/1 |

**Photo end to end** (`run_capture.py outputs/photo_inputs/single_scan_with_ceiling__c7d28f72c6__rotate/photos
--tier photo`): 5 rooms named R1, R4, R2, R6, R5, no overlaps, footprint 18.56 m² [3.0, 34.1] vs 62.39. Schema
clean. Runtime 37.6 s (39 s wall, 1.9 GB; depth and gravity came from the photo cache). In this run the front-end
did not place R7's linked block with the others, so all 5 links point to folders without a room and **adjacency is
empty**. One touching pair (R2–R4, no link) was dropped. Two earlier end-to-end runs on the same folder this session
gave 5 rooms with 2/2 correct links, but R7 was 7.92 m² in one and 2.14 m² in the other. The extractor is
deterministic on a fixed scene (checked twice on video and photo scenes), so this run-to-run spread comes from the
photo front-end.

## v4.5 LiDAR unchanged

- `extract_plan` (default call, as `run_capture` makes it) on the three benchmark scenes
  (`outputs/benchmark/lidar/<capture>/drift_on/scene`) before and after: plan JSON **identical** apart from the new
  `meta.tier_v4 = {tier: lidar}`. single_room 3 rooms / 17.61 m², floor_only 8 / 61.90, with_ceiling 8 / 62.50.
- The LiDAR overrides are empty and every new code path is behind a parameter that is off by default.
- The only LiDAR output change is in `run_capture`: lower bounds below 0 are clipped. The values are unchanged.
  Five intervals in the existing benchmark plans have a negative lower bound: single_room has 1 opening width (0.010
  m [−0.015, 0.035]), floor_only has 4 wall thicknesses ([−0.018, 0.134], [−0.012, 0.151], each twice).
- `tests/test_stability.py` and `tests/test_export.py`: 19 passed. The S-3 rule is untouched.

## v4.6 Issues log

| # | Issue | Root cause | Status |
|---|---|---|---|
| T-I1 | Video plan 1 room, 23.3 m² (V2-I9) | TSDF weight threshold removes walls seen by few keyframes; extractor only read the TSDF | fixed by T-1 (3 rooms / 17.5 m² and 16.9 m² on two scenes) |
| T-I2 | Video interval did not cover LiDAR | only scale + fit terms; learned-depth surface noise missing | T-2 (5 cm, from run B residuals); end-to-end run C covered on every quantity |
| T-I3 | Video R3 (1.9 × 1.06 m room) becomes a 4.0 × 0.34 m "corridor" strip (runs A, B) or is dropped (run C) | free space carved through the door into the unentered space beyond; the partition walls there are still weak | open: flagged by its label `corridor`; its interval does not cover LiDAR R3 |
| T-I4 | Video run A R1 +11–12% | a 0.4 m left-wall leak where neither TSDF nor raw surface has a wall in the band | open; run B/C R1 within 1.6% on one side, +9.4% on the other (run C) |
| T-I5 | Photo rooms are partial (median dimension error 33–54%) | 2–8 photos see the floor as view wedges; the room outline is where the wedges end | open. Intervals widened (T-5, folder widen); hull fill made it worse elsewhere (overlap) |
| T-I6 | Photo with_ceiling/rotate: hallway folder R7 grows into its neighbours (+257% area in one run) | neighbour folders without a placed seed (R3: 0 seeds, R4: 1 camera) leave their floor to the hallway's basin | open; area interval does not cover; v4 never claims an adjacency for it unless linked |
| T-I7 | Photo front-end gives different scenes on the same folder (R7 7.9 / 2.1 / not placed) | front-end non-determinism (photo_tier owner) | reported; the extractor is deterministic |
| T-I8 | Negative lower bounds (LiDAR opening/thickness; photo after widen ×4) | symmetric Gaussian intervals on sizes near 0 | clipped at 0 in `run_capture` |
| T-I9 | Video with_ceiling has no floor → TypeError in the extractor | front-end found no floor (video Issue 15) | open (should fail with a clear message) |

## v4.7 Limitations and next steps

1. **Photo room size is not measured yet.** Only adjacency, naming and no-overlap are right. The 8% gate fails on
   the simulated photos. Next step: fit a Manhattan rectangle per folder to that folder's own wall evidence (each
   room's walls are seen from its spin centre) instead of trusting where the view wedges end. Then use the doorway
   gap width to snap fallback rooms (photo_tier.md P-6a).
2. **Values tuned on the sample.** T-1's voxel and curvature, T-2's 5 cm and T-5's 0.5 m come from 1–3 video
   runs of one capture and 4 simulated photo scenes. Re-check them on the 08:00 own-home capture against the tape:
   - the per-room dimension errors after scale (T-2),
   - the coverage of the photo intervals (T-5).
3. **Video door leak (T-I3).** Use `kf_trusted` / camera positions to limit carving to rays from cameras inside
   the region (video carving is long-range), or cut free space at the door width as in Part 1's neck rule.
4. **`recommended_measurement_cloud = "points"`** (video_tier change request) is not implemented. Wall *positions*
   are still refined on raw points. On run B the dimensions agree within 1.6–2.2% after T-1, so it was not needed
   for the gate, but it is untested.

## v4.10 Photo v3: rooms from each folder's spin layout (added 04:00-05:00, 4 Oct)

**Problem.** v4 photo rooms were the right rooms, joined the right way, but much too small (T-I5). Example:
with_ceiling/rotate footprint 17.9 vs 62.4 m². 8 of 13 room-area intervals missed the LiDAR value.

**Change** (photo tier only; `info["room_layouts"]` from photo_tier.md v3; nothing new runs without it):

| Step | What | Where | Why |
|---|---|---|---|
| 6' (T-6) | After the v4 rooms are built, each folder with a usable layout gets a **rectangle**: its 4 wall offsets from its own spin (photo_tier v3), centred on the spin centre (placed spin photos), or on the v4 room centroid if no spin photo was placed (position flagged `inferred`) | `tiers.layout_plan`, `extract._layout_rooms` | sizes from walls the room's own photos saw, not from floor wedges |
| 6'a | **Neighbour clip**: a side that reaches past the spin centre of another anchored room within its span is cut back to that room's facing wall (− 10 cm) and becomes `inferred` (σ ≥ 25%) | `tiers._clip_by_neighbours` | a room cannot contain the spot where the photographer stood in another room: such a wall was seen through a door (PT-15) |
| 6'b | **Push apart**: overlapping rectangles move along the axis of least penetration until 10 cm apart. The least certain one moves first (not anchored, then larger widen, then smaller). Kept v4 rooms are fixed obstacles | `tiers._push_apart` | gate: no overlaps |
| 6'c | Walls, area, perimeter, bbox and footprint as **Monte Carlo** quantiles over the 4 offsets (400 draws). Kept v4 rooms add their own interval to the footprint | `tiers.layout_plan` | calibrated intervals (photo_tier v3.2, P-15/P-16 and the σ term) |
| 6'd | **Openings from doorway pairs**: a pair whose two photos are both placed and whose rooms the front-end linked is a `door` at the mean of the two cameras; width not measured (`not_observed`); adjacency `via` that opening, else `shared_wall`; adjacency still only from photo links | `tiers._pair_openings`, `extract._folder_adjacency` | the doorway pair IS the door evidence |
| – | Folders without a usable layout keep the v4 room (already widened). No usable layout at all → the v4 plan unchanged | `extract._layout_rooms` | – |

The v4 folder widening (×2/3.5/4) is **not** applied to layout rooms. Their size does not depend on the room's
placement; their uncertainty is the per-side Monte Carlo plus the calibrated 50% term. `run_capture` then adds the
tier scale term, as before.

**Results** (cached scenes, scored against `outputs/benchmark/lidar/<capture>/drift_on/plan.json`, which is the
`run_capture` default; folders mapped to LiDAR rooms by polygon overlap; table and JSON in photo_tier.md v3.2 and
`outputs/photo_tier/v3/cached_*.json`):

| Scene | Extractor | rooms / folders | adjacency correct / wrong | overlaps | footprint (LiDAR) | median room-area error | dims within 8% | dims in 95% interval | areas in interval |
|---|---|---|---|---|---|---|---|---|---|
| with_ceiling / rotate | v4 | 5/7 | 2 / 0 | 0 | 17.9 (62.5) | 78.8% | 0/8 | 5/8 | 1/5 |
| | **v4 + photo v3** | **7/7** | **5 / 0** | 0 | **45.8 [22.9, 68.7]** | 56.7% | 0/12 | 10/12 | 5/7 |
| floor_only / rotate | v4 | 5/6 | 2 / 0 | 0 | 37.5 (61.9) | 27.5% | 0/8 | 3/8 | 3/5 |
| | **v4 + photo v3** | **6/6** | 2 / 0 | 0 | 36.8 [16.6, 57.1] | 58.5% | 2/10 | 7/10 | 3/6 |

End to end: photo_tier.md v3.5. **Verdict:** the stitch gates improve (all folders become rooms, more correct links,
0 overlaps, and the with_ceiling footprint interval now contains LiDAR). The 8% size gate is **not** met on
simulated photos, and floor_only's median area error is worse than v4's. The intervals are widened to what the
errors measure.

**LiDAR and video unchanged** (verified this session). `extract_plan` was run on the three benchmark LiDAR scenes and
the v4 video scene, with a pristine copy of the code (my edits reverted) and with the new code. The plan JSON SHA-256
is identical (timing fields removed):

| Scene | rooms | footprint m² | hash (before = after) |
|---|---|---|---|
| LiDAR single_room | 3 | 17.610 | 2ef94ef03fc107f8 |
| LiDAR floor_only | 8 | 61.901 | 49df65b104a94537 |
| LiDAR with_ceiling | 8 | 62.498 | 9159870d937593d9 |
| video single_room (`outputs/runs/plan_v4_video_single_room/scene`) | 2 | 15.835 | 1be875db7d04bcf3 |

Script: `outputs/photo_tier/v3/same.py`. The only shared-code edit is a guard in `extract._tilts`: skip a wall id
that does not exist. It cannot fire on LiDAR, where every wall exists, as the identical hashes show.

**Issues (v4.10).** T-I10: the hallway folder over-reaches through its doors (+257% area on with_ceiling): open.
T-I11: L-shaped rooms are fitted as one rectangle: open (photo_tier v3.6 item 2). T-I12: a layout rectangle overlapped a
kept v4 room (1.06 m², corners end to end) → kept rooms are now fixed obstacles in the push-apart (fixed).
T-I13: footprint interval collapsed when no layout was used → the v4 plan is returned unchanged (fixed).
