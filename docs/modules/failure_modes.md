# Module: failure_modes (`floorplan/qa/`)

- Files:
  - `floorplan/qa/__init__.py`, `floorplan/qa/surfaces.py`: the detectors.
  - `scripts/analyze_failure_modes.py`: the evidence.
  - `docs/FAILURE_MODES.md`: the repo-level write-up per condition and per tier.
- Evidence: `outputs/failure_modes/`, with the headline numbers in `summary.json`.

## 1. Purpose and where it sits

Mirrors, glass, glossy floors, low light and motion blur corrupt the geometry *before* the plan is built. This module
sits between the front-ends (stage [1] SCENE) and plan extraction ([2]), and also serves damage detection ([5]).

It does two things:
1. It **measures** how each condition corrupts each tier on the sample captures, as evidence.
2. It provides **reusable detectors**. Each returns a `SurfaceFlag` holding:
   - `kind`, `score` and `where`;
   - `evidence`: the numbers;
   - `handling`: the recommended actions;
   - a one-line `message` for the user.

Detectors never delete data silently. The caller applies the handling, and the flag goes into the JSON, so every
exclusion can be audited.

**Interface for other modules** (all in `floorplan.qa`):

| Caller | Function | Input | Use |
|---|---|---|---|
| video / photo front-end | `frame_quality(img)` → `FrameQuality` | BGR or grey image | Per-frame sharpness, luma, clipped, dark, noise |
| video / photo front-end | `flag_frames(list[FrameQuality], timestamps or None)` → `(weight, blurry, flags)` | — | Frame weights; capture-level re-capture flags |
| video / photo front-end | `pick_sharpest(sharpness, timestamps, 0.5)` | — | Keyframe choice |
| LiDAR front-end | `lowconf_blob_mask(conf)`, `depth_confidence_stats(conf)` | ARKit confidence map | Glass, mirror and dark-glossy regions per frame |
| LiDAR front-end | `floor_reflection_stats(depth, conf, K, T_wc, floor_y, rgb_small)` | One frame in the aligned frame | Glossy-floor statistics |
| any tier after levelling | `below_floor_points(points, floor_y)` → `(mask, flags)` | Aligned points | Stairwell or floor-reflection points to exclude |
| plan extractor | `classify_wall_gaps(points, origins, walls, reference, floor_y)` → flags | Raw points + camera centres; `walls` = Plan walls or dicts `{id, p0, p1, normal}` | Mirror / glazing / opening per wall gap |
| any tier with fused 3D | `detect_mirrors(points, origins, VoxelOccupancy(surface_pts, normals), reference)` → `(flags, phantom_mask)` | — | Points seen through observed surfaces: `mirror`, `see_through`, `floor_reflection`, `inconsistent_geometry` |
| damage module | flags with `mask_for_damage_detection` | — | Do not detect stains on highlights or reflections |

Camera centres for the LiDAR scene: `origins = raw_points - raw_ray * raw_range[:, None]` (from `scene.npz`).

**How to run:**
```bash
python scripts/analyze_failure_modes.py   # all stages, about 10 min CPU + 30 s GPU
python scripts/analyze_failure_modes.py --stages frames,geometry,synthetic,lowlight,examples,learned,summary --captures single_room
```

## 2. How it works, step by step

1. **Image quality per frame** (`frame_quality`).
   - The image is resized to a 640 px long side, so that 12 MP photos and 1080p video frames give comparable numbers.
   - It then computes: the Laplacian variance (sharpness), median luma, the fraction of dark pixels (< 25) and of
     clipped pixels (≥ 250), and sensor noise. Noise is measured with the Immerkær operator on the flatter half of
     the pixels of a 1280 px copy.
   - Without this step, blurred or noisy frames enter matching and learned depth at full weight.
2. **Blur and low-light flags** (`flag_frames`).
   - Blur is judged *relative* to the median sharpness of the frames within ±1 s. A white wall is never sharp in
     absolute terms, but a frame half as sharp as its neighbours is blurred.
   - Low light is judged *absolutely*: luma < 60 or noise > 3, because it is a property of the room.
   - Capture-level flags (more than 30% blurry, more than 50% dark) trigger re-capture advice.
3. **LiDAR confidence blobs** (`lowconf_blob_mask`).
   - ARKit marks every depth edge as low confidence, as 1–3 px lines.
   - A morphological opening and a minimum area keep only the *blobs*: glass, mirrors, dark glossy panels and far
     see-through regions.
4. **Glossy floor** (`floor_reflection_stats`, `below_floor_points`).
   - Every pixel ray is cast to the levelled floor plane.
   - Measured points clearly *below* the floor can only be reflections or a real lower level (a stair).
   - We also count low-confidence pixels and highlights on the floor.
5. **Seen-through test** (`VoxelOccupancy`, `occlusion_violations`).
   - Each point's viewing ray is marched through a sparse 5 cm voxel map of the fused surface.
   - Crossing an observed surface counts only when the surface normal faces the ray (|cos| > 0.34), so grazing along
     a wall is ignored.
   - A real surface cannot be seen through a wall. Mirror phantoms, things behind glass that other views saw as
     solid, moved doors, people and drift can.
6. **Classify each cluster** (`detect_mirrors`).
   1. Cluster the flagged points in plan view.
   2. Fit the plane through the *crossing* points.
   3. Decide the class:
      - not planar (RMS > 5 cm) → `inconsistent_geometry`;
      - horizontal with the points below it → `floor_reflection`;
      - vertical with a reflection match → `mirror`;
      - otherwise → `see_through`.
7. **Wall-aware check** (`points_behind_walls`, `classify_wall_gaps`).
   - With the plan's walls, find the points whose ray crosses a wall segment and lands behind it.
   - Group them by *where the ray crosses the wall*, which gives the extent of the pane or opening.
   - For each group, run the reflection test against its shifted-plane control, then the sill test: is wall surface
     observed 0.1–0.5 m above the floor under the gap?
   - Result: mirror, glazing, or opening/glass door.
8. **Evidence script.** Stages: `frames`, `geometry` (including the cluster-to-frame verification figure),
   `synthetic` (ground-truth room), `lowlight` (simulation), `examples` (hall mirror, stairwell), `learned` (MoGe-2
   against LiDAR on hard frames) and `summary`.

## 3. Decisions

- **D-FM-1: geometric tests rather than learned mirror or glass segmentation.**
  - Options considered:
    - (a) a learned mirror/glass segmenter (research models such as MirrorNet or GDNet, with research licences, which
      would need `transformers` or extra envs);
    - (b) geometry: seen-through consistency and reflection symmetry.
  - Chose (b). It is tier-agnostic: it works on any point cloud with camera centres. It is explainable and has no
    licence risk. The synthetic test recovers mirror, glass and door extents within one 0.2 m bin.
  - Revisit if the own-home capture shows mirrors that the geometry misses, because the reflected room was never
    scanned.
- **D-FM-2: the reflection test needs a shifted-plane control.**
  - Evidence: without it, a `single_room` doorway (wall r0_w4) scored a match of 0.52 and was labelled a mirror. Its
    ±25 cm control scored 0.48. In the synthetic room, the doorway scored 0.88 against a control of 0.81.
  - Rule: a mirror needs a match ≥ 0.5 **and** a margin over the control ≥ 0.25.
- **D-FM-3: tell glazing from an opening with a sill test, not with the lowest crossing height.**
  - Evidence: the crossing-height version labelled 66 of 69 sample see-through regions as glazing. A chest-height
    phone rarely sends a ray through the bottom of a doorway to a point within 4 m, so the lowest crossing was
    0.34–1.1 m even for doors.
  - The sill test gives 4 glazing, 64 openings and 1 mirror candidate, and is correct on all three synthetic cases.
- **D-FM-4: a non-planar crossing surface means `inconsistent_geometry`, not a reflection.**
  - Evidence: the first version called 3 large clusters (6.9k / 25k / 14k points) "floor reflections". Their points
    sit 0.5–1.3 m *above* the floor, and their crossing points have an RMS of 0.21–0.41 m about any plane.
  - Real reflectors are flat.
- **D-FM-5: blur relative to temporal neighbours, low light absolute, and check noise before trusting sharpness.**
  - Evidence: blur rises with rotation speed (31–49% of frames above 90°/s against 0.4–1.0% below 30°/s).
  - In the simulation, noise inflates sharpness 2.8–11x, so a noisy dark frame would otherwise look "sharp".
- **D-FM-6: the noise estimate uses the mean on flat pixels, not a robust median.**
  - Evidence: on H.264 frames the median-based Immerkær estimate was exactly 0.2471 for every capture, because it
    collapses to one grey level after the codec removes the noise. The mean form gives continuous values (0.25–0.27).
- **D-FM-7: the learned-depth comparison gives MoGe the true field of view.**
  - Why: this isolates depth and scale errors from focal-length errors. That is the best case for the photo tier, so
    the reported errors are a lower bound.

## 4. Issues log

| Symptom | Root cause (evidence) | Fix options | Chosen fix and why |
|---|---|---|---|
| Reflection "match" high for every cluster (0.84–0.98 unreflected) | The reference cloud contained the phantom points themselves | Exclude the cluster; restrict the reference to the camera side | Reference = surface points on the camera side of the plane, plus the shifted control (D-FM-2) |
| Doorway labelled mirror (0.52) | A cluttered room matches anything within 5 cm | Tighter tolerance; a control | Control and contrast threshold |
| Large clusters labelled floor reflection although the points are above the floor | PCA of a diffuse crossing cloud gives a "horizontal" normal | Require the points below the plane; require planarity | Both: an RMS < 5 cm gate and a below-the-plane test (D-FM-4) |
| 66 of 69 sample gaps labelled glazing | The lowest crossing height is not a sill (D-FM-3) | Points' lowest height (fails for windows: the outside drops below the floor); crossing height; sill coverage | Sill coverage |
| Pane extent wrong in the synthetic test (0–3.2 m for a 1–2 m mirror) | Grouping by the phantom point's along-wall position, not by where its ray crosses | — | Group by ray–wall crossing |
| Noise σ identical across captures | Median quantisation on codec-smoothed video (D-FM-6) | — | Mean on flat pixels |
| "Below-floor" points looked like glossy-floor reflections | Inspection: they are the stairwell (`lidar_mirror_and_stairwell.png`) | — | Flag renamed `below_floor` ("stairwell or reflection"); the plan extractor decides |
| GPU OOM on the MoGe run | GPU shared with other agents | Wait; CPU | Fall back to CPU when less than 3 GB is free (4 frames) |

## 5. Results

All numbers come from `outputs/failure_modes/summary.json`. Captures are given in the order `single_room` /
`floor_only` / `with_ceiling`.

**Blur:**
- 6.3% / 5.1% / 6.2% of frames are blurry.
- Below 30°/s rotation: 0.4% / 1.0% / 0.85%. Above 90°/s: 49% / 31% / 38%.
- Spearman correlation of relative sharpness with rotation speed: −0.56 / −0.41 / −0.39.
- Figures: `<capture>/frames_summary.png`, `<capture>/frames_timeseries.png`.

**Light:**
- Median luma 150 / 154 / 153; 10th percentile 117 / 131 / 131. No frames have more than 1% clipped pixels.
- So the sample is not low-light. The simulation (`lowlight_simulation.csv`) flags 100% of frames at 16x and 64x less
  light, and 5–8% at 1–4x less.

**LiDAR confidence:**
- Low + medium: 6.1% / 10.3% / 9.4% of pixels.
- Blobs: 4.9% / 8.8% / 7.9% on average; 16% / 37% / 33% at the 95th percentile.
- Wall-band blob fraction: 5.9% / 9.8% / 8.4%.
- Figure: `<capture>/geometry_plan.png`, right panel.

**Mirror (hall, `floor_only` #2016):**
- Depth in the mirror region has a median of 6.1 m, against 2.5 m for the frame.
- Confidence there: 82% low, 10% medium, 7.8% high. 7.8% passes the D-006 filter.
- Figure: `lidar_mirror_and_stairwell.png`.

**Glossy floor:**
- Floor rays landing below the floor: 0.03% / 0.56% / 0.70%. The latter two are the stairwell.
- Low confidence on floor pixels: 5.1% / 6.5% / 5.7%.

**Seen through a surface:**
- 1.1% / 9.5% / 5.0% of the 400k queried points.
- Clusters: `single_room` 3 see-through, 2 inconsistent and 1 small horizontal; `floor_only` 7 inconsistent and 1
  see-through (glass shower screen, #4251); `with_ceiling` 7 inconsistent.
- Figure: `<capture>/phantom_clusters_frames.png`.

**Wall gaps** (plan_alpha walls): 64 opening/glass door, 4 glazing, 1 mirror candidate. The candidate is `floor_only`
wall r0_w8: 250 points, match 0.82 against a control of 0.47. It is unverified.

**Synthetic ground truth** (`synthetic.json`):

| Case | Recovered | Truth | Other numbers |
|---|---|---|---|
| Mirror | 1.0–2.0 m | 1.0–2.0 m | Match 1.00 vs control 0.43 |
| Glazing | 1.0–2.6 m | 1.0–2.5 m | Sill coverage 0.95 |
| Opening | 0.6–1.6 m | 0.6–1.5 m | Sill coverage 0.10; correctly not called a mirror (0.88 vs 0.81) |

**Learned depth on hard frames** (MoGe-2 with the true field of view, against LiDAR; `learned.json`,
`hard_frames.png`):

| Frame | LiDAR/MoGe scale | AbsRel after scale correction | Pixels > 10% error |
|---|---|---|---|
| Bathroom mirror/glass #4547 | 0.94 | 5.5% | 34% |
| Dark window #2598 | 1.40 | 1.4% | 8% |
| Shower screen #914 | 1.68 | 1.5% | 13% |
| Glossy floor + balcony #3897 | 1.13 | 7.2% | 44% |

## 6. Limitations and next steps

- **Glass.** A frameless glass door or partition with no sill is indistinguishable from an open doorway with LiDAR
  alone. It is flagged "opening_or_glass_door" with a wider interval. Next step: the RGB frame edge, or a reflection
  highlight inside the gap.
- **Mirrors.** The reflection test needs the reflected part of the room to have been scanned. A mirror facing an
  unscanned area falls back to `see_through` (excluded, interval widened). No *confirmed* mirror survives D-006 in the
  sample, so the real-data validation of the mirror class rests on the synthetic test and the own-home capture.
- **Wet floors.** A high-gloss or wet floor acting as a horizontal mirror is untested on real data, because the
  sample's floors do not produce sub-floor phantoms.
- **Low light.** The thresholds come from simulation (luma 60, noise 3). Calibrate them on the own-home capture and
  check how noise interacts with HEIC denoising on real phones.
- **Clutter, open plan, non-Manhattan walls, tiny rooms.** These belong to the plan extractor; see the
  `FAILURE_MODES.md` §5 table.
- **`inconsistent_geometry` rate (5–9.5% on long captures).** Hand it to the drift module as a before/after metric:
  pose-graph correction should lower it.
- **Wiring.** Front-ends should call `flag_frames` and `pick_sharpest`. Plan extractors should call
  `classify_wall_gaps`, `detect_mirrors` and `below_floor_points` before the final wall fits. The export should
  include `SurfaceFlag.to_dict()` in the plan JSON `meta`.
- **Next measurement.** Count SIFT/LightGlue matches against sharpness ratio to turn the blur threshold into a direct
  matching-quality threshold.

## 7. Proposed commits

1. `qa: SurfaceFlag + Handling vocabulary for difficult surfaces`
   - Files: `floorplan/qa/__init__.py`, the `surfaces.py` header and dataclasses.
   - Why: one auditable format for every exclusion or re-label.
2. `qa: per-frame image quality, blur/low-light flags, sharpest-frame picker`
   - Files: `surfaces.py` (`frame_quality`, `flag_frames`, `pick_sharpest`).
   - Why: frame selection and re-capture advice for the photo and video tiers.
3. `qa: LiDAR confidence blobs and glossy-floor statistics`
   - Files: `surfaces.py` (`lowconf_blob_mask`, `floor_reflection_stats`, `below_floor_points`).
   - Why: glass and mirror regions; sub-floor points.
4. `qa: seen-through test with sparse voxel occupancy + cluster classification`
   - Files: `surfaces.py` (`VoxelOccupancy`, `occlusion_violations`, `cluster_plan`, `detect_mirrors`).
   - Why: phantom geometry from mirrors, glass, drift and people.
5. `qa: reflection test with shifted-plane control; wall-gap classifier with sill test`
   - Files: `surfaces.py` (`reflection_consistency`, `is_mirror`, `points_behind_walls`, `_sill_coverage`,
     `classify_wall_gaps`).
   - Why: mirror vs glazing vs opening.
6. `scripts: analyze_failure_modes (frames, geometry, synthetic, lowlight, examples, learned, summary)`
   - Files: `scripts/analyze_failure_modes.py`.
   - Why: reproducible evidence.
7. `docs: FAILURE_MODES.md + modules/failure_modes.md`
   - Why: the brief's "cover mirrors, glass, wet-look, low light".

## 8. Concepts to explain in the defense

- **Why mirrors create phantom rooms.** Every depth sensor assumes light travels straight to a surface and back. A
  mirror folds the path, so the reflected object appears as far *behind* the mirror as it really is in front of it.
- **Reflection symmetry test.** Fold the suspicious points back through the wall. If they land on furniture that is
  really in the room, it was a mirror.
  - The control: fold through a plane 25 cm off. A real mirror only matches at the right plane. A cluttered room
    matches "a bit" everywhere.
- **Seen-through test (visibility consistency).** If one view saw a solid wall here, no other view can have seen
  something behind that wall through it. A violation means a reflection, glass, a moved object or drift. Planarity
  of the crossings tells a pane apart from a disagreement between passes.
- **Sill test.** A window has wall below it; a doorway goes to the floor. We check whether wall surface was observed
  in the band 10–50 cm above the floor under the gap.
- **Confidence blobs vs edges.** ARKit's low confidence is mostly thin lines on object edges, which are harmless. Glass
  and mirrors give *areas*. A morphological opening keeps the areas.
- **Why learned depth fails on glass.** A single image cannot tell a reflection from a scene. The shape is often right
  (1.5% error after scaling), but the absolute scale on glass-heavy frames was wrong by 28–41%. Hence scale is
  fused over many frames, never taken from one.
- **Blur vs rotation.** Hand-held blur is mostly rotational: 0.4–1% of frames are blurry below 30°/s and 31–49% above
  90°/s. So the protocol says "turn slowly", and the video tier keeps the sharpest frame per half second.
- **Noise inflates sharpness.** The Laplacian measures high-frequency energy, and noise is high-frequency. So we check
  noise before trusting sharpness.
