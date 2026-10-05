# Decision log

Every design choice, why it was made, what else was considered and what evidence supports it.
Newest entries go at the bottom.

Template: **Context** (the problem) → **Options** (what we could do) → **Decision** → **Why** → **Evidence** →
**Risks / revisit if**.

---

## D-001 Explore the data before designing anything

- **Context.** We received three captures with no documentation. A wrong assumption about conventions silently
  destroys cm-level accuracy. For example, the wrong camera-axis convention shifts every point by tens of cm.
- **Options.** (a) Assume the ARKit conventions from Apple's docs and start building. (b) Measure the conventions
  from the data first.
- **Decision.** (b). Write `scripts/explore_capture.py` and record the findings in `docs/DATA_NOTES.md`.
- **Why.** Measuring takes minutes, while a wrong assumption costs hours of debugging later. It also gives us figures
  to show *what the data contains*: rooms, loop closures, glass, ceiling coverage.
- **Evidence.** The ARKit-axes assumption would have been wrong: 22–32 cm re-projection error versus 6–7 mm for the
  OpenCV axes (D-005).

## D-002 Scope: the provided sample data is the target input

- **Context.** The case study says "We provide no captures" (Part 1). The email, however, says "Please run your code
  on this Sample Data". I have no iPhone or LiDAR device, so I cannot record my own captures, build
  the Part 2 benchmark set, or run the Part 3 head-to-head.
- **Options.**
  - (a) Block until the recruiters reply.
  - (b) Treat the sample data as the target input, build the full pipeline on it, and keep the capture-dependent
    parts (Part 1 route, own benchmark, Part 3) as clearly labelled gaps.
- **Decision.** (b). A clarification email was drafted (see `docs/OPEN_QUESTIONS.md`).
- **Why.** Every possible answer still requires a pipeline that works on this data, so that work is never wasted.
  The deadline does not allow waiting.
- **Risks / revisit if.** If the recruiters reply that Part 1 is still required, a Route 2 protocol takes about
  1 hour to write. Stray Scanner, which produced this data, is the natural tool for it.

## D-003 Order of work: LiDAR tier first

- **Context.** All three tiers are mandatory, but they are not equally hard or equally informative.
- **Options.** Start with photos (the hardest), video, or LiDAR.
- **Decision.** Build in this order: LiDAR, then drift handling, then the evaluation harness, then video, then
  photos, then damage, then the fix loop.
- **Why.**
  1. The sample data is LiDAR-native, so this tier can be built and checked first.
  2. Once trusted, the LiDAR result becomes the **reference** for the video and photo tiers. Those tiers are derived
     from the same captures, so they can be scored against LiDAR without tape ground truth.
  3. The plan-building back-end (rooms, walls, openings, dimensions, stitching, JSON, rendering) is shared by all
     tiers. Building it first on the cleanest input means later tiers only add a front-end that produces metric
     geometry.
  4. The fix loop (Part 4, 25% of the score) needs a benchmark with failing gates, so an evaluation harness must
     exist early.

## D-004 Measure with geometry; use learned models only where geometry is missing

- **Context.**
  - Learned floor-plan models (RoomFormer, Raster2Seq, PolyLayout, SpatialLM) predict room polygons.
  - Their benchmarks score corners within 10 px on a 256 px map, which is decimetres, not centimetres.
  - Their weights are trained on research-only datasets (Structured3D, ScanNet++), which is a licence risk.
- **Options.** (a) Learned polygons for everything. (b) Classical geometry for every number. (c) Learned models
  for structure, geometry for numbers.
- **Decision.**
  - Every reported dimension comes from fitting planes and lines to measured points. That fitting is deterministic,
    auditable, and yields an uncertainty.
  - Learned models are used only where the sensor gives no metric geometry: video and photo tiers (scale, depth)
    and damage recognition.
  - Learned room polygons are kept as an optional cross-check, not as the source of dimensions.
- **Why.**
  - The gates are 1–2 cm.
  - A plane fitted to thousands of LiDAR points has a standard error of millimetres, while a learned polygon vertex
    is quantised to the raster.
  - Geometry also explains its own failures, which matters for the fix loop.

## D-005 Pose convention: camera-to-world, OpenCV camera axes; world is y-up

- **Context.** ARKit uses camera axes x right, y up, z backward. Many exporters convert them, some do not.
- **Evidence.** Re-projection test (`explore_capture.py`), median depth disagreement:

  | Capture | OpenCV axes | ARKit axes |
  |---|---|---|
  | single_room | **6.4 mm** | 223 mm |
  | floor_only | **6.9 mm** | 285 mm |
  | with_ceiling | **7.0 mm** | 319 mm |

  The fused floor plane's normal is within 0.02–0.6° of +y.
- **Decision.** Use the `odometry.csv` pose directly as `T_world_camera` with OpenCV camera axes. Treat +y as up.
- **Why.** Measured, not assumed.

## D-006 Fuse only high-confidence LiDAR depth, up to 4 m

- **Context.**
  - ARKit marks each depth pixel low, medium or high confidence.
  - Low and medium pixels sit on depth edges, glass, dark or specular surfaces, and far ranges.
  - LiDAR noise grows with range; Apple's sensor is specified to about 5 m.
- **Options.** (a) Use all pixels. (b) Use medium and high. (c) Use high only, with a range cap.
- **Decision.** (c): confidence = 2 and 0.2 m < depth < 4.0 m. Both are config values.
- **Why.**
  - High-confidence pixels are 89–94% of all pixels, so we keep most of the data.
  - Dropping the rest removes the worst outliers: glass "holes", flying pixels at edges, and mirror reflections.
  - TSDF fusion of the remaining frames averages out the remaining noise.
- **Risks / revisit if.** Glass and mirror surfaces end up with *no* points. We must detect them as "unobserved"
  rather than treat them as open space (handled in openings detection).

## D-007 Dense fusion: Open3D tensor VoxelBlockGrid (TSDF)

- **Context.** A single LiDAR frame is noisy (about 1 cm) and sparse (256x192). We need one consistent surface for
  the whole capture.
- **Options.**
  - (a) Concatenate raw points: noisy, with duplicates.
  - (b) TSDF fusion (KinectFusion-style): averages many observations per voxel.
  - (c) Neural or Gaussian surface reconstruction: slow, GPU-heavy, and no accuracy gain on LiDAR input.
- **Decision.** (b), using Open3D's tensor `VoxelBlockGrid` (MIT).
- **Why.**
  - TSDF is the standard, explainable way to average depth.
  - It runs on CPU in seconds (the exploration fused 2,437 frames in under 30 s).
  - Its output (points and normals) feeds plane fitting directly.
- **Evidence.** Open3D 0.20's legacy `ScalableTSDFVolume` silently returns an empty volume. This was found
  during environment setup, which is why we use the tensor API.
- **Risks / revisit if.** TSDF slightly rounds corners and can erode thin structures. Dimensions are therefore taken
  from planes fitted to *raw* high-confidence points near each wall, not from the TSDF surface.

## D-008 Keyframes: 5 cm / 5° / 0.5 s, skip mostly-invalid frames

- **Context.**
  - The phone records at 60 Hz, so consecutive frames are nearly identical.
  - Fusing all 9,745 frames of the longest capture adds time but no information.
- **Options.**
  - (a) Use every frame.
  - (b) Use every k-th frame.
  - (c) Pick frames by motion.
- **Decision.** (c).
  - A frame becomes a keyframe after 5 cm of travel, 5° of rotation, or 0.5 s, whichever comes first.
  - Frames where fewer than 25% of depth pixels survive the D-006 filter are skipped.
- **Why.**
  - Fixed stride (b) over-samples when the operator stands still and under-samples fast turns.
  - Motion-based selection keeps coverage uniform in space.
- **Evidence.** Frames kept out of the total:

  | Capture | Keyframes / frames |
  |---|---|
  | single_room | 308 / 1,715 |
  | floor_only | 1,041 / 5,251 |
  | with_ceiling | 1,978 / 9,745 |

  Scene build times are 9 s, 24 s and 47 s on CPU.

## D-009 Level the scene with the floor plane, then rotate walls onto the axes (Manhattan)

- **Context.**
  - Heights such as ceiling height and sill height are measured along "up".
  - The gate for ceiling height is 1.5 cm.
  - A 0.5° tilt over a 5 m room moves a height by 4.4 cm.
- **Options.**
  - (a) Trust ARKit's gravity.
  - (b) Refine gravity with the fitted floor plane.
- **Decision.** (b), but only when the floor tilt is below 2°.
  - A larger tilt suggests a ramp or a mis-detected floor, so ARKit gravity is kept instead.
  - Then rotate about the vertical axis so the dominant wall direction lies on x.
  - That direction is estimated from wall normals using the "4θ" circular mean: angles 90° apart coincide once multiplied by 4.
- **Why.**
  - The floor is a direct measurement of "level" over metres.
  - Axis-aligned walls turn wall fitting into fitting one offset per wall instead of an angle and an offset, which is more stable.
  - The plans then look like magicplan / poly.cam plans.
- **Evidence.** Tilt corrected and wall alignment per capture:

  | Capture | Floor tilt corrected | Wall points within ±5° of an axis |
  |---|---|---|
  | single_room | 0.53° | 59–64% |
  | floor_only | 0.19° | 59–64% |
  | with_ceiling | 0.04° | 59–64% |

  The rest of the "vertical" surfaces are furniture and clutter.
- **Risks / revisit if.** Non-rectangular rooms. Walls off the Manhattan axes are kept and fitted at their own angle.

## D-010 Build the room/wall extractor twice, two ways, and let evidence pick

- **Context.**
  - Turning a 3D scan into rooms, walls and doors is the crux of the product. Every gate depends on it.
  - The walk-in test (30%) runs it cold on a space we have never seen.
- **Options.**
  - (a) Pick one approach and iterate.
  - (b) Build two approaches that fail in different ways and compare them on the same evidence:
    - "boundary-first": detect walls, partition space into cells, label the cells;
    - "space-first": segment the free floor space into rooms, then fit walls.
- **Decision.** (b).
  - The approaches were built in parallel.
  - A judge compares them on repeatability using the two scans of the same apartment, then on plausibility, robustness, clarity and speed.
  - The loser's good ideas are borrowed.
- **Why.** The choice of segmentation paradigm is the biggest risk in the project, and building both costs only wall-clock time.
- **Evidence.** `docs/modules/judge_plan_extractor.md` (verdict: D-018).

## D-012 Scope after the recruiters' reply: own captures are mandatory; complete scope

- **Context.** The recruiters answered our questions (`docs/OPEN_QUESTIONS.md`):
  - Our own captures are mandatory.
  - The sample data is acceptable for LiDAR.
  - There are no native photos to share.
  - Complete the scope with as little deviation as possible.
- **Decision.**
  - **LiDAR tier:** sample data (Stray Scanner), plus the ARKitScenes laser ground truth for absolute accuracy.
  - **Photo and video tiers:** my own home, captured with my phone following the protocol, with tape
    ground truth, staged damage (2 classes), repeat captures and a consumer-app head-to-head.
  - **Sample data:** we also run video and photo tiers derived from the sample `rgb.mp4`, for cross-tier agreement
    with LiDAR.
  - **Part 1:** Route 2 protocol (`docs/CAPTURE_PROTOCOL.md`).
  - **Part 3:** a consumer app on 2 of our rooms, compared at the photo/video tier (disclosed deviation, since LiDAR is
    not possible on own rooms).
- **Why.** This follows the recruiters' instruction and covers every part of the brief. The only deviations are the
  ones forced by hardware, and each is stated.

## D-013 The capture protocol includes a sheet of paper on the floor as a scale reference (SUPERSEDED by D-067)

- **Status (4 Oct).** Superseded by D-067: no reference object in any capture. The sheet detector stays in the code as
  an option, off by default.
- **Context.**
  - Photos and plain video have no metric scale.
  - Learned metric depth (MoGe-2, Depth Anything 3, MapAnything) estimates scale to roughly a few percent.
  - The video gate is ±3% on wall lengths, so a few percent is too loose to rely on alone.
- **Options.**
  - (a) Learned scale only.
  - (b) A printed fiducial marker (AprilTag/ArUco). Accurate, but needs a printer, and printer "fit to page" scaling
    silently changes its size.
  - (c) A credit card. Too small at room scale.
  - (d) A plain **A4 or US-Letter sheet** flat on the floor. Every home and office has one, its size is
    standardised regardless of any printer, and its 4 corners lie on the floor plane.
  - (e) Priors: camera height, door height.
- **Decision.** (d), fused with (a) and (e): every available cue enters an uncertainty-weighted estimate, and the
  interval reflects which cues were present. The pipeline still runs without the sheet ("any picture in, results
  out"), just with wider intervals.
- **Why.** It is the cheapest strong scale cue a non-engineer can follow literally at the walk-in, and it needs no
  printing.
- **Risks / revisit if.**
  - A white sheet on a white floor has low contrast. The detector must handle it, and we test it on our captures.
  - A sheet that is not flat (curled) is detected by its non-planarity and down-weighted.

## D-011 Measure the LiDAR noise from the data; weight points by it; use exact depth intrinsics

- **Context.**
  - Intervals need a sensor-noise term, and fits should trust precise points more.
  - Guessing "1 cm" is not good enough for 1–2 cm gates. Accuracy comes first (my standing rule).
- **Experiment.** `scripts/noise_model.py`:
  - Re-project frame i's depth into frame j (0.1–1 s later) with the ARKit poses.
  - Compare with frame j's measured depth.
  - Robust σ of one sample = 1.4826 · MAD / √2.
  - Binned by range and by ARKit confidence; 250 frame pairs per capture, hundreds of thousands of points per bin.
- **Results** (σ of one high-confidence sample, worst of the three captures):

  | Range | 0.2–0.5 m | 0.5–1 | 1–1.5 | 1.5–2 | 2–2.5 | 2.5–3 | 3–3.5 | 3.5–4 |
  |---|---|---|---|---|---|---|---|---|
  | σ (mm) | 6 | 7 | 7 | 7 | 10 | 16 | 18 | 19 |

  - **Medium confidence: 15–40 mm**, 3–5x worse at every range. This confirms D-006: drop medium and low pixels.
  - **Range bias** (median signed disagreement): within ±3 mm below 2.5 m, up to ±10 mm at 3–4 m. Prefer near points.
  - **Depth-intrinsics convention:** the pixel-centre-correct scaling, c' = (c + 0.5)·s − 0.5, beats plain scaling
    c' = c·s on **all three** captures (6.70 vs 6.78, 7.58 vs 7.61, 7.13 vs 7.21 mm). The gain is tiny but
    consistent, so we switch.
- **Decision.**
  1. Keep the confidence = 2 filter.
  2. Fit walls with **inverse-variance weights** w = 1/σ(r)². A point at 1 m counts about 6x more than one at 3.5 m.
  3. Use σ(r) as the sensor term of intervals.
  4. Use pixel-centre-correct depth intrinsics.
  5. The model lives in `floorplan/uncertainty/noise.py` and `lidar_noise.json`.
- **Why.** These are measured, not assumed. Weighting is the textbook optimal estimator for points of unequal
  precision. Points near the wall dominate, which is also how a careful surveyor works.
- **Caveat.** The σ values include a little pose error, so they are an upper bound: conservative intervals.

## D-014 Photo-tier inputs for the sample data are simulated from the video, without leaking poses

- **Context.**
  - The recruiters have no native photos of the sample apartment.
  - The photo tier must still run on the same rooms, so the LiDAR plan can be the reference.
  - Our own home's native phone photos (with tape ground truth) are the primary photo benchmark.
- **Options.**
  - (a) Random frames per room.
  - (b) Simulate a person following our protocol: choose frames that together see every wall, plus doorway "looking out"
    shots.
- **Decision.** (b), in `scripts/make_photo_folders.py`.
  - Uses the LiDAR plan's room polygons and doors, and the true poses, **only to choose frames**.
  - Writes 2–8 upright JPEGs per room folder, with an EXIF-style 35 mm-equivalent focal length (rounded to an integer,
    as phones write it).
  - Puts poses, depth and intrinsics in a separate `truth/` file used only for scoring.
- **Why.**
  - Random frames would be unrealistically blurry or redundant.
  - The protocol-shaped selection matches what the walk-in capture will look like.
- **Limitations (disclosed).**
  - Video frames are compressed and lower quality than real photos.
  - The phone was held in portrait, so the frames are portrait.

## D-015 Score own captures against tape ground truth by geometry, not by hand

- **Context.** My own home is the photo/video benchmark. Its ground truth is a hand-written list of
  wall lengths (W1..Wn clockwise from the main door), ceiling heights, opening widths and damage extents.
- **Decision.** `floorplan/benchmark/gt_eval.py` pairs ground truth with predictions automatically, in three steps:
  1. **Rooms:** Hungarian assignment on wall-length profiles, or exact name match for photo folders.
  2. **Walls:** the best cyclic order, clockwise seen from above (mirror-aware, I-001), preferring a door wall as W1.
     Superseded by D-072.
  3. **Openings:** Hungarian assignment on width. Missed and phantom openings count as misses, as the gate requires.

  It then reports the gates and the empirical coverage of our 95% intervals.
- **Why.** Hand matching is slow and can be biased. Geometric matching is reproducible, so reviewers can regenerate
  every number.
- **Evidence.** The synthetic test (`tests/benchmark/test_gt_eval.py`) recovers an injected 12 mm wall error exactly and
  finds the door wall. It also exposed a bug: a door shared by two rooms was flagged as phantom when only one room's
  ground truth listed it. Fixed.

## D-016 One place where intervals widen with thinner input: the tier scale term

- **Context.** The case study requires the same output contract from each tier, "with intervals that widen honestly as
  sensor data thins".
- **Decision.** The plan extractor's intervals cover fit error, sensor noise and drift.
  `floorplan/uncertainty/tier_budget.py` then adds the tier's relative scale uncertainty σ_s in quadrature:
  - lengths and heights: σ_s·L;
  - areas: 2σ_s·A, because area scales with the square of scale.

  σ_s is 0 for LiDAR, since its depth is metric. For video and photos it comes from the scale-cue fusion: paper sheet,
  learned depth, priors. (Since D-067 the sheet is opt-in and off by default.)
- **Why.** A scale error multiplies every length in the reconstruction. Treating it as one shared, explicit term is
  correct, and it can be audited: the term is written into each measurement's `method` string.

## D-017 Photo protocol revised: turn on the spot in the middle, plus doorway pairs (instead of corner shots)

- **Context.** On the sample data, photos taken from corners and doorways shared too few features.
  - SfM registered only 17 of 46 photos, in 5 separate pieces.
  - Only 3–4 of 9 rooms could be joined by image evidence (`docs/modules/photo_tier.md`).
  - The photo-tier stitch gate needs **all** rooms joined, with correct adjacency.
- **Options.**
  - (a) Keep the corner shots and rely on a door-matching fallback.
  - (b) **Turn on the spot from the room's middle (6 shots, about 1/3 overlap each)**, as panorama-based capture
    (Zillow 3D Home, Matterport phone capture) does.
  - (c) Use the ultra-wide lens. Its strong distortion and lower quality hurt metric depth.
- **Decision.** (b), plus **doorway pairs**: standing in a doorway, one photo into each adjacent room, seconds apart.
  - The EXIF capture times pair them, so the two photos share a camera centre. That links the rooms even when no
    features match.
  - The door-matching fallback is kept as a last resort, with wider intervals.
- **Why.**
  - Shots from one spot overlap reliably, and rotation is easy to estimate between them.
  - Metric depth per photo then gives the room layout around a common centre.
  - The doorway pair turns "same spot, two rooms" into a hard geometric link.
- **Also (withdrawn by D-067: no sheet).** The paper sheet goes on the darkest open floor patch. The detector finds 84%
  of sheets when the paper is brighter than the floor, but only 20% of white paper on a near-white floor.
- **Risks / revisit if.** Simulating this protocol from the sample's walk-through video is imperfect. My
  own capture is the real test, and the wave-3 photo agent is validating the change overnight.

## D-018 LiDAR plan extractor: plan_beta (space-first) as the base, with four plan_alpha parts ported

- **Context.** The judge (`docs/modules/judge_plan_extractor.md`) ran both extractors on identical scenes and scored
  them with the shared harness.
- **Evidence.**
  - **Repeatability is a statistical tie, and both fail by roughly 10x:**
    - beta: 11.2% of walls within max(1 cm, 0.5%), Wilson interval [6.4, 19.0]%;
    - alpha: 8.5%.
  - **Where beta is ahead:**
    - same-topology wall agreement: 3.0 vs 4.3 cm;
    - topology-error size: 22 vs 70 cm;
    - self-stability under re-sampling: 89% vs 79%;
    - interval coverage: 0.61 vs 0.54.
  - **What decided it:** beta **never invents space**. Alpha added unentered "rooms" of 5–9 m², and on a degenerate
    input reported 4.4 m² of rooms from 2.25 m² of data. That would be a liability with mirrors and glass at the
    walk-in.
  - **Where alpha is better:** door widths (±1.4 cm), ceiling handling under drift, coverage, and speed.
- **Decision.** Use beta, and port alpha's:
  - wall-line detector, as segmentation barriers;
  - jamb-in-core door width;
  - ceiling double-layer rule;
  - inconsistency term in the intervals.
- **Fix plan.** The defect list B-1..B-10 is the v2 work plan.
  - The biggest is B-1: rooms merge when a partition is unseen above 1 m.
  - A one-parameter experiment showed matched rooms going 6 → 10 and the median wall Δ going 11 → 2.5 cm.
  - A frozen copy, `floorplan/plan/beta_v1`, keeps the "before" for the Part 4 fix loop.

## D-019 Canonical LiDAR scenes: drift correction ON by default; per-point source frame kept

- **Context.** Drift correction (`docs/modules/drift.md`) brings revisit separation on floor_only from 14 cm (ARKit
  as-is) to 1.4 cm, and the ceiling level spread on with_ceiling from 6.0 to 0.6 cm. The plan extractors were first
  developed on drift-OFF scenes.
- **Decision.**
  - `scripts/build_canonical_scenes.py` builds `outputs/scenes_v2/` with drift ON and the D-011 intrinsics.
  - `scripts/run_capture.py` runs drift ON by default; `--no-drift` exists for the ablation only.
  - Raw points now carry their source frame (`raw_frame`), so a room can be measured from a single visit. My
    idea: within one visit, tracking is consistent to about 0.7 cm, while two passes disagree by
    1.5–2.5 cm even after correction.
- **Why.** "Poses used as-is" is an automatic fail of the drift gate. Single-visit measurement attacks the
  repeatability floor directly.

## D-020 Video tier v2: treat the video like the LiDAR tier, with learned depth

- **Context.** Video v1 works on the short capture (scale +1.2..+4.2%) but fails on the long multi-room captures:
  +187% / +284% scale error. Visual odometry loses track on blank walls, and the pieces are chained with the wrong
  relative scale.
- **Decision.**
  - Split the video into segments at tracking breaks.
  - Give each segment metric scale from per-frame learned depth plus the paper sheet (opt-in since D-067).
  - Join the segments and close loops with the same fragment pose graph as the LiDAR drift module, using learned
    depth as pseudo-LiDAR.
  - Read the focal length from video metadata when present.
- **Why.**
  - One mechanism, already measured on LiDAR, explains the whole system.
  - Loop closure is exactly what multi-room walkthroughs need.
- **Status.** In progress (wave 5).

## D-021 Correct the LiDAR inward surface offset (0.68 cm per surface); pre-registered and validated on unseen rooms

- **Context.**
  - On ARKitScenes (iPad LiDAR vs FARO laser), wall-to-wall distances read about 1.4 cm short.
  - Ceilings read short too: only 3 of 5 rooms within the 1.5 cm gate.
- **Investigation.** `docs/modules/lidar_bias.md` tested five hypotheses with evidence:
  - **H1 range-dependent depth scale: rejected.** The offset is flat across ranges.
  - **H2 trajectory scale: rejected.** Per-axis scales disagree.
  - **H3 registration: rejected.** ±1 cm / ±0.3° moves the result by ≤ 0.2 cm.
  - **H4 our estimator: minor.** All 12 estimators read short.
  - **H5 the laser: rejected.** Five scans agree, and planes fit to 2–5 mm.
  - **Cause.** Apple's LiDAR surfaces sit about 0.68 cm *into the room* compared with the laser, like the hook
    offset on a tape measure.
- **Decision.** Add 2 × 0.68 cm to interior distances and ceiling heights (LiDAR tier only), and carry the
  correction's own uncertainty in the interval: ±1.6 cm at 95%, widened to about ±2.1 cm because the iPhone itself
  is not validated. Switch: `--no-bias-correction` or `FLOORPLAN_LIDAR_BIAS=off`.
- **Evidence, pre-registered.**
  - The value and the predicted result were fixed on 3 development rooms before 2 unseen rooms were run.
  - Hold-out mean error −1.25 → **+0.11 cm** (predicted 0.0 ± 0.7).
  - Mean absolute error 1.41 → 0.71 cm.
  - Within 1 cm: 20% → 60%.
  - Ceiling heights: all 5 rooms within 0.7 cm (before, 3 of 5 within 1.5 cm).
- **Honest limits.**
  - The correction removes the average offset, not the room-to-room scatter. Two widths that were already right got
    worse (by +1.5 cm).
  - Spread after the fix: 1.05 cm, against the 0.7 cm predicted.
  - Only iPad was validated. The sample data is iPhone.
- **Part 4.** This is a complete fix-loop rehearsal: failing number, root cause with evidence, pre-registered
  prediction, shipped fix, measured result. The declared fix loop targets the worst gate of the final benchmark.

## D-022 Photo protocol tweaks from the v2 test

From the photo-tier v2 evidence (`docs/modules/photo_tier.md` v2):
1. **Always turn clockwise.** When the code had to guess the direction, it was wrong in 2 of 13 rooms.
2. **Keep the same tilt for every shot.** A tilt difference of 20° or more matched only 24% of the time, against
   100% under 10°.
3. **7 spin shots (about 50° steps) in rooms with one door, 6 with two.** At 60° steps the overlap was only about 13%,
   linkable 0–36% of the time; about 26% overlap is linkable 60–75% of the time.
4. **Doorway pair:** on the threshold, both photos within 5 s, a full turn.
5. **Keep EXIF.** It gives the capture times for pairing and the focal length for scale. Without EXIF focal length,
   scale was 22% off.

Plus, from damage v2: paint the staged stain **clearly** brown/yellow. Pale simulated blotches were missed. Applied
to `docs/CAPTURE_PROTOCOL.md` and `docs/HOUSE_CAPTURE_GUIDE.md`.

## D-023 Damage reporting has two tiers: confirmed (drives scope) and review (flagged, no scope)

- **Context.** The first integrated run on the clean `single_room` produced one 10 cm "crack" at confidence 0.17. It
  was above the detector's 0.10 cutoff, so it generated scope items: crack fill plus a 4.8 m² repaint. The brief
  penalises confident garbage, and a phantom scope line costs an insurer or a homeowner money.
- **Options.**
  - (a) Raise the detection cutoff. That loses true faint cracks.
  - (b) Tune the detector further. There is no time, and one apartment is a weak basis.
  - (c) **Two tiers**: everything detected is reported, but only "confirmed" detections (confidence ≥ 0.35) create
    scope items and concealed-damage flags. "Review" detections are listed for a human.
- **Decision.** (c), in `floorplan/damage/__init__.py` (`CONFIRM_CONFIDENCE = 0.35`).
- **Why the threshold.**
  - Damage v2 measured true stains at 0.37–0.98 and true cracks at 0.18–0.82, and the false crack scored 0.17.
  - At 0.35 every true stain is confirmed. Faint true cracks drop to "review": still visible, just not auto-quoted.
  - This is a precision-first choice for money-relevant outputs.
- **Evidence.** The clean-capture false crack becomes `review` with 0 scope items. The threshold must be checked on
  the real staged decals (own capture).

## D-024 Reject "bars" (handles, edges, grout lines) as cracks with a dominant-line test

- **Context and evidence.** See I-003: a wardrobe handle (20 cm straight vertical bar with stand-offs) was accepted
  as a crack at confidence 0.61, because overall straightness (0.74) sits between the existing rules.
- **Decision.**
  - Fit the single best straight line to the crack skeleton (RANSAC).
  - If ≥ 70% of skeleton pixels lie within max(3 mm, 1.5 px) of it, **and** it is within 8° of horizontal or
    vertical, reject the candidate as a bar.
- **Why.**
  - Physical: cracks zig-zag; manufactured edges are straight and axis-aligned.
  - Works at every tier: no depth needed.
- **Trade-off.** Perfectly straight, axis-aligned hairline cracks are not reported. We accept this: such lines are
  indistinguishable from joints and edges in images alone.

## D-025 Protruding-object check, made local to the detected line

- **v1** compared the whole bounding box with a ring around it.
  - It rejected a 2.3 cm "relief", but that came from a real handle elsewhere in the box.
  - It also **wrongly rejected 4 of 9 injected cracks** (crack recall 0.60 → 0.33) that sat near furniture, skirting or
    handles.
- **v2** compares the median height of LiDAR points **on** the line (within 6 mm of its skeleton) with points
  **beside** it (1.5–4 cm). It abstains when there are too few points.
  - A flat crack next to furniture stays flat relative to its own sides, so it is kept.
  - A protruding object on the line is rejected (thresholds: LiDAR 1.5 cm, learned depth 4 cm).
- **Evidence.** Crack recall restored to 9/15, with 0 false positives on injected and clean sets at τ ≥ 0.10
  (`outputs/damage/v2d`).

## D-026 Cracks need stronger evidence than stains before they generate scope (0.65 vs 0.35)

- **Context.** After D-024 and D-025 one clean-capture false crack remains: confidence 0.61, on a wood-grain wardrobe
  front that the plan treats as a wall.
  - Along the line itself the LiDAR shows no relief (−0.1 to +0.03 cm). It is a dark straight-ish line on furniture: a
    door gap, groove or grain.
  - The "textured surface" rule did not fire: too few ridge candidates.
- **Evidence for the class split.** Every false candidate on the clean captures so far is a *crack* (line-like
  features are common in clean homes). No false *stain* has appeared at any tier.
- **Decision.** Confirm thresholds per class: water stain 0.35, crack 0.65.
  - Below the threshold a detection is still reported, as `review`.
  - Only confirmed detections create scope items and concealed-damage flags.
- **Trade-off (from the v2d sweep).**
  - At 0.65, about 0.27–0.33 of injected cracks are *confirmed*; the rest are found but marked for review.
  - Precision of confirmed = 1.00 on everything we have.
  - For a money-relevant output we prefer this to quoting phantom crack repairs.
- **Open.** A furniture-vs-wall surface classifier (material/texture) would let us lower the crack threshold. Not
  built.

## D-027 LiDAR bias correction depends on corner type (convex +b, reflex −b); the no-correction path still widens

- **Found by** the ARKitScenes plan-level benchmark (`docs/modules/benchmark_arkitscenes_accuracy.md`).
  - The first version added +2b to every wall length.
  - Moving every face outward by b lengthens a wall by b at a convex corner but shortens it by b at a reflex
    (inward) corner.
  - Walls with a reflex end were not short raw (−0.55 cm), and the blanket +2b pushed them to +0.81 cm.
- **Decision.**
  - Shift each wall length by b × (s_start + s_end), with s = +1 at a convex corner and −1 at a reflex corner. The
    full two-surface uncertainty is kept even when the shifts cancel.
  - `--no-bias-correction` now calls the same code with `enabled=False`, so the interval is still widened one-sided
    and covers the truth.
- **Evidence.**
  - Unit test on an L-shaped room: the two walls meeting at the reflex corner shift 0, all others +2b.
  - The benchmark agent's offline measurement: walls within 1.5 cm of the laser went from 82% to 94%.
- **Perimeter and area** were already exact. For a simple orthogonal polygon, convex corners minus reflex corners
  = 4, so ΔP = 8b and ΔA = bP + 4b².

## D-028 Video protocol tweaks from the video-tier v2 evidence

From `docs/modules/video_tier.md` v2:
- **Keep away from blank walls.** Every visual-odometry restart happened on blank walls 0.3–1 m away. Stay ≥ 1.5 m
  from walls and keep a corner or the floor line in view.
- **Turn slowly.** The sample turned at a median 23–25°/s, with bursts of 66–84°/s. Cap it at about 20°/s.
- **Pause at doorways.** No join across a tracking restart could be verified without views of the doorway, so pause
  2 s facing into the next room.
- **Show the sheet properly (withdrawn by D-067: no sheet).** Camera height above the video's own floor scatters
  ±4–6%. Keep the sheet fully in view from two spots.
- **Pitch slightly down.** Gravity estimation broke at +6° median pitch.
- **One lens, no stabiliser modes.** A lens switch or crop mid-clip breaks the single-focal-length model. Landscape is
  about 62° wide against about 37° in portrait.
- **Close the loop.** Re-film the first view at the end.

Applied to `docs/CAPTURE_PROTOCOL.md` and `docs/HOUSE_CAPTURE_GUIDE.md`.

## D-029 Deterministic drift solve: single-threaded ICP, parallel loop candidates

- **Problem.** Running the same command twice gave a different plan: one room was 13.99 m² in one run and 11.46 m²
  in the other.
- **Root cause.** Open3D's multithreaded ICP sums its equations in a run-dependent order. That changes edge values
  by 2–4e-15, and the pose solver amplifies this to 0.005–0.023 mm pose differences.
- **Decision.** Run each ICP single-threaded and parallelise across loop candidates in Python threads.
- **Evidence.** Three runs are bit-identical, across processes and worker counts. The cost is about 40% more drift
  runtime.
- **Also.** Fused TSDF points are now sorted, so scene alignment no longer varies at the 1e-10 level.
- **Why it matters.** Deliverable 4 requires regenerable numbers; a plan that changes between identical runs is not
  reproducible.

## D-030 The plan extractor includes enclosed, partly seen floor (S-3); the multi-offset vote is rejected

- **Problem.** With deterministic poses, a 2.51 m² patch of unobserved floor (probably under a bed) still flipped in
  or out of a room. Its coverage, 0.535 or 0.457, straddled a 50% rule.
- **Decision (S-3).** Include a partly seen region (≥ 25% covered) when it lies inside the room's own measured walls
  and opens onto the room without a wall in between.
- **Evidence.** floor_only footprint is 61.90 m² on 3 of 3 runs, against 61.90 / 59.37 / 61.90 before.
- **Rejected.** The multi-offset majority outline:
  - the four grid-offset builds agree on only 59–73% of walls;
  - the vote picked the shrunken outline;
  - repeatability fell from 14.9% to 11.6%.
- **Caveat.** The S-3 thresholds were set while looking at this one case.

## D-031 A water stain must lie on the wall surface

- **Problem.** A confirmed false stain (confidence 0.73) on the clean floor_only fired a "window seal leak" flag and
  2 scope items.
- **Root cause.** The gold lid of a black jar on a bathroom shelf, standing 3.1 cm proud of the wall:
  - 68% of its points are more than 1.5 cm proud;
  - the wall orthophoto accepts anything within about 9.5 cm of the wall plane.
- **Decision.** Compare the median height of the points inside the candidate with the local wall level, measured
  from near-plane points at least 2 cm outside it, and reject candidates off the surface:
  - in front of it: an object;
  - behind it: glass, a mirror or an opening.
- **Evidence.**
  - Integrated runs on all three clean captures: 0 confirmed detections, 0 scope items.
  - Injection recall is unchanged.
  - This corrects D-026's claim that no false stain had occurred: one did, and it is now handled.

## D-032 A missing floor must never crash a cold run: estimate it, say how, or return an explained empty plan

- **Problem.** The video tier on with_ceiling crashed in the extractor: `float(None)` on `floor_y`.
  - The video front-end reported `floor_not_found`.
  - Its gravity estimate broke on that capture (+6° median camera pitch): the camera path sits about 17 m "below" the
    origin, so there is no floor in that frame.
- **Decision.** `scripts/run_capture.py` calls `floorplan/plan/floor_fallback.py` when a front-end reports no floor.
  1. **Points cue:** the lowest dense layer of points 0.6–2.5 m below the cameras, with surface normals ignored
     (σ 3 cm).
  2. **Camera-height prior:** chest height 1.40 ± 0.15 m below the median camera.
  3. The source is written to `plan.meta.floor_fallback`.
  - When the reconstruction itself is broken, the extractor returns an empty plan with its coverage warning, and
    JSON, render and DXF all handle an empty plan (tested).
- **Why.** At the walk-in the founders run the pipeline cold; a stack trace is the worst outcome. An empty plan with
  a stated reason is honest.
- **The root cause is upstream.** The video gravity estimate fails at upward pitch. The protocol tweak (D-028) asks
  for a slight downward pitch.

## D-033 Drop physically implausible openings (doors/passages < 0.45 m, windows < 0.25 m)

- **Evidence.** The single_room LiDAR plan contained a 0.01 m "passage", an extractor artefact. Every other opening
  across the 9 benchmark plans is 0.57–2.07 m (doors), 0.73–1.26 m (passages) or 0.41–0.99 m (windows).
- **Decision.** After extraction, drop openings narrower than any real one.
  - The narrowest real doorway, a closet door, is about 0.45 m.
  - Each drop is recorded in `plan.meta.dropped_openings` with its reason, and wall and adjacency references are
    cleaned.
- **Why.** The opening gate counts phantom openings as misses, and a 1 cm doorway is not something a homeowner should
  see. The filter is a physical bound, not a fit to this capture.

## D-034 Video plans from unverified segment joins get a 20% relative-sigma floor and a low-reliability flag

- **Problem.**
  - The final benchmark found the video tier confidently wrong on the long walkthroughs: footprint −37.9% (floor_only)
    and −36.8% (with_ceiling) against the LiDAR plan.
  - Its intervals covered the reference only 31–38% of the time.
  - Yet the video front-end itself reported `whole_scene_consistent=False`: segments joined across tracking restarts
    without verified geometry.
- **Decision.** In `scripts/run_capture.py`, when `whole_scene_consistent` is False:
  - floor the tier's relative scale sigma at 20%, so the 95% interval is about ±40%, matching the measured error;
  - write `plan.meta.reliability = "low: ..."`.
  - Consistent videos keep their measured sigma.
- **Why.** The brief: "confident garbage on thin input caps your total score". When we know the scene may be wrong,
  the intervals must say so. Honest width beats false precision.
- **Better fix (open).** Verified joins: doorway pauses in the capture protocol (D-028) and geometric join checks.
  That would let the floor drop again, on evidence.

## D-035 A simulator for walk-in rehearsal: Isaac Sim renders colour only; all geometry comes from the same USD triangles

- **Why a simulator now.** The real house capture was postponed, and 30% of the score is a cold run on an unseen
  space captured by someone else following our page. A simulator rehearses that on many houses and "people" with
  exact ground truth, and gives the fix loop (25%) cheap before/after runs. Sim results are labelled simulated and
  never replace the real benchmark.
- **Scene.** InteriorAgent (Hugging Face `spatialverse/InteriorAgent`, Isaac-Sim-ready USD, own terms of use).
  - Development scene: `kujiale_0065`, a true 1 BHK (living, bedroom, kitchen, bathroom, balcony; 59 m²). It was
    picked because it is the smallest download (0.2 GB) of the 25 scenes.
  - Generalisation scenes: `kujiale_0038` (another 1 BHK) and `kujiale_0022` (2 BHK).
- **Split of work.**
  - Isaac Sim (RTX, headless) renders only the colour frames (`sim/render.py`).
  - Depth, ground truth and the walkable map are computed from the same triangles with Open3D ray casting
    (`sim/scene_geom.py`).
  - Phone formats and noise are added afterwards in the main venv (`sim/emulate.py`).
- **Evidence it is consistent.** Ray-cast depth vs Isaac's own `distance_to_image_plane` on four views: **0.00 mm at
  the 99th percentile** with principal point (W/2 − 0.5, H/2 − 0.5). The same pose code drives both.
- **Alternatives rejected.**
  - Rendering depth in Isaac too: twice the GPU time and storage, and every noise variant would need a re-render.
  - Blender: no ready-made furnished houses with semantic labels.
  - Habitat/HM3D: real scans, but holes and no clean architectural ground truth.

## D-036 Simulator ground truth = the visible surfaces, measured like a tape (ray casting), not the scene's room outlines

- **Problem.** `rooms.json` holds the designer's outlines. In kujiale_0065 the visible surfaces differ:
  - a TV-wall cladding sits 1.8 cm in front of the outline on part of the east wall;
  - a wall panel sits 0.3 cm in front on the west wall;
  - tray ceilings are 2.80 m in the middle and 2.61 m at the perimeter.

  At cm level, those differences decide pass or fail.
- **Decision (`sim/scene_gt.py`).**
  - **Walls.** Each outline edge is moved to the median wall surface hit by horizontal rays (0.8–1.7 m high, every
    5 cm). Corners are intersections of the moved edges, and wall length is corner to corner. The surface spread is
    kept in the notes. Edges that are almost all doorway use rays above and below the opening.
  - **Ceilings.** Upward rays near the most open point of the room, as the tape guide says ("near the middle"); the
    distinct levels are listed.
  - **Doors.** Clear width between the frames at ~1 m. Across the wall thickness the width changes (wall hole,
    jamb lining, sliding track, a closed leaf and its lock), so the value is the narrowest width that ≥ 20% of the
    rays agree on. The wall hole width is noted. Example: bathroom 0.710 m between frames vs a 0.800 m hole.
  - **Windows.** The wall hole (glass ignored): width, height, sill.
- **Format.** The same CSV as the tape GT (`gt_eval.read_gt`). Walls are W1..Wn clockwise from above, W1 being the
  wall with the room's main door (breadth-first from the entrance).

## D-037 Interactive capture: a pure-Python phone rig plus a thin Isaac layer; photos on P, never Space

- **Request.** The operator drives a phone through the house (WASD, tilt) and records only what a phone would record.
- **Decision.**
  - `sim/rig.py` holds the motion, collision, recording and photo-folder logic. It needs no Isaac Sim and is shared
    with the scripted sessions and the tests.
  - `sim/teleop.py` binds keys, the viewport and a HUD with a minimap.
  - The session file stores the intended path and the shots. Rendering is a separate pass, so the operator drives at
    interactive speed and the same drive can be re-rendered at any quality.
- **Details that matter.**
  - Photos go to the room in front of the camera, so doorway pairs sort themselves as the protocol requires.
  - Walls block motion; furniture only warns.
  - Space is not used: in Isaac Sim it plays the timeline, and InteriorAgent furniture carries rigid-body physics.
    The tool also stops the timeline if anything starts it. `render.py` disables Replicator's capture-on-play for
    the same reason.

## D-038 Human imperfection is part of every simulated capture

- **Request.** Video at slightly different angles and heights throughout the run, because people never hold them
  constant. The overlap between consecutive photos should vary too, with changing height and angle.
- **Decision.**
  - **Walks** (`rig.humanize`, applied at render time to scripted and keyboard paths alike): mean-reverting drift
    of height (σ 7 cm, about 6 s), tilt (σ 5°), roll (σ 2°), aim (σ 1.5°) and side sway (3 cm), plus walking bob and
    hand tremor (0.35°).
  - **Photo spins** (`auto_capture.py`): each turn step leaves a different overlap, drawn from 10–50% of the view.
    Height and tilt drift from shot to shot (correlated), the feet shuffle (6 cm), and each shot gets its own small
    aim error (`rig.humanize_photo`).
- **Why.** A pipeline tuned on perfectly level, evenly spaced captures fails at the walk-in. These magnitudes are
  what people show when they try to follow "keep the same tilt".

## D-039 Phone emulation: the walk-in's iPhone 15 by default, with the real sample's sensor statistics

- **Decision (`sim/emulate.py`).** The default profile is `iphone15`, because the walk-in uses an iPhone 15 or newer.
  A `nord` profile (my own phone; no video focal key) is also available.
- **LiDAR.**
  - Each depth pixel averages 3×3 ray-cast sub-rays (edges blend), with noise from the measured σ(range)
    (`lidar_noise.json`), spatially correlated.
  - Confidence fades with range (2.5 → 4.5 m), at grazing angles and at depth edges.
  - Glass passes the pulse. A mirror reflects it into a "room behind the mirror" at high confidence.
- **Check.** High-confidence points land 5–9 mm (median) from the true surfaces.
- **Confidence calibrated on the real sample.** High-confidence share by range in the three real Stray recordings
  (every 40th frame) vs the emulation after calibration:

  | Range | Real | Emulated |
  |---|---|---|
  | < 1.5 m | 0.93–0.95 | 0.96 |
  | 1.5–2.5 m | 0.87–0.91 | 0.91 |
  | 2.5–3.5 m | 0.67–0.73 | ~0.75 |
  | 3.5–4.5 m | 0.56–0.64 | ~0.6 |

  The first version was far too pessimistic at close range (0.53). Its no-return pixels were written as depth 0,
  whereas real ARKit depth maps are dense (no zero pixels; far or unknown pixels get a value at confidence 0).
- **Poses.** ARKit-like (Y up, origin and heading at the first frame, OpenCV camera axes) with VIO drift: 0.25°/√m
  yaw, 0.6 cm/√m position, 0.4% scale. This gives ~10–15 cm revisit gaps over a flat; the real sample had 14 cm.
- **Images.** Auto exposure with gain-dependent noise, rotational motion blur, JPEG/H.264/HEVC.
- **Metadata as phones write it.** EXIF `FocalLengthIn35mmFilm` is an integer using the standard diagonal
  definition, and iPhone videos carry the QuickTime 35 mm-equivalent key. This exposed issue I-009.
- **Real-data parity.** The output folder is the real capture layout, so `scripts/process_own_capture.py` runs
  unchanged. It now also runs `lidar/<take>/` Stray folders.

## D-040 Scripted sessions follow the capture page literally, and doing so found three ambiguities in it

- **Decision.** `sim/auto_capture.py` turns `docs/CAPTURE_PROTOCOL.md` into paths: per-room spins and doorway pairs
  for photos, and one walk route for video and LiDAR (the LiDAR page already says "walk the same route as the
  video"). Room order is depth-first from the entrance room.
- **Ambiguities found** (the brief: "if the page is ambiguous, the capture you get reflects that"):
  1. **Photo count in rooms with 3+ doors** is unspecified (only "7 with one door, 6 with two"). The rule implied by
     the 8-photo cap is N = 8 − doors, minimum 4. The page now needs to say so.
  2. **Doorway pairs on the way back.** Taking them again would overflow the 8-photo cap of hub rooms. Rule: a
     doorway pair only the first time you pass a door.
  3. **Duration.** Followed literally (a ≤ 20°/s full turn per room, two sheet views, doorway pauses), a 5-room flat
     takes ~6–7 min, not "2–4 min". Rule: about 1 minute per room.
- **Merged sweep.** The LiDAR ceiling and floor sweeps are folded into the room's slow full turn (the tilt swings
  between floor line and ceiling line while turning). This covers both without extra time, and gives the video tier
  ceiling views too.

## D-041 Simulated walks are rendered at 10 fps

- **Measured cost.** 0.40–0.43 s per 1920×1440 RTX frame on the RTX 2000 Ada. Most of it is Replicator's
  render-and-wait step (a bare `app.update()` is 0.08 s, but it does not refresh the annotator). Two render products
  per step gave only 27% more throughput, and recreating them crashed Replicator.
- **Decision.** Walks render at 10 fps (a 7-minute walk is ~28 min to render). Video and LiDAR use the same frames.
- **Why it is acceptable.**
  - LiDAR keyframes are chosen by motion (5 cm / 5° / 0.5 s), and 10 fps at 0.45 m/s still gives one candidate every
    4.5 cm.
  - DPVO's auto stride then sees every frame; it still needs testing at 10 fps.
  - Use `--fps 30` for a final check.

## D-042 The scorer reports the whole-property stitch gate (rooms, adjacency, overlaps, footprint)

- **Gap.** Part 2's photo-tier stitch row ("one stitched plan with correct adjacency and no room overlaps; footprint
  within ±8%") was not computed by `floorplan/benchmark/gt_eval.py`. It scored walls, openings and ceilings only.
- **Decision.** `evaluate()` now also reports:
  - per-room floor area against GT `area` rows;
  - `gates.stitch`:
    - matched vs GT rooms, and the missing rooms;
    - total footprint error;
    - adjacency found / missing / wrong, against the door graph in the GT notes (`rooms A+B`);
    - room-polygon overlaps (> 3% of the smaller room, with shapely);
    - pass/fail with the ±8% footprint tolerance.
- **Where the data comes from.** The simulator's GT writes areas and door room pairs. For a tape GT, measure the room
  areas from the sketch, and the check runs on whatever exists.

## D-043 Emulated phone videos are encoded with the GPU's hardware encoder (NVENC)

- **Measured.** CPU encoding (x264 and x265 at 1920×1440, alongside six depth workers and an Isaac render) held
  emulation at 0.42 s/frame: about 29 min for a 7-minute walk.
- **Decision.** `h264_nvenc` / `hevc_nvenc` (quality-targeted, `-cq`), with libx264/libx265 kept as a fallback
  (`emulate.USE_NVENC`).
- **Why it is also more faithful.** Phones encode with hardware encoders too. The pipeline only sees an ordinary H.264
  (.MOV) or HEVC (`rgb.mp4`) file.

## D-044 Photo protocol: stand far from the walls, and aim doorway shots at the room's open middle (found in sim)

- **Evidence (simulated, kujiale_0065).** Two rounds of the scripted person:
  - **v1** stood at the most open spot, which was the dining end of the 8 m hub living room. Four spin photos covered
    about 220°.
  - **v2** stood at the room's centroid, but the L-shaped outline (hallway niche) pulls the centroid to 0.8 m from a
    marble wall. Three of the four spin photos were wall close-ups.
  - In both, doorway back-shots aimed straight across a narrow hallway showed only a wall 0.7 m away.
- **Decision.**
  - **Scripted person (v3).** Stands where the distance to the walls is largest (capped at 1.6 m), clear of furniture,
    then nearest the centroid. Doorway shots aim at the open middle of each room.
  - **Capture page and house guide.** "Stand in the open middle, as far from every wall as you can (ideally ≥ 1.5 m)"
    and "aim the doorway shots at the room's open middle".
- **Why.** Close-ups of blank walls carry no features to match and no geometry to fit. These are protocol details a
  non-engineer can follow, and they decide whether the photo tier gets usable input.

## D-045 The scripted walker looks toward open space when its path runs at a near wall

- **Evidence (simulated).** On the first 1 BHK walk, 10.6% of frames were filled by something under 0.6 m away (walls,
  the kitchen range hood, curtains, cabinets), and 19.3% under 1 m. The page says "never fill the picture with a blank
  wall", which a person does by looking along the room.
- **Decision.** When the view along the path is blocked within 1.5 m, the walker looks toward the most open direction
  within ±60° (2D ray marching on the walkable map, slightly preferring the path direction).
- **Result on the same route.** < 0.6 m: 10.6% → 8.8%; < 1 m: 19.3% → 14.3%. The remainder comes from full turns in
  small rooms (a 2.2 m kitchen), which real people cannot avoid either, so it is kept as realistic difficulty.
- **Scope.** Applies to the walks rendered after 11:00 (kujiale_0038, kujiale_0022). The kujiale_0065 walk keeps the
  older, sloppier walker; it is a useful robustness case and is labelled as such.

## D-047 Simulated captures use the real phone formats

| Tier | What the sim writes | Real-phone reference |
|---|---|---|
| Photos | **4032×3024, 4:3 landscape** JPEG, EXIF 26 mm (35 mm equivalent) / 6.24 mm, capture times | iPhone 15 default 12 MP photo; Android phones also default to 4:3 |
| Video | **1920×1080, 16:9** H.264 `.MOV` with the QuickTime 35 mm-equivalent key | iPhone video default (16:9) |
| LiDAR | Stray Scanner folder, 1920×1440 HEVC + 256×192 depth | ARKit's format |

- **Photos.** They were 2000×1500 until 10:55. The pipeline downsizes to ≤ 1024 px anyway, but the sheet detector and the
  JPEG statistics now see real-size files.
- **Protocol.** The capture page and the house guide now say "keep the default 4:3 photo format" (16:9 is the video
  format). The page never said it, so a 16:9-configured phone could have produced crops we never tested.
- **Remaining deviation: frame rate.** Walks render at 10 fps for throughput (D-041), while phones record 30 fps.
  One house (kujiale_0022) is rendered at the real 30 fps as the full-fidelity check of the video tier.

## D-048 The photo tier converts EXIF 35 mm-equivalent focal length with the standard diagonal definition (fixes I-009)

- **Root cause (I-009).** v1 used f = f35 / 36 mm × long side. The 35 mm equivalent is defined on the 43.27 mm
  diagonal, which the video tier already uses. On a 4:3 photo the v1 rule gives a focal length 3.8% too short, so the
  assumed field of view is too wide.
- **Evidence it matters (simulated, exact ground truth).** Same 40 iPhone-15-like photos of the 1 BHK (kujiale_0065),
  only the rule changed:

  | Metric | v1 rule | diagonal rule |
  |---|---|---|
  | Footprint error | −46.9% | −13.2% |
  | Wall median error | 28.2% | 16.7% |
  | Walls within ±8% | 25% | 45% |
  | 95% interval coverage | 73% | 89% |

  The 3.8% focal error is amplified, not absorbed: it changes MoGe-2's metric depth (field of view is an input) and the
  per-room layout fit.
- **Why the diagonal rule is right for the walk-in phone.** iPhone EXIF values match it. Example: iPhone 13 Pro main
  camera, 5.7 mm lens on a 9.57 mm-diagonal sensor: 5.7 × 43.27 / 9.57 = 25.8, written as 26. The width rule would
  give 27.
- **Consistency.** `scripts/make_photo_folders.py` (the phone-like test photos made from the sample) now writes EXIF
  with the same diagonal rule. Before, both sides used the 36 mm rule, so the error cancelled and the benchmark could
  not see it.
- **Before/after reproducible.** `--photo-param f35_rule=36mm` restores v1.
- **Remaining photo-tier error (−13% footprint).** It comes from the per-room layout fit: furniture fronts (wardrobe,
  sofa back) and through-door walls taken as walls. That is the next target.

## D-049 Photo tier: a room ceiling outside 2.0–4.0 m is not a ceiling; unseen ceilings use the whole-scene value, inferred

- **Evidence (simulated, exact ground truth).** The spin photos are aimed slightly down (protocol), so few see the
  ceiling. Their downward-facing points are cabinet and lamp undersides, yet they were reported as *measured* room
  heights: bedroom 1.73 m, kitchen 0.90 m, bathroom 1.82 m, against true 2.80 / 2.56 / 2.59 m. That is confident
  garbage, which the brief penalises.
- **Decision.**
  1. A per-room height outside 2.0–4.0 m is rejected (the LiDAR extractor already has `ceiling_min_h_m` = 1.9 m).
  2. A room without a plausible height takes the whole scene's ceiling minus floor (all photos together, which do see
     ceilings), status **inferred**, σ = 10 cm.
- **Result (same photos).** Every room is 2.745 m inferred. Errors are −5 / −5 / +19 / +15 / +15 cm, all inside the
  intervals, where before they were −1.6 to −0.2 m outside them. The ceiling gate (1.5 cm) stays out of reach for
  photos, and the plan says so honestly instead of claiming a measurement.

## D-046 Headers over wide openings separate rooms (coplanar header-band faces), on by default

- **Problem (I-010, found in simulation).** Rooms joined by an opening wider than any door neck (> 1.3 m: sliding doors,
  wide doorways) merged into one room. In the sim 1 BHK, the kitchen (2.12 m sliding door) and the balcony (2.29 m)
  merged into the living room.
- **Physics.** A room boundary is a wall that reaches the ceiling, and an opening is a hole in it, with a header
  above, below ~2.0–2.4 m. True open plan has no header.
- **Rule (`plan/beta/lines.header_cut_mask`).**
  1. Detect face lines in the 2.05–2.75 m band. Parallel peaks within 20 cm merge (frame, track, drift doubling).
  2. Keep a header line only if it lies within 12 cm of a low-band (0.3–2.0 m) wall line with the same facing, and
     that wall line has at least 30 cm of wall somewhere along it.
  3. It must span a stretch where that wall is open for ≥ 1.3 m.
  4. Draw the cut there, leaving a 0.8 m door-sized gap in the middle. The neck rule then separates the rooms, and
     the opening is measured jamb to jamb using the header's open span as the search width (`_necks_on_headers`).

  Faces that are not in a wall's plane are ignored: tray-ceiling soffit edges inside the room, wall cabinets,
  wardrobe tops.
- **Why not paired faces (first attempt).** The header's far side is rarely seen: from inside a small room it is above
  and behind the operator who just walked in. Pairing found nothing.
- **Results.**

  | Case | Before | After |
  |---|---|---|
  | Sim 1 BHK, LiDAR rooms | 3 of 5 | 4 of 5 (kitchen separated) |
  | Kitchen sliding door width (true 2.119 m) | missed | **2.124 m (+4.6 mm)** |
  | Openings within 2 cm | 57% | 75% |
  | Wall median error | 49% | 29% |
  | Real LiDAR scenes (3 sample captures, 5 laser-scanned ARKitScenes rooms) | — | **identical** rooms, walls and openings |

- **Not solved.** The balcony stays merged. No surface at all was reconstructed above its opening (curtains and a
  pelmet hang there), so there is no header evidence. Next options (I-010): the protocol's doorway pause as a door
  marker, or a ceiling-height step.

## D-050 Photo protocol A/B in simulation: "about a quarter overlap" stays; "even circle" is not better (one seed)

- **Question.** Would spreading the N spin photos evenly over the full circle cover a hub room better than
  "overlap the previous photo by about a quarter" (which, with 4 photos in a 4-door room, covers only ~220°)?
- **Test.** Same 1 BHK (kujiale_0065), same person seed and standing spots; only the spin rule changed.

  | Spin rule | Footprint | Wall median error | Walls within ±8% | 95% interval coverage |
  |---|---|---|---|---|
  | Quarter overlap (page) | −13.2% | 16.7% | 45% | 100% |
  | Even circle | +25.1% | 14.9% | 29% | 86% |

- **Reading.** The even spin does see the living room's far end (5.1 m), but then over-sizes the room (41.7 vs 28.8 m²).
  The other rooms barely change. With one seed and mixed signs this is not evidence for a change, so the page keeps
  "about a quarter overlap".
- **Next.** Repeat over several seeds and houses before changing the page.

## D-051 The capture protocol, designed once from published practice and my review (v2)

- **Why redesign.**
  - My review of the first simulated captures found the camera inside furniture (bedroom wardrobe, kitchen
    range hood, balcony curtains) and too many photos for small rooms.
  - I wanted a floor pass then a ceiling pass, and less jitter.
  - The research (docs/CAPTURE_PRACTICES_RESEARCH.md, 31 sources) gave the industry practice to copy.
- **Protocol (docs/CAPTURE_PROTOCOL.md; scripted in sim/protocol_v2.py).**
  - **Photos.**
    - Normal rooms: turn on the spot from the open middle (Matterport, ZInD, magicplan, HorizonNet), 7 − doors photos,
      plus one ceiling photo.
    - Small rooms: from the doorway looking in plus from the far end looking back (CubiCasa, Hover, ZInD), plus one
      ceiling photo.
    - Doorway pairs once per door. At most 8 per room.
  - **Video and LiDAR.**
    - Walk forward, never sideways, about 1 m in from the walls, glancing into corners (CubiCasa, Hover). No standing
      turn in the middle of a room (CubiCasa).
    - Small rooms: step in and pan.
    - Pass 1 aimed at the floor line, pass 2 the same route aimed at the ceiling line (ARKitScenes' separate
      sequences; the provided sample's floor_only / with_ceiling captures).
- **Scripted person v2.**
  - **Obstacles.** Isaac Sim's occupancy map (PhysX colliders, `sim/isaac_omap.py`) united with every object's
    triangles at 0.10–1.90 m, door parts included, doorway passages re-opened, with a 25 cm body/phone margin. Routes
    use only that strict free space.
  - **Human imperfection** is applied when the path is made, and every pose is checked in 3D. Result: **no camera
    closer than 0.26 m to any surface** in all three houses (before: 5.5% of frames under 5 cm, i.e. inside furniture).
  - **Less jitter**: hand tremor 0.35° → 0.10°, walking bob 1.2 → 0.5 cm, slower sway and aim wobble. Slow
    height and tilt drift kept.
- **Bugs found on the way.**
  - The walk helper reset the tilt at every segment, so the ceiling pass was not looking up.
  - Standing turns before every leg added 200 s of pirouettes.
  - The obstacle raster used `cv2.fillPoly` with the even-odd rule, so a box's top and bottom faces cancelled.
  - Waypoints snapped to cells beside furniture.
- **Frame rate settled (D-041).** On the identical path, 10 fps gave 9 VO segments and 30 fps gave 14. Frame rate is
  not why the video tier fragments; 10 fps renders stay.

## D-052 Photo tier reads the v2 protocol: ceiling photos are recognised and measure the room height

- **Why.** Protocol v2 (D-051) adds one ceiling photo per room, and gives small rooms 2 positional shots instead of a
  spin. The v1 front-end put every non-doorway photo of a folder into one clockwise spin with ~60° steps, so the
  ceiling photo (tilted up, arbitrary yaw) would have been forced into the spin.
- **Decision.**
  - A photo whose levelled optical axis points up by more than `ceiling_photo_min_pitch_deg` (18°) is that room's
    ceiling photo (`protocol.photo_pitch_deg`); the last photo of a series from 10° (D-074). It is excluded from the
    spin prior.
  - Room height = the spin photos' camera height above their floor plane + the median height of the downward-facing
    points of the ceiling photo above the camera (`layout.ceiling_above_camera`), if inside 2–4 m. Else the D-049
    fallback (whole-scene ceiling, inferred).
  - Small rooms (doorway shot + far-end shot) fall below the 2-photo spin minimum and get no false spin prior. They
    are placed by feature edges and doorway links.
- **Status.** To be measured on the v2 simulated photo sets (exact GT).

## D-053 Photo tier: a small room is boxed from its threshold photo (protocol v2)

- **Protocol v2 small rooms** have no spin: one photo from the doorway looking in (often the doorway pair's photo
  into the room), one from the far end looking back, one ceiling photo.
- **Model.** The threshold photo measures the far wall and both side walls directly. The door wall is where the
  camera stands: its room-side face is about half a wall thickness (`door_threshold_inset_m` = 0.10 m) in front of the
  camera.
- **First result (simulated 1 BHK).** Kitchen area +7% (width 2.22 vs 2.19 m), bathroom +13%. Under v1 these were
  +68% and −57%.

## D-054 Photo tier: hub rooms use their doorway photos in the box fit; glass-walled small rooms fall back

- **Hub rooms.** Under the 8-photo cap, a room with 4 doors gets 3 turning photos, 4 doorway photos and 1 ceiling photo.
  The 3-photo spin leaves walls unseen (simulated living room: 0.86 m wide instead of 3.45 m). The doorway photos look
  in from every side, so they now join the fit with their pose-graph position relative to the placed spin photos
  (within 6 m). The refit is kept only if it measures more sides.
- **Balcony.** A doorway view of a balcony sees mostly its window wall (no wall points through glass). The far-end
  photo, looking back at the solid door wall, is used instead.
- **Protocol: bigger rooms get more photos.** Turning photos scale with area: about 4 under
  10 m², 5 up to 18 m², 6 above, within the brief's 8-photo cap (turning + ceiling + one doorway photo per door).
- **Result (simulated 1 BHK, v2 photos).** Living room area −90% → −13%, wall median 6.9%. The balcony stays wrong
  (+339%) with one far-end photo; see D-055.

## D-055 Photo protocol v2.1: every room seen in and out through its doors; small rooms left and right

- **Review (4 Oct, 14:20).**
  - "Every image is almost left facing; we should have right-facing images as well, in the kitchen and bathroom."
  - "Add a photo of the balcony from the hallway."
  - "There should always be photos from each room looking out and in, so that we can patch them together."
- **Protocol change.**
  - **Doorway pair at every door between two rooms**, also doors the route does not walk through. Each room is then
    seen *in* from each of its doors, and a cross-room feature match is only accepted between paired rooms
    (`features_need_pair`), so every true adjacency can now carry one.
  - **Normal rooms: the turning series starts facing the door you came in through.** Its first photo looks *out*
    through that door into the previous room, giving shared views with that room's photos, at no extra photo.
  - **Small rooms: 5 photos** (was 3).
    - Doorway photo turned left (the pair's photo).
    - Two far-end photos back at the door, turned left of it and then right of it. The door is in both: these are
      the *out* views.
    - The ceiling photo.
    - On the way out, a doorway photo turned right.
  - **Shallow rooms (< 2.2 m deep from the door: balcony).** One more photo, taken last from ~2 m back in the next
    room through the door ("balcony from the hall").
  - The brief's 2–8 photos per folder still holds:
    - simulated 1 BHK: living 8, bedroom 7, kitchen 5, bathroom 5, balcony 6;
    - the 2 BHK bathroom with two doors: 6.
- **The ceiling photo ends a room's turning series.** The 1–2 photos after it (second doorway photo, balcony from
  the hall) come from other spots, so they are kept out of the spin group and its "same spot" prior.
  - Guard for human captures: this applies only with ≥ 2 photos before the ceiling photo and ≤ 2 after it
    (`extra_photos_max`).
  - v1 captures have no ceiling photo and are unchanged.
- **Pipeline.**
  - A v2 room with ≤ 2 spin photos plus a ceiling photo is a small room: the far-end pair looks back at the door and
    is not a spin of the room. It is boxed from its threshold photo (D-053).
  - The second doorway photo joins that fit when the pose graph puts it on the same threshold (≤ 0.6 m,
    `same_spot_m`).
  - The glass fallback (D-054) now uses both far-end photos as a 2-photo spin, refit with the doorway and extra photos.
- **Timing.** Simulated capture times now come from the walking distance between spots (0.8 m/s) plus aiming time.
  A doorway pair (a turn on the spot, 2.3–3.2 s) is always quicker than walking to the next spot, as with a person.
  On the 1 BHK, every other room-to-room gap is ≥ 6.8 s against a pairing limit of ~5.6 s.
- **Status.** Datasets regenerated (`outputs/sim/k65v2`, `k22v2`, `k38v2`, `k65v2_dim`; the v2 photo sets are kept
  in each `old_v2/`). Scores to follow.
- **Addendum (15:00).** A small room's ceiling photo taken at the far end pointed, at random, at the wall
  0.4 m away (a close-up of the window corner). It now aims along the long side towards the side with room in front
  (< 1.2 m on one side → the other). Same random draws, so only those photos changed: k65 2 photos, k22 1 photo,
  re-rendered with `render.py --only`.

## D-056 Photo tier: the A4 sheet was never used; wired in, and the detector fixed for 12 MP photos (off by default since D-067)

- **Found (4 Oct, 15:00).** Every photo-tier run reported `sheet_status: floorplan.scale has no observe_image()`.
  The adapter between the photo tier and the sheet module was never written, so the sheet in every room was ignored
  and the scale was MoGe-2's alone.
- **Second bug, in the detector.** Candidates are sorted by area and only the 50 largest are evaluated. A 12 MP
  photo of a furnished room has more than 50 larger bright regions (tiles, panels, windows), so a clearly visible sheet
  was never evaluated (k65 balcony photo: sheet 400 px wide, nearest candidate 232 px off). Video frames (1920 px)
  rarely hit the cap.
- **Fix.**
  - `floorplan.scale.observe_image` detects on the full-resolution photo and turns the sheet into a depth correction
    (`depth_map_scale`).
  - The photo tier searches every non-ceiling photo in parallel.
  - The detector drops candidates whose closed-form rectified shape cannot be a sheet *before* the cap:
    - aspect outside 1.15–1.75, or corners > 15° from square;
    - with gravity, not within 25° of flat.

    The final checks are unchanged and much stricter.
- **Result (sim).**
  - Sheet height: +0.11% (balcony) and −0.6% (living room) against truth.
  - Detection is now 0.2–0.6 s per photo, was 2–3 s.
- **Open.**
  - Only 2 of 18 photos with a sheet in frame give a detection. On light marble the white sheet (albedo 0.88) is
    rejected as "darker than surroundings". This is a real-home risk too (white vitrified tiles).
  - The two readings disagree in the common frame (×1.056, ×1.155) because the pose graph rescaled those two
    photos (0.89, 0.76). The fused global scale ×1.10 oversized the rooms. To fix: the sheet should scale the room
    whose photo sees it, not the whole scene through a cross-room link.
- **Fix to the fix (same afternoon, from sim truth).**
  - **Evidence.** Each photo's MoGe-2 scale was compared with ray-cast truth (`depth_truth.json`):
    - raw spread between photos: MAD 5.6%, median 0.975;
    - the pose graph's per-photo corrections: MAD 11.3%, so worse than none;
    - the sheet ratio against true depth at the sheet pixels: 0.966/0.968 and 1.028/1.029.
  - **Change.** The sheet now enters the fusion as its photo's direct ratio, with MoGe-2's photo-to-photo spread
    (6%) added to its sigma.
  - **Result (k65).** Global scale ×1.100 → ×0.995 ± 2.9% (ideal 0.975).

## D-057 Photo tier: ceiling-photo heights scaled by their room, else fused with a residential prior

- **Evidence (sim truth, k65).** MoGe-2's metric/predicted ratio on the five ceiling photos is 0.59–1.18. The other
  photos have a median of 0.975. Kitchen and bathroom ceilings came out +50% and +33% with narrow intervals:
  confident garbage.
- **Tried: link each ceiling photo to its room's photos by feature matches.** The ratio of distances between matched
  3-D points needs no pose. It fails in practice. Tilted +40° against −12°, the photos share no verified matches,
  except the living room's (31 and 27), and those matches sit where the depth is invalid (windows). The code stays:
  it applies whenever matches exist.
- **Shipped.**
  - An unlinked ceiling photo gets a 25% sigma on its above-camera part.
  - The height is fused with a disclosed residential prior of 2.70 ± 0.25 m.
  - The plan prefers this per-room ceiling-photo height over the free-space room's ceiling points, which mix in
    every photo's depth scale (living room 3.56 m vs 2.80 m true). This applies only to v2 ceiling-photo heights,
    so v1 captures are unchanged.
- **Result (k65, same photos and caches; v2.1 before → after D-056 + D-057).**

  | | Before | After |
  |---|---|---|
  | Wall median error | 14.6% | 6.3% |
  | Walls within 8% | 30% | 75% |
  | 95% interval coverage | 87% | 97% |
  | Footprint | +5.2% | −14.9% |

  - Ceilings are 2.60/2.80, 2.69/2.80, 2.86/2.56, 2.81/2.59 and 2.60/2.60 m (predicted/true), all inside their
    intervals.
  - The footprint's earlier +5.2% was an accident: the ×1.10 scale hid the L-shaped living room's −31% (one
    rectangle; the dining end unseen and inferred).

## D-058 The plan step gets a hard time limit (a live run must never hang)

- **Found (4 Oct, 16:00).** An orphaned photo-tier run (old `k38_s1` set, started 13:02) sat inside
  `extract_plan` for 3 hours at 112% CPU. Its log stops at "scene saved", with no "plan:" line. In the walk-in test,
  where the pipeline runs cold in front of the assessors, that is the worst possible failure.
- **Guard (`scripts/run_capture.py`).**
  - The plan step runs under a SIGALRM time limit (`--plan-timeout`, default 300 s; normal runs take 30–60 s).
  - On expiry, the alpha extractor gets its own limit; if it fails too, the run exits with a clear error.
  - `run_report.json` records `plan_timeout`.
  - Tested: a busy Python loop is interrupted after the limit.
- **Root cause.** Being reproduced under a faulthandler time limit (`photo_hangtest`); see I-012.

## D-059 Video: upright rotation by per-frame votes, not by the median angle

- **Found.** The sim 1 BHK video came out as 0 rooms, all 36 segments untrusted.
- **Cause.** With the rotation tag at 0, GeoCalib voted 180°: median up-angle 1.33° for 180° against 1.39° for 0°. The
  frames were upright, so depth was computed on upside-down frames. GeoCalib reads almost any rotated frame as
  "nearly upright", so the medians tie on every capture; the real samples score 4.7–12° for every candidate.
- **Fix.** Use the per-frame vote counts: sim 0° wins 7 of 12; real samples 90° wins 7–8 of 12. Ties go to the
  container's value.
- **Result.** The sim video gives 5 of 5 rooms. It is still inaccurate: 28 VO segments, footprint +127%.

## D-060 Photo tier: walls are what an ADE20K segmenter calls "wall"

- **Model.** SegFormer-B5 (ADE20K) runs in its own env (`envs/seg`, I-002) as a cached subprocess:
  `scripts/seg_walls.py` + `floorplan/photo/semantic.py`. Only wall-labelled points define a side; a side with no
  wall-labelled points keeps its geometric fit.
- **Results (exact GT).**
  - k65: footprint −14.9% → −8.3%, living room −31% → −17%.
  - k38: footprint −8.0% → −3.8%.
  - k22: −49% → −45%.
- **Cost.** CPU 3.5 s per photo; GPU about 0.3 s.

## D-061 Video: live-run time caps

- **Re-tracking.** Re-run DPVO only on segments of at least 60 keyframes, longest first, at most 8. It had cost
  1570 s of the sim's 3435 s and 715 s of real with_ceiling's 1735 s.
- **Sheet search.** At most about 200 keyframes in total (was 1,120, 246 s).

## D-062 Photo tier: a spin made of wall close-ups is boxed from the doorway photo

- **Trigger.** Half or more of a room's turning photos have median depth under 1 m.
- **Example.** k22 bedroom2: 2 of 4 turning photos were blank wall at 0.6 m.

## D-063 Photo tier: an unplaced second doorway photo is tried at the same spot

- Tries the 4 Manhattan-consistent yaws.
- Keeps one only if it measures more sides without moving any measured side by more than 25 cm.

## D-064 Capture protocol v2.2: audit the data first, then a coverage-aware scripted person

- **Principle (18:40).** Rather than build a plan and then discard it as wrong, first check that the data is
  accurate.
- **Audit tools.**
  - `sim/audit_photos.py`: ray-cast, per photo and per wall, in the 1–2 m band.
  - Three agents reviewed every photo by eye (`outputs/sim/<house>/verify/visual_review.md`).
- **Findings (consistent across all three houses).**
  - Turns cover only 108–260°.
  - Doorway photos lose 25–55% of the frame to jambs.
  - Small rooms' "left/right" photos sit 20° apart; far-end photos through wide openings show the next room.
  - The second take duplicates the first.
  - Bedroom2 is all close-ups.
- **Changes.**
  - **Seer**: what the camera would show. The turning spot is chosen by wall coverage, with views clear for at least
    1.2 m; blocked directions are skipped.
  - Cramped rooms use the small-room method. Doorway shots turn away from blocked views.
  - A coverage pass adds a photo of any wall under 30% seen, within 8 per room.
  - Shallow rooms are photographed along their length from each end, with doorway photos turned about 60°.
  - The second take uses a different spot (0.6–1.2 m away).
  - Body clearance 25 → 18 cm (walkable area +16–19%); camera clearance 25 → 15 cm.
- **Walk.**
  - Room by room: floor line, then ceiling line (changed at 19:00; was the whole house twice).
  - The loop looks 30° into the room, so it covers all the walls while walking. There is no turning on the spot,
    which monocular tracking cannot handle.
  - Small rooms are walked into ("never from inside").

## D-065 Photos with the 0.5× ultra-wide lens

- **Why.** Every iPhone 15 has it. 106° across instead of 67°: 3 turning photos close the circle within the brief's
  8-photo cap, jambs shrink, and the ceiling line is in view.
- **Planned wall coverage, 1× → 0.5×.**

  | House | Room | 1× | 0.5× |
  |---|---|---|---|
  | k65 | living room | 47% | 67% |
  | k65 | balcony | 22% | 37% |
  | k22 | bedroom2 | 26% | 51% |
  | k22 | kitchen | 20% | 35% |
  | k38 | bedroom | 43% | 60% |
- **Pipeline.** Unchanged: the EXIF f35 (13 mm) gives the focal length. The sim renders, emulates and writes EXIF
  per photo. Video and LiDAR stay on 1×.

## D-066 Video: a keyframe pair votes on scale only if its cost curve is sharp

- **Cause.** The local scale comes from depth agreement between keyframe pairs. Near-rotation pairs (turns, doorway
  pauses) still showed > 0.005 contrast from noise, so they voted with arbitrary minima: local scale 0.002–45 and
  fake 2× "jumps". That gave 15–36 segments on the sim and real walks.
- **Fix.** `scale_min_contrast` 0.005 → 0.015: ±20% must change the pair's cost by 15% of the truncation. Pairs that
  fail carry no opinion.
- **Status.** Measured in the 4 Oct evening end-to-end runs (sim k65, real with_ceiling).

## D-068 Photos back to the 1× main lens (D-065 reverted for now)

- **Measured.** Simulated 1 BHK, same house, photo tier with today's code, scored on exact GT.
  - 0.5× photos (v2.3): footprint −48.8%, living room −92.8%, walls within 8%: 38%, 0 openings.
  - 1× photos (v2.1): footprint −8.3%, living room −17%, walls within 8%: 58%. Today's code on the old 1×
    photos is being re-scored to separate lens from code.
- **Cause.** Coverage is better at 0.5×, but the wide view sees straight through the 2.1–2.3 m kitchen and balcony
  openings. The box fit took the kitchen's and balcony's far walls as the living room's sides (10.6 m vs 8.0 m
  true), and the plan's overlap step then crushed the room. The pipeline's through-door handling was tuned on 1×.
- **Decision.** The capture protocol and the house guide go back to the 1× main lens for photos, just before my
  own capture. The other v2.2 changes stay:
  - every wall checked;
  - cramped rooms photographed like small rooms;
  - doorway pairs at every door;
  - the per-room video walk.

  The 0.5× path (through-door wall rejection for wide views) is future work.
- **D-063 switched off (20:30).** Its four-way yaw search accepted an impossible −130° turn for the k65 kitchen's
  second doorway photo, which "measured more sides" from the wrong walls: kitchen +78%. It never helped k22. Off by
  default (`threshold_same_spot_unplaced`).
- **k65, 1× photos, today's code, exact GT (best configuration).**

  | | D-063 on | D-063 off |
  |---|---|---|
  | Kitchen | +78% | +12.6% |
  | Walls within 8% | 46% | 58% |
  | Wall median error | — | 5.6% |
  | Interval coverage | — | 100% |
  | Footprint | −3.1% | −9.0% |

  The remaining footprint gap is mostly the L-shaped living room (−18%).

## D-067 No reference object in any capture; the paper-sheet cue is opt-in

- **Rule (4 Oct, evening).** No A4 sheet, AprilTag or other reference object in any capture: not every customer
  will have one, and competitor apps need none.
- **Decision.** The capture pages no longer ask for a sheet, and the pipeline does not look for one by default.
  This supersedes D-013 and the sheet parts of D-016, D-017, D-020, D-028, D-040, D-047, D-056 and D-061. A printed
  marker stays rejected, as in D-013 option (b).
- **What replaces the sheet.** Scale comes from the capture itself:
  - learned metric depth (MoGe-2, 1-sigma 4%), in both tiers;
  - the focal length stored in each file: photo EXIF; for video the container metadata, fused with SfM and GeoCalib;
  - photos: the camera-height prior (1.35 ± 0.15 m).
  - Not used: standard door heights. `floorplan/photo/door_scale.py` was built and measured, but it stays off and is
    not wired in (D-069).
- **Measured cost (`outputs/handoff/no_sheet_baseline.md`).** Scale error = scale / ideal scale − 1; a negative value
  means the plan comes out too small.

  | Run | With the sheet | Without |
  |---|---|---|
  | Photos, sim k65 (2 sheet readings) | +0.6% | +0.5% |
  | Photos, sim k22 (7 readings) | −0.4% | −0.5% |
  | Photos, sim k38 (1 reading) | −2.5% | −3.4% |
  | Video, sim k65 take 1 (40 sightings) | −18.7% | −1.7% |
  | Photos, 3 real samples (no sheet in them) | – | −3.6%, −4.2%, −7.5% |

  - **Photos.** The sheet moved the scale by at most 0.9 points (k38).
  - **Video.** The sheet made the scale worse. The sheet heights were within 1.7% of truth, but they were divided by
    a camera height taken from an untrusted segment that was never joined to the rest. The correction (×0.827) was
    then copied to all 28 segments.
  - **Detection.** Only 2 of 18 sim photos with a sheet in frame gave a reading (D-056): a white sheet on a light
    floor is rejected.
- **What remains.** Without a sheet the photo scale is MoGe-2's own: it carries 88.5% of the fusion weight. The
  camera-height prior carries 11% and sits 3–8% below the true heights (1.39–1.47 m). The reported 1-sigma of 3.8%
  under-covers the worst real sample (−7.5%).
- **Code.** The detector and its fusion stay as an option, off by default.
  - Photos: `PhotoParams.use_sheet = False`; `--photo-param use_sheet=true` switches it on.
  - Video: `VideoParams.use_sheet = False`; `scripts/run_video_frontend.py --sheet` switches it on.
  - `scene_info.json` keeps its keys: `scale.sheet_status` reads "off (…)" for photos, `scale.sheets_found` is 0
    for video.
  - Sim: `sim/protocol_v2.py` has `LOOK_AT_SHEETS = False`, so the scripted walk no longer looks at the sheet. The
    corner glances are unchanged. Nothing was re-rendered: the existing renders still contain the sheet props and
    the old looks.
- **Capture pages.** `docs/CAPTURE_PROTOCOL.md` and the house guide no longer ask for a scale sheet. No remaining
  step changed its number. The paper decals for the staged damage stay.
- **Risks / revisit if.** The own-home tape measurements show a scale error outside the interval. The fix is then
  MoGe-2's bias on real images and the per-room scale consistency, not a reference object.

## D-069 Door-height scale cue: built and measured, kept OFF

- **Why try it.** Without a reference object (D-067) the photo scale is MoGe-2's own. Every home has doors, and an
  interior door is about 2.0–2.13 m tall.
- **What was built.** `floorplan/photo/door_scale.py`.
  - It finds doors in the ADE20K class map of each photo (D-060).
  - It keeps a door only if its top edge is inside the image, it stands on the floor, and its size is plausible for a
    room door.
  - It measures the height from a plane fitted to the door's points and the photo's floor level. The top edge is
    not read from the depth map.
  - Prior 2.06 m, sigma 3.5%, systematic per home: all doors in one home are alike, so more doors do not shrink it.
  - It gives a cue only with at least 2 doors in at least 2 photos.
- **Measured (`outputs/handoff/door/*.json`).** Scale error = scale / ideal scale − 1.

  | Set | Doors accepted | Fused scale error, without → with | Door cue alone |
  |---|---|---|---|
  | Simulated k65, 1× photos | 2 of 11 candidates, in 2 photos | +0.5% → −0.4% | −4.4% |
  | Simulated k65, 0.5× photos | 1 of 8: below the 2-door minimum, no cue | +0.8% (unchanged) | – |
  | Real sample frames (single_room, floor_only, with_ceiling) | 0 of 5, 0 of 11, 0 of 17 | unchanged | – |

  - Sim 1×: the cue moved the scale by 0.9 points, but the error only shrank from 0.54% to 0.36%. The two door
    readings differ by 15% (scale 1.014 vs 0.882): on the true depth, the same measurement gives 2.11 m for the
    living-room door but 2.49 m for the kitchen one, which is not a standard door. The cue's own sigma is therefore
    8.1%; the fused sigma goes from 3.8% to 3.4%.
  - Real frames: of the 33 door candidates, 16 were cut by the top border, 7 were too small, 5 were in frames with no
    floor plane, 2 did not stand on the floor and 3 failed other checks. These are video frames, not photos. The
    photo tier is not benchmarked on this dataset (D-071); this only shows how often the checks fail.
- **Status.** `PhotoParams.door_scale_cue = False`. The module is NOT wired into `floorplan/photo/frontend.py`: the
  hook was lost in concurrent edits, so the parameter has no effect today. Wiring it needs one line after the cue
  list, and a validation on real photos of homes.
- **Why off.**
  - Too few accepted doors per capture: 2, 1 and 0, against a minimum of 2.
  - Not validated on real photos.
  - The change on the simulator (a 0.9-point shift; the error shrank by only 0.2 points) is inside the noise: the
    fused 1-sigma is 3.4–3.8%.
- **Revisit if.** The own-home photos (OnePlus Nord, tape ground truth, `scripts/process_own_capture.py`) show at least
  2 whole doors per capture. Wire the hook, then keep the cue only if it brings the scale closer to the tape.

## D-070 Video closet: a corridor seen through a doorway, not ghost geometry; the fix needs a written policy first (proposed, not merged)

- **Finding.** See I-014. The extra area in the bad single_room run is a real corridor (and a bathroom seen through
  its door) that the camera stepped into for about 3 s. Whether it shows up as a 1.2–1.6 m² sliver or a 7.96 m² box
  depends only on whether a stray wall line exists on the region's far side (`walls.py:179-182`).
- **Proposed fix (three parts, only together).**
  1. Close the arrangement at the region's own extent in all tiers (no silent truncation).
  2. A written rule for briefly entered space: when the walked path inside a region stays below a threshold (dwell
     and path length to be set from the sample captures), the region is drawn as "partially observed", hatched,
     without dimensions and outside the footprint. It is not a room.
  3. Video: free space counts only when seen from at least two viewpoints at least 0.12 m apart.
- **Why not merged tonight.** Each part alone made 3 of 4 runs worse; together they move the LiDAR plans too (the
  LiDAR reference itself loses this corridor through the same truncation). Validation needs every real LiDAR scene,
  the sample videos and the simulator.
- **Status.** Not merged; today's code is unchanged. Candidate for the fix loop (Part 4).

## D-071 Benchmark scope per tier: photo tier on the simulator, sample captures for video and LiDAR only

- **Decision (4 Oct, 21:20).** The recruiters' sample captures (single_room, floor_only, with_ceiling) are used
  for the video and LiDAR tiers only. The photo tier is never benchmarked on them. It is benchmarked on the simulated
  k65 flat (exact ground truth) and on my own home photos (tape ground truth).
- **Why.** The sample captures contain no photos. Our photo sets for them were cut from the Stray Scanner video
  (`scripts/make_photo_folders.py`): 1920×1440 video frames with motion blur, video exposure and frame intrinsics
  written into EXIF by us. They do not represent "2 to 8 stills per room from any iPhone 15 or newer", so a score on
  them says little about the photo tier a customer would use.
- **What changed.**
  - The photo jobs on the sample captures were stopped (21:20).
  - `scripts/bench_final.py`: the sample-capture photo block was to become opt-in. Not done: it still scores every
    photo run it finds under `outputs/benchmark/final/photo/`.
  - Photo-tier numbers quoted in the report come from k65 (walls median 5.6%, 58% within 8%, footprint −9.0%,
    `outputs/sim/k65v2/runs_iphone15/eval_1x_today`) and from the own capture once scored.
  - Earlier photo rows for the sample captures (REPORT.md §1, COMPLIANCE.md, TECHNICAL_REPORT.md)
    are history; the submission-checklist audit lists each place to update.
- **Cost.** No real-camera photo benchmark exists until the own capture is scored. Say so in the report.

## D-072 Tape ground truth: walls are paired by geometry on an outline, not by order

- **Context.** D-015 paired walls by cyclic order. One extra 3–10 cm stub or door reveal in a plan shifts every later
  pair, so walls that are right within 1 cm scored tens of centimetres. On the simulated k65 LiDAR plan, the 33 cm
  median wall error came from the scorer, not the plan.
- **Decision** (`floorplan/benchmark/gt_eval.py`, `floorplan/benchmark/wall_match.py`).
  - **With GT outlines** (`scripts/eval_own_capture.py --gt-json`, else `sim_gt.json` next to the CSV, which the
    simulator writes): the predicted room is moved onto the GT room by a rigid transform (no scale), and each GT wall
    takes the predicted walls on its line (same direction ±10°, offset ≤ 0.30 m; collinear pieces merge). When the
    whole plan fits the GT (IoU ≥ 0.5), rooms without a matching label are paired by overlap.
  - **With tape lengths only:** `tape_method="geometric"`, the default of `gt_eval.evaluate`. The outline is rebuilt
    from W1..Wn (clockwise, right-angled corners), the predicted room is registered on it by IoU (the door wall breaks
    near-ties; lengths never choose), and walls are paired as above. If several outlines fit the tape, each is scored
    and the worst is reported. If none fits, a constrained order pairing is used, with the axis and the door as hard
    constraints.
  - `anchored` and `order-merged`, the earlier order pairings, stay for comparison. The plain cyclic order is still
    reported as `wall_order_based`, outside the gates.
- **Own house.** `scripts/house_gt.py` writes `gt_polygons.json` next to `ground_truth.csv`: room outlines from the
  hand sketch's layout and the tape lengths. The sketch is not to scale, so the outlines only decide the pairing;
  every scored length comes from the tape.
- **Repeated room labels (5 Oct).** The scorer kept the last plan room whose label is the GT room's id. The video
  tier calls most rooms "room", so on my video plan the bedroom was scored against R8, a 1.6 m² slice of the hall.
  Now a label that several plan rooms carry does not decide. The GT room takes the one that overlaps its outline
  best (without outlines: the lowest wall-length profile cost), and the report warns when that room fits another
  GT room better. On that plan it picks R1 (IoU 0.69) and warns that R1 fits the hall better (0.77). The video
  diagnosis shows why: the bedroom is split into R3 and R5, so no single plan room is the bedroom.
- **Not yet consistent.** `scripts/eval_own_capture.py --tape-pairing` offers only `anchored` (its default) and
  `order-merged`. So a tape-only GT scored from the command line without `--gt-json` is paired by the anchored
  order. For the own house, run `scripts/eval_own_capture.py --gt-json <gt>/gt_polygons.json`.
  `scripts/process_own_capture.py` passes it when `gt/gt_polygons.json` exists (5 Oct).
- **Evidence.** `tests/benchmark/test_gt_eval.py` runs the default outline path: it finds the door wall as W1 and
  recovers an injected 12 mm wall error exactly.

## D-073 Head-to-head app on Android: Matterport (free) first, CubiCasa (one free scan) second; not magicplan

- **Constraint.** My phone is a OnePlus Nord (Android, ARCore yes, no LiDAR). The brief wants a free tier
  ("cost is not an accepted reason; free tiers exist") and an app that measures the rooms itself.
- **magicplan is out.** It removed the camera/AR scan from Android in version 2024.24.0: "Android devices are not
  supported for magicplan's scan features" (https://help.magicplan.app/supported-devices). On Android it only draws
  rooms from typed lengths, which would repeat the tape numbers. Checked on 4 Oct with the vendor pages and Play
  reviews (`outputs/handoff/apps/`).
- **Checked and rejected.** ARPlan 3D and CamToPlan measure with ARCore, but recent reviews say the free tier locks
  saved plans or screenshots (a paid trial is needed); Floor Plan Creator has no camera measuring.
- **Chosen (free apps only).**
  1. Matterport, free plan (no card, one space): 360° phone scans, model built in the cloud (under an hour to about
     8 hours), Measurement Mode for walls, ceiling height, doors and windows. OnePlus is on its supported list.
  2. CubiCasa, one free scan (needs Android 12+ and ARCore): a walk-through scan, a plan with room width × length and
     area within about 6 hours (vendor AI plus a human check).
- **Deviation from the brief.** Part 3 asks for our LiDAR tier against the app. I have no LiDAR device, so the
  head-to-head compares our camera tiers (video, photo) with these apps on the same rooms and the same tape; the
  LiDAR tier's accuracy is shown on ARKitScenes laser data instead. Say this in the report.

## D-074 Photo tier: the last photo of a room's series is its ceiling photo from 10° up

- **Context.** On my phone the ceiling photo came out tilted up only 16.6° (lit take) and 16.7° (dim take), under the
  18° cut-off of D-052. So it joined the turning photos with a yaw about 90° off. On the lit take it alone set the
  window-wall side at 1.01 m from the spin centre; the other photos see that wall at 2.29 m.
- **Decision.** `ceiling_photo_last_min_pitch_deg` = 10°. The last photo of a room's series, where the protocol puts
  the ceiling photo, is a ceiling photo when tilted up more than 10°. Photos mid-series keep 18°.
- **Why not 10° for every photo.** On the sample's with_ceiling/rotate set (video frames), three turning frames are
  tilted up 11.9–16.9°. With 10° for every photo they left the spin, and 3 runs gave 3 different plans. With the
  last-photo rule the set gave the same plan as at 18°, within 4 cm (`outputs/own_house/diag/fix_refute/`).
- **Evidence.** Same photos and cached depth, before and after (`outputs/fixes/`):

  | Run | Before | After |
  |---|---|---|
  | Own lit: walls median / within 8% / ceiling (tape 2.629 m) | 44.0%, 0 of 6, 2.314 m | 9.1%, 1 of 6, 2.514 m |
  | Own dim: the same | 28.6%, 0 of 6, 2.305 m | 22.5%, 0 of 6, 2.562 m |
  | k65 (sheet cue on): walls median / within 8% / footprint | 5.6%, 15 of 26, −9.0% | the same (rows within 0.2 mm) |

- **Limits.**
  - The simulator cannot test this cut-off: no simulated photo is tilted up between 10° and 18°.
  - Most of the ceiling gain comes from the 2.70 m prior. The ceiling photo shares no features with the room's other
    photos, so its own scale gets a 25% sigma. Its measured part is 2.345 m (lit) and 2.429 m (dim).
  - The one lit wall within 8% (W4, −3.0%) comes from two side errors that cancel: the window side about 34 cm too
    far, the wardrobe side 70 cm too near.
  - A weakly tilted ceiling photo that is not the last one is still missed (small-room protocol: a doorway photo
    follows it).

## D-075 Video scale from PnP on MoGe-2 depth (fix loop; the gate still fails)

- **Context.** The declared fix-loop gate (`FIX_LOOP.md` Part 1): video walls within ±3% on my take1, 0 of 14 in both
  before runs. The local scale came from depth agreement between keyframe pairs, and few pairs voted: 31 and 25 of 902.
  On the 23:28 run DPVO's scale dropped about 13× at t 34.5 s, no pair voted between t 12 and 37 s, and the ±1.5× clamp
  hid the drop.
- **Options.** Each was replayed on the 23:28 run and rejected (`FIX_LOOP.md` Part 1, item 2): another depth model
  (MoGe-2 is within 6% inside one frame), the plan step (3 rooms on the PnP-scaled scene), the D-066 gate back at 0.005,
  no clamp, and PnP votes plus a cut at the jump.
- **Decision** (`floorplan/video/scale.py`, `dced951`, tag `after-fix`). For keyframe pairs 2, 4 and 6 apart: SIFT
  matches, the MoGe-2 depth of the first frame, `solvePnPRansac` and an LM refine. A pair votes when its metric step is
  at least 8 cm and it agrees with DPVO within 35° in direction and 4° in rotation. The vote is metric step / DPVO step.
  The local scale is the running median of the votes within ±8 keyframes (at least 3 votes, log-interpolated in
  between). The Gaussian window and the ±1.5× clamp are gone. Segment cuts, DPVO re-runs, the self-check, the pose graph
  and the plan step are unchanged.
- **Evidence** (`FIX_LOOP.md`, "After the fix": two runs, then each run replayed on its own cached DPVO run with the old
  and the new scale step).
  - Gate: 0 / 0 → 3 / 0 of 14 within 3% (predicted 0, range 0–2). On the same DPVO runs, old → new: 1 → 1 and 1 → 0.
    The gate did not move beyond run-to-run noise.
  - The walk holds together. On r1's DPVO run kf1–kf42 goes from 10.47 to 0.68 m (0.50 m by PnP), and the footprint from
    37.8 to 29.2 m² (the house is about 29–31 m²). The real r1's bedroom is 11.02 m² against 11.09 m², with 3 walls
    within 3%.
  - Plans: 2 and 1 rooms, against 3 (3–4) predicted.
- **Limits / revisit if.**
  - The self-check's spread limit (2.0) assumed the clamp, which kept the spread at 2.25 or less. The PnP scale follows
    DPVO's real drift (spreads of 2 to 107 here), so drifting segments are now dropped. On take1 r2's DPVO run 20
    keyframes stay, against 121 with the old step: 1 room against 2.
  - On the sample videos the after runs are worse than the before runs: single_room +25.3% footprint, floor_only 0
    rooms, with_ceiling 1 room. The old step on the same DPVO runs is also far off (+19.9%, 0 rooms, and 3 rooms with one
    48 m² room; for with_ceiling one DPVO segment had to be run again), so this is mainly the DPVO run (I-007), not this
    change. On single_room the new step's plan is still 1.6 m² larger (+29.3% against +19.9%).
  - Next, not done: a self-check that fits the PnP scale (for example the votes' scatter around the running median, not
    the spread of the scale itself), then the after runs again.
- **Status.** In the code since `dced951`. Keeping it, or going back to the clamp until the self-check is redone, is not
  decided here.

## D-076 Video scale: depth agreement is the default again; PnP votes are opt-in (fix-loop follow-up)

- **Context.** A follow-up to the closed fix loop; `FIX_LOOP.md` Part 1 and its results are unchanged. D-075 made the
  PnP scale the default and left two problems on take1. (a) The self-check's spread limit (2.0) assumed the clamp; the
  PnP scale follows DPVO's real drift (spreads 2–107), so drifting segments were dropped. (b) On before r2's DPVO run no
  segment was dropped, yet the plan had 1 room, with the camera 3.25 m above the plan floor.
- **Cause (b)** (`outputs/fixloop/followup/cause_b.py`, `NOTES.md`).
  - The camera path drops 2.2 m. 1.47 m of it is two keyframe steps, kf 135 to 137 (t 78–80 s).
  - DPVO's step kf 136 to 137 points mostly down (0.0259 units down, 0.0078 across). The local scale there is 44.2, so
    it becomes 1.14 m. With the old scale (5.44) it is 0.14 m.
  - PnP solves no pair between kf 131 and 137. The one pair over the step differs from DPVO by 38° in direction and
    16.5° in rotation: a DPVO glitch where PnP finds no matches.
  - The floor seen after kf 137 sits about 1.7 m below the floor seen before. The pose graph anchors only floors within
    0.15 m of the lowest ones (3 fragments). The plan takes the lowest floor and loses the hall and bedroom.
- **What "pnp" now does** (`5fe6ae6`; "depth_agreement" is the code before D-075: window, clamp, spread test).
  1. Self-check: no spread test. A segment needs ≥ 3 votes within ±8 keyframes on at least half of its keyframes
     (vote coverage ≥ 0.5), plus the depth residual ≤ 0.058 and ≥ 20 keyframes as before. Set from the scale step
     alone, before any replay (probes, `probe_stats.json`, `probe_truth.json`): the sample segments that ARKit puts
     113–213% off had coverage 0.00–0.44; take1's segments 0.67–1.00. single_room's one segment has 0.31–0.39 although
     ARKit puts it within +3.4%; a run whose only segment fails keeps it, with the 25% sigma floor. The vote residual
     (0.015–0.215) did not separate good from broken segments, so it is reported, not judged.
  2. Floor levelling before the pose graph. Each keyframe measures its camera height above the floor in its own depth
     (lowest strong level 0.5–2.2 m below). Keyframes within 0.30 m of the run's median height count (bed tops are
     0.5–0.8 m closer). A running median per segment (±8 keyframes) makes the floor's height in the scene one value.
     On before r2's run the path moves −0.14 to +2.31 m, the floor's p10–p90 goes from 1.89 to 0.15 m, and the plan
     from 1 to 3 rooms.
  3. `VideoParams.scale_method` = "depth_agreement" | "pnp"; `run_capture.py --video-scale pnp`. `floor_level` is on
     with "pnp" and off with "depth_agreement" unless set.
- **Rule**, committed before any replay (`3d8b4f1`). Per run, room-count error (take1: distance from 3–4 rooms; samples:
  from the LiDAR plan's rooms) and footprint error (take1 against 30 m²; samples against the LiDAR plan). New is worse
  if its room-count error is larger or its footprint error more than 5 points larger. "pnp" becomes the default only if
  (A) it is not worse on more than half of the runs and (B) on take1 it has a lower median footprint error and a lower
  total room-count error.
- **Replays** (each run's own cached DPVO, MoGe-2, GeoCalib and SfM arrays; CPU, one at a time;
  `outputs/fixloop/followup/summary.json`). Walls: tape walls within 3% / within 10% / found, of 14.

  | Run | Old: rooms, footprint | New: rooms, footprint | Walls, old → new | Verdict |
  |---|---|---|---|---|
  | take1 before r1 | 4, 24.1 m² (−20%) | 2, 29.2 m² (−3%) | 2/4/12 → 1/4/9 | worse (rooms) |
  | take1 before r2 | 4, 40.3 m² (+34%) | 3, 31.3 m² (+4%) | 0/1/8 → 3/4/10 | better |
  | take1 after r1 | 2, 37.8 m² (+26%) | 2, 27.4 m² (−9%) | 1/1/6 → 0/0/6 | better |
  | take1 after r2 | 2, 18.7 m² (−38%) | 2, 30.8 m² (+3%) | 1/1/8 → 2/2/10 | better |
  | take1 23:28 | 9, 37.6 m² (+25%) | 3, 29.8 m² (−1%) | 0/1/8 → 1/1/12 | better |
  | single_room d069 r1 (LiDAR 3, 17.61 m²) | 3, 17.8 m² (+1%) | 3, 21.3 m² (+21%) | | worse |
  | single_room d069 r2 | 3, 17.1 m² (−3%) | 3, 22.6 m² (+28%) | | worse |
  | single_room d069 r3 | 3, 16.7 m² (−5%) | 2, 14.8 m² (−16%) | | worse |
  | single_room after run | 3, 21.1 m² (+20%) | 3, 19.1 m² (+8%) | | better |
  | floor_only after run (LiDAR 8, 61.90 m²) | 0 rooms | 3, 22.6 m² (−63%) | | better |
  | floor_only d066 | 5, 31.4 m² (−49%) | 1, 7.4 m² (−88%) | | worse |
  | with_ceiling after re-run (LiDAR 8, 62.50 m²) | 3, 54.3 m² (−13%) | 2, 21.0 m² (−66%) | | worse |

- **Decision.** (B) holds: on take1 the median footprint error falls from 26.0% to 2.7%, and the room-count error from
  7 to 3. (A) fails: "pnp" is not worse on 6 of 12 runs, not more than half. So "depth_agreement" is the default again
  and "pnp" is opt-in. The rule's line in `FIX_LOOP.md` named only the floor_only after run; `NOTES.md`, written at the
  same time, added d066 if its cache replays. It did, so it counts. Without it (A) would pass, 6 of 11; I take the
  stricter count. with_ceiling's final run (4 Oct) cannot be replayed on CPU: its cache lacks a fresh DPVO run.
- **Why the two sides differ** (ablations after the decision, `ablation_summary.json`, not part of the rule).
  - take1: the PnP scale is what helps. Without the floor levelling "pnp" gives 2–4 rooms and 27.6–29.5 m² on four
    runs, and before r2's run collapses again (1 room, 7.1 m²). The old scale with the floor levelling gives 29.5,
    19.3, 40.2 and 18.3 m² (before r1, r2, after r1, r2; the 23:28 run crashed in the plan step, a shapely topology
    error).
  - Samples: SIFT finds 0–15 keypoints on many of these blurred white-wall frames, so votes are sparse (single_room 22
    over 164 keyframes). Where they are sparse the PnP scale is interpolated: on d066 the first segment comes out
    +200.6% against ARKit, and the coverage test drops it with most of the plan. On floor_only's after run the old
    step's plan floor is 5.3 m below the median camera (0 rooms); "pnp" levels the floor (p10–p90 3.56 to 0.14 m).
  - single_room follows its closet (LiDAR 1.93 m²): 1.3–1.7 m² or 4.1–7.6 m² depending on run and arm, also with the
    old scale plus levelling (6.9 and 6.0 m² on r3 and the after run). The two main rooms stay at 8.2–11.0 and
    5.0–7.9 m² (LiDAR 8.31 and 7.56). That is the plan step's closet (I-014, D-070), not the scale. Against ARKit
    the PnP path is +3.0% and +1.8% (d069 r1, after run), the depth-agreement path +4.6% and +7.7%.
  - Segment by segment against ARKit on the same DPVO runs (`probe_old_vs_arkit.json`), neither scale wins
    everywhere. PnP is closer on single_room and on with_ceiling's segment 1 (−12.7% against −27.9%). Depth
    agreement is closer on floor_only's first segments (+3.3% against +13.8%; d066 +18.1% against +200.6%).
  - Noise: the same arm replayed again gave the same plan on 3 of 4 checks; before r1's old arm gave 24.1 and then
    22.4 m² (the fused floor moved 0.08 mm; Open3D's fusion is multi-threaded). The old arm gives the same plans as
    the fix loop's before-fix replays on 6 of 7 caches; the seventh is that before r1 run.
- **Limits.** The thresholds were set on these runs, with no held-out set. The floor levelling assumes one floor level
  (no stairs or split levels). The old step keeps its own take1 failure: the hidden 13× drop, a median footprint error
  of 26%.
- **Revisit if** "pnp" gets votes on low-texture frames (denser features, or the depth-agreement scale where votes
  are sparse), then replay the 12 runs with the same rule.

## D-077 Photo tier: a wall seen beside furniture replaces it in the room box (far-wall rule; fix-loop follow-up)

- **Context.** On my own Room the side fit took a wardrobe front (lit and dim takes) and an open door leaf (dim) as
  walls; the segmenter labels both "wall" (layout diagnosis, `outputs/own_house/diag/photo_workflow_result.json`). The
  diagnosis's rule B (farthest peak with >= 0.8 m of wall, at most 1.5 m beyond the nearest such peak) fixed those
  sides but cost the simulator. This is a follow-up to the closed fix loop; `FIX_LOOP.md` is unchanged.
- **Why rule B lost walls** (sim truth poses and GT outlines, `fw_gt.py`):
  1. One wall seen by two photos at two depths: k65 living room 1.57 m (two photos) vs 1.91 m (a third), truth
     1.69 m; k65 bedroom 1.11 vs 1.48 m (a close-up of a blank wall), truth 1.19 m; k38 living room 2.62 vs 3.08 m,
     truth 2.60 m. Such peaks are 0.30-0.45 m apart (ratios 1.15-1.33). B takes the farther one.
  2. B starts from the nearest >= 0.8 m peak, not from the code's choice, so it can move a side in: k38 living room
     2.46 m -> 0.65 m (an inner wall; truth 4.30 m).
  Not doorways, small rooms or L-shapes as such.
- **Decision** (`layout._far_wall`, `layout._far_wall_in_room`, `params.layout_far_wall*`). Start from the code's
  choice and only move out. A farther wall-length peak replaces the chosen surface when it has >= 0.8 m of wall and
  >= 40 points; lies 0.5-1.5 m and >= 1.33x farther; at most 20% of its rays cross the chosen plane where the chosen
  surface was seen (a wall cannot be seen through it); at least 80% cross it past an END of that surface (the door
  line: the wardrobe ends at the door and the wall continues beyond it; a wall seen on both sides of the far patch has
  an opening and stays); and, once the 4 sides are fitted, it overlaps the room's extent along it by >= 0.3 m (else it
  is the next room through that door). Main side fits only; the D-060 geometry-only fits keep the old rule.
- **Evidence** (same cached photos and depth, CPU; rule off = `--photo-param layout_far_wall=false`, the same as
  the code before `5dfd531`; tape GT with `gt_polygons.json`, sim with `sim_gt.json`; `outputs/own_house/diag/far_wall/`):

  | Set | Rule off | Rule on | Sides moved |
  |---|---|---|---|
  | Own lit: walls median / within 8% / area (outline 11.09 m²) | 9.1%, 1 of 6, 9.25 m² (−17%) | 8.3%, 2 of 6, 10.66 m² (−4%) | W1 side 1.09 -> 1.78 m (wardrobe -> wall) |
  | Own dim: the same | 22.5%, 0 of 6, 7.68 m² (−31%) | 21.7%, 1 of 6, 10.69 m² (−4%) | W6 side 0.57 -> 1.58 m (door leaf), W1 side 1.11 -> 1.73 m (wardrobe) |
  | Own box W5–W1 × W4–W6 (tape 3.73 × 3.00 m) | lit 3.38 × 2.74, dim 3.60 × 2.13 | lit 4.07 × 2.74, dim 3.60 × 3.15 | |
  | k65 (sheet cue on): walls median / within 8% / footprint | 5.6%, 15 of 26, −9.0% (3 runs) | 5.6%, 15 of 26, −9.0% (2 runs) | none |
  | k22 | 23.5-26.0%, 5-6 of 20, −43.8 to −49.5% (3 runs) | 26.0%, 6 of 20, −44.2 and −44.5% (2 runs) | living room 2.95 -> 4.41 m (truth 4.49 m) |
  | k38 | 5.8-6.2%, 12 of 26, +7.6% (4 runs; 1 more crashed) | 5.8-7.3%, 11-12 of 26, +7.6 to +10.4% (3 runs; 1 crashed) | none |
  | 13 other sim takes (other takes of k65, k22, k38; k38_s1 is not held out, see below) | | same score in 10; k65v23, no side moved: 10 and 4 of 26 in two runs (off 6) | k65v2_dim living 1.21 -> 2.73 m (truth 2.89 m) in one of two runs; the rule-off run already had 2.72 m there, so its 11 -> 12 of 26 is noise; k65v23 take 2 bedroom 0.90 -> 1.89 m (furniture -> wall), 0 -> 2 of 6; k38_s1 bathroom 1.15 -> 2.43 m rejected (next room, overlap −0.13 m) |

  The room-extent check was added after the first run on these 13 takes. Without it the rule moved k38_s1's
  bathroom side 1.15 -> 2.43 m (truth 1.06 m), into the next room: k38_s1 went from 7 to 5 of 26 walls within 8%
  (median 13.1% -> 14.8%) and its bathroom area from −9% to +43%. The final runs overwrote those files. So k38_s1 is
  not held out for that check.

  Other rules on the same caches (one run each; walls median, within 8%, area or footprint):

  | Rule | Own lit | Own dim | k65 | k22 | k38 |
  |---|---|---|---|---|---|
  | B (diagnosis) | 9.2%, 1/6, +21% | 21.7%, 1/6, −4% | 5.6%, 12/26, −3.6% | 13.9%, 7/20, −34.2% | 8.9%, 9/26, −7.4% |
  | B2 (guarded B, fix decision) | 9.2%, 1/6, +21% | 21.7%, 1/6, −4% | 5.6%, 14/26, −2.8% | 24.1%, 5/20, −49.5% | plan step crashed |
  | Pose-graph camera positions (C) | 10.5%, 2/6, −12% | 19.1%, 0/6, −23% | 5.6%, 15/26, −9.0% | 23.5%, 5/20, −47.3% | 6.2%, 10/26, +4.1% |
  | B + C | 11.8%, 2/6, +17% | 10.8%, 2/6, −18% | 5.6%, 12/26, −3.3% | 14.3%, 6/20, −37.6% | plan step crashed |
  | This rule, reach 2.0 m (before the room-extent check) | as above | as above | as above | as above | 12.6%, 7/26 and 8.7%, 10/26 |

  The earlier numbers for B (k65 15 -> 11, k38 12 -> 4 of 26) came from scripts that imported `dummy_repo`'s older
  `floorplan/` (outputs/ is a symlink into it).
- **Ideas measured, not used.**
  - Upper band (1.9-2.4 m) or the wall-ceiling line: the own Room's spin photos (22-31° down) see these planes only up
    to 1.41-1.65 m. No points.
  - Far wall seen in >= 2 photos: each own-Room far wall is seen by one photo (223846, 223956, 223938). It would
    remove all three moves.
  - Floor reaching the far wall: the floor reaches the wardrobe too (1.02-1.03 of its range); the lit W1 wall has
    none (a box in front, 0.0). It does not separate.
  - C (above): with it this rule moves no own-Room side; alone it costs k38 12 -> 10 of 26 and k22 6 -> 5 of 20.
  - Reach 2.0 m: also moves the k38 living room side to its wall (2.46 -> 4.09 m, truth 4.30 m), but k38 then gave 7
    and 10 of 26.
- **Limits.**
  - Lit W5–W1 is now 4.07 m against 3.73 m (+9%): the photos were not taken from one spot. The rule picks the right
    surface; the distance still carries the camera offset (C did not fix it robustly).
  - The polygon step now cuts a false notch at the wardrobe corner (lit 0.69 × 0.70 m, dim 1.01 × 0.63 m), and W5 is
    scored against the cut wall (−32%, −29%). With notches off (measurement only): lit 8.8%, 1 of 6, 11.14 m²; dim
    4.4%, 3 of 6, 11.33 m². Next: no notch behind a surface this rule set aside as furniture (`polygon.py`).
  - The thresholds rest on 3 own-Room sides (2 takes of one room) and the simulator: gaps 0.57-1.01 m against
    0.30-0.45 m, ratios 1.51-2.78 against 1.15-1.33. Not covered: furniture standing more than 1.5 m out, a wall seen
    only in a gap between two pieces of furniture, and furniture less than 0.5 m deep.
  - k38's plan step crashes in some runs (GEOS TopologyException in `plan/beta/extract.py` `_canonical_outlines`, not
    caught): 1 of 6 runs without the rule, 1 of 4 with it. Not this change.

## D-078 Room names from what is in each room

- **Context.** Plans called every room "room" or "corridor"; photo rooms carried their folder name. A floor plan
  names its rooms: CubiCasa's plan of my home says Living room, Eat-in kitchen, Bedroom, Foyer.
- **Options.**
  - (a) Shape only (size, aspect). It cannot tell a bedroom from a living room.
  - (b) A scene classifier on whole images (Places365). One more model; a view through a doorway votes for the wrong
    room.
  - (c) The ADE20K classes the photo tier already computes (SegFormer-B5, D-060): bed, sofa, stove, toilet, ... in the
    images of each room, then shape for what objects cannot tell.
- **Decision.** (c), in `floorplan/plan/room_types.py`, on by default in `run_capture.py` (`--no-room-names` turns
  it off); `scripts/name_rooms.py` names a finished run and scores the names. plan.json schema 1.1: optional `name`,
  `type` and `type_evidence` per room (the rule, images, top classes with pixel shares, score per type, shape). Both
  drawings print the name.
  - Images of a room. Photo: the photos of its folder; a folder named after a type (`01_living_room`) keeps that
    name. Video and LiDAR: up to 60 frames with the camera in each room plus 200 spread over the capture. Each pixel
    votes for the room its 3D point lies in (depth + pose), so a kitchen seen from the hall counts for the kitchen.
    Self-check: pixels the segmenter calls floor must land within 25 cm of the plan's floor, else each frame votes
    for the room its camera is in. Glass and sky have no depth, so the balcony test uses that camera-in-room view.
  - Score of a type = weighted pixel share of its objects: bed 1, pillow 0.3; stove, oven, hood 1, refrigerator 0.6,
    countertop 0.4; sofa, coffee table 1, TV 0.8; toilet, bathtub, shower 1, mirror 0.25. A sink (0.6) is the
    bathroom's when a toilet, bathtub or shower covers 0.5% of the pixels, else the kitchen's. A type needs 0.8%.
  - Shape. Under 2 m wide: bedroom or living room evidence must be 3x stronger; glass and outdoors >= 10% and 3x the
    best object score, with no kitchen or bathroom objects, is a balcony; 1.8x as long as wide (or at most 1.2 m wide
    and 1.5x as long) is a passage. A room of 4 m² or less with the entrance is a foyer. Else "Room".
  - One kitchen and one living room per home. A second claim within 0.5 m of the first is the same room split by the
    plan; elsewhere it takes its next type. With no sofa, TV or coffee table anywhere, the largest room of 9 m² or
    more (at the entrance when one has it) is the living room; a room with a bed qualifies only when another room's
    bed scores 2.5x more. My hall has a cot in it (bed 0.6% of its pixels, bedroom 5.6%).
- **Evidence** (`outputs/room_names/`, `summary.txt`; true types from `sim_gt.json`, own home hall = living room,
  room = bedroom; plan rooms paired with the GT room that covers most of them under the scorer's whole-plan fit,
  IoU >= 0.3):

  | Runs | Paired rooms | Right | Camera-in-room vote |
  |---|---|---|---|
  | Photo, sim k22, k38, k65 (folder names ignored) | 15 | 15 | |
  | Photo, own Room lit and dim | 2 | 2 (Bedroom) | |
  | LiDAR, sim k22, k38, k65, k65_s0 | 21 | 20 | 19 |
  | Video, own take1: after r1, old 8-room run | 5 | 5 | 5 |
  | Video, own r2 (one 6.35 m² room, fit IoU 0.23) | 0 | named Room | Kitchen |
  | Video, sim k65 (fit IoU 0.30; rooms of 59 and 49 m² in a 59 m² flat) | 2 | 0 | 0 |

  - The miss: k38's kitchen is split in two (3.74 + 1.95 m²); the small part has no stove pixels but 1.1% "shower"
    and becomes Bathroom 2. With camera votes the other part got no frames ("Room"). k22's kitchen is split too: the
    camera vote called the part without frames a passage; pixel projection names both parts Kitchen.
  - Self-check: LiDAR floor pixels land 1.2-1.8 cm from the floor; own r1 6.3 cm. The own 8-room run (26.7 cm) and
    sim k65 video (40.4 cm) fell back to camera votes.
  - Cost: about 40 s per LiDAR run (decoding and segmenting about 475 frames), 2-6 s once the class maps are cached.
  - No truth types, checked by eye on the frames of each room (`outputs/room_names/`): the recruiters' single_room
    LiDAR plan gives Living room (sofa 18%), Bathroom (toilet 3.9%), Passage (1.06 m wide); floor_only gives Living
    room (sofa 10%), Bathroom 1 and 2, Foyer, Passage and three "Room" (a landing with stairs, a hall, a desk room).
    Today's own-video runs (`outputs/presentable/`): pnp r1 Living room 22.1 m² and Bedroom 9.0 m²; default r1 one
    Bedroom (bed 8.2%); default r2 one 19.4 m² room called Bedroom (cabinet 15.7%, bed 3.0%, TV 0.7%): a plan of one
    room keeps what its objects say.
- **Limits.**
  - The weights and rules were set on these same runs (no held-out set). k65_s0 (4 of 4) was only scored.
  - A plan that merges rooms gets one name: own r1's hall, kitchen and passage are one "Living room".
  - "Eat-in kitchen", dining and study are not types; ADE20K has no class that separates them.
  - The foyer rule needs a separate room of 4 m² or less at the entrance. No plan here has one.

## D-081 Photo tier: rooms are stitched at the doors they share (door-anchored stitching)

- **Context.** On the k65 simulated flat the photo tier measured room sizes well but put rooms 0.5-2.5 m off and one
  room "beside the block" (overlap IoU 0.39, MyHouse_Dataset/k65_photo_plan_vs_gt.png). The idea came from the user:
  stitch at the doors the out-looking photos see. Every room has a door, and the protocol always takes a photo looking
  OUT through it (the first turning photo faces the door one came in by; a small room's far-end photos look back at
  its door; the doorway pair stands on the threshold). The room that photo matches is the neighbour, and the match
  says where. Then the two rooms snap together at that door.
- **Decision** (`floorplan/photo/door_stitch.py`, one hook in `frontend.py` after the room layouts are fitted,
  `--photo-param door_stitch=true|false`; the layout now stores each fitted photo's local pose, `fit_poses`):
  1. Doors per room, in the room's own layout frame: pixels whose depth runs > 0.25 m past a fitted wall at door
     heights, cast along their rays onto that wall (no sill: open down to the floor; windows, sky, curtains and
     mirrors from the ADE20K map are not openings), plus ADE20K door pixels on the wall. A doorway-pair photo placed in
     the room stands in one of its doors (the threshold).
  2. Door correspondences: each verified photo pair across two rooms, both ways, by PnP of the other photo's pixels on
     this photo's MoGe-2 depth (RANSAC + LM). Kept when the relative pose is level (tilt <= 3 deg), has a real
     baseline (>= 0.3 m), the matched points lie outside this room (seen through its door) and inside the other
     room, and the rooms do not overlap. A room pair needs >= 20 such inliers in one consistent placement, and no
     rival placement with half as many. Why these thresholds: on k65 with true poses, real cross-room pairs gave
     PnP errors of 3-42 cm and 0.0-2.3 deg with tilt <= 1.9 deg; matches on the outdoor scene through two windows
     gave hundreds of "inliers" (643 bedroom-balcony) as a pure rotation (baseline 0), and every other false pair
     had tilt 5-90 deg or under 15 inliers.
  3. Relative placement: the PnP pose puts B's frame in A's (rotation snapped to the Manhattan axes); the door snap
     then makes the facing door intervals coincide along the wall, faces parallel and a wall thickness apart. The
     thickness is the jambs' depth when they are seen (0.23 m on k65, 0.25 m on k22; the simulator's walls are
     0.24 m), else 0.15 +- 0.05 m. A snap that disagrees with the PnP placement by > 0.8 m is not used.
     A doorway pair alone links two rooms only when both door walls are known: a seen door, a small room's door side,
     or a threshold photo square (<= 12 deg) to the wall behind it.
  4. Joint adjustment: rotations from a spanning tree of the strongest links, translations by robust least squares
     over door snaps (normal 0.05 m, along the wall 0.1-0.35 m), PnP links (0.15 m), doorway-pair same-spot priors
     (0.3 m) and an overlap penalty. Rooms only move (sizes unchanged). A room with no matched door keeps its
     placement and is flagged; each stitched room's cameras move with it in the scene.
- **Tried and dropped.** "The only door left" for a pair photo not placed in its room put the k65 bathroom on a
  phantom door (7 m off); the wall behind a diagonal threshold photo put the k65 bedroom door in the wrong wall.
- **Evidence** (cached photos, depth and features, CPU; one run each with `door_stitch=false` / `true` on the same
  working tree, `--no-damage --no-semantic-openings --no-room-names`; `outputs/door_stitch/replay.sh`, scored by
  `scripts/eval_own_capture.py` and `scripts/eval_door_stitch.py`). "Placement vs living room" is where each
  anchored room's frame sits relative to the living room's, plan vs the simulator's true camera poses (it does not
  depend on room sizes or on the whole-plan alignment).

  | House | IoU off -> on | Footprint off -> on | Walls median off -> on | Within 8% off -> on | Rooms stitched |
  |---|---|---|---|---|---|
  | k65 | 0.376 -> 0.439 | -5.0% -> -3.8% | 6.6% -> 6.6% | 38% -> 42% | kitchen, balcony (bedroom, bathroom kept) |
  | k22 | 0.334 -> 0.618 | -44.5% -> -31.6% | 26.0% -> 21.7% | 30% -> 30% | bedroom2, bathroom, kitchen (bedroom kept) |
  | k38 | 0.815 -> 0.797 | +8.6% -> +11.0% | 5.7% -> 9.8% | 50% -> 35% | kitchen, bathroom (bedroom, balcony kept) |

  | Placement vs living room (m), off -> on | k65 | k22 | k38 |
  |---|---|---|---|
  | kitchen | 0.34 -> 0.10 | 0.30 -> 0.16 | 0.31 -> 0.24 |
  | balcony | 0.55 -> 0.31 | - | 0.53 -> 0.56 |
  | bathroom | 9.27 -> 9.27 (not stitched) | 0.17 -> 0.22 | 0.45 -> 0.27 |
  | bedroom / bedroom2 | not anchored | 0.15 -> 0.16 / 0.23 -> 0.23 | 0.41 -> 0.12 |

  Doors in each room's own frame (k65, k22; frame put on the GT by true poses, fit residual 0.4-2.3 deg): k65 6 found
  (along the wall median 0.17 m; widths of 3 seen doors off by 0.04-0.77 m: a door's leaf hides part of it), 3
  phantoms (one is the bedroom's window), 3 missed (the closed front door, the bathroom door no living-room photo
  sees, the bedroom door at the end of a 0.7 m passage); k22 13 found (along 0.11 m median), 3 phantoms, 1 missed.
- **Default: off.** The rule was "on only if IoU improves on all three flats and nothing else gets worse": k38 got
  worse (IoU -0.02, walls median 5.7% -> 9.8%). Part of that is run-to-run noise (the pose graph differs between two
  runs of the same photos: k38 kitchen and balcony sides moved 0.17-0.20 m, and the unstitched bedroom's placement
  error changed 0.41 -> 0.12 m), but one run each cannot show the stitch is not to blame. What holds: on all three
  flats the stitched rooms sit closer to the living room's true relative position in 6 of 7 cases, and the k65 and
  k22 plans improve a lot. Before it goes on: repeat runs per flat, and make the photo front end's pose graph
  deterministic. Pictures: `MyHouse_Dataset/k65_photo_plan_vs_gt.png` (before) and `..._after.png`.
