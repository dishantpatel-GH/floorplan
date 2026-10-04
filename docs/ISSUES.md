# Issues log (integration level)

Format: **Symptom** → **Root cause** (with evidence) → **Fix options** → **Chosen fix and why** → **Status**.
Module-level issues live in `docs/modules/<module>.md`, section 4.

---

## I-001 Plan coordinates are mirror-reversed if drawn with v pointing up

- **Symptom.** The plan frame is (u, v) = (x, z) of the aligned world. Plotting it with v increasing up the page
  shows the apartment as a mirror image.
- **Root cause.**
  - The world is right-handed with y up, and x × z = −y.
  - So the (x, z) axes form a left-handed frame when viewed from above.
  - A correct top-down view, with the viewer looking down along −y, uses (x, −z) with v up, or equivalently (x, z)
    with v pointing *down* the page.
  - The exploration figures inverted the z axis and looked right; `prepare_scene.py`'s figures did not and are mirrored.
- **Fix options.**
  - (a) Redefine plan coordinates as (x, −z) everywhere.
  - (b) Keep (x, z) as data, but require every renderer to draw v downward (image convention).
- **Chosen fix.** (b) for now.
  - Several modules are already being built against (x, z).
  - Lengths, areas and adjacency are unaffected by the mirroring.
  - Only the drawing and the meaning of "counter-clockwise" change.
  - The schema documents it ("v axis points down when viewed from above"), and the renderer flips the axis.
  - Revisit at integration: if the convention causes confusion, switch to (a) in one place, a single transform at
    export.
- **Status.** Open; check at integration.

## I-002 Installing `transformers` would silently break the cached foundation models

- **Symptom.** `uv pip install --dry-run transformers` resolves to `huggingface-hub` **2.1.1 → 1.33.0** (a downgrade).
- **Root cause.**
  - `transformers` 5.18 pins `huggingface-hub` < 2.
  - The MapAnything, Depth Anything 3 and MoGe-2 weights were downloaded with hub 2.1.1, whose cache layout stores
    file bytes under `blobs/<xx>/`, as noted during environment setup.
  - A downgrade risks re-downloads or load failures in the middle of the build.
- **Fix options.**
  - (a) Install anyway and re-test everything.
  - (b) Keep `transformers` out of the main env; give any module that needs it its own env.
  - (c) Avoid zero-shot detectors that need `transformers`.
- **Chosen fix.** (b) or (c), decided by the damage module.
  - The main env stays exactly as smoke-tested (15/15 tests passing).
  - Separating envs is cheap: torch is hard-linked from the uv cache, so a second env costs about 100 MB.
- **Status.** Applied (nothing installed).

## I-003 A wardrobe handle was reported as a crack (confidence 0.61) in the integrated pipeline

- **Symptom.**
  - `run_capture.py` on the clean `single_room` (LiDAR) reported 3 cracks, at confidence 0.61, 0.20 and 0.15.
  - The damage module's own validation on the same capture reported 0.
  - The 0.61 one would pass the D-023 "confirmed" threshold and generate scope items: confident garbage.
- **Root cause (evidence).**
  - The surface overlay (`outputs/submission_docs/fresh_env_lidar_single_room_v2/damage/surfaces/R1-W1.jpg`) shows
    a vertical door handle on a wood-grain wardrobe front.
  - Plan v2 measures the wardrobe front as wall R1-W1, so the detector analyses it. The module's own test used an
    older plan, without this surface.
  - Its metrics:
    - orientation 91.4° (1.4° off vertical);
    - straightness 0.74, from a 20 cm bar plus stand-off "hooks";
    - so it slips between the existing rules, "straight axis-aligned edge" (straightness > 0.95) and "tortuous"
      (< 0.70).
- **Fix options.**
  - (a) Raise the confirm threshold. That loses real cracks: the true ones score 0.18–0.82.
  - (b) Exclude furniture fronts. That needs semantics we don't have.
  - (c) A geometric relief check: a handle protrudes. Only possible at the LiDAR tier.
  - (d) A **dominant-line test**: if ≥ 70% of the skeleton lies within max(3 mm, 1.5 px) of one straight line within
    8° of horizontal or vertical, it is a bar (handle, edge, grout line), not a crack. Real cracks zig-zag.
- **Chosen fix.** (d) (D-024).
  - It works at every tier, it is explainable, and it targets the physical difference.
  - Accepted risk: a perfectly straight, axis-aligned hairline crack (along a drywall joint) would be rejected.
    Diagonal and jagged cracks are unaffected.
- **Status.** Resolved by policy, after three iterations. Each step was measured on six injection sets plus clean
  captures (`outputs/damage/v2c`, `v2d`).
  1. **The bar rule (D-024)** removed other weak line candidates.
  2. **Bounding-box relief (v1)** removed the candidate, but cost 4 true cracks.
  3. **Line-local relief (D-025)** restored all 4 true cracks. It showed the remaining candidate is **flat**: not a
     handle but a dark line on the wood-grain wardrobe front (door gap or grain).
  4. **Final.** Per-class confirm thresholds (D-026: crack 0.65, stain 0.35). The candidate is reported as `review`,
     with 0 scope items.
- **Lesson.** An evidence-free "obvious" explanation (the handle) was wrong. The relief measurement showed what the
  object really was.

## I-004 Drift correction can worsen ceiling height in tiny single-room captures

- **Symptom.** ARKitScenes room 47895909: ceiling −2.14 cm with drift correction ON (fails the 1.5 cm gate), −0.99 cm
  with it OFF. The other 3 rooms with a plan pass either way.
- **Root cause (evidence).** The drift report shows the across-fragment ceiling spread growing from 0.06 cm to 0.55 cm
  after correction. In a short capture with no revisit loops, only floor anchors constrain the vertical. They pull
  each fragment's floor level together, and the ceiling, which was consistent, is tilted apart.
  - Contrast: on with_ceiling, which has many revisit loops, the correction **reduced** ceiling spread from 6.0 to
    0.6 cm.
- **Fix options.**
  - (a) Skip the vertical anchor terms when the capture has no accepted revisit loops.
  - (b) Add ceiling anchors when a ceiling is observed.
  - (c) Weight vertical anchors by the evidence.
- **Status.** Open. A clean fix-loop candidate: one gate (ceiling ≤ 1.5 cm), one root cause, a before/after on 4
  rooms.

## I-005 The same command gave different plans on identical input

- **Status.** Resolved (D-029, D-030). Evidence: `outputs/stability/`, `tests/test_stability.py`.

## I-006 A false confirmed water stain on floor_only (jar lid)

- **Status.** Resolved (D-031). Evidence: `outputs/damage_precision/`.

## I-007 The same video gives different plans on different runs (long walkthroughs)

- **Evidence.** `docs/modules/benchmark_final.md` §7, `outputs/benchmark/final/dpvo_determinism/`.
  - floor_only video, two runs of the identical command: 0 rooms then 8 rooms (05:05 report).
  - After the cuDNN fix: 5 rooms / 40.91 m² then 3 rooms / 18.95 m² (06:28 report).
- **Root causes (measured, two independent ones).**
  1. **pycolmap's two-view geometric verification is not seeded.** `random_seed` and `num_threads=1` don't cover it.
     The SfM features and matches are identical, but the solved focal length differs (1564.6 vs 1566.0 px), so DPVO
     gets different inputs.
  2. **DPVO itself is non-deterministic from frame 1**, even with identical inputs and cuDNN deterministic. It uses
     custom CUDA kernels with atomic adds. The shape of the path agrees to 0.05%, but its arbitrary scale differs 2×.
     Downstream decisions then flip: 2 vs 4 segments, and camera pitch −22° vs −27°, which changes the floor used for
     levelling.
- **Fix attempted.** cuDNN deterministic mode in the DPVO runner. **It did not resolve it.**
- **Mitigation in place.** D-034: unverified multi-segment videos get a 20% relative-sigma floor and a low-reliability
  flag. Coverage rose to 60–88%, but footprint can still miss when whole rooms are missing.
- **Consequence for the fix loop.** Any video before/after comparison needs at least 2 runs per state (FIX_LOOP.md).
- **Proper fix (not done).**
  - Seed or replace the SfM verification step.
  - Run DPVO's affected kernels deterministically, or replace DPVO in the video path with a deterministic tracker.
  - On own captures, check whether the protocol (doorway pauses, ≥ 1.5 m from walls) removes the restarts that make
    the outcome fragile.
- **Status.** Open; documented, and mitigated only by honest intervals.

## I-008 Isaac Sim's Replicator crashes in my env_isaaclab (numpy 2.4.2; Isaac Sim 5.1 needs numpy 1.26.0)

- **Evidence.**
  - Attaching any annotator fails with
    `TypeError: Unable to write from unknown dtype, kind=f, size=0` in `omni.syntheticdata`.
  - `isaacsim-kernel` requires `numpy==1.26.0`, and `isaaclab` requires `numpy<2`, yet numpy 2.4.2 was installed
    (11 Feb, by some later install). My other Isaac Lab scripts may be affected too.
- **Fix (non-invasive).** `envs/isaacsim_np126` is a uv venv on the same Python 3.11.14 with numpy 1.26.0. A `.pth`
  file adds `env_isaaclab`'s site-packages after it, and `env_isaaclab` is untouched. All Isaac scripts use
  `envs/isaacsim_np126/bin/python`.
- **Status.** Worked around. Downgrading numpy in `env_isaaclab` is a separate decision, outside this project.

## I-009 The photo tier converts EXIF 35 mm-equivalent focal length with a 36 mm-width rule; phones use the diagonal

- **Evidence.**
  - `floorplan/photo/images.py` computes f = f35 / 36 × long side.
  - The 35 mm equivalent is defined on the 43.27 mm diagonal, which our own video tier uses
    (`video/metadata.focal_px_from_f35`).
  - On a 4:3 photo the two differ by 3.9%. The sim's iPhone-15-like photos (EXIF 26 mm, true f = 1502 px at
    2000×1500) are read as f = 1444 px (**−3.8%**).
  - Our earlier "phone-like" photos (`scripts/make_photo_folders.py`) wrote EXIF with the same 36 mm rule, so the
    two errors cancelled and the benchmark could not see it.
- **Expected impact.** A wider assumed field of view for MoGe-2 and the bundle-adjustment seed, so scale and shape
  errors of the order of the focal error.
- **Status.** FIXED by D-048: the diagonal rule is now the default. Simulated before/after on the same photos:
  - footprint −46.9% → −13.2%;
  - wall median error 28.2% → 16.7%;
  - walls within ±8%: 25% → 45%.

  `--photo-param f35_rule=36mm` reproduces the before.

## I-010 Rooms joined by a WIDE opening (sliding door, open kitchen) merge into one room (found in simulation)

- **Evidence (simulated, kujiale_0065, LiDAR tier, iPhone-15-like capture).** The kitchen (2.12 m sliding door) and
  the balcony (2.29 m sliding door) merged into the living room:
  - 3 rooms instead of 5; living room 39.2 m² vs 28.8 m² true;
  - the two wide doors missed;
  - wall matching for the merged room meaningless.

  What did work in the same run:
  - total footprint −2.2%;
  - normal doors within ±5 mm (0.706 vs 0.710 m, 0.650 vs 0.645 m);
  - ceilings within 1.0–1.4 cm;
  - bathroom walls within 0.3–1.3 cm.
- **Root cause.** `plan/beta/segment.py` separates rooms only at necks narrower than `door_max_m` = 1.30 m ("rooms are
  wide, doors are narrow"). A 2.1–2.3 m opening is wider than any neck, so the watershed regions merge. The wall
  lines that could cut (`lines.py`) use the 0.3–2.0 m band, below the header that spans the opening.
- **Tried (11:15).** Header-band cuts (paired faces at 2.05–2.75 m over an open low band; `lines.header_cut_mask`,
  `use_header_cuts`). This found nothing on this capture:
  - The header's far side is barely observed. From inside a small room it sits 0.6–1.1 m above the camera, behind the
    person who walked in, so pairing fails.
  - The near-side header lines come out fragmented (~1 m runs over a 2.1 m opening).

  The switch is kept off (no change to any plan).
- **Next options.** In order of robustness:
  1. A header face collinear (≤ 6 cm) with a low-band wall face of the same line, over a gap ≥ 1.3 m.
  2. The protocol's doorway pause as a door marker: the operator stands 2 s in every opening, facing in.
  3. A ceiling-height step between the two lobes.

  Every option must be checked on the real LiDAR benchmark (no new splits in single_room / floor_only /
  with_ceiling / ARKitScenes) before it ships.
- **Status.** Partly fixed by D-046 (header-band cuts, on by default):
  - the kitchen now separates, and its 2.12 m sliding door is measured to 4.6 mm;
  - real LiDAR plans are unchanged (8 scenes identical);
  - the balcony still merges, because its header is hidden by curtains (no points reconstructed there).

## I-011 LiDAR plan: rooms at wide openings take the wall thickness; stubs and reveals become walls (open)

- **Symptom (sim, exact GT, v2 walks).**
  - The kitchen behind a 2.12 m opening measures 2.689 m against a true 2.474 m (+0.215 m, about one wall
    thickness).
  - Its opening side is drawn as five walls (0.564, 0.328, 0.081, 1.297 and 0.081 m) instead of one wall with an
    opening. The 1 BHK plan has 52 walls for 26 true ones, so wall-by-order scoring shows a 33 cm median error that
    is an artifact. The correctly paired walls are close (3.441 vs 3.452 m, 2.202 vs 2.189 m).
- **Not the cause.** The doorway-bump remover (`remove_doorway_bumps`, outward bumps ≤ 1.30 m wide). Allowing
  2.6 m wide bumps changed nothing (tested on all three houses, reverted): the opening side is pulled to the wall's
  far face by the header cut (D-046) and steps back to the inner face at the stubs.
- **Why it matters.** The walk-in test measures wall to wall with a laser. A plan whose wall list has 8–34 cm stubs
  at every wide opening, and rooms 20 cm too long there, loses points on rooms we otherwise measure to ~1 cm.
- **Next.**
  1. Place the cut on the room's inner face.
  2. Merge collinear wall segments across an opening into one wall carrying the opening.
  3. Make the scorer match walls by geometry rather than order.
- **Fix (2026-10-04): `plan/beta/walls.outline_at_openings`, after B-2, LiDAR only.** Video and photo are off through
  `opening_outline=False` in `tiers.TIER_OVERRIDES`.
  - **Root cause (measured).** There are two separate failures, not one.
    1. The D-046 cut lies on the header face that is coplanar with the NEAR room's wall. The far room's side
       therefore sits on the wall's far face: +21.5 cm for the kitchen and +19 cm for the balcony (k65), +25 cm for
       the balcony (k38).
    2. The five kitchen "walls" are not at the sliding door. They are the reveal of the kitchen WINDOW: an 8.1 cm
       deep, 1.297 m wide bump, plus the 0.564 and 0.328 m pieces beside it. A window recess has no open floor
       beyond it, so `remove_doorway_bumps` keeps it.
  - **Rule a: header far face.** `lines.header_cut_mask` now also reports the header's other face (`far`). It must
    face the opposite way, lie 5–40 cm behind the cut, and be seen over ≥ 20 cm within 1 m of the open span. An
    edge of the far room that lies on the cut moves to that face, or onto the room's own line on it. It is then
    refit on raw points: wall band first, header band if the whole side is the opening.
  - **Rule b: excursions into the slab.** Bumps (staircase reveals included) and steps at a corner, 5–40 cm deep and
    ≤ 3.5 m long, close onto the room's measured inner wall line. They close only with opening evidence:
    - a header open span covers ≥ 30% of the excursion; or
    - it is a window recess: a lintel on the inner plane (header band ≥ 30%), the window seen at the recess back
      (wall band ≥ 25%; sim 38–42%) and nothing at sill height (≤ 20%).

    Inward notches (pilasters, ducts) are never excursions. A niche shows its back at sill height and stays.
  - **Not used: door necks.** On the real repeat pair they fired in one capture and not the other. Walls passing the
    gate fell from 15/87 to 11/83, and the median rose from 6.39 to 8.22 cm. Narrow doors stay with
    `remove_doorway_bumps`.
    - The geometry is not the problem. In all 3 sim and 4 real cases the excursion lies on the door's neck cut
      (0.4–5 cm away): the narrow-door version of rule a.
    - Each capture's plan has different excursions at the same doors, so the change is asymmetric.
    - Revisit this with a second real repeat pair.
  - **Sim results.** Run with `scripts/run_plan_beta.py` (raw values, without the +1.36 cm bias). Walls are paired
    to the GT polygons by direction and position.

    | | k65 1 BHK | k22 | k38 |
    |---|---|---|---|
    | Walls (GT 26 / 20 / 26) | 52 → **42** | 48 → 48 | 40 → **38** |
    | Walls < 35 cm | 19 → **11** | 9 → 9 | 5 → **4** |
    | Paired-wall median error | 14.8 → **4.8 cm** | 109.5 → 109.5 cm (split kitchen) | 35.1 → **16.2 cm** |
    | Footprint, m² (GT room sum 59.20 / 60.86 / 63.28) | 57.18 → 55.80 | 55.94 → 55.80 | 61.22 → 60.55 |

    | Room | Walls (GT) | Depth or length, m (GT) | Area, m² (GT) |
    |---|---|---|---|
    | k65 kitchen | 8 → **4** (4) | 2.689 (bbox 2.770) → **2.471** (2.474) | 5.99 → **5.41** (5.42) |
    | k65 balcony | 8 → 6 (4) | 1.647 → **1.417** (1.456) | 5.61 → 4.88 (5.06) |
    | k65 bathroom | 10 → 6 (4) | 3.234 → **3.045** (2.996) | 5.42 → 5.35 (5.26) |
    | k38 balcony | 6 → 6 (4) | 1.914 → **1.727** (1.542) | 5.79 → 5.23 (5.62) |
    | k38 kitchen part R5 (window side) | 6 → **4** | 2.212 → 2.124 (kitchen split) | — |
    | k22 kitchen part R5 | 4 → 4 | 1.862 → **1.790** (1.750) | — |

    - The kitchen's opening wall (2.189 m) carries the 2.125 m opening. Its window wall carries the window.
    - Wall thickness is now measured at both wide openings (0.236 and 0.238 m). It was a zero-thickness shared
      line before.
    - Scorer (`eval_own_capture.py`, current version), k65 wall median: 4.75 → 3.23 cm; within 2 cm: 27% → 35%.
  - **Real LiDAR.** single_room, floor_only, with_ceiling and the 5 ARKitScenes rooms give identical plans: rooms,
    walls, openings, areas and footprint. The with_ceiling vs floor_only pair is unchanged: 15/87 walls pass,
    median 6.39 cm, same topology 14/30.
  - **Worse or not fixed.**
    - **Footprint.** It moves away from the GT room sum. The wall slabs that were removed (k65: kitchen +0.58 m²,
      balcony +0.55 m²) had been masking rooms that are too small for other reasons (k65 bedroom −3.42 m²).
    - **Window widths.** The two windows now detected are under-measured: 0.516 vs 1.200 m and 0.496 vs 0.800 m.
      `openings.py` counts see-through ≥ 10 cm behind the plane, and the sim glass sits ~8 cm behind it. Before the
      fix these windows were not detected at all.
    - **Remaining stubs.**
      - k65 bathroom door side: a 7.5 cm step, with only neck evidence.
      - Full-width window walls of the balconies: k65 −5.6 cm, k38 +11.9 cm.
      - 3.6 cm jogs on the k65 living wall (not at an opening).
      - k22 living window side, where the true wall lies between two pieces.
      - k22 and k38 kitchens are still split.
    - **Header far face.** It is not exactly the wall plane: k38 balcony +6.6 cm on that side.
  - **Status.** Partly fixed. Next: a door-neck rule that is consistent across captures (it needs a second real
    repeat pair), and window width measured at the recess depth.

## I-012 A plan step hung for 3 h; two crash paths in live runs (guarded)

- **Hang.**
  - An orphaned photo run (13:02 code) sat in `extract_plan` for 3 h. It is not reproducible on current code (the
    plan step takes 2.9 s).
  - Guard: SIGALRM time limits on the plan step (300 s, then alpha) and on damage (300 s, then skipped), in
    `run_capture.py` (D-058).
- **Crash 1.** `estimate_floor` called `min` on an empty peak set when a degenerate video segment put non-finite points
  in the fusion (real with_ceiling video). Non-finite values are now filtered and "no floor" is returned.
- **Crash 2.** A GEOS topology error in `_undo_overlapping_refinements` (k38 photo runs). It is caught, and the rooms
  keep their refined lines.

## I-013 Capture v2.2: remaining photo problems found by the visual re-review (k22, 4 Oct 19:30)

v2.2 (0.5× lens, coverage-aware) is "good enough to test the pipeline". Coverage gaps are much smaller than v2.1.
Useless/weak photos: 7/7 of 38 (v2.1: 6/9). For the next capture round:
1. **Doorway shots next to wardrobes and cabinets.** With 106° across, anything within 0.5 m of the threshold fills
   the frame. Either step 0.4–0.5 m in (this would loosen the doorway pair's "same spot" prior, so the pipeline's pair
   sigma would need to grow from 0.2 m) or turn away from any surface that fills more than ~25% of the frame. The Seer
   can measure that fraction.
2. **Rooms whose entry corner is hidden by a partition** (k22 living room: entry door and SW corner). The coverage
   pass should search spots outside the room's open middle too, e.g. the entry corridor.
3. **Headings still only partly screen-driven.** "Faces entry door" photos don't always show the door. Take-2 turns
   can repeat or leave gaps. A cramped bedroom's far-end photo can be a blank wall.
4. **Realism.** The renders are cleaner than a phone (no noise, vignetting, chromatic aberration, residual distortion),
   so sim results are best-case for real phones. Soft dark smudges on plain walls are a render artefact. EXIF
   physical focal fixed for the ultra-wide (1.54 mm); the f35 (13 mm) was already right.

## I-014 Video (and LiDAR) plan: a briefly entered space decides the room shape; arrangement cells are truncated (open)

- **Symptom.** Real sample video single_room, identical command, today's code: footprint 23.85 m² (+35.4% against
  the LiDAR plan, 17.61 m²) at 20:58, then 17.32 / 16.82 / 17.14 m² (−1.6 / −4.5 / −2.7%) in three repeats at 22:00.
  The 1.06 × 1.88 m closet came out as a 4.15 × 1.93 m box once and as a 4.1 × 0.3–0.4 m sliver three times.
- **Cause (measured; `outputs/handoff/video_fix/`, pictures `d066_before_*.png`, `zoom_hall.png`, `frames_closet.jpg`).**
  1. The "closet" is a small hall. Frames 1554–1712 (33.7–37.1 s) step 0.2–0.5 m west into a real corridor and look
     into a bathroom through its door. On video that floor is ray-carved free space (70% of a 1 × 1.2 m box; LiDAR
     9%), so the 0.93 m neck that separates hall and corridor on LiDAR disappears: hall, corridor and bathroom glimpse
     become one basin (`floorplan/plan/beta/freespace.py:116`, `segment.py:49-53, 126-139`).
  2. `arrangement_polygon` (`floorplan/plan/beta/walls.py:179-182`) builds cells only between the outermost candidate
     wall lines; free space beyond the last line is silently cut off. With no line on the corridor's south side the
     region becomes the sliver; one stray line at z = 1.46 gives the full box. Strips under 1.2 m² are dropped
     (`extract.py:354-355`); that is also how the LiDAR reference loses the corridor (a 0.2 m² strip).
- **Not the cause.** Upstream numbers agree across the four runs: 1 segment, 0 VO restarts, path 14.0–14.3 m, floor
  level within 2.8 cm, focal within 1.6%, scale 1-sigma 5.7–6.2%.
- **Per-room error even when the footprint looks good.** r1–r3: bottom room +6 to +12%, top room −9 to −16%,
  closet −17 to −36%. The footprint errors cancel. Wall-line placement on noisy video depth moves each room by about
  0.5–0.7 m².
- **Candidates measured, none merged** (plan step re-run on the saved scenes, `variants.py`): multi-view free space
  (0.12 m baseline) + a closed arrangement puts the hall at 1.68–2.26 m² (LiDAR 1.93) in 3 of 4 runs, but the corridor
  (66 body positions inside, 67% measured wall) then becomes a fourth room of 6.1–6.5 m² under the visited-or-enclosed
  rule (D-B10). Each change alone makes 3 of 4 runs worse. Closing the arrangement on the LiDAR scene gives 4 rooms,
  28.42 m² (+61%).
- **Next step.** D-070: write the policy for briefly entered space first, then ship the three changes together and
  validate on every real LiDAR scene, the sample videos and the simulator. Candidate for the fix loop.
