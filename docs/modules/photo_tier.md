# Module: photo_tier (per-room photo folders → one metric scene)

Files:

| File | What it is |
|---|---|
| `floorplan/photo/__init__.py` | Entry point `build_scene_from_photos(photo_root, params=None, work_dir=None) -> (scene, info)` |
| `floorplan/photo/params.py` | `PhotoParams`: every knob with the reason for its value |
| `floorplan/photo/images.py` | Photo loading: EXIF orientation, HEIC (pillow_heif), EXIF 35 mm focal → pixel focal |
| `floorplan/photo/sfm.py` | ALIKED + LightGlue on ALL photo pairs, COLMAP epipolar verification, diagnostic joint SfM, `verified_matches()` |
| `floorplan/photo/depth.py` | MoGe-2 metric depth + normals per photo (EXIF field of view as input), flying-pixel cleaning |
| `floorplan/photo/views.py` | Per-photo levelled metric point cloud: GeoCalib gravity, refined by the floor plane |
| `floorplan/photo/geometry.py` | Rotations, normals, floor refinement, Manhattan yaw |
| `floorplan/photo/link.py` | Metric photo-photo edges, dense free-space check, pose graph, loop pruning, weak-bridge removal |
| `floorplan/photo/scale.py` | Scale cues (learned depth, camera-height prior, paper-sheet adapter: opt-in, off by default since D-067) and their fusion |
| `floorplan/photo/recon.py` | MapAnything runner (Apache weights, bf16 backbone, crop bookkeeping), GPU wait helper |
| `floorplan/photo/intra.py` | Experimental: MapAnything proposals inside a room, verified by depth (OFF by default, P-2c) |
| `floorplan/photo/frontend.py` | The pipeline: steps 1-7 below |
| `floorplan/photo/evaluate.py`, `report.py` | Evaluation against the simulated-photo truth + LiDAR (never an input) and the top-view figure |
| `scripts/run_photo_frontend.py` | One command: photo folders → `outputs/photo_tier/<name>/scene.npz` + `scene_info.json` (+ evaluation) |

How to run (from the repo root):

```bash
PY=../.venv/bin/python
$PY scripts/run_photo_frontend.py outputs/photo_inputs/single_scan_with_ceiling__c7d28f72c6/photos
#  -> outputs/photo_tier/single_scan_with_ceiling__c7d28f72c6/{scene.npz, scene_info.json, evaluation.json,
#     photo_vs_lidar_topview.png, work/}
$PY scripts/run_photo_frontend.py ../TakeHome/OwnCaptures/photos --out outputs/photo_tier/own_home   # native photos
```

Runtime: 100-220 s for 46 photos (feature matching ~45 s on the GPU, MoGe-2 + GeoCalib ~30 s, the rest CPU).
GPU peak about 2 GB (MoGe-2 fp16 at 518 px); the code waits for free GPU memory instead of crashing (shared GPU).

---

## 1. Purpose and where it sits

```
photos/<room>/*.jpg ──► PHOTO FRONT-END ──► scene.npz (+ scene_info.json) ──► the SAME plan back-end as LiDAR
 (2-8 per room, no depth,                    points/normals/colors, raw_points/raw_range/raw_ray,
  no poses, any phone)                       traj, kf, T_align, T_wc + photo keys (cam_room, point_room, ...)
```

The photo tier is the thinnest input of the three. The case study wants it to produce the same deliverable as the
other tiers (rooms, walls, openings, intervals) and, specifically for photos, that **per-room folders stitch into
ONE plan with correct adjacency and no overlaps**, wall lengths within 8%, and honest calibration ("confident
garbage on thin input caps your total score").

This module only produces **metric geometry in one aligned frame**, in the format the LiDAR scene builder writes
(`floorplan/pipeline/scene.py`), so that plan extraction, uncertainty, damage and export are shared across tiers
(D-003). Per-photo extras let the back-end know which room each point and camera came from and how trustworthy
each room's placement is.

## 2. How it works, step by step

1. **Load the photos** (`images.py`).
   - What: every sub-folder is a room; JPEG/PNG/HEIC are read, rotated upright with EXIF Orientation, and the
     focal length is taken from EXIF `FocalLengthIn35mmFilm` (f_px = f35 / 36 mm × long side).
   - Why: learned depth needs upright images and the true field of view; a 10% focal error is a ~10% depth error.
   - Without it: sideways images and a guessed field of view; MoGe-2 then guesses the focal itself (worse).
   - Absent EXIF: a 70° field-of-view default, recorded per photo in `info.focal_source` (honest widening).
2. **Match every photo against every other photo, across rooms** (`sfm.py`).
   - What: ALIKED keypoints + LightGlue matching on all N(N-1)/2 pairs (1,035 pairs for 46 photos), then COLMAP's
     epipolar RANSAC keeps only geometrically consistent matches. A joint SfM is also run, but only as a diagnostic.
   - Why: the doorway "looking out" photos of the protocol share features with the next room's photos; a match
     between two rooms is direct image evidence of how they connect. Exhaustive pairs cannot miss such a link.
   - Without it: rooms could only be placed by guessing (fallback, step 6).
3. **A metric, levelled point cloud per photo** (`depth.py`, `views.py`).
   - What: MoGe-2 predicts metric depth and normals per photo (EXIF field of view as input); GeoCalib predicts the
     gravity direction; the floor plane in the photo's own depth (if visible) refines "up".
   - Why: once each photo is levelled, placing it needs only 4 numbers (yaw, x, y, z) plus a scale correction,
     instead of 7. Gravity from a plane over metres is more precise than a single-image network (same reasoning as
     LiDAR D-009). 38 of 46 sample photos were levelled by their floor.
4. **Metric edges between photos, verified three ways, then one pose graph** (`link.py`).
   - What: each verified match is lifted to 3-D in both photos; a RANSAC gravity-constrained similarity (2-point
     minimal sets: yaw, shift, scale) aligns the two photos. The edge is then checked:
     1. **scale sanity**: the two photos' learned depth scales must agree within 33%;
     2. **dense free-space check**: photo a's whole depth map is moved into photo b; if > 20% of the points land
        clearly IN FRONT of the surface b observed, the edge is a contradiction and is rejected;
     3. **loop consistency**: the graph is solved, the edge that disagrees most with the solution is removed, and
        this repeats while any edge disagrees by more than max(30 cm, 15%) or 8°;
     4. **weak bridges**: an edge that alone connects two photo groups must carry ≥ 20 inliers.
     The surviving edges form a graph; a maximum spanning tree initialises the poses, robust least squares
     (soft-L1) refines yaw, position and a per-photo scale correction over all edges.
   - Why: feature matches can be consistent and still wrong (repeated doors, identical white cabinets, mirrors).
     Every check removes a class of false links that the previous one cannot see (evidence in section 3, P-5).
   - Without it: 16 of 51 raw edges on the sample were wrong (up to 6 m off), and one wrong edge in the spanning
     tree moves whole rooms by metres (first run: camera error median 1.76 m, relative rotation error median 97°).
5. **Global scale** (`scale.py`).
   - What: inverse-variance fusion (in log space) of every available cue: MoGe-2's metric scale (σ 4%) and the
     chest-height camera prior (1.35 ± 0.15 m). The paper sheet (`floorplan.scale.observe_image()`, D-056) enters
     only with `PhotoParams.use_sheet=True`: off by default since D-067. The fused σ is reported and inflated if the
     cues disagree (χ²).
   - Why: photos carry no scale; learned scale is only good to a few percent. The sheet (D-013) was meant to be the
     strong cue; since D-067 no capture uses one.
6. **Rooms that no photo links** (`frontend._place_components`).
   - What: the largest linked group of photos is the anchor block. A group containing rooms not yet placed is rotated
     onto the block's Manhattan axes and put **beside** the block (no overlap, floors level), flagged
     `placement = fallback_beside_block`, with an interval-widening factor of 4 instead of 2. A group of photos whose
     rooms are already placed but which has no link to them cannot be placed and is listed in
     `info.unplaceable_fragments` (its geometry is not used: no guessing inside a measured room).
   - Why: the deliverable must be one frame, but a guessed position must never look like a measured one.
7. **Alignment and fusion** (`frontend.build_scene_from_photos`).
   - What: all photos' points are moved into the common frame and scaled; the LiDAR tier's own
     `floorplan.plan.align.align_scene` levels the scene on the floor and rotates walls onto x/z; a 2 cm voxel mean
     gives the surface (`points/normals/colors`), a 1 cm grid the measurement points (`raw_points`, `raw_range`,
     `raw_ray`). Final per-photo metric depth maps are saved for damage mapping (`work/depth_final.npz`).
   - Why: same alignment code, same scene format → the same plan extractor runs unchanged.

### Output contract (what other modules may rely on)

`scene` (saved with `floorplan.pipeline.scene.save_scene`):

| Key | Meaning |
|---|---|
| `points, normals, colors` | fused surface (2 cm voxel mean), aligned frame (+y up, walls on x/z); normals face the camera |
| `raw_points, raw_range, raw_ray` | 1 cm de-duplicated metric points, their range from the photo, unit ray camera→point |
| `T_align` | photo-common frame → aligned frame |
| `traj`, `kf` | camera centres of the PLACED photos (aligned frame); `kf = arange(n_placed)` |
| `T_wc` | (n,4,4) camera-to-world of placed photos (OpenCV camera axes) |
| `cam_names, cam_room, cam_placement` | per placed photo: file, room folder, `linked` or `fallback_beside_block` |
| `room_names, point_room, raw_point_room` | room list and per-point room index (the room folder of the photo that saw it) |

`info` (scene_info.json): `floor_y, ceiling_y, ceiling_height_global`, `scale = {factor, sigma_rel, cues,
sheet_status}`, `rooms = {room: {photos, placed, placement, widen}}`, `links` (room pair → edges, inliers, photo
pairs), `all_rooms_one_frame`, `components`, `unplaceable_fragments`, `edges`, `pruned_edges`, `edge_residuals`,
`per_photo_scale_correction`, `up_source`, `focal_source`, `sfm`, `interval_widen_factor`, `scale_sigma_rel`.

## 3. Decisions

### P-1 Link rooms by verified feature matches across ALL photos, not by per-room reconstruction

- **Options.** (a) Reconstruct each room separately, then place rooms by doorway geometry. (b) Joint SfM over all
  photos (hloc + COLMAP). (c) Exhaustive cross-room matching, but solve placement with metric edges (matches lifted
  by learned depth) instead of SfM.
- **Evidence.** Joint SfM on the 46 sample photos registered **17/46 photos in 5 models** (rerun: 18 in 5), with a
  focal 9% off the truth. The protocol's photos are wide-baseline and few; incremental SfM needs a chain of
  overlapping triangulated points that these photos do not have. But 96-97 photo pairs have ≥ 15 epipolar-verified
  matches, 37 of them across rooms.
- **Decision.** (c). A pair of photos needs only its own matches + each photo's depth to give a metric relative pose:
  no third photo, no triangulation. SfM stays as a logged diagnostic (`info.sfm`).

### P-2 MapAnything is not the geometry source here (measured), MoGe-2 depth is

- **Experiment** (`floorplan/photo/recon.py`, run per room, photo depth vs the LiDAR depth of the source frame):

  | Model | per-photo depth ratio vs LiDAR | relative rotation error per room (median) |
  |---|---|---|
  | MapAnything, images + EXIF intrinsics | 0.65-0.94 (all rooms short) | R1 85°, R2 11°, R3 44°, R4 5°, R5 10°, R6 4°, R7 85°, R8 64°, R9 24° |
  | MapAnything + MoGe-2 depth as input | 0.68-1.05 | similar (R2 9°, R9 67°) |
  | MoGe-2 alone (EXIF focal) | 0.74-1.06, typically 0.92-1.0 | (single image) |

  MapAnything also ignored the given focal in its output (predicted fx/W 0.75-1.25 vs true 1.11).
- **Decision.** MoGe-2 for per-photo metric depth (agrees with the video tier's V-5 finding, 0.95 median), poses
  from matched pairs (P-1). MapAnything is kept as a runner for experiments only.
- **P-2c (rejected after measuring).** MapAnything as a *proposal* for same-room photo pairs without matches, reduced
  to 4 DoF, refined by ICP, accepted only if the dense check passes and the photos overlap: 87 proposals, 6 accepted,
  **3 of the 6 still 0.6-2.5 m wrong** (sliding along a wall passes a depth check). So `intra_room_proposals=False`.

### P-3 Gravity: GeoCalib per photo, refined by the floor plane

- **Why.** Levelling makes every edge 4-DoF (fewer unknowns, more robust with 12-50 inliers) and makes heights
  meaningful. A plane fitted to thousands of floor points over metres is more precise than a single-image network.
- **Evidence.** 38/46 photos found their floor and were refined; final relative rotation error between linked photos
  median 3.7°, p90 6.1° (this includes yaw error, not only tilt). The final scene's floor tilt before alignment
  was 0.27° (`info.floor_tilt_before_deg`).

### P-4 Scale: fuse all cues; pluggable paper sheet

- **Status (D-067).** The sheet cue is opt-in and off by default (`PhotoParams.use_sheet = False`). No capture asks
  for a sheet.
- **Options.** Learned scale only; priors only; sheet only; fusion.
- **Decision.** Fusion in log space with χ²-inflation (`scale.fuse`). Current cues: MoGe-2 (σ 4%), camera height
  prior (σ 11%), sheet (when available). The sheet adapter expects
  `floorplan.scale.observe_image(rgb, K, depth, up_cam) -> [ {scale, sigma_rel, ...} ]` (scale multiplies that
  photo's depth so the sheet has its true size); each observation is mapped to the common frame through the photo's
  own scale correction. If the module or the function is missing, `info.scale.sheet_status` says so and the
  pipeline runs on (D-013: "any picture in, results out").
- **Why not "correct" MoGe's measured −3.5% bias?** That bias was measured on this apartment; applying it would be
  tuning on the test set. It stays inside σ.
- **Evidence.** Final per-photo depth ratio vs LiDAR median 0.966 (p10-p90 0.88-0.99); camera-height cue
  1.35 m median (agrees, scale 0.999); global scale error from camera positions −5.2% with σ 3.8% (1.4σ).

### P-5 Three independent filters against false links

- **Context.** Before filtering, 16 of 51 metric edges were wrong (checked against truth after the fact), some with
  44-61 inliers and 2 cm RMSE: geometrically self-consistent and wrong (repeated structures).
- **Measured discrimination** (dense free-space check, worst direction): wrong edges 0.21-0.54 (most > 0.25), correct
  edges mostly < 0.12; a 20% threshold rejects most wrong edges and loses 3 correct ones. Scale-ratio limit removes
  wrong edges with scale 0.37-0.67. After both: **31 correct / 1 wrong**. The remaining wrong edge (12 inliers) was a
  bridge, which loop consistency cannot test; every wrong edge that passed the dense check had ≤ 19 inliers, hence
  `bridge_min_inliers = 20`. After that: no wrong edge in the linked block (relative rotation p90 6.1°).
- **Trade-off, chosen deliberately.** Fewer rooms linked (4/9 instead of 7/9) but every link is correct. A wrong
  link silently produces a confident wrong plan; a missing link produces a flagged, widened fallback.

### P-6 Fallback placement: beside the block, flagged, widened

- **Options.** (a) Doorway matching (find a door gap of equal width in both rooms and snap them). (b) Place beside
  the block under a no-overlap constraint, flag it. (c) Drop unlinked rooms.
- **Decision.** (b) now, (a) as the next step. (a) needs openings, which only the plan back-end detects; it belongs
  after plan extraction, where door widths and wall positions are measured (it can reuse this module's flags).
  (c) would fail "one plan for the property".
- **What is still measured for a fallback room.** Its own geometry (walls, sizes) at the photo tier's scale; only
  its position relative to the other rooms is a guess. `info.rooms[r].widen = 4` vs 2 for linked rooms.

### P-7 Same scene format and alignment code as LiDAR

- `align_scene` (D-009) is reused unchanged; the plan extractor needs only the documented keys. Normals face the
  camera (floors up, ceilings down), exactly like TSDF normals, so floor/ceiling detection works unchanged.

## 4. Issues log

### PT-1 Ground-truth poses looked 90° off for every photo
- **Symptom.** MapAnything relative rotation errors of 85-140° in every room.
- **Root cause.** The simulator rotates video frames upright (`np.rot90`), which rolls the camera; the truth file
  stores the pose of the un-rotated frame. Evidence: image-up in world was horizontal before the fix, 0.87-0.98
  vertical after.
- **Fix.** `evaluate.truth_T_wc`: R_photo = R_frame (A^k)ᵀ with A the quarter-turn matrix. Evaluation-only.

### PT-2 Similarity alignment of 4-6 cameras was meaningless
- **Symptom.** Scale "errors" of 100-1600% for rooms whose depth was within 10%.
- **Root cause.** Umeyama on camera centres that span a metre is ill-conditioned. **Fix.** rotation-first
  alignment (chordal mean of orientations, then scale and shift) and relative-rotation errors per camera pair.

### PT-3 Pose graph residuals of 1-3 m on correct edges
- **Symptom.** After the solve, correct edges disagreed with the solution by 1-3 m, and loop pruning then removed
  11 correct edges.
- **Root cause.** After the solve I re-centred the per-photo log-scales (mean 0) without scaling the positions, which
  breaks t_b = t_a + s_a R_a t_ab. **Fix.** Treat it as a global similarity: scale positions by the same factor.
  Residual median after the fix: 3 cm, max 30 cm.

### PT-4 GPU out of memory at model load (shared GPU)
- **Root cause.** MapAnything was moved to the GPU in fp32 (4.9 GB) before the bf16 cast; another agent held 6.6 GB.
  **Fix.** Cast on the CPU first, then move (2.6 GB); `wait_for_gpu()` polls free memory before loading any model.

### PT-5 Few photos placed on the sample (open)
- **Symptom.** 6-8 of 46 photos in the linked block, 4/9 rooms linked; 18-22 unplaceable fragments.
- **Root cause (evidence).** The simulated photos come from a LiDAR walk-through video: camera pitched 25-30° down,
  close to walls, chosen by a wall set-cover, so photos of the same room rarely overlap (verified pairs inside rooms
  are few). Within a room, 75 of 87 photo pairs could not even be checked for overlap.
- **Fix options.** (a) Protocol: require overlap between consecutive photos of a room (a "pan" of 4-6 shots from the
  centre plus corners). (b) Room-layout reasoning (rectangle closure, Manhattan wall matching). (c) Doorway snap (P-6a).
- **Chosen.** (a) proposed to the protocol owner (see section 6), (c) next. Status: open.

## 5. Results

Sample apartment, `single_scan_with_ceiling` (46 simulated photos, 9 rooms; truth used only for scoring):

| Metric | Value | Evidence |
|---|---|---|
| Photos placed by image evidence (linked block) | 6 photos, rooms R2, R6, R7, R8 | `scene_info.json` |
| Rooms in one frame | 9/9 (4 linked + 5 fallback beside the block), `all_rooms_one_frame = false` | `scene_info.json` |
| Global scale error (camera positions) | −5.2% (reported σ 3.8%) | `evaluation.json` |
| Per-photo depth ratio vs LiDAR (final) | median 0.966, p10-p90 0.88-0.99 | `evaluation.json` |
| Camera position error, linked photos | median 8.5 cm, p90 22 cm | `evaluation.json` |
| Relative rotation error, linked photos | median 3.7°, p90 6.1° | `evaluation.json` |
| Photo points → LiDAR surface (rigid, own scale) | median 5.0 cm, p90 16 cm | `evaluation.json` |
| Same after similarity (shape only) | median 3.6 cm, p90 13 cm | `evaluation.json` |
| Ceiling height (global) | 2.97 m vs LiDAR 3.10 m (−4.2%, same sign and size as the scale error) | `scene_info.json` |
| Edge consistency after the solve | median 3 cm / 0.3°, max 30 cm | `scene_info.json` |
| Runtime | 100-220 s | log |

Figure: `outputs/photo_tier/single_scan_with_ceiling__c7d28f72c6/photo_vs_lidar_topview.png` (photo points
coloured by room over the LiDAR surface; red lines = each camera to its true position; the linked rooms sit on the
LiDAR walls, fallback rooms are visibly placed beside the block, as flagged).

Per-room extents inside the LiDAR plan polygons (`evaluation.json → room_extents`) are not yet a fair wall-length
score: rooms are only partly covered by placed photos. The real wall-length gate (±8%) is scored after the plan
extractor runs on `scene.npz`.

### 5b. Other sample captures (photos simulated from their LiDAR plans by `make_photo_folders.py`)

| Capture | Photos / rooms | Linked block | Scale (depth ratio vs LiDAR) | Cam error (linked) | Points → LiDAR (rigid) |
|---|---|---|---|---|---|
| `single_scan_floor_only` | 21 / 7 | 6 photos, rooms R1, R3, R6; R2, R4, R5, R7 fallback | median 0.950 (p10-p90 0.86-1.09) | median 12 cm, p90 17 cm; rel. rotation 2.9° / 11° | median 3.7 cm, p90 11 cm |
| `single_room` | 7 / 2 | none: 0 verified pairs among 7 photos; one photo per room kept, both flagged | n/a | not scorable | n/a |

Two caveats, measured: (1) the "global scale error" computed from camera positions (+10.3% on floor_only, −5.2%
on with_ceiling) is noisy with 6 cameras a few metres apart; the per-photo depth ratio against LiDAR is the direct
measure (−3.4% to −5%) and it agrees with the reported σ ≈ 4%. (2) `single_room` shows the honest failure mode of
thin, non-overlapping photos: the module does not invent links; it returns one flagged photo per room (widen 4×).
Evidence: `outputs/photo_tier/<capture>/{scene_info,evaluation}.json`, `photo_vs_lidar_topview.png`.

## 6. Limitations, failure modes, next steps

- **Thin overlap** (PT-5) is the dominant limit: placement needs matches, and the dense check needs shared depth.
  Proposed protocol addition: "in each room, also take 3-4 photos turning on the spot in the middle of the room, each
  overlapping the previous by about a third." This adds the overlap chains that link corner photos.
- **Glass, mirrors.** A mirror creates matches and depth "behind" the wall; such edges are what the dense check
  rejects (contradicting depth). Mirror points still enter the cloud from single photos; the plan back-end's mirror
  detection applies.
- **Low light / blur.** Fewer keypoints → fewer edges → more fallback rooms (flagged, not wrong).
- **Clutter.** MoGe-2 depth is fine on furniture, but walls behind furniture are hidden in single photos.
- **Learned-scale bias.** −3.5% on this apartment. Without the sheet, a 4% σ is honest but cannot pass the ±3% video
  gate; for photos (±8%) it is enough. No capture uses a sheet since D-067: on the simulator it moved the photo
  scale by at most 0.9 points.
- **EXIF missing** (WhatsApp-stripped photos): default 70° FOV, recorded per photo; next step GeoCalib focal.
- **Next steps.** (1) Doorway snap for fallback rooms using the back-end's openings (P-6a). (2) Sheet detector:
  integrated (D-056), off by default since D-067. (3) Bundle-adjust the final graph on reprojection error
  with depth priors. (4) Run on the own-home photos (tape ground truth) — the primary photo benchmark.

---

# photo_tier v2 (4 Oct 2026, night): the revised protocol D-017 (spin + doorway pairs)

> Everything above this line is v1 and is kept as history. v2 adds structure from the revised capture protocol to the
> same pipeline; nothing of v1 was removed (the v1 behaviour is `{"spin_prior": false}` and no EXIF times).

New and changed files:

| File | What changed |
|---|---|
| `floorplan/photo/protocol.py` (new) | doorway pairs from EXIF times, spin groups, per-photo Manhattan yaw, prior edges, room-overlap check, wall-layout registration of a doorway photo |
| `floorplan/photo/images.py` | reads EXIF `DateTimeOriginal` + `SubSecTimeOriginal` → `Photo.time_s` |
| `floorplan/photo/link.py` | `Edge.kind` (`features` / `spin_prior` / `doorway_pair` / `layout`) with explicit sigmas; solver, spanning tree, loop pruning and bridge removal respect the kind |
| `floorplan/photo/frontend.py` | protocol edges, two-phase room-overlap verification (with doorway-pair yaw repair), layout registration, door-matching fallback, anchor = component with most rooms, MoGe/GeoCalib view cache, new `info` keys |
| `floorplan/photo/params.py` | v2 knobs, each with its reason / measurement |
| `floorplan/photo/report.py` | reads `capture_name` from the truth file (folders now carry a protocol suffix) |
| `scripts/make_photo_folders.py` | `--protocol rotate|corners` (default rotate), EXIF capture times, disclosed simulation |
| `outputs/photo_tier/v2/*.py` | evidence scripts: `edge_truth_check.py`, `overlap_experiment.py`, `evaluate_v2.py`, `debug_pair_overlap.py`, `run_batch.sh` |

Run (same command as v1; corner-style folders need `--params outputs/photo_tier/v2/params_corners.json`):

```bash
PY=../.venv/bin/python
$PY scripts/make_photo_folders.py ../TakeHome/Dataset/single_scan_with_ceiling/c7d28f72c6 \
    outputs/plan_beta/single_scan_with_ceiling__c7d28f72c6/plan.json outputs/single_scan_with_ceiling__c7d28f72c6 --protocol rotate
$PY scripts/run_photo_frontend.py outputs/photo_inputs/single_scan_with_ceiling__c7d28f72c6__rotate/photos
$PY outputs/photo_tier/v2/evaluate_v2.py            # corners vs rotate table + plan_beta on the photo scenes
```

## v2.1 Purpose

v1 linked only 3-4 of 9 rooms by image evidence: corner/doorway shots of one room share few features. D-017 changed
the protocol: per room ~6 photos **turning on the spot** near the middle, plus **doorway pairs** (standing in a
doorway, one photo back into the room being left and one into the next room, seconds apart, each filed under the
room it looks into). v2 makes the front-end use that structure, re-simulates the sample photos under the new
protocol, measures corners vs rotate, and gives a verdict on the protocol before the 08:00 capture.

## v2.2 How it works, step by step (only what is new; v1 steps 1-7 are unchanged)

1. **Capture times** (`images.py`). EXIF `DateTimeOriginal` (+ `SubSecTimeOriginal`) becomes `Photo.time_s`. Only
   the *original* time is used (`DateTime` is the edit time). Missing → `None` (WhatsApp-stripped photos).
2. **Doorway pairs** (`protocol.doorway_pairs`). All photos sorted by time. Two photos form a pair when they are
   in different folders, **consecutive** in time (nothing shot in between), and closer in time than
   `min(15 s, 2 × the photographer's median same-room interval)`. Each photo joins at most one pair.
   - Why consecutive: the last spin shot of room A and the doorway photo into B can be < 15 s apart.
   - Why the rhythm limit: walking from one room's spin to the next room's spin took 12 s in the simulated schedule
     and a flat 15 s limit paired them (measured: 3 false pairs on floor_only); a doorway pair is shot at the same
     rhythm as spin shots.
3. **Spin groups** (`protocol.spin_groups`). A room's photos that are not in a pair, in time order. No EXIF times →
   no spin prior for that room (we cannot tell spin shots from doorway shots), recorded in `info`.
4. **Per-photo Manhattan yaw** (`protocol.photo_manhattan`). Each photo's levelled MoGe-2 wall normals give the
   wall direction modulo 90° in the photo's own frame. For a photo whose walls are on the room axes, its pose yaw
   θ ≡ m (mod 90°). This is *measured* per photo, not assumed.
5. **Spin prior edges** (`protocol.spin_edges`). Consecutive spin photos get an edge: shared centre (σ 0.3 m:
   arm + body sway), yaw step = the Manhattan-consistent value nearest to the protocol's −60° (clockwise), σ 10°
   (25° if a photo sees too little wall), scale tie σ 0.05. Feature edges inside the group are kept; if they
   contradict a prior, loop pruning removes the prior (its badness is measured in its own sigmas).
6. **Doorway-pair edges** (`protocol.pair_edge_prior`). Shared centre (σ 0.2 m), relative yaw ≈ 180° ("turn
   round") snapped to the two photos' Manhattan axes (σ 6°). No feature match is needed.
7. **Cross-room link checks** (`frontend._features_need_pair`, `frontend._verify_pairs`). (a) Before loop pruning:
   when the capture has doorway pairs, a cross-room FEATURE edge is kept only between two rooms that a doorway pair
   joins (every opening walked through has a pair; measured 4/4 false removed, 9/9 true kept, P-11). (b) After loop
   pruning and weak-bridge removal: every DOORWAY PAIR is judged with only its two rooms' intra-room edges; if room
   B's surfaces then stand > 30% and > 1.5 m inside the free space room A's photos observed (score > 0.35), the
   pair's other three Manhattan yaws are tried (the photographer may not have turned a full 180°) and the yaw with
   the least overlap wins; a pair that still fails is removed, worst first.
8. **Layout registration** (`protocol.register_to_room`). A doorway photo that shares no features with its own
   room's spin group is aligned to that group's walls in top view: 4 Manhattan yaws × 5 scales, FFT correlation
   over translation, camera constrained to the room's wall line (it stands in a doorway). Accepted only if the score
   is ≥ 0.45, beats every rival placement by ≥ 15%, and passes the free-space test against all spin photos.
9. **Door-matching fallback** (`frontend._door_match`), before "beside the block": doorway photos stand in doors,
   so a pair whose partner sits in another component gives a door position on both sides; every (door, door,
   Manhattan rotation 0/90/180/270) placement is tried and the unique one with no overlap (< 5% of 10 cm cells)
   wins (`placement = door_match`, widen 3.5). Otherwise v1's beside-the-block fallback (widen 4).
10. **Anchor** = the component spanning the most rooms (v1: most photos; a big single-room spin group must not
    displace a multi-room block).
11. **New `info` keys**: `protocol_v2 = {doorway_pairs, spin_groups, pair_checks, layout_registration, door_match,
    photo_link_kinds}`, `room_links_v2 = {"A|B": {kind: count}}`, `edges[*].source/yaw_deg/note`,
    `rooms[r].placement ∈ {linked, door_match, fallback}`.

## v2.3 Decisions

### P-8 Spin group = shared centre + Manhattan-snapped 60° steps, not pure-rotation estimation from matches

- **Options.** (a) Pure-rotation relative orientation from matches with known K (homography / bearing alignment)
  between consecutive spin photos. (b) Keep v1 metric feature edges and add a shared-centre prior only.
  (c) Shared centre + yaw step from each photo's own Manhattan yaw, snapped to the protocol's step (−60°, clockwise),
  with feature edges added wherever they exist.
- **Evidence** (`outputs/photo_tier/v2/overlap_experiment.json`, real frames of the sample apartment, ALIKED +
  LightGlue + F-RANSAC, same-spot pairs (< 0.3 m) with < 10° pitch difference):

  | horizontal overlap | 0.82 | 0.65 | 0.49 | 0.33 | 0.19 | 0.03 |
  |---|---|---|---|---|---|---|
  | pairs ≥ 15 verified matches | 5/5 | 6/6 | 60% | 75% | 36% | 0% |

  With the Nord's 1× lens in landscape (~69° HFOV, to be confirmed from my phone's EXIF) a 60° step leaves
  ~13% overlap: feature links between consecutive spin photos will be the exception, so (a) alone would leave most
  spin photos unconnected. Same-spot pairs whose pitch differs by ≥ 20° matched in only 24% of cases even at < 20°
  heading change (n = 45): **a level phone matters as much as the overlap**.
- **Decision.** (c). The yaw comes from geometry measured in every photo (wall normals), the protocol only resolves
  the 90° ambiguity. Measured per-step yaw error on the simulated spins: see v2.5 (most within ±10°, worst 16°;
  the walk-through-derived "spins" have 40-90° steps, a real spin is more regular).

### P-9 Turning direction fixed by the protocol (clockwise), not inferred

- **Evidence.** With direction inferred from the Manhattan residuals ("auto"), 2 of 13 simulated rooms were resolved
  in the wrong direction (floor_only R1: steps of 40-45°, where ±45° are indistinguishable modulo 90°; with_ceiling R5:
  a 92° step), giving 89-160° yaw errors for the whole room. Fixing the direction removes the failure; the cost of
  the other direction is still computed and a `direction_warning` is raised if it fits clearly better.
- **Consequence for the protocol.** The guide must say **"turn clockwise (to your right)"** — it already says "like a
  clock hand"; make it explicit.

### P-10 Doorway pair = consecutive-in-time + rhythm limit; yaw ≈ 180° snapped, repaired by no-overlap

- **Evidence.** Pairing by "within 15 s" alone produced 3 false pairs on floor_only (12 s room-to-room walks);
  consecutive + `≤ 2 × median same-room interval` (8 s here) gave exactly the 3 simulated pairs. True pairs had yaw
  errors of −0.8° to 4.0° when the photographer turned ~180°; one simulated pair turned only 126° and was mis-snapped
  by 90°; the no-overlap repair found the right yaw (error 2.5°).
- **Why it is strong.** It links two rooms with no shared features at all; its uncertainty (0.2 m, 6°) is set by how
  still a person stands, not by image content (white walls, dark rooms do not matter).

### P-11 Cross-room links: feature edges must join rooms that a doorway pair joins; pairs get a room-overlap test

- **First attempt (measured, then replaced).** A room-level dense free-space check (every photo of room A against
  every photo of room B) on every cross-room link. On floor_only it removed the two false R5~R6 edges (90° wrong,
  6.6-7.2 m) that had made loop pruning delete a correct 48-inlier edge. On with_ceiling it rejected 6 CORRECT
  feature edges (one with 285 inliers, 0.1° error) and kept a false one. Judged in isolation on all 13 truth-labelled
  cross-room feature edges (`outputs/photo_tier/<capture>__rotate/room_check_discrimination.json`): false edges
  scored 0.06-1.0 (strict) / 0.0-0.30 (gross), true edges 0.0-1.0 / 0.0-0.59: **no separation**. Rooms placed by
  priors (0.2-0.5 m, 5-10°) are not precise enough for a dense test.
- **What does separate them.** Under D-017 every opening the photographer walked through has a doorway pair. All 4
  false edges joined rooms with no pair between them; all 9 true edges joined paired rooms.
- **Decision.** With ≥ 1 doorway pair in the capture, a cross-room feature edge is kept only between paired rooms,
  applied BEFORE loop pruning (otherwise a false edge can win against a true pair). Without any pair (no EXIF
  times, corner-style folders) feature edges are handled exactly as in v1.
- **Doorway pairs** still get the room-overlap test, gross version (point > 30% and > 1.5 m in front of the other
  room's surface, threshold 0.35), because its job there is different: pick the right one of four Manhattan yaws
  when the photographer did not turn a full 180°. Margins measured on the pair R3_07~R6_01 (correct yaw vs the three
  wrong yaws): strict 15% test 0.94 vs 0.98-1.0 (useless); 0.6 m → 0.72 vs 0.95-1.0; 1.0 m → 0.55 vs 0.84-0.97;
  **1.5 m → 0.002 vs 0.50-0.93**. Each link is judged with only its two rooms' intra-room edges (one false link must
  not make another look contradictory).
- **Caveats.** The pair rule trusts the protocol: a doorway walked through without a pair loses its feature links
  (they come back as a fallback placement, flagged). Thresholds were set on one apartment.

### P-12 Layout registration of an unmatched doorway photo: strict acceptance

- It never fired on the sample (0/3 and 0/6 in the final runs): the best wall alignment was always within 15% of a rival placement
  (sliding along a wall, P-2c's lesson). Accepting the best anyway would place rooms by a coin flip, so it stays
  strict and reported (`info.protocol_v2.layout_registration`). The doorway-on-the-wall-line constraint was added
  but did not resolve the ambiguity.

### P-13 Door-matching fallback uses doorway-photo positions, not detected openings

- The task asked for "match openings by width and position under a Manhattan rotation". The front-end has no
  openings (only the plan back-end detects them). A doorway photo's camera centre *is* a door position, so v2
  matches those under 4 rotations with a no-overlap test. Widths cannot be compared without openings; that needs
  per-room plan extraction (see v2.6). On the sample it did not trigger (no dangling pairs).

### P-14 The simulator keeps the honest imperfections of a walk-through

- Spin "groups" are frames within 0.4 m with headings in contiguous ~60° sectors (±20°); the walk-through rarely
  offers 6 such headings: 2-5 per room, cluster 0.1-1.5 m from the room centre (recorded per room).
- Doorway pairs are two frames < 0.8 m from the door centre looking into each room with ≥ 120° between them; their
  real time gap (8-195 s) is replaced by protocol-ordered EXIF times, **disclosed** in `photo_truth.json`
  (`exif_time_simulated: true`, `doorway_pairs[*].real_time_gap_s`, `dist_m`).
- Upright Stray frames are portrait (48° HFOV) vs the Nord's ~69° landscape: the simulation is pessimistic on
  overlap.

## v2.4 Issues log (v2)

### PT-6 False doorway pairs from fast room-to-room walks
- **Symptom.** 6 pairs on floor_only/rotate, only 3 simulated. **Root cause.** The 3 extra were the last spin shot
  of a room and the first spin shot of the next, 12 s apart (< 15 s). **Fixes.** (a) tighter fixed limit; (b) rhythm
  limit relative to the photographer's own pace; (c) geometric check only. **Chosen.** (b) + consecutive-in-time +
  (c) as a second line (P-11): adapts to slow and fast photographers; protocol note "take the pair quickly".

### PT-7 Spin direction ambiguity
- **Symptom.** floor_only R1 spin 89° wrong, with_ceiling R5 90-160° wrong. **Root cause.** Steps of ~45° (and
  90°) are ambiguous modulo 90°; auto-direction picked the other sense. **Chosen.** Direction fixed by protocol (P-9).

### PT-8 Loop pruning removed a correct edge; a dense room check did not fix it in general
- **Symptom.** floor_only: R6_01~R6_04 (48 inliers, correct) pruned. **Root cause.** Two false R5~R6 feature edges
  agreed with each other (same repeated structure), so in their triangle the correct edge had the largest residual.
- **Fix tried.** Dense room-overlap check of cross-room feature edges: fixed floor_only, broke with_ceiling (6 correct
  edges rejected). **Chosen.** Pair-consistency rule before pruning (P-11): 4/4 false removed, 9/9 true kept.

### PT-9 Correct doorway pair rejected by the dense check
- **Symptom.** R3_07~R6_01 rejected at 0.94-0.99 for every yaw. **Root cause** (`debug_pair_overlap.py`):
  under TRUE poses the violation is 0.04; under solved poses R6's photos are 0.3 m and 16% in scale off and see
  R3's surfaces through the door, so their points land 0.9 m in front of R3's surface. The check was designed for
  cm-level feature edges. **Chosen.** Gross-error margin for pairs (P-11), scale prior σ 0.10 → 0.05.

### PT-10 Verification loop skipped already-reported edges (bug)
- **Symptom.** A false edge flagged with violation 0.55 stayed in the graph. **Root cause.** The loop skipped edges
  already in its report after the first removal. **Fix.** Re-evaluate every edge in every round.

### PT-11 Anchor chosen by photo count
- **Symptom.** A falsely linked 2-room block of 8 photos was the anchor. **Fix.** Anchor = most rooms, then photos.

### PT-12 Shared GPU full (other agents at 6.2 GB)
- OOM inside MoGe-2 / ALIKED. **Fix.** Batch runner retries on OOM; MoGe/GeoCalib views are cached in the work dir
  (re-runs of the graph stages take ~1 min instead of 5); the overlap experiment ran on the CPU.

### PT-13 A slightly wrong feature edge beat a correct doorway pair in loop pruning (fixed by P-11)
- **Symptom.** with_ceiling: the false edge R2_05~R5_04 (55 inliers, 24.7° wrong) survived every check and placed R2
  1.8 m off. **Root cause.** It looked as good as true edges to the dense check (0.056). **Fix.** Pair rule (P-11):
  R2 and R5 share no doorway pair → removed before pruning. Remaining: the R2~R7 pair is still pruned by loop
  consistency (badness 3.5) against three correct R2~R7 feature edges (its Manhattan snap was off) — the intended
  behaviour.

### PT-14 Missing EXIF focal reported a 3.8% scale sigma (confident garbage)
- **Symptom.** `phone_like/photos_no_exif`: ceiling 3.59 m vs 2.93 m for the same photos with EXIF (+22%), reported
  σ 3.8%. **Root cause.** The 70° default FOV vs the true 48° (portrait frames) changes MoGe-2's metric scale; the
  learned-depth cue kept its 4% sigma. **Fix.** If any placed photo has a guessed focal, the learned-depth cue gets
  σ 20% (`scale_sigma_no_focal`). After the fix: fused σ 9.7% (with the camera-height cue), ceiling 3.73 m vs LiDAR
  3.10 m = +20%, i.e. 2σ: honest but still wide. Next: GeoCalib focal estimate.

## v2.5 Results

All on the sample apartment; truth (ARKit poses, LiDAR plan) used only for scoring. Both protocols re-simulated with
the same current LiDAR plan (`outputs/plan_beta/<capture>/plan.json`, 7 rooms). "Corners" = v1 protocol run with the
v2 code and `{"spin_prior": false, "doorway_pair_max_dt_s": 0}` (no spin or pair structure exists in those folders).
Evidence: `outputs/photo_tier/v2/comparison.json` (from `evaluate_v2.py`), `outputs/photo_tier/<capture>__<protocol>/
{scene_info,evaluation}.json`, `outputs/photo_tier/v2/edge_truth_*.tsv`.

**Whole-property stitch**

| Capture / protocol | photos | rooms in the anchor block | other multi-room blocks | room links claimed: correct / wrong | LiDAR door adjacencies found | camera error in block, median / p90 | relative rotation, median / p90 | runtime |
|---|---|---|---|---|---|---|---|---|
| with_ceiling, corners | 36 | 3/7 (R2, R3, R6), **wrongly joined** | R1-R2, R4-R7 | 3 / 2 | 3/6 | 1.60 m / 2.51 m | 91° / 91° | 9 s* |
| with_ceiling, rotate | 41 | 4/7 (R2, R4, R5, R7) | R1-R3-R6 (pairs) | **5 / 0** | **5/6** | 0.77 m / 3.12 m | 7.2° / 26° | 78 s* |
| floor_only, corners | 23 | 3/7 (R1, R3, R6) | R4-R7 | 3 / 0 | 3/5 | **0.08 m / 0.10 m** | **3.8° / 7.2°** | 7 s* |
| floor_only, rotate | 28 | 2/6 (R3, R6) | R1-R2 (pair + features) | 2 / 0 | 2/5 | 0.23 m / 0.35 m | 9.5° / 23° | 60 s* |

\* re-runs with cached features/depth; a cold run is 150-280 s on the shared GPU. Corner runs use
`{"spin_prior": false, "doorway_pair_max_dt_s": 0}` (no spin or pair exists in those folders). The same corner folders
run with time pairing switched ON produced 12 / 5 "pairs" from video timestamps, rooms 100% overlapping and 87°
median rotation error (`outputs/photo_tier/v2/comparison_corners_with_time_pairing.json`): **pairing must only be
used when the photographer followed the pair protocol** (`doorway_pair_max_dt_s = 0` otherwise).

Reading the table honestly: no run reaches "all rooms in one frame", so the photo-tier stitch gate still fails on
the simulated sample with either protocol. Rotate gave the only error-free adjacency set on with_ceiling (5/5
correct, 5 of 6 doors) where corners joined wrong rooms; on floor_only corners was more accurate (feature-linked
photos 8 cm vs prior-linked 23 cm) but linked one room more. Rooms not in the anchor block are placed beside it,
flagged `fallback`, widen 4 (v1 behaviour).

**What the protocol structure measures (rotate runs, both captures)**

| Element | Result | Evidence |
|---|---|---|
| Doorway pairs found from EXIF times | 3/3 (floor_only) and 6/6 (with_ceiling), 0 false (3 false with a flat 15 s limit) | `info.protocol_v2.doorway_pairs` |
| Doorway-pair relative yaw error | −0.8° to +4.0° for all 8 pairs that stayed in the graph (one after the no-overlap yaw repair: snapped 90° off → 2.5°); true centre distance 0.01-0.59 m | `edge_truth_*.tsv`, `photo_truth.json` |
| Doorway-pair adjacency | 8/8 kept pairs join truly adjacent rooms; the 9th was removed by loop consistency against 3 correct feature edges | `room_links_v2`, `pruned_edges` |
| Spin-step yaw error (38 steps) | median 4.5° (floor_only) / 2.6° (with_ceiling); 31/38 within 10°; worst 70° on a simulated 92° step (a real 60° spin cannot produce it; that step was down-weighted to σ 25°) | `edge_truth_*.tsv` |
| Spin-group camera spread (truth) | 0.3-0.8 m within a group, cluster 0.1-1.5 m from the room centre | `photo_truth.json → selection[*].spin` |
| Doorway photo attached to its own room's spin group | by features 7/18; by wall-layout registration 0/9 (always ambiguous, rival within 15%); pair bridge 0/3 (10 and 13 joint hypotheses pass the no-overlap test) | `layout_registration`, `pair_bridge` |
| Cross-room feature edges (truth-labelled) | 4 false (all between unpaired rooms) removed, 9 true (all between paired rooms) kept | `cross_room_edges_*.txt`, `room_check_discrimination.json` |

**Per-room size.** The bounding-box proxy of each linked room's wall points vs the LiDAR plan
(`linked_room_extent_err_pct`) is −82%…+487%: points seen through doors and from rooms placed with prior-quality
poses dominate a 2-98 percentile box, so this proxy is not a wall-length score. The real number needs the plan
extractor (next item).

**plan_beta on the photo scenes** (`extract_plan(scene, info, ...)`, unchanged, `comparison.json`):
with_ceiling/rotate → 3 rooms (5.5, 2.2, 1.8 m²), no adjacency, footprint 9.4 m² (the LiDAR plan has 7 rooms);
floor_only/rotate → 6 rooms (7.5, 6.6, 6.0, 3.8, 1.8, 1.3 m²), 3 adjacencies via openings, 4 openings, footprint
27.0 m²; with_ceiling/corners → 2 rooms; floor_only/corners → 1 room (an earlier run with false time pairs crashed it:
`ValueError: max() arg is an empty sequence`). The extractor runs on photo scenes, but on sparse single-view geometry
it segments few or split rooms and it ignores the folder labels, so its rooms cannot be matched to the photo rooms
and wall lengths cannot be scored yet (`outputs/photo_tier/<run>/plan_beta_on_photo_scene.json`).

**Interface change plan_beta needs to use per-room labels** (not made: not my file):
1. Read `scene["cam_room"]`, `scene["traj"]` and `scene["raw_point_room"]` (already written by the photo tier).
2. Seed the watershed with ONE marker per room folder (that folder's camera centres) instead of distance-transform
   maxima, and forbid merging two regions whose markers belong to different folders.
3. Name output rooms after the folder (`Room.label = folder`) so scope items keep the operator's room names.
4. Accept `info["rooms"][folder]["widen"]` and `["placement"]` and multiply that room's interval half-widths by
   `widen` (linked 2, door_match 3.5, fallback 4), and mark adjacency between rooms that touch only because of a
   fallback placement as `status="inferred"`.
5. Free space: treat a photo camera's 2-D position as "visited" (it already does via `traj`), but carve only up to
   each raw point (`raw_range`); photo points are sparse, so `fill_furniture_holes` and the minimum-component area
   need photo-tier parameters (`BetaParams` per tier).

**Phone-file robustness** (`outputs/photo_tier/v2/phone_like_*`, corner-style folders of the sample, 46 photos):

| Input | Loaded | Focal source | Levelled by floor | Rooms linked | Capture times | Scale σ |
|---|---|---|---|---|---|---|
| `photos_oneplus_exif` (sensor-order pixels + EXIF Orientation 6) | 46/46 upright (1440×1920) | EXIF f35 46/46 | 38/46 | 4/9 (R2, R6, R7, R8) | none in files → no pairs/spin (logged) | 3.8% |
| `photos_no_exif` | 46/46 | default 70° FOV 46/46 | 37/46 | 4/9 (same rooms) | none | **9.7%** after PT-14 fix (was 3.8%) |
| `photos_heic` | 46/46 (pillow_heif) | EXIF f35 46/46 | 38/46 | 3/9 (R2, R6, R8) | none | 3.8% |

Runtime 74-85 s each (cold: features, MoGe-2 and GeoCalib computed; shared GPU).

## v2.6 Limitations and next steps

1. **The blocking link is "doorway photo ↔ its own room's spin".** Pairs link rooms reliably (±4°), spins give each
   room a layout (median 3-5°), but a doorway photo must also attach to its room. Features did so 4/16 times on the
   simulation (portrait 48° FOV, walk-through frames); wall-only registration is ambiguous along walls.
   **Next (code, ~1-2 h):** a door-gap constraint: in the spin group's top view, carve free space along every depth
   ray; a doorway is where the spin photos see THROUGH the wall line. A doorway photo's camera must sit in such a
   gap, which removes the sliding ambiguity of P-12 and makes the pair bridge unique.
2. Loop pruning by per-kind thresholds → χ² test per edge (each edge judged under its own covariance).
3. The simulation cannot show the protocol's real benefit: a walk-through has no true spins (2-5 headings per room,
   steps 40-92°) and no true doorway pairs (frames 8-195 s apart, 0-0.6 m apart). My own capture is the
   real test; the per-element numbers above (pair yaw ±4°, spin steps ~3-5°) are what transfers.
4. Thresholds of the room-overlap check were set on this apartment (physically motivated, not cross-validated).
5. Door-matching by opening WIDTH needs openings from per-room plan extraction (interface change above).
6. Ultra-wide/HDR/night modes, people in frame: untested.

## v2.7 Verdict on the revised protocol (before the 08:00 capture)

**Keep D-017 (spin + doorway pairs) with five changes.** Evidence for keeping: doorway pairs are the only link that
does not depend on texture or overlap: on the simulation every kept pair joined truly adjacent rooms with a relative
yaw error ≤ 4°, the pairs gave the only error-free adjacency set on with_ceiling (5/5 vs 3 correct + 2 wrong for
corners), and they are what lets the code reject false feature links (4/4). Spin steps were within 10° in 31/38
cases. Against: on the simulation it did not link more rooms into one block (2-4 vs 3), because a doorway photo must
also attach to its own room (7/18 did) — a code limitation (v2.6 item 1), not a protocol one. Evidence for the changes:

1. **Turn clockwise (to your right), always.** Direction inference picked the wrong sense in 2/13 simulated rooms and
   then the whole room was 90-160° wrong (PT-7, P-9). The code now assumes clockwise.
2. **Keep the same tilt for every photo** (floor line just visible at the bottom, phone level left-right). Same-spot
   frames with ≥ 20° pitch difference were linkable in 24% of cases vs 100% at < 10° (`overlap_experiment.json`).
3. **7 photos per turn instead of 6 in rooms with one door** (about 50° per step, "a bit less than two clock hours"),
   6 in rooms with two doors; never more than 8 per folder including doorway photos. With the 1× lens in landscape
   (~69° wide) 6 steps of 60° overlap ~13% (≈ 0-36% linkable), 7 steps of ~51° overlap ~26% (≈ 60-75% linkable).
   Hallways: unchanged (2 from each end + doorway pairs).
4. **Doorway pair: stand ON the threshold, take the two photos within ~5 s, nothing in between, turn fully round.**
   The pairing rule needs them consecutive and quicker than twice your normal shot rhythm (PT-6); an under-rotated
   pair (126° instead of 180°) was snapped 90° wrong before the no-overlap repair.
5. **Keep EXIF intact** (copy files by cable / Google Photos "original", never WhatsApp). Without capture times there
   are no pairs and no spin priors (verified on `phone_like/photos_no_exif`); without the focal length the scale
   shifts ~20% (PT-14).

Withdrawn (D-067): the A4 sheet on the darkest floor patch (D-013/D-017). No capture uses a reference object. The
photo scale is therefore ±4% (learned depth), which passes the 8% photo gate only if the depth prior holds.

---

# photo_tier v3 (4 Oct 2026, ~04:00): room sizes from each room's own spin

> v1 and v2 above are unchanged. v3 adds one new file (`floorplan/photo/layout.py`), a few `PhotoParams` knobs, one
> call at the end of `build_scene_from_photos` (`info["room_layouts"]`), and the photo-only consumer in plan_beta
> (`tiers.layout_plan`, plan_beta.md Part v4 "v4.10"). Switch off with `{"room_layouts": false}`: the plan is then
> the v4 plan. Evidence: `outputs/photo_tier/v3/` (scripts and JSON), end-to-end runs `outputs/runs/photo_v3_*`.

## v3.0 Problem (plain language)

The v4 photo plan had the right rooms joined the right way, but the rooms were the wrong size: with_ceiling/rotate
footprint 17.9 m² against 62.5 m² (LiDAR), median room-area error 29-79%, and 8 of 13 room-area intervals missed the
LiDAR value. Root cause (plan_beta.md T-I5): the free-space extractor builds a room from the floor it sees. A few
photos see the floor as separate wedges, so each room came out as the fragment where the wedges overlap.

## v3.1 How it works, step by step

| Step | What | Why | Without it |
|---|---|---|---|
| 1 | Per folder, the **spin photos** (v2 `spin_groups`, time order) get local yaws from the Manhattan-snapped steps (v2 P-8). All sit at one centre (the spin spot) | a spin is a panorama in 6-7 stills: no feature match is needed to put them together, the same idea as panorama layout methods (LayoutNet / HorizonNet) | rooms depend on the pose graph, which placed only 27 of 41 photos on with_ceiling/rotate |
| 2 | Each photo's MoGe-2 metric depth × the fused scale → levelled points around the centre | metric depth is what the plan needs; the scale is the same one the scene uses | – |
| 3 | **Wall points**: MoGe-2 normal within ~17° of horizontal, 1.0-2.0 m above the floor, and only the **farthest such point in each image column** | above 1 m most furniture (beds, tables, sofas, counters) is gone; in a column, the wall is the last vertical surface | with all band points, median dimension error 16.1% vs 11.3% (proxy, below) |
| 4 | Rotate onto the room's Manhattan axes. Per side (+x, −x, +z, −z): histogram of wall offsets (5 cm bins). The score of a bin is the wall **length** it covers (distinct 10 cm cells along the wall). Take the nearest peak that is ≥ 60% of the longest peak, then the median of its points | a wall seen through a door is a door-wide patch; a real wall covers most of the side. Preferring the nearer strong peak removes the next room's wall seen through a doorway | on the sim spins, R5's −x wall at 4.97 m (the next room's wall through a door; LiDAR ~1.25 m) |
| 5 | Per-side uncertainty, Monte Carlo (400 draws): each photo that saw the side moves by its sway (σ 0.3 m) and scales by its own depth error (σ 5%); surface noise 5 cm; plus `layout_side_sigma_rel` = 50% of the distance (calibration, v3.3) | a wall seen by one photo is as uncertain as that photo's position; the calibration term covers what the model does not see (furniture, doors, partial walls) | intervals 1-2 cm wide on walls that are 30% wrong (confident garbage) |
| 6 | A side no photo saw is **inferred**: at least as far as the neighbouring walls were seen to reach (a corner cannot be inside the room), else the opposite wall's distance (the protocol asks to stand near the middle), σ = max(35%, 0.4 m) | an unseen wall is not a measurement; it must be wide and labelled | – |
| 7 | Room height = median ceiling points − floor points (when seen) | – | plan_beta keeps its own ceiling when it has one (it is better: v3.5) |
| 8 | **Anchor**: the placed spin photos give the rotation (common yaw − local yaw) and the centre (their mean position); then into the aligned frame (`T_align`) and the 4 sides are snapped to the plan axes | the scene and the layout must share one frame | – |
| 9 | `info["room_layouts"][folder]` = sides (offset, sigma, status, photos, rival ratio), anchor, height | plan_beta builds the rooms (plan_beta.md v4.10) | – |

## v3.2 Decisions (options, evidence)

### P-15 Fit each room from its own spin, not from the fused free space

- **Options.** (a) v4: free-space segmentation per folder seed. (b) v4 + per-folder convex-hull fill (T-4, rejected
  in v4: an overlap). (c) v3: per-room Manhattan rectangle from the spin's wall points. (d) A learned layout network
  (HorizonNet etc.) on a stitched panorama: needs ≥ 50% overlap between neighbours to stitch; the protocol gives ~13-26%.
- **Evidence** (cached scenes, LiDAR benchmark `drift_on` = the `run_capture` default; `outputs/photo_tier/v3/cached_*.json`):

  | Scene | | rooms / folders | adjacency correct / wrong (LiDAR pairs) | overlap m² | footprint vs 62.5 / 61.9 m² (in interval?) | median area error | dims within 8% | dims in interval |
  |---|---|---|---|---|---|---|---|---|
  | with_ceiling / rotate | v4 | 5/7 | 2 / 0 (10) | 0 | 17.9 (−71%, no) | 78.8% | 0/8 | 5/8 |
  | | **v3** | **7/7** | **5 / 0** | 0 | **45.8 (−27%, yes)** | 56.7% | 0/12 | 10/12 |
  | floor_only / rotate | v4 | 5/6 | 2 / 0 (11) | 0 | 37.5 (−39%, no) | 27.5% | 0/8 | 3/8 |
  | | **v3** | **6/6** | 2 / 0 | 0 | 36.8 (−41%, no) | 58.5% | 2/10 | 7/10 |
  | with_ceiling / corners | v4 = v3 | 2/7 | 0 / 0 | 0 | 17.3 (−72%) | 32.3% | 0/4 | 2/4 |
  | floor_only / corners | v4 = v3 | 1/7 | 0 / 0 | 0 | 0.9 (−99%) | 75.7% | 0/2 | 0/2 |

  Corners folders have no spin, so no layout is made and v3 = v4 (that is the intended contrast). On rotate, v3 keeps
  every folder as a room, joins more of them correctly (5/0 vs 2/0), keeps no overlap and moves the with_ceiling
  footprint from −71% to −27%. **Per-room sizes are still not within 8%.** floor_only's median area error got worse
  (27.5% → 58.5%), while its dimension coverage improved (3/8 → 7/10).
- **Decision.** (c) is on by default for spin folders. The reason is the gates it fixes: one plan with all rooms,
  correct adjacency, no overlaps, and a footprint whose interval contains the truth. The size gate is not met on the
  simulation, and the doc says so (v3.5).

### P-16 One point per image column, nearest strong peak, wall length as the score

Evidence: the full-turn proxy (v3.4, true poses, 12 dimensions on with_ceiling):

| Variant | median abs dim error | within 8% | median abs area error |
|---|---|---|---|
| **chosen**: farthest per column, prefer-near 0.6, 5 cm bins | 11.3% | 5/12 | 25.8% |
| all band points (no per-column filter) | 16.1% | 1/12 | 21.6% |
| no prefer-near (longest peak wins) | 8.9% | 6/12 | 27.7% |
| prefer-near 0.3 | 11.0% | 5/12 | 27.0% |
| band 0.5-2.2 m | 11.7% | 3/12 | 25.5% |
| 10 cm bins, 20 cm inliers | 10.4% | 5/12 | 17.1% |

These differences are small on 12 dimensions. The per-column filter is the only clear gain. Prefer-near is kept for
its effect on the simulated spins: R5's −x wall went from 4.97 m to 3.71 m (the next room's wall seen through a door; LiDAR ~1.25 m). The neighbour clip in plan_beta removes the rest. No-prefer-near is slightly better
on the proxy. That is a real trade-off, not a settled choice.

### P-17 A spin that is not from one spot is not a layout

A layout is used only when (i) the folder has capture times (otherwise no spin order: `phone_like` photos have none,
and v4 is used), (ii) its spin photos give ≥ 200 wall points, and (iii) the placed spin photos stand ≤ 1.0 m from
their mean (corner shots placed by features are metres apart). If no spin photo is placed, the sizes still come from
the spin, but the position is **inferred** (the centroid of the folder's free-space room, else of its placed photos).
This is recorded in `meta.tier_v4.layouts.per_folder[f].centre`.

## v3.3 Issues log (v3)

| # | Issue | Root cause | Status |
|---|---|---|---|
| PT-15 | Corridor / room walls fitted to the NEXT room's wall seen through a door (sim R5 +139%, R7 hallway +115%) | in the 1-2 m band a doorway shows the far room; the longest-peak rule picked it | prefer-near rule (P-16) and plan_beta's neighbour clip (a room cannot contain another room's spin centre): R5 +166% → +56% area, floor_only R3 +49% → +9%. The hallway R7 is still +257% |
| PT-16 | Even with TRUE poses, rooms are 20-40% small on the simulated spins | the simulated "spins" are walk-through frames: 3-5 headings spanning 85-200°, so whole walls are never seen; MoGe-2 depth is ~10% short on these frames (v2: depth ratio median 0.905); furniture in front of walls | measured, not fixable in the extractor. Oracle (true yaw + position) on with_ceiling sim spins: per-side median abs error 19.7%, 4/24 sides within 8% |
| PT-17 | `run_capture` on corner folders (EXIF times present) treated corner shots as a spin | `spin_prior` is on by default | spread guard (P-17). The proper corner run sets `spin_prior: false`, and then no layout is made |
| PT-18 | A layout rectangle overlapped a kept free-space room (1.06 m², with_ceiling/corners end to end) | the push-apart only knew layout rectangles | kept rooms are fixed obstacles too; 0 m² after the fix |
| PT-19 | Footprint interval collapsed to the scale term when no layout room was used | the Monte Carlo summed fixed values | no layout → v4 plan returned unchanged; kept rooms add their own interval |
| PT-20 | floor_only R4: 13 wall points | its spin photos see almost no wall in the 1-2 m band (open) | room keeps its v4 free-space shape |
| PT-21 | Layout heights 1.6-2.4 m on the sim (LiDAR ~2.5 m) | few ceiling points in photos aimed at the floor line | plan_beta's ceiling is used when it has one |

## v3.4 The question for 08:00: what would a REAL spin give? (full-turn proxy)

The simulated spins cannot answer this (PT-16). Proxy: for every LiDAR room, the walk-through video frames inside
the room whose headings cover a full turn (6 targets, 60° apart, ±30°), nearest to the room centre. MoGe-2 is run
with the pipeline settings, with true gravity and true heading. Then the v3 fit runs (a) at the true camera
positions and (b) with all cameras at their mean (the spin assumption). These frames are 0.3-2.2 m apart, not a
real spin. (`outputs/photo_tier/v3/fullspin.py`, `fullspin6_*.json`.)

| Room | frames | camera spread (m) | LiDAR L × W | true positions: L × W (error) | common centre (error) |
|---|---|---|---|---|---|
| with_ceiling R2 | 6 | 1.44 | 4.37 × 3.03 | 4.03 × 3.99 (−7.9%, +31.7%) | (−2.6%, −6.6%) |
| with_ceiling R3 (L-shaped) | 6 | 2.23 | 4.85 × 3.01 | 3.03 × 2.68 (−37.6%, −10.8%) | (−55.1%, −33.8%) |
| with_ceiling R4 | 6 | 1.08 | 3.17 × 3.02 | 3.03 × 2.85 (**−4.5%, −5.7%**) | (−19.7%, −24.1%) |
| with_ceiling R5 | 5 | 0.40 | 3.10 × 2.52 | 2.96 × 2.44 (**−4.4%, −3.1%**) | (−20.6%, −22.8%) |
| with_ceiling R6 | 5 | 0.34 | 2.22 × 1.87 | 1.81 × 1.38 (−18.4%, −25.9%) | (−0.8%, −12.8%) |
| with_ceiling R7 | 6 | 0.86 | 2.08 × 1.59 | 4.09 × 1.77 (+96.5%, +11.7%) | (+69.6%, +9.3%) |
| floor_only R1 | 5 | 1.54 | 3.97 × 3.89 | 3.05 × 2.02 (−23.2%, −48.1%) | (−41.3%, −73.0%) |
| floor_only R3 (L-shaped) | 6 | 1.33 | 4.80 × 3.01 | 2.95 × 2.64 (−38.5%, −12.2%) | (−52.5%, −42.6%) |

**True positions: median abs dimension error 15.3%, 5 of 16 within 8%. Common centre: 23.5%, 3 of 16.** The two
rectangles that are right (R4, R5) are rooms where every wall is visible above furniture. The failures have causes
we can name:
- R3 and floor_only R3 are L-shaped: one rectangle cannot fit them.
- R7 sees through a door.
- R6 and R1 have walls hidden behind tall furniture or MoGe-skewed walls.
- The MoGe-2 scale is short on these frames.

The common-centre column is pessimistic: these frames stand 0.3-2.2 m apart, while a real spin moves ~0.1-0.3 m.

## v3.5 Results end to end (`run_capture.py <folders> --tier photo`, this session)

Outputs: `outputs/runs/photo_v3_<input>/` (plan.json, run_report.json, scene/, stdout.log, time.log), scored
by `outputs/photo_tier/v3/score_e2e.py` against `outputs/benchmark/lidar/<capture>/drift_on/plan.json` (the
`run_capture` default; there is no `default/` folder). The scale term (`widen_plan`) and the clip at 0 are
included. `schema_problems: []` in every run.

| Input (`--tier photo`) | rooms / folders | adjacency correct / wrong (LiDAR pairs) | overlaps m² | footprint m² [95%] vs LiDAR | median room-area error | areas within 8% | areas in interval | dims within 8% | dims in interval | layouts used | wall time / RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| with_ceiling / rotate | **7/7** | **5 / 0** (10) | **0** | 45.7 [22.9, 68.6] vs 62.5 (−27%, **covered**) | 56.7% | 0/7 | 5/7 | 0/12 | 10/12 | 7 (3 with inferred position) | 1:24 / 1.9 GB |
| floor_only / rotate | **6/6** | 2 / 0 (11) | **0** | 36.5 [16.5, 56.5] vs 61.9 (−41%, not covered) | 58.5% | 0/6 | 3/6 | 2/10 | 7/10 | 5 (R4: 13 wall points) | 0:59 / 1.7 GB |
| with_ceiling / corners (default params) | 5/7 | 1 / 0 (10) | 0 | 43.4 [20.1, 66.7] vs 62.5 (−31%, covered) | 27.5% | 1/5 | 5/5 | 1/10 | 9/10 | 4 | 0:47 / 1.7 GB |
| floor_only / corners (default params) | 4/7 | 2 / 0 (11) | 0 | 47.2 [12.1, 82.3] vs 61.9 (−24%, covered) | 34.1% | 1/4 | 4/4 | 1/6 | 6/6 | 4 | 0:21 / 1.6 GB |
| phone_like / oneplus_exif (9 corner folders, no capture times) | 4/9 | 1 link | 0 | 18.5 [7.0, 29.9] vs 62.5 | – | – | – | – | – | **0**: no times → no spin order → v4 plan | 0:46 |
| phone_like / heic | 4/9 | 1 link | 0 | 18.2 [6.1, 30.4] vs 62.5 | – | – | – | – | – | 0 (same) | 0:58 |
| phone_like / no_exif | 3/9 | 0 links | 0 | 12.7 [0.0, 25.4] vs 62.5 | – | – | – | – | – | 0 (no times) | 0:54 (first two attempts: cuDNN error on the shared GPU, retried) |

The v4 baseline on the same end-to-end scenes, re-extracted (`score.py --scene ... --v4`,
`outputs/photo_tier/v3/e2e_scene_*_v4.json`):
- with_ceiling/rotate: 5/7 rooms, 2/0 adjacency, footprint 15.5 (−75%, not covered), median area error 78.6%,
  dimensions in interval 4/8, areas in interval 2/5.
- floor_only/rotate: 5/6 rooms, 2/0 adjacency, footprint 36.3 (−41%, not covered), median area error **13.5%**
  (better than v3), dimensions in interval 3/8, areas in interval 3/5.

**Gates on simulated photos (honest):**
- One stitched plan: yes.
- Adjacency: no wrong links. Not all LiDAR links are found: 5 of 10 and 2 of 11.
- No overlaps: yes.
- Footprint within 8%: **no** (−24% to −41%).
- Wall lengths within 8%: **no** (2 of 22 dimensions on rotate).
- Intervals: dimensions 17/22 and areas 8/13 contain LiDAR on rotate. That is better than v4 (7/16 dimensions and
  5/10 areas on the same scenes) but still below 95%. The remaining misses are gross errors (hallway through doors,
  partly seen rooms), not noise.

Corner folders through `run_capture` use the default params (spin prior on, because EXIF times exist), so corner
shots were treated as spins and 4 layouts were made. The proper corner run sets `spin_prior: false` (v2), and then
no layout is made (cached scenes, v3.2 table: v3 = v4). The corner rows are contrast, not a recommendation.


## v3.6 Limitations and next steps

1. **The 8% size gate is not met on any simulated photo set**, nor on 11 of 16 dimensions of the full-turn proxy.
   What v3 delivers reliably is the topology: every folder becomes a room, adjacency comes from links, there are no
   overlaps, and the footprint interval contains LiDAR on with_ceiling. The intervals are wide (±50% per side) because
   the measured errors are that large. The real test is the 08:00 capture against the tape.
2. **L-shaped rooms.** A rectangle is the only model so far. Next: fit two rectangles when a side has two strong
   peaks ≥ 1 m apart that each span part of the side (the rival ratio is already stored per side).
3. **Scale.** The proxy rooms are 5-10% small before any other error. The A4 sheet (D-013) was meant to fix that, but
   no capture uses one since D-067. Without it the photo tier cannot meet 8% here even on perfect walls. Sheet-free
   options: scale fusion weighted by view type (`docs/LAYOUT_METHODS_RESEARCH.md` §1.3) and the door-height cue
   (built, kept off: D-069).
4. **Photos without capture times** (WhatsApp, `phone_like`): no spin order → no layout. Next: order by file name
   (phones name files by time).
5. Doorway photos are not used in the layout. With true poses they helped some rooms (R2 length −46% → +1%, oracle run with the first fitter) and hurt
   others (R1 −29% → −46%). They need the v2.6 door-gap registration first.
6. Hallways: the 1-2 m band of a hallway is mostly doors, so the layout over-reaches (R7). Treat a folder with ≥ 2
   doorway pairs and a narrow free space as a corridor (next).

# photo_tier v3-poly (4 Oct 2026, ~18:00): rooms that are not rectangles

> One new file (`floorplan/photo/polygon.py`), one call at the end of each room's fit in `layout.room_layouts`
> (`lay["polygon_local"]`, no existing key changes), one switch `PhotoParams.layout_polygon` (default on), and a marked
> block in `plan/beta/tiers.layout_plan` (`use_polygon`). Switch off with `--photo-param layout_polygon=false`.
> Evidence: agentC scratchpad (`off/`, `dumpplan/`, `e2e/`), scored with `scripts/eval_own_capture.py` on exact sim GT.

## Problem
Open-plan living rooms are not rectangles: k38 living room has 10 vertices (an alcove with the bathroom and bedroom
doors, fill 0.78), k65 living room 8 (a door alcove, fill 0.81, and a far end seen by no spin photo). One rectangle per
room gave k65 living -31% area, k38 -10% (-46% after D-060, see below).

## How it works (room's layout frame: spin centre = origin, the rectangle's Manhattan axes)
| Step | What | Why |
|---|---|---|
| 1 | Photos: the spin photos plus every other placed photo of the folder (doorway-pair photos, extra photos) at its pose-graph position relative to the placed spin photos (as `_refit_with_door_shots`) | door shots see the far parts of big rooms |
| 2 | Wall points: the rectangle fit's rule (horizontal normal, 1-2 m band, farthest per column, D-060 semantic mask). Visibility: per photo and azimuth, the farthest band point; nearer = seen free, behind = blocked | free space says where the room is, walls where it ends |
| 3 | Per side, the offset peaks of its wall points (wall length in 10 cm cells) = candidate wall lines | a step is a second wall line on a side |
| 4 | A measured side whose wall was seen only OUTSIDE the room's extent along it is replaced by the nearest farther wall of that side that overlaps the room | D-060 k38: the +z "wall" at 2.49 m was the kitchen's wall seen through its 1.6 m door, beside the room |
| 5 | An INFERRED side reaches at least the doorway photo standing on it (its room-side wall face 0.10 m in front of the camera, the D-053 convention) | a doorway photo is taken on the threshold; k65 far wall 3.41 -> 4.38 m (true 5.15: the door shot is placed ~0.6 m short) |
| 6 | Alcove: doorway photo(s) of this room >= 0.4 m beyond a measured side, corroborated (a second doorway photo, or a wall seen just beyond it); depth to the photo or that wall, span +-0.45 m stretched to seen perpendicular walls within 1.5 m; rejected if the spin photos saw it mostly blocked (> 40%) | one door shot alone can be its placement error (0.03-0.54 m measured on k22/k38) |
| 7 | Notch: a rectangle corner cell >= 60% blocked, <= 5% free, its walls toward the room >= 40% seen, no camera in it | strict: never fired on k22/k38/k65 (relaxed it only produced a false notch on k38 balcony) |
| 8 | At most 2 changes; polygon = boundary of the inside cells of the line grid; each edge keeps its line (status, sigma) | unseen parts keep the rectangle |
| 9 | plan_beta: lines into the plan frame (`side_to_plan`), Monte Carlo per line (same calibration as the sides: hypot(sigma, 0.5 x distance)), one wall per edge, push-apart on the polygon's parts; a side cut back by a neighbour -> plain rectangle; any polygon failure -> rectangle | walls with intervals, no overlap, room ids unchanged |

## Results (exact sim GT, `eval_own_capture.py`; before = same scene, polygon off)
| Scene | living area | footprint | wall median | walls matched | interval coverage | living vertices (GT) |
|---|---|---|---|---|---|---|
| k65 photo_d057 | -31.0% -> **-20.1%** | -14.9% -> **-9.6%** | 5.6% -> 5.6% | 19 -> 19 /26 | 1.00 -> 1.00 | 4 -> 4 (8) |
| k65 current code (D-060), run's own inputs | -29.2% -> **-18.3%** | -14.4% -> **-9.0%** | 5.6% -> 5.6% | 19 -> 19 /26 | 1.00 -> 1.00 | 4 -> 4 (8) |
| k38 photo_d057 | -10.0% -> **-3.3%** | -8.0% -> **-4.6%** | 8.7% -> 9.0% | 18 -> 21 /26 | 1.00 -> 1.00 | 4 -> 8 (10) |
| k38 current code (D-060), run's own inputs | -45.8% -> **+1.6%** | -26.2% -> **-1.7%** | 10.9% -> **9.0%** | 15 -> 21 /26 | 1.00 -> 1.00 | 4 -> 8 (10) |
| k38 current code, fresh end-to-end run (same scene re-planned with the polygon off) | -45.8% -> **-4.1%** | -12.6% -> +9.0% (this run's bathroom is +185%, no longer offset by the small living room) | 8.4% -> **6.0%** | 17 -> 20 /26 | 1.00 -> 1.00 | 4 -> 8 (10) |
| k22 (control) photo_v21b_d057 | living unchanged; bedroom -28.1% -> -22.5% (6 vertices, GT 4) | -49.3% -> -48.1% | 29.4% -> **34.3%** | 13 -> 14 /20 | 0.65 -> 0.63 | 4 -> 4 (4) |
| k22 (control) photo_d060b (D-060) | no change: the living room's 6-vertex polygon is dropped (side clipped by a neighbour) | -45.2% | 25.9% | 14 /20 | 0.67 | 4 (4) |

Living-room IoU with GT (scorer's geometric match): k38 0.87 -> 0.92 (d057), 0.54 -> 0.93 (current), 0.54 -> 0.91
(fresh run); k65 0.69 -> 0.80 (d057), 0.71 -> 0.82 (current). k38 "run's own inputs": two of three fresh polygon-on runs of k38 crashed in `extract._undo_overlapping_refinements` (GEOS
TopologyException in the free-space rooms, before the layout step; the same scene crashes with the polygon off too, the
scenes differ run to run), so this row is the polygon-off run's scene re-planned with polygons fitted from that run's
own `room_layouts` inputs (dumped). On k65 that procedure reproduces the fresh run exactly.

Known failures: (i) k22 living: a 6-vertex alcove is fitted (one doorway photo plus a wall seen beyond it; GT is a
rectangle) and is only dropped because a neighbour clips that room; on the photo_d057 scenes the same rule gave k22
bedroom 6 vertices (area -28% -> -22%, wall median 29% -> 34%); (ii) k65's door alcove (1 m2) is not found (its door
shot is only 0.33 m beyond the side); (iii) the polygon walls' intervals are as wide as the sides' (50% calibration
term per line).
