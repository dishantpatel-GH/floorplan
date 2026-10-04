# Module: ARKitScenes validation (absolute accuracy of the LiDAR tier against a laser scanner)

Files:

- `floorplan/io/arkitscenes.py`: an adapter that makes an ARKitScenes capture look like a Stray Scanner capture.
- `scripts/validate_arkitscenes.py`: runs our stage-1 pipeline on it, registers the Faro laser scan, and measures the errors.
- `docs/modules/arkitscenes_validation.md`: this document.

Evidence is in `outputs/arkitscenes_validation/`:

- `results.json`: every number in this document.
- `run_log.txt`
- `plan_overlay.png`
- `raw_error_map.png`
- `error_stats.png`
- `T_laser_to_aligned.txt`

---

## 1. Purpose and where it sits in the pipeline

**The problem.** The sample captures we were given contain no ground truth: no tape and no laser measurements. On them we can
measure *repeatability*, because two scans of the same apartment should agree, but not *accuracy*. Both scans
could be wrong by the same 2 cm and still agree.

**The answer.** [ARKitScenes](https://github.com/apple/ARKitScenes) is a public Apple dataset. Each room in it has
two recordings:

- an iPad Pro capture: LiDAR depth, confidence and ARKit poses, the same kind of data Stray Scanner records on an iPhone Pro;
- a **Faro Focus S70 laser scan** of the same room, survey grade with millimetre accuracy.

We take scene 47895909, a small bathroom: dark, tiled, with a door, a sink, a washing machine and a shower area. We run our pipeline on it **unchanged** and compare
what it measures with what the laser measures.

**Where it sits.**

```
ARKitScenes raw scene ──> ArkitScenesCapture (adapter: same interface as StrayCapture)
                              │
                              ▼
            stage 1 = the same functions as floorplan/pipeline/scene.py build_scene:
            select_keyframes -> fuse_tsdf -> collect_raw_points -> align_scene
                              │
Faro laser PLY ──> laser cloud (y-up, normals) ──> registration (align_scene on the laser too,
                              │                     4 yaw hypotheses, FFT shift, ICP)
                              ▼
            metrics: cloud-to-cloud error, plane-to-plane dimensions, error breakdowns, ablations
```

It is **not** part of the product pipeline. It is the **evidence** for the LiDAR tier's absolute accuracy, and it
feeds two other modules:

- the **confidence-interval model**: it shows how large the systematic error is, and that a purely statistical
  interval is far too narrow;
- the **fix loop (Part 4)**: it found a measurable failure, which section 4 diagnoses.

---

## 2. How it works, step by step

### 2.1 Adapter: `floorplan/io/arkitscenes.py`

The adapter's `load_arkitscenes(scene_dir, pose_mode="nearest")` returns an `ArkitScenesCapture` with exactly the
fields and methods that the pipeline uses from `StrayCapture`:

- fields: `n`, `capture_id`, `timestamps`, `T_wc`, `K_rgb`, `rgb_size`, `depth_files`, `conf_files`;
- methods: `K_depth(i)`, `depth(i, min_conf, min_m, max_m)`, `confidence(i)`, `iter_rgb(indices)`.

| Step | What it does | Why | What goes wrong without it |
|---|---|---|---|
| 1. Read `lowres_wide.traj` | Reads each row as `t, rx, ry, rz, tx, ty, tz` (axis-angle in radians, translation in metres). Each row is **world→camera**, so it is inverted to get camera→world. | The pipeline expects `T_wc` = camera→world. This is the same contract as Stray Scanner. | Using the row directly puts points up to 46 cm off. The data README measured this against the ARKit mesh. |
| 2. Convert the world from z-up to y-up | `T_wc ← C · T_wc` with `C`: (x, y, z) → (x, z, −y), a −90° rotation about x. | ARKitScenes stores its world with **+z up**. Our `align_scene` looks for the floor along **+y**, which is ARKit's native convention and what Stray Scanner gives. | The floor search would find a wall as the "floor", and the plan would be a side view. |
| 3. Match 10 Hz poses to 60 Hz depth | Default `"nearest"`: keep the depth frame closest to each pose if it is within 5 ms. The measured gap is at most 0.50 ms, which gives 337 frames. Option `"interpolate"`: every depth frame inside the trajectory, with slerp on rotation and lerp on translation (2,015 frames). | Each depth frame needs a pose. | A wrong pose time smears the cloud. The ablation in §3, D2 compares the two options. |
| 4. Per-frame intrinsics | Reads `.pincam` (`w h fx fy cx cy`) for every frame. fx varies between 209.6 and 222 px. | Focus changes the focal length. One fixed K would scale the points. | About 3% lateral scale error at the extremes. |
| 5. Depth and confidence | `depth()` applies exactly the same filtering as Stray: uint16 mm, confidence ≥ `min_conf`, 0.2–4 m. | Same input contract, so the same downstream behaviour. | — |
| 6. Sanity checks | Frame sizes must be 256×192 (the pipeline imports `DEPTH_W/H` from the Stray loader). Missing confidence or RGB files raise an error. | Fail loudly rather than fuse garbage. | — |

The camera axes are already OpenCV (x right, y down, z forward), so no camera-axis flip is needed. This was
verified with the 0.8–2.0 cm agreement against the ARKit mesh in the data README.

### 2.2 Stage 1 on the adapter: `build_stage1`

`floorplan/pipeline/scene.py::build_scene` calls `load_stray(folder)` internally, so it cannot take an adapter object.
The script therefore calls **the same four shared functions in the same order** with the default `Config`:

1. `select_keyframes`: 5 cm, 5° or 0.5 s → 217 keyframes out of 337 posed frames.
2. `fuse_tsdf`: VoxelBlockGrid with 2 cm voxels, high confidence only → 72,842 surface points. Colour is off because it does not change the geometry.
3. `collect_raw_points`: raw high-confidence LiDAR points de-duplicated on a 5 mm grid → 1,637,679 points.
4. `align_scene`: floor levelling plus Manhattan yaw. The measured floor tilt was 0.19° and the yaw 27.4°.

Two additions are needed only for the evaluation:

- **PCA normals for the raw points.** They come from a 5 cm radius with up to 30 neighbours, and are flipped to face the
  camera that saw each point (`raw_ray`). The plane fitter needs normals, and the raw points have none.
- **Sorting the TSDF points.** VoxelBlockGrid's extraction order depends on threads, so the points are sorted to make the run deterministic.

The requested change in §6 removes this duplication.

### 2.3 Laser preparation: `load_laser`

1. **Read the PLY.** It is a 1.8 GB binary file holding 41,033,222 points. The script memory-maps it and reads every 10th point
   (4.1 M). This is fast, and on nearby walls the spacing is still below 5 mm.
2. **Find the scanner position** from `<scan>_pose.txt` (row 3 = translation). The PLY is *already* in that registered
   frame (data README), so the pose is used only to know where the scanner stood. Applying it again would move the scan by 12 m.
3. **Convert z-up → y-up** with the same matrix `C` as the adapter.
4. **Crop** to points within 8 m of the scanner. The scanner sees through doors into other rooms.
5. **Build two densities.** The 5 mm version (1,044,530 points, median spacing 3.2 mm) is the reference for the error metrics.
   The 2 cm version (103,285 points) is used for registration.
6. **Compute normals.** PCA normals are oriented toward the scanner. This is the same rule as TSDF normals: they face the side the sensor observed, which is into the room.

### 2.4 Registration: `register_laser`

The dataset ships **no** transform from the laser frame to the ARKit frame. The two frames differ by a yaw, a
translation and a small tilt. The registration has four steps:

1. **Run `align_scene` on the laser cloud.** This is the *pipeline's own* gravity and Manhattan step, and the laser floor tilt it removes is 0.16°.
   Both clouds now have their walls on the x and z axes, so what remains is a yaw that is a multiple of 90° plus a translation.
2. **Try the four 90° yaws.** For each one, take a wall slice (1.0–1.5 m above the floor) from each cloud, rasterise it
   at 2 cm, and find the horizontal shift by **FFT cross-correlation**. Keep the yaw with the highest normalised score.
   The scores were 0.705 for 0°, against 0.278, 0.310 and 0.266 for the others, so the choice is unambiguous. The vertical shift is the
   difference of the two floor levels.
3. **Refine with point-to-plane ICP.** The source is our TSDF at 2 cm. The target is the laser cropped to our bounding box plus
   30 cm. The correspondence distance shrinks over 10 → 5 → 2 cm.
   - Result: inlier RMSE 1.21 cm, fitness 0.53.
   - The ICP correction to the coarse alignment was only 0.72° / 0.73 cm, so the coarse step already did most of the work.
4. **Cross-check against an independent registration.** The data-download smoke test produced its own laser→ARKit transform with a
   different pipeline (no `align_scene`). Ours differs from it by **0.16° and 3.6 cm** (`--reference-transform`).

Why the fitness is only 0.53: a single, stationary scanner cannot see behind the toilet, the washing machine or the
cabinets, and the floor under the tripod is a hole. Those parts of our surface have no laser partner within 2 cm.

### 2.5 Surface accuracy: `surface_metrics`, `completeness`

For every point of ours, the script finds the nearest laser point and records three things:

- the point-to-point distance;
- the **signed point-to-plane distance** along the laser normal. A positive value means our point lies *in front of* the true surface, on the camera side;
- whether the point is **in overlap**, meaning a laser point lies within 10 cm. Points without one are almost all places the laser did not see
  (occlusion, the tripod hole), so they would measure laser coverage, not our error.

Statistics are reported over the overlap and over all points. **Completeness** is the reverse check: the fraction of laser points
inside our bounding box that have one of our points within 5 cm.

### 2.6 Dimensions: the same robust plane fit on both clouds (`measure_against_laser`)

The floor plan reports distances between planes, not point errors. That makes this the number that matters for the gates.

1. **Orientation families.** Points are split into six families by normal: ±x walls, ±z walls, floor (+y) and ceiling (−y). A
   point belongs to a family if its normal is within 25° of the family direction. This works because `align_scene` put the walls on the axes.
2. **Sequential surface extraction.** This is like sequential RANSAC, but 1-D. For each family:
   1. Take the strongest 1 cm histogram peak along the family axis.
   2. Fit a plane with **trimmed least squares**: refit while keeping only points within 3, 2, 1.5 and finally 1 cm of the current plane.
   3. Remove the points of that plane.
   4. Repeat.

   A surface is kept if it covers at least 0.15 m² of 5 cm cells, where a cell counts only if it holds at least 20% of the points a
   fully seen cell would have at that cloud's resolution.
3. **Match each surface to the laser.** Our surface only identifies *which* laser surface to compare with: same family,
   within 10 cm, same cells. The laser plane is then found from **laser points alone**, using its own histogram peak and the same trimmed
   fit. Both planes are then **refitted on the cells both sensors see**, so a wall the laser saw only partly is
   compared on the same stretch.
4. **Facing pairs.** Every pair consists of a +axis surface on the low side and a −axis surface on the high side. The two must overlap in plan and
   be at least 30 cm apart. For each pair, the distance between the two planes is measured on a line through the middle of the two surfaces,
   for us and for the laser. This is what a tape measure between those two surfaces would read.
   Pair kinds:
   - `full_height`: both surfaces reach above 2.0 m (walls, tall cabinets);
   - `partial_height`: furniture fronts;
   - `height`: floor to ceiling.
5. **Room box.** This is the headline number. The room centre is the median of the camera path (the operator walks inside the room). On
   each side of the centre, take the **largest** matched surface of the right orientation that spans the centre. Measure width x, depth z and height.
6. **Error versus distance.** Fit `error = offset + slope · distance` over all pairs. A scale error (for example, odometry scale drift) appears as
   slope. A per-surface depth bias appears as offset, because every interior distance loses twice the bias whatever its length.

**Why dimensions are the registration-independent check.**

- A wall-to-wall distance does not change if the laser is shifted.
- A residual rotation θ changes it only by a factor of cos θ, which is 1 − 1e-5 for 0.25°.

ICP improves the cloud-to-cloud numbers by construction, because it minimises exactly that error. The dimension numbers do not benefit from it.

### 2.7 Error breakdowns and ablations

The breakdowns cover raw-point error by range, by incidence angle and by distance to the nearest *other* room-box plane (the corner).
The script also counts raw points more than 10 cm *behind* a room-box surface, as the ghost-geometry check for mirrors and glass.

There are four ablations:

- pose interpolation;
- confidence threshold 2, 1 and 0;
- an edge margin (dropping the rim of each surface);
- corner exclusion (fitting room-box planes only on points ≥ 0.2, 0.3 or 0.4 m from the other box planes).

---

## 3. Decisions

| # | Decision | Options considered | Choice and why | Evidence |
|---|---|---|---|---|
| D1 | How ARKitScenes enters the pipeline | (a) a separate ARKitScenes pipeline, as in the smoke test; (b) an adapter with the StrayCapture interface | **(b).** The accuracy we measure must be the accuracy of *our* code (keyframes, TSDF, raw points, alignment), not of a lookalike. | `build_stage1` calls the same four shared functions with the default `Config` |
| D2 | 10 Hz poses vs 60 Hz depth | (a) nearest depth frame to each pose; (b) interpolate a pose for every depth frame | **(a) "nearest".** Every pose used is a real ARKit estimate. Interpolating over 100 ms of hand motion adds error, and keyframing would discard most of the extra frames anyway. | Raw point-to-plane median **1.12 cm** (nearest) vs 1.21 cm (interpolate). p90 3.59 vs 3.84 cm. Full-height pair median 1.54 vs 2.19 cm. Room box +0.25/−1.52/−1.92 vs +0.96/−1.55/−1.98 cm |
| D3 | World up-axis conversion | (a) change `align_scene` to accept z-up; (b) convert in the loader | **(b).** One convention inside the pipeline (+y up, as ARKit and Stray use). The loader is the only place that knows the dataset's quirks. | After conversion `align_scene` finds the floor with 0.19° tilt, matching the data README's 0.25° |
| D4 | Registration | (a) reuse the smoke test's transform file; (b) feature-based global registration (FPFH + RANSAC); (c) the pipeline's own `align_scene` on both clouds, then 4 yaws + FFT + ICP | **(c).** It uses only our code plus standard ICP, and it runs cold on any ARKitScenes room. (a) depends on an external artifact. (b) is less robust in repetitive, tiled rooms and is not deterministic. | Yaw score 0.705 vs ≤ 0.31 for the others. ICP RMSE 1.21 cm. Agrees with the independent transform to 0.16° / 3.6 cm |
| D5 | Error metric | point-to-point NN; point-to-plane; mesh distance | **Report both, with point-to-plane as primary.** Point-to-point is inflated by the laser's sampling spacing (3.2 mm median). Point-to-plane measures distance to the *surface* and gives a **sign** (in front or behind), which was the key to diagnosing the bias. | p2p median 1.42 cm vs p2pl 1.12 cm on the same points |
| D6 | Which points count | all points; points with a laser point within 10 cm | **Report both, and headline the overlap.** The laser is a single scan with occlusions and a tripod hole, so non-overlap points mostly measure laser coverage. Overlap is 86% of our raw points. | All-points p95 = 27.8 cm (laser holes) vs overlap p95 = 4.66 cm. See the grey areas in `raw_error_map.png` |
| D7 | How to compare dimensions | (a) detect walls independently in each cloud (first version); (b) detect on ours, then fit the laser independently on the same surface and the same stretch | **(b).** (a) failed: see Issue 1. A single scanner sees some walls only in fragments, so "strongest wall" picked different walls in the two clouds. (b) separates *measurement accuracy* (what we test here) from *wall detection* (tested by the plan-extractor modules). The laser position still comes from laser points only. | (a): −92 cm / +39 cm "errors" from wrong correspondences. (b): +0.25 / −1.52 / −1.92 cm |
| D8 | Plane-fitting method | RANSAC; plain least squares; median offset; trimmed least squares | **Trimmed LS (3 → 1 cm).** It is deterministic, uses thousands of points (standard error < 0.1 mm), and ignores clutter and corner droop beyond 1 cm. It is the same function on raw, TSDF and laser. | Plane RMSE: raw ≈ 5 mm, laser 1–3 mm |
| D9 | Measure on raw points or on the TSDF (tests D-007) | — | **Raw.** The validation confirms D-007: the TSDF is *biased* for dimensions, not just smoother. | Room box TSDF −2.47 / −2.62 / −2.48 cm vs raw +0.25 / −1.52 / −1.92 cm. Pair-fit offset TSDF −2.33 cm vs raw −1.47 cm |
| D10 | Confidence threshold (tests D-006) | conf ≥ 2, ≥ 1, ≥ 0 | **Keep ≥ 2.** It gives the best surface accuracy. Dimensions are within noise of each other on this room. | Raw p2pl median 1.12 / 1.21 / 1.25 cm. p90 3.59 / 3.88 / 4.16 cm. Height error −1.92 / −1.28 / −1.52 cm (no consistent winner, single room) |
| D11 | Edge margin for plane fits | 0, 5, 10 or 20 cm rim removed | **0.** Removing the rim does not remove the bias, and it destroys the comparison set, because the matched surfaces collapse from 23 to 5. | See Issue 3 |
| D12 | Room-box wall choice | nearest wall to the centre; widest pair; largest surface on each side | **Largest.** Walls are the largest surfaces. "Nearest" picked a 0.70 m gap between a shower screen and a cabinet. "Widest" spans the door into the vestibule. | Issue 2 |

---

## 4. Issues log

**Issue 1: the first dimension comparison gave −92 cm and +39 cm errors.**

- **Symptom.** Measuring "the strongest inward-facing plane on each side" independently on each cloud gave width_x
  1.547 vs 2.469 m and width_z 2.006 vs 1.616 m.
- **Root cause.** A diagnostic top view at two heights showed what happened:
  - The scanner stood at (0.64, −0.06) in our frame and, through the door, saw a long vestibule wall at x = −0.60. That
    became its "strongest +x wall", while ours was the bathroom wall at x = 0.33.
  - On the other axis, the laser saw the back wall (z = 1.08) only in fragments behind the shower curtain and cabinet, so its
    strongest −z surface was a cabinet front at z = 0.70.
  - Neither cloud was wrong. The *correspondence* was.
- **Possible fixes.**
  - (a) Restrict the laser to our bounding box: this still leaves the cabinet-front problem.
  - (b) Use the band 1.0–1.5 m instead of 0.6–1.9 m: this does not address occlusion.
  - (c) Match surfaces: find them on ours, then fit the laser on the same surface and the same cells, from laser points only.
- **Chosen.** (c). It is the only option that compares like with like when one sensor sees part of a wall. It also yields
  many pairs (43 on raw) instead of three numbers, which gives a distribution.

**Issue 2: too many surfaces, duplicates, and a "room box" of 0.70 m.**

- **Symptom.** The first matched version found 147 surfaces in the raw cloud. Each noisy wall appeared several times at 5 cm
  spacing, and the room box picked a 0.70 m "width".
- **Root cause.**
  - Independent histogram peaks do not remove a surface's points, so the 5 mm raw noise (p90 3.6 cm) produced extra peaks.
  - Cells were counted as occupied by a single stray point.
  - The "smallest separation spanning the centre" rule picked a shower screen and a cabinet.
- **Fix.**
  - Sequential extraction that removes each plane's points.
  - A cell must hold at least 20% of a fully seen cell's points at that cloud's resolution.
  - The largest surface on each side defines the room box.
- **Result.** 25 surfaces (raw) and 21 (TSDF), each corresponding to a visible structure in `plan_overlay.png`.
- **Alternatives considered.** RANSAC: random and slower. Region growing on normals: more parameters, and no benefit
  once the walls are axis-aligned.

**Issue 3 (main finding): our interior distances are about 1.5 cm short. Ceiling height is −1.92 cm, outside the 1.5 cm gate.**

- **Symptom.** Errors on the facing pairs are mostly *negative*. Room-box height is −1.92 cm and depth −1.52 cm; full-height
  pairs have a signed median of −1.54 cm.
- **Evidence.**
  1. Point level: the signed median is **+0.38 cm** (raw) and **+0.48 cm** (TSDF). Our surfaces sit in front of the true ones, toward
     the camera.
  2. It is **not a depth scale error**. The signed median by range is flat: +0.61, +0.23, +0.51, +0.56 and +0.23 cm for 0.2–0.5 m through 2–3 m.
  3. It is **not an incidence-angle effect**. The signed median is flat at +0.34 to +0.40 cm from head-on to grazing.
  4. Per surface, the inward offset (ours minus laser, measured into the room) has a median of **+0.58 cm**. An interior distance loses about twice that, which is consistent with
     the −1.47 cm offset of the error-versus-distance fit.
  5. The bias is **concentrated near concave corners**. The signed median by distance to the nearest other room-box plane is:

     | Distance to corner | 0–5 cm | 5–10 cm | 10–20 cm | 20–40 cm | 40–100 cm |
     |---|---|---|---|---|---|
     | Signed median | +0.28 cm | **+0.84 cm** | +0.58 cm | +0.41 cm | **+0.00 cm** |

     The 0–5 cm bin is low only because points drooping more than 5 cm fall outside the selection.
  6. A one-off diagnostic (scratch script, not in the repo) of the ceiling along x showed the same droop:
     - raw ceiling 6.4 cm below the laser at the door wall (x = 0.3 m);
     - 0.5–0.7 cm below mid-room;
     - 6.5 cm below at the far wall.

     The door header shows +3.5 cm.
- **Root cause (hypothesis, partly supported).** iPad "scene depth" is a learned densification of a sparse
  LiDAR dot pattern, and it rounds concave corners toward the camera. A further small, uniform inward
  offset (about 0.5 cm, visible on the ceiling mid-room and on the floor, +0.53 cm) remains. One room cannot tell whether
  that offset comes from the device's range bias, the glossy tiles, or the laser.
- **Fixes tried.**

  | Fix | Margin | Result | Why it failed |
  |---|---|---|---|
  | (a) Edge margin: drop the rim of each matched surface | 0 / 5 / 10 / 20 cm | Matched surfaces fall from 23 to 19, 12 and 5. Inward offset median 0.58 → 0.64 → 0.46 → (too few). Room-box height −1.92 → −1.80 → −1.61 cm, depth −1.52 → −1.45 → −2.17 cm | The bias extends to about 40 cm, and the single-scan coverage is too patchy to survive erosion |
  | (b) Corner exclusion: fit each room-box plane only on points ≥ m from the other box planes | 0 / 0.2 / 0.3 / 0.4 m | Height −1.91 → −1.81 → −1.77 → −1.86 cm. Depth worsens −1.52 → −2.03 → −2.61 → −4.16 cm as the back wall's support drops from 6,640 to 606 points. Width +0.27 → +0.18 → −0.15 → +1.08 cm | The height bias persists mid-surface |

- **Remaining options.**
  - (c) A calibrated outward correction of about 0.5 cm per surface, plus a corner-aware weighting.
  - (d) Do not correct, but put the systematic term into every confidence interval.
  - (e) Measure more ARKitScenes rooms before choosing.
- **Chosen for now.** (d) + (e).
  - With one room, a correction learned here would be fitted and tested on the same data. That is the overfitting the fix loop must avoid.
  - The CI model should add a systematic term of about **±1.5 cm per interior distance** (95%). The statistical
    standard error of a plane fit is 0.04–0.08 mm, about 200× too small.
  - Next step: run this script on 3–5 more single-scan ARKitScenes rooms with different materials. The script takes any
    `--scene/--laser`, and each laser scan is about 1.8 GB, fetched by `../data/arkitscenes/fetch_parallel.sh` (with the data, outside the repo).
  - If the offset is stable, (c) becomes a Part 4 fix with a hold-out room.

**Issue 4: the edge-margin erosion first deleted almost every surface.**

- **Symptom.** With 10 cm erosion of the *common* (ours ∩ laser) cells, only 9 surfaces matched and the room box lost an axis.
- **Root cause.** The laser coverage is patchy, so eroding the intersection removes most of it.
- **Fix.** Erode *our* dense patch, which is where the physical rim is, and then intersect with the laser cells.
- **Result.** Better, but still not useful (Issue 3a).

**Issue 5: the earlier description of the room (data README: "about 2.0 × 2.5 m") differs from our room box (1.54 × 2.01 m).**

- **Root cause.** The smoke test measured across the open door into the vestibule (x −0.60 → 1.88 m). Our room box uses the
  bathroom's own door wall at x = 0.33 m.
- **Evidence.** Both pairs exist in `results.json`. The 2.47 m distance across the door is in four pairs, because different
  surfaces face each other there. Its error is +0.03 cm on the main vestibule-wall surface and −3.6 to −8.8 cm on small partial surfaces.
- **Status.** Not an error. It shows how much "room" depends on segmentation, which is the plan-extractor's job.

**Issue 6: open3d and determinism.**

- **Symptom.** The VoxelBlockGrid's point order changes between runs.
- **Fix.** Sort the TSDF points. Voxel subsampling uses `np.unique` (the first point per voxel) instead of Open3D's hash map.
- **Evidence.** Repeated runs give identical numbers (room box +0.25/−1.52/−1.92 in 5 runs; ICP RMSE 1.214 every time).

---

## 5. Results on the captures

Command and timing:

- The run is in `run_log.txt`.
- Scene 47895909: 33.6 s, 2,061 depth frames, 337 poses.
- About 23 s per variant and 139 s in total with the ablations, on a shared CPU. No GPU is used.

**Registration.**

| Quantity | Value |
|---|---|
| Laser floor tilt | 0.16° |
| Yaw scores | 0.705 for 0°, ≤ 0.31 for the others |
| ICP inlier RMSE | 1.21 cm |
| ICP fitness | 0.53 |
| Difference from the independent smoke-test transform | 0.16° / 3.6 cm |

**Surface accuracy** (cm, overlap = has a laser point within 10 cm):

| | Points | Overlap | Point-to-plane median / p90 / p95 | Point-to-point median / p90 / p95 | Signed median | Completeness (laser within 5 cm of ours) |
|---|---|---|---|---|---|---|
| TSDF surface | 72,842 | 87.3% | **0.96 / 3.34 / 4.46** | 1.28 / 4.94 / 6.79 | +0.48 | 74.9% |
| Raw LiDAR points | 1,637,679 | 86.4% | **1.12 / 3.59 / 4.66** | 1.42 / 4.76 / 6.40 | +0.38 | 85.6% |

Over all points, point-to-point is 1.58 / 13.3 / 24.8 cm (TSDF) and 1.73 / 15.7 / 27.8 cm (raw). The large tails are laser holes, not our error.

**Dimensions** (ours from raw points; the laser is fitted by the same function):

| Room box | Ours | Laser | Error |
|---|---|---|---|
| Width x | 1.5363 m | 1.5337 m | **+0.25 cm (+0.17%)** |
| Depth z | 2.0094 m | 2.0246 m | **−1.52 cm (−0.75%)** |
| Floor-to-ceiling height | 2.3416 m | 2.3608 m | **−1.92 cm (−0.81%)** |

The same room box measured from the TSDF errs by −2.47, −2.62 and −2.48 cm.

All facing pairs measured from the raw points (|error| in cm):

| Kind | n | Median | p90 | Max | Within 1 cm | Within 2 cm |
|---|---|---|---|---|---|---|
| Full height | 17 | 1.54 | 6.09 | 6.26 | 29% | 53% |
| Partial height | 24 | 3.45 | 7.70 | 11.09 | 8% | 13% |
| Height | 2 | 1.80 | — | — | — | — |

Error-versus-distance fit over the 43 pairs: offset −1.47 cm, slope −0.89 cm/m.

**Error breakdowns** for the raw points (median |e|, cm):

| Range | 0.2–0.5 m | 0.5–1 m | 1–1.5 m | 1.5–2 m | 2–3 m |
|---|---|---|---|---|---|
| Median \|e\| | 1.58 | 0.96 | 1.22 | 1.31 | 1.22 |

| Incidence | 0–15° | 15–30° | 30–45° | 45–60° | 60–75° | 75–90° |
|---|---|---|---|---|---|---|
| Median \|e\| | 1.03 | 1.03 | 1.14 | 1.10 | 1.16 | 1.33 |

The corner profile is in Issue 3.

**Ablations** (room-box errors are width x / depth z / height):

| Variant | Keyframes | Raw p2pl median / p90 | TSDF p2pl median | Room-box error (cm) |
|---|---|---|---|---|
| **nearest, conf ≥ 2 (default)** | 217 | **1.12 / 3.59** | **0.96** | +0.25 / −1.52 / −1.92 |
| interpolate, conf ≥ 2 | 316 | 1.21 / 3.84 | 1.06 | +0.96 / −1.55 / −1.98 |
| nearest, conf ≥ 1 | 221 | 1.21 / 3.88 | 1.02 | +0.38 / −1.49 / −1.28 |
| nearest, conf ≥ 0 | 226 | 1.25 / 4.16 | 1.09 | +0.31 / −1.50 / −1.52 |

**What the gates say for this one room.**

- Wall-to-wall distances are within 2 cm on 53% of full-height pairs.
- The room box is within 2 cm on all three axes.
- Ceiling height is −1.92 cm, which **fails the 1.5 cm ceiling-height gate** in absolute terms.

The cause is the systematic inward bias (Issue 3), not noise. This is the honest number to report.

**Figures.**

- `plan_overlay.png`: top view of the laser (left) and our raw points (right). It shows every matched surface fitted on ours (solid; thick = room box) and on the laser (dashed), with the camera path.
- `raw_error_map.png`: |error| per raw point in a top view, plus each room-box wall face-on and the ceiling. Grey means the laser has no point there. You can see:
  - the door opening in the x = 0.33 m wall;
  - the header above it (dark, about 3–5 cm);
  - the back wall at z = 1.08 m, which the laser barely saw (mostly grey);
  - the ceiling strip near the door wall.
- `error_stats.png`:
  - signed error histograms for raw vs TSDF (both shifted positive);
  - error by range, with the signed median flat;
  - error by incidence angle, also flat.

---

## 6. Limitations, failure modes and next steps

- **One room, one device, one scanner position.** Every number above has an unknown room-to-room spread. The 0.5 cm
  inward offset needs more rooms before we correct for it (Issue 3). Next: 3–5 more single-scan rooms,
  different materials.
- **Registration is ICP-optimised.** The cloud-to-cloud numbers are therefore an *optimistic* (best rigid fit) measure of shape accuracy. Dimensions
  are the registration-independent check.
- **Single-scan laser occlusion.**
  - About 14% of our points have no laser partner.
  - Several walls are compared on small stretches. The back wall had 0.31 m² in common out of 1.74 m².
  - The tripod leaves a hole of about 0.6 m radius in the floor.
- **Low light: covered.** The scene is dark, with a mean RGB luma of 43/255. LiDAR does not need light, and 84.5% of pixels are still
  high-confidence. The numbers above *are* low-light numbers for the LiDAR tier. RGB-based tiers will not be this lucky.
- **Mirrors and glass: partly covered.**
  - The bathroom has a small mirror cabinet and a shower area.
  - We count raw points more than 10 cm behind a room-box surface (`raw_beyond_room_box`): 258,112 points (15.8%), of which **81% are
    confirmed by the laser**. They are the vestibule, seen through the open door, and a high opening in the x+ wall, not ghosts.
  - So this capture shows no measurable mirror "room behind the wall". The check is in place for other captures.
  - The confidence filter (D-006) is what removes most mirror and glass returns. At conf ≥ 0 the raw error p90 grows from 3.59 to 4.16 cm.
  - What we cannot test here: a large wall mirror or a glass partition, which would create a planar "surface" the laser also does not
    see behind.
  - Mitigation for the pipeline:
    - flag surfaces whose points lie behind an already-fitted wall;
    - flag large planar holes at confidence 0/1 as `not_observed` instead of open space (D-006).
- **Wet-look and glossy surfaces.** The glazed tiles here are glossy. They may be the reason for the inward offset, but this is not separable with one room.
- **Clutter.** Furniture-front pairs (`partial_height`) have larger errors: median 3.45 cm, max 11 cm. Small, partly seen
  surfaces fit worse. This is a reason for the room box to use the largest surfaces.
- **The validation reimplements a plane fitter.** The wall and plan modules have their own. When those are final, this script
  should call them, so the validation tests the actual measurement code (requested change 3).

**Requested changes to shared code** (not made; owners decide):

1. `floorplan/pipeline/scene.py`: let `build_scene` accept a capture object, or a loader callable, in addition to a Stray
   folder. This removes the duplicated stage 1 in `build_stage1` and guarantees the validation runs the literal same function.
2. `floorplan/io/__init__.py`: add a `load_capture(path)` that detects the format (`odometry.csv` → Stray;
   `lowres_wide.traj` → ARKitScenes), so the CLI and `scripts/prepare_scene.py` also work on ARKitScenes.
3. Wall and plane measurement module: expose its plane fit as a function
   `(points, normals, seed plane) → plane + support`, so `validate_arkitscenes.py` can use it instead of its own
   `fit_plane_trimmed`.
4. Confidence-interval model: add a systematic term for LiDAR interior distances of about ±1.5 cm at 95%. It is derived here, and should be re-derived
   when more rooms are measured. Do not rely on the statistical standard error (0.04–0.08 mm).
5. `docs/DECISIONS.md`: add the decision entry suggested below.

> **D-0xx Absolute accuracy of the LiDAR tier, measured on ARKitScenes with a Faro laser.**
>
> - Decision: use nearest-pose matching, conf ≥ 2, and measure on raw points.
> - Evidence: room box +0.25 / −1.52 / −1.92 cm (raw) vs −2.5 cm (TSDF). Surface error median 1.1 cm.
> - Systematic inward bias of about 0.5 cm per surface, worst near corners. Not corrected yet; carried in the confidence interval.
