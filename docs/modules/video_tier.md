# Module: video tier (`floorplan/video/`)

Status: runs end to end on all three sample captures.

- On the short capture (single_room, 37 s) it recovers metric scale to +1.2% to +4.2% (4 runs) with a 10–13 cm
  camera-path error over 14 m. Its claimed interval covers the error.
- On the two long whole-apartment captures, DPVO loses track and restarts its scale. The pieces are metric locally
  (most 30-keyframe windows within ±10%), but they are placed wrongly relative to each other. The output is
  **flagged** (`whole_scene_consistent = false`), not usable for a stitched plan.
- The 3% video gate is **not** met reliably (sections 5 and 6).
Every number in this document comes from code in this module. The evidence files are listed in section 5.

**v2 (end of this document)** adds handling for long captures and keeps the v1 text above as the record. The
changes:

- segments at VO restarts, with a fresh tracking run and a self-check for each;
- untrusted segments are dropped from the scene;
- a pose graph on predicted depth (reusing `recon/drift.py`);
- focal length from metadata, SfM and GeoCalib, with its uncertainty propagated;
- the paper-sheet cue with gates (opt-in, off by default since D-067);
- honest `scale_sigma_rel` and `whole_scene_consistent` keys.

Results with v2:

- floor_only, trusted part: +2.9% scale error and 36 cm ATE over 35 m (v1: +187%, 455 cm), still flagged as not one
  rigid map.
- single_room: +2.2% and 10.2 cm (v1: +4.2%, 12.7 cm).
- with_ceiling: 6 of 8 trusted segments within ±4%, one self-check miss, flagged.

---

## 1. Purpose and where it sits in the pipeline

The case study defines the video tier as *"a handheld walkthrough clip from any iPhone 15 or newer"*. That means
**RGB video only**: no depth, no poses, no IMU, no intrinsics file. This module turns such a clip into the same
**metric, gravity-aligned 3D scene** that the LiDAR tier produces (`scene.npz` + `scene_info.json`, the format of
`floorplan.pipeline.scene.save_scene`). The plan extractors (walls, rooms, openings, stitching) then run unchanged
on either tier.

```
rgb.mp4 ──► [video tier front-end]  ──► scene.npz (points, normals, colors, raw_points, traj, kf, T_align, ...)
                                         scene_info.json (floor_y, ceiling_y, scale + its uncertainty, ...)
                                              │
LiDAR capture ──► [LiDAR front-end] ──────────┤  same format
                                              ▼
                          plan extraction (shared back-end) ──► dimensioned plan + intervals
```

The design principle is the same as D-003/D-004: **tiers differ only in how they get metric geometry and how wide
their intervals are.**

On the sample data the video is `rgb.mp4` from a Stray Scanner capture. The front-end never opens the depth,
odometry, camera-matrix or IMU files. The evaluation code (`evaluate.py`, `experiments.py`) does read them, but only
to *score* the video tier against the LiDAR tier of the same capture. The LiDAR tier is a reference here, not ground
truth.

Entry point:

```python
from floorplan.video import build_scene_from_video, VideoParams
scene, info = build_scene_from_video("TakeHome/Dataset/single_room/c00a170fe1", VideoParams())
```

Command line (saves the scene, and evaluates it if the LiDAR scene of the same capture exists):

```bash
python scripts/fetch_weights.py                 # once: pinned weights (add --experiments for the comparison models)
HF_HUB_OFFLINE=1 python scripts/run_video_frontend.py TakeHome/Dataset/single_room/c00a170fe1
python -m floorplan.video.experiments <capture_dir> outputs/video_tier/<name>    # option comparison (V-3, V-5)
```

**Dependencies and environment.**

- No packages were installed for this module.
- Main env (`.venv`): GeoCalib, hloc/LightGlue/ALIKED, pycolmap, MoGe-2, Open3D. The comparison also uses DA3 and
  MapAnything.
- DPVO runs as a subprocess in its own env (`envs/dpvo`, compiled CUDA ops). Its paths come from
  `FLOORPLAN_DPVO_PYTHON`, `FLOORPLAN_DPVO_REPO` and `FLOORPLAN_DPVO_SHIMS`, which default to this machine's layout.
- GPU use stays small on the shared 8 GB card: DPVO 0.8 GB, MoGe-2 at 504 px about 2 GB, GeoCalib about 1 GB. The
  models are loaded one after the other and freed.
- Runtime is 252 s for 37 s of video (single_room) and 633 s for 115 s (floor_only). DPVO, then MoGe-2, dominate.

**Licences of what runs in the pipeline.** DPVO (MIT code; weights without a separate licence), MoGe-2 (MIT),
GeoCalib (Apache-2.0 code, CC-BY-4.0 weights: attribution), ALIKED (BSD-3), LightGlue (Apache-2.0), COLMAP/pycolmap
(BSD). The comparison-only models are DA3-BASE and DA3METRIC-LARGE (Apache-2.0) and `facebook/map-anything-apache`
(Apache-2.0). The non-commercial variants are deliberately not used.

---

## 2. How it works, step by step

| # | Step | File | What it does | Why it is needed | What goes wrong without it |
|---|---|---|---|---|---|
| 1 | Scan the video | `frames.py` | Decodes every frame once at 160 px width. Records the timestamp, the image shift relative to the previous frame (phase correlation) and the sharpness (variance of the Laplacian). | Without poses, the only way to know "how much the view changed" is to measure it in the image. | Fixed-rate sampling. It wastes frames when the operator stands still and leaves gaps with no overlap during fast turns, and the gaps break the tracking. |
| 2 | Keyframes | `frames.py` | A new keyframe is taken when the picture has moved 15% of its width (about 9°) or 1 s has passed. Within each trigger, the sharpest of the last 6 frames is kept. Keyframes are snapped to the frames DPVO sees (every 2nd frame). | It is the LiDAR keyframe rule (D-008) translated to image space. Picking the sharpest frame avoids motion blur. | Blurred keyframes give bad depth and bad colour. |
| 3 | Upright rotation | `orientation.py` | Uses the container's rotation flag if it has one (iPhone Camera-app videos do). Otherwise GeoCalib votes on which 90° rotation makes gravity point down the screen. | The phone was held in portrait, but `rgb.mp4` stores the frames sideways and carries no flag. Learned models expect upright images. | Depth and calibration models run on sideways images and degrade. |
| 4 | Focal length | `sfm.py`, `orientation.py` | Self-calibrates one shared focal length with SfM (ALIKED + LightGlue + COLMAP bundle adjustment) on the first 120 keyframes. GeoCalib's median focal is the prior and the fallback. | No intrinsics file exists. The focal length sets the field of view, which every depth and pose model needs. | A focal error of e% stretches lateral dimensions by about e%. |
| 5 | Camera path | `vo.py`, `dpvo_runner.py` | DPVO (deep patch visual odometry) on every 2nd frame at 480x640, in its own interpreter. | It gives a smooth, locally accurate camera path for every frame, including stretches with blank walls. | SfM breaks into fragments on this footage (V-3). |
| 6 | Gravity + flip check | `orientation.py`, `frontend.py` | GeoCalib's per-frame "up" vector is rotated into the DPVO world by the camera orientations, and outliers are removed from the mean. Then a check: phones film walkthroughs looking down, so if the median camera pitch comes out *above* the horizon, the frames are upside down. In that case they are turned 180° and "up" is flipped. | Heights (ceiling, sills) need "up". GeoCalib alone cannot tell 0° from 180° (Issue 2). | The scene could be built upside down. |
| 7 | Metric depth | `depth.py` | MoGe-2 depth for each upright keyframe (504 px), with the focal length passed in. Pixels beyond 4 m and "flying pixels" at depth edges are removed. | This is the only metric signal in an RGB video. | There would be no scale and no dense surface. |
| 8 | Scale + drift correction | `scale.py` | For keyframe pairs, finds the translation scale at which the two metric depth maps agree, and cuts the path at scale *jumps*. The local scale inside a segment is the sum of the depth-agreement curves over a time window, clamped to ±1.5× of the segment's scale (the default again since D-076). Since D-087 (5 Oct) the default is `scale_method = "auto"`: PnP's scale on a segment whose votes cover at least half of its keyframes, else this depth agreement. With `scale_method = "pnp"` (D-075, opt-in) it is the running median of PnP votes (SIFT matches on keyframes 2, 4 and 6 apart, MoGe-2 depth, metric step / DPVO step, ±8 keyframes), with a vote-coverage self-check and the path levelled to the floor each keyframe sees. The path is re-integrated with the local scale. It also estimates the scale uncertainty (block bootstrap over the votes plus a model-bias term). | DPVO's scale is unknown, drifts and can jump (V-6). | Wall lengths would be off by 20% or more (measured). |
| 9 | Fusion | `frontend.py` → `floorplan.recon.fusion.fuse_tsdf` | The predicted depth maps are fused with the **same TSDF code as the LiDAR tier**, through a small adapter object (`VideoCapture`). | Averaging many noisy depth maps gives one consistent surface. | Duplicate, noisy surfaces. |
| 10 | Alignment | `floorplan.plan.align` | Gravity is refined on the floor plane (up to 10°). Then Manhattan yaw, floor level and ceiling level are computed with the shared LiDAR-tier code. | Heights are measured along "up", and walls should run along x and z. | Tilted floors and walls at odd angles. |

The output has the same keys as the LiDAR scene. Three things differ:

- `T_wc` and `traj` cover every decoded frame. DPVO poses (every 2nd frame) are interpolated: linearly for
  position, with slerp for rotation.
- `raw_points` are back-projected *predicted* depth points, not sensor points. They are noisier than the TSDF
  surface (C2C median 8.1 cm versus 7.6 cm on single_room, final run; 7.6 versus 5.9 cm in the first run). So `info["recommended_measurement_cloud"] = "points"`: the plan
  extractors should fit walls on the TSDF points for this tier (see the change request at the end of section 6).
- `info["scale"]` holds the scale uncertainty, `sigma_rel_total` (1-sigma, relative). Every length measured on this
  scene must carry at least this relative uncertainty.

---

## 3. Decisions

### V-1 Upright rotation: container flag, then a GeoCalib axis vote, then a pitch-sign check

- **Context.** `rgb.mp4` stores 1920x1440 landscape frames of a portrait capture, and ffprobe shows no rotation
  tag. The IMU must not be used in this tier.
- **Options.**
  1. The IMU gravity direction. Forbidden in the video tier.
  2. The container rotation tag. This is the right answer for real Camera-app videos, but it is absent here.
  3. GeoCalib roll/up on all 4 rotations, keeping the one where "up" points up the screen.
  4. A semantic orientation classifier. None is installed, and it would be one more model.
- **Decision.** (2) when present. Otherwise (3) to pick the *axis*, followed by a physical sign check after the
  odometry.
- **Why.** GeoCalib resolves the axis but not the sign (Issue 2). The sign check uses a strong behavioural prior:
  the median camera pitch during a walkthrough is clearly negative. Measured with ARKit on the three captures:
  -31°, -27° and -19°.
- **Evidence.** In the final runs the vote picked 90° (7, 8 and 8 of 12 probe frames). That matches ARKit: world-up
  projects to image -x in the stored frames, so rotating 90° clockwise makes the image upright. The median camera
  pitch after the check was -31.5° on single_room.
- **Risk.** If a user films mostly the ceiling, the pitch prior fails. With broken gravity, the pitch can sit near
  0° (Issue 15), so a flip requires a margin of +10°. The flip is logged in `info.rotation.flipped_by_pitch_check`.

### V-2 Keyframes from image motion (phase correlation), 15% of the width or 1 s, sharpest of 6

- **Options.** Fixed rate (for example 3 fps); optical-flow magnitude; phase correlation; learned frame selection.
- **Decision.** Phase correlation on 160 px frames, the cheapest global shift estimator. The keyframe threshold is
  0.15 of the width.
- **Evidence.** Phase-correlation shift against ARKit rotation on single_room: for the 1,661 frames with peak
  response > 0.4, the median shift is 0.68% of the width per frame, against 0.82% predicted from ARKit rotation.
  Below response 0.1 (5 frames) the shifts were random, so they are capped at 5%. With an 8% threshold we got 365
  keyframes for 37 s, which is too many for SfM and depth. At 15% we get 186, or 164 after snapping to the DPVO
  frames.

### V-3 Camera trajectory: DPVO (chosen) vs SfM vs MapAnything vs Depth Anything 3

All four were run on single_room's keyframes and scored against ARKit. Evidence:
`outputs/video_tier/single_room__c00a170fe1/experiments.json`.

| Option | What happened | Number |
|---|---|---|
| **hloc SfM** (ALIKED + LightGlue, COLMAP incremental mapper) | Breaks into fragments. In the 186-keyframe set, 43 of the 185 consecutive-keyframe pairs (9 stretches) have **zero** verified matches: blank walls, curtains, close-ups (Issue 5) | 35/164 keyframes in the largest model (41/186 at the 15% threshold before snapping) |
| **MapAnything** (Apache weights), images only, 30-keyframe windows | "Metric", but unreliable on this footage | Over the 5 windows: focal -36% to +3%, metric scale error -9% to +71%, Sim3 ATE 10–54 cm, rotation error 3.5–46° |
| **Depth Anything 3** (DA3-BASE) any-view, 30-keyframe windows | Good in some windows, broken in others. All 186 frames at once does not fit 8 GB at 504 px; at 336 px it fails (1.69 m ATE, scratch test) | Sim3 ATE 3.7–43 cm per window, rotation error 1.2–43° |
| **DPVO** (chosen) | Tracks all 857 frames. Locally excellent, but the scale drifts (and jumps on floor_only, V-6) | Sim3 ATE per 30-keyframe window 1.3–3.9 cm; scale per window varies 0.84–1.05 (first run: 0.87 → 0.69) |

- **Decision.** DPVO for the path, with metric scale from depth (V-6).
- **Why.** It is the only option that tracks *through* the textureless stretches. It is fast (109 s for 1,714 frames
  on single_room, 0.83 GB of GPU), and its remaining failure (scale drift) can be measured and corrected.
- **Licence.** DPVO code is MIT; the weights ship without a separate licence (commercial use probably fine:
  `docs/DISCLOSURES.md` §1).
- **Risk.** DPVO has no relocalisation. The DPV-SLAM loop closure is tested as an ablation (section 5).

### V-4 Focal length: SfM self-calibration with a GeoCalib prior

ARKit's per-frame focal length (median) is the reference: 1599.7 px (single_room) and 1599.2 px (floor_only) at
1440x1920.

| Capture | SfM self-calibration, 4 runs on the same frames | GeoCalib median |
|---|---|---|
| single_room | 1587.8 / 1549.5 / 1561.5 / 1545.8 px (-0.7% to -3.4%; 35 images registered) | 1536.4 (-4.0%) |
| floor_only | 1545.3 / 1555.7 / 1551.9 / 1549.2 px (-2.7% to -3.4%; 23 images registered) | 1586.4 (-0.8%) |

- **Decision.** Use SfM when its model registers at least 20 images, otherwise GeoCalib.
- **Honest note.** Neither source is reliably better than about 3% on this footage, and SfM is not even repeatable
  (Issue 13c). The focal error goes into the scale almost 1:1 (Issue 13e), so **focal estimation is the largest
  error source on the short capture**. Reading the focal length from the video metadata is the first item in
  section 6.

### V-5 Depth model: MoGe-2 (chosen) vs DA3METRIC-LARGE

Measured against high-confidence LiDAR depth on the same 164 single_room keyframes, with the focal length passed to
both models:

| Model | Median per-frame scale ratio (pred/LiDAR) | p10–p90 | AbsRel after per-frame scale | Time (164 frames) |
|---|---|---|---|---|
| **MoGe-2 ViT-L** (MIT) | **0.952** (-4.8%) | 0.82–1.03 | **2.8%** | ~60 s |
| DA3METRIC-LARGE (Apache-2.0) | 0.918 (-8.2%) | 0.76–0.98 | 4.3% | ~17 s |
| floor_only, MoGe-2 (432 keyframes) | 0.929 (-7.1%) | 0.79–1.01 | 2.8% | |
| floor_only, DA3METRIC | 0.897 (-10.3%) | 0.74–0.98 | 3.9% | |

- **Decision.** MoGe-2. It has half the bias and a better shape, at a runtime cost we can afford.
- **Evidence.** `experiments.json → depth_models_vs_lidar`; floor_only numbers are in section 5.

### V-6 Metric scale: summed depth-agreement cost curves, local window, jump segmentation

- **Context.** The DPVO scale per 30-keyframe window on single_room goes 0.87, 0.91, 0.79, 0.80, 0.69 (raw DPVO,
  ARKit Sim3). On floor_only it **jumps**: about 0.37 → 0.94 at keyframe ~150 and → 10.9 at ~330. A single global
  scale gives 16.8 cm SE3 ATE on single_room and destroys floor_only.
- **Options.**
  1. One global scale from the median ratio of mono depth to triangulated SfM points. This is impossible here: SfM
     fragments, and the scale is not constant anyway.
  2. Per-pair minimum, then a median. Measured: per-pair errors have a MAD of 10–36%, so this is too noisy.
  3. **Sum the per-pair cost curves** over a time window, then take the minimum.
  4. Feed depth into a dense RGB-D SLAM (DROID-SLAM). Not installed, and a large integration.
- **Decision.** (3), with a Gaussian window (σ = 15 keyframes) that never crosses a detected **scale jump**.
  - A jump is a >2x step in a running median (±8 keyframes) of per-pair estimates that persists for ≥ 20
    keyframes.
  - Inside a segment, the local scale is clamped to within 1.5x of the segment's scale, because windows with
    little information (for example the last keyframes) otherwise wander.
  - Changed by D-075 (5 Oct, fix loop): the Gaussian window and the 1.5x clamp were replaced by a running median of
    PnP votes. The jump cuts and the merge below stay.
  - D-076 (5 Oct, follow-up): the window and the clamp are the default again (`scale_method = "depth_agreement"`). The
    PnP votes are opt-in (`"pnp"`, `run_capture.py --video-scale pnp`): better on my own take1, worse on the sample
    videos, where SIFT finds few features on white walls.
  - After the fine curves are computed, adjacent segments whose fine scales differ by less than sqrt(2) are merged
    again. This is a false-cut guard (Issue 13b).
  - Every scene reports `info.scale.vo_restarts` and `whole_scene_consistent`. After a restart, each segment is
    metric on its own, but the placement of segments relative to each other depends on DPVO's single step across the
    restart. The plan back-end must not stitch across that boundary as if it were rigid.
- **Evidence (single_room).**

  | Variant | SE3 ATE | Scale error |
  |---|---|---|
  | Global scale only | 16.8 cm | -3.3% |
  | Local, σ = 10 | 11.3 cm | +1.75% |
  | Local, σ = 20 | 10.8 cm | +1.55% |

  On floor_only (one DPVO run, scratch test on the same depth and poses): without segmentation the windows after
  the restarts were off by -93%. With it, most windows are within ±10% (whole-path SE3 ATE 170 → 76 cm in that
  run). DPVO restarts differ between runs (Issue 13c); the final run's whole path is still wrong (section 5).

### V-7 Scale uncertainty: bootstrap (statistical) ⊕ depth-model bias (systematic)

- **Context.** Bootstrapping the pairs only captures noise. MoGe-2's systematic bias (-4.8% on single_room, see V-5)
  does not average out.
- **Decision.** σ_total = sqrt(σ_stat² + σ_model²) with σ_model = 4% (a parameter). σ_stat comes from a block
  bootstrap (blocks of 10 consecutive pairs, 2,000 resamples) per scale segment, weighted by keyframes.
- **Evidence.**
  - single_room, final run: claimed 95% interval ±12.8%; measured whole-path scale error +4.2%; all 5 windows
    inside the interval (`scale_error_windows.png`).
  - The interval is honest but wide, wider than the ±3% gate, so the video tier **cannot certify** the gate on its
    own (section 6).
  - The measured MoGe-2 bias is -4.8% (single_room) and -7.1% (floor_only), so σ_model = 4% is at the low edge.
    Calibrating it on ARKitScenes is the fix (section 6).

### V-8 Gravity: GeoCalib up vectors, then floor-plane refinement

- **Decision.** Use only frames within GeoCalib's ±45° training range of pitch. Take the robust mean (drop > 10°
  outliers). Then refine on the fused floor plane with the shared `gravity_correction` (tolerance raised to 10°).
- **Evidence.** single_room: GeoCalib spread 1.9°, residual floor tilt corrected 0.86°. floor_only: spread 4.2°.

### V-9 Fusion: reuse the LiDAR tier's TSDF through an adapter; weight threshold 5

- **Why.** One fusion implementation, already debugged (D-007). The adapter (`VideoCapture`) exposes `depth(i)`,
  `K_depth(i)`, `K_rgb`, `rgb_size` and `iter_rgb`.
- **Settings.** The weight threshold is 5 instead of 3, because predicted depth needs more views to agree. The range
  cap is the same 4 m. Pixels on depth jumps of more than 5% are removed.

### V-10 Evaluation protocol

- The trajectory is reported after SE3 alignment (what a user gets) and after Sim3 alignment (shape only). The Sim3
  factor gives the scale error.
- Rotation error is aligned on orientations (chordal mean), not on camera centres (Issue 9).
- Cloud-to-cloud (C2C) distance runs from video points to the LiDAR TSDF after the trajectory SE3 alignment, with
  no ICP. It therefore includes the trajectory error, which is the honest number.

---

## 4. Issues log

| # | Symptom | Root cause (evidence) | Possible fixes | Chosen and why |
|---|---|---|---|---|
| 1 | Frames look sideways | Portrait capture stored landscape. ffprobe shows no rotation tag. ARKit shows world-up projecting to image -x | Read the IMU (forbidden); ask the operator; detect from the image | Detect from the image (V-1). Real Camera-app videos carry the tag, and it is read first |
| 2 | GeoCalib scores 0°/180° (and 90°/270°) **identically** | GeoCalib's up-field always points to the top of the screen (mean up-field (-0.2, -0.97) for every rotation of the same frame). It has no upside-down examples to learn from | A semantic classifier; geometry after reconstruction (points mostly below the camera: 94%/92%/56%); camera-pitch prior (-31/-27/-19°) | Pitch prior. Strongest signal, and available before depth, so depth runs on upright frames. The point-height cue is too weak on with_ceiling (56%) |
| 3 | The video tier and ARKit disagree by one frame | OpenCV drops the first packet of `rgb.mp4`, which has a negative PTS (-0.0167 s). Decoded frame *i* is odometry row *i+1*: offset 1 gives a 5.7 ms max residual against 33 ms for offset 0. The clocks also drift by up to 29 ms over 215 s | Index arithmetic; timestamp matching | Timestamp matching (`evaluate.match_frames`), robust to both effects. **Also affects the shared loader** (change request) |
| 4 | 8% threshold gave 365 keyframes in 37 s, with bursts of 1-frame gaps | Real fast motion plus a few unreliable phase-correlation peaks (response < 0.1: random shifts) | Raise the threshold; cap unreliable shifts; use optical flow | Both: threshold 15% and cap at 5% (V-2) |
| 5 | SfM registers 41/186 images in 5 models | The capture was made *for LiDAR*: the phone sweeps 0.3–1 m from white walls and curtains. 43 of 185 consecutive-keyframe pairs (9 stretches) have 0 inliers (contact sheet inspected: blank wall, curtain, blur) | More keypoints; denser keyframes; dense matchers (LoFTR/RoMa, not installed, ScanNet-trained = NC risk); learned VO | Learned VO (DPVO) for the path; SfM kept only for focal self-calibration (V-3, V-4) |
| 6 | DA3 on all 186 keyframes: OOM at 504 px, 1.69 m ATE at 336 px | Global attention memory grows with frames × tokens on an 8 GB shared GPU; at low resolution the poses collapse | Windows + Sim3 chaining | Tested windows: too inconsistent (V-3) |
| 7 | MapAnything windows: focal -36%, scale up to +115% | Images-only mode is only roughly metric (noted during environment setup). Close-up blank walls give it no cues | Feed intrinsics; feed poses | Not pursued: DPVO + depth is better and lighter |
| 8 | DPVO scale 0.87 → 0.69 in 37 s; jumps x2.6 and x9 on floor_only | Monocular VO has no metric reference, and DPVO re-initialises scale after losing track on blank walls | Global scale; local scale; jump segmentation; loop closure | Local scale + segmentation (V-6); loop closure as ablation |
| 9 | Rotation error 2.8° on a good SfM fragment | The alignment rotation came from camera *centres* of a short, nearly straight path, which constrains rotation about the path axis poorly | Align on orientations | Chordal-mean rotation alignment (1.2° on the same fragment) |
| 10 | Scale came out ×51 (6,198% error) | Bug: scales at which the views stop overlapping returned `inf`, which my normalisation turned into **zero** cost, so "no overlap" won | Penalise non-overlap | Non-overlap costs TRUNC (the maximum), so it can never win |
| 11 | Per-pair scale estimates: MAD 10–36% | Pure rotations give flat curves; depth errors shift single curves | Weight by sharpness; sum curves | Sum curves and drop flat ones (contrast < 0.005 at ±20%) |
| 12 | floor_only: after the first scale fix, 30 segments, several with nonsense scale (×6 inside a ×0.8 stretch), SE3 ATE 1.9 m | A short running median (±4) produced spurious jumps, and keyframes without information got the grid's first value (1e-3) | Longer median; minimum segment length; fill uninformed keyframes from neighbours | All three. In that run: 6 segments and 76 cm SE3 ATE (the later re-runs got different DPVO restarts, Issue 13c) |
| 13 | MoGe-2 depth is 4.8% too short on single_room | The systematic bias of monocular metric depth at 0.3–2 m (close-ups dominate this capture) | Calibrate the bias on external LiDAR data (ARKitScenes); a known-size object; leave it in the interval | Leave it in the interval (σ_model = 4%) for now. Calibration is in section 6 |
| 13b | With jump = 1.5, single_room (which has no real jump) was cut into 4 segments; SE3 ATE rose from 11 cm to 63 cm | The threshold was too close to the noise of the running median. Real restarts on floor_only are x2.6 and larger | A higher threshold; a longer minimum segment; clamping | jump = 2.0, minimum 20 keyframes, clamp within 1.5x of the segment value |
| 13c | Re-running the same capture gives different numbers (single_room focal 1587.8 / 1549.5 / 1561.5 / 1545.8 px; floor_only DPVO restarts at different keyframes, from 2 to 4 segments) | GPU feature extraction and matching (fp16 LightGlue), multi-threaded COLMAP BA and DPVO's CUDA kernels are not bit-deterministic, even with fixed seeds | Single-threaded BA; deterministic CUDA flags (slow, and not all DPVO kernels have them); repeated runs with the spread reported | Single-threaded BA was tried: still not repeatable (the matches change), so it is kept only as a partial measure. The run-to-run spread is **reported as a result** (section 5). It is a repeatability failure in the sense of the Part 2 gate |
| 13d | A floor_only re-run gave a +64% scale error and 23° rotation error, while claiming σ_total = 9%: **confident garbage** | DPVO lost track several times on that run (4 scale segments). Its rotation also broke. Neither is covered by a statistical interval on the scale | Detect restarts and flag them; refuse whole-scene output; loop closure / pose graph | Flag (`vo_restarts`, `whole_scene_consistent`) plus a log warning now. Pose graph in section 6 |
| 13e | The scale error follows the focal error, about 1:1 (single_room runs: focal -0.6/-3.0/-2.3% → scale +1.2/+4.2/+3.6%) | A wrong field of view makes MoGe-2 place the same pixels at a different metric depth, and DPVO's geometry changes with it | Better focal estimation; read the focal from the video's metadata | Documented; the metadata route is in section 6 (real iPhone videos carry the lens focal length in QuickTime metadata; this Stray Scanner file does not) |
| 15 | with_ceiling: the pitch check flipped a correctly oriented video upside down (final rotation 270°, ARKit says 90°) | DPVO restarted 6 times and gravity was poor (GeoCalib spread 8.2°), so the median pitch came out about +2°: its sign was noise. The prior assumes a reliable gravity estimate | Require a margin; use the point-height cue as a second vote; weight frames by gravity confidence | A margin: flip only if the median pitch > +10° (`flip_min_pitch_deg`). After the fix, with_ceiling keeps 90° (correct), but its estimated median pitch was still +6.4° against a true -18.7°: when DPVO breaks, gravity breaks too, and the margin is only a stopgap. The real fix is the same as for the trajectory (loop closure / pose graph) |
| 14 | MapAnything bf16 hook crashed in the experiment script | My shortened hook cast only tensors; MapAnything returns nested structures | Copy the smoke-test recipe | Copied the recipe (`_bf16_backbone`) |

---

## 5. Results on the captures

Commands: `scripts/run_video_frontend.py <capture>`. Outputs: `outputs/video_tier/<capture>/{scene.npz,
scene_info.json, evaluation.json, video_vs_lidar_topview.png}`. Option comparison:
`outputs/video_tier/<capture>/experiments.json`. Logs: `outputs/video_tier/log_*.txt`.

### Final runs (all with the code as it is now)

| Capture | Frames / keyframes | Focal (ARKit ≈ 1599.5 px) | VO restarts | Claimed 1-σ scale | Whole-path scale error | SE3 ATE | Sim3 ATE | Rot. error (median) | Path | C2C median TSDF / raw | 30-kf windows within ±3% / ±10% | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_room | 1714 / 164 | 1545.8 (-3.4%) | 0 | 6.5% | **+4.2%** | **12.7 cm** | 10.0 cm | 3.0° | 14.2 m | 7.6 / 8.1 cm | 3 / 4 of 5 | 252 s |
| floor_only | 5250 / 432 | 1551.9 (-3.0%) | 2 (flagged) | 5.5% | +187% | 455 cm | 276 cm | 13.8° | 53.7 m | ≥ 50 cm (capped) | 3 / 10 of 14 | 633 s |
| floor_only + DPV-SLAM loop closure | 5250 / 432 | 1549.2 (-3.2%) | 1 (flagged) | 6.0% | +107% | 342 cm | 296 cm | 25.5° | 53.7 m | ≥ 50 cm | 5 / 7 of 14 | 915 s |
| with_ceiling | 9744 / 836 | 1574.0 (-1.6%) | 9 (flagged) | 6.8% | +284% | 560 cm | 312 cm | 22.9° | 99.0 m | ≥ 50 cm | 8 / 14 of 28 | 2048 s |

How to read it:

- "Whole-path scale error" is the Sim3 factor over the *whole* path. After a VO restart it mostly measures how
  wrongly the segments are placed relative to each other, not the scale inside a room.
- The window columns show the scale inside rooms: 30 consecutive keyframes, about 2–5 m of camera path. They are
  plotted in `scale_error_windows.png` against the claimed interval.

**single_room run-to-run spread** (the same video run 4 times while the code evolved; the scale method is the same
in all 4):

| Run | Focal error | Scale error | SE3 ATE |
|---|---|---|---|
| 1 | -0.7% | +1.2% | 11.3 cm |
| 2 (jump threshold 1.5, false cuts, Issue 13b) | -3.0% | +3.8% | 63 cm |
| 3 | -2.4% | +3.6% | 12.7 cm |
| 4 (final) | -3.4% | +4.2% | 12.7 cm |

**Drift ablation (DPVO loop closure on/off, floor_only).**

- Loop closure removed one of the two restarts and improved some windows: 5 within ±3% instead of 3.
- It made the global rotation worse (25.5° vs 13.8°) and is still unusable as a whole.
- The uncorrected DPVO path (no scale correction at all) has a 406% whole-path scale error and 11.6 m SE3 ATE on
  floor_only (`experiments.json → dpvo_raw`). The depth-based correction is what makes the windows metric. Without
  it, "poses used as-is" would be the automatic fail the brief describes.

**Option comparison (single_room, `experiments.json`).** See V-3 and V-5:

- SfM registers only 35 of 164 keyframes in its largest model. That fragment has 2.6 cm Sim3 ATE, so it is accurate
  where it works.
- Raw DPVO, per 30-keyframe window: Sim3 ATE 1.3–3.9 cm, window scale 0.84–1.05 (it drifts).

**Figures** (all checked visually):

- `outputs/video_tier/single_room__c00a170fe1/video_vs_lidar_topview.png`: the video path follows the ARKit path
  closely; the right, top and bottom walls overlay the LiDAR walls.
- `outputs/video_tier/single_room__c00a170fe1/scale_error_windows.png`: all windows are inside the claimed 95%
  interval (±12.8%); 3 of 5 are inside ±3%.
- `outputs/video_tier/single_scan_floor_only__1a8384c3f6/video_vs_lidar_topview.png`: pieces of rooms are placed
  wrongly after the restarts.
- `outputs/video_tier/*/scale_error_windows.png` for the long captures: errors concentrate in the windows that
  straddle a restart.

**Gates this module touches** (Part 2):

- **Video wall lengths ±3%.**
  - Short capture: the whole-path scale is +4.2% in the final run (+1.2% to +4.2% over 4 runs). **Not met
    reliably**, but the claimed interval covers the error.
  - Long captures: not met, and flagged.
- **Calibration.** The interval covered the error on single_room in every run. On the long captures the *local*
  windows are mostly within the interval, but the whole scene is not; the `whole_scene_consistent` flag is what
  prevents confident garbage there.
- **Drift accountability.** Scale drift is measured (raw DPVO) and corrected (windows). The loop-closure on/off
  ablation is reported above. Rotation drift is not yet corrected.


---

## 6. Limitations, failure modes and next steps

**What the numbers say.**

- On a short capture the video tier recovers scale to about 1% and the path to about 1% of its length.
- On long LiDAR-style sweeps, DPVO's tracking restarts, and the joints between scale segments plus rotation drift
  put whole rooms in the wrong place. The video scenes of the whole apartment are therefore **not** yet good enough
  for stitched plans. Their claimed intervals are wide (σ_total ≈ 5–6%), so the plan will be flagged as low
  confidence rather than confidently wrong ("confident garbage caps your total score").

**Glass, mirrors, wet-look surfaces, low light** (required by the brief):

- **Glass (the shower screen, windows).** Monocular depth either sees through glass (the depth of the room behind)
  or paints the glass as a surface at the wrong distance. Feature tracking finds reflections that move wrongly.
  - Today: the 4 m range cap and the TSDF weight threshold remove part of it.
  - Next: a glass/mirror segmentation model, with masked pixels treated as "unknown" (as D-006 does for LiDAR).
- **Mirrors.** These are worse: a mirror shows a *consistent* virtual room. Depth and DPVO both believe it, which
  produces phantom rooms behind walls.
  - Next: detect planar regions whose content mirrors the scene; at minimum, reject geometry behind a fitted wall
    plane.
- **Wet-look and glossy surfaces (tiles, polished floors).** Specular highlights move with the camera. DPVO patches
  on highlights give wrong flow, and depth models flatten or dent glossy floors.
  - Mitigations: the truncated cost (a 10% cap per pixel), the Laplacian sharpness frame choice, and the floor-plane
    fit using only up-facing TSDF normals.
- **Low light.** Noise and motion blur both rise. Sharpness-based keyframe choice helps; DPVO fails earlier (more
  scale restarts) and GeoCalib's gravity spread grows.
  - Next: report the per-video median sharpness and phase-correlation response in `info`, and widen intervals or
    refuse when they fall below calibrated thresholds.
- **Clutter.** It hides wall bases. Mono depth is good on furniture, but the walls behind it are unobserved. This is
  the same rule as LiDAR: unobserved is not free space.

**Next steps, in order of expected gain:**

1. **Loop closure / global consistency.**
   - DPV-SLAM loop closure (ablation run in section 5).
   - Or a pose graph (gtsam, installed) with Sim3 edges from DPVO segments and loop edges from re-matching revisited
     places (ALIKED + LightGlue on keyframes close in the corrected path).
   - Bridge scale segments with the depth-agreement scale *across* the jump instead of trusting the DPVO step.
2. **Depth-model bias calibration** on ARKitScenes (installed, with a Faro laser scan). Measure MoGe-2's scale bias
   versus scene depth, and use the measured residual as σ_model instead of a fixed 4%.
3. **Per-frame depth-scale alignment before fusion.**
   - MoGe's per-frame scale varies by ±10% (p10–p90: 0.82–1.03).
   - Aligning each depth map to its neighbours under the metric poses would sharpen walls (C2C today: 5.9 cm median).
4. **Focal estimation (largest single error source on single_room, Issue 13e).**
   - First, read the focal length from the video's metadata. iPhone Camera-app videos store the lens and its
     35 mm-equivalent focal length; this sample file has no such tag.
   - Otherwise, combine SfM and GeoCalib by their uncertainties, and run SfM on the *most textured* window instead
     of the first 120 keyframes.
5. **A real walkthrough.** The video tier was derived from LiDAR-scanning footage (close-ups of blank walls).
   - A normal walkthrough video, held 1.5–3 m from walls, is much easier for every component.
   - The capture guidance for this tier should say so ("walk the middle of the room, phone at chest height, slow
     turns").

**Requested changes to shared code** (not modified by this module):

1. `floorplan/recon/fusion.collect_raw_points` hard-codes the 256x192 LiDAR grid (`np.mgrid[0:DEPTH_H, 0:DEPTH_W]`).
   Using `depth.shape` would let other tiers reuse it. This module has its own `_raw_points` for now.
2. `floorplan/io/stray.StrayCapture.iter_rgb` assumes decoded frame *i* is odometry row *i*. OpenCV drops the first
   packet (negative PTS), so colour is one frame (17 ms) late. This only affects TSDF colour; it should match by
   timestamp (see `video/evaluate.match_frames`).
3. The plan extractors should read `info.get("recommended_measurement_cloud", "raw_points")`. For the video tier it
   is `"points"` (the TSDF surface), because the predicted raw points are noisier than the fused surface.
4. `floorplan.plan.align.gravity_correction` is used with `max_tilt_deg=10` here. Its docstring could say that
   other tiers need a larger tolerance than ARKit's 2°.

---

# Video tier v2: long captures, focal length, phone files, integration

Sections 1–8 above describe v1 and stay as written: they are the evidence for why v2 exists. This part only adds.
Every number below comes from the runs listed in V2-§5 (logs `outputs/video_tier/log_v2_*.txt`).

**Status in one paragraph.** On the long captures, v1 chained DPVO pieces with the wrong scale and placement
(+187% / +284% whole-path scale error) while claiming a 6–7% interval. v2 cuts the walk into segments at VO restarts and
gives each its own metric scale. It then checks each segment's path against its own depth maps, with no ground truth.
Segments that fail the check are left out of the scene and reported, and the rest are joined. On floor_only the two
trusted segments (35 m of path) now have scale errors of +1.6% and -1.7%. Their claimed 1-σ is 5–7%, and the joint
trusted path has a +2.9% scale error and 36 cm SE3 ATE (v1: 455 cm for the whole path). Joining by geometry (ICP on
predicted depth) did **not** verify across any restart, so `whole_scene_consistent` stays **false**. That is the honest
answer. On single_room the fragment pose graph improves the v1 result from +4.2% to +2.2% scale error and from 12.7 to
10.2 cm ATE (same DPVO, depth and focal; on/off ablation).

On with_ceiling (12 segments), 6 of the 8 trusted segments are within ±4%, and all 4 dropped segments were really
wrong. **One wrong segment (+338%) passed the self-check** (V2-I11); the scene is flagged.

On the phone-format copies, orientation is correct for all three files. The results spread with the focal estimate
(±5% on identical pixels).

The 3% gate is still **not certified** by the video tier: the claimed 1-σ is 5–7%.

## V2-§1 How v2 works (plain language)

```
video ─► metadata (ffprobe: rotation, codec, 35 mm focal if the phone wrote it)
      ─► keyframes, upright rotation (v1)
      ─► focal = fusion of {metadata, SfM self-calibration, GeoCalib} with a 1-σ           (V2-2)
      ─► DPVO ─► metric scale per keyframe from depth agreement (v1) ─► SEGMENTS at scale jumps
      ─► a fresh DPVO run from every restart, rescaled per segment                        (V2-1)
      ─► self-check of every segment: does its path make its own depth maps agree?         (V2-5)
      ─► per-segment gravity levelling + "the camera cannot teleport" at restarts          (V2-3)
      ─► fragment pose graph on predicted depth (drift.py machinery): odometry inside segments,
         ICP bridges across restarts, ICP loops, floor/Manhattan anchors                   (V2-4)
      ─► TSDF fusion of trusted segments only ─► alignment (v1)
      ─► paper sheet (opt-in, D-067; if seen, flat, consistent) ─► per-segment scale fusion (V2-6)
      ─► scale_sigma_rel (honest), whole_scene_consistent (honest)                         (V2-7)
```

In words:

1. **Segments.** DPVO is the visual odometry. When it loses track (on blank walls), it restarts with a new, unrelated
   scale. v1 already detected these jumps. v2 treats the stretches between them as separate **segments**. Each segment
   gets its own metric scale and its own uncertainty, and is never smoothed into its neighbour.
2. **Fresh tracking per segment.** After a failure DPVO's internal state stays corrupted for a while. So v2 re-runs DPVO
   from the start of each segment, which gives it a clean initialisation.
3. **Self-check.** A segment whose camera path is right makes its keyframes' depth maps agree with each other. Its
   metric scale also drifts only slowly. A broken segment shows large depth disagreement and a scale that saturates
   the clamps. Broken segments are **dropped from the scene**, because their walls would be confident garbage, and they
   are listed in `info.scale.segments[*].reasons`.
4. **Joining.** Each segment is levelled with its own gravity estimate. The pieces are then placed end to end ("the
   camera did not teleport across the restart"). Finally the LiDAR tier's fragment pose graph refines everything,
   using predicted depth as pseudo-LiDAR. A join counts as *verified* only if ICP between the two sides passes the
   checks.
5. **Honesty flags.** `scale_sigma_rel` is the worst 1-σ over the large trusted segments. It includes the scatter,
   the depth-model bias and the focal-length uncertainty. `whole_scene_consistent` is true only if there is one
   segment, or if every segment is tied to the others by verified geometry and all segments passed the self-check.

## V2-§2 Decisions (options, evidence)

### V2-1 Long captures: split into segments and re-track each segment
- **Options.**
  1. Keep one DPVO run and chain (v1).
  2. Cut at the scale jumps, keep the first run's poses inside each segment.
  3. Cut, and run DPVO afresh from each cut (chosen).
  4. DPV-SLAM loop closure (v1 ablation: +107%, rotation worse).
  5. A different SLAM (DROID-SLAM: not installed).
- **Evidence (floor_only).** In v1, segment 0 has a +3.7% scale error and 31 cm Sim3 ATE, and segment 2 has +130% and
  139 cm. In v2, segment 0 has +1.6% and 17 cm, and segment 1 has -1.7% and 18 cm. Segment 2 is still broken with the
  fresh run (+83%, 137 cm). The video itself is the problem there: it is a LiDAR sweep past blank walls. The fresh run's
  rotations agree with the first run to a median of 1.9° and 6.1°, so the rotation hand-over is sound.
- **with_ceiling.** See V2-§4.
- **Cost.** One extra DPVO pass over the frames after the first restart: 84 + 87 s on floor_only. Results are cached
  per segment (`work/dpvo_from<a>_to<b>.npz`).
- **Why not just (2).** The fresh run is what a user would get if the walk had started there. It never makes a
  segment worse by construction: the self-check (V2-5) still judges it.

### V2-2 Focal length: metadata, then image estimates, with the uncertainty carried into the scale
- **Context (v1 Issue 13e).** The scale error follows the focal error about 1:1. On single_room the focal error was
  -0.7% to -3.4% over 4 runs, giving scale errors of +1.2% to +4.2%.
- **Options.**
  1. SfM only (v1).
  2. GeoCalib only.
  3. Container metadata only.
  4. **Robust fusion of all available sources, each with a measured σ (chosen).**
- **Metadata.** `floorplan/video/metadata.py` reads every format and stream tag with ffprobe. It takes an explicit
  35 mm-equivalent key (iPhone: `com.apple.quicktime.camera.focal_length.35mm_equivalent`) and converts it with the
  diagonal convention, f_px = f35 · diag_px / 43.27, the same rule as the photo EXIF path. It is tested on a clip
  tagged the way iOS 17+ writes it: f35 = 26, lens model `iPhone 15 back camera 6.24mm f/1.6`, make and model parsed.
  The Stray Scanner sample and all three phone_like files carry **no** focal tag. Android MP4s usually do not either,
  so the OnePlus Nord capture will use the image estimates. σ_meta = 5%: the video mode crops the sensor and
  stabilisation crops a few % more, and whether the stored value accounts for that is undocumented.
- **Fusion.** `floorplan.scale.fuse_scale` is reused on focal "cues" (log space; robust rejection with ≥ 3 sources;
  Birge inflation when the sources disagree). The fused σ is floored at 2%.
- **σ per source** (in `params`, updated after the phone runs).
  - SfM: 4%. Over 11 runs the errors were -7.1% to +4.1% (RMS 3.9%). The 4:3 originals gave -0.7% to -3.4%; the 16:9,
    30 fps copies gave -7.1%, +3.9% and +4.1%.
  - GeoCalib: 5% (-4.0, -0.8, +1.6, +2.5 and +10.9%).
  - The runs in this document were made with the first values (3% / 3.5%), which are cached in each `calib.json`.
- **Propagation.** The scale σ per segment is sqrt(σ_stat² + σ_model² + (1.0 · σ_focal)²). It was
  sqrt(σ_stat² + σ_model²) in v1.
- **Evidence** (ARKit: 1599.5 px at 1920 px):

  | Run | Fused focal | Error | Claimed 1-σ |
  |---|---|---|---|
  | Integration run, single_room (fresh SfM) | 1543.8 px | -3.5% | 2.3% |
  | phone h264 (rotation metadata) | 1549.3 px | -3.1% | 4.9% |
  | phone hevc | 1647.1 px | +3.0% | 2.3% |
  | phone baked | 1710.1 px | **+6.9%** | 3.1% (2.2σ, outside the 95% interval) |

  - The three phone files hold **the same pixels**. Their focal estimates spread by ±5%, because SfM is not repeatable
    and different keyframes get picked (V2-I12).
  - The image estimates alone **cannot** reach 1%. Real iPhone metadata is the route to that (V2-§7).

### V2-3 Per-segment gravity levelling and a continuity prior at restarts
- **Why.** A DPVO restart can also turn DPVO's world. One global "up" then tilts every later room, and the 4-DoF pose
  graph (yaw and translation only, as in drift.py) cannot fix tilt.
- **What.**
  - Each segment gets the robust mean of **its own** GeoCalib up-vectors (≥ 5 frames) and is rotated about its first
    camera to the common up.
  - Across a restart, a camera step faster than 1.5 m/s is replaced by a zero step.
- **Evidence (floor_only).**
  - Segment tilts relative to the global up were 6.0°, 2.7° and 31.6°. The last one is the broken segment, and the
    self-check drops it anyway.
  - The restart steps were 0.92 m in 0.08 s and **64 m in 0.05 s**. Both were reset.
- **Evidence (with_ceiling).**
  - Segments 0–3 were tilted 83–168° relative to the global up, while GeoCalib's spread inside each segment was only
    3–6°. DPVO's world rotation broke there, and per-segment levelling is what can repair it.
  - **Risk:** if one segment's GeoCalib up is wrong, levelling turns that segment wrongly. A cap on plausible tilt
    would need the same ground-truth check, which was not done in the time available.

### V2-4 Joining and loop closure: drift.py's fragment pose graph on predicted depth
- **Reused from `floorplan/recon/drift.py`:**
  - `Fragment`, `Edge`, `split_fragments`, `register` (coarse-to-fine point-to-plane ICP, Tukey), `verify_loop`,
    `loop_edges`;
  - the plane anchors (`measure_plane_anchors`, `anchor_sets`);
  - the robust 4-DoF solver (`build_graph`, `solve_4dof`, IRLS with a Cauchy kernel), `edge_residual`,
    `anchor_spread`.
- **New in `floorplan/video/posegraph.py`:**
  - fragments are cut inside segments, never across them;
  - ICP *bridges* across each restart (last 2 × first 2 fragments), with a weak continuity edge (0.5 m, 20°) that
    keeps the graph connected when no bridge verifies;
  - per-segment correction blending;
  - a segment connectivity test.
- **Thresholds for learned depth** (`video_graph_params`). The LiDAR settings (8→4→2 cm voxels, 1 cm RMSE) assume a
  sensor; MoGe-2 has 2.8% AbsRel plus a per-frame scale scatter of p10–p90 0.82–1.03. v2 uses:
  - ICP voxels 12→6→4 cm, correspondence gates 40→15→8 cm;
  - acceptance: RMSE ≤ 3 cm, fitness ≥ 0.30 at 6 cm, jump ≤ 50 cm / 10° (bridges 1 m / 25°);
  - odometry σ 4 cm/√m and 1°/√m;
  - floor anchor σ 3 cm, Manhattan σ 2°;
  - yaw smoothing off (DPVO heading error is not a gyro-bias ramp).
- **Evidence.**
  - **single_room on/off ablation** (same DPVO, depth and focal; only the graph differs):

    | | Scale error | SE3 ATE | Sim3 ATE | Rotation error | C2C median (TSDF) |
    |---|---|---|---|---|---|
    | graph off (= v1) | +4.21% | 12.7 cm | 10.0 cm | 2.99° | 7.6 cm |
    | graph on | **+2.22%** | **10.2 cm** | 9.4 cm | 2.62° | **6.7 cm** |

    2 of 6 loop candidates were accepted. The surface separation of the accepted loops fell from 7.0 to 4.6 cm.
  - **The bridges did not verify on floor_only (0/8 in the first run, 0/4 in the final run).** Root cause (measured
    with `dbg_graph`): even between *consecutive* fragments of one segment, where DPVO is good to a few cm, ICP on
    predicted-depth fragments moves a median of **21 cm / 3.7°** from DPVO. The median fitness is only 0.31. ICP on
    these clouds is therefore not a trustworthy measurement. Right at a restart the fragments are also tiny (3,449
    points against a typical 30,000–60,000), because the restart *happens* on a blank wall.
  - The loops in the long captures were mostly rejected (jump 5, fitness 7, RMSE 2, degenerate 2). Accepting them
    with looser gates would add wrong edges, not information.
- **Decision.**
  - Keep the graph: it helps where loops verify, and the robust kernel protects it.
  - Report the joins as **unverified**.
  - Do **not** loosen the gates to make the flag turn green.

### V2-5 Segment self-check (no ground truth)
- **Signals.**
  1. Local-scale spread inside the segment (p90/p10 of the per-keyframe metric scale).
  2. The median truncated |log depth ratio| between keyframes 1–2 apart under the metric poses.
- **Thresholds.** Spread ≤ 2.0 and residual ≤ 0.058, with at least 20 keyframes.
  - The spread threshold started at 1.6. It was raised once the stride-1 h264 phone run's good segment (+1.5% error)
    measured 1.67, while broken segments sit at 2.18–2.25, where the clamps are hit.
  - Since D-075 there is no clamp. In the fix-loop runs, segments failed on spreads of 2.1–107 while their residuals
    passed (`FIX_LOOP.md`, after the fix). The limit has not been re-tuned for the PnP scale.
  - D-076: with `scale_method = "pnp"` the spread is reported, not judged. The segment needs votes of its own on at
    least half of its keyframes (vote coverage ≥ 0.5). "depth_agreement", the default again, keeps the spread test.
- **Measured.**
  - Segments with a small error: spread 1.00–1.87, residual 0.036–0.055 (single_room, floor_only, with_ceiling,
    phones).
  - Segments that were really broken: residual 0.063–0.085 (floor_only 2; with_ceiling 0, 4, 10, 11; phone h264
    stride 2, segment 2; phone hevc stride 2, segment 1). All were caught.
  - **One miss**: with_ceiling segment 3 (+338%) had spread 1.24 and residual 0.050 (V2-I11).
- **Honest caveat.** The residual is the signal that separates; the spread mainly catches saturated clamps. Both
  thresholds were tuned on these same captures, with no held-out set.
- **Action.** Untrusted segments' depth is left out of the fusion: unobserved, not free, as with LiDAR. If **no**
  segment passes, everything is kept and the claimed σ gets a floor of 25%, so the scene says "uncertifiable" instead
  of guessing.

### V2-6 Paper sheet (`floorplan/scale`): wired in, gated, and honestly weak in the video tier
- **Status (D-067).** Opt-in, off by default (`VideoParams.use_sheet = False`; `scripts/run_video_frontend.py
  --sheet`). No capture asks for a sheet. The text below describes the opt-in cue.
- **What.**
  - The 40 most downward-looking keyframes per segment are searched (`find_sheet`, CPU, 6 processes).
  - The detector gets the **gravity normal in that camera's frame**, so only quads lying flat on a horizontal plane
    pass.
  - A sheet's camera height h_sheet is compared with the camera height above the reconstructed floor, h_recon.
  - A cue that disagrees with its segment's learned-depth scale by more than 3σ is rejected as "not on the floor".
  - The surviving cues (max 6 per segment, grouped, so many sightings of one sheet do not count as independent) are
    fused with the learned-depth scale by `fuse_scale`.
  - Segments without a sheet borrow the fusion of all sheets plus 3% (the depth model's room-to-room bias varies:
    -4.8% vs -7.1%).
  - If a correction is applied, depth and path are rescaled and the pose graph, fusion and alignment run again.
- **Evidence 1: false positives.** The first single_room run "found" a sheet in 2 keyframes and would have rescaled
  the scene by **0.26x**: confident garbage. The flatness gate removed both. After the fix: 0 detections on all
  three samples, which have no sheet.
- **Evidence 2: the reconstruction side is the weak link.** There is no real sheet in the samples, so the
  "sheet oracle" in `scripts/run_video_frontend.py` replaces h_sheet by the true height (ARKit + LiDAR floor). The
  ratio true/video camera height was:

  | Run | True/video height | Single-keyframe scatter |
  |---|---|---|
  | single_room, graph on | 1.042 | 4.4% |
  | single_room, graph off | 0.938 | 7.5% |
  | floor_only segments | 0.953 and 0.972 | 4.6–4.8% |

  Even a *perfect* sheet would therefore move single_room from +2.2% to +6.5%, and floor_only's segments from +1.6%
  to -3.2% and from -1.7% to -4.4%. The camera height above the *video* floor is biased by ±4–6%: MoGe-2's floor depth
  at grazing angles, and vertical drift that the pose graph's floor anchors change.
- **Decision.**
  - Keep the cue, but with its measured reconstruction-side σ of 5% (`sheet_floor_rel_sigma`). It can then only
    tighten an interval slightly and never dominate it.
  - Say plainly that in the video tier **the sheet is not yet a 3% fix**.
  - The better route is in V2-§7.

### V2-7 What "honest" means for the two output keys
- **`info["scale_sigma_rel"]`** (also `info["scale"]["sigma_rel"]` and `sigma_rel_total`). It is the worst 1-σ over
  trusted segments that hold ≥ 10% of the keyframes. Per segment it is sqrt(stat² + 4%² + focal²), fused with the
  sheet cues when present (opt-in, off by default since D-067). If no segment is trusted, it is floored at 25%.
  `scripts/run_capture.py` reads this key and adds it to every wall interval (verified, V2-§4).
- **`info["whole_scene_consistent"]`** (also `info["scale"]["whole_scene_consistent"]`). It is true only if all
  segments passed the self-check **and** (there is one segment **or** the segments are connected by accepted ICP
  edges that the robust solve kept with weight ≥ 0.5).
- **Per segment:** `scale.segments[*]` holds `trusted`, `reasons`, `sigma_rel`, `scale_source` and the cues. The scene
  carries `kf_segment` and `kf_trusted`.

## V2-§3 Issues log (v2)

| # | Symptom | Root cause (evidence) | Possible fixes | Chosen and why |
|---|---|---|---|---|
| V2-I1 | floor_only segment 2: +130% scale and 139 cm Sim3 ATE, with a claimed 1-σ of 5.7% | DPVO's path is wrong in **shape**, not only in scale. A fresh run is also wrong (+83%, 137 cm): the footage is blank walls at 0.3–1 m | Re-track; reject; a stronger SLAM | Re-track (V2-1) plus the self-check that **drops** the segment (V2-5). It is reported, never fused |
| V2-I2 | 0 of 8 ICP bridges verified | ICP on predicted-depth fragments is unreliable (21 cm / 3.7° median jump even where DPVO is right; fitness 0.31). Restarts happen on blank walls (3,449-point fragment) | Looser gates; global registration (FPFH); 2D wall-map matching; visual place recognition | None yet: the flag stays false. Loosening the gates would only add wrong edges (V2-4) |
| V2-I3 | First fresh-run attempt crashed: empty fragment → `estimate_floor` on zero points | A segment's depth was all removed (dropped or blank) | Guard | Fragments with < 500 points are skipped; anchors are measured per fragment, inside try |
| V2-I4 | The sheet detector "found" 2 sheets on single_room → scale x0.26 | White rectangles that are not on the floor | Flatness gate; consistency gate; ignore sheets | Both gates (V2-6). 0 false detections afterwards |
| V2-I5 | The integration run and the phone runs failed inside DPVO (missing `dpvo.npz`; `slam is None`) | Relative paths: DPVO runs with its repo as the working directory. **This also affected v1 whenever `work_dir` or the video was a relative path** (run_capture passes a relative out dir) | Absolute paths | `resolve()` both the work dir and the video path |
| V2-I6 | with_ceiling fresh DPVO run OOM | The integration run's SfM (LightGlue, 6.45 GB) shared the 8 GB GPU | Retry; serialise | Retry up to 3 times with a 30 s wait, then keep the first run for that segment (logged as `failed`). Runs were serialised |
| V2-I7 | Quality dict overwrote `segments[*].keyframes` with a count (crash) | Key clash | Rename | `n_keyframes` |
| V2-I8 | Camera height above the video floor is biased by ±4–6% (sheet oracle) | MoGe-2 floor depth at grazing angles plus vertical drift | Measure the sheet in the depth map over many sightings; calibrate on a real sheet capture | σ = 5% for now (V2-6); next steps in V2-§7 |
| V2-I11 | with_ceiling segment 3: +338% scale error but trusted (spread 1.24, residual 0.050) | Probably near-pure rotation in a stretch where DPVO invents translation: depth agreement is then weakly sensitive to scale, so neither signal fires (not verified in the time available) | Require a minimum informative-pair fraction or baseline; cross-check a fresh run against the first run per segment (Sim3 shape agreement); rotation-only detection from the phase-correlation and rotation ratio | **Open.** Reported; the scene-level flag is false |
| V2-I12 | The baked-upright copy gives 289 keyframes; the rotated copies of the same pixels give 193–198 | The keyframe trigger is "image shift > 15% of the *stored* width", so a portrait-stored frame (1080 wide) triggers more often than a landscape-stored one (1920 wide). With different keyframes, SfM gives a different focal (1549 / 1647 / 1710 px) | Normalise the shift by the upright width or the diagonal | **Open** (changing it invalidates every cache). One-line fix in `frames.select_keyframes`; proposed commit 32 |
| V2-I13 | The baked run reports a 72° rotation error | Evaluation artefact: ARKit's camera axes belong to the *stored* frames of `rgb.mp4` (sideways). For the baked file the stored frames are already upright, so the reference needs one more 90° turn. The final rotation (0°) and the median pitch (-30°) are correct | Pass the reference rotation | Noted; the scale and position metrics (Umeyama on centres) are unaffected |
| V2-I10 | The 30 fps phone copies of single_room lost track 1–2 times; the 60 fps original lost it 0 times | DPVO stride 2 was fixed for the 60 fps sample, so a 30 fps file gave DPVO only 15 fps: half the frame-to-frame overlap (the 16:9 crop also narrows the view) | Fixed stride 1; stride from the frame rate | `dpvo_stride = 0` means automatic: stride = round(fps / 30). Same stride 2 on the samples. Evidence in V2-§5 (stride 2 vs 1 on the same files) |
| V2-I9 | Video plan area 23.3 m² vs LiDAR plan 17.4 m² (+34%) on single_room, while the scale error is only +2.8% | Not scale: the plan extractor on predicted-depth surfaces (1 room instead of 2, longer walls). The wall interval adds only the tier's *scale* term, so geometric noise of the video surface is not covered: **the area interval [18.3, 28.4] misses 17.4** | Video-specific geometry noise term in the extractor; fit on the TSDF points (`recommended_measurement_cloud`) | Reported to the plan-extractor owner (change request below). It is outside this module |

## V2-§4 Results

Commands:

- `python scripts/run_video_frontend.py <capture> --tag __v2`
- `--no-graph` for the ablation
- `--eval-capture` for re-encoded copies

Outputs are in `outputs/video_tier/<capture>__v2/` and `outputs/video_tier/phone_like/`. The long-capture v2 runs reuse
the v1 runs' DPVO, depth and focal caches, so the before/after difference is the v2 logic only. single_room was also
run cold through `run_capture`.

**Whole captures, before (v1) and after (v2):**

| Capture | v1 whole-path scale error / SE3 ATE | v2 segments (trusted) | v2 trusted-path scale error / SE3 ATE / path | v2 claimed 1-σ | whole_scene_consistent |
|---|---|---|---|---|---|
| single_room | +4.21% / 12.7 cm | 1 (1) | **+2.22% / 10.2 cm** / 14.2 m | 7.2% | true |
| single_room, cold `run_capture` | – | 1 (1) | +2.84% / 11.2 cm / 14.2 m (C2C 6.4 cm) | 5.5% | true |
| floor_only | +187% / 455 cm | 3 (2) | **+2.93% / 35.9 cm** / 35.0 m | 6.8% | **false** (no verified bridge) |
| with_ceiling | +284% / 560 cm | 12 (8) | +380% / 594 cm / 92.6 m (one trusted segment is wrong, see below) | 11.3% | **false** (0/28 bridges) |

**Per segment** (Sim3 per segment against ARKit; "covered" = the error lies inside the claimed 95% interval):

| Capture / segment | Keyframes | Path | v1 scale error | v2 scale error | v2 Sim3 ATE | v2 claimed 1-σ | Covered | Trusted |
|---|---|---|---|---|---|---|---|---|
| single_room / 0 | 0–163 | 14.2 m | +4.21% | +2.22% | 9.4 cm | 7.2% | yes | yes |
| floor_only / 0 | 0–144 | 19.6 m | +3.72% (0–159) | +1.55% | 17.1 cm | 6.8% | yes | yes |
| floor_only / 1 | 145–300 | 15.4 m | -0.08% (160–300) | -1.68% | 17.7 cm | 5.1% | yes | yes |
| floor_only / 2 | 301–431 | 18.6 m | +129.7% | +82.9% | 136.5 cm | 8.4% | **no** | **no → dropped** |
| with_ceiling / 0 | 0–18 | 0.8 m | +181.7% | +181.7% | 1.5 cm | 5.0% | **no** | **no → dropped** (residual 0.070, 19 kf) |
| with_ceiling / 1 | 19–47 | 2.9 m | -12.4% | -3.39% | 7.6 cm | 12.5% | yes | yes |
| with_ceiling / 2 | 48–102 | 6.9 m | +37.0% | **-0.61%** | 5.4 cm | 9.6% | yes | yes |
| with_ceiling / 3 | 103–155 | 7.8 m | +284.5% | **+337.9%** | 137.0 cm | 8.5% | **no** | **yes: self-check MISS** |
| with_ceiling / 4 | 156–159 | – | – | (4 keyframes) | – | – | – | **no → dropped** (residual 0.084) |
| with_ceiling / 5 | 160–311 | 19.4 m | +4.7% (156–311) | +3.96% | 36.4 cm | 5.5% | yes | yes |
| with_ceiling / 6 | 312–488 | 14.5 m | -5.4% | -3.55% | 22.4 cm | 5.7% | yes | yes |
| with_ceiling / 7 | 489–595 | 16.5 m | +3.9% | +2.30% | 10.2 cm | 6.6% | yes | yes |
| with_ceiling / 8 | 596–722 | 17.2 m | -13.5% | -11.76% | 27.3 cm | 11.3% | yes (1.0σ) | yes |
| with_ceiling / 9 | 723–764 | 7.0 m | -2.8% (723–788) | -2.16% | 2.9 cm | 5.3% | yes | yes |
| with_ceiling / 10 | 765–788 | 0.7 m | – | +465.4% | 2.2 cm | 7.2% | **no** | **no → dropped** (residual 0.080) |
| with_ceiling / 11 | 789–835 | 4.7 m | +156.6% | +404.8% | 65.2 cm | 17.1% | **no** | **no → dropped** (residual 0.085) |

**with_ceiling summary.**
- v1: 3 of 10 segments within ±5%, and 4 segments off by +37% to +285% were all fused into the scene.
- v2: 6 of the 8 trusted segments are within ±4% (92.6 m of trusted path), and all 4 untrusted segments were really
  wrong (+182% to +465%) and were left out.
- **One miss.** Segment 3 (+338%, 7.8 m) passed the self-check (spread 1.24, residual 0.050), so its geometry is in the
  scene. The whole scene is flagged `whole_scene_consistent = false` (0/28 bridges, 2/20 loops), but the segment's own
  interval (8.5%) does not cover its error. This is the remaining confident-garbage risk (V2-I11).
- **Gravity.** Per-segment levelling found segments 0–3 tilted 83–168° relative to the global up. DPVO's world rotation
  broke there, while GeoCalib's up stayed consistent inside each segment (spread 3–6°). The fused scene still had no
  detectable floor ("[align] WARNING: no floor found"), so heights are unavailable for this capture (v1 Issue 15
  persists).

The v1 segment boundaries differ slightly from v2's, because v2 forces cuts at the fresh-run starts.

**Phone formats (V2-§5)** are below. **Integration** (`run_capture.py --tier video --extractor beta --no-damage` on
single_room, cold, 559 s):

- The front-end runs end to end. The fresh SfM focal gives 1543.8 px (-3.5%), 1-σ 2.3%.
- `scale_sigma_rel` = 5.51% is read by `run_capture` and appears on every wall as "+ tier scale term 5.51% (x1)", and
  on the area as "(x2)".
- The plan has 1 room, 20 walls and 2 openings, against 2 rooms and 18 walls for the LiDAR plan. The area is 23.3 m²
  against 17.4 m²: see V2-I9. That is the extractor's issue on video surfaces, not scale (+2.84%).

## V2-§5 Phone formats (`outputs/phone_like/video`, re-encodes of single_room, scored against its ARKit)

These files are 16:9, 1080p, 30 fps crops of single_room (made by the docs agent; see `outputs/phone_like/README.md`).
"Stride" is how many recorded frames DPVO skips. The v1 default was 2, which gives DPVO only 15 fps on a 30 fps file.
The fix makes it automatic, giving stride 1 here (V2-I10).

**Orientation (the point of this test). All three come out upright and agree:**

| File | Container rotation | How the front-end decided | Final rotation | Median camera pitch |
|---|---|---|---|---|
| `phone_portrait_h264_1080p30` | display matrix -90 (OpenCV `ORIENTATION_META` = 90) | container metadata | 90° | -31.7° |
| `phone_portrait_hevc_1080p30` | display matrix -90 | container metadata | 90° | -30.0° |
| `phone_baked_upright_h264_1080p30` | none, pixels already upright | GeoCalib vote tied 0° vs 180° (5:5), so the pitch check chose 0° | 0° | -30.0° |
| original `rgb.mp4` (for reference) | none, stored sideways | GeoCalib vote | 90° | -31.3° |

ARKit's median pitch on this capture is -31°. HEVC and H.264 decode identically through OpenCV/FFmpeg, and the
display matrix is read through `CAP_PROP_ORIENTATION_META` with auto-rotation off; the same value is shown by
`metadata.video_metadata`.

**Results** (scored against single_room's ARKit by timestamp):

| File | Stride | Focal (error) | Restarts | Trusted segments | Trusted-path scale error / SE3 ATE / path | Claimed 1-σ | whole_scene_consistent |
|---|---|---|---|---|---|---|---|
| h264 (rotation metadata) | 2 | 1549 px (-3.1%) | 2 | 2 of 3 | -2.0% / 17.3 cm / 11.0 m (segment 1 of 1.8 m: -11.6%) | 9.9% | false |
| hevc (rotation metadata) | 2 | 1647 px (+3.0%) | 1 | 1 of 2 | +17.0% / 44.8 cm / 11.1 m (2.0σ) | 8.4% | false |
| h264 | **1** | 1549 px (-3.1%) | 2 | 1 of 3 | **+1.5% / 14.7 cm / 11.1 m** (the 2 dropped segments were +300% and +65,000%) | 7.4% | false |
| hevc | **1** | 1647 px (+3.0%) | **0** | 1 of 1 | **+6.0% / 29.4 cm / 14.3 m** (whole path, C2C 10.5 cm) | 7.5% | true |
| baked upright | 1 | 1710 px (+6.9%) | 4 | 4 of 5 | +28.7% / 60.6 cm / 13.7 m. Segments: -4.6%, +3.3%, -12.6% (0.9 m), +27.6% (4.2 m, **not covered**) | 8.0% | false |
| *original 4:3, 60 fps (V2-§4)* | 2 | 1545.8 px (-3.4%) | 0 | 1 of 1 | +2.2% / 10.2 cm / 14.2 m | 7.2% | true |

Reading:

- The 16:9, 30 fps copies are much harder than the 4:3, 60 fps original, which had 0 restarts and +2.2%. At stride 2
  they restart 1–2 times.
- Stride 1 helps. hevc goes to 0 restarts (whole path +6.0%, 29 cm). h264's first 11.1 m are now one segment with
  +1.5% error.
- The results do **not** match each other closely, even though the pixels are the same. The spread comes from the
  focal estimate (±5%, V2-2) and the keyframe choice (V2-I12), and it is larger than the 3% gate. This is the strongest
  argument for reading the focal length from iPhone metadata and for the protocol tweaks below.

## V2-§6 Limitations (v2)

- **The 3% gate is not certified by the video tier.**
  - Inside a trusted segment the scale errors were -1.7% to +2.9%, but the claimed 1-σ is 5–7%.
  - The two largest terms are the depth-model bias (4%, which needs calibration on external LiDAR data) and the focal
    length (2–5% without metadata).
- **Joining across restarts is not verified** by geometry on the samples, so stitched multi-room plans from a video
  with restarts are flagged. Room *dimensions* inside a trusted segment remain usable.
- **Self-check thresholds** come from 4 good and 1 broken segment; with_ceiling is the first semi-independent check.
- **The sheet route** in the video tier is limited by the reconstructed floor (±4–6%), not by the sheet (0.55%).
- **The self-check can miss.** with_ceiling segment 3 (+338%) passed (V2-I11). On the phone runs, trusted short
  segments were off by -12.6% and +27.6%: the baked run, with a +6.9% focal error.
- **Focal from images is unstable to ±5%** on identical pixels (V2-2, V2-I12). Without metadata this alone exceeds
  the 3% gate.
- **Runtime.** Long captures pay for one extra DPVO pass after the first restart, plus a second scale estimation
  (floor_only 451 s with caches).
- **Repeatability.** SfM focal and DPVO are still not bit-deterministic (v1 Issue 13c). The integration run's focal
  differed from the cached run's (1543.8 vs 1545.8 px).

## V2-§7 Next steps (in order of expected gain for the walk-in test)

1. **Real iPhone metadata focal.** Read it (done). Check it on one founders' iPhone clip against SfM. If it agrees
   within 1%, lower σ_meta and the focal term drops out.
2. **Sheet measured in the depth map, not via the floor.** Dropped by D-067: no capture has a sheet, and the cue is
   opt-in.
   - `depth_map_scale`: sheet-plane depth vs MoGe depth at the sheet pixels, averaged over every sighting in the
     segment. This removes the floor bias of V2-I8.
   - It was to be calibrated on my own capture, which now has no sheet.
3. **Bridges that work on blank walls.** 2D wall-map matching of the two segments (gravity known, so 3 DoF), or
   matching keyframes 2–5 s before and after the restart with LightGlue. Accept only with a degeneracy test.
4. **Depth-bias calibration** on ARKitScenes (v1 next step 2), to shrink the 4% model term.

**Change request to the plan extractors** (V2-I9): for `tier == "video"`, add a geometry-noise term to wall offsets,
for example from the scene's C2C of predicted surfaces (about 6–8 cm median), and use
`info["recommended_measurement_cloud"]`.

## V2-§8 Video protocol tweaks (for `docs/CAPTURE_PROTOCOL.md`), with the evidence behind each

1. **Record at 60 fps if the phone offers it at 1080p/4K; otherwise 30 fps is fine now that the stride is automatic.**
   - Evidence: the same single_room footage as a 30 fps phone file, tracked at DPVO stride 2 (15 fps effective), lost
     track 1–2 times. The 60 fps original at stride 2 (30 fps effective) lost track 0 times.
   - Fix in code: `dpvo_stride = 0`, so DPVO now sees ~30 fps whatever the recording rate (V2-I10).
2. **Landscape, main lens at 1x, and turn off automatic lens switching and "super/ultra steady" or "action"
   stabilisation.**
   - iPhone Pro: Settings → Camera → Macro Control off.
   - OnePlus: Ultra Steady off.
   - Why: the pipeline assumes one focal length for the whole clip and estimates it to 2–5% (V2-2). A lens switch or
     a stabilisation crop changes it mid-clip.
   - Why landscape: in portrait, a 16:9 phone video is 37° wide; in landscape it is 62°. The narrower view is what
     lost track in the phone_like files.
3. **Stay at least 1.5 m from walls. Never fill the frame with a blank wall; keep a corner or the floor line in
   view.**
   - Evidence: every DPVO restart and every failed bridge was on blank walls at 0.3–1 m (v1 Issue 5; V2-I2: the
     fragment at the floor_only restart had 3,449 points against a typical 30,000–60,000).
4. **Turn at most about 20°/s (a full turn in ≥ 20 s).**
   - Evidence: the sample captures turned at a median of 23–25°/s, with p90 66–84°/s (ARKit). The fast p90 bursts
     are where the phase-correlation response dropped.
5. **Pause 2 s in every doorway, facing into the next room with both door jambs and its floor in view, then walk
   through facing forward.**
   - Why: transitions are where tracking breaks and where the pose graph needs one fragment that sees both rooms to
     join segments (V2-4: 0 verified joins on footage without such views).
6. **Withdrawn (D-067): no sheet.** This tweak asked to keep the A4 sheet in view for about 2 s from two spots. No
   capture uses a reference object any more, and the sheet cue is off by default.
7. **Lights on; do not pan across bright windows slowly.** Low light and exposure pumping cause blur and
   restarts (v1 §6).
8. **End where you started and re-film the first view for about 3 s**, so the end of the clip can close a loop with
   its start.
9. **Keep the phone slightly pitched down (the floor line in the lower third).**
   - Why: the orientation check relies on a median pitch of -20° to -30°. With_ceiling's median pitch of +6° is where
     gravity and the flip check became unreliable (v1 Issue 15).
