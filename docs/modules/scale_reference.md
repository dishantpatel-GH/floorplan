# Module: scale_reference (`floorplan/scale/`)

Files: `floorplan/scale/{__init__.py, sheet.py, fuse.py}`, `scripts/test_sheet_scale.py`.
Evidence: `outputs/scale_reference/{results.json, cases.csv, accuracy.png, examples.jpg}`.
Decision context: D-013 (a plain A4/Letter sheet on the floor is the scale reference).

**Status (D-067, 4 Oct):** optional and OFF by default (`PhotoParams.use_sheet`, `VideoParams.use_sheet`). No capture
instruction asks for a sheet any more. The numbers below describe the opt-in detector on rendered sheets. The cue
fusion in `fuse.py` is still used by the video tier's focal-length fusion.

## 1. Purpose and where it sits

- Photos and plain video recover geometry only **up to an unknown scale factor**. Every wall length inherits any
  scale error 1:1, and the gates are 8% (photos) and 3% (video).
- Learned metric depth (MoGe-2, DA3, MapAnything) gives scale to "a few percent": not enough on its own for 3%.
- Until D-067 the capture protocol put one plain A4 or US-Letter sheet flat on the floor of each room. Its size is
  standardised (printer scaling does not apply: nothing is printed). No capture uses one now.
- This module:
  1. `sheet.py` finds the sheet in an image and measures **the camera's height above the floor in metres**, with an
     interval (Monte Carlo over corner noise, focal length, principal point and paper type).
  2. `fuse.py` turns that (and any other cue) into the **scale factor** of a reconstruction:
     `scale = metric height / height in reconstruction units`, and fuses all cues (sheet, learned depth, priors) with
     robust inverse-variance weighting. The interval says which cues were present.
- Callers: the photo tier and the video tier (they own the reconstruction; this module never touches it).

```python
from floorplan.scale import load_image_with_intrinsics, find_sheet, sheet_cue, camera_height_prior, fuse_scale
img, intr = load_image_with_intrinsics("photos/kitchen/IMG_0001.HEIC")   # K from EXIF 35 mm focal (or a guess)
meas = find_sheet(img, intr, floor_normal_cam=None)    # SheetMeasurement or None (no trustworthy sheet)
cues = [sheet_cue(meas, h_recon=cam_height_in_recon_units)] if meas else []
cues.append(camera_height_prior(median_cam_height_in_recon_units))
scale, sigma, report = fuse_scale(cues)                 # metric = scale * recon units; report lists cues + weights
```

Other entry points: `detect_sheets` (all detections + rejected candidates with reasons), `measure_sheet`,
`depth_map_scale` (scale a learned depth map from the sheet's pixels), `plane_depth`, `learned_depth_cue`,
`door_height_prior`, `ceiling_height_prior`.

## 2. How it works, step by step

### 2.1 Detection (`detect_sheets`)
1. **Candidates (several generators).** Work on a copy downscaled to <= 1600 px.
   - Global brightness thresholds every 2.5 L* units between the 40th and 99.7th percentile (like MSER).
   - Local (adaptive) thresholds at three window sizes: "brighter than its own surroundings".
   - Regions enclosed by Canny edges (two sensitivities, with and without gap closing).
   - Each region -> convex hull -> 4-vertex polygon, kept only if the region fills its hull (solidity >= 0.88).
   - Why several: white paper on a white floor differs by only ~3-8 L* units. Measured: on rendered tests, single
     generators each miss cases the others find (e.g. a sheet touching a grout line has an open Canny contour but
     a clean threshold blob, and vice versa).
   - Near-identical quads are clustered; up to 3 variants per cluster are tried in turn.
2. **Sub-pixel edges (full resolution).** For each side, 12-80 intensity profiles across the edge; the steepest
   bright(inside)->dark(outside) step is located with a parabola; a robust (Tukey) straight line is fitted. Corners =
   intersections of adjacent lines.
   - Why lines and not corner detectors: corners are the most blurred/damaged part; a side has 50+ edge samples.
   - The line residual is also the **straightness** measure: a curled sheet has curved edges.
3. **Verification (hard gates, all must pass).**
   - Brighter than a strip outside **every** side (min contrast 1.5 L*), uniform low-texture interior (no text,
     wood grain), near-neutral chroma relative to the scene's white.
   - Straight edges (rms residual <= 0.4% of the side and <= 1.2 px), >= 75% of each side has a consistent edge
     (occlusion check).
   - Edges **end at the corners**: tiles, cabinet doors and window frames continue as a grid (<= 3 of 8
     side-extensions may carry an edge).
   - With K: the back-projected quad must be a right-angled rectangle (<= 4 deg) with the aspect ratio of A4
     (1.414) or Letter (1.294) within 4.5%, for some focal inside +-2.5 sigma of the prior. This is the strongest
     false-positive filter: a random bright quadrilateral is almost never a metric A4/Letter rectangle.
   - Plausible camera-to-plane distance 0.15-8 m; optional: parallel to the floor if the caller knows the floor normal.

### 2.2 Measurement (`measure_sheet`)
1. **Known-rectangle pose fit (IPPE + LM).** For each paper hypothesis (A4, Letter), the rectangle of that exact
   size is fitted to the 4 corners: maximum-likelihood pose under known K. Camera-to-floor distance = distance of
   the camera centre to the sheet plane.
2. **Monte Carlo (600 draws per hypothesis).**
   - Corners: each edge line is perturbed by its fit uncertainty plus a calibrated **edge bias** (0.1 px per side end).
   - Focal: log-normal with the caller's relative sigma (EXIF ~4%, no EXIF 10%); principal point: optional sigma.
   - Each focal draw is weighted by how well the **observed** corners fit the rectangle under that focal (Gaussian
     likelihood of the reprojection residual, width x1.5, calibrated). Under strong perspective the sheet itself tells a wrong
     focal apart (self-calibration); for a nearly frontal sheet the weights are flat and the prior is kept.
   - The same likelihood gives P(A4) vs P(Letter). The output distribution is the **mixture** of both hypotheses, so
     an ambiguous type widens the interval instead of guessing.
3. Output `SheetMeasurement`: height, sigma, split into **random** (corner noise; independent between photos) and
   **systematic** (intrinsics; shared by all photos of one camera), 95% interval, plane normal/centre in camera
   coordinates, paper posterior, focal posterior.

### 2.3 Fusion (`fuse_scale`)
- Log-space (scale errors are multiplicative).
- Cues of the same `group` share `sigma_sys`: their random parts average down, the shared part does not
  (several photos from one phone share its focal error; treating them as independent would claim sqrt(N) too much).
- Robust: leave-one-out z-test of every cue against the fusion of the others; the worst with |z| > 3 is removed,
  repeatedly, while >= 3 cues remain (with 2 cues there is no majority, so we inflate instead).
- Birge ratio: if surviving cues disagree more than their sigmas allow, the interval is inflated by sqrt(chi2/dof).
- Priors (wide on purpose, they arbitrate and never dominate): camera height 1.40 +- 0.13 m (chest-height shots),
  door height 2.04 +- 0.05 m, ceiling 2.65 +- 0.25 m.
- With priors only, `report["status"] = "priors_only"` and the interval is ~+-18% (95%): honest, not a guess dressed up.

## 3. Decisions

| # | Question | Options | Choice | Evidence |
|---|---|---|---|---|
| S-1 | Pose from 4 corners | (a) unconstrained parallelogram back-projection (closed form, aspect-free); (b) known-size rectangle fit (IPPE+LM) per paper type | **(b)**, (a) kept for gating/aspect | Same detected corners on a foreshortened view: (a) +1.4% height error, (b) -0.13%. With 0.3 px corner noise: (a) 2.1% sd, (b) 0.37% sd (one-off experiment on rendered case single_room frame 835; script not kept in the repo). Enforcing right angles + aspect removes the weakly-constrained direction. |
| S-2 | A4 vs Letter unknown | (a) ask the user; (b) classify, pick one; (c) keep both hypotheses as a mixture | **(c)** | A4 vs Letter differs by 6% in length: a wrong pick is a 2-4% scale error. The mixture costs nothing when the type is clear (posterior ~1) and widens honestly when not. Side note: the size measure L^0.31 W^0.69 is identical (234.0 mm) for both papers; it is used in the aspect-free gate. |
| S-3 | Corner uncertainty | (a) line-fit residuals only; (b) + calibrated edge-bias term (0.1 px per side end) | **(b)** | (a) predicts ~0.02 px corner sd; measured corner errors on rendered sheets are ~0.2-0.3 px (10x). Calibration table in section 5: 0.05 px is over-confident (coverage 77%), 0.1 px gives 98%. |
| S-4 | Focal uncertainty | (a) ignore (trust EXIF); (b) prior only; (c) prior x rectangle likelihood | **(c), likelihood width x1.5** | The rectangle likelihood does self-calibrate: with a 3% focal error the median height error is 0.49% (it would be ~2-3% from the prior alone; sensitivity of height to focal is 0.6-1.0). Width x1.0 is over-confident (coverage 94%, z rms 1.12), x2.0 too wide (z rms 0.67); x1.5 gives 98% / 0.78. |
| S-5 | Candidate generation | single threshold / Canny / MSER / LSD-quad grouping / several cheap generators | **several cheap generators + clustering** | Each single generator missed cases others caught on rendered tests; LSD grouping is combinatorial and slower. |
| S-6 | False-positive control | appearance only / + metric shape test | **appearance + metric A4/Letter rectangle + grid-continuation test** | Real-frame negatives, section 5. A wrong scale poisons every measurement in the room, so we prefer a miss (the pipeline falls back to other cues) over a false detection. |
| S-7 | Correlated cues in fusion | treat all independent / group-aware | **group-aware** | N photos with the same 3% focal error do not give 3%/sqrt(N). |
| S-8 | Two disagreeing cues | reject one / inflate | **inflate (Birge)** | With 2 cues there is no evidence which is wrong; rejection needs a majority (>= 3 cues). |
| S-9 | Testing without real sheet photos | wait for the capture / synthetic images / rendered sheets in real frames | **rendered sheets in real sample frames** | Real lighting, floors and confounders; exact truth from LiDAR pose + floor plane. Rendered: anti-aliased (4x supersampling), paper brightness = local floor illumination x albedo ratio (shadows and light patches carry over), fibre texture, blur, noise, JPEG re-encoding. |

## 4. Issues log

- **I-S1 Rendered sheets were half a pixel off.** Symptom: every detected corner ~0.5-0.7 px from truth, same
  direction. Root cause: the 4x supersampled renderer used `x*4` instead of the pixel-centre convention
  `x*4 + 1.5`. Fix: corrected mapping; corner errors fell to ~0.2-0.3 px. (Test bug, not a detector bug, but it would
  have mis-calibrated the edge-bias floor.)
- **I-S2 Paper rendered grey / blocky.** Symptom: sheets darker than the floor; visible checkerboard on the paper.
  Root cause: paper brightness taken from a ring that included dark furniture; per-cell illumination on the mesh.
  Fix: paper = smoothed floor illumination under the sheet x albedo ratio, applied per pixel after rasterisation.
- **I-S3 Sheets placed over furniture.** Symptom: rendered sheets straddling a fridge/wall boundary. Root cause:
  placement checked only a median LiDAR agreement. Fix: sheet centre chosen among depth pixels that LiDAR confirms
  lie on the floor plane (+-3 cm), and 80% of sheet samples must agree (90th percentile < 4 cm).
- **I-S4 Parallelogram pose was 5x noisier than necessary** (see S-1). Fix: known-rectangle IPPE fit.
- **I-S5 Focal self-calibration biased** (see S-4). Fix: tempered likelihood.
- **I-S6 Tukey line fit rejected good low-contrast edge points** (residual scale floor 0.05 px was below JPEG jitter),
  so clean low-contrast sheets were rejected as "broken edge". Fix: scale floor 0.25 px.
- **I-S7 Detection slow (up to 13 s/image)** because the photometric checks ran Sobel and full-image masks per
  candidate. Fix: one shared gradient image and crops around each quad (now ~0.4-1 s/image on a loaded CPU).
- **I-S9 Weighted-percentile intervals collapsed.** Symptom: coverage 81% while z rms was 0.66. Root cause: when the
  focal likelihood is sharp, few Monte Carlo draws carry the weight, so the corner-noise spread was estimated from a
  handful of samples. Fix: variance = (importance-weighted variance of the observed-corner fits over focal/paper) +
  (unweighted corner-noise variance); interval = exp(mu +- 1.96 sigma). Coverage 98%.
- **I-S10 A curled sheet's flat part accepted as a smaller sheet (open).** Symptom: 1 of 11 curled cases accepted
  with +9.4% error at 3.3 sigma. Root cause: the shading change at the fold forms a straight edge, so the flat part is
  a perfect rectangle with a paper-like aspect (1.235, within the gate once focal freedom is allowed); misfit and
  continuation tests do not separate it (rectangle misfit 0.08 px, inside the range of true sheets). Possible fixes:
  require the strip outside each side to be NOT paper-like (fails on white floors); tighter aspect gate (would drop
  real far sheets: true sheets reach 0.044); rely on fusion. Chosen for now: fusion (a 9% outlier among >= 3 cues
  is rejected by the z-test) + protocol "flat sheet"; flagged as open.
- **I-S8 Early pre-check discarded the right candidate** when the coarse quad was a few pixels off (strip contrast
  measured across the true edge). Fix: pre-check only rejects regions darker than their surroundings; the full
  contrast test runs after sub-pixel refinement.

## 5. Results

All numbers: `python scripts/test_sheet_scale.py --per-capture 30 --neg-per-capture 40` (seed 0), written to
`outputs/scale_reference/results.json` and `cases.csv`; figures `accuracy.png` (error vs distance with 95% intervals,
z-score histogram, detection rate) and `examples.jpg` (gallery of rendered cases, red = accepted outline).
Setup: real frames of the 3 sample captures; rendered A4 (60%) / Letter (40%) sheets on LiDAR-confirmed floor;
the detector gets a focal length perturbed by N(0, 3%) (EXIF-like error) and is told sigma = 3%. Truth = exact
camera-to-floor distance of the rendering pose. Runtime ~0.6 s/image (detection) + ~0.5 s (Monte Carlo), CPU only.

**Flat sheets (n = 73 rendered, 47 detected):**

| Metric | Value |
|---|---|
| Detection rate (correct location) | 64% (84% when paper is >= 1.35x brighter than the floor; 20% for white-on-white < 1.15x) |
| Detections at a wrong location | 0 |
| Height (= scale) error, median / p95 | **0.55% / 3.1%** (1.5-2.5 m: 0.35% / 1.4%; 2.5-3.5 m: 1.5% / 5.5%) |
| Bias (median signed error) | -0.15% |
| Reported sigma (median) | 1.2% (0.8% at 1.5-2.5 m, dominated by the 3% focal prior) |
| 95% interval coverage | **97.9%** (46/47); z rms 0.77 (slightly conservative) |
| Paper type correct / left ambiguous | 87% / 13%; **never confidently wrong** on flat sheets |

**Calibration of the two noise parameters** (same 47 detections re-measured, `edge_bias_px`, `like_temper`):

| edge bias (px) | temper | focal prior | z rms | cov95 | median sigma | median abs err |
|---|---|---|---|---|---|---|
| 0.05 | 1.0 | 3% | 1.80 | 0.77 | 0.48% | 0.42% |
| 0.10 | 1.0 | 3% | 1.12 | 0.94 | 0.82% | 0.43% |
| **0.10** | **1.5** | 3% | **0.78** | **0.98** | 1.20% | 0.49% |
| 0.10 | 2.0 | 3% | 0.67 | 1.00 | 1.41% | 0.54% |
| 0.10 | 1.0 | exact K | 0.68 | 0.98 | 0.23% | **0.11%** |

Read: with an exact focal the sheet gives the height to ~0.1% (median); the **focal length dominates** the budget.
The chosen setting errs on the conservative side (score rule: confident garbage is worse than a wide interval).
(Earlier versions, for the record: parallelogram pose + 0.2 px bias gave coverage 100% with z rms 0.58 and median
error 0.8%; a weighted-percentile interval collapsed when the focal weights were sharp: coverage 81%, fixed by I-S9.)

**Rejections (must not produce a confident wrong scale):**

| Case | n | Accepted |
|---|---|---|
| Real frames without a sheet (cabinets, door panels, white/glossy tiles, light patches, glass) | 113 | **0 false positives** |
| Rendered square 30x30 cm (tile, sticky note) | 10 | 0 |
| Rendered 2:1 rectangle (envelope, tile) | 14 | 0 |
| Rendered printed page (dense text) | 11 | 0 |
| Partially occluded sheet (object over a corner) | 14 | 0 (all rejected: conservative) |
| Curled sheet (last third lifted 3-7 cm) | 11 | 1 accepted at the wrong outline: **+9.4% error, sigma 2.8% (3.3 sigma)**, see I-S10 |

**Fusion (simulated rooms, n = 300):** 2-4 sheet photos (same camera, shared focal error) + camera-height prior, and in
half the rooms a deliberately wrong learned-depth cue (+25%, claimed 3%): median error 0.41%, p95 2.1%, median
sigma 1.6%, **coverage 99%**, wrong cue rejected in 98% of the rooms that had one.


## 6. Limitations, failure modes, next steps

- **Low light / strong shadows across the sheet**: a shadow edge crossing the sheet breaks the "straight consistent
  edges" test -> rejected (a miss, not a wrong scale). Protocol says all lights on.
- **Glossy floors / specular highlights**: a highlight on the sheet or next to it can lower side contrast; the
  rectangle test keeps highlights themselves from being accepted.
- **Mirrors and glass**: a sheet seen in a mirror is a valid rectangle but at the wrong (virtual) distance. The
  optional floor-normal gate rejects mirrored sheets on walls; a mirror on the floor is not handled.
- **Detection rate is the main weakness** (64% per rendered view; 20% for white paper on a near-white floor at
  < 1.15x contrast). Misses mostly come from candidate generation (no region with the sheet's outline) and from
  shadows or neighbouring edges breaking a side. A miss is safe (other cues take over), and the protocol asks for
  the sheet in >= 2 photos and the video sees it in many frames. Next: line-segment (LSD/DeepLSD) quad hypotheses as
  a fourth generator; a darker floor spot for the sheet in the protocol.
- **Folded/curled sheets** can pass as a smaller rectangle (I-S10).
- **Clutter**: partial occlusion of a side -> rejected; occlusion of a corner only may still measure correctly (the
  corner comes from line intersection).
- **Curled sheets**: rejected through curved edges or non-right angles when the curl is visible; a gently bowed sheet
  whose edges still look straight biases the height by roughly the lift (a few mm, < 0.5%).
- **Printed pages** are rejected (text = texture). The D-013 protocol said plain sheet. A C5 envelope (229x162 mm) has the A4
  aspect ratio and would be accepted with a 23% wrong scale; fusion with priors would flag it only if it disagrees
  > 3 sigma with the other cues.
- **Lens distortion**: phone photos are distortion-corrected by the ISP; raw video frames may not be. Residual
  distortion near image borders biases corners; the edge-bias floor absorbs part of it. Next: undistort if the video
  tier estimates distortion.
- **Focal length dominates the error budget** for EXIF-quality focal (~3-4%): see results. Next: share one focal
  posterior across all photos of a room (the sheet likelihood multiplies), and accept the photo/video tier's
  self-calibrated focal (SfM/GeoCalib) with its own sigma.
- **Scale = height ratio** assumes the reconstruction's floor plane and camera centre are right; their errors enter
  through `h_recon_rel_sigma` in `sheet_cue` (caller supplies).
- Real-photo validation is still missing: run `find_sheet` on `TakeHome/OwnCaptures/photos/*` with the tape ground
  truth tomorrow (same API; nothing capture-specific is tuned).

## 7. Proposed commits

1. `scale: paper-sheet detector (candidates, sub-pixel edges, verification gates)` - `floorplan/scale/sheet.py`
   (detection half), `floorplan/scale/__init__.py` - the free scale reference of D-013 needs a detector first.
2. `scale: known-rectangle metric measurement with Monte Carlo intervals` - `floorplan/scale/sheet.py`
   (`fit_rectangle`, `measure_sheet`, `depth_map_scale`) - turns corners into metres with honest uncertainty.
3. `scale: robust, group-aware fusion of scale cues and priors` - `floorplan/scale/fuse.py` - one scale with an
   interval from whatever cues exist; correlated cues not double-counted.
4. `test: blind rendered-sheet benchmark on real sample frames` - `scripts/test_sheet_scale.py` - detection rate,
   error, calibration, false positives with exact LiDAR truth.
5. `docs: scale_reference module write-up` - `docs/modules/scale_reference.md`, `outputs/scale_reference/*` - the
   defense needs every decision and number traceable.

## 8. Concepts to explain in the defense

- **Why a known-size object fixes scale:** a picture of a 297 mm sheet that spans 200 px is twice as far away as one
  spanning 400 px (for a fixed lens). With the perspective of four corners we get the full 3D pose of the sheet, so
  the camera's height above the floor in metres. The same height in the reconstruction gives the scale ratio.
- **Homography / pose of a plane:** four points on a plane fix how that plane maps into the image. If we also know the
  rectangle's real size and the lens (K), there is one 3D placement that explains it.
- **Why the rectangle constraint matters:** with the shape forced to be an exact A4 rectangle, the fit cannot "absorb"
  corner noise by skewing the shape, so the distance is 5x more stable.
- **Monte Carlo uncertainty:** we jiggle the inputs (corners, focal) the way they could plausibly be wrong, refit
  many times and read the spread. That spread is the interval.
- **Calibration / coverage:** if we say "95% interval", the truth must fall inside about 95% of the time. We check
  that on hundreds of rendered sheets where the truth is known.
- **Inverse-variance weighting:** precise cues count more (weight 1/sigma^2); the combined sigma is smaller than each.
- **Correlated errors:** ten photos from the same phone share its focal error; averaging them does not remove it.
- **Robust rejection / Birge ratio:** a cue far from the consensus is dropped when a majority exists; if the remaining
  cues still disagree, we widen the interval rather than pretend.
- **Focal self-calibration from a rectangle:** a wrong focal length makes the back-projected sheet's corners not
  90 degrees; under a tilted view this tells us the focal.
