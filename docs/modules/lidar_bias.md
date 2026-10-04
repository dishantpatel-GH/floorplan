# Module: LiDAR bias (why LiDAR wall-to-wall distances read short, and the fix)

Files:

- `scripts/investigate_lidar_bias.py`: tests five hypotheses on any ARKitScenes room with a Faro laser scan (`arkit`), runs the laser-free internal tests on Stray Scanner captures (`stray`), and pools all rooms and evaluates the correction (`summary`).
- `floorplan/uncertainty/bias.py`: the correction and its systematic error term (`LidarBiasConfig`, `apply_to_plan`, `correct_plane_offset`, `correct_interior_distance`).
- `docs/modules/lidar_bias.md`: this document.

Evidence is in `outputs/lidar_bias/`:

- `summary.json`, `summary_before_after.png`: every room pooled, with before and after numbers.
- `preregistration.json`: the correction value and the prediction, written **before** the two hold-out rooms were analysed.
- `arkit_<video_id>/`: one folder per room, each holding:
  - `results.json`;
  - `h1_depth_vs_range.png`;
  - `h4_estimators.png`;
  - `surface_profiles.png`;
  - `surface_maps.png`.
- `stray_<capture>/results.json`: the internal tests on the three sample captures.

Data: ARKitScenes rooms, each recorded by an iPad Pro (LiDAR + ARKit poses) and scanned by a Faro laser.

- 47895909 is the room from the validation module.
- The other four rooms were fetched for this module with `data/arkitscenes/fetch_parallel.sh` and the official `download_data.py`:
  - 42445884 (visit 422009);
  - 47331133 (visit 470348);
  - 47430003 (visit 470537, *Validation* split);
  - 47333561 (visit 469650).
- Each room is a different visit, so a different home. Each laser scan is single-scan and about 1.8 GB.

---

## 1. Purpose

`docs/modules/arkitscenes_validation.md` (Issue 3) found that interior distances measured from raw LiDAR points read **short** against
the laser:

- room box −1.52 cm (depth) and −1.92 cm (height);
- a 43-pair fit of error = −1.47 cm − 0.89 cm/m · distance.

With gates of 1–2 cm (ceiling height ≤ 1.5 cm), this decides pass or fail. This module finds out **why**, decides whether a
correction is justified and generalises, implements it, and sizes the systematic term that every LiDAR interval must carry. It is
written so that it can be read as a **Part 4 fix-loop declaration** (§5.6).

## 2. How it works, step by step

For each ARKitScenes room (`investigate_lidar_bias.py arkit`):

1. **Same pipeline, same registration.** The script imports `scripts/validate_arkitscenes.py` unchanged:
   - stage 1 (keyframes → TSDF → raw points → `align_scene`);
   - laser loading;
   - registration (`align_scene` on the laser, 4 yaws + FFT, point-to-plane ICP);
   - surface matching (`measure_against_laser`).

   So the baseline numbers are exactly the validation's numbers.
2. **All observations.** Besides the de-duplicated raw points (one per 5 mm voxel), the script re-reads **every** high-confidence
   depth sample of the keyframes (stride 2), with its range, viewing ray and incidence angle. That is 1–4 million per room.
3. **Per matched surface.** The laser plane is the reference, and its normal points into the room. On the cells that both sensors see, the script collects:
   - our raw points;
   - our observations;
   - the laser points;
   - each point's offset along the laser normal ("inward" = + = into the room);
   - the in-plane distance to the edge of our patch (corner, opening or occluder).
4. **Translation-free bias.** A leftover registration shift t moves every +axis surface by +t and every −axis surface by −t.
   - So per axis, `b = (median inward of the + side + median inward of the − side)/2` cancels t exactly. This is `balanced_bias`.
   - Per surface, the script fits `inward_s = b + sign_s · t_axis` (`bias_translation_fit`).
5. **H1 (depth scale or offset).**
   - (a) The translation-free b binned by range, by incidence and by edge distance. A depth scale error would make b rise linearly with range.
   - (b) **Internal near-vs-far test, no laser.** Within each 10 cm patch (a fixed effect), regress the depth error on range. The pooled slope is
     Σnum/Σden over surfaces, with a cluster bootstrap over surfaces.
   - (c) **Internal constant-offset test.** Within a patch, regress the inward offset on cos(incidence). A constant along-ray depth offset b_d gives slope +b_d.
6. **H2 (VIO scale).** Point-to-plane registration with a scale, written as our own Gauss-Newton:
   - 7 DoF (isotropic) and 9 DoF (one scale per axis);
   - run on raw points, walls only, and TSDF;
   - 0.5 m spatial-block bootstrap.
7. **H3 (registration).** Re-measure everything with the laser moved ±1 cm in x, y and z, ±0.3° in yaw and +0.2° in tilt, and (room 47895909) with
   the independent smoke-test transform.
8. **H4 (measurement method).** Re-measure every surface with 12 estimators while the laser reference stays fixed:
   - the current trimmed-LS plane;
   - mean, median, trimmed offset and KDE mode within 3 cm;
   - RANSAC + LS;
   - the 1.0–1.5 m band;
   - all observations without de-duplication (trimmed, mode, 1/σ(r)² weighted);
   - observations at range < 1.5 m only;
   - observations ≥ 0.3 m from the patch edge only.

   The script also redoes the pair statistics with **surfaces** as the unit (cluster bootstrap), because the 43 pairs reuse 22 surfaces.
9. **H5 (laser).** Laser plane RMSE, the laser's own estimator spread, and the best-agreeing pairs.

`stray` runs steps 5b and 5c on the three sample captures, which have no ground truth. `summary` pools the rooms and applies `floorplan/uncertainty/bias.py`:

- **leave-one-room-out** on the three development rooms;
- the **pre-registered** value on the two hold-out rooms.

## 3. Decisions

| # | Decision | Options | Choice and why | Evidence |
|---|---|---|---|---|
| B1 | Get more rooms before deciding | (a) decide on the one validation room; (b) fetch more single-scan ARKitScenes rooms | **(b).** With one room, any correction is fitted and tested on the same data. Fetching 4 rooms took about 40 min in the background (the CDN gives about 20 MB/s with 8 range requests) | 5 rooms, 5 visits, 14 room-box dimensions |
| B2 | Development / hold-out split | fit on all; LORO only; dev + pre-registered hold-out | **3 dev rooms (LORO) + 2 hold-out rooms analysed only after `preregistration.json` was written.** This is the strongest guard against tuning the fix to the test | `preregistration.json` timestamp 00:45; hold-out runs started afterwards |
| B3 | Unit of analysis for "error vs distance" | 43 pairs as independent; surfaces as clusters | **Surfaces (cluster bootstrap).** A small, badly matched surface appears in up to 6 pairs, and that is what produced the −0.89 cm/m "slope" | Slope 95% CI over surfaces: [−4.9, +2.5] cm/m in 47895909. Slopes across rooms: −0.89, +0.27, +0.48, +0.98, −0.07 cm/m (no consistent sign) |
| B4 | Model of the error | depth scale; VIO scale; constant inward offset per surface | **Constant per-surface inward offset b.** It is flat with range in every room, the pair *offsets* are consistently negative while the *slopes* flip sign, and the scale fits are anisotropic inside a room | §5.1–5.3 |
| B5 | Correct, or only widen the intervals | (a) widen only (the validation's choice with one room); (b) correct + residual sigma | **(b).** The hold-out test confirmed the prediction: the mean room-box error goes from −1.25 cm to +0.11 cm, and the mean \|error\| from 1.41 to 0.71 cm | §5.5 |
| B6 | Where to apply it | shift points along rays; shift planes along normals; scale distances | **Planes along their normals (plan quantities, `apply_to_plan`).** The bias does not grow with range, so it is not along-ray depth (the internal cos test even has the opposite sign). Shifting fitted planes is exact for what the plan reports | §5.1, §5.2 |
| B7 | Value of b | per-surface area-weighted fit; pair-fit offset/2; half the mean room-box error | **Half the mean room-box error of the dev rooms: b = 0.68 cm.** It is the quantity the gates use. The per-surface fit gives 0.61–1.04 cm and depends on small, edge-dominated surfaces | Refit on all 5 rooms: 0.66 cm (difference 0.2 mm) |
| B8 | Default for iPhone (Stray) captures | off; on with validated sigma; on with extra device term | **On, with `device_validated=False`.** This adds b in quadrature: σ per distance = √(0.81² + 0.68²) = 1.06 cm. iPhone and iPad use the same ARKit sceneDepth pipeline, so on is the best estimate. Without iPhone ground truth the honest interval is wider | §6 |
| B9 | Estimator | trimmed LS (current); mode; band; interior | **Keep trimmed LS.** No estimator removes the bias in every room. Rankings change from room to room, so picking one would be overfitting | §5.4 |

## 4. Issues log

**Issue 1: "−0.89 cm/m" looked like a scale error.**

- **Root cause.** Pseudo-replication: 43 pairs, but only 22 surfaces. A few small surfaces near corners and openings, with 3–6 cm inward
  offsets (vestibule seen through the door, door header, back wall behind the shower curtain), take part in many pairs.
- **Evidence.**
  - Cluster bootstrap CI for the slope: [−4.9, +2.5] cm/m.
  - Shifting the laser by 1 cm changes which small surfaces match, and the fit moves to offset −0.71 … −1.65 cm and slope −0.82 … −1.23 cm/m, while the room box moves by at most 0.1 cm.
  - Other rooms: slopes +0.27, +0.48, +0.98 and −0.07 cm/m.
- **Fixes considered.** (a) Weighted regression; (b) cluster bootstrap; (c) report room boxes plus per-surface b.
- **Chosen.** (b) + (c). A regression on pseudo-replicated pairs should not drive a decision.

**Issue 2: the first internal near-vs-far slope came out as +41 mm/m.**

- **Root cause.** A bug. The absolute coordinate (about 1 m) was divided by cos(incidence) before the per-patch demeaning, so the incidence spread turned into metres.
- **Fix.** Use the offset relative to the plane.
- **Result after the fix.** −7.2 mm/m, CI [−18.5, +3.0].

**Issue 3: the per-surface "shared bias + translation" fit was dominated by small surfaces.**

- **Symptom.** b = 1.52 cm unweighted vs 1.04 cm area-weighted in 47895909, with an implausible 1.7 cm z translation.
- **Fix.** Use `balanced_bias` (per-axis medians over all observations) for the H1 curves. Use room boxes for the correction value (B7).

**Issue 4: the trimmed-LS plane and a fixed-normal offset disagree by up to 1.7 cm on one surface.**

- **Example.** The 47895909 ceiling: plane fit +1.59 cm inward, point offsets +0.17 to +0.33 cm.
- **Root cause.** Our ceiling plane is tilted **1.62°** relative to the laser's. Evaluating a tilted plane at the bbox centre of a partly seen patch
  extrapolates. Several walls are tilted 1.3–1.6° (`surfaces[].normal_angle_deg`).
- **Status.** This is a second, random (not one-signed) error source of about ±0.5–1 cm in partly seen surfaces.
  - It is why my per-surface reconstruction of 47430003's width differs from the validation's pair measure (−0.42 vs +0.40 cm).
  - Not fixed here: the plan extractors own their plane fits.
  - Recommendation: when a pair of surfaces is measured, constrain both planes to a shared normal (parallel-plane fit). This takes the
    tilt out of the distance. See §6.

**Issue 5: `pkill -f <script>` killed the shell that ran it**, because the shell's own command line matched the pattern. Use explicit PIDs.

**Issue 6: the KDE-mode estimator crashed on a laser patch narrower than its kernel.**

- **Root cause.** `np.convolve(..., "same")` returns the longer of the two lengths.
- **Fix.** Pad the grid by 4 bandwidths. The hold-out room 47430003 was rerun after the fix.

## 5. Results

### 5.1 H1: depth scale error in the sensor. **Rejected** as the cause of the shortfall

The translation-free inward bias b (cm) by range does not rise with range in any room:

| Range (m) | 0.5–0.75 | 0.75–1.0 | 1.0–1.25 | 1.25–1.5 | 1.5–2.0 |
|---|---|---|---|---|---|
| 47895909 | 0.96 | 0.58 | 0.64 | 1.40 | 0.64 |
| 42445884 | 0.68 | 0.65 | 0.83 | 0.89 | – |
| 47331133 | 0.29 | 0.55 | 0.55 | 0.58 | 0.62 |
| 47333561 | 0.58 | 0.61 | 0.97 | 0.98 | 0.79 |
| 47430003 | 0.05 | 0.41 | 0.18 | 0.19 | −0.57 |

A −0.9% scale error would add 0.9 cm per metre, which means +1.2 cm between the first and the last column. No room shows that. Instead
b is about 0.5–0.9 cm **at every range**.

- **Internal near-vs-far slope** (no laser; the same 10 cm patch seen from different ranges; pooled, 95% cluster CI, mm/m):

  | Room | Slope | 95% CI |
  |---|---|---|
  | 47895909 | −7.2 | [−18.5, +3.0] |
  | 42445884 | +1.9 | [−11.4, +11.4] |
  | 47331133 | −8.7 | [−19.1, +26.2] |
  | 47333561 | +0.2 | [−23.2, +11.7] |
  | 47430003 | +9.1 | [+7.3, +25.8] |

  The sample captures (iPhone, Stray Scanner) give +14.0 [+0.7, +35.6] (c00a170fe1), +4.6 [+1.5, +10.5] (1a8384c3f6) and
  +5.0 [+2.0, +8.2] (c7d28f72c6).

  - Per-surface slopes are heterogeneous: −37 to +65 mm/m, each with SE ≈ 1 mm/m.
  - So this test is dominated by pose and drift differences between the near and far frames, not by a sensor law.
  - On ARKit rooms, where the laser shows no range trend, it still swings by ±9 mm/m.
  - **Conclusion:** the internal test cannot resolve a sub-1% scale, and it must not be used as a calibration.

  The positive iPhone slopes (0.5–1.4%) are noted as an open item in §6, not acted on.
- **Internal constant-offset test** (inward offset vs cos(incidence) within a patch): −4, −13, −10, −10 and −17 mm. This is the **opposite**
  sign to a constant along-ray depth offset, which would give +6 mm. Within a patch, grazing observations sit *further* into the room than
  head-on ones. That fits beam-footprint or densification blur, not a range offset.
- Absolute b by incidence is roughly flat (for example 47331133: 0.63, 0.51, 0.52, 0.62, 0.74 cm from 0–15° to 60–75°). So the shift exists
  even head-on.

### 5.2 H2: pose / VIO scale. **Rejected**

The similarity registration does find a "scale" (ours too small):

| Room | Isotropic scale − 1 | Per-axis x / y / z |
|---|---|---|
| 47895909 | +0.57% | 0.19 / 0.32 / 1.38% |
| 42445884 | +1.06% | 1.77 / 0.81 / 1.04% |
| 47331133 | +0.50% | 0.54 / 0.30 / 0.68% |
| 47333561 | +0.82% | 0.92 / 0.69 / 0.81% |
| 47430003 | +0.47% | 0.16 / 0.49 / 0.61% |

Within one room, the per-axis bootstrap CIs do not overlap (47895909: x [0.10, 0.28] vs z [1.31, 1.46]). A VIO scale error is
isotropic, so this is not one. In a 1.5–2.5 m room, moving every surface 0.7 cm inward looks to a scale fit exactly like a shrink
of 0.7/(half-size) ≈ 0.5–1%. The fit is absorbing the offset. The offset-vs-slope test on pairs separates the two:

- pair offsets −1.47, −1.89, −2.05 and −4.70 cm (+0.64 cm in the 6-pair room);
- slopes of inconsistent sign.

### 5.3 H3: registration pulls the cloud. **Rejected**

Over all perturbations (±1 cm in x, y, z; ±0.3° yaw; +0.2° tilt; the independent transform for 47895909), the room box moves:

- by at most **0.2 cm** in 4 rooms;
- by 0.64 cm on width_z of 47430003, the small room with only 6 pairs.

The independent smoke-test transform gives +0.24 / −1.52 / −1.92 cm against +0.25 / −1.52 / −1.92. Distances between facing planes are
translation-invariant by construction, and the measurement confirms it. Registration only moves the per-surface offsets, which is why the
analysis uses translation-free b.

### 5.4 H4: our measurement method. **A minor contributor, not the cause**

Room-box error (cm) by estimator. The laser reference is fixed. These are per-surface reconstructions, so the trimmed-LS row differs slightly from the validation baseline (Issue 4).

| Room (x / z / h) | trimmed LS | median | KDE mode | 1.0–1.5 m band | all obs (no dedup) | interior ≥ 0.3 m |
|---|---|---|---|---|---|---|
| 47895909 | +0.23 / −1.56 / −2.12 | +0.45 / −2.25 / −0.67 | +0.87 / −3.21 / +0.07 | +1.16 / −0.63 / – | +0.45 / −2.53 / −0.74 | +0.63 / −1.66 / −0.40 |
| 42445884 | −1.97 / −1.44 / −1.61 | −1.30 / −1.13 / −1.64 | −0.12 / −1.29 / −1.61 | −1.15 / −1.40 / – | −1.37 / −1.20 / −1.50 | −1.21 / – / −1.62 |
| 47331133 | −1.41 / −1.18 / −0.67 | −1.14 / −1.25 / −0.61 | −1.07 / −1.04 / −0.17 | −1.19 / −1.46 / – | −1.25 / −1.07 / −0.71 | −0.77 / −2.38 / −0.59 |
| 47333561 | −2.43 / −1.72 / −0.97 | −2.25 / −1.29 / −1.25 | −2.01 / −1.12 / −1.47 | −2.55 / −1.11 / – | −2.45 / −1.49 / −1.44 | −2.33 / −0.95 / −1.73 |

- **Every estimator is short in every room.**
- The per-surface histograms (`surface_profiles.png`) show the **whole** distribution of our points shifted by 0.5–1 cm (for example
  47331133, +z wall: peak at +0.9 cm), not a one-sided tail. A robust estimator cannot remove a shifted peak.
- De-duplication, 1/σ(r)² weighting and near-range-only change the room box by a few millimetres.
- Edge proximity adds to the bias in 47895909, where b is 0.8–0.95 cm at 5–30 cm from an edge and 0.05 cm beyond 30 cm. That is the validation's
  "corner" finding. It does not generalise: 42445884 is flat (0.8–1.0 cm everywhere), and 47333561 is 0.8–1.3 cm at all edge distances.
- What H4 *does* explain:
  - the pair statistics (Issue 1, pseudo-replication on small edge surfaces);
  - a ±0.5–1 cm random term from plane tilt and extrapolation on partly seen surfaces (Issue 4).

### 5.5 H5: the laser or its registration. **Rejected** within what this data can test

- Laser plane RMSE is 2.0–5.0 mm (median per room).
- The laser's own estimator spread is a median of 0.3–1.0 mm, so the reference does not depend on the estimator.
- In every room some facing pairs agree to **0.10–0.25 cm**. A laser scale error would affect every pair proportionally.
- Five different Faro scans from five visits give the same sign and size. The Faro S70 range accuracy is ±1 mm.

What cannot be excluded: a material-dependent laser effect of 1–2 mm, for example sub-surface penetration on paint or tile. It is too
small to explain 7 mm.

### 5.6 Root cause, the fix, and the fix-loop declaration

**Root cause (supported).** Surfaces reconstructed from ARKit LiDAR depth (iPad Pro) sit about **0.66 cm into the room** relative to a survey
laser.

- The offset is constant: independent of range (§5.1), roughly independent of incidence, of the estimator (§5.4) and of the registration (§5.3).
- It is present in 5 of 5 homes.
- Every interior distance therefore reads about 1.3 cm short.

The physical mechanism inside Apple's depth pipeline (dToF timing offset, or blur in the RGB-guided densification) cannot be
separated with this data. The internal tests show it is *not* a pure along-ray range offset. Correcting it is a black-box calibration,
like the hook offset of a tape measure.

**Fix-loop declaration (Part 4 format).**

| Step | Content |
|---|---|
| Failing number | Ceiling height −1.92 cm on 47895909, which fails the 1.5 cm gate. Over 3 development rooms the 9 room-box errors average **−1.37 cm**; only 22% are within 1 cm and 56% within 1.5 cm |
| Root-cause hypothesis | A constant per-surface inward offset of about 0.7 cm (not a scale, not registration, not the estimator) |
| Evidence | b flat with range in 5 rooms. Pair offsets consistently −1.5 to −4.7 cm while slopes flip sign. Scale fits anisotropic within a room. Room box moves ≤ 0.2 cm under registration perturbations. Every estimator short in every room. Whole point distribution shifted |
| Fix | `floorplan/uncertainty/bias.py`: move each LiDAR surface outward by b = 0.68 cm, so interior distances +2b, and add the residual systematic σ in quadrature (switch: `LidarBiasConfig.enabled` or `FLOORPLAN_LIDAR_BIAS=off`) |
| Predicted number (pre-registered before the hold-out) | Hold-out room-box mean error −1.4 ± 0.7 cm before, **0.0 ± 0.7 cm after**, mean \|e\| from about 1.4 to about 0.6 cm. Falsified if \|mean\| after > 1 cm |
| Measured after: development rooms (leave-one-room-out, 9 dims) | Mean −1.37 → **+0.00 cm**. Mean \|e\| 1.42 → **0.55 cm**. Within 1 cm: 22% → **78%**. Within 1.5 cm: 56% → **89%** |
| Measured after: **hold-out rooms** (2 rooms, 5 dims, never seen when b was chosen) | Mean −1.25 → **+0.11 cm**. Mean \|e\| 1.41 → **0.71 cm**. Within 1 cm: 20% → **60%**. Within 1.5 cm: 60% → **80%**. Std after 1.05 cm (predicted 0.7) |
| Ceiling heights (all 5 rooms) | Before −1.92, −1.82, −0.99, −1.50, −1.07 cm (3 of 5 within 1.5 cm). After −0.40, −0.68, +0.46, −0.14, +0.29 cm: **5 of 5 within 0.7 cm** |
| What got worse | Two widths that were already right: 47895909 width_x +0.25 → +1.78 cm (LORO) and 47430003 width_z +0.40 → +1.76 cm (6-pair room). Both are within 2σ of the residual spread. The correction removes the mean, not the room-to-room scatter |

Per-room detail (cm, x / z / h):

| Room | Role | Before | After |
|---|---|---|---|
| 47895909 | dev (LORO b 0.76) | +0.25 / −1.52 / −1.92 | +1.78 / −0.00 / −0.40 |
| 42445884 | dev (LORO b 0.57) | −2.24 / −1.40 / −1.82 | −1.09 / −0.27 / −0.68 |
| 47331133 | dev (LORO b 0.72) | −1.36 / −1.30 / −0.99 | +0.08 / +0.15 / +0.46 |
| 47333561 | **hold-out** (b 0.68) | −2.47 / −1.59 / −1.50 | −1.11 / −0.23 / −0.14 |
| 47430003 | **hold-out** (b 0.68) | – / +0.40 / −1.07 | – / +1.76 / +0.29 |

**Systematic term for intervals.**

- After correction, the residual per interior distance is σ = **0.81 cm** (1σ; the std of all 14 room-box errors), so ±1.6 cm at 95%. This is
  what `bias.py` adds in quadrature.
- For devices other than the validated iPad class, b is added in quadrature as well (σ = 1.06 cm).
- With the correction **off**, the interval is stretched one-sidedly toward longer by 2b, so it still covers the truth.
- The plane-fit standard error (0.04–0.08 mm) is about 100× too small and must never be used alone.

**How to measure walls to minimise the error** (recommendation to the plan extractors):

1. Measure on raw points, not the TSDF. The validation found the TSDF more biased: room box −2.5 cm on all axes vs +0.25 / −1.52 / −1.92 cm raw.
2. Use a large, mid-wall stretch.
3. Fit facing walls as **parallel planes with a shared normal**, so relative tilt does not enter the distance (Issue 4).
4. Use surfaces, not pairs, as the statistical unit.
5. Apply `apply_to_plan` once, LiDAR tier only, before `tier_budget.widen_plan`.

The estimator (trimmed LS), the height band and near-range weighting change results by only millimetres, and no choice wins across rooms.

## 6. Limitations and next steps

- **Five rooms, one device class (iPad Pro), single laser scans.** The hold-out has 5 dimensions. The residual std after correction (0.8–1.05 cm) is the honest
  number, and more rooms would tighten b (σ_b ≈ 0.81/√14/2 ≈ 0.1 cm already).
- **iPhone transfer is unverified.**
  - The LiDAR-tier sample data and Stray Scanner are iPhone. The internal near-vs-far slopes there are positive (0.5–1.4%), but on ARKit rooms
    this test proved unreliable (§5.1). So nothing is concluded from it, and the device term is added instead (B8).
  - Fix: one tape or laser-measurer check of a room on an iPhone Pro, or one ARKitScenes-like iPhone capture with ground truth.
- **Mechanism not identified.** This is a black-box calibration. If Apple changes the depth pipeline (iOS version), b may change. Store `ios_version`
  with captures and revalidate.
- **Opening widths are not corrected.** Jamb geometry was not validated, so they get σ only. Next: measure door widths on the ARKitScenes rooms against the laser.
- **The plane-tilt term (Issue 4)** is real and random (±0.5–1 cm on partly seen surfaces). It should be fixed in the plan extractors with parallel-plane
  fits, then re-measured here.
- Wall thickness (−2b) and area or perimeter (polygon offset) corrections follow from geometry but were not checked against the laser.
- `bias.py` is not wired into `scripts/run_capture.py` (not my file). Proposed: call `apply_to_plan(plan, LidarBiasConfig.load())` on LiDAR-tier plans
  just before `widen_plan`, and add `lidar_bias_correction: bool = True` to `floorplan/config.py`.

## 7. Proposed commits (in order)

| # | Message | Files | Why |
|---|---|---|---|
| 1 | `chore(data): fetch 4 more single-scan ARKitScenes rooms (visits 422009, 470348, 470537, 469650)` | `scripts/fetch_arkitscenes.sh` (the loop used here; strip `\r` from the mapping CSV) | More rooms, so a correction is not fitted on one room |
| 2 | `feat(eval): lidar-bias investigation (H1-H5) reusing the validation pipeline` | `scripts/investigate_lidar_bias.py` (arkit mode) | Evidence for each hypothesis, with translation-free statistics |
| 3 | `fix(eval): internal near-vs-far slope used absolute coordinates` | same file | Issue 2 (+41 → −7 mm/m) |
| 4 | `feat(eval): stray mode (internal tests on sample captures) and pooled summary with LORO` | same file | Generalisation evidence |
| 5 | `docs(eval): pre-register LiDAR bias correction before hold-out rooms` | `outputs/lidar_bias/preregistration.json` (or copy into the doc) | Makes the hold-out test credible |
| 6 | `fix(eval): pad KDE-mode grid` | same script | Issue 6 |
| 7 | `feat(uncertainty): LiDAR inward-bias correction + systematic sigma (switchable)` | `floorplan/uncertainty/bias.py` | The fix |
| 8 | `feat(pipeline): apply LiDAR bias correction before tier widening` | `scripts/run_capture.py`, `floorplan/config.py` (owners) | Wire it in |
| 9 | `docs: lidar_bias module + decision D-0xx; amend arkitscenes_validation Issue 3` | `docs/modules/lidar_bias.md`, `docs/DECISIONS.md`, `docs/modules/arkitscenes_validation.md` | Explain the decision |

Suggested decision entry:

> **D-0xx LiDAR surfaces sit about 0.7 cm into the room. Correct it, and carry ±1.6 cm (95%) systematic per interior distance.**
>
> - Evidence: 5 ARKitScenes rooms with laser ground truth. Not scale, not registration, not the estimator.
> - Fix pre-registered on 3 rooms, tested on 2 hold-out rooms: mean room-box error −1.25 → +0.11 cm.
> - Ceiling heights 5/5 within 0.7 cm after correction.

## 8. Concepts to explain in the defense

- **Bias vs noise.**
  - Noise averages down with more points; bias does not.
  - Here the plane-fit standard error is 0.05 mm, while the bias is 7 mm.
  - Only a reference instrument (the laser) reveals a bias.
- **Scale error vs offset error.**
  - A scale error grows with distance (slope).
  - An offset per surface costs every interior distance the same amount, 2b (intercept).
  - Test: regress error on distance, and check whether the bias grows with range.
- **Why a similarity registration "finds" a scale.**
  - In a small room, moving every wall inward by b is almost the same as shrinking by b/half-size.
  - The per-axis fit exposes it: x ≠ z inside one room is impossible for a true VIO scale.
- **Translation-free statistics.** A registration shift moves opposite walls in opposite inward directions, so averaging the + and − sides cancels it.
- **Pseudo-replication and the cluster bootstrap.**
  - 43 pairs built from 22 surfaces are not 43 independent samples.
  - Resample the surfaces, not the pairs, to get honest intervals.
- **Fixed effects.** Comparing observations *within* the same 10 cm patch removes everything about the patch (its true position, registration error),
  leaving only how the measurement depends on range or incidence.
- **Leave-one-out and pre-registration.**
  - Write down the correction and its predicted effect *before* looking at the test rooms.
  - The hold-out confirmed the prediction (−1.25 → +0.11 cm). That is the difference between a calibration and curve fitting.
- **Black-box calibration.** Like the hook offset of a tape measure: you do not need to know *why* it is offset to correct it, but you need to know
  that it is stable (five homes) and how much it scatters (σ 0.8 cm). That scatter goes into every interval.
- **Why the correction moves planes, not points.** The bias does not grow with range and is not along the ray, so the right object to shift is the fitted surface, along its normal.
