# Module: drift correction (fragment pose graph + ICP loop closures + plane anchors)

Files:

- `floorplan/recon/drift.py`: the module. Public entry point: `correct_drift(cap, kf, T_wc, cfg_or_params) -> (T_wc_corrected (N,4,4) for ALL frames, report dict)`.
- `scripts/drift_ablation.py <capture_dir> [--sweep] [--no-variants] [--folds 5]`: the on/off ablation, the variants, the held-out validation, the figures.
- `docs/modules/drift.md`: this document.

Evidence is in `outputs/drift/<capture>/`. Every number below comes from these files.

| File | What it holds |
|---|---|
| `T_wc_corrected.npy` | corrected camera-to-world poses for every frame; the plan extractors can be re-run on them |
| `scene_on/` | the corrected scene, same format as `outputs/<capture>/scene.npz` |
| `drift_report.json` | the module report, held-out residuals, ICP noise floor, `drift_sigma` |
| `ablation.json` | the OFF / ON / variant table |
| `sigma_sweep.json` | noise-model sweep, written with `--sweep` |
| `fig_overlay.png` | OFF vs ON walls in one frame, with zooms where they disagree most |
| `fig_side_by_side.png` | each scene in its own aligned frame, with footprint proxies |
| `fig_corrections.png` | per-fragment corrections plus floor, wall and ceiling anchors, ARKit vs corrected |
| `fig_loops.png` | per-loop residual, in-sample and held-out |
| `run_<capture>.log` | the console output of the run |

The cross-capture registration with drift on vs off is in `outputs/drift/registration_off/` and
`outputs/drift/registration_on/`. It was produced by the eval module's `scripts/register_scenes.py`.

---

## 1. Purpose and where it sits in the pipeline

**The problem.** ARKit's visual-inertial odometry (VIO) is very good over a few metres, but it is *dead reckoning*: each
step's small error is added to all the previous ones. After a long walk the camera's estimated position and heading
are off. When it comes back to a room it saw before, the same wall is drawn a second time, a few centimetres or
degrees away. On the stitched plan that appears as:

- doubled or blurred walls;
- rooms that do not close;
- wrong room-to-room offsets;
- a ceiling measured at a different height from the floor it belongs to.

**The gate.** The case study's *Drift accountability* row requires two things:

1. The report states what we do about accumulated drift on the multi-room capture.
2. An ablation shows the stitched footprint with the correction **on and off**.

"Poses used as-is" is an automatic fail.

**Where it sits.**

```
load_stray -> select_keyframes (ARKit poses)
                    │
                    ▼
            correct_drift(cap, kf, T_wc)          <- this module
              fragments -> plane anchors -> 4-DoF pose graph <-> ICP loop closures (2 rounds)
              -> per-fragment corrections -> blended onto every frame
                    │  T_wc_corrected
                    ▼
build_scene(capture, cfg, T_wc=T_wc_corrected) -> TSDF + raw points + alignment -> plan extractors
```

Today `build_scene` does not call it yet; it accepts `T_wc=` (see *Requested changes to shared code*). The ablation
script builds both scenes, and saves `T_wc_corrected.npy` plus `scene_on/` so the plan modules can be re-run on
corrected poses.

**What it delivers to other modules.** It also delivers `drift_sigma`: how far two observations of the same surface
made at different times still disagree after correction. This is the residual-drift term of every interval (the
error budget in `docs/TECHNICAL_REPORT.md` §4).

---

## 2. How it works, step by step

### Step 1: cut the walk into fragments (`split_fragments`, `build_fragments`)

- **What.** Walk along the keyframes and start a new fragment every **1.5 m of camera travel** or **5 s**, whichever
  comes first.
- **Building each fragment.** All its keyframes' high-confidence LiDAR points (every 4th pixel) are back-projected
  into the frame of its **middle** keyframe (the "anchor"), using ARKit's poses. They are then voxelised at 2 cm and
  given normals oriented towards the camera.
- **Why.** ARKit is locally near-exact, so a fragment is a trustworthy rigid piece of geometry.
  - Drift only builds up *between* fragments, so we need one pose per fragment, not one per frame.
  - with_ceiling has 9,745 frames and 1,978 keyframes, but only 66 fragments. That is a small, well-posed problem.
- **Why the middle frame.** A rotation correction applied at the anchor moves the fragment's ends the least
  (smallest lever arm).
- **Without it.**
  - Per-frame registration: 256x192 depth frames are too small and noisy for ICP to be well constrained.
  - Per-room registration: rooms are not known yet, because drift correction comes *before* room segmentation.

### Step 2: measure plane anchors per fragment (`measure_plane_anchors`, `anchor_sets`)

For each fragment, under its ARKit pose, we measure two things.

- **Dominant wall direction** (`manhattan_yaw`, reused from `plan/align.py`).
  - Most interior walls are mutually perpendicular, so a fragment's wall normals cluster at one angle modulo 90°.
  - If ARKit's heading drifted by 2°, every wall in that fragment appears turned by 2° relative to the building.
  - It is an **absolute heading reference that needs no loop closure**.
- **Floor level** (`estimate_floor`, reused, refined to sub-cm by the median of the floor points).
  - One flat floor per apartment, so a fragment whose floor reads 2 cm high has drifted 2 cm vertically.
- **Gates.**
  - A fragment counts as Manhattan if at least 50% of its wall normals agree.
  - The Manhattan anchors switch on only if at least half of all fragments are Manhattan, so a non-Manhattan
    building turns them off automatically.
  - Floor readings more than 10 cm from the global floor are ignored. Those are beds and tables, which are always
    higher than the floor, never lower.
- **Ceiling level.** It is measured too, but **only as a check**. It is never used in the solve, because rooms may
  legitimately have different ceiling heights. That makes it an independent witness for the vertical correction
  (section 5).

### Step 3: odometry edges (`odometry_edges`)

- **What.** Consecutive fragments are linked by **ARKit's own relative pose**: a stiff spring whose stiffness falls
  with distance travelled.
  - Translation σ = 1 cm/√m.
  - Yaw σ = 0.6°/√m.
  - Plus a yaw-smoothness term (step 5).
- **Why ARKit and not ICP.** Over 1.5 m ARKit beats fragment-to-fragment ICP (Issue 1).

### Step 4: loop closures (`loop_edges`, `register`, `verify_loop`)

1. **Candidates.**
   - Every pair of non-consecutive fragments is a candidate.
   - It is kept if, under the *current* pose estimate, at least 30% of either fragment lies within 5 cm of the other
     (a KD-tree test on 5 cm-voxelised clouds, after a cheap bounding-sphere test).
2. **ICP.** Coarse-to-fine point-to-plane ICP at 8 → 4 → 2 cm, with correspondence gates of 20 → 8 → 4 cm and a
   Tukey robust loss. It starts from the current estimate of the relative pose.
3. **Verify.** A loop is accepted only if all four checks pass:
   - fitness ≥ 0.30;
   - point-to-plane RMSE ≤ 1.0 cm;
   - ICP moved less than 30 cm / 5° from its start;
   - the weakest translation direction is constrained by ≥ 1% of correspondences (the degeneracy check).
4. **Information matrix.** Computed as **point-to-plane** (`point_to_plane_stats`). In a corridor, motion along the
   corridor gets almost no weight, so a sliding ICP cannot drag the graph.
5. **Two rounds.** Loops are searched again after each solve. The first solve uses the plane anchors only, and it
   removes most heading drift. That brings revisits within ICP's convergence basin. On floor_only, raw ARKit had
   revisits 16–34 cm and 2–3° off, and ICP from there failed (Issue 3).
6. **"Local" vs "revisit".** Loops more than 20 s apart are "revisits", the real drift test. Closer ones are "local".

### Step 5: the robust 4-DoF pose graph (`build_graph`, `solve_4dof`)

- **Unknowns.** Per fragment k:
  - a yaw correction ψ_k about the vertical through its anchor;
  - a shift d_k = (dx, dy, dz).

  Plus the building's Manhattan direction φ and the global floor level. Fragment 0 is fixed: this is the "gauge",
  because the whole map could otherwise slide or rotate freely.
- **Why 4-DoF.** Roll and pitch are **not** unknowns. The accelerometer measures gravity directly, so ARKit's tilt
  does not drift. A 6-DoF solver only lets ICP noise tilt fragments: the Open3D 6-DoF baseline tilts them by up to
  0.7–1.2° (Issue 2). VINS-Mono's loop-closure pose graph is 4-DoF for the same reason.
- **Residuals** (all whitened by their σ, so they are unit-free):

  | Term | Formula | σ |
  |---|---|---|
  | edge translation | `c'_s − c'_t − Ry(ψ_t)·R_t·p_meas` | odometry: 1 cm/√m. loops: 1.5 cm, shaped by the point-to-plane information |
  | edge yaw | `ψ_s − ψ_t − yaw(R_t·R_st·R_sᵀ)` | odometry: 0.6°/√m. loops: 0.4° |
  | Manhattan | `wrap90(θ_k − ψ_k − φ)` | 1° |
  | floor | `floor_k + dy_k − floor_level` | 1 cm |
  | yaw smoothness | `ψ_{k+1} − 2ψ_k + ψ_{k−1}` | 0.03° |

- **Yaw smoothness.** Heading drift comes mainly from residual gyro bias, so it is a slow **ramp**, not jitter. The
  second difference is zero for a ramp and large for zig-zag. This term was added after the wall-sharpness metric
  showed the solver injecting yaw jitter (Issue 5).
- **Robustness: iteratively re-weighted least squares (IRLS) with a Cauchy kernel.**
  1. Solve weighted least squares (scipy `least_squares`, TRF, sparse Jacobian).
  2. Give every loop and anchor a weight w = 1 / (1 + (χ²/dof) / 2²).
  3. Repeat (up to 8 times).

  A loop that disagrees with the rest by 3.5σ ends with w < 0.25 and is reported as "down-weighted". This plays the
  role of Open3D's line process in Choi et al. Odometry is never down-weighted.

### Step 6: apply to every frame (`interpolate_corrections`)

- **The correction.** Per fragment, the world-frame correction is C_k = X'_k X_k⁻¹. Each frame's correction is
  blended in time between the two nearest anchors: SLERP for rotation, linear for translation.
- **Why blend.** Applying one rigid correction per fragment would create seams (steps) at fragment boundaries.
  Blending gives a continuous trajectory. Frames before the first anchor and after the last take the nearest
  correction.

### Step 7: the ablation (`scripts/drift_ablation.py`)

Each step builds both scenes (OFF = ARKit as-is, ON = corrected), puts them in one frame, and computes:

- **Loop residuals.** For each accepted loop, the **surface separation**: the displacement implied by
  (poses vs ICP measurement) projected on the surface normals. This is how far the two observations of the same
  surfaces sit apart, ignoring sliding along a surface, which ICP cannot see.
  - In-sample.
  - **Held-out**: 5-fold, where the correction is recomputed without the fold's loops. The honest number.
- **ICP noise floor.** ICP vs ARKit on *consecutive* fragments, where ARKit is near-exact. Residuals at this level
  cannot be distinguished from ICP noise.
- **Plane-anchor spreads.** Wall direction, floor level, and **ceiling level (independent)**.
- **Wall sharpness (independent of ICP and of the anchors).**
  - Raw LiDAR wall points, labelled by oriented Manhattan face, so the two sides of a wall are never mixed.
  - Split into 0.5 m pieces, and grouped by gaps larger than 10 cm across the wall.
  - Spread around each group's median: MAD-σ, and % within 1 cm.
  - A doubled wall (two passes 3 cm apart) widens it.
- **Footprint proxies.** Observed floor area, filled footprint area, robust bounding box.
- **Variants.** Loops-only, anchors-only, and the textbook Open3D 6-DoF baseline.
- **Figures** (section 5).

---

## 3. Decisions

| # | Decision | Options considered | Choice and why | Evidence |
|---|---|---|---|---|
| DR-1 | What to correct drift with | (a) Nothing ("ARKit is good"). (b) Re-run full SfM / bundle adjustment on RGB (COLMAP/hloc). (c) Per-frame ICP odometry. (d) **Fragment pose graph + ICP loops** (Choi et al. 2015, Open3D system). (e) Learned SLAM (DROID-SLAM, MASt3R-SLAM). | (d), with plane anchors added. (a) is an automatic fail, and the data shows drift up to 65 cm / 5.9°. (b) throws away the LiDAR depth and the metric scale, and takes much longer to run. (c) is noisier than ARKit at short range. (e) means GPU weights plus a licence check, and is not better than VIO+LiDAR for metric cm. Fragments keep ARKit where it is good (locally) and fix only what it is bad at (globally). | Section 5 tables; Issue 1. |
| DR-2 | Degrees of freedom | 6-DoF (Open3D `global_optimization`) vs **4-DoF (x, y, z, yaw)** | 4-DoF. Gravity is observed by the accelerometer, so roll and pitch do not drift. 6-DoF lets ICP noise tilt the map. | Open3D 6-DoF baseline: max tilt change 0.97° (with_ceiling), 0.72° (floor_only), 1.17° (single_room). Its floor-level spread is worse than ARKit's (2.00 vs 0.72 cm, 1.69 vs 1.27 cm) and its ceiling spread is 2.81 cm vs 0.64 cm for ours. A 1° tilt over 5 m moves heights by 8.7 cm, against a 1.5 cm ceiling gate. |
| DR-3 | Odometry edge measurement | ICP-refined (Choi et al.) vs **ARKit relative pose** | ARKit. On consecutive fragments, ICP disagrees with ARKit by a median 0.73–1.04 cm surface separation and 0.33–0.48°. That is ICP noise, not ARKit error. ICP odometry also fails outright on low overlap. | Baseline: 34/65 (with_ceiling), 18/35 (floor_only), 6/9 (single_room) ICP odometry edges failed the sanity check (fitness < 0.3 or > 5 cm / 1° from ARKit) and fell back to ARKit. A development run saw one odometry ICP slip by 108 cm / 139°. `icp_noise_floor` in `drift_report.json`. |
| DR-4 | Fragment size | 0.75 m … 3 m of travel; Choi et al. use ~50 frames | **1.5 m or 5 s.** It is the scale over which ARKit is still near-exact, and a fragment still contains 2–3 walls, so ICP is well-posed. Smaller fragments give more degenerate ICPs; larger ones bake drift into a fragment. | 66 / 36 / 10 fragments for with_ceiling / floor_only / single_room. Not swept (open item, section 6). |
| DR-5 | Information matrix | Open3D point-to-POINT information vs **point-to-plane `Σ JᵀJ`, J = [q×n, n]** | Point-to-plane. It encodes that a corridor does not constrain motion along its walls, so a degenerate ICP is weighted correctly in the solve instead of being rejected. | 6 / 3 / 0 loops rejected as degenerate. Many more were accepted with anisotropic weights (`constraint_frac` per edge in `drift_report.json`). |
| DR-6 | Loop verification thresholds | fitness, RMSE, jump, degeneracy | fitness ≥ 0.3, **point-to-plane** RMSE ≤ 1 cm, jump ≤ 30 cm / 5°, constraint ≥ 1%. Point-to-POINT RMSE is dominated by the 2 cm voxel sampling (two samplings of the same wall are ~1 cm apart in-plane), so it cannot tell good from bad. Point-to-plane can. | First version used point RMSE ≤ 1.5 cm and rejected loops with fitness 0.6 and 1.7 cm jumps. 35/80, 14/30, 3/5 accepted now. |
| DR-7 | Outlier handling | Hard thresholds only; Open3D line process; switchable constraints; **IRLS with Cauchy kernel** | IRLS-Cauchy. Same idea as the line process, works with our own 4-DoF solver, and gives an explicit per-loop weight we can report. | 0 loops ended below w = 0.25 on all captures (`down_weighted`). The verification gates already remove the bad ones. |
| DR-8 | Plane anchors (the "plane-anchored correction" the brief suggests) | none; walls-to-shared-plane landmarks (full plane SLAM); **per-fragment Manhattan yaw + floor level priors** | Per-fragment priors. They are cheap, need no data association, give an absolute heading/vertical reference without loops, and switch off by themselves in non-Manhattan scenes. Full plane landmarks (each physical wall a variable) are the next step (section 6). | anchors-only reduces floor_only's revisit separation (13.95 → 9.10 cm) but cannot fix translation drift. Loops-only leaves floor spread 1.21–1.61 cm. **Both together** give the best wall sharpness on all three captures (section 5). |
| DR-9 | Loop search order | from raw ARKit vs **from the anchor-corrected estimate, 2 rounds** | From the corrected estimate. Raw ARKit put floor_only revisits outside ICP's basin. | floor_only: ICP from ARKit accepted 0 of 5 revisits (20–34 cm, 1.8–3.3° jumps, Issue 3). From the anchored estimate, 8 revisits accepted. |
| DR-10 | Noise model sigmas | sweep | odometry 1 cm/√m and 0.6°/√m; yaw smoothness 0.03°; loop 1.5 cm / 0.4°; Manhattan 1°; floor 1 cm. Chosen on **two criteria**: held-out loop separation (ICP-based) **and** wall sharpness (independent of ICP). | `sigma_sweep.json` (144 combinations). Held-out RMS spans only 1.66–2.15 cm (with_ceiling) and 2.21–2.73 cm (floor_only) over the whole grid, so the result is insensitive to sigmas. Wall-sharpness sweep in Issue 5. |
| DR-11 | Yaw smoothness prior | none; stiffer first-order yaw; **second-difference (constant-drift-rate) prior** | Second difference, σ = 0.03°. A first-order σ could not satisfy both captures (Issue 5). A ramp is free, jitter is not, which matches how gyro-bias drift behaves. | Wall sharpness with_ceiling 15.49 (ARKit) → 15.13 mm; floor_only 11.15 → 9.51 mm. Without it (σ_yaw 0.6) with_ceiling got *worse* than ARKit (16.04 mm). |
| DR-12 | Applying corrections to frames | one rigid correction per fragment vs **time-blended (SLERP + linear)** | Blended: no seams at fragment boundaries. | Corrections vary smoothly (`fig_corrections.png`). |
| DR-13 | How we validate without ground truth | in-sample loop residuals only vs **held-out loops + independent checks** (wall sharpness, ceiling level, cross-capture registration) | In-sample residuals are what the solver minimised, so they are always good. We report held-out (k-fold) loops plus three metrics the solver never saw. | Section 5. |
| DR-14 | Ceiling as an anchor? | use it in the solve vs **check only** | Check only. Rooms can have different ceilings (soffits are visible: 0.95 m vs 1.58 m levels in the same capture), and the ceiling height is itself a gated measurement, so forcing ceilings equal would bias it. Keeping it out lets it validate the vertical correction. | with_ceiling ceiling MAD-σ: 6.03 cm (ARKit) → 0.64 cm (ON). |

---

## 4. Issues log

**Issue 1: ICP-refined odometry is worse than ARKit.**

- **Symptom.** The first version (Open3D 6-DoF, ICP odometry as in Choi et al.) on with_ceiling: 34 of 65 odometry
  ICPs failed the sanity check, and the optimised graph tilted fragments by up to 1.4°.
- **Root cause.** Over 1.5 m ARKit is near-exact, while fragment-to-fragment ICP has a noise floor.
  - ICP vs ARKit on consecutive fragments: median surface separation 0.73–1.04 cm, 0.33–0.48°.
  - Overlap is often low (fitness 0.12–0.42).
  - One pair slipped by 108 cm / 139° (single_room, fragments 0→1).
- **Fixes considered.** Stricter ICP gates; weight ICP odometry less; use ARKit's relative pose.
- **Fix chosen.** ARKit relative pose. It is the better measurement at this scale. The baseline keeps
  `refine_odometry=True` so the comparison is reproducible.

**Issue 2: a 6-DoF solver introduces tilt.**

- **Symptom.** The Open3D baseline changes fragment tilt by up to 0.72–1.17°. Floor-level spread gets *worse* than
  ARKit's: 1.69 vs 1.27 cm on floor_only, 2.00 vs 0.72 cm on with_ceiling.
- **Root cause.** ICP's rotation noise (~0.3–0.8°) enters all 3 axes. Nothing in the 6-DoF graph knows that gravity
  is observed.
- **Fixes considered.** Gravity-prior edges in Open3D (an extra fixed node with rotation-only information); project
  the corrections onto yaw afterwards; a **4-DoF solver**.
- **Fix chosen.** The 4-DoF solver. The physics is right by construction (VINS-Mono does the same), every residual
  is one line of code, and it is easy to add our own terms (anchors, smoothness). Post-hoc projection would make the
  loops inconsistent again.

**Issue 3: on floor_only, ARKit drifted too far for ICP to close the loops.**

- **Symptom.** On floor_only, ICP from ARKit poses accepted 0 of 5 revisits:
  - jumps of 16–34 cm and 1.8–3.3°;
  - the start=end pair needed 41 cm / 12° and failed.
- **Root cause.**
  - Real heading drift: under ARKit, the per-fragment wall direction moves steadily from about −3.4° to +2.2°
    (`fig_corrections.png`).
  - The same apartment in with_ceiling shows no trend, so the walls really are parallel.
  - floor_only was filmed pointing at the floor (median depth 1.1 m), which is hard for VIO.
- **Fixes considered.** Global registration (FPFH + RANSAC, as in Open3D's system); a wider ICP basin; correct the
  heading first with Manhattan anchors, then search loops.
- **Fix chosen.** Anchors first, then 2 loop rounds. FPFH+RANSAC in repetitive rooms gives confident wrong matches
  that must then be caught. The anchor solve is cheap and deterministic.
- **Result.** 8 revisits accepted. Held-out revisit separation 13.95 → 1.85 cm.

**Issue 4: residual metric inflated by corridors.**

- **Symptom.** Some loops had a 15 cm "residual" after correction, and they were the degenerate ones (corridors).
- **Root cause.** The full-displacement metric counts motion *along* the corridor, which ICP cannot measure.
- **Fixes considered.** Drop degenerate loops from evaluation; project the error onto the information matrix;
  **project the displacement on the surface normals** (surface separation).
- **Fix chosen.** Surface separation. It is literally "how far apart the two copies of the wall are", the quantity
  that matters for a plan. Full displacement is still reported.

**Issue 5: the correction made with_ceiling's walls slightly blurrier.**

- **Symptom.** With σ_yaw = 0.6°/√m, wall sharpness on with_ceiling got *worse* than ARKit: 16.04 vs 15.49 mm.
  anchors-only was much worse (21.0 mm).
- **Root cause.** The yaw corrections zig-zagged by up to ±1.4° between neighbouring fragments.
  - Per-fragment Manhattan readings have about 0.5–0.7° of noise. with_ceiling's ARKit wall direction varies by
    0.71° without any trend, so that is measurement noise.
  - Loose odometry let each fragment follow its own noisy reading.
- **Evidence (sweep of first-order σ_yaw, wall MAD-σ in mm).**

  | σ_yaw (°/√m) | with_ceiling | floor_only |
  |---|---|---|
  | 0.1 | 14.81 | 9.87 |
  | 0.2 | 15.50 | 9.41 |
  | 0.3 | 15.69 | 9.33 |
  | 0.6 | 16.04 | 9.28 |

  The two captures want opposite things. floor_only's held-out residual at σ = 0.1 is also bad (2.80 cm).
- **Fixes considered.** A stiffer first-order σ (hurts floor_only); a larger Manhattan σ (barely helps: 15.35 mm);
  a **second-difference smoothness prior**.
- **Fix chosen.** The second-difference prior. Gyro-bias drift is a ramp, so it costs nothing, while jitter is
  penalised.
- **Result.** 15.03 / 9.39 mm in the sweep, and 15.13 / 9.51 mm in the final runs. Both beat ARKit. Yaw corrections
  became a smooth ±0.6° wave.
- **Trade-off, stated honestly.** On held-out loops, the no-smoothing settings score about 0.3 cm RMS better
  (`sigma_sweep.json`). Loop residuals are measured by ICP, which has its own ~1–1.7 cm RMS noise floor, so we
  prefer the metric that is independent of ICP.

**Issue 6: a vertical correction of 10 cm. Real or an ICP artefact?**

- **Symptom.** The 4-DoF solve moved the last 40 s of with_ceiling (the ceiling-facing second pass) up by 9.9 cm.
  Three local loops (fragments 52–58, 53–57, 53–58) claimed ARKit was 5.7–6.7 cm off vertically within about 15 s.
- **Suspected cause.** ICP sliding on a ceiling-only overlap.
- **Check.** The per-fragment ceiling level under ARKit, which is not in the solve.
  - The main ceiling reads 1.57–1.60 m at t ≈ 112–154 s but 1.465 m at t ≈ 205–214 s.
  - The lower soffit reads 0.95 m at t ≈ 172 s and slides steadily to 0.874 m at t ≈ 194 s.
  - So ARKit really sank by about 11 cm while the phone faced the ceiling.
- **Fix chosen.** Keep the correction, and add the ceiling level as a permanent independent check.
- **Result.** Ceiling MAD-σ 6.03 → 0.64 cm.
- **Why it matters.** Without the correction, rooms measured late in that walk would get ceiling heights about 10 cm
  wrong, against a 1.5 cm gate.

**Issue 7: the ceiling-spread metric was hiding the improvement.**

- **Symptom.** The ceiling std only fell from 4.42 to 3.89 cm, although the figure shows the drifted fragments being
  fixed.
- **Root cause.** A few fragments see a *different* ceiling (another room or a soffit, about 8–10 cm lower) that
  passes the ±15 cm gate. The standard deviation is dominated by those few points.
- **Fix chosen.** Use a robust MAD-σ: 6.03 → 0.64 cm. The plain std is not the right statistic when a few points
  belong to a different population.

**Issue 8: small residual non-determinism.**

- **Symptom.** Re-runs differ in the 7th significant digit, e.g. drift_sigma 0.0193667 vs 0.0193665. Occasionally a
  borderline loop flips accept/reject: identical re-runs moved one wall-sharpness cell by up to 0.25 mm (loops-only
  on with_ceiling: 17.68 vs 17.93 mm) and ON on floor_only by 0.02 mm.
- **Root cause.** Open3D's multithreaded ICP sums in a different order between runs.
- **Fix chosen.** None for now: the conclusions do not change. Every other random choice is seeded (`RNG_SEED = 0`).
  If bit-exact replay is required, run ICP single-threaded (`OMP_NUM_THREADS=1`) at about 3x the runtime.
- **Consequence.** Differences below ~0.3 mm in wall sharpness should not be read as real.

**Issue 9: rename bug while refactoring.**

- **Symptom.** A global `_yaw` → `yaw_angle` rename also turned `manhattan_yaw(` into `manhattanyaw_angle(`, which
  crashed the run.
- **Fix chosen.** Fixed by hand. Lesson: use an IDE rename, not `str.replace`.

---

## 5. Results on the captures

Run with: `python scripts/drift_ablation.py <capture_dir> [--sweep]`. Runtime of `correct_drift` alone:

- with_ceiling: 64 s (66 fragments, 80 loop ICPs);
- floor_only: 13 s;
- single_room: 10 s (CPU).

The full ablation with variants takes 4–6 min per capture.

### 5.1 The on/off ablation (from `ablation.json`)

Column definitions:

- **revisit sep**: median surface separation of accepted revisit loops, in-sample.
- **floor / ceil**: MAD-σ of per-fragment levels. The ceiling column is independent.
- **wall MAD**: raw-wall-point spread. Independent.
- **footprint**: filled floor area.

**single_scan_with_ceiling** (walked twice, 215 s, 100 m):

| variant | revisit sep cm | yaw sd ° | floor cm | **ceil cm** | **wall MAD mm** | <1 cm | footprint m² | bbox m |
|---|---|---|---|---|---|---|---|---|
| OFF (ARKit as-is) | 2.89 | 0.71 | 0.72 | 6.03 | 15.49 | 48.6% | 41.8 | 10.80 x 12.94 |
| **ON (ours)** | **1.21** | 0.50 | 0.39 | **0.64** | **15.13** | **49.3%** | 41.6 | 10.84 x 12.87 |
| loops only | 1.32 | 0.54 | 1.21 | 0.64 | 17.93 | 44.2% | 41.6 | 10.83 x 12.87 |
| anchors only | 2.43 | 0.55 | 0.30 | 6.19 | 18.46 | 43.7% | 41.9 | 10.82 x 12.91 |
| Open3D 6-DoF baseline | 1.10 | 0.61 | 2.00 | 2.81 | 16.74 | 46.0% | 40.8 | 10.83 x 12.84 |

**single_scan_floor_only** (115 s, 54 m, starts and ends at the same spot):

| variant | revisit sep cm | yaw sd ° | floor cm | wall MAD mm | <1 cm | footprint m² | bbox m |
|---|---|---|---|---|---|---|---|
| OFF (ARKit as-is) | 13.95 | 1.64 | 1.27 | 11.15 | 59.6% | 43.1 | 10.30 x 10.87 |
| **ON (ours)** | **1.44** | 0.60 | 0.38 | **9.51** | **64.1%** | 43.0 | 10.39 x 10.98 |
| loops only | 1.54 | 0.71 | 1.61 | 9.62 | 63.7% | 43.2 | 10.40 x 10.97 |
| anchors only | 9.10 | 0.59 | 0.32 | 10.15 | 61.1% | 43.4 | 10.31 x 10.91 |
| Open3D 6-DoF baseline | 4.44 | 1.47 | 1.69 | 10.42 | 61.9% | 40.6 | 10.37 x 10.92 |

**single_room** (37 s, 14.5 m; only 3 loops, so treat these numbers as indicative):

| variant | revisit sep cm | floor cm | wall MAD mm | <1 cm | footprint m² | bbox m |
|---|---|---|---|---|---|---|
| OFF | 4.98 | 2.02 | 12.06 | 57.2% | 13.4 | 6.64 x 6.10 |
| **ON** | 0.92 | 0.22 | **11.10** | **60.7%** | 13.4 | 6.63 x 6.14 |
| Open3D 6-DoF baseline | 2.15 | 0.80 | 11.61 | 58.6% | 12.9 | 6.64 x 6.17 |

**Reading the tables.**

- ON gives the sharpest walls on all three captures. It is the only variant that beats ARKit on wall sharpness on
  with_ceiling, where ARKit was already good.
- Loops fix the vertical: the ceiling column. Anchors fix the heading. You need both.
- Footprint and bbox move by less than 1% (0.1–0.4 m² and ≤ 11 cm on a 10–13 m bbox). Drift here mainly *blurs and
  doubles* walls rather than changing the overall envelope. The plan modules feel it as wall offsets and doubled
  wall candidates.

### 5.2 Held-out loop residuals (5-fold, `drift_report.json` → `heldout_loop_residuals`)

Surface separation, median / RMS in cm:

| capture | loops | ARKit | **corrected** | ICP noise floor (consecutive fragments) |
|---|---|---|---|---|
| with_ceiling | 35 | 2.99 / 3.39 | **1.64 / 1.94** | 0.74 / 1.68 |
| floor_only | 14 | 5.75 / 10.87 | **1.72 / 2.48** | 0.73 / 1.02 |
| single_room | 3 | 4.73 / 4.58 | **2.01 / 1.93** | 1.04 / 1.28 |

More held-out detail:

- **Revisits only.** with_ceiling 2.89 → 1.69 cm (n = 25); floor_only 13.95 → 1.85 cm (n = 8). Rotation: floor_only
  2.97° → 0.51° median.
- **Vertical RMS.** with_ceiling 2.86 → 1.27 cm.
- After correction, residuals are within about 1 cm of the ICP noise floor, so what is left is at the limit of what
  this measurement can see.

### 5.3 Size of the drift ARKit accumulated (`frame_correction`)

| capture | max position correction | median | max yaw | max vertical | tilt changed |
|---|---|---|---|---|---|
| with_ceiling | 13.9 cm | 5.3 cm | 0.67° | 9.9 cm | 0 |
| floor_only | 65.5 cm | 23.0 cm | 5.86° | 3.0 cm | 0 |
| single_room | 14.1 cm | 5.5 cm | 1.53° | 3.3 cm | 0 |

### 5.4 Repeatability across captures (independent; eval module's registration)

Registration is rigid 2-D, whole capture to whole capture. Source files: `outputs/drift/registration_{off,on}/summary.json`.

| pair | inliers OFF → ON | local tile disagreement median (p90) OFF → ON | local yaw OFF → ON |
|---|---|---|---|
| floor_only ↔ with_ceiling (same apartment) | 0.40 → **0.81** | 5.2 (15.0) → **2.0 (4.1) cm** | 0.85° → **0.26°** |
| single_room ↔ floor_only | 0.57 → 0.80 | 4.2 (9.5) → 2.3 (7.6) cm | 2.14° → 0.74° |
| single_room ↔ with_ceiling | 0.73 → 0.84 | 1.9 (6.7) → 3.6 (4.7) cm | 0.41° → 0.69° |

**Three-capture cycle** (A←B←C vs A←C): 3.70° / 70.7 cm max displacement (OFF) → **0.17° / 5.7 cm (ON)**.

- This bears directly on the Part 2 repeatability gate: drift-corrected captures of the same apartment agree far
  better.
- One honest regression: single_room ↔ with_ceiling median tile disagreement rose from 1.9 to 3.6 cm, while its p90
  fell from 6.7 to 4.7 cm. Only 5–7 tiles overlap, so this is noisy, but it is not proven better.

### 5.5 `drift_sigma` for other modules (`drift_report.json` → `drift_sigma`)

**Definition.** RMS of held-out surface separation, i.e. how far a surface observed at two different times is offset
after correction.

| capture | drift_sigma_m (ON) | vertical (ON) | ARKit (OFF) |
|---|---|---|---|
| with_ceiling | 0.019 | 0.013 | 0.034 |
| floor_only | 0.025 | 0.006 | 0.109 |
| single_room | 0.019 (3 loops) | – | 0.046 |

**Recommendation.**

- `drift_sigma_m = 0.02` for any distance between surfaces observed at **different times**: stitched room-to-room
  offsets, and a room scanned on two passes.
- `0.01` (the current placeholder) for a wall observed within one sweep of a room, where ARKit is locally exact.
  This is bounded above by the ICP noise floor of 0.7–1.0 cm median.
- Vertical: `0.013 m` across passes.
- These values include ICP's own noise, so they are conservative.

### 5.6 Figures to look at

- `outputs/drift/single_scan_floor_only__1a8384c3f6/fig_side_by_side.png`
  - OFF: the corridor wall at v ≈ 3.6–3.8 is doubled, and the bottom wall is tilted.
  - ON: both are single and straight.
- `outputs/drift/single_scan_floor_only__1a8384c3f6/fig_corrections.png`
  - ARKit's heading drift as a steady ramp (wall direction −3.4° → +2.2°), flattened to about ±0.5° by the correction.
- `outputs/drift/single_scan_with_ceiling__c7d28f72c6/fig_corrections.png`
  - Bottom-left: ARKit ceiling levels sink by about 6 cm at the end; corrected, they sit near 0. This is the
    independent check.
  - Bottom-right: the 9.9 cm vertical correction.
- `outputs/drift/single_scan_with_ceiling__c7d28f72c6/fig_overlay.png`
  - Zoom 2: a wall doubled under ARKit (orange) that collapses onto a single line when corrected.
- `outputs/drift/*/fig_loops.png`
  - Per-loop separation, in-sample and held-out.
- `outputs/drift/registration_on/*_overlay.png` vs `registration_off/`
  - Cross-capture overlays.

---

## 6. Limitations, failure modes and what to do next

**Mirrors.**

- A mirror returns a "virtual room" behind the glass. Seen from two places it is geometrically inconsistent.
- ARKit marks most mirror returns as low confidence (D-006 drops them).
- Whatever survives can make ICP lock onto phantom geometry. Guards:
  - the verification gates (point-to-plane RMSE, jump limits);
  - the Cauchy weights.
- Plane anchors are not fooled much, because a mirrored wall is still Manhattan.
- Next: mask mirror candidates (a planar hole bordered by a frame, with geometry "behind" a wall) before building
  fragments.

**Glass (shower screen, dark window).**

- Mostly no LiDAR returns, so these surfaces contribute nothing, which is safe.
- A fragment looking mostly through glass has little geometry. It then fails the fitness gate, and odometry carries
  it.

**Wet-look / specular floors.**

- Specular floors return sparse or no high-confidence depth at grazing angles. Floor anchors then disappear for those
  fragments, and the floor gate ignores outliers.
- Vertical drift in such rooms relies on loops only.
- Risk: a glossy floor reflecting the ceiling produces a "floor" below the true floor. The floor gate's
  "lowest-of-strong-levels" logic could pick it. It was not observed here.

**Low light.**

- This is where **VIO drifts most**: fewer visual features, more motion blur. It is exactly when this module matters.
- The LiDAR is active illumination, so fragments and ICP loops still work in the dark. This is the main robustness
  argument for depth-based loop closure over image-based relocalisation.
- Manhattan anchors also work in the dark, because they come from depth normals.

**Clutter.**

- Furniture lowers the per-fragment Manhattan score. Fragments below 0.5 lose their heading prior.
- If fewer than half of the fragments qualify, anchors switch off for the whole capture, and we fall back to
  loops-only (see the "loops only" rows).
- Clutter helps ICP: more 3-D structure, less degeneracy.

**Non-Manhattan or curved buildings.** Anchors switch off automatically, and we are back to loops-only, still 4-DoF.

**Split-level floors, sunken bathrooms.**

- The floor prior assumes one floor level.
- Steps over 10 cm are gated out. A 1–3 cm sunken bathroom is pulled up slightly, at most about 1σ = 1 cm, which is
  noticeable against the 1.5 cm ceiling gate.
- Next: per-room floor levels once rooms exist (a second pass after segmentation), or a floor-level variable per
  connected floor region.

**Long single-pass captures with no revisits.**

- No loops means only anchors, and translation drift along the walk cannot be corrected.
- The residual drift is then unmeasured. We should report ARKit's drift rate measured here (floor_only: about 65 cm /
  5.9° over 54 m) as the interval term instead.
- Part 1 protocol advice: "end where you started, and re-enter the first room".

**ARKit relocalisation jumps.** These were not present here: only 50 ms gaps. A jump would violate the stiff
odometry assumption. Next: detect frame-to-frame jumps and cut the odometry chain there, letting loops reconnect it.

**Corridors and other degenerate geometry.** Handled by point-to-plane information, but a long featureless corridor
still leaves drift *along* the corridor uncorrected.

**Fragment size not swept** (DR-4). Next: sweep 1.0 / 1.5 / 2.5 m against wall sharpness.

**Spatially uniform `drift_sigma`.** Next: per-fragment marginal covariances from the solver (GTSAM gives them
directly). Each wall would then get its own drift term: small near the start, larger far from any loop.

**Real plane-anchored correction.** Make each physical wall a shared plane variable seen by many fragments
(plane-landmark SLAM). That replaces "Manhattan direction" with "the same wall", and fixes translation as well as
heading.

---

## Requested changes to shared code

1. **`floorplan/pipeline/scene.py` → `build_scene`: run drift correction by default.**
   - Order: select keyframes on ARKit poses, then `correct_drift(cap, kf, cap.T_wc, cfg)`, then fuse with the
     corrected poses, **reusing the same `kf`**.
   - Save `T_wc_arkit` and the drift report summary (minus `_state`) into the scene and `scene_info.json`.
   - Currently `build_scene` re-selects keyframes from the given `T_wc`. That is harmless, but it makes OFF and ON
     use slightly different keyframes.
2. **`floorplan/config.py`:** add `drift_correction: bool = True`, and optionally the `DriftParams` fields, or a path
   to them, so that one config drives everything.
3. **`scripts/prepare_scene.py`:** add a `--no-drift` flag for the ablation.
4. **Plan modules (`plan/alpha/params.py`, `plan/beta/params.py`):**
   - Keep `drift_sigma_m = 0.01` for walls seen within one sweep.
   - Use `0.02` for cross-pass and room-to-room (stitched) distances.
   - Use `0.013` vertical when floor and ceiling of a room come from different passes.
   - Better: read `outputs/drift/<capture>/drift_report.json → drift_sigma`.
5. **`docs/DECISIONS.md`:** add a D-entry summarising DR-1/2/8/11.
6. **`docs/COMPLIANCE.md` G4:** point to `outputs/drift/*/fig_overlay.png` and `ablation.json`.

No packages were installed.

---

# Part S (stability): the drift solve is now deterministic

Added 2026-10-04 by the stability task. Nothing above is changed; this part records why `drift.py` changed and the
evidence. Evidence folder: `outputs/stability/drift_det/` (probe: `outputs/stability/drift_determinism.py`,
solver probe: `outputs/stability/drift_solver_sensitivity.py`).

## S.1 Problem

The LiDAR benchmark (`docs/modules/benchmark_lidar_sample.md` §4.1) ran the same one-command on floor_only three
times: footprints 61.90 / 59.37 / 61.90 m², the largest room 13.99 vs 11.46 m², while the drift-corrected
trajectories differed by only ~0.06 mm. with_ceiling run twice gave 70 vs 74 walls. A pipeline that gives two
answers to one input cannot be "regenerable from raw inputs" (Deliverable 4).

## S.2 Root cause (with evidence)

Probe: `correct_drift` three times in one process on floor_only, saving the corrected `T_wc`, every fragment cloud,
every ICP edge transform, accept flag and robust weight (`drift_determinism.py`).

| Stage | Run-to-run difference (before) |
|---|---|
| Fragment clouds (voxel grid, normals: Open3D) | identical (sums of points and normals bit-equal) |
| Loop candidates and accept/reject decisions | identical (67 edges, 51 accepted, 0 flips) |
| ICP edge transforms (Open3D `registration_icp`) | **differ by 2–4e-15** |
| Robust IRLS weights | differ by up to 4.3e-5 |
| Corrected poses (all frames) | **differ by 0.005–0.023 mm** (benchmark: 0.06 mm) |

- **Source 1, the only nondeterminism: Open3D's ICP.** Open3D 0.20 parallelises with oneTBB. ICP accumulates the
  point-to-plane normal equations (JᵀJ, Jᵀr) with a parallel reduction whose split/join order depends on thread
  scheduling. Floating-point addition is not associative, so the last bits differ from run to run. Proof: with
  `o3d.utility.set_max_threads(1)` for the whole process, three runs are **bit-identical** (`before/o3d1thread_*`).
- **Source 2, an amplifier, not a source: the solver's stopping tolerance.** Re-solving the final graph with every
  edge translation perturbed by N(0, ε) moves the poses by 1.3–2.3e-5 m for **every** ε from 1e-15 to 1e-6
  (`solver_sensitivity_before.txt`). The poses are only defined to ~0.02 mm by scipy's default tolerances
  (ftol = xtol = gtol = 1e-8) and the IRLS stop rule; any input change, however small, lands somewhere in that
  0.02 mm ball. This is harmless in itself (0.02 mm is 100x below the registration floor of 2 cm), but it is why a
  1e-15 ICP difference became a 1e-5 m pose difference.
- Not sources (checked): cKDTree queries (single-threaded), scipy `least_squares` (deterministic for identical
  input), loop-candidate order (nested `range` loops, deterministic), random sampling (`edge_residual` uses
  `default_rng(0)` and is evaluation only).
- Then the plan extractor amplified 0.06 mm into 2.5 m² (one arrangement cell sitting at the 50% inclusion
  threshold). That half is fixed in `docs/modules/plan_beta.md`, Part v3.

## S.3 Options

| Option | Deterministic? | Cost | Verdict |
|---|---|---|---|
| (a) `set_max_threads(1)` for the whole process | yes | drift 9–12 s → 26–29 s on floor_only; slows TSDF and normals too | rejected: 3x slower for no reason outside ICP |
| (b) single thread only around `registration_icp` | yes | 19–23 s | better, still ~2x |
| (c) **(b) + run the loop-candidate ICPs concurrently in Python threads** (Open3D releases the GIL; `map` keeps order) | **yes** | **13.5–17 s** under the same load | **chosen** |
| (d) re-implement ICP in numpy with a fixed summation order | yes | an hour of work and a new ICP to validate | not needed |
| (e) tighten scipy tolerances to 1e-12 so the solve is unique to ~1e-12 | removes the amplifier only | the solve did not finish in 10 min on floor_only (stopped) | rejected; with (c) the input is bit-identical, so the amplifier has nothing to amplify |

## S.4 Chosen fix

- `deterministic_open3d()`: a context manager that sets Open3D's thread limit to 1 and restores the previous limit.
- `loop_edges` first collects the candidate pairs (same order as before), then runs `register` + `verify_loop` for
  all of them inside `deterministic_open3d()` with a `ThreadPoolExecutor(icp_workers)`; `odometry_edges` wraps its
  (baseline-only) ICP the same way. Each ICP is single-threaded, so its summation order is fixed; the result does
  not depend on `icp_workers` (proved below).
- New parameter `DriftParams.icp_workers = min(8, cpu_count)`. No other parameter or behaviour changed: the same
  edges are proposed, verified and solved.

## S.5 Before / after

| floor_only, `correct_drift` x3 | Before | After |
|---|---|---|
| Max difference of corrected poses (any frame) | 0.0053 mm, 0.0226 mm | **0.0 (bit-identical)** |
| Max difference of ICP edge transforms | 2.2e-15, 4.4e-15 | **0.0** |
| Robust-weight difference | 8e-6, 4.3e-5 | **0.0** |
| Loop accept flips | 0 | 0 |
| Across processes and settings (whole-process 1 thread vs ICP-only 1 thread vs pool of 8) | – | **bit-identical** (`compare` of `before/o3d1thread_run0`, `after/icp1thread_run0`, `after/pool8_run{0,2}`) |
| Drift runtime (shared machine, load ~20) | 9.2 / 11.7 / 9.5 s | 16.9 / 14.1 / 13.5 s |

| with_ceiling, `correct_drift` x3 (after) | |
|---|---|
| Max difference of corrected poses | **0.0 (bit-identical)**, 147 edges, 100 accepted in every run |
| Runtime (machine load 40–55 during these runs) | 98 / 74 / 78 s (benchmark's quiet-machine "before": 34 s; its loaded first pass: 104 s) |

End to end (`run_capture.py --no-damage` twice per capture, `outputs/stability/e2e/`): `T_wc` and every loop edge
in the drift report are bit-identical between runs; the plans are identical (floor_only 61.9009 m², 70 walls;
with_ceiling 62.4985 m², 70 walls), against 61.90 / 59.37 m² and 70 / 74 walls before. The drift stage took
25.2 / 24.7 s (floor_only) and 58.6 / 58.3 s (with_ceiling) with four pipelines running at once.

Runtime cost: about +40% on floor_only at equal load. Timings on this shared machine are noisy; a quiet-machine
re-time of with_ceiling is still to do.

Unit test: `tests/test_stability.py::test_icp_bit_reproducible_in_parallel_pool` (8 ICPs from 4 Python threads
give the bit-identical transform) and `test_deterministic_open3d_restores_thread_limit`.
