# Module `eval`: registration, matching, repeatability, gates, benchmark

Files:
- `floorplan/eval/register.py`: rigid 2D registration between captures, plus loop and local-disagreement checks.
- `floorplan/eval/match.py`: matches rooms, walls and openings between two plans.
- `floorplan/eval/repeatability.py`: per-wall, per-room and per-opening comparison of two captures.
- `floorplan/eval/gates.py`: every Part 2 gate as data, plus pass/fail logic and interval calibration.
- `scripts/register_scenes.py`: registers every pair of prepared scenes and writes figures, loop checks and tiles.
- `scripts/benchmark.py`: discovers every `plan.json`, then writes `outputs/benchmark/report.md`, `gates.csv`,
  `repeatability_walls.csv` and `benchmark.json`.
- `tests/eval/`: a synthetic plan fixture, known-answer tests, and an end-to-end benchmark smoke test (16 tests).

Evidence lives in `outputs/eval/`:
- `registration/`: the original ARKit scenes.
- `registration_drift_on/`: the drift-corrected scenes.
- `ablations/`: the ablation scripts and their JSON.

How to run (from the repo root):

```bash
PY=../.venv/bin/python
PYTHONPATH=. $PY scripts/register_scenes.py --verify-3d            # ~1 min, 3 scene pairs
PYTHONPATH=. $PY scripts/register_scenes.py --scenes 'outputs/drift/*/scene_on/scene.npz' \
              --out outputs/eval/registration_drift_on             # same, drift correction ON
PYTHONPATH=. $PY scripts/benchmark.py                               # all plans found under outputs/
PYTHONPATH=. $PY -m pytest tests/eval -q                            # 16 tests, ~10 s
```

---

## 1. Purpose and where it sits in the pipeline

The pipeline produces plans: rooms, walls, openings and ceiling heights, each with a 95% interval. This module answers
the question **"how good are those plans?"** in the terms the case study scores, which are the Part 2 gates.

```
capture -> scene (recon) -> plan (plan_alpha / plan_beta / video / photo) -> plan.json
                                                                                |
          scenes ---> register.py (capture-to-capture transform) ---------------+
                                                                                v
                                    match.py -> repeatability.py -> gates.py -> benchmark.py -> report.md, gates.csv
```

Why it matters for the score:
- **Part 2 (gates).** Every gate is computed from files on disk, the same way every run.
- **Part 4 (fix loop, 25%).** It needs "the single worst-performing gate with the failing number", plus a before run
  and an after run that can be regenerated. The benchmark's "Failing gates" section is that list, and running the
  benchmark before and after a fix gives both runs.
- **Calibration** is "scored at every tier", so it is a gate here too, not an afterthought.

There is **no ground truth** for the sample data. The module is therefore built around three ideas:
1. **Repeatability needs no ground truth (GT).** The two whole-apartment captures (`floor_only`, `with_ceiling`) and
   the `single_room` capture show the same rooms. Any registered pair of captures is a repeat pair for the rooms
   they share.
2. **Reference-based evaluation.** Video and photo tiers are compared with the LiDAR-tier plan of the same capture.
   Every such row is labelled `lidar_reference`, because LiDAR is our own estimate and not truth.
3. **"Not verifiable" is a real status.** Gates that need GT and have none say so, with the reason. A GT plan in the
   same JSON schema can be dropped in later (`--gt-dir`), and the same code then scores against it with basis
   `ground_truth`.

---

## 2. How it works, step by step

### 2.1 Registration (`register.py`): put two captures in one frame

Each capture starts wherever the phone was switched on and faces an arbitrary heading. Before we can say "this wall
in capture A is that wall in capture B", we need the rotation and translation between them.

1. **Evidence extraction** (`scene_evidence`). We keep only two kinds of TSDF surface point:
   - **Wall points:** vertical surfaces (|n_y| < 0.3) between 0.3 m and 2.0 m above the floor.
     - Below 0.3 m: skirting, floor clutter and furniture legs.
     - Above 2.0 m: cabinets, ceiling edges, lamps.
     - These points are projected to the plan (u, v) = (x, z), and each keeps its 2D normal.
   - **Floor points:** up-facing points within 5 cm of the floor level.
   - *Without this step:* tables, sofas and the floor itself would dominate the matching.
   - It also explains why we register on *geometry* and not on the plans. If registration used the plans being
     evaluated, a wrong plan could drag the registration towards itself and hide its own error.
2. **Global search** (`global_search`).
   - Both scenes are already Manhattan-aligned (walls along x and z), so the unknown rotation is k·90° plus a small
     residual. For each quadrant k = 0..3, and each fine yaw in ±5° (step 0.5°), we:
     - rotate B's wall points;
     - rasterise them on a 5 cm grid into **4 channels by normal direction** (+u, −u, +v, −v);
     - compute the cross-correlation with A's rasters in one FFT, whose peak gives the best translation for that yaw.
   - The best (quadrant, yaw, translation) is then refined on a 2 cm grid (±0.5°, step 0.1°).
   - *Without the normal split:* the two faces of a 10 cm partition wall look identical, so the correlation can lock
     onto the wrong face.
   - *Without a global search:* ICP alone converges to whichever local minimum is nearest the start. The start is
     arbitrary here.
3. **Local refinement** (`icp_point_to_line`).
   - 2D point-to-line ICP: every B wall point is pulled towards the *line* through its nearest A wall point.
     - The correspondence must face the same way (normals within ~37°).
     - The search radius shrinks from 15 cm to 3 cm.
   - Huber weights (1 cm) stop clutter and doors that moved from dominating the fit.
   - This takes the 2 cm-grid answer to sub-centimetre residuals.
4. **Diagnostics** (in `Registration`):
   - RMS and median residual of inlier wall points.
   - Inlier fraction.
   - Floor-area overlap: IoU, and how much of A and of B is covered.
   - **Distinctiveness:** the best correlation divided by the best correlation in any other 90° quadrant. A value
     near 1 means a symmetric layout and an ambiguous answer; the benchmark rejects anything below 1.2.
   - **Yaw-at-edge flag:** set when the optimum sits on the edge of the yaw window.
   - **Constraint ratio:** the smallest over the largest eigenvalue of the translation block. Near 0 means
     corridor-like geometry that can slide along its walls.
   - Hessian sigmas (see I-3: optimistic, not used for decisions).
5. **Consistency checks** (`cycle_error`, `local_disagreement`):
   - **Loop check.** With three captures, A←B←C must equal A←C. A residual means at least one capture is internally
     bent (drift), or one registration is wrong.
   - **Tiles.** B is cut into 3 m tiles, and each tile is re-registered alone, starting from the global answer. Tiles
     that want to move show where the two captures disagree locally. Tiles whose walls all run one way (constraint
     ratio ≤ 0.1) are flagged, because their motion along the wall is meaningless.

### 2.2 Matching (`match.py`): which element is which

1. **Transform B into A's frame** with the registration (`transform_plan`). Lengths are unchanged.
2. **Rooms:** polygon IoU between every pair. A one-to-one assignment by the **Hungarian algorithm** maximises total
   IoU, and pairs with IoU < 0.3 are dropped.
   - *Why Hungarian:* greedy matching can give one big room two partners. Hungarian guarantees one-to-one.
3. **Local correction per matched room** (`local_room_transform`).
   - The global rigid fit still leaves parts of two whole-apartment captures 5–15 cm apart, because of drift (I-2).
   - Each matched room pair is therefore re-aligned on its own walls with the same 2D ICP. The correction is kept
     only if at least 50% of the room's wall samples find a partner.
   - It shifts and turns a room but never changes a length, so it cannot hide a length error.
4. **Walls, inside each matched room pair.** Candidates must:
   - be parallel within 10°;
   - face the same way (normal cosine ≥ 0.5);
   - have a line-to-line offset ≤ 15 cm;
   - overlap by ≥ 50% of the shorter wall.

   The Hungarian cost is offset/15 cm + (1 − overlap).
5. **Openings:** centre distance ≤ 0.5 m, assigned by Hungarian on distance. The kind (door / window / passage) is
   reported but not required to agree: a kind error is a different failure from a detection error.
6. **Unmatched elements are returned explicitly**, because they are failures too: a missed or phantom opening, or a
   wall that exists in only one capture.

`as_plan_dict` accepts a `Plan` dataclass, a dict or a path. It normalises both interval encodings: the dataclass
`{lo, hi}` and the published schema's `{ci95: [lo, hi]}` (I-8).

### 2.3 Repeatability (`repeatability.py`)

- **Per matched wall:** Δ = L_B − L_A, compared with the gate tolerance max(1 cm, 0.5% · mean length).
  - Status is pass or fail; it is `not_verifiable` when a length was not observed, and `unmatched` when a wall of a
    matched room has no partner.
  - A **z-score**, Δ / sqrt(σ_A² + σ_B²), is computed for interval calibration (2.4).
- **Per matched room:** floor-area Δ (absolute and relative), and ceiling heights with their spread.
  - `classify_ceiling` names the outcome the way the gate demands: `unrepeatable`, `repeatable_but_biased` (needs
    GT), `repeatable_bias_unknown`, `pass` or `not_verifiable`.
- **Per matched opening:** width Δ.
- **More than two captures:** `ceiling_groups` links (capture, room) nodes through room matches (union-find), so the
  spread "across captures" covers every capture of that room, not only pairs.
- **Output:** a markdown table (`to_markdown`) and CSV (`walls_csv`).

### 2.4 Gates (`gates.py`)

- **Gate definitions:** `GATES` is the Part 2 table as data: id, row, requirement text, tiers, whether GT is needed,
  thresholds. The report prints it, so the encoded rules can be checked against the brief.
- **Result format:** every gate function returns a `GateResult` with these fields:
  - `status`: pass | fail | not_verifiable | not_applicable.
  - `basis`: ground_truth | lidar_reference | repeat_pair | declaration | self_check | none.
  - `value`, `threshold`, `n`, and a human-readable `detail`.

| Gate | How it is computed |
|---|---|
| `opening_width` | hits (matched with \|Δw\| ≤ 2 cm) / (matched + missed + phantom) ≥ 0.85 |
| `ceiling_abs` | every matched room with a reference ceiling within 1.5 cm. A room whose tested plan did not observe the ceiling counts against it. |
| `ceiling_spread` | worst group spread ≤ 1 cm. The detail says *unrepeatable*, or *repeatable; bias not verifiable without GT*. |
| `repeatability` | every wall of every matched room within max(1 cm, 0.5%); unmatched walls fail |
| `drift` | the plan declares a method that is not "as-is" **and** an on/off footprint ablation exists. "As-is" is an automatic fail, as the brief says. |
| `photo_stitch.*` | `one_plan` (≥ 2 rooms, connected adjacency graph), `no_overlap` (no pair overlapping > 1% of the smaller room), `adjacency` (precision = recall = 1 vs the reference), `footprint` (within ±8% **and** the reference inside the 95% interval) |
| `wall_length_photo/video` | every matched wall within ±8% / ±3% **and** the intervals not overconfident; wall recall reported |
| `calibration` | coverage of the 95% intervals over walls, areas, ceilings and openings (below) |
| `round1_lidar` | Round 1 thresholds are not in the Round 2 brief: always `not_verifiable`, never invented |

**Interval calibration** (`interval_calibration`):
- *Strict coverage:* the reference lies inside [lo, hi]. This is correct when the reference is exact (GT).
- *Noise-aware coverage:* |value − ref| ≤ 1.96·sqrt(σ_pred² + σ_ref²). This is correct when the reference has its own
  error: the LiDAR reference, or the other capture of a repeat pair.
- *Status:*
  - `overconfident` if the Wilson 95% upper bound on coverage is below 0.95, **or** RMS z > 2 (with n ≥ 10).
    Intervals are too narrow, which is the "confident garbage" the brief penalises.
  - `underconfident` if RMS z < 0.5 (with n ≥ 10): intervals more than 2x too wide, so useless.
  - `calibrated` otherwise; `insufficient_n` below 5 samples.
- **Repeat-pair calibration (needs no GT):** across repeat pairs, |z| ≤ 1.96 should hold for about 95% of walls.
  This checks the *random* part of the error only; a bias shared by both captures is invisible to it.

### 2.5 Benchmark (`scripts/benchmark.py`)

1. **Discovery.**
   - Finds every `outputs/<producer>/<capture>/plan.json`.
   - The tier comes from the plan's `tier` field.
   - The capture is mapped to a prepared scene by name prefix.
   - Plans whose `meta.synthetic` is true (e.g. `outputs/export/stress`) are skipped unless `--include-synthetic`
     is given.
   - Unreadable files are listed, not fatal.
2. **Repeatability.**
   - Compares plans of the **same producer and tier** only; different producers are different methods.
   - Any two plans of different captures form a candidate pair. There are no hard-coded capture names: if they share
     rooms, they are a repeat pair.
   - Two candidate transforms are tried: the scene registration (independent of plans) and a plan-to-plan
     registration. The one that matches more walls wins, and the report states which (I-11).
3. **Reference-based gates.**
   - Each video or photo plan is compared with every LiDAR plan of the same capture: plan-to-plan registration, then
     matching, then the gates.
   - With `--gt-dir`, any plan is also compared with `<gt-dir>/<capture>/plan.json` (basis `ground_truth`).
   - LiDAR plans without GT get explicit `not_verifiable` rows.
4. **Drift.**
   - The plan's own `meta.drift` declaration decides the drift gate.
   - The drift module's `outputs/drift/<capture>/ablation.json` is quoted when the plan does not declare its handling.
   - The cross-capture loop residual is attached as independent evidence.
5. **Report.**
   - Lists inputs, the registration table, loop checks, all gates, **failing gates (fix-loop candidates)**,
     repeat-pair calibration, the full repeatability tables, notes on everything skipped, and the gate definitions.

---

## 3. Decisions

Each entry gives the options, the choice, the reason and the evidence.

**E-1. Rigid registration, never similarity (no scale).**
- *Options:* rigid (3 DoF in 2D); similarity (adds scale); non-rigid.
- *Choice:* rigid.
- *Why:* the video and photo gates measure scale error (±3% / ±8%). A similarity transform would absorb a 3% scale
  error and report a perfect plan. Non-rigid would also warp lengths.
- *Evidence:* the synthetic test `test_video_wall_gate_detects_scale_error` fails a plan scaled by 1.04, passes one
  scaled by 1.02, and fails the 1.02 plan when its intervals are too narrow.

**E-2. Register scenes (geometry), not plans; plan-to-plan registration only as a fallback.**
- *Why:* the transform must not depend on the thing being judged.
- *Fallback:* needed when no scene exists (photo tier), or when a plan was built in another frame (I-11). Even then,
  the plan registration uses only wall *positions*, never lengths.

**E-3. Global search with Manhattan quadrants + FFT cross-correlation, then ICP.**
- *Options:*
  - (a) ICP from identity: fails, because headings are arbitrary.
  - (b) Feature matching (FPFH + RANSAC): random, needs tuning, and is weak on flat walls.
  - (c) Exhaustive search over rotation, with translation from FFT correlation.
- *Choice:* (c).
- *Why:* the Manhattan alignment shrinks the rotation search to 4 quadrants × a few degrees. FFT correlation tests
  *every* translation at once, so there is no local minimum in translation. It is deterministic, with no random
  sampling, which matters for regenerable before/after runs.
- *Evidence:* all three pairs pick the correct quadrant with distinctiveness 1.98 / 2.32 / 2.59 (> 1.2). Runtime is
  7–14 s per pair (13–27 s while other agents shared the CPU).

**E-4. Four normal-direction channels instead of one raster.**
- *Ablation* (`outputs/eval/ablations/register_ablation.json`): distinctiveness, single raster → split:

  | pair | single raster | split |
  |---|---|---|
  | room→floor | 1.51 | 1.98 |
  | room→ceil | 1.74 | 2.32 |
  | floor→ceil | 1.92 | 2.59 |

  That is +31–35% margin against picking the wrong quadrant. On these captures the final yaw after ICP is the same
  (within 0.2°), so the split buys robustness, not accuracy.

**E-5. Cell weight = sqrt(point count).**
- *Why:* a TSDF has about one point per 2 cm voxel, so the count in a cell is about the height of the vertical
  surface there. Full-height walls should outweigh furniture sides, but one very tall wall should not drown the rest.
- *Honesty note:* not ablated; it is a reasoned default.

**E-6. Yaw window ±5°, with a flag when the optimum is on the edge.**
- *Options:* ±3° (the first version); ±5°; a full 360° search.
- *Evidence:* with ±3° the room→floor correlation rises monotonically to the −3° edge (figure
  `registration/…single_room…_yaw.png`). The true answer is −3.73° off the quadrant (I-1).
- *Choice:* ±5°. It matches `align.py`'s 5° Manhattan tolerance, costs 16 more FFTs per quadrant, and the edge flag
  catches anything beyond it. A full 360° search is the right next step for non-Manhattan buildings (section 6).

**E-7. 2D point-to-line ICP (own code) rather than Open3D 3D point-to-plane ICP.**
- *Why:* we want exactly yaw + translation. In 3D the floor dominates the point count and couples tilt into the
  answer.
- *Why our own code:* the 2D version is about 25 lines that we can explain, and it exposes the residuals and Jacobian
  for diagnostics.
- *Cross-check:* Open3D 3D point-to-plane ICP, started from our answer, moves it by only **0.21–0.33° and ≤ 2.0 cm**
  on all three pairs (`summary.json → verify_3d`), and by 0.10–0.19° and ≤ 2 cm on the drift-corrected scenes.
  Part of that small change is real tilt between captures, which a 2D fit cannot express.

**E-8. Huber loss at 1 cm in ICP.**
- *Why:* LiDAR noise is about 5–7 mm (DATA_NOTES), so points within 1 cm get full weight. Moved chairs and doors at
  5–15 cm get weight 1 cm/|r|, so they cannot pull the fit.
- *Alternative:* hard trimming. It needs a choice of percentage and is discontinuous.

**E-9. Matching thresholds:** room IoU ≥ 0.3; walls ≤ 10°, ≤ 15 cm, ≥ 50% overlap; openings ≤ 0.5 m.
- *Why:* the thresholds must be wider than the *registration* uncertainty but narrower than the spacing between
  distinct elements:
  - ICP basins sit 5 cm apart (I-4) and drift tiles move 2–15 cm (I-2), so a 15 cm wall offset is needed.
  - Partition walls are ~10 cm thick, but they face opposite ways, so the normal test separates them.
  - Doors are ≥ 0.6 m wide, so their centres are > 0.5 m apart.
- *Test:* `test_matching_survives_local_drift` shifts one room by 12 cm and still matches all 12 walls with zero
  length error.

**E-10. One-to-one matching via the Hungarian algorithm** (scipy `linear_sum_assignment`).
- Greedy nearest-neighbour matching can assign one element twice, or steal a partner. That would turn a phantom into
  a "match" and hide a detection error.

**E-11. Per-room local correction before wall matching.**
- *Options:* (a) only widen tolerances; (b) non-rigid registration (thin-plate spline / CPD); (c) a per-room rigid
  correction.
- *Choice:* (c).
- *Why:* rooms are rigid at the scale that matters. (b) would bend walls, so it could change lengths, which is the
  quantity we measure. (a) alone risks matching a wall to its neighbour when drift reaches 15 cm.
- *Evidence:* the tile analysis (I-2), and the 12 cm drift test.

**E-12. Reading of "within 1 cm or 0.5% per wall": tolerance = max(1 cm, 0.5% · L).**
- "Or" means either condition suffices, so the lenient one applies: 1 cm for walls under 2 m, 0.5% above.
- The percentage uses the mean length of the two captures, so the test is symmetric.
- *Test:* `test_wall_tolerance_reading_of_or`.
- *Risk:* if the founders meant the stricter reading, it is one line (`wall_tolerance`) to change.

**E-13. Unmatched walls in matched rooms count as repeatability failures.**
- "Same room in, same plan out" means a wall that appears in only one capture is a non-repeatable result. Dropping
  it would flatter the pass rate.

**E-14. Ceiling: name the failure mode.**
- The gate text requires the report to say whether the result is repeatable-but-biased or unrepeatable.
- Bias needs GT, so without GT the honest statuses are `unrepeatable` or `repeatable_bias_unknown`. With a GT value,
  `classify_ceiling` returns `repeatable_but_biased` or `pass`.
- *Test:* `test_ceiling_classification_and_groups`.

**E-15. Reference-based evaluation, labelled, with noise-aware coverage.**
- *Options:* (a) mark every GT gate "not verifiable"; (b) score video/photo against LiDAR, labelled as such.
- *Choice:* both. LiDAR plans get `not_verifiable` rows, and video/photo plans get `lidar_reference` rows.
- *Why:* (b) still catches real failures, such as a 4% scale error or a missing room, which is useful for the fix
  loop.
- *Coverage:* noise-aware coverage stops the LiDAR reference's own error from counting against the tested tier.

**E-16. Calibration status = Wilson bound + RMS z band [0.5, 2].**
- *Why Wilson:* at n = 12, a plain ±1.96·sqrt(p(1−p)/n) interval is meaningless near p = 1.
- *Why RMS z:* the Wilson rule alone passed a case with RMS z 2.2 (I-7). RMS z is the size of the typical error in
  units of the claimed sigma, and should be ≈ 1. Outside [0.5, 2] the intervals are wrong by more than 2x.
- *Test:* `test_calibration_detects_over_and_under_confidence` uses 2,000 simulated samples. With intervals ×1 the
  status is calibrated and coverage is 0.95 ± 0.015; with ×0.3 it is overconfident; with ×5 it is underconfident.

**E-17. Opening gate denominator = matched + missed + phantom.**
- The brief says detection is scored: "a missed opening and a phantom opening each count as a miss".
- *Test:* `test_opening_gate_scores_detection` gives 2 hits out of 4 (one 1.5 cm hit, one exact hit, one missed, one
  phantom), so the rate is 0.5 and the gate fails.

**E-18. Tier wall gates need every matched wall within tolerance AND non-overconfident intervals.**
- "With calibrated intervals" is part of the row. Wall recall is reported so that a plan which drops the hard walls
  cannot pass on the easy ones.

**E-19. Drift gate = declaration + ablation, with the cross-capture loop check as independent evidence.**
- The brief's rule is about the report ("states what you do … an ablation shows"), so the plan must declare its
  method.
- *Independent check:* our loop check measures drift without trusting the drift module at all. It drops from
  3.70° / 70.7 cm to 0.29° / 5.9 cm with correction on (section 5).

**E-20. Round 1 gates are `not_verifiable` rather than guessed.**
- The Round 2 brief says "Round 1 gates apply" but does not list them. Inventing thresholds would make the report
  look complete while being wrong.

**E-21. Repeat pairs are discovered from registration and overlap, not from capture names.**
- *Why:* the walk-in test brings unseen captures, so nothing may depend on our three names. `single_room` vs either
  apartment scan is also a valid repeat pair, for the rooms it shares.

---

## 4. Issues log

**I-1. The yaw optimum sat on the edge of the ±3° window (room → floor_only).**
- *Symptom:* the yaw curve rose monotonically to −3°. After ICP the answer was −93.73°, outside the window.
- *Root cause:* `floor_only`'s living area is rotated about 3.7° relative to `floor_only`'s own dominant Manhattan
  axes. The capture is bent by drift.
- *Evidence:*
  - ICP restarted from −90.03° (the value implied by the other two registrations) converges back to −93.72° (RMSE
    1.09 cm).
  - The overlay at −93.7° fits the living area; the cycle-implied transform visibly does not
    (`registration/…floor_only…single_room…_overlay.png`).
  - `floor_only`'s own aligned overview shows its living-area walls slanted.
- *Possible fixes:* (a) widen the window; (b) a full 360° search; (c) register per room.
- *Chosen:* (a), ±5° plus an edge flag. It is the cheapest fix that covers `align.py`'s tolerance and reports
  anything beyond it. (b) is the next step for non-Manhattan buildings, and (c) lives in `match.py` (E-11).
- *Result:* no pair is at the edge any more (`yaw_at_search_edge = false`).

**I-2. The three captures cannot be made rigidly consistent (loop residual 3.70°, up to 70.7 cm).**
- *Symptom:* room→floor ∘ floor→ceil ≠ room→ceil.
- *Root cause:* accumulated drift inside the whole-apartment captures, mostly `floor_only`. The registrations
  themselves are correct.
- *Evidence:*
  - The tiles of `floor_only` against `with_ceiling` want to move by a median 5.2 cm (p90 15 cm, median |yaw| 0.85°).
  - The drift module independently measured 34 cm median and 3.0° revisit displacement in `floor_only`'s ARKit poses
    (`outputs/drift/…/ablation.json`).
  - On the drift-corrected scenes, the loop residual drops to **0.29° / 5.9 cm**, and the tiles to a median 2.4 cm,
    p90 4.0 cm and median |yaw| 0.45°.
- *Consequences:*
  1. One global rigid transform cannot match whole-apartment captures at cm level, hence the per-room correction in
     matching (E-11).
  2. The loop residual is a GT-free drift metric for the drift gate's report.
- *Chosen:* report it, and use it as independent evidence. Fixing drift is the drift module's job, not ours.

**I-3. The Hessian-based sigma is wildly optimistic.**
- *Symptom:* σ_yaw = 0.002° and σ_t < 0.1 mm.
- *Root cause:* σ²(JᵀJ)⁻¹ assumes independent residuals. Neighbouring TSDF points are strongly correlated, and the
  real error is dominated by non-rigidity.
- *Evidence:* restarting ICP from 20 perturbed starts (±1°, ±10 cm; `ablations/icp_restart.json`) gives a yaw spread
  (std) of **0.04° / 0.09° / 0.36°** and a centre displacement of up to **2.4 / 5.5 / 11.8 cm** for
  room→floor / room→ceil / floor→ceil.
- *Possible fixes:* (a) inflate by an effective-sample-size factor; (b) report the empirical restart spread; (c) a
  bootstrap over tiles.
- *Chosen:* keep the Hessian only as a *relative* indicator (constraint ratio). Use the restart spread and tiles as
  the honest numbers. Nothing in the gates depends on the registration sigma.

**I-4. Two ICP basins about 5 cm apart (room → with_ceiling).**
- *Evidence:* restarts converge either to the reported answer or to one shifted by 5.2–5.4 cm along v. Both have
  inlier fraction ≈ 0.72 and RMSE ≈ 0.87 cm.
- *Interpretation:* the room's geometry constrains v only weakly at that level of detail.
- *Impact:* lengths are unaffected, and matching tolerates it (E-9).
- *Not fixed:* the extra precision would not change any gate.

**I-5. Local tiles gave meaningless displacements.**
- *Symptom:* displacements such as 47 cm.
- *Root cause:* a tile whose walls all run one way can slide freely along them.
- *Fix:* compute each tile's translation constraint ratio, and flag or hide tiles with ratio ≤ 0.1.
- *Remaining:* one tile (43 cm, ratio 0.29) at the edge of `with_ceiling`'s coverage still looks like a mis-jump of
  the 30 cm first ICP radius. Tiles are a diagnostic, so median and p90 are what we report.

**I-6. The video-gate test failed, and the test was wrong, not the gate.**
- *What happened:* a 2% scale error on a 7.1 m wall is 14 cm. The synthetic ±6 cm intervals do not cover it, so the
  gate said "overconfident → fail".
- *Fix:* the test now checks three cases: within tolerance and calibrated; within tolerance but overconfident; and
  out of tolerance.
- *Lesson:* intervals must scale with length when the error is a scale error. This is input for the video-tier
  module.

**I-7. The calibration rule passed overconfident intervals.**
- *Symptom:* 12 walls with coverage 0.83, Wilson CI [0.55, 0.95] and RMS z 2.22 were labelled "calibrated", because
  the Wilson upper bound touched 0.95.
- *Root cause:* at small n, a coverage test has little power.
- *Options:* (a) require a larger n; (b) add an RMS-z band; (c) a chi-square test on z.
- *Chosen:* (b), RMS z > 2 counts as overconfident. It is symmetric with the < 0.5 underconfident rule and easy to
  explain. (c) is equivalent but harder to explain.

**I-8. Two interval encodings exist.**
- The dataclass form is `{lo, hi}`; the published schema (`export/json_export.py`) is `{ci95: [lo, hi]}`.
- *Fix:* `as_plan_dict` normalises both formats, so eval reads either. Verified on `outputs/export/stress/plan.json`.

**I-9. The export module's stress-test plan would have been scored as a real capture.**
- *Fix:* plans with `meta.synthetic: true` are skipped and listed as skipped.

**I-10. The synthetic fixture failed JSON serialisation** (numpy int64 from integer room coordinates).
- *Fix:* cast to float. It is trivial, but listed because the fixture must mimic a real `plan.json` round trip.

**I-11. Plans built on the drift-corrected scene live in a different frame.**
- The drift-on floor→ceil translation is (6.507, 0.248) m, against (6.045, −0.011) m for the original scenes.
- *Fix:* `best_match` tries both the scene registration and a plan-to-plan registration, and keeps the one that
  matches more walls. The report says which transform was used.
- *Alternative:* require producers to declare which scene variant they used (requested below). That is cleaner, but
  depends on other modules.

**I-12. The ceiling-spread gate cannot be verified on the sample data.**
- Only `with_ceiling` observes the ceiling (DATA_NOTES), so no room has two ceiling measurements. The gate reports
  `not_verifiable` with that reason; it is not a fabricated pass.
- *What would fix it:* a second capture with the ceiling in view (a capture-protocol item for Part 1).

**I-13. Real LiDAR plans fail repeatability by decimetres.**
- *Symptom:* only 0–6 walls pass per pair; median |Δ| is 18.5–29.7 cm on the apartment pair; unmatched walls
  outnumber matched ones by 1.4–4x.
- *First suspicion: an evaluation bug.* Checked by printing matched pairs: room IoUs are 0.53–0.93 and wall offsets
  2–15 cm, so the matches are the right walls. The overlay figures (`outputs/benchmark/repeat_*.png`) show the
  disagreement is real.
- *Breakdown by wall length* (`outputs/eval/ablations/real_plan_breakdown.json`, apartment pair):

  | producer | walls | n | pass | median \|Δ\| (cm) | p25 \|Δ\| (cm) |
  |---|---|---|---|---|---|
  | plan_alpha | short (< 1 m) | 29 | 1 | 22.2 | 6.6 |
  | plan_alpha | long (≥ 1 m) | 25 | 1 | 16.3 | 11.4 |
  | plan_beta | short (< 1 m) | 16 | 3 | 40.9 | 17.6 |
  | plan_beta | long (≥ 1 m) | 20 | 3 | 28.1 | 5.4 |

  Even long walls are off by decimetres. The problem is not centimetre noise but *different polygons*:
  - rooms extend into alcoves or doorways in one capture and not the other;
  - rooms have 14–22 walls, many of them 2–15 cm jogs, and are segmented differently per capture;
  - the plans were built from scenes without drift correction, so the same room is bent differently in each capture
    (I-2).
- *Hypotheses for the fix loop, in order of expected impact:*
  1. Simplify room polygons: snap or merge jogs shorter than ~20 cm into the main wall line.
  2. Build plans on the drift-corrected scenes.
  3. Define a wall's length by its plane extent within the room, not by the corner-to-corner path through jogs.

  These are owned by the plan modules; this module measures them.
- *What eval itself must not do:* loosen matching or ignore short walls to make the number look better. Both would
  hide a real failure.

**I-14. Discovery mis-read `outputs/runs/<capture>/<tier>/plan.json`.**
- The pipeline runner's layout made the tier folder look like the capture.
- *Fix:* `_capture_of` takes the path component that matches a known scene, skipping tier folder names.


---

## 5. Results on the captures

**Registration** of the original ARKit scenes (`outputs/eval/registration/summary.json`; figures `*_overlay.png`,
`*_residual.png`, `*_yaw.png`, `*_tiles.png`):

| B → A | yaw (°) | RMSE (cm) | inliers | floor IoU | cover A / B | distinct | 3D ICP check Δ |
|---|---|---|---|---|---|---|---|
| single_room → floor_only | −93.73 | 1.08 | 0.57 | 0.30 | 0.31 / 0.97 | 1.98 | 0.32°, ≤ 2.0 cm |
| single_room → with_ceiling | −0.98 | 0.94 | 0.73 | 0.28 | 0.29 / 0.90 | 2.32 | 0.21°, ≤ 1.8 cm |
| floor_only → with_ceiling | 89.05 | 1.15 | 0.40 | 0.79 | 0.89 / 0.87 | 2.59 | 0.33°, ≤ 1.8 cm |

How to read it:
- `single_room` lies 90–97% inside each apartment scan. That is expected, since it is a subset.
- The two apartment scans overlap 0.79 IoU and cover 87–89% of each other.
- The inlier fraction for floor→ceil is only 0.40: rigid registration of drifted captures (I-2).

**Loop check** (`cycles.json`) and **tiles**, with drift correction OFF and ON (`outputs/eval/registration_drift_on/`):

| | loop yaw | loop max displacement | floor↔ceil inliers | floor↔ceil distinct | tiles median / p90 displacement | tiles median \|yaw\| |
|---|---|---|---|---|---|---|
| ARKit as-is | 3.70° | 70.7 cm | 0.40 | 2.59 | 5.2 / 15.0 cm | 0.85° |
| drift-corrected | **0.29°** | **5.9 cm** | **0.79** | **4.26** | **2.4 / 4.0 cm** | **0.45°** |

In plain words, measured without any ground truth and without trusting the drift module's own metrics: after drift
correction, the two apartment captures agree with each other, and with `single_room`, about ten times better.
- Room→floor yaw becomes −89.97°, almost exactly a quadrant: `floor_only`'s living area is no longer bent.
- Figures: `registration_drift_on/*_residual.png`, `*_tiles.png`.

**Ablations** (`outputs/eval/ablations/`):
- *Normal-split channels* (E-4): distinctiveness +31–35%.
- *±3° window:* the room→floor optimum is flagged at the edge (E-6).
- *ICP restarts:* yaw std 0.04–0.36°, and up to 2.4 / 5.5 / 11.8 cm centre spread (I-3).

**Match / repeatability / gates on synthetic plans** (`tests/eval`, 16 tests pass):
- A known transform (91.3°, (5, −2) m) is recovered exactly (0.000°, 0.0 mm) by plan-to-plan registration when the
  two plans are identical. With the perturbed plan B (two rooms resized by 7 mm and 3 cm) it is recovered to 0.010°
  and 4.5 mm at the plan centre; the fit splits the 3 cm resize. The tests assert < 0.05° and < 2 cm.
- Known perturbations are recovered exactly (to 1e-9 m):
  - +7 mm on two walls → pass;
  - +3 cm on two walls → fail;
  - ceiling +5 mm → `repeatable_bias_unknown`;
  - ceiling +2 cm → `unrepeatable`;
  - opening +1.5 cm.
- One missed and one phantom opening are reported. The opening gate gives 2/4 = 0.5 → fail.
- 12 cm local drift of one room: all walls are still matched, and the gate passes.
- Scale 1.02 / 1.04 on the video tier gives pass / fail. Scale 1.02 with ±6 cm intervals gives fail (overconfident).
- Photo stitch: overlapping rooms and a missing adjacency fail their sub-rows.
- Benchmark end-to-end on a temporary outputs tree:
  - repeatability fails with n = 12 (two walls off);
  - ceiling spread is reported as `unrepeatable`;
  - the video wall gate passes, with basis `lidar_reference`;
  - LiDAR GT gates are `not_verifiable`;
  - the synthetic plan is skipped;
  - an empty outputs tree still writes a report.

**Benchmark on the real LiDAR plans** (`outputs/benchmark/report.md`, `gates.csv`, `repeatability_walls.csv`,
figures `outputs/benchmark/repeat_*.png`).

This is a snapshot at 22:34 on 3 Oct. `plan_alpha` and `plan_beta` were still being regenerated by their agents
while this ran, so rerun `scripts/benchmark.py` for current numbers. There were no video or photo plans yet, so no
`lidar_reference` rows exist.

| producer | pair | rooms matched | walls pass / fail / unmatched | median \|Δ\| (cm) | p90 \|Δ\| (cm) | openings matched / unmatched |
|---|---|---|---|---|---|---|
| plan_alpha | single_room vs floor_only | 3 | 0 / 12 / 32 | 50.6 | 274.7 | 1 / 18 |
| plan_alpha | single_room vs with_ceiling | 3 | 0 / 10 / 26 | 122.0 | 268.2 | 1 / 15 |
| plan_alpha | floor_only vs with_ceiling | 10 | 2 / 52 / 72 | 18.5 | 111.4 | 6 / 17 |
| plan_beta | single_room vs floor_only | 2 | 1 / 13 / 24 | 20.8 | 76.6 | 4 / 21 |
| plan_beta | single_room vs with_ceiling | 2 | 1 / 7 / 32 | 137.4 | 195.7 | 2 / 23 |
| plan_beta | floor_only vs with_ceiling | 7 | 6 / 30 / 82 | 29.7 | 222.2 | 11 / 24 |

What the benchmark says:
- **Repeatability: fail for both producers on every pair.** This is currently the worst gate and the natural fix-loop
  target (I-13).
- **Repeat-pair interval calibration: overconfident for both producers.**
  - plan_alpha: n = 76, coverage 0.16, RMS z 37.4.
  - plan_beta: n = 58, coverage 0.19, RMS z 32.4.

  The plans' wall intervals are millimetre-wide, while the captures disagree by decimetres.
- **Ceiling spread: not verifiable.** Only `with_ceiling` sees a ceiling (I-12).
- **Drift: not verifiable.** The plans do not declare `meta.drift`. The drift module's ablation exists and is quoted
  in the report row.
- **GT-dependent LiDAR rows** (opening widths, ceiling, calibration, Round 1): not verifiable, with the reason stated.
- The synthetic stress plan is skipped. `outputs/runs/<capture>/lidar/plan.json` is discovered with the right capture
  (I-14).

---

## 6. Limitations and failure modes

- **Mirrors.** LiDAR and RGB see a "room behind the mirror": a mirrored copy of real walls appears as extra wall
  evidence.
  - *Registration:* it is usually harmless, because both captures see the same mirror. But a mirror seen in only one
    capture adds unmatched evidence and lowers the inlier fraction.
  - *Plans:* a mirror can create a phantom room or opening. The opening gate counts phantoms as misses, and the
    photo-stitch overlap check catches a phantom room that overlaps a real one.
  - *Next:* mask evidence beyond a wall plane that sits behind a known wall (geometrically impossible returns).
- **Glass** (the shower screen, the dark window). It produces no high-confidence LiDAR returns, so wall evidence has
  holes. Registration uses the rest of the room, and the tiles flag low-evidence areas. Matching may then leave a
  glass wall unmatched, which the repeatability gate counts as a failure. That is the honest outcome, since the
  plans really do disagree there.
- **Wet-look and specular floors.** Floor points drop out, so the floor overlap (IoU) under-reads. This is a
  diagnostic only and never used in a gate.
- **Low light.** LiDAR is unaffected, so registration and LiDAR plans are fine. Video and photo tiers degrade, and
  their reference-based rows should fail or widen. Calibration then shows whether their intervals widened honestly
  (`underconfident` / `overconfident`).
- **Clutter and furniture that moved between captures.** Furniture faces in the 0.3–2.0 m band count as wall
  evidence. Huber weighting and the normal check limit the damage, and the inlier fraction drops visibly. The tiles
  show where.
- **Symmetric layouts.** Two quadrants look alike, so distinctiveness falls below 1.2, and the benchmark refuses the
  registration instead of guessing.
- **Non-Manhattan buildings** (45° walls, curved walls). The search assumes k·90° ± 5°. *Next:* a full 360° yaw sweep
  at 1° on the 5 cm grid (about 360 FFTs, under 1 min), which the same code supports by changing the yaw list.
- **Corridor-like captures.** Translation along the corridor is unconstrained. The constraint ratio reports it.
  *Next:* use the floor outline (free-space raster) as a second correlation channel.
- **The registration uncertainty is empirical, not analytic** (I-3). *Next:* a bootstrap over tiles, giving a proper
  covariance.
- **No ground truth.** Absolute-accuracy gates are `not_verifiable` on the sample data. The `--gt-dir` hook is ready
  for tape or laser measurements written as a `plan.json`, or for the walk-in test's laser numbers.
- **Reference-based ≠ accuracy.** A video plan that copies LiDAR's error would pass. Every row is labelled
  `lidar_reference` for that reason.
- **Ceiling spread** cannot be checked until two captures observe the same ceiling (I-12).
- **Wall identity across producers.** Repeatability is only computed within a producer, so a method change between
  captures is never mistaken for non-repeatability.

### Requested changes to shared code / other modules

1. **Plan producers:** write `meta.drift = {"method": "<name>", "ablation": {"footprint_on_m2": x,
   "footprint_off_m2": y, "figure": "<path>"}}`. Without it, the drift gate stays `not_verifiable` (or `fail` for
   "as-is").
2. **Plan producers:** write `meta.scene = "<path of the scene.npz used>"`. The benchmark then knows the plan's frame,
   instead of discovering it by trying two transforms (I-11).
3. **Plan producers:** write plans to `outputs/<producer>/<capture>/plan.json` with `tier` set. Test or stress plans
   must set `meta.synthetic: true`.
4. **Video and photo modules:** make the length interval proportional to length when the dominant error is scale
   (I-6).
5. **Ground truth:** record tape or laser measurements as a `plan.json` in the published schema under
   `ground_truth/<capture>/plan.json`. Every GT gate then runs with basis `ground_truth`, with no code change.
6. **Move the ablation scripts** in `outputs/eval/ablations/*_cmd.py` to `scripts/ablate_registration.py` in the real
   repo. They are kept with their evidence here because scripts outside this module's file list were not mine to
   create.

---

## 7. Proposed commits (to replay in the real repo, in order)

| # | Message | Files | Why |
|---|---|---|---|
| 1 | `feat(eval): rigid 2D scene registration (Manhattan quadrants, FFT correlation, point-to-line ICP)` | `floorplan/eval/register.py` | Captures have arbitrary headings; every cross-capture comparison needs a transform that does not depend on the plans |
| 2 | `feat(eval): register_scenes script with overlay, residual and yaw-curve figures` | `scripts/register_scenes.py` | Evidence you can look at; the first figures exposed I-1 |
| 3 | `fix(eval): widen yaw window to ±5° and flag optima on the window edge` | `register.py` | floor_only's living area is rotated 3.7° (I-1) |
| 4 | `feat(eval): loop-consistency check and tile-wise local disagreement` | `register.py`, `register_scenes.py` | Shows the captures are not rigidly consistent (3.70° / 70.7 cm), i.e. drift (I-2) |
| 5 | `fix(eval): flag degenerate tiles by translation constraint ratio` | `register.py`, `register_scenes.py` | One-direction tiles produced meaningless 47 cm displacements (I-5) |
| 6 | `exp(eval): registration ablations: normal-split channels, yaw window, ICP restarts, 3D ICP cross-check` | `scripts/ablate_registration.py` | Evidence for E-4, E-6, E-7 and I-3 |
| 7 | `test(eval): synthetic plan fixture with exactly known geometry` | `tests/eval/synthetic.py`, `tests/eval/conftest.py` | Develop matching and metrics before real plans exist |
| 8 | `feat(eval): plan matching (rooms by IoU + Hungarian, per-room ICP correction, walls, openings)` | `floorplan/eval/match.py`, tests | One-to-one, drift-tolerant element matching (E-9 to E-11) |
| 9 | `feat(eval): repeatability tables: per-wall gate, ceiling-spread classification, union-find groups` | `floorplan/eval/repeatability.py`, tests | The one accuracy-type gate fully testable without GT (E-12 to E-14) |
| 10 | `feat(eval): Part 2 gates as data, reference-based evaluation and interval calibration` | `floorplan/eval/gates.py`, tests | Regenerable pass/fail rows for the fix loop (E-15 to E-20) |
| 11 | `test(eval): video wall gate checks calibration as well as tolerance` | tests | The first test expectation was wrong; the gate was right (I-6) |
| 12 | `fix(eval): RMS z > 2 counts as overconfident` | `gates.py`, tests | The Wilson bound alone passed RMS z 2.2 at n = 12 (I-7) |
| 13 | `feat(eval): accept published-schema ci95 intervals` | `match.py` | The export schema differs from the dataclasses (I-8) |
| 14 | `feat(bench): benchmark script (discovery, repeatability, reference gates, drift evidence, report)` | `scripts/benchmark.py`, `tests/eval/test_benchmark_smoke.py` | One command regenerates every gate; robust to missing pieces |
| 15 | `fix(bench): skip synthetic plans; choose scene vs plan transform by matched walls` | `scripts/benchmark.py` | I-9, I-11 |
| 16 | `exp(eval): cross-capture loop check with drift correction on vs off` | `register_scenes.py` (`--scenes`, `--out`) | GT-free evidence for the drift gate: 3.70° / 70.7 cm → 0.29° / 5.9 cm |
| 17 | `feat(bench): repeatability overlay figures and per-length breakdown` | `scripts/benchmark.py`, `match.py` (`aligned_plan_b`) | Shows *where* plans disagree; this is the evidence for the fix-loop hypothesis (I-13) |
| 18 | `fix(bench): find the capture in <capture>/<tier>/plan.json layouts` | `scripts/benchmark.py` | Pipeline runner layout (I-14) |
| 19 | `exp(bench): first real benchmark: LiDAR repeatability fails (median 18-30 cm on the apartment pair)` | none (numbers in the message) | Records the before state for Part 4 |
| 20 | `docs(eval): module documentation and decision log` | `docs/modules/eval.md` | Defense preparation |

`outputs/` is never committed. The commit messages quote the numbers, so the history explains itself.

---

## 8. Concepts to explain in the defense

- **Rigid vs similarity transform.**
  - Rigid = rotate + shift; distances are unchanged.
  - Similarity = rigid + uniform scale.
  - We use rigid, so that a scale error stays visible.
- **Manhattan world.** Most indoor walls meet at 90°. After aligning each capture to its dominant wall directions,
  two captures differ by a multiple of 90° plus a small residual. That turns a 360° search into 4 small ones.
- **FFT cross-correlation.**
  - Sliding image B over image A and summing the products at every offset is a correlation.
  - The FFT computes that sum for *all* offsets in one shot (multiply the spectra, transform back).
  - The highest peak is the best translation.
  - It is exhaustive, so it cannot get stuck in a local minimum.
- **ICP (Iterative Closest Point), point-to-line.**
  - Repeat: pair each point with its nearest neighbour in the other set; solve for the small rotation and shift that
    reduce the distances; apply.
  - Point-to-*line* measures the distance along the wall's normal only, so points may slide along a wall freely.
    This converges faster and more accurately on flat walls.
- **Huber loss.** Quadratic for small residuals (trust them), linear for large ones (limit their pull). It is robust
  to outliers without throwing them away.
- **Gauss-Newton and its Hessian covariance.**
  - Linearise the residuals, then solve the normal equations JᵀJ·δ = −Jᵀr.
  - σ²(JᵀJ)⁻¹ is the parameter covariance *if* the residuals were independent. Here they are not, which is why we
    also report restart spreads.
- **Degeneracy / constraint ratio.** If all walls run one way, nothing stops sliding along them. The eigenvalues of
  JᵀJ show which directions are constrained; their ratio near 0 means degenerate.
- **Loop (cycle) consistency.** A←B←C must equal A←C. The leftover rotation and shift measure how non-rigid the
  captures are, i.e. drift, without any ground truth.
- **IoU (intersection over union).** Overlap area divided by combined area; 1 means identical shapes.
- **Hungarian algorithm.** Optimal one-to-one assignment for a cost matrix, in polynomial time. It prevents one
  element from being matched twice.
- **Union-find.** Merge nodes linked by matches into groups, e.g. "room X in captures 1, 2 and 3 is one room".
- **Repeatability vs accuracy.**
  - Repeatability: the same input gives the same output. Needs no truth.
  - Accuracy: the output equals the truth. Needs GT.
  - "Repeatable but biased" = consistently wrong. "Unrepeatable" = varies run to run.
- **Reference-based evaluation.** Comparing a weaker method (video/photo) with a stronger one (LiDAR) on the same
  data. It measures disagreement, not accuracy, and inherits the reference's error.
- **Confidence interval and coverage.**
  - A 95% interval should contain the truth 95% of the time.
  - Coverage = the fraction of cases where it did.
  - Coverage too low means overconfident ("confident garbage"); intervals far too wide means useless
    (underconfident).
- **Wilson score interval.** An interval for a proportion (such as coverage) that behaves well with few samples and
  near 0% or 100%. We use its upper bound to decide whether a coverage below 95% is real or just bad luck.
- **z-score and RMS z.**
  - z = error / claimed sigma. If the sigmas are honest, z has spread 1, so RMS z ≈ 1.
  - RMS z = 3 means the errors are 3x larger than claimed.
- **Noise-aware coverage.** When the reference has its own error, compare the difference with the *combined* sigma,
  sqrt(σ₁² + σ₂²), not with the tested interval alone.
- **Detection-scored metric.** Missed and phantom elements enter the denominator as failures, so a method cannot score
  well by only reporting the easy openings.
