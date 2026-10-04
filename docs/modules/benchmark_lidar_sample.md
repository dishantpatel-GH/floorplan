# Benchmark: LiDAR tier on the sample captures

What this is: the LiDAR-tier part of the benchmark report (case study Deliverable 5), measured on the three Stray
Scanner sample captures. Every number below comes from a run made for this report on 2026-10-04, using the one
command per capture and default settings. Nothing is copied from earlier module documents. Where an earlier
document gives a different number for the same thing, both are shown and the difference is explained.

The gate verdicts are in `outputs/benchmark/lidar/gates.md`. The raw results are in
`outputs/benchmark/lidar/results.json`.

## 1. What was run

**Inputs.** Three captures from `TakeHome/Dataset/`:

- `single_room/c00a170fe1`: 37 s, one room plus an alcove and a bathroom.
- `single_scan_floor_only/1a8384c3f6`: the whole apartment, one loop, floor only.
- `single_scan_with_ceiling/c7d28f72c6`: the same apartment, walked twice, ceilings included.

floor_only and with_ceiling are the **repeat pair**: the same rooms, captured twice at the same tier.

**Command.** The one command, with default settings. Defaults are:

- drift correction ON (D-019);
- LiDAR inward-bias correction ON (D-021);
- damage two-tier reporting ON (D-023);
- extractor plan_beta v2.

```
python scripts/run_capture.py <capture_dir> --tier lidar --out outputs/benchmark/lidar/<capture>/<variant> [--no-drift]
```

| Script (new, `scripts/bench_*`) | What it does |
|---|---|
| `scripts/bench_lidar_runs.sh` | First pass, all 5 runs in parallel. Default runs on all 3 captures (`drift_on`), and `--no-drift` on floor_only and with_ceiling (`drift_off`, the plan-level drift ablation). Each run is wrapped in `/usr/bin/time -v`. |
| `scripts/bench_lidar_retime.sh` | Second pass, one capture at a time: the default command again on all 3 captures (`drift_on_rerun2`). Gives cleaner timing and a run-to-run check. |
| `scripts/bench_lidar_eval.py` | All scoring. It only reads the run folders and reuses existing code unchanged (list below). |
| `scripts/bench_lidar_rerun.py` | Same capture, same command, twice: floor_only `drift_on` against `drift_on_rerun_nodamage`. |

Code reused unchanged by `bench_lidar_eval.py`:

- `floorplan/eval/register.py`: scene-to-scene registration from geometry, the same call as `plan_v2_eval.register_pair`.
- `judge_common.compare`: matching, the gate max(1 cm, 0.5% L), unmatched = fail, topology split, Wilson interval.
- `plan_v2_eval.same_topology_coverage`.
- `judge_plausibility.plan_stats` / `floor_recall`.
- `drift_ablation.wall_sharpness` / `footprint` / `to_common_frame`.

**Regenerate everything** (from the repo root, about 10–20 min depending on load):

```
scripts/bench_lidar_runs.sh                       # 5 runs -> outputs/benchmark/lidar/<capture>/{drift_on,drift_off}/
scripts/bench_lidar_retime.sh                     # 3 runs -> outputs/benchmark/lidar/<capture>/drift_on_rerun2/
python scripts/run_capture.py ../TakeHome/Dataset/single_scan_floor_only/1a8384c3f6 --tier lidar \
    --out outputs/benchmark/lidar/floor_only/drift_on_rerun_nodamage --no-damage
env -u PYTHONPATH python scripts/bench_lidar_eval.py    # -> results.json, walls_*.csv, repeat_*.png, drift_overlay_*.png
env -u PYTHONPATH python scripts/bench_lidar_rerun.py   # -> rerun_floor_only.json
```

**Machine note.** The first pass ran 5 captures in parallel while other agents were also using the CPU (load average
35–70 on 22 cores), so its wall times are upper bounds. The second pass ran sequentially on a quieter machine.

**Code version note.** Other agents changed pipeline code while this benchmark was running.

| When | File | Change |
|---|---|---|
| 02:06 | `floorplan/damage/__init__.py` | D-026: crack confirm threshold 0.65 |
| 02:09 | `floorplan/uncertainty/bias.py` and `scripts/run_capture.py` | D-027: corner-type-aware length correction |

Which runs used which code:

- **First pass, 01:46–01:56** (`drift_on`, `drift_off`): code from *before* D-026 and D-027.
- **`drift_on_rerun_nodamage`**, about 02:05–02:07: same as the first pass for the plan. Its plan stage finished
  before 02:09.
- **Second pass, 02:10–02:19** (`drift_on_rerun2`): the current code.

What this changes:

- The drift on/off comparison is internally consistent, because both arms of the first pass use the same code.
- The repeat pair is reported for both passes, and both give 15 passing walls.
- D-027 moves individual wall lengths by ±b at reflex corners. It leaves areas, perimeters and footprints unchanged.

## 2. Timing

Wall time is for the whole command: front end, drift, scene, plan, damage and export. Peak RSS is the peak memory
use.

| Capture | Pass | Wall s | CPU s (user) | Drift s | Scene saved at s | Plan done at s | Damage s | Peak RSS GB |
|---|---|---|---|---|---|---|---|---|
| single_room | parallel (loaded) | 155 | 286 | 6.8 | 43 | 51 | 84 | 2.0 |
| floor_only | parallel (loaded) | 373 | 805 | 45.0 | 136 | 167 | 156 | 3.3 |
| with_ceiling | parallel (loaded) | 561 | 1422 | 103.6 | 261 | 299 | 223 | 4.8 |
| floor_only `--no-drift` | parallel (loaded) | 354 | 744 | – | 91 | 122 | – | 3.2 |
| with_ceiling `--no-drift` | parallel (loaded) | 508 | 1297 | – | 169 | 222 | – | 4.9 |
| floor_only `--no-damage` | alone (load ~4) | 69 | 241 | 11.3 | 44 | 65 | off | 3.3 |
| single_room | sequential (load 4–14), **current code** | **65** | 217 | 2.9 | 18 | 21 | 39 | 2.0 |
| floor_only | sequential, current code | **170** | 621 | 10.4 | 38 | 55 | 103 | 3.3 |
| with_ceiling | sequential, current code | **305** | 1238 | 33.6 | 100 | 132 | 156 | 4.7 |

**Reading.**

- On a quiet machine the one command takes **65 s / 170 s / 305 s** (single_room / floor_only / with_ceiling).
- Damage is the most expensive stage: 39–156 s, about half the run.
- Drift takes 3–34 s and the plan extractor 3–32 s.
- Under the shared-CPU first pass, everything was about 1.8–2.4× slower.
- The case study sets no processing-time gate. It asks for "README to running on a fresh capture in under 15
  minutes", and all three captures finish well inside that.

## 3. Drift accountability (gate: state the method and show the footprint with drift handling on and off)

**Method** (full description in `docs/modules/drift.md`):

1. Cut the walk into fragments of 1.5 m or 5 s.
2. Build a 4-DoF pose graph. ARKit's relative poses are the odometry edges.
3. Add ICP loop closures between revisited fragments: two rounds, each loop verified by fitness, plane RMSE and
   jump limits.
4. Add per-fragment plane anchors: the Manhattan wall direction (absolute heading) and the floor level.
5. Solve robustly with IRLS.
6. Blend the corrections onto every frame before the TSDF and raw points are built.

The ceiling level is never used in the solve, so it serves as an independent witness. `--no-drift` uses the ARKit poses
as-is: this is the ablation only, since poses as-is fail the gate.

**How on and off are compared here.**

- (a) Plan-level numbers from each run's own plan.json.
- (b) The drift module's scene metrics, which the solver never saw:
  - wall sharpness: the spread of raw LiDAR wall points around their local wall line, per oriented face, where a
    doubled wall widens it;
  - footprint proxies.
- (c) The OFF plan placed in ON's frame two ways:
  - through the shared world gauge (the first fragment is fixed), which shows the accumulated drift;
  - registered by scene geometry, which leaves only the shape differences.
- (d) The OFF plan scored against the ON plan with the repeatability code.

Figures: `outputs/benchmark/lidar/drift_overlay_floor_only.png`, `drift_overlay_with_ceiling.png`.

### 3.1 Plan level (the stitched footprint)

| | floor_only ON | floor_only OFF | with_ceiling ON | with_ceiling OFF |
|---|---|---|---|---|
| Rooms | **8** | 7 | 8 | 8 |
| Walls (shorter than 15 cm) | 70 (4) | 66 (9) | **70 (9)** | 82 (20) |
| Openings | 8 | 6 | 13 | 13 |
| Footprint m² [95%] | **61.90** [61.06, 62.47] | **52.59** [51.78, 53.35] | 62.50 [61.66, 62.91] | 61.99 [61.23, 62.53] |
| Plan bbox m | 10.37 × 9.67 | 10.27 × 9.56 | 9.67 × 10.35 | 9.64 × 10.33 |
| Floor recall (observed floor inside rooms) | **0.940** | 0.860 | 0.952 | 0.938 |
| Plan connected | yes | **no** | yes | yes |
| Overlap m² | 0 | 0 | 0 | 0 |
| OFF scored against ON (rooms matched; walls within gate / judged; unmatched; median \|Δ\|) | – | 7 / 8; 11 / 89; 46; 4.0 cm | – | 8 / 8; 13 / 95; 38; 4.5 cm |
| OFF registered onto ON (inliers; tile disagreement median / p90) | – | 0.53; 5.4 / 10.5 cm | – | 0.69; 3.7 / 5.9 cm |

**floor_only** is where drift handling decides the plan.

- ARKit's heading drifted by up to 5.95° (median frame correction 4.4°, 26 cm; max 64 cm).
- With poses as-is:
  - one room disappears;
  - the plan breaks into disconnected parts;
  - the footprint is 15% smaller (52.6 vs 61.9 m²);
  - 46 of 89 walls have no counterpart.
- In the left panel of the overlay, the OFF rooms are visibly rotated about 4° and offset by up to about 0.6 m.

**with_ceiling** was already fairly good under ARKit, because it walks the loop twice. There, drift handling mainly
removes fragmented, doubled walls: 82 → 70 walls, and 20 → 9 walls shorter than 15 cm. The envelope changes by 0.8%.

### 3.2 Scene level, independent of the plan (drift module metrics)

| | floor_only ARKit (OFF) | floor_only ON | with_ceiling ARKit (OFF) | with_ceiling ON |
|---|---|---|---|---|
| Revisit-loop surface separation, median (in-sample) | 14.29 cm (n = 10) | **2.27 cm** | 2.83 cm (n = 25) | **1.15 cm** |
| Local-loop surface separation, median | 4.01 cm | 1.43 cm | 3.31 cm | 0.84 cm |
| Floor level spread (MAD σ over fragments) | 1.22 cm | 0.36 cm | 0.81 cm | 0.40 cm |
| Wall-direction spread (σ) | 1.69° | 0.59° | 0.70° | 0.49° |
| **Ceiling level spread** (independent witness) | – | – | **6.04 cm** | **0.54 cm** |
| Wall sharpness, raw wall points: MAD σ | 10.96 mm | **9.49 mm** | **15.42 mm** | 15.77 mm |
| Wall sharpness, share within 1 cm | 60.0% | **64.2%** | **48.7%** | 48.0% |
| Wall sharpness, p90 | 49.3 mm | **32.2 mm** | **92.3 mm** | 103.0 mm |
| Filled floor footprint m² (scene) | 43.18 | 43.01 | 41.80 | 41.61 |
| Wall bbox m (scene) | 10.29 × 10.86 | 10.40 × 10.95 | 10.80 × 12.93 | 10.81 × 12.88 |

**One honest regression.**

- On with_ceiling in this run, the corrected scene's walls are *not* sharper than ARKit's: 15.77 vs 15.42 mm MAD σ,
  and p90 103 vs 92 mm.
- `docs/modules/drift.md` §5.1 reported 15.13 vs 15.49 mm, i.e. the opposite sign, from its own ablation run.
  drift.md itself notes that borderline loops flip between runs, and that sharpness differences below about 0.3 mm are
  not meaningful. This difference is 0.35 mm, at that limit.
- The with_ceiling verdict on wall sharpness is therefore **a tie, not a gain**. The ceiling witness (6.04 → 0.54 cm)
  and the plan-level fragmentation (20 → 9 tiny walls) still favour ON there.
- floor_only improves on every metric.

### 3.3 Ceilings with drift on and off (with_ceiling)

| Room | ON m [95%] | OFF m [95%] (status) | OFF − ON |
|---|---|---|---|
| R1 (14.3 m²) | 2.481 [2.454, 2.508] | 2.424 [2.397, 2.451] measured | **−5.7 cm**, the OFF interval excludes ON |
| R2 | 3.098 | 3.098 measured | −0.1 cm |
| R3 | 3.078 | 2.970 [2.943, 3.082] *inferred* | −10.8 cm, the widened interval covers ON (B-6 double-layer rule) |
| R4 | 3.091 | 3.095 | +0.4 cm |
| R5 | 2.372 | 2.358 | −1.4 cm |
| R6 | 2.393 [2.366, 2.420] | 2.294 [2.267, 2.321] measured | **−9.9 cm**, the OFF interval excludes ON |
| R7 | 2.480 | 2.451 [2.387, 2.478] inferred | −2.9 cm, just excluded |
| R8 | 2.485 | 2.484 | −0.1 cm |

Without drift handling, 2 of 8 ceilings would be confidently wrong by 6–10 cm. This is the vertical drift that the
ceiling-level spread measures: 6.04 cm across fragments.

## 4. Repeatability (with_ceiling = A, floor_only = B, drift ON)

**Registration.** From scene geometry, never from a plan: yaw 89.07°, residual RMSE 0.98 cm, inliers 0.79. Local
tiles disagree by a median of 2.0 cm (p90 4.1 cm) over 15 tiles. This matches `plan_v2_ablation.md` (89.08°, 0.98 cm,
2.0 / 4.1 cm).

| Metric | First pass (`drift_on`, pre-D-026/027 code) | Second pass, same command (`drift_on_rerun2`, current code) | plan_v2_ablation.md final row (scenes_v2) |
|---|---|---|---|
| Rooms A / B / matched | 8 / 8 / 8 | 8 / 8 / 8 | 8 / 8 / 8 |
| Walls pass / judged | **15 / 86 = 17.4%** | **15 / 90 = 16.7%** | 13 / 87 = 14.9% |
| Wilson 95% | [10.9, 26.8]% | [10.4, 25.7]% | [8.9, 23.9]% |
| Unmatched walls | 32 | 36 | 36 |
| Median \|Δ\| (matched walls) | 5.9 cm | 7.3 cm | 8.2 cm |
| Walls within 2 cm | 18 / 54 matched | 18 / 54 matched | – |
| Same topology n / pass / median | 33 / 14 / 2.0 cm | 30 / 14 / 1.8 cm | 26 / 12 / 1.7 cm |
| Different topology n / pass / median | 21 / 1 / 42.9 cm | 24 / 1 / 33.9 cm | 25 / 1 / 32 cm |
| Walls measured in both, n / pass / median | 21 / 9 / 2.0 cm | – | – |
| Walls ≥ 1 m, matched / pass / median | 34 / 7 / 8.6 cm | – | – |
| Footprint A / B m² | 62.50 / 61.90 | 62.56 / 61.90 | 62.65 / 61.90 (no bias correction) |
| Floor recall A / B | 0.952 / 0.940 | – | 0.952 / 0.940 |
| Interval coverage, all / same-topology | 0.69 / 0.94 | 0.67 / 0.97 | 0.63 / 0.96 |

**Verdict: FAIL.** The gate needs every wall within max(1 cm, 0.5%), and 17% pass.

**Why it fails.**

- 32 of 86 walls are unmatched, and 21 matched walls have different neighbours: a median of 43 cm, of which 1 passes.
  Those are segmentation and topology differences, not measurement error.
- Walls whose corners are built the same way agree to a median of 2.0 cm. That equals the residual misalignment
  between the two drift-corrected captures (registration tiles: 2.0 cm median). So measurement sits at the drift floor,
  and even perfect segmentation would pass only about 40–45% of walls.

The bias correction does not affect this gate: it adds the same 2b to both captures.

**Per-wall table:** `outputs/benchmark/lidar/walls_with_ceiling__floor_only.csv`, one row per wall. Columns:

- A and B ids, both lengths;
- Δ and the tolerance;
- status;
- same-topology flag;
- z;
- measured or inferred status in each capture.

Figure: `repeat_with_ceiling__floor_only.png`. Per room (A's rooms; "B" means walls only in floor_only):

| Room (A) | pass | fail | unmatched (A) | unmatched (B) |
|---|---|---|---|---|
| R1 | 3 | 3 | 2 | 0 |
| R2 | 1 | 4 | 5 | 5 |
| R3 | 4 | 6 | 4 | 0 |
| R4 | 2 | 6 | 0 | 4 |
| R5 | 1 | 4 | 1 | 5 |
| R6 | 3 | 8 | 1 | 1 |
| R7 | 0 | 3 | 3 | 1 |
| R8 | 1 | 5 | 0 | 0 |

**Room boxes** (what a homeowner checks with a tape; shorter side paired with shorter side):

| Room | Area A / B m² | Box A (m) | Box B (m) | Δ (cm) |
|---|---|---|---|---|
| R1 | 14.31 / 13.99 | 3.879 × 3.974 | 3.888 × 3.973 | +0.9 / −0.0 (both pass) |
| R2 | 12.81 / 12.12 | 3.030 × 4.370 | 2.895 × 4.606 | −13.5 / +23.6 |
| R3 | 11.17 / 11.33 | 3.008 × 4.854 | 3.005 × 4.801 | −0.3 / −5.3 |
| R4 | 9.33 / 9.26 | 3.018 × 3.170 | 3.022 × 3.445 | +0.4 / +27.5 |
| R5 | 7.67 / 7.68 | 2.517 × 3.100 | 2.555 × 3.089 | +3.8 / −1.1 |
| R6 | 3.39 / 3.62 | 1.867 × 2.217 | 1.822 × 2.362 | −4.5 / +14.5 |
| R7 | 2.47 / 2.84 | 1.585 × 2.082 | 1.613 × 1.759 | +2.8 / −32.3 |
| R8 | 1.96 / 1.68 | 1.041 × 1.974 | 0.891 × 1.938 | −15.0 / −3.6 |

### 4.1 Same capture, same command, run twice (a repeatability floor nobody asked for, but it matters)

`scripts/bench_lidar_rerun.py`, floor_only, `drift_on` against `drift_on_rerun_nodamage`. Damage is off in the second
run; it runs after the plan and does not change it.

| | Run 1 | Run 2 |
|---|---|---|
| Drift: max frame correction | 64.222 cm, 5.9454° | 64.223 cm, 5.9454° |
| Scene: surface points / raw points | 549,840 / 8,465,750 | 549,792 / 8,465,923 |
| Trajectory max difference | – | 0.06 mm |
| Plan: rooms / walls | 8 / 70 | 8 / 72 |
| Footprint | 61.90 m² | **59.37 m²** |
| Largest room | 13.99 m² | **11.46 m²** |
| Wall gate, run 2 against run 1 | – | **68 / 72 (94%) [87, 98]**; 7 of 8 rooms identical to 0.1 cm² |

**Three runs of the same command on each capture.**

| Capture | Run | Footprint | Rooms | Walls |
|---|---|---|---|---|
| floor_only | `drift_on` | 61.90 m² | 8 | 70 |
| floor_only | `drift_on_rerun_nodamage` | **59.37 m²** | 8 | 72 |
| floor_only | `drift_on_rerun2` | 61.90 m² | 8 | 70 |
| with_ceiling | `drift_on` | 62.50 m² | 8 | 70 |
| with_ceiling | `drift_on_rerun2` | 62.56 m² | 8 | **74** |
| single_room | `drift_on` and `drift_on_rerun2` | identical (17.61 m²) | | |

The flip is intermittent. D-027 changes wall lengths only, not wall counts or areas, so these differences are
run-to-run.

**What it shows.**

- On a fixed scene the extractor is deterministic: re-run on each saved scene, it gives the same footprint both times
  (61.9009 / 61.9009 and 59.3681 / 59.3681).
- The drift solve is not bit-reproducible: it moves the poses by about 0.06 mm. The resulting change of a few dozen
  of 550,000 surface points is enough to flip one room's outline by 2.5 m².
- The same "small input → different topology" sensitivity is what `plan_v2_ablation.md` B-5 found with a 1.3 cm shift
  (41–48% of walls changed). Here it is shown to be triggered by a sub-millimetre change.

**Proposed fixes (not applied; pipeline code is out of scope for this task).**

1. Make the drift solve reproducible. The likely source is multi-threaded ICP or KD-tree reductions in
   `floorplan/recon/drift.py`. Fix the thread count for the ICP calls and sort the loop-candidate lists, then
   verify by checking that two runs give identical `T_wc` arrays.
2. More importantly, stabilise the outline: phase-averaged outlines, as proposed in `plan_v2_ablation.md`, "Part 4
   candidate". Fixing (1) alone would only hide the instability. A second capture always differs by more than
   0.06 mm.

### 4.2 Opening widths across the repeat pair

13 opening slots in the 8 matched rooms: 8 matched, plus 5 present only in with_ceiling (O8–O12, missed by
floor_only or phantoms in with_ceiling; there is no ground truth to say which).

| A | B | Kind A / B | Width A cm | Width B cm | Δ cm | ≤ 2 cm | z |
|---|---|---|---|---|---|---|---|
| O1 | O7 | door / door | 67.7 (measured) | 82.8 (measured) | +15.1 | no | 8.35 |
| O2 | O6 | door / door | 89.4 | 90.3 | +0.9 | **yes** | 0.51 |
| O3 | O2 | passage / passage | 74.6 (inferred) | 85.7 (inferred) | +11.1 | no | 1.54 |
| O4 | O3 | door / door | 88.3 | 90.5 | +2.2 | no (by 2 mm) | 1.22 |
| O5 | O4 | passage / passage | 116.1 (inferred) | 125.6 (inferred) | +9.6 | no | 1.32 |
| O6 | O1 | door / passage | 71.0 (measured) | 72.5 (inferred) | +1.5 | **yes** | 0.29 |
| O7 | O5 | passage / passage | 113.7 | 116.1 | +2.4 | no | 1.35 |
| O13 | O9 | window / window | 40.9 | 68.9 | +28.0 | no | 15.55 |

**Result.** 2 / 13 slots agree within 2 cm (15%; the gate needs 85% against ground truth). Of the matched openings,
2 / 8 agree (median |Δ| 6.0 cm). Of the 3 door pairs measured in both captures, 2 agree within 2.2 cm, and one differs by 15 cm.

**Interval check on openings.** 6 / 8 matched openings are within |z| ≤ 1.96. The two that are not are both confident
and wrong:

- door O1 / O7: 15 cm apart, z = 8.4;
- window O13 / O9: 28 cm apart, z = 15.6.

### 4.3 single_room against the matching rooms of the two big captures

single_room has 3 rooms: a bedroom, a study and a small bathroom. All 3 match rooms of both big captures. The
registration is from scene geometry.

| | vs with_ceiling | vs floor_only |
|---|---|---|
| Registration (inliers; tile median / p90) | 0.83; 2.8 / 4.1 cm | 0.83; 3.2 / 5.4 cm |
| Rooms matched | 3 / 3 | 3 / 3 |
| Walls pass / judged | **1 / 26 (3.8%)** [0.7, 18.9] | **0 / 33 (0%)** [0, 10.4] |
| Unmatched walls | 14 | 20 |
| Median \|Δ\| | 16.2 cm | 6.9 cm |
| Same topology n / pass / median | 5 / 1 / 1.8 cm | 6 / 0 / 4.3 cm |
| Openings within 2 cm (slots) | 1 / 4 (door 71.0 vs 70.9 cm) | 1 / 4 (door 72.5 vs 70.9 cm) |
| Same-topology interval coverage | 1.00 (n = 5) | 1.00 (n = 6) |

| single_room room | Area | with_ceiling | floor_only | Box Δ vs with_ceiling / floor_only (cm) |
|---|---|---|---|---|
| R1 bedroom | 8.31 m² (2.758 × 3.036) | R4 9.33 m² (3.018 × 3.170) | R4 9.26 m² (3.022 × 3.445) | −26.0, −13.4 / −26.4, −40.9 |
| R2 | 7.56 m² | R5 7.67 m² | R5 7.68 m² | +1.4, −7.3 / −2.4, −6.1 |
| R3 bath | 1.93 m² | R8 1.96 m² | R8 1.68 m² | +2.0, −7.5 / +17.0, −4.0 |

single_room's bedroom is about 1 m² (11%) smaller than the same room in both apartment captures. Its plan loses a
26 cm strip. This is the "single_room −1.8 m²" red flag already listed as open in `plan_v2_ablation.md`; it is
reproduced here and remains unresolved.

## 5. Ceiling heights (gate: ≤ 1.5 cm per room; spread across captures ≤ 1 cm)

Only with_ceiling observes ceilings. In floor_only and single_room every room reports `not_observed` with no value,
which is the correct behaviour: nothing is invented.

| with_ceiling room (area) | Ceiling m | 95% interval | Status | Share of the room at this level (other levels seen) |
|---|---|---|---|---|
| R1 (14.31 m²) | 2.481 | [2.454, 2.508] | measured | 36% (2.06 m on 14%) |
| R2 (12.81) | 3.098 | [3.071, 3.125] | measured | 42% (2.47 m on 24%) |
| R3 (11.17) | 3.078 | [3.051, 3.105] | measured | 90% (2.46 m on 13%) |
| R4 (9.33) | 3.091 | [3.064, 3.118] | measured | 96% |
| R5 (7.67) | 2.372 | [2.346, 2.399] | measured | 83% |
| R6 (3.39) | 2.393 | [2.366, 2.420] | measured | 100% |
| R7 (2.47) | 2.480 | [2.453, 2.507] | measured | 100% |
| R8 (1.96) | 2.485 | [2.458, 2.512] | measured | 100% |

Each interval is ±2.7 cm. It includes the D-021 residual systematic term: ±1.6 cm at 95%, widened for an unvalidated
device. The values include the +1.36 cm bias correction.

- **Absolute (≤ 1.5 cm): not verifiable on the sample**, because there is no ground truth. ARKitScenes evidence
  (`docs/modules/lidar_bias.md`, fix-loop table): after the D-021 correction, ceiling height errors on 5 laser-scanned
  rooms are −0.40, −0.68, +0.46, −0.14 and +0.29 cm, so 5 of 5 are within 0.7 cm. Before correction, 3 of 5 were
  within 1.5 cm. Two of those rooms were held out when the correction was chosen. Two caveats:
  - that test measures the room box plane-to-plane on an iPad, not this plan extractor on the sample captures' iPhone;
  - so it supports the sensor-plus-correction part of the claim, not the per-room extractor.
- **Spread across captures (≤ 1 cm): not measurable on the sample.** Only one capture observes the ceilings, so we
  cannot say "repeatable-but-biased" or "unrepeatable". The closest internal evidence is within with_ceiling, which
  walks the apartment twice: the ceiling level seen by different fragments spreads 6.04 cm with ARKit poses and
  0.54 cm after drift correction. A real cross-capture spread needs a repeat capture with the
  ceiling scanned.

## 6. Damage on clean captures (false positives)

The sample apartment has no staged damage. Every confirmed detection below was inspected in its surface image
(`<run>/damage/surfaces/*.jpg`) and is a false positive.

| Run | Detections | Confirmed (scope-generating) | Review | Scope items | Concealed flags | What it is |
|---|---|---|---|---|---|---|
| single_room (default) | 1 | 0 | 1: crack, conf 0.148 | **0** | 0 | Thin line on a wall; correctly kept at review (D-023) |
| floor_only (default) | 1 | **1: "water stain", conf 0.733** (R6-W10) | 0 | **2** | **1: "below window: possible seal leak"** | A dark object or shadow at a window edge: a confident false positive with a money-relevant scope line |
| with_ceiling (default) | 0 | 0 | 0 | 0 | 0 | – |
| floor_only `--no-drift` | 1 | 1: "crack", conf 0.557 (R1-W3) | 0 | 2 | 0 | Door-frame or skirting edge |
| with_ceiling `--no-drift` | 1 | 1: "water stain", conf 0.412 (R7-W2) | 0 | 1 | 0 | Object on a shelf above a glass screen |
| single_room (second run, current code with D-026) | 2 | 0 | 2: cracks, conf 0.581 and 0.148 | 0 | 0 | Under D-026 (crack threshold 0.65) the 0.581 crack is review; under the first-pass code (0.35) it would have been confirmed |
| floor_only (second run, current code) | 1 | **1: the same "water stain", 0.733** | 0 | **2** | **1** | Reproduced. D-026 does not touch stains |
| with_ceiling (second run, current code) | 0 | 0 | 0 | 0 | 0 | – |

**Result.**

- Default runs give 1 confirmed false positive in 3 clean captures, with the first-pass code and with the current
  code alike. It is the floor_only "water stain" at 0.733 on a dark object or shadow at a window edge, and it fires a concealed
  "window seal leak" flag and 2 scope items.
- D-026's evidence says that no false *stain* had appeared at any tier. This one is a false stain on a clean capture,
  so that statement no longer holds.
- **Proposed next step (not applied):** reject stain candidates whose colour change comes from a 3-D object rather
  than the wall surface. Either require LiDAR relief ≈ 0 *and* a planar fit residual at the wall, or test whether the
  region's depth is in front of the wall plane. A dark object on a sill sits in front of the plane.
- The single_room crack at 0.581 is review only because of D-026's 0.65 threshold. With the first-pass code it would
  have been confirmed, so the threshold change is doing real work here. The drift-OFF runs produce different false positives on different surfaces. So the detector's
output depends on the poses: the surface orthophotos change, and with them the candidates. D-023's 0.35 threshold
does not stop a dark-object "stain" at 0.73.

Small logging inconsistency: single_room logs `[damage] 1 regions, 0 flags, 2 scope items`, but its plan.json has
0 scope items. The log counts scope before the D-023 tiering removes the review-tier items.

## 7. Calibration (are the 95% intervals honest?)

The repeat pair is the only place on the sample where two independent estimates of the same quantity exist.
z = Δ / √(σ_A² + σ_B²), and a calibrated interval gives |z| ≤ 1.96 about 95% of the time.

| Quantity | n | Share \|z\| ≤ 1.96 | Median \|z\| | Reading |
|---|---|---|---|---|
| All matched walls | 54 | **0.69** | – | Under-covers. Different-topology walls (corner built from a different neighbour, tens of cm) are segmentation errors that no measurement σ covers |
| Same-topology walls | 33 | **0.94** | 0.35 | About nominal. The median \|z\| of 0.35 says typical intervals are about 2× wide |
| Opening widths (matched) | 8 | 0.75 | – | Two confident failures: z = 8.4 (door) and 15.6 (window) |
| single_room vs big captures, same-topology walls | 5 / 6 | 1.00 / 1.00 | 0.28 / 0.49 | Small n |
| Ceilings across captures | 0 | – | – | Not measurable (one capture sees ceilings) |
| Ceilings, drift OFF against drift ON (ablation) | 8 | 5 / 8 OFF intervals contain the ON value | – | Without drift handling, 2–3 ceilings are confidently wrong |

Absolute calibration against a laser is in `docs/modules/lidar_bias.md`: residual σ = 0.81 cm per interior distance
after correction, which the intervals carry. On the sample, interval coverage holds for measurement but not for
segmentation. Topology changes are not represented in any interval, and that is the biggest calibration gap at the
LiDAR tier.

## 8. What fails and why (summary)

1. **Repeatability: 17% of walls.** Segmentation and topology instability is the cause (32 unmatched, 21 different
   topology). It is exposed even by the same command run twice (94%, with one room flipping by 2.5 m²). Measurement
   on same-topology walls (2.0 cm) sits at the drift floor (2.0 cm registration tiles).
2. **Opening widths: 2 / 13 slots.** Detection differs between captures (5 unpaired openings), and two widths are
   confidently wrong: a door by 15 cm and a window by 28 cm.
3. **Damage: 1 confirmed false positive** with a concealed flag and scope lines on a clean capture.
4. **Not verifiable on the sample:** absolute openings and ceilings, and the cross-capture ceiling spread. ARKitScenes
   covers the ceiling bias only.
5. **What passes:**
   - drift accountability (method plus ablation: on floor_only the OFF plan loses a room and 15% of its area);
   - stitch internal checks (8 connected rooms, no overlaps, 8/8 rooms matched across captures);
   - same-topology interval coverage (0.94).

## 9. Notes for reviewers

- **Harness adapter (not a pipeline bug).** The published plan.json writes a room's `bbox_dims` as `{length, width}`.
  `scripts/judge_common._bbox_rows` expects the dataclass list form and crashes with `TypeError: string indices must be
  integers` on run_capture outputs. `bench_lidar_eval.plan()` converts the dict to a list before scoring.
  Proposed fix in `judge_common._bbox_rows`:
  `dims = a["bbox_dims"].values() if isinstance(a["bbox_dims"], dict) else a["bbox_dims"]`.
- **Run-to-run differences.** These runs are not bit-identical to `outputs/scenes_v2` or `outputs/plan_v2`. For
  example, the with_ceiling footprint is 62.50 m² here (with bias correction) against 62.65 m² there (without). The
  same repeat-pair code gives 15 / 86 here against 13 / 87 there. The source is §4.1.
