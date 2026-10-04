# Benchmark: absolute accuracy of the LiDAR plan against a laser (ARKitScenes)

Part of the benchmark report (Deliverable 5). Script: `scripts/bench_arkitscenes.py`. Evidence: `outputs/benchmark/arkitscenes/`.
Every number below comes from runs made for this document on 2026-10-04 (02:00–02:45).

## 1. What this answers

The sample captures have no ground truth, so they can only show repeatability. Here, the **whole LiDAR-tier pipeline**
(the same steps `scripts/run_capture.py --tier lidar` runs) is run on 5 ARKitScenes rooms that also have a Faro laser scan, and
every number the plan reports (wall lengths, room box, ceiling height, area, perimeter) is compared with the same quantity measured on the laser.

| Room | Role for the bias correction (D-021) | Frames (posed) / keyframes | Plan produced |
|---|---|---|---|
| 47895909 | development (b was fitted on it) | 337 / 217 | 1 room, 6 walls |
| 42445884 | development | 288 / 200 | 1 room, 10 walls |
| 47331133 | development | 588 / 334 | 1 room, 10 walls, 2 openings |
| 47333561 | **hold-out** (analysed after b was pre-registered) | 637 / 371 | 1 room, 8 walls |
| 47430003 | **hold-out** | 172 / 102 | **no room** (see 5.1) |

**Important caveat on independence.** The correction b = 0.68 cm was fitted on room-box distances of the 3 development rooms, measured
on raw points by `validate_arkitscenes.py`. The plan measures with its own code (plan_beta v2 walls, its own ceiling estimator), so the
dev numbers here are not an exact re-test of the fit, but they are not independent either. Only the hold-out rows are a test. Because
47430003 produced no plan, the hold-out evidence is **one room**: 2 room dimensions, 5 wall lengths, 1 ceiling.

## 2. How it works

1. **Build (mirrors `run_capture.py`, LiDAR path).** `load_arkitscenes` (the adapter, nearest-pose mode), `correct_drift` (ON),
   then the body of `floorplan.pipeline.scene.build_scene` (keyframes, TSDF, raw points, `align_scene`), copied line for line,
   because `build_scene` only accepts a Stray folder. Then plan_beta v2 `extract_plan`, then `apply_to_plan(plan, LidarBiasConfig.load())` (D-021), then
   `widen_plan(plan, 0.0, ...)`. Two plans are saved from the **same** extraction:
   - `plan_corrected.json`: the default pipeline;
   - `plan_raw.json`: what `--no-bias-correction` produces.

   Damage (D-023) is skipped, because it does not change walls, rooms or heights. The run uses `device_validated=False`, which is what
   `run_capture` uses. Strictly, the iPad is the validated device, so the intervals are wider here than they need to be.
2. **Registration of the laser into the plan's frame.** This uses `validate_arkitscenes.register_laser`, unchanged: the laser is levelled
   and Manhattan-aligned, then 4 yaws plus FFT, then point-to-plane ICP. ICP inlier RMSE was 1.1–1.2 cm and fitness 0.53–0.68. Distances
   between facing planes do not depend on the translation (lidar_bias.md §5.3).
3. **Ground truth on the laser, item by item, following the plan's own definitions.**
   - **Wall position.** For each plan wall line, the laser position is found with the validation's robust idea: a 1 cm histogram peak
     nearest the plan line, then the trimmed-mean schedule 3 / 2 / 1.5 / 1 cm. The laser points used:
     - have a normal facing the room, within 37° of the wall normal;
     - lie within ±12 cm of the line, inside the wall span with 15 cm left out at each corner;
     - lie at tape height, 0.8–1.6 m above the floor (the plan's own measuring band, V2-D5), falling back to 0.3–2.0 m if needed.
   - **GT wall length.** The corners are the intersections of consecutive GT wall lines, and the GT length is the corner-to-corner
     distance, which is the plan's definition. A wall's GT is missing if it, or either neighbour, has no laser surface.
   - **GT room box.** For each axis, the laser positions of the plan walls that form the polygon's two extremes.
   - **GT ceiling height.** Laser ceiling level minus laser floor level. Each level is a dominant 1 cm peak plus a trimmed mean, using
     up- or down-facing laser points inside the plan polygon shrunk by 15 cm.
   - **Cross-check.** These laser heights agree with the lidar_bias room-box laser heights to within 0.2–3.2 mm: 2.3632 vs 2.3608,
     2.2613 vs 2.2615, 2.7014 vs 2.6982 and 2.5831 vs 2.5838 m.
4. **Scoring.** For each item: error = plan − laser. Reported: median |error|, mean (signed), share within 1, 1.5 and 2 cm, coverage
   (the share of laser values inside the plan's 95% interval), and the median interval half-width. Dev and hold-out are shown separately,
   with and without the correction, and with drift correction on and off.

## 3. Regenerate

```
PY=../.venv/bin/python
cd floorplan-capture
for r in 47895909 42445884 47331133 47333561 47430003; do
  env -u PYTHONPATH $PY scripts/bench_arkitscenes.py build   --room $r              # ~10-45 s pipeline + ~10 s laser
  env -u PYTHONPATH $PY scripts/bench_arkitscenes.py measure --room $r
  env -u PYTHONPATH $PY scripts/bench_arkitscenes.py build   --room $r --no-drift   # drift ablation
  env -u PYTHONPATH $PY scripts/bench_arkitscenes.py measure --room $r --no-drift
done
env -u PYTHONPATH $PY scripts/bench_arkitscenes.py diag --room 47430003             # diagnostic only (relaxed rule)
env -u PYTHONPATH $PY scripts/bench_arkitscenes.py summary
# or everything in one go:  env -u PYTHONPATH $PY scripts/bench_arkitscenes.py all
```

Data: `../data/arkitscenes/raw/{Training,Validation}/<video_id>` and `../data/arkitscenes/laser_scanner_point_clouds/<visit>/<scan>.ply`
(fetched for lidar_bias, see docs/modules/lidar_bias.md).

Outputs:
- `<room>/`: `plan_raw.json`, `plan_corrected.json`, `plan.png`/`.svg`, `scene/`, `laser_registered.npz`, `build_report.json` (timings, drift report, registration), `accuracy.json` (every item, the laser wall fits, the GT polygon), `overlay.png`;
- `<room>_nodrift/`: the same, with drift correction off;
- `47430003_diag_unvisited0.4/`: the diagnostic for the empty plan;
- `items.csv` (one row per item and variant), `summary.csv` / `summary.json` (pooled tables, timing), `errors_by_kind.png`, `overlays_all_rooms.png`.

## 4. Results (drift correction ON, the default pipeline)

### 4.1 Per item (cm, plan − laser)

Room box and ceiling:

| Room | Role | Item | Laser (m) | Raw error | Corrected error (default) | Corrected 95% half-width | Laser value in interval? |
|---|---|---|---|---|---|---|---|
| 47895909 | dev | box length (x) | 2.4754 | −0.92 | +0.44 | 6.41 | yes |
| 47895909 | dev | box width (z) | 1.3535 | −0.21 | +1.15 | 6.58 | yes |
| 47895909 | dev | **ceiling** | 2.3632 | −3.50 | **−2.14** | 2.69 | yes |
| 42445884 | dev | box length (x) | 1.9848 | −0.93 | +0.43 | 6.38 | yes |
| 42445884 | dev | box width (z) | 1.7128 | −1.67 | −0.31 | 6.85 | yes |
| 42445884 | dev | ceiling | 2.2613 | −2.18 | −0.82 | 2.70 | yes |
| 47331133 | dev | box length (x) | 2.1902 | −1.15 | +0.21 | 6.54 | yes |
| 47331133 | dev | box width (z) | 2.0737 | −2.04 | −0.68 | 6.38 | yes |
| 47331133 | dev | ceiling | 2.7014 | −1.13 | +0.23 | 2.68 | yes |
| 47333561 | **hold-out** | box length (z) | 3.0412 | −2.54 | −1.18 | 6.39 | yes |
| 47333561 | **hold-out** | box width (x) | 1.6856 | −1.55 | −0.19 | 5.83 | yes |
| 47333561 | **hold-out** | ceiling | 2.5831 | −1.32 | +0.04 | 2.68 | yes |
| 47430003 | **hold-out** | everything | – | no plan | no plan | – | – |

Wall lengths: all 34 plan walls are in `items.csv`. 19 have laser GT; the other 15 lack it because the wall or a neighbour has no
laser surface. Errors in cm, raw → corrected:

| Room | Walls with GT: raw → corrected |
|---|---|
| 47895909 | W3 −0.21→+1.15, W4 −0.92→+0.44, W5 −0.56→+0.80 |
| 42445884 | W1 −1.55→−0.19, W2 +0.43→+1.79, W3 +0.62→+1.98, W4 −2.11→−0.75 |
| 47331133 | W1 −0.86→+0.50, W2 −2.84→−1.48, W3 +0.07→+1.43, **W4 +8.12→+9.48**, W5 −0.37→+0.99, **W6 −10.17→−8.81**, W10 −0.01→+1.35 |
| 47333561 (hold-out) | W1 −1.74→−0.38, W2 −1.03→+0.33, W3 +0.19→+1.55, W4 −2.43→−1.07, W8 −2.54→−1.18 |

### 4.2 Pooled

| Set | Item | Correction | n (with GT / plan items) | Median \|e\| | Mean e | ≤1 cm | ≤1.5 cm | ≤2 cm | Coverage of 95% interval | Median half-width |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | room box | raw | 6/6 | 1.04 | −1.15 | 50% | 67% | 83% | 100% | 6.1 |
| dev | room box | **corrected** | 6/6 | **0.43** | +0.21 | 83% | **100%** | 100% | 100% | 6.5 |
| hold-out | room box | raw | 2/2 | 2.04 | −2.04 | 0% | 0% | 50% | 100% | 5.8 |
| hold-out | room box | **corrected** | 2/2 | **0.68** | −0.68 | 50% | **100%** | 100% | 100% | 6.1 |
| dev | wall length | raw | 14/26 | 0.74 | −0.74 | 64% | 64% | 71% | 86% | 6.1 |
| dev | wall length | corrected | 14/26 | 1.25 | +0.62 | 43% | 71% | 86% | 86% | 6.5 |
| hold-out | wall length | raw | 5/8 | 1.74 | −1.51 | 20% | 40% | 60% | 100% | 6.0 |
| hold-out | wall length | corrected | 5/8 | 1.07 | −0.15 | 40% | 80% | 100% | 100% | 6.4 |
| dev | ceiling | raw | 3/3 | 2.18 | −2.27 | 0% | 33% | 33% | **33%** | 1.7 |
| dev | ceiling | corrected | 3/3 | 0.82 | −0.91 | 67% | 67% | 67% | 100% | 2.7 |
| hold-out | ceiling | raw | 1/1 | 1.32 | −1.32 | 0% | 100% | 100% | 100% | 1.7 |
| hold-out | ceiling | corrected | 1/1 | 0.04 | +0.04 | 100% | 100% | 100% | 100% | 2.7 |
| all | wall length | raw | 19/34 | 0.92 | −0.94 | 53% | 58% | 68% | 89% | 6.1 |
| all | wall length | corrected | 19/34 | 1.15 | +0.42 | 42% | 74% | 89% | 89% | 6.4 |
| all | ceiling | corrected | 4/4 (5 rooms) | 0.53 | −0.67 | 75% | 75% | 75% | 100% | 2.7 |

What it shows:

- **Room box (wall to wall).** The correction does what it was designed to do.
  - The raw plan reads short by 1.15 cm (dev) and 2.04 cm (hold-out).
  - Corrected, all 8 dimensions are within 1.5 cm, with median |e| 0.43 cm.
  - The pre-registered hold-out prediction (mean 0.0 ± 0.7 cm) holds on the 2 plan dimensions: −1.18 and −0.19 cm.
- **Wall lengths.** The correction makes the median |e| *worse* (0.92 → 1.15 cm), because it is applied wrongly to walls with a reflex corner (bug 1 in §6).
- **Ceiling.** The correction removes most of the shortfall: mean −2.03 → −0.67 cm over 4 rooms. The one remaining failure is −2.14 cm on 47895909, and it comes from drift correction (§4.4).

### 4.3 The wall-length correction is wrong at reflex corners (evidence for bug 1)

Moving every wall outward by b lengthens a wall by +b at each **convex** end and shortens it by b at each **reflex** end. Only a wall
with two convex ends gains +2b. `apply_to_plan` adds +2b to every wall. The table below excludes the 2 structural outliers (|raw| > 5 cm, §5.2):

| Wall type | n | Raw mean | Default (+2b everywhere) mean | Corner-aware (proposed) mean | ≤1.5 cm raw / default / corner-aware |
|---|---|---|---|---|---|
| both ends convex | 10 | −1.20 | **+0.16** | +0.16 | 50% / 100% / 100% |
| one reflex end | 7 | **−0.55** | +0.81 | −0.55 | 86% / 57% / 86% |
| all 17 | 17 | −0.93 | +0.43 | −0.13 | 65% / 82% / **94%** |

- Walls with a reflex end are, as predicted, **not** short raw (−0.55 cm). The default correction over-lengthens them by about 2b.
- The corner-aware rule gives median |e| 0.80 cm, 94% within 1.5 cm and 94% within 2 cm on these 17 walls.
- **This is a proposed fix measured offline**, not the pipeline's output.
- With n = 7 the reflex-end mean has a wide spread, but the direction and size match the geometry exactly.

### 4.4 Drift correction on vs off (corrected values)

| Item (all rooms with a plan) | Drift ON: median \|e\| / ≤1.5 cm / coverage | Drift OFF: median \|e\| / ≤1.5 cm / coverage |
|---|---|---|
| room box (8) | 0.43 / 100% / 100% | 0.52 / 88% / 88% |
| wall length | 1.15 / 74% / 89% (19 with GT) | 1.16 / 78% / 89% (18 with GT) |
| ceiling (4) | 0.53 / 75% / 100% | 0.51 / **100%** / 100% |

The drift correction matters in two places, in opposite directions:

- **It hurts the ceiling of 47895909.** The error goes from −0.99 cm with drift off to −2.14 cm with drift on (corrected), which fails the gate.
  - The drift report shows why. Across the 9 fragments, the ceiling spread (MAD) goes from **0.06 cm with ARKit's poses to 0.55 cm after correction**,
    while the floor spread goes from 0.65 to 0.47 cm.
  - The correction uses floor anchors only (0 Manhattan anchors in this room). Levelling an uneven or partly occluded floor pushes fragments
    vertically, and that spreads the ceiling.
  - Room 47333561 shows the same pattern: ceiling MAD 0.80 → 1.07 cm.
- **It fixes a structural error in 47333561.** With drift off, the plan has 10 walls instead of 8, and the box width reads **+9.4 cm** (outside its interval).
  With drift on, the error is −0.19 cm.
- **Overall, keep drift ON.** The ceiling side-effect is recorded as an open issue (§7).

### 4.5 Timing (drift ON, CPU shared with other jobs, load average ≈ 68 on 22 cores)

| Room | Drift | Scene | Extract | Pipeline total | Laser load + registration (benchmark only) |
|---|---|---|---|---|---|
| 47895909 | 14.0 s | 4.9 s | 3.1 s | 25.7 s | 7.9 s |
| 42445884 | 4.7 s | 4.2 s | 2.6 s | 15.2 s | 10.7 s |
| 47331133 | 18.0 s | 9.8 s | 5.0 s | 38.5 s | 9.5 s |
| 47333561 | 23.9 s | 8.8 s | 4.7 s | 43.7 s | 9.1 s |
| 47430003 | 1.8 s | 2.4 s | 1.1 s | 7.3 s | 10.3 s |

## 5. What the plan extractor gets structurally wrong in these small rooms

See `overlays_all_rooms.png` and the per-room `overlay.png`.

1. **47430003 (hold-out): no room at all.**
   - The capture is 17 s with 172 poses. The camera looks into a small room without entering it.
   - The only region is dropped: "never entered and only 48% of its boundary is measured wall" (rule `unvisited_min_enclosure = 0.6`).
     The plan carries `coverage_warning`, which is the honest behaviour.
   - A diagnostic re-extraction with the rule relaxed to 0.4 (`47430003_diag_unvisited0.4/`; not the pipeline) does not rescue it. It gives a
     0.6 × 3.9 m strip that lies mostly **outside** the laser's room: 5 of 8 walls have no laser surface, and 0.18 m² of the laser floor lies inside it.
   - So relaxing the rule would only make things worse. For the gates this room is a **miss**.
2. **A jog that is not where the laser puts it (47331133 W5).**
   - The plan has a 9.6 cm step (W5). The laser surface nearest it lies 9.2 cm further out.
   - Both neighbouring walls are therefore wrong by about 9 cm: W4 +8.1 cm and W6 −10.2 cm.
   - W4 is outside its interval, and W6 is outside its interval too. These are the only 2 interval misses among the lengths.
3. **Walls with no laser surface behind them** (phantom or closing walls):
   - 47895909 W1 (13 cm jog);
   - 42445884 W6 (23 cm) and W9 (19 cm), both short jogs;
   - 47331133 W8 (13 cm);
   - 47333561 W6 (1.35 m).

   47333561's W6 is the **Manhattan assumption failing**: the laser shows a diagonal wall (about 60° off-axis) between (0.2, 1.5) and
   (0.45, 0.6), and the plan replaces it with an axis-aligned wall at x = 0.44. Its two neighbours (W5, W7) therefore have no GT length.
4. **Many tiny jogs.**
   - 8 of 34 walls are shorter than 25 cm (9–24 cm).
   - In bathrooms these are often cabinets or vanity faces read as wall. They add corners, and each corner is one more place the
     topology can differ from the laser.
   - `canonicalize` removes steps of 10 cm or less only when unsupported. These are supported by raw points, just not by a wall.
5. **Rooms truncated at the capture's edge.** The laser sees floor beyond the plan in 47895909, 42445884 and 47331133. The plan closes the room with a wall where the iPad stopped seeing:
   - 47895909 W4 has laser support on only 35% of its span;
   - in 42445884 the free space continues to z ≈ 3 m past W7;
   - 47331133 W6 has laser support on 32% of its span, and a large area lies beyond it.

   Part of this is the scanner seeing through doorways into neighbouring spaces, which the plan rightly leaves out. Without an annotated
   floor plan, how much is a true miss is not measured here.
6. **Openings.** The plans contain 2 openings in total (47331133). Opening widths were **not** scored against the laser (no jamb GT
   extraction in this time box), so this evidence says nothing about the opening gate.

## 6. Bugs found (pipeline code not modified; proposed fixes)

**Bug 1. `floorplan/uncertainty/bias.py::apply_to_plan` adds +2b to every wall length.**

- **Problem.** The rule is right only for walls whose two corners are both convex. At a reflex corner the length *shrinks* by b.
- **Evidence.** §4.3: reflex-end walls go −0.55 → +0.81 cm under the default, while convex-convex walls go −1.20 → +0.16 cm.
- **Proposed fix.** Per wall, shift by b × (s_start + s_end), where s = +1 for a convex corner and −1 for a reflex corner, taken from the room polygon. Then:
  - convex-convex walls: +2b;
  - walls with one reflex end: 0;
  - walls with two reflex ends: −2b.

  The perimeter (+8b) and area (+bP + 4b²) rules are already right for any orthogonal polygon, and so is the room box (+2b).

  ```python
  # in apply_to_plan, replace the per-wall loop:
  walls = {w.id: w for w in plan.walls}
  for r in plan.rooms:
      V = np.asarray(r.polygon, float); n = len(V)
      orient = np.sign(np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1]))
      def s(k):   # +1 convex, -1 reflex corner at vertex k
          a, b_ = V[k] - V[k - 1], V[(k + 1) % n] - V[k]
          return 1 if np.sign(a[0] * b_[1] - a[1] * b_[0]) == orient else -1
      for k, wid in enumerate(r.wall_ids):
          ends = s(k) + s((k + 1) % n)          # 2, 0 or -2
          _shift_and_widen(walls[wid].length, ends * cfg.inward_bias_m, sig, cfg, "wall length between corners (corner-aware)")
  ```

  The sigma can stay `sig` for every wall, because the residual systematic term does not cancel at a reflex corner. Corner-aware
  improves within 1.5 cm from 82% to 94% on the 17 non-structural walls.

**Bug 2. `scripts/run_capture.py --no-bias-correction` skips `apply_to_plan` entirely**, instead of calling it with `enabled=False`.

- **Problem.** `bias.py` documents that the disabled mode "widens the intervals one-sidedly (toward longer) by the uncorrected bias, so
  it still covers the truth". `run_capture` never reaches that code, so raw intervals carry no bias term.
- **Evidence.** Raw ceiling coverage is **2 of 4** (47895909: −3.50 cm against a ±1.71 cm interval; 42445884: −2.18 cm against ±1.72 cm).
  Corrected coverage is 4 of 4.
- **Proposed fix.** Replace the `if ... and not a.no_bias_correction:` block with:
  ```python
  if a.tier == "lidar":
      from floorplan.uncertainty.bias import LidarBiasConfig, apply_to_plan
      apply_to_plan(plan, LidarBiasConfig.load(enabled=not a.no_bias_correction))
  ```

**Not a bug, a finding.** For iPad captures, `run_capture` uses `device_validated=False`. That adds b in quadrature to the systematic
sigma, although the iPad *is* the validated device class. On ARKitScenes the intervals are therefore slightly wider than needed. That is harmless
for coverage and is left as it is.

## 7. Gates-style summary (LiDAR tier, absolute accuracy against the laser)

| Gate / claim | Evidence here | Verdict |
|---|---|---|
| **Ceiling height ≤ 1.5 cm per room** | Default pipeline: 47331133 +0.23, 42445884 −0.82, 47333561 (hold-out) +0.04, **47895909 −2.14**, 47430003 no plan. With drift OFF: −0.99, −0.71, +0.30, −0.22 (4/4). Raw (no correction): 2 of 4 within 1.5 cm | **Not supported as shipped: 3 of 5 rooms pass.** It fails in 47895909 because drift correction spreads the ceiling (§4.4), and 47430003 has no plan. The bias correction itself is validated: hold-out +0.04 cm |
| Ceiling: repeatable but biased, or unrepeatable? | Raw ceilings are short in 4 of 4 rooms (−1.1 to −3.5 cm): a **bias** (≈ −2b), which D-021 removes. The remaining scatter (σ ≈ 1 cm) is room to room | Repeatable-but-biased raw; after correction, unbiased with σ ≈ 1 cm |
| Wall-to-wall (room box) accuracy | Corrected: 8 of 8 within 1.5 cm, median 0.43 cm. Hold-out 2 of 2 within 1.5 cm (−1.18, −0.19). Raw: 4 of 8 within 1.5 cm | **Supported** for rooms the extractor gets right, within the 2 cm tolerance used for the opening gate; ~1 cm only for 75% |
| Wall lengths | Corrected: 74% within 1.5 cm, 89% within 2 cm (19 walls with GT; 15 of 34 plan walls have none). Corner-aware fix: 94% / 94% on the 17 non-structural walls | Partly supported. Bug 1 costs about 0.5 cm on walls with a reflex corner. 2 structural misses of about 9 cm |
| Topology in small rooms | 1 of 5 rooms not produced. 1 non-Manhattan wall squared off. 1 misplaced 10 cm jog. 8 of 34 walls < 25 cm | **The main source of large errors**, not the sensor |
| Calibration (95% intervals) | Lengths: coverage 89% (2 misses = structural), but half-widths of about 6.4 cm against a median \|e\| of about 1 cm, so intervals are **about 4× conservative**. Room box 100% (8/8). Ceiling corrected 100% (4/4) with ±2.7 cm. Ceiling raw 50% (bug 2) | Calibrated or conservative wherever the topology is right. Wall and box intervals are too wide to be useful against a 1–2 cm gate. The drift term (2 cm per wall across passes, V2-D6) dominates them |
| Bias correction D-021 on plan quantities | Hold-out room box −2.04 → −0.68 cm, ceiling −1.32 → +0.04 cm. Dev room box −1.15 → +0.21 cm | **Confirmed** on plan outputs (one hold-out room) |

### Open items
- Run more ARKitScenes rooms. The hold-out is one room, and 5 ceilings is a small sample.
- Look at the drift correction's floor anchoring in small rooms with clean ARKit poses. For example, skip the correction when the ARKit ceiling and floor
  MADs are already below 0.3 cm, or add ceiling anchors.
- Wire bug fixes 1 and 2, then rerun `bench_arkitscenes.py all`.
- Score opening widths against laser jambs.
- The output folder is about 0.6 GB (cached scenes and registered laser), all regenerable.
