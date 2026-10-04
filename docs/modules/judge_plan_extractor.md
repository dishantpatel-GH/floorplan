# Judge: which plan extractor the LiDAR pipeline uses (plan_alpha vs plan_beta)

**Decision: `plan_beta` (space-first) is the base extractor, as a hybrid that ports four parts of `plan_alpha`.**
The ported parts are the wall-line detector, the door-jamb measurement, the ceiling double-layer rule and the
"inconsistency" term in the intervals.

It was a close call. On the gate that ranks first (repeatability) the two are statistically tied, and **both fail by
an order of magnitude**. The choice rests on three things:

- beta leads, by a small margin, on most of the finer repeatability numbers;
- beta does not invent space, which matters at the walk-in because of mirrors and glass;
- beta fails loudly and honestly on bad input.

Alpha is better at doors, ceilings under drift, coverage and speed. Those parts are what we take from it.

This document is self-contained: every number below comes from the scripts listed in section 2, and the outputs are
in `outputs/judge/`.

---

## 0. The answer in one table (repeat pair, drift-corrected scenes)

Same apartment, captured twice (floor_only and with_ceiling). Both extractors were run by the judge on the same
scenes and scored by the same harness (`floorplan/eval`). The gate is "every wall within max(1 cm, 0.5%)".

| Metric (drift-corrected poses) | plan_alpha | plan_beta | Better |
|---|---|---|---|
| Walls passing the gate (pass / judged) | 8 / 94 = **8.5%** [4.4, 15.9]% | 11 / 98 = **11.2%** [6.4, 19.0]% | tie (Fisher p = 0.63) |
| Same, ARKit poses as-is | 0 / 125 = 0% | 4 / 117 = 3.4% | beta (pooled over both pose sets: p = 0.14) |
| Matched walls: median \|Δ length\| | **6.5 cm** | 11.0 cm | alpha |
| Walls with the same neighbours in both captures ("same topology"): n, pass, median \|Δ\| | 31, 7, 4.3 cm | 32, 11, **3.0 cm** | beta |
| Walls whose corners come from different neighbours: n, median \|Δ\| | 19, 70 cm | 22, **22 cm** | beta |
| Walls with no partner at all | 44 | 44 | tie |
| Footprint, with_ceiling vs floor_only | 67.91 vs 65.75 m² (**3.2%**) | 63.06 vs 59.41 m² (5.8%) | alpha |
| Doors matched across captures, width within 2 cm | 2 of 6 | 1 of 5 | tie (too few) |
| Repeat-pair interval coverage (target 0.95) | 0.54 | 0.61 | both overconfident |
| Self-stability: same scene with 10% of points removed (pass rate vs itself) | 153/194 = 79% | 139/157 = **89%** | beta |
| Self-stability: same scene shifted 1.3 x 0.7 cm | **100%** | 96% | alpha |
| Self-stability: same scene turned 90° | 127/213 = 60% | 98/173 = 57% | tie |
| Unentered "rooms" added to the plan (with_ceiling / floor_only) | 2 (8.6 m²) / 2 (5.3 m²) | 0 / 0 | beta |
| Degenerate input (1.5 x 1.5 m crop) | 2 rooms, **4.4 m² from 2.25 m² of data** | empty plan, reason recorded | beta |
| Runtime, with_ceiling (machine loaded) | **37 s** | 44–48 s | alpha |

Figures to look at first: `outputs/judge/repeat_drift_on.png` (where the two captures disagree, per extractor),
`outputs/judge/plans_drift_on.png` (both extractors drawn by one renderer at the same scale) and
`outputs/judge/invariance.png` (each extractor against itself, after a rigid move or a re-sample).

---

## 1. Purpose and where it sits in the pipeline

```
capture ─► prepare_scene ─► drift correction ─► scene.npz ─► [plan extractor] ─► Plan JSON ─► export / eval
                                                              ▲
                              plan_alpha (boundary-first) ────┤  built in parallel (D-010)
                              plan_beta  (space-first)  ──────┘  THIS MODULE picks one, with evidence
```

The plan extractor turns a 3-D scene into rooms, walls, doors, windows and dimensions. Every gate depends on it,
and so does the walk-in test (30% of the score). Decision D-010 built it twice, in two ways that fail differently:

- **plan_alpha, boundary-first.** Find wall lines first. Extend them into a grid of cells, decide inside/outside
  with a graph cut, then split rooms at narrow necks.
- **plan_beta, space-first.** Find the free floor space first and split it into rooms at narrow necks (watershed).
  Then fit walls around each room.

The judge's job:

1. run both on identical inputs;
2. score them with the shared evaluation harness, in the order of criteria we were given;
3. pick one, or a hybrid;
4. list what to take from the other;
5. list the defects to fix next, ordered by impact on the gates.

The judge does not change either extractor. Its scripts only call their public `extract_plan` functions, sometimes
with a parameter override for an experiment.

Criteria, in the order given:

1. repeatability on the floor_only vs with_ceiling pair;
2. plausibility;
3. robustness for a cold run on an unseen capture;
4. explainability and code clarity;
5. runtime.

---

## 2. How it works, step by step

All scripts are run from the repo root with `../.venv/bin/python`.

### Step 1. Run both extractors on identical inputs (`scripts/judge_run.py`, `scripts/judge_common.py`)

- **What.**
  - Both extractors run on all three captures, twice: on the ARKit-pose scenes (`raw`) and on the drift module's
    corrected scenes (`drift_on`).
  - Each run writes `outputs/judge/runs/<variant>/<producer>/<capture>/plan.json`, plus `run.json` (runtime, any
    error) and the extractor's own `debug.png`.
  - 12 base runs, 5 in parallel (`outputs/judge/jobs.txt`).
- **Why.**
  - The authors' own outputs were produced at different times, some on different scenes. Beta had no drift-on
    runs at all.
  - A fair comparison needs the same scene, the same code version and the same harness.
- **Without it.** We would compare alpha-on-corrected-poses with beta-on-ARKit-poses. That is exactly what the
  authors' two summaries do, so their numbers cannot be put side by side.
- **Check.** Determinism. The judge's re-run reproduces each author's committed plan exactly: max |Δ| = 0.0 on every
  wall, for both extractors (`repeatability.json → determinism`).

### Step 2. Repeatability on the repeat pair (`scripts/judge_repeatability.py`)

1. **Registration.** floor_only is put into with_ceiling's frame with the scene-to-scene transform written by
   `scripts/register_scenes.py`:
   - raw: yaw 89.05°, residual 1.15 cm;
   - drift_on: yaw 88.93°, residual 0.93 cm.

   The transform comes from the reconstructed geometry, never from a plan, so neither extractor can make itself
   look better.
2. **Matching** (`floorplan/eval/match.py`):
   - rooms by polygon overlap (IoU ≥ 0.3), one-to-one (Hungarian);
   - each matched room re-aligned locally, which absorbs residual drift;
   - walls matched inside matched rooms.
3. **Gate** (`floorplan/eval/repeatability.py`): |Δ length| ≤ max(1 cm, 0.5% L). A wall with no partner in a
   matched room counts as a failure.
4. **Extra numbers the judge adds** (`judge_common.compare`), because a pass rate alone does not say *why* a plan
   fails:
   - **Same / different topology.** A matched wall has "the same topology" when both of its neighbours are also
     matched to each other, so both corners are built from the same walls.
     - Disagreement on these walls is a **measurement** problem.
     - Disagreement on the others, and the unmatched walls, is a **segmentation** problem.
     - The two need different fixes.
   - **Wilson 95% interval** on each pass rate, and a Fisher exact test between the extractors. This tells us
     whether "8 vs 11 passes" is a real difference or noise.
   - **Room bounding boxes** (what a homeowner checks with a tape), footprint agreement, and opening-width
     agreement.

### Step 3. Self-stability: invariance tests (`judge_run.py --shift / --rot90 / --keep-frac`, scored by `judge_repeatability.py`)

- **What.** The same scene, changed in a way that should not change the plan, is extracted again and compared with
  the original using the same harness.
  - **shift:** the whole scene moved by 1.3 cm x 0.7 cm.
  - **rot90:** the whole scene turned 90° about the vertical. An exact integer rotation, so no resampling.
  - **thin90:** a random 90% of the surface and raw points kept, fixed seed. This is the same walls sampled
    slightly differently: the closest thing to a second capture without drift or coverage changes.
- **Why.** The repeat pair mixes three causes: real capture differences, residual drift, and the extractor's own
  instability. The invariance tests isolate the third. Whatever an extractor changes on an identical scene is a
  floor that its repeatability can never get below.
- **Without it.** We could not tell "this extractor is unstable" apart from "these two captures differ".

### Step 4. Plausibility, robustness, runtime (`scripts/judge_plausibility.py`)

There is no ground truth, so plausibility is measured with checks that any floor plan must satisfy, computed
identically for both extractors:

- **Structure:** rooms do not overlap; every room has ≥ 4 walls; the adjacency graph is connected.
- **Floor recall:** the share of observed floor (up-facing surface within 5 cm of the floor, 5 cm raster) that lies
  inside some room. Low recall means seen space is missing from the plan.
- **Unentered rooms:** rooms the camera path never entered. They are space seen through a door, window or mirror.
  Plausible for a balcony; a phantom for a mirror.
- **Slivers:** rooms under 1.5 m², or narrower than 0.7 m.
- **Measurement state:** the share of wall lengths measured on raw points, and the number of openings with a
  measured width.
- **One renderer for both.** Both extractors are drawn by the same renderer at the same scale
  (`plans_<variant>.png`), so their different debug figures do not bias the eye.
- **Robustness runs:**
  - `crop_half`: the left half of with_ceiling, a partial capture;
  - `tiny`: a 1.5 x 1.5 m crop, a degenerate capture.

  Each `run.json` records whether the run finished and what it returned.

### Step 5. Code clarity (`scripts/judge_code_metrics.py`)

Code lines (no blanks, comments or docstrings), number of functions, functions longer than 50 lines, the longest
function, and the number of tunable parameters. These are proxies, read together with the code itself.

### Step 6. One-parameter experiments (`judge_run.py --set KEY=VALUE`, `scripts/judge_experiments.py`)

- **What.** Each suspected defect is tested by changing one public parameter of the extractor and re-scoring the
  repeat pair exactly like the default runs.
- **Why.** A defect list ordered "by impact on the gates" needs evidence of impact, not opinion. This is also the
  evidence format the Part 4 fix loop asks for: hypothesis, then a predicted number.

---

## 3. Decisions

| # | Decision | Options considered | Choice | Why | Evidence |
|---|---|---|---|---|---|
| J-1 | Which scenes to judge on | authors' existing outputs; raw scenes only; **raw and drift_on, re-run by the judge** | re-run both | The pipeline must report drift-on (poses used as-is fails the drift gate), but raw shows behaviour under bad poses | Authors' numbers were not comparable: alpha quoted drift-on, beta only raw |
| J-2 | Registration for the repeat pair | per-extractor plan registration (what `benchmark.py` can fall back to); **scene registration from geometry** | scene | Plan-to-plan registration lets a plan pick the transform that flatters it; the scene transform is independent of both plans | `outputs/eval/registration*/…json`: residual 1.15 / 0.93 cm |
| J-3 | Headline repeatability number | gate pass rate only; median \|Δ\| only; **pass rate + Wilson CI + topology split + footprint + bbox + openings** | many | Pass rate is the gate, but with ~95 walls a difference of 3 passes is noise. The topology split tells which fix is needed. | Fisher p = 0.63 (drift_on), 0.14 (both pose sets pooled) |
| J-4 | How to measure an extractor's own instability | none; repeat-pair only; **rigid moves + re-sampling of the same scene** | invariance tests | Separates the extractor's noise from capture differences and drift | `invariance.png`, `repeatability.json → invariance` |
| J-5 | What counts as an implausible room | visual check only; **plus measurable checks (floor recall, unentered rooms, slivers, connectivity)** | both | Visual checks are biased by the two authors' different debug figures. The numbers are reproducible. | `plausibility.json`, `plans_*.png` |
| J-6 | Final choice | alpha; beta; **beta + four alpha parts** | hybrid on beta | See section 5 "Why" | Sections 0 and 5 |
| J-7 | How to test a suspected defect | argue from the code; **one-parameter experiment on the repeat pair** | experiment | Gives a measured impact (and exposed that one "fix" was gaming the gate, Issue J-I-5) | `experiments.json` |
| J-8 | Bounding-box sides paired by size, not direction | by direction after registration; **by size** | by size | Independent of the 90° turn between captures. Wrong only for near-square rooms whose sides swap order, and then visibly (a large Δ), not hidden. | `judge_common._bbox_rows` |

---

## 4. Issues log

**J-I-1. The authors' own numbers could not be compared.**
- Symptom: alpha reported 8/50 walls passing (drift-on); beta reported "within 0.0–5.7 cm on the same surfaces"
  (raw, its own opposite-wall metric).
- Root cause: different scenes (drift-on vs raw), different metrics (gate pass rate vs opposite-wall distances),
  and different moments in their development.
- Possible fixes:
  - (a) trust each author's metric;
  - (b) ask both authors to re-run;
  - (c) re-run both in one harness.
- Chosen: (c). It is the only option that guarantees the same code version, scenes and metric. It cost about 5 min of
  compute.

**J-I-2. A sub-cell shift changed nothing for alpha. Is that real?**
- Symptom: alpha's shifted run matched the original exactly (88/88 walls, Δ = 0).
- Root cause: both extractors anchor their raster to the data's own extent. Alpha uses the 0.1 percentile of the
  points; beta uses the minimum point.
  - A rigid shift moves that anchor too, so the raster phase relative to the walls does not change.
  - For alpha the shift test is therefore only a floating-point test. Alpha passes it exactly; beta changes 6 walls
    (4 fail, 2 unmatched) on float-level differences, a sign of hard thresholds somewhere in its pipeline.
- Fix: the thin90 test was added. Removing 10% of the points moves the data extremes and the anchor, so it really
  changes the raster phase. It is the honest test of raster sensitivity:
  - beta: 139/157 walls pass against itself (89%);
  - alpha: 153/194 (79%). On with_ceiling alpha is only 65/100.

**J-I-3. Alpha's plans change when the same data is thinned by 10%.**
- Symptom: 21 failing and 14 unmatched walls on with_ceiling (`invariance.png`, top right).
- Root cause, from the figure:
  - the walls that change are the tilted ones (r1's left wall, r3, r4);
  - alpha accepts a wall tilt only if it makes a "clean plane" (decision A-9). That is a hard yes/no threshold;
  - with 10% fewer points some walls cross it, the line tilts or untilts, and both corners move.
- Experiment: tilt off (`--set max_tilt_deg=0`) on the repeat pair. Result: 6/94 pass (vs 8/94), median Δ 6.3 cm
  (vs 6.5), same-topology median 3.7 cm (vs 4.3).
- Conclusion: neutral on the gate. The tilt gate costs alpha self-stability without buying repeatability. Beta's
  choice (walls stay on the axes, tilt goes into the interval) is the safer one to keep.

**J-I-4. Beta merged two rooms in one capture and not the other.**
- Symptom: drift-on floor_only has 6 rooms against 7 in with_ceiling. The with_ceiling room R4 matches floor_only's
  R1 with IoU 0.53 (9.3 vs 16.6 m²), and every wall of that pair fails or is unmatched.
- Root cause:
  - beta only counts a vertical surface as a wall if it covers ≥ 40 cm of the **1.0–2.0 m** height band (decision
    D-B2, chosen so desks are not walls);
  - floor_only was captured with the phone aimed down (DATA_NOTES: median depth 1.1 m), so the partition between
    the two rooms has little surface above 1 m;
  - free space leaked through it, and the watershed merged the rooms.
  - Evidence: `runs/drift_on/beta/single_scan_floor_only__1a8384c3f6/debug.png` (R1, 6.09 x 3.47 m) and the band
    experiment below.
- Experiment: wall band from 0.6 m instead of 1.0 m (`--set wall_band_lo_m=0.6`):

  | | default (1.0 m) | band 0.6 m |
  |---|---|---|
  | rooms (with_ceiling / floor_only) | 7 / 6 | 10 / 11 |
  | rooms matched | 6 | 10 |
  | walls passing | 11 / 98 | **17 / 106** |
  | median \|Δ\| | 11.0 cm | **2.5 cm** |
  | walls within 2 cm | 12 | 27 |
  | same-topology median | 3.0 cm | **1.5 cm** |

- **But see J-I-5.** The band change is a diagnosis, not the fix.

**J-I-5. The best repeatability number came from a worse plan.**
- Symptom: band 0.6 m gave the best repeatability of any run, but with_ceiling's footprint fell from 63.1 to
  57.5 m² (−9%).
- Root cause (`plans_drift_on_band60.png`): furniture between 0.6 and 1.0 m (beds, wardrobes, counters) now counts
  as wall. Rooms are cut short, the same way in both captures. For example, the bedroom drops from 11.3 to 6.6 m².
  The result is repeatable and wrong.
- Lesson: repeatability can be gamed. The gate text itself warns: "repeatable-but-biased and unrepeatable both
  fail". Every repeatability experiment must also report footprint and floor recall. `judge_experiments.py` should
  print them; the doc lists this as a next step.
- Chosen: keep the 1.0 m band for free space, and get room-*separation* evidence from long straight wall lines
  instead. That is alpha's line detector (defect B-1).

**J-I-6. Alpha turned a 2.25 m² crop into 4.4 m² of rooms.**
- Symptom: on the degenerate 1.5 x 1.5 m input, alpha returned two rooms (3.12 + 1.28 m²). Their walls run along the
  edge of the raster, beyond any data (`runs/raw_tiny/alpha/…/debug.png`).
- Root cause:
  - the graph cut labels cells "inside" wherever there is interior evidence, and LiDAR rays cross the whole crop;
  - where no wall bounds the space, nothing stops the room until the raster's 30 cm margin.
  - The same mechanism produces alpha's unentered balcony room r5 on with_ceiling, which extends to the scene's
    edge at v = −4.2 m.
- Consequence: invented area in the footprint, labelled `inferred` but counted. In front of a large mirror, the
  "room" behind the glass would be counted the same way (alpha's own doc lists this as open).
- Beta on the same input: an empty plan, footprint `not_observed`, and the reason in `meta.dropped_regions`. That is
  the right behaviour, although the CLI should also say so loudly (defect B-8).

**J-I-7. Beta's ceiling is confidently wrong under drift; alpha flags it.**
- Symptom: raw with_ceiling bedroom. Beta: 2.958 m ± 1.7 cm, `measured`. Alpha: 3.064 m, `inferred`,
  [2.947, 3.079].
- Root cause: on ARKit poses the two passes see the ceiling 11 cm apart (vertical drift). Beta picks the level that
  covers the largest area; alpha detects two big layers within 20 cm and widens the interval (decision A-14).
- Evidence that alpha is right: on the drift-corrected scene both extractors give **3.065 / 3.066 m**. In fact all 7
  matched rooms agree within 0.1 cm between the two independent implementations (section 5). The 2.958 m was the
  drift layer.
- Chosen: port A-14 into beta (defect B-6). A ±1.7 cm claim on an 11 cm error is the "confident garbage" the case
  study penalises.

**J-I-8. The runtime numbers are noisy.**
- Symptom: the same beta run took 44 s in one batch and 101 s in another.
- Root cause: other agents share the 22 cores. The load average was 40–85 during the judge's batches, and the
  experiment batch ran 6 extractor jobs at once.
- Chosen: report runtimes from the same batch for both extractors, which ran under the same load, and treat them as
  relative. Beta reaching 101 s under heavy load is still a warning sign against the 90 s budget (defect B-9).

**J-I-9. A first version of the scripts could not import their own helper.**
- Symptom: `ModuleNotFoundError: scripts.judge_common`.
- Root cause: `scripts/` is not a package, and the import ran before the repo root was put on `sys.path`.
- Fix: `judge_common.py` puts the repo root on `sys.path` itself, and the scripts import it as a sibling module.

---

## 5. Results

### 5.1 Repeatability: repeat pair, both pose sets (`outputs/judge/repeatability.json`, `repeat_raw.png`, `repeat_drift_on.png`)

| Poses | Extractor | Rooms A / B / matched | Pass / fail / unmatched | Pass rate [Wilson 95%] | Median \|Δ\| | Same-topology n, pass, median | Different-topology n, median | Coverage |
|---|---|---|---|---|---|---|---|---|
| ARKit | alpha | 10 / 11 / 10 | 0 / 53 / 72 | 0% [0, 3.0] | 23.1 cm | 19, 0, 6.6 cm | 34, 39 cm | 0.25 |
| ARKit | beta | 7 / 7 / 7 | 4 / 33 / 80 | 3.4% [1.3, 8.5] | 27.4 cm | 13, 4, 1.9 cm | 24, 73 cm | 0.30 |
| drift-on | alpha | 9 / 10 / 8 | 8 / 42 / 44 | 8.5% [4.4, 15.9] | **6.5 cm** | 31, 7, 4.3 cm | 19, 70 cm | 0.54 |
| drift-on | beta | 7 / 6 / 6 | **11** / 43 / 44 | **11.2%** [6.4, 19.0] | 11.0 cm | 32, 11, **3.0 cm** | 22, **22 cm** | **0.61** |

Readings:

- **Both fail the gate badly.** The pass rates' Wilson intervals overlap almost completely, so the gate itself
  cannot separate the two extractors.
- **Drift correction matters more than the choice of extractor.**
  - Alpha's median |Δ| drops from 23 to 6.5 cm, beta's from 27 to 11 cm.
  - The registration between the captures improves too: the median local disagreement between the two scenes,
    measured tile by tile by the eval module, falls from 5.2 to 2.4 cm.
- **Where the failures come from.** About two thirds of the judged walls fail for *segmentation* reasons:
  unmatched, or different neighbours.
  - Alpha: 44 + 19 = 63 of 94.
  - Beta: 44 + 22 = 66 of 98.
- **Where the rooms agree, beta measures more repeatably.** Same-topology median: 3.0 vs 4.3 cm (drift-on), 1.9 vs
  6.6 cm (raw).
- **When the topology differs, beta's error is smaller.** Median 22 vs 70 cm. Beta's rooms stay rectangular; alpha's
  extended wall lines and tilted walls move whole corners.
- **Alpha's footprint is more repeatable.** 3.2% vs 5.8%: beta's floor_only merge (J-I-4) and its dropped
  partial spaces cost it here.
- **Room bounding boxes, the bedroom** (the one room both segment the same way): beta 4.859 x 2.997 vs
  4.859 x 3.001 m, within the gate on both sides. Alpha 4.823 x 2.998 vs 3.887 x 2.997 m: one side fails, because
  alpha cut that room differently in floor_only. All rows are in `repeatability.json → room_bbox`.
- **Doors, cross-capture width difference** (drift-on, matched doors):
  - alpha: −10.0, **0.6**, 4.0, 9.2, **1.3**, −10.0 cm (2 of 6 within 2 cm);
  - beta: −12.8, **0.0**, 11.4, 12.0, 8.5 cm (1 of 5).

### 5.2 Self-stability (`repeatability.json → invariance`, `invariance.png`)

Each extractor is compared against its own plan of the same scene.

| Test | alpha with_ceiling | alpha floor_only | beta with_ceiling | beta floor_only |
|---|---|---|---|---|
| shift 1.3 x 0.7 cm | 88 / 88 | 94 / 94 | 79 / 85 | 70 / 70 |
| rot 90° | 39 / 119 (rooms 10 → 9) | 88 / 94 | 54 / 93 | 44 / 80 (rooms 7 → 6) |
| thin to 90% of points | 65 / 100 | 88 / 94 | 78 / 87 | 61 / 70 |

Neither extractor is stable under a 90° turn of the same data. Both use watershed and raster scans whose tie-breaking
depends on scan order, and their room merges depend on single-cell widths. Alone, this caps their repeatability at
about 60% of walls, even with perfect captures. This belongs on the defect list (B-5) whichever extractor we keep.

### 5.3 Plausibility (`plausibility.json`, `plans_raw.png`, `plans_drift_on.png`)

| Drift-on | Rooms | Footprint m² | Floor recall | Unentered rooms (m²) | Doors / passages / windows | Opening widths measured | Overlap | Min walls | Connected | Median wall ± (95%) |
|---|---|---|---|---|---|---|---|---|---|---|
| alpha single_room | 3 | 24.62 | 0.975 | 0 | 2 / 0 / 0 | 2 of 2 | 0 | 6 | yes | 5.0 cm |
| beta single_room | 2 | 17.72 | **0.606** | 0 | 0 / 1 / 0 | 0 of 1 | 0 | 8 | yes | 10.2 cm |
| alpha floor_only | 10 | 65.75 | 0.981 | **2 (5.25)** | 6 / 0 / 5 | 9 of 11 | 0 | 4 | yes | 4.6 cm |
| beta floor_only | 6 | 59.41 | 0.940 | 0 | 3 / 3 / 2 | 2 of 8 | 0 | 10 | yes | 10.1 cm |
| alpha with_ceiling | 9 | 67.91 | 0.994 | **2 (8.57)** | 7 / 0 / 4 | 9 of 11 | 0 | 4 | yes | 4.8 cm |
| beta with_ceiling | 7 | 63.06 | 0.955 | 0 | 4 / 2 / 2 | 4 of 8 | 0 | 8 | yes | 9.8 cm |

On ARKit poses the adjacency graph is disconnected once for each extractor: alpha with_ceiling, beta floor_only.

What the figures show:

- **alpha:**
  - Covers everything seen: floor recall 0.97–0.99.
  - Adds space the camera never entered as rooms that count in the footprint: the balcony seen through the window
    (r5, 5.4 m², extending to the scene edge) and a 3.2 m² room seen through a door. These are flagged in the room
    label only.
  - On ARKit poses its walls tilt with the drift (`plans_raw.png`, floor_only r3 and r4).
  - Detects and measures more doors, jamb to jamb.
- **beta:**
  - Rooms are rectangular and conservative. It never adds an unentered room (D-B10).
  - Pays for this with omissions. In single_room it drops the 6–7 m² corridor seen from the doorway (recall 0.61).
  - Merges two rooms in floor_only (J-I-4) and, as its own doc reports, merges the bathroom behind a glass shower
    screen into the bedroom.
- **Ceilings agree almost exactly.** On drift-on with_ceiling the two independent implementations give the same
  per-room ceiling heights within 0.1 cm in all 7 matched rooms:

  | alpha / beta room | alpha (m) | beta (m) |
  |---|---|---|
  | r0 / R1 | 2.469 | 2.469 |
  | r1 / R2 | 3.082 | 3.083 |
  | r2 / R3 | 3.066 | 3.065 |
  | r3 / R4 | 3.087 | 3.087 |
  | r4 / R5 | 2.364 | 2.364 |
  | r6 / R6 | 2.380 | 2.380 |
  | r8 / R7 | 2.479 | 2.478 |

  This is not ground truth, but it is good evidence that the ceiling measurement itself is sound once the poses are
  corrected.

### 5.4 Robustness (`runs/raw_crop_half`, `runs/raw_tiny`, `plausibility.json → runs`)

- **No crashes.** Neither extractor crashed on any of the 28 runs: base, shift, rot90, thin90, half crop, tiny crop.
- **No capture-specific tuning.** Neither has capture-specific constants. Every parameter is physical and documented
  in `params.py`: 69 for alpha, 66 for beta.
- **Half crop** (a partial capture):
  - alpha: 8 rooms, 2 of them "seen through an opening, not entered";
  - beta: 5 rooms, none unentered.
- **Tiny crop** (a degenerate capture):
  - alpha: invents 4.4 m² of rooms from 2.25 m² of data (J-I-6);
  - beta: an empty plan with the reason recorded.

  For a cold run at the walk-in, failing honestly beats confident garbage.

### 5.5 Code clarity (`code_metrics.json`)

| | alpha | beta |
|---|---|---|
| Files / code lines | 13 / 1,353 | 12 / 1,670 |
| Functions / longer than 50 lines | 103 / 1 (`extract_plan`, 103 lines) | 120 / 0 (longest 45) |
| Tunable parameters | 69 | 66 |
| Public functions without docstring | 19 | 23 |

- **alpha:** the core idea is one clean energy (graph cut over a cell complex) and is elegant to explain. Its
  orchestration, though, is a single 103-line function.
- **beta:** smaller functions, but its overlap handling grew by patches. Its doc's I-4 lists four successive
  fixes: band, max shift, revert, snap.
- Both docs are thorough (15 issues each, every parameter justified).
- Verdict: a tie, with a slight edge to alpha's concept and to beta's code shape.

### 5.6 Runtime (seconds, same batch, 5 jobs in parallel plus other agents; load average 40–85 on 22 cores)

| | single_room | floor_only | with_ceiling |
|---|---|---|---|
| alpha, ARKit / drift-on | 10.4 / 10.6 | 29.8 / 24.4 | 37.4 / 37.0 |
| beta, ARKit / drift-on | 7.7 / 5.8 | 33.4 / 29.0 | 48.0 / 44.5 |

Both are under the 90 s budget; alpha is about 20–25% faster on the big captures. Under the heaviest load (the
experiment batch) beta took 75–101 s, which is a risk (B-9).

### 5.7 Experiments (`experiments.json`)

All on the drift-corrected repeat pair.

| Run | Rooms matched | Pass / fail / unmatched | Median \|Δ\| | Same-topology median | Footprint with_ceiling | Verdict |
|---|---|---|---|---|---|---|
| beta default | 6 | 11 / 43 / 44 | 11.0 cm | 3.0 cm | 63.1 m² | baseline |
| beta jog snap 6 → 15 cm | 6 | 6 / 34 / 34 | 13.2 cm | 3.5 cm | not computed (52 walls vs 78) | worse: snapping moves corners differently in each capture |
| beta wall band from 0.6 m | 10 | 17 / 51 / 38 | **2.5 cm** | **1.5 cm** | 57.5 m² (−9%) | confirms the merge root cause, but cuts rooms at furniture (J-I-5) |
| alpha default | 8 | 8 / 42 / 44 | 6.5 cm | 4.3 cm | 67.9 m² | baseline |
| alpha tilt off | 8 | 6 / 44 / 44 | 6.3 cm | 3.7 cm | not computed (same room count) | neutral |

Alpha's own experiment (its A-12) found the same about jogs: raising its jog tolerance from 12 to 20–30 cm did not
help. Bigger jog snapping is not the fix for either extractor.

### 5.8 The decision, and why

**Choose plan_beta as the base. Make it a hybrid by porting four parts of plan_alpha (section 5.9).**

Why, criterion by criterion:

1. **Repeatability: tie on the gate, lean to beta on the details.**
   - Pass rate: 11.2% vs 8.5% on drift-on; 15/215 vs 8/219 pooled over both pose sets. The difference is not
     significant (p = 0.63 and 0.14).
   - Beta is ahead on the measurements that point to the *fixable* parts:
     - measurement repeatability where the rooms agree: 3.0 vs 4.3 cm;
     - size of topology errors: 22 vs 70 cm;
     - self-stability under re-sampling: 89% vs 79%;
     - interval coverage: 0.61 vs 0.54.
   - Alpha is ahead on overall median |Δ| and footprint agreement.
   - Most important: beta's biggest repeatability failure has a **known, measured root cause with a demonstrated
     effect** (J-I-4: rooms matched 6 → 10, median |Δ| 11 → 2.5 cm when the partition is seen). That gives a
     concrete Part 4 fix-loop path.
   - Alpha's biggest failures (corner topology, the tilt gate, room extents running to the raster edge) are spread
     over several mechanisms.
2. **Plausibility: mixed, but beta's errors are the safer kind.**
   - Alpha is more complete and finds and measures more doors.
   - But alpha **adds space that was never entered**, with walls running to the scene edge. At the walk-in, a large
     mirror or a glass wall (both named in the brief's constraints) would add a phantom room to the stitched plan
     and the footprint.
   - Beta's errors are omissions and merges. They show up as missing or merged rooms, are recorded in
     `meta.dropped_regions`, and are fixable with better wall evidence (B-1).
3. **Robustness: beta.** Beta returns an empty, explained plan on degenerate input, where alpha invents area. It is
   also more stable on re-sampled data. Neither has capture-specific constants, and neither crashed.
4. **Explainability: tie.** Alpha's concept is cleaner; beta's code is better decomposed.
5. **Runtime: alpha** (20–25% faster). Beta is still inside the budget at normal load.

The decision is reversible. Both extractors emit the same `Plan` type, and the judge scripts re-run the whole
comparison in about 10 minutes. If the hybrid does not move the repeatability gate, re-run the judge.

### 5.9 What to borrow from plan_alpha (the hybrid)

| Alpha part | Beta defect it fixes | Why alpha's is better | Port effort |
|---|---|---|---|
| `lines.py`: 1-D peak lines per face sign over **0.3–2.0 m**, kept if occupied over ≥ 0.4 m | B-1 room merges, B-7 glass partition | A long straight vertical line is a wall even when it is only seen low down. Furniture rarely forms long straight lines in a Manhattan room. Separate face signs keep both sides of a partition. | Medium: use the lines as a "barrier" for beta's segmentation only, while carving and the 1.0 m band stay as they are |
| `doors.py`: contact detector between facing room edges, plus jamb-to-jamb width **inside the wall core** with a relative tall-face rule | B-3 opening widths | 9 of 11 widths measured at ±1.4 cm (beta: 2–4 of 8). The same door measures 0.911 vs 0.917 m across captures. | Medium: beta's thickness partners give the two facing faces, and the width code is self-contained |
| `heights.py` A-14: two big ceiling layers within 20 cm → `inferred`, interval covers both | B-6 ceiling under drift | It turns an 11 cm error claimed at ±1.7 cm into an honest interval (J-I-7) | Small |
| `walls.py` / A-8: an inconsistency term √(rmse² − noise²) in each wall's sigma | B-4 calibration | Widens the interval exactly where a face is smeared: curtains, two passes | Small |
| A-17 speed-ups: `bincount`, local KD-trees | B-9 runtime | with_ceiling went from 72 to 15 s on alpha | Small |

**Do not borrow:**

- Alpha's graph-cut extension of rooms to the raster margin (J-I-6).
- Its unentered-room inclusion.
- Its gated wall tilt (J-I-3).

---

## 6. Limitations, failure modes, and the defects to fix next

### 6.1 Defects in plan_beta, ordered by impact on the gates

| # | Defect | Gate hit | Evidence | Proposed fix (and why this one) | Predicted effect |
|---|---|---|---|---|---|
| **B-1** | Rooms merge when the partition is not seen above 1.0 m (aimed-down capture, glass, low furniture walls) | Repeatability: every wall of the merged rooms is unmatched. Also adjacency and stitched plan. | J-I-4. The band 0.6 experiment shows the effect: rooms matched 6 → 10, pass 11 → 17, median 11.0 → 2.5 cm. | Use alpha's long Manhattan lines (0.3–2.0 m, ≥ 0.4 m occupied) as segmentation barriers. Keep the 1.0 m band for free space. Unlike lowering the band, this does not turn furniture into walls (J-I-5). | Rooms 7 / 7 matched; pass rate about 15% (band60 reached 16%) with the footprint within 2% of today's. To be confirmed: run `judge_experiments.py` and also check footprint and recall. |
| **B-2** | Different jogs, notches and doorway bumps in the two captures, so corners come from different walls | Repeatability: 22 different-topology walls (median 22 cm), many of the 44 unmatched | Section 5.1. The jog-snap experiment made it worse (6 passes). | Make the room outline canonical *before* refinement: drop polygon edges with no raw-point support shorter than 30 cm, and merge collinear walls across them. Measure the room's dimensions plane to plane (the opposite-wall distance a laser measures). Bigger snapping is ruled out by two experiments. | Unproven. A proper experiment is needed: predict 30–40% of the 44 unmatched become matched. |
| **B-3** | Few opening widths measured on jambs; cross-capture width differences of 8–13 cm on 4 of 5 doors | Opening widths (≤ 2 cm on ≥ 85%) | Section 5.1 doors; 5.3 "opening widths measured" | Port alpha's jamb-in-core measurement (5.9) | Measured share from about 40% to about 80%. Cross-capture within 2 cm: at least alpha's 2 of 6 level. |
| **B-4** | Intervals too narrow: coverage 0.61 vs 0.95 target; RMS z ≈ 30 per eval.md | Calibration (scored at every tier; "confident garbage caps the score") | Section 5.1 coverage column | (a) The drift sigma from the drift module's measured 1.9–2.5 cm, not the 1 cm placeholder. (b) Alpha's inconsistency term. (c) Walls next to an unsupported or inferred corner get the inferred sigma on that end. Topology errors cannot be covered by a sigma, so B-1 and B-2 come first. | Coverage of same-topology walls about 0.9 |
| **B-5** | Order and raster-phase dependence: a 90° turn of the same scene keeps only 57% of walls; thinning by 10% flips 9–14% | Repeatability floor | Section 5.2 | Make the room-merge decision use a neck width measured on fitted lines, not a single raster cell width. Break watershed ties deterministically by position, not scan order. Add the rot90 and thin90 tests to CI. | Self-stability ≥ 95% on all three tests |
| **B-6** | Ceiling reported "measured ±1.7 cm" on a drift double layer that is 11 cm off | Ceiling height ≤ 1.5 cm; ceiling spread | J-I-7 | Port alpha's A-14 | Raw-pose ceiling becomes `inferred` with an honest interval. Drift-on is unchanged (already agrees with alpha within 0.1 cm). |
| **B-7** | A glass shower screen gives no wall, so the bathroom merges into the bedroom | Repeatability, room count, glass coverage required by the brief | plan_beta.md I-8 | Same mechanism as B-1: the screen's frame and lower returns form a long straight line. Add a "no-return band behind a supported line" flag for glass. | The bathroom becomes its own room |
| **B-8** | Degenerate or very partial input gives an empty plan with exit status OK | Walk-in usability | J-I-6, `runs/raw_tiny/beta` | Return a non-zero exit code and a top-level `meta.coverage_warning` when the footprint is under about 4 m² or the floor recall is under 0.5 | — |
| **B-9** | Runtime 44–48 s normally, 75–101 s under heavy load | 90 s budget | J-I-8 | Port alpha's A-17 speed-ups | About 25 s on with_ceiling under the same load |
| **B-10** | Partly seen spaces dropped entirely (single_room recall 0.61) | Stitched-plan completeness | Section 5.3 | Keep them as `partial` rooms (status `inferred`, excluded from the footprint, listed separately) rather than discarding them | Recall about 0.9 without adding phantom area to the footprint |

### 6.2 Real-property failure modes (brief: mirrors, glass, wet-look surfaces, low light)

- **Mirrors.** A mirror shows a reflected room behind the wall.
  - Alpha would count that space as an unentered room in the footprint (J-I-6).
  - Beta's D-B10 drops unentered regions unless 60% of their outline is measured wall. A reflected room's "walls"
    are reflections, so they are not measured from inside, and the region is dropped.
  - Both also run a reflection test on see-through regions.
  - Residual risk: a full-height wardrobe mirror next to a real doorway. Neither extractor has been tested on a
    known mirror. This needs an own capture.
- **Glass.**
  - Clear glass gives no or low-confidence returns, which are dropped (D-006). The space behind becomes "seen
    through".
  - Glass partitions merge rooms (B-7); windows are found by see-through and no-return tests.
  - A black TV also gives no return, which is why beta keeps no-return regions as `window_candidates`, not
    openings.
- **Wet-look and glossy floors.** They give phantom points below the floor. Both extractors use "seen from above /
  below" rules and fixed height bands, so these points do not enter walls. The floor level uses a robust peak.
- **Low light.** LiDAR depth does not depend on light, so the LiDAR-tier plan is unaffected. Low light hurts the
  photo and video tiers and ARKit tracking, which shows up as drift. Drift is the main repeatability limiter here:
  even same-topology walls differ by a median of 1.5–3 cm.
- **Clutter.** Furniture in the 1.0–2.0 m band (wardrobes, tall shelves) is taken as wall by both extractors. That
  shortens rooms consistently: repeatable but biased, J-I-5. Without ground truth this bias is invisible to
  repeatability. Only tape/laser ground truth, or ARKitScenes, can measure it.

### 6.3 Limits of this judgement

- **One repeat pair**, about 95 judged walls per extractor. Differences of a few walls are not significant, as the
  Fisher tests show. A second apartment would change the confidence more than any extra metric would.
- **No ground truth.** Plausibility is measured by consistency checks, not accuracy. A plan can be repeatable and
  wrong (J-I-5).
- **The repeat pair is not independent of drift.** Residual local misalignment between the corrected captures is
  2.4 cm median (eval registration, `tiles_summary`). That alone makes "within 1 cm" unreachable for many walls, whichever
  extractor runs. The repeatability gate also needs the drift module to improve.
- **Topology split assumption.** It relies on each room's `wall_ids` being in polygon order; this was verified for
  both extractors on with_ceiling (0 breaks in 86 and 78 walls).

### 6.4 What I would do next

1. Implement B-1 in beta.
2. Re-run `judge_experiments.py`, adding footprint and recall to its printout (J-I-5 lesson).
3. Then B-3 and B-6, which are small, well-evidenced ports.
4. Use B-1 as the **Part 4 fix-loop** candidate:
   - worst gate: repeatability, 11.2% of walls (drift-on);
   - root cause: J-I-4, with the band experiment as evidence;
   - predicted after the fix: about 15% of walls, with rooms matched 7 / 7.

   This is honestly short of the gate. The report must say why: residual drift of about 2.4 cm between the
   captures, plus topology (B-2).
5. Add the shift / rot90 / thin90 invariance tests to `tests/` as regression tests with a threshold, so that future
   changes cannot silently make the extractor less stable.

### Requested changes to shared code

- `docs/DECISIONS.md` D-010: replace "Evidence: … (pending)" with a pointer to this doc. Add a new entry, D-018
  "LiDAR plan extractor: plan_beta + alpha's lines, jambs, ceiling rule, inconsistency term", using the summary in
  section 5.8.
- `docs/COMMIT_PLAN.md`: append section 7's commits after the two extractors' commits.
- `scripts/benchmark.py` / `floorplan/eval/gates.py`: next to every repeatability number, report the footprint and
  floor-recall change. A repeatability gain that shrinks the plan is a red flag (J-I-5).
- `floorplan/eval/match.py`: optionally expose the same / different topology split (`judge_common._topology_split`).
  It is the most useful diagnostic for the fix loop.

---

## 7. Proposed commits (to replay in the real repo)

These come after both extractors' own commit lists (plan_alpha.md §7, plan_beta.md §7) and after the eval module's.
The history then shows both approaches being built, judged, and the hybrid chosen with evidence.

| # | Message | Files | Why |
|---|---|---|---|
| 1 | `exp(judge): shared runner that drives both plan extractors on identical scenes, with rigid-move, re-sample and crop perturbations` | `scripts/judge_common.py`, `scripts/judge_run.py` | Same inputs, same code version for both: the authors' numbers were not comparable (J-I-1) |
| 2 | `exp(judge): repeat-pair, invariance and determinism scoring through floorplan/eval, with topology split and Wilson CIs` | `scripts/judge_repeatability.py` | Criterion 1 with a plan-independent registration (J-2); the topology split tells segmentation from measurement errors |
| 3 | `exp(judge): plausibility, robustness and runtime checks, one renderer for both extractors` | `scripts/judge_plausibility.py` | Criteria 2, 3 and 5 without ground truth; one renderer avoids visual bias |
| 4 | `exp(judge): code-size and function-length metrics` | `scripts/judge_code_metrics.py` | Criterion 4, with measured proxies |
| 5 | `exp(judge): one-parameter experiments on the repeat pair (jog snap, wall band, wall tilt)` | `scripts/judge_experiments.py`, `scripts/judge_run.py` (`--set`) | Evidence of impact for the defect list; exposed the "repeatable but wrong" trap (J-I-5) |
| 6 | `docs(judge): choose plan_beta + four alpha parts; ordered defect list` | `docs/modules/judge_plan_extractor.md`, `docs/DECISIONS.md` (D-010 evidence, D-018) | The decision and its evidence, for the defense |
| 7 | `feat(plan): beta uses alpha's Manhattan wall lines as segmentation barriers` (tag `before-fix` on #6, `after-fix` here if it is the Part 4 fix) | `floorplan/plan/beta/freespace.py`, `segment.py`, `floorplan/plan/lines.py` (moved from alpha) | B-1 |
| 8 | `feat(plan): jamb-in-core door width from alpha` | `floorplan/plan/beta/openings.py`, `floorplan/plan/jambs.py` | B-3 |
| 9 | `fix(plan): ceiling double layer within 20 cm is inferred with a covering interval` | `floorplan/plan/beta/levels.py` | B-6 |
| 10 | `refactor(plan): retire plan_alpha's segmentation; keep its line detector and jamb code as shared modules` | `floorplan/plan/alpha/*` | Keeps the repo honest about what runs; the history keeps the alternative |

Commits 7–10 are future work: they are the hybrid, not part of this scratch judge.

---

## 8. Concepts to explain in the defense

- **Repeat pair.** Two captures of the same rooms. With no ground truth, "same room in, same plan out" is the one
  accuracy-like property we can measure. It catches instability, not bias: a plan that is wrong the same way twice
  passes.
- **Scene registration vs plan registration.** To compare two plans, one must be moved into the other's frame. If
  the transform came from the plans, a plan could look good just by moving itself onto the other. Registering the
  reconstructed scenes (the geometry) gives a transform neither plan controls.
- **Hungarian matching.** The optimal one-to-one pairing of rooms (by overlap) and of walls (by offset and overlap).
  "Optimal" means the total cost is minimal, so the result is reproducible and not greedy.
- **The gate, max(1 cm, 0.5% L).** A wall passes if the two captures' lengths differ by at most 1 cm, or 0.5% of its
  length when that is larger: walls longer than 2 m get more than 1 cm. A wall with no partner fails.
- **Wilson interval and Fisher exact test.** With about 95 walls, 8 vs 11 passes could easily be luck. The Wilson
  interval is a 95% range for a pass rate that behaves well near 0%. Fisher's test gives the probability of seeing
  a difference this big if the two extractors were really equal: p = 0.63, so we cannot tell them apart on the
  gate alone.
- **Topology vs measurement errors.** A wall's length is the distance between its two corners, and each corner is
  where it meets a neighbour.
  - If both captures build the corners from the same neighbours, any difference is how precisely the walls were
    measured.
  - If a small jog or notch appears in one capture only, the corner jumps to a different wall and the length
    changes by decimetres, however good the measurement.
  - The two need different fixes, so we count them separately.
- **Invariance test.** Feed the same data in a form that should not matter (moved, turned, 10% fewer points) and
  check the output does not change. Whatever changes is noise produced by the algorithm itself, and no capture can
  remove it.
- **Raster phase.** Both extractors draw the scene on a 2 cm grid anchored to the data's edge. Where a wall falls
  inside a cell depends on that anchor. Thresholds that look at single cells can then flip when the anchor moves
  by a millimetre.
- **Floor recall.** The share of the floor the scanner saw that ends up inside some room. Low recall means seen
  space was left out of the plan.
- **Unentered room.** Space the camera path never entered: seen through a door, window or mirror. Counting it as a
  room risks a phantom (mirror); dropping it risks a missing balcony. We prefer to flag it and keep it out of the
  footprint.
- **Interval coverage.** If our 95% intervals are honest, about 95% of repeat-pair differences fall within
  1.96 × the combined sigma. Coverage of 0.54–0.61 means the intervals are too narrow ("overconfident"), which the
  case study penalises at every tier.
- **"Repeatable but biased".** A change can make two captures agree better by making both wrong the same way.
  Lowering beta's wall band made rooms stop at the furniture in both captures (J-I-5). This is why every
  repeatability result must be read together with the footprint and coverage.
- **Space-first vs boundary-first.**
  - Space-first finds where you can stand, then puts walls around it. It is robust when walls are hidden, but
    rooms leak when a partition is not seen.
  - Boundary-first finds walls, then decides which regions between them are rooms. It gives cleaner corners, but
    can invent rooms where nothing stops them.
  - The hybrid takes beta's space-first segmentation, plus alpha's wall lines as hard barriers.

---

## How to reproduce everything in this document

```bash
cd floorplan-capture
PY=../.venv/bin/python
# 1. runs (about 10 min with 5 in parallel); the job lists are in outputs/judge/jobs*.txt
cat outputs/judge/jobs.txt outputs/judge/jobs_thin.txt | xargs -P 5 -L 1 $PY scripts/judge_run.py
cat outputs/judge/jobs_exp.txt | xargs -P 5 -L 1 $PY scripts/judge_run.py
$PY scripts/judge_run.py --producer beta --capture single_room__c00a170fe1 --variant drift_on \
    --set wall_band_lo_m=0.6 --tag band60 --no-render
# 2. scoring
$PY scripts/judge_repeatability.py          # repeatability.json, repeat_*.png, invariance.png
$PY scripts/judge_plausibility.py drift_on_band60   # plausibility.json, plans_*.png
$PY scripts/judge_code_metrics.py            # code_metrics.json
$PY scripts/judge_experiments.py drift_on_jog15 drift_on_band60 drift_on_notilt   # experiments.json
```

Note: `outputs/judge/jobs.txt` omits the two raw single_room runs, which were run first by hand as a smoke test,
with the same command form (`--producer {alpha,beta} --capture single_room__c00a170fe1 --variant raw`).
