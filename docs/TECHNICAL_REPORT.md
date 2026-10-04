# Technical report: phone captures to dimensioned, stitched floor plans

**DRAFT 3, 4 Oct 06:45 IST.** Cap: 6 pages. Every number cites the file it comes from. Paths are relative to `docs/`
unless they start with `outputs/`. "REPORT" means `outputs/benchmark/final/REPORT.md`: the final benchmark, run on one
code fingerprint (`57bec9dc`) from 05:08 to 06:28 and scored at 06:28. Text in **[PENDING: ...]** is a placeholder,
and nothing in a placeholder is a claim. Three things are still pending: the own capture against tape, the
head-to-head, and the declared fix loop.

## 1. Architecture

There is one back-end and three front-ends (D-003). The tiers differ only in **how they obtain metric geometry** and
**how wide their intervals are**. Everything after the scene is shared, so a fix to plan extraction helps every tier.

```text
LiDAR (Stray Scanner)        Video (RGB only)                         Photos (2-8 per room folder)
keyframes by motion          DPVO, split into segments at restarts    ALIKED+LightGlue links, doorway pairs (EXIF time),
drift: fragment pose graph   MoGe-2 depth -> scale per segment        spin groups -> per-room layout; MoGe-2, GeoCalib
TSDF + raw points            self-check; same pose graph on depth     scale-cue fusion (depth, focal, priors)
            \_______________________ same scene.npz + scene_info.json ______________________/
   plan_beta: free space -> rooms (watershed + wall-line cuts) -> walls fitted on raw points -> openings, ceilings,
   adjacency (tier-aware, v4) -> LiDAR bias correction (corner-aware) -> tier scale term -> damage (two tiers) + scope
   -> plan.json (schema v1.0.0) + SVG/PNG + DXF                     one command: scripts/run_capture.py --tier ...
```

Five principles, each with its evidence:

- **Measure conventions before trusting them.** The ARKit-axis assumption would have cost 22–32 cm, against 6–7 mm
  (D-005).
- **Models propose, geometry measures.** Learned polygon models are accurate to decimetres and research-licensed
  (D-004).
- **Unobserved is not free space.** Glass and mirror holes are `not_observed` (D-006).
- **Intervals widen where data thins:** fit ⊕ sensor ⊕ drift ⊕ tier scale (D-016, D-034).
- **A cold run never crashes.** A missing floor is estimated or explained (D-032), and physically impossible
  openings are dropped (D-033).

**Plan extractor choice.** We built it twice, in two ways that fail differently (D-010), and judged both on
identical scenes (`modules/judge_plan_extractor.md` §0). Repeatability was a statistical tie: 11.2% against 8.5% of
walls in gate (Fisher p = 0.63). The space-first `plan_beta` won because it never invents space; alpha added 5–9 m²
of rooms nobody entered. Four alpha parts were ported (D-018). The v2 change, alpha's wall lines used as
segmentation cuts, took matched rooms from 6 to 8/8 (`modules/plan_beta.md` v2.4). v4 makes extraction tier-aware.
For video it adds a raw-point surface (1 room → 3 rooms on single_room) and a 5 cm surface-noise σ. For photos it
seeds one room per folder and takes adjacency only from photo links (plan_beta.md Part v4).

## 2. Tiers, honest accuracy, capture protocol

| Tier | Hardware | Metric source | Scale term (1σ) | Evidence |
|---|---|---|---|---|
| LiDAR | iPhone 12 Pro+ / iPad Pro 2020+, Stray Scanner | LiDAR depth (confidence 2, 0.2–4 m), ARKit poses + our drift correction | none (metric) | 3 sample captures; 5 ARKitScenes rooms with a Faro laser |
| Video | any phone camera, 1080p, 1× lens | DPVO segments + MoGe-2 depth; focal from metadata or SfM | 6.9% (single_room); **floored at 20%** when joins are unverified (D-034) | sample `rgb.mp4` against the LiDAR plan; **[PENDING: own video vs tape]** |
| Photos | any phone camera, 6–8 per room | MoGe-2 depth + GeoCalib; spin layout per room; doorway pairs | 3.8% with EXIF focal (REPORT self-report rows), 9.7% without | simulated folders against the LiDAR plan; **[PENDING: own photos vs tape]** |

**LiDAR, against the laser** (ARKitScenes, final code; REPORT §1, `outputs/benchmark/final/arkitscenes/summary.json`):

- Room dimensions: **8/8 within 1.5 cm**, median |e| 0.43 cm.
- Wall lengths: median 0.86 cm, 84% within 1.5 cm, max 8.81 cm (n 19).
- Ceilings: **3 of 4 rooms** with a plan within 1.5 cm (median 0.53 cm). The miss is 47895909 at −2.16 cm, where drift
  correction spreads the ceiling (I-004). The fifth room produced no plan: a 17 s capture that never enters the room,
  flagged `coverage_warning`.
- The room-box study behind the bias correction: hold-out mean −1.25 → +0.11 cm, pre-registered (D-021).
- Large errors come from topology, not the sensor: a 10 cm jog placed 9 cm off, and a 60° wall squared off
  (`modules/benchmark_arkitscenes_accuracy.md` §5).
- Only the iPad is validated. iPhone transfer is carried as an extra σ.

**Video** (D-020, `modules/video_tier.md` V2-§4). v1 chained DPVO pieces with the wrong relative scale (+187% /
+284%). v2 does four things:

- cuts the video at tracking restarts;
- re-tracks each segment;
- scales it from learned depth;
- self-checks it against its own depth and drops the segments that fail.

| Capture | v1 scale / ATE | v2 trusted segments | v2 trusted-path scale / ATE | `whole_scene_consistent` |
|---|---|---|---|---|
| single_room (14 m) | +4.21% / 12.7 cm | 1 of 1 | **+2.22% / 10.2 cm** | true |
| floor_only (54 m) | +187% / 455 cm | 2 of 3 | **+2.93% / 35.9 cm** over 35 m | false (no verified bridge) |
| with_ceiling (100 m) | +284% / 560 cm | 8 of 12 | +380%: one wrong segment (+338%) **passed the self-check** | false (0/28 bridges) |

Good trajectories do not yet give good plans. The final benchmark, scored against the LiDAR plan of the same capture
(REPORT §1):

| Capture | Video plan | Footprint, m² [95%] | vs LiDAR |
|---|---|---|---|
| single_room | 3 rooms | 23.46 [17.03, 29.88] | +33.2% |
| floor_only | 5 rooms | 40.91 [8.82, 73.01] | −33.9% (interval covers) |
| with_ceiling | 5 rooms in 4 pieces | 19.40 [4.16, 34.64] | −69.0% (interval misses) |

- Room dimensions within ±3%: 2/6, 3/10 and 2/6. **The 3% video gate fails.**
- The single_room excess is a door leak (plan_beta T-I3): a 7.74 m² room against LiDAR's 1.93 m². Earlier code states
  gave −1.7% and −6.7% (REPORT, "earlier code states").
- **D-034:** when segments are joined without verified geometry, the scale σ is floored at 20% and
  `meta.reliability = "low"` is set. Coverage on the long videos rose: floor_only 31% → 88%, with_ceiling 38% → 60%
  (`modules/benchmark_final.md` §7.2).
- **It still misses when rooms are missing** (BF-13). A scale term cannot cover area that is not in the plan.
- **Not repeatable (I-007).** The same command twice on floor_only gave 5 rooms / 40.91 m², then 3 rooms / 18.95 m²
  (REPORT §2). There are two measured causes (BF-12):
  - pycolmap's two-view verification varies, so the self-calibrated focal is 1564.6 against 1566.0 px;
  - DPVO itself varies from frame 1 even with identical inputs and cuDNN deterministic, with a 2× scale difference.
  - The cuDNN flag did not fix it. `ISSUES.md` still says "mitigated"; that is out of date.

**Photos** (D-017, D-022; `modules/photo_tier.md` v2.5 and v3):

- Doorway pairs are found from EXIF times: 9/9, with 0 false.
- v3 fits each room as a Manhattan rectangle from its own spin. A spin is a small panorama, so no feature match is
  needed.

Final benchmark on simulated folders (REPORT §1):

| | single_room | floor_only | with_ceiling |
|---|---|---|---|
| Rooms matched | 3/3 | 6/6 | 7/7 |
| Room dimensions within ±8% | 0/6 | 3/12 | 0/14 |
| Footprint vs LiDAR | +16.5% | −41.0% | −26.9% (interval covers) |
| Plan pieces | 3 | 4 | 2 |
| Reference adjacency pairs found | 0/2 | 1/11 | 4/10 |

- There are 0 m² of overlap.
- The simulated "spins" are walk-through frames covering 85–200°, so whole walls are never seen (PT-16).
- With true poses and a full turn, 5/16 dimensions are within 8% (v3.4). L-shaped rooms and hallways seen through
  doors are the named misses.
- Photos without capture times (messaging apps) get no layout (v3.5).

**No reference object** (D-067). Scale comes from learned metric depth, the stored focal length and priors. The
paper-sheet detector (`modules/scale_reference.md`) stays as an option, off by default. On the simulator it moved the
photo scale by at most 0.9 points and made the video scale worse (−18.7% with it, −1.7% without).

**Our own evidence revised the protocol three times** (Route 2, `CAPTURE_PROTOCOL.md`):

| Decision | Change | Evidence |
|---|---|---|
| D-017 | Photos: spin from the room's middle + doorway pairs, instead of corner shots | corner shots registered 17/46 photos in 5 pieces |
| D-022 | Turn clockwise; same tilt; 7 shots with one door; pair within 5 s; keep EXIF | direction guessed wrong in 2/13 rooms; a tilt difference ≥ 20° linked 24% vs 100%; no EXIF focal → 22% scale error |
| D-028 | Video: ≥ 1.5 m from blank walls, ≤ 20°/s turns, 2 s pause at doorways, pitch down, one lens, close the loop | every VO restart happened 0.3–1 m from a blank wall; the sample turned at 23–25°/s with bursts of 66–84°/s |

## 3. Drift handling and determinism

### 3.1 Method and ablation

The walk is cut into 1.5 m / 5 s fragments of raw high-confidence LiDAR; ARKit is trusted locally. The graph has
three kinds of edges:

- ARKit odometry;
- verified ICP loop closures (multi-scale point-to-plane, checked on fitness, RMSE and degeneracy);
- **plane anchors**: per-fragment Manhattan wall direction and floor level.

A robust 4-DoF pose graph is solved with IRLS-Cauchy and blended onto every frame (`modules/drift.md`). Gravity is
observed, so only yaw and position move. A 6-DoF graph tilted fragments by about 1° and was rejected. The ceiling
never enters the solve, so it is an independent witness. Drift correction is on by default (D-019).

| One command, ON vs OFF | floor_only ON | floor_only OFF | with_ceiling ON | with_ceiling OFF |
|---|---|---|---|---|
| Rooms / one connected plan (REPORT §1) | **8 / yes** | 7 / **no** | 8 / yes | 8 / yes |
| Footprint m² (REPORT §1) | **61.90** | 52.59 | 62.50 | 62.16 |
| Walls (REPORT §2, code-change rows) | 70 | 66 | 70 | 82 |
| Revisit surface separation, median (`benchmark_lidar_sample.md` §3) | **2.27 cm** | 14.29 cm | **1.15 cm** | 2.83 cm |
| Ceiling-level spread, the witness (same file) | – | – | **0.54 cm** | 6.04 cm |

ARKit's heading drifted up to 5.95° on floor_only. With poses used as-is, the plan loses a room, falls apart and
shrinks by 15%. On with_ceiling, drift mainly doubles walls. Held-out 5-fold loop residuals give the interval's drift
term: 1.9–2.5 cm between passes and 0.7–1.0 cm within one sweep (`drift.md` §5.5).

**The side effect (I-004).** With no revisit loop, floor anchors are the only vertical constraint. In ARKitScenes
room 47895909 (0 revisit loops), they spread the ceiling from 0.06 to 0.55 cm and push it to −2.16 cm
(`outputs/benchmark/final/arkitscenes/47895909/build_report.json`). This is fix-loop candidate C2 (`FIX_LOOP.md`).

### 3.2 Same command, same plan

Early on, the same command gave 61.90 / **59.37** / 61.90 m² on floor_only. There were two causes, both fixed:

1. **Open3D's multi-threaded ICP** summed in a scheduler-dependent order. Fix: each ICP runs single-threaded and the
   candidate pairs run in a thread pool. The poses are now bit-identical (D-029).
2. **A knife-edge in the extractor:** a cell under a bed had 0.535 against 0.457 coverage on a 50% rule. Fix: floor
   enclosed by the room's own walls belongs to the room (D-030). A multi-phase vote was built and **rejected**: it was
   predicted at about 17/87 walls and measured 11/94 (`plan_v2_ablation.md` Part S).

**Final run.**

- The LiDAR rerun of floor_only is identical: 70/70 walls, 61.90 / 61.90 m².
- All five LiDAR plans are bit-identical to the 04:32 and 04:45 runs (REPORT §2).
- The one difference from the 03:43 run is single_room, where D-033 dropped a 1 cm "passage".
- **Video is not deterministic** (I-007, §2).
- The extractor is still sensitive to raster phase: a quarter-cell shift changes 20–40% of walls (`plan_beta.md`
  v3.5). That is the main open problem behind the repeatability gate.

## 4. Error budget (LiDAR tier, one interior distance, 95%)

| Term | Size | Source | In the interval? |
|---|---|---|---|
| Sensor noise per point | σ 6–7 mm at < 2 m, 16–19 mm at 2.5–4 m | `scripts/noise_model.py`, D-011 | yes: 1/σ² fit weights and a sensor term |
| Plane-fit standard error | 0.04–0.08 mm | `modules/arkitscenes_validation.md` §6 | yes, negligible |
| Systematic surface offset (Apple LiDAR 0.68 cm into the room) | −1.37 cm per distance before correction | D-021 | **corrected**, corner-aware (D-027); residual ±1.6 cm, ±2.1 cm with device transfer |
| Residual drift | 2.0 cm across passes, 1.0 cm within one visit | `drift.md` §5.5 | yes |
| Inconsistency (two passes, curtains) | √(rmse² − σ_sensor²) | `plan_beta.md` v2.1 | yes |
| Surface tilt on partly seen walls | ±0.5–1 cm | `lidar_bias.md` Issue 4 | yes |
| Segmentation / topology (which surface became the wall) | tens of cm when it happens | `benchmark_lidar_sample.md` §4 | **no**: not a Gaussian term; handled by status, flags and the fix loop |

**D-027.** Moving every face outward by b lengthens a wall at a convex corner and **shortens** it at a reflex corner.
The corner-aware shift took walls within 1.5 cm of the laser from 82% to 94%
(`benchmark_arkitscenes_accuracy.md` §4.3). `--no-bias-correction` still widens the interval one-sided.

Video and photo add the scale term σ_s·L (2σ_s·A for areas, D-016). Learned-depth tiers also carry a 5 cm
surface-noise σ on wall offsets (plan_beta v4, T-2). Photo layout sides add 50% of their distance, which is
calibrated on the simulated spins (photo_tier v3.1).

## 5. Calibration (do the 95% intervals contain the truth 95% of the time?)

| What | Coverage (target 0.95) | Source |
|---|---|---|
| LiDAR vs laser: wall and room lengths | 0.93 (n 27, median half-width 6.4 cm: conservative) | REPORT §5 |
| LiDAR vs laser: ceilings | 4/4 (half-width 2.7 cm) | REPORT §5 |
| LiDAR repeat pair: same-topology walls / all matched walls | 0.97 (n 30) / 0.68 | REPORT §1 |
| LiDAR repeat pair: opening widths | 0.75 (n 8) | REPORT §5 |
| Video against the LiDAR plan, single_room / floor_only / with_ceiling | 0.70 (n 10) / 0.88 (n 16) / 0.60 (n 10) | REPORT §1 |
| Photo v3 against the LiDAR plan, same order | 1.00 at a half-width of 79% / 0.58 / 0.77 | REPORT §1 |
| Damage: stain area / crack length (injected) | 11/11 / 7/9 | `damage.md` v3.5 |

Where the error is **measurement** (the laser lengths, same-topology walls), intervals are calibrated or
conservative. Where it is **topological** (a different corner, a missing room, a wrong segment), no σ covers it.
Instead the plan marks such walls `inferred`, drops untrusted video segments and raises flags:
`whole_scene_consistent`, `reliability` (D-034), `coverage_warning`, `not_observed`. D-034 raised video coverage
from 31–38% to 60–88%. It still fails on two long runs where rooms are missing (BF-13). A photo coverage of 100% at
±79% is honest but uninformative. **[PENDING: coverage at the photo and video tiers against tape, own capture.]**

## 6. The fix loop

The full template, the candidates and the morning procedure are in `FIX_LOOP.md`.

**Rehearsal completed: LiDAR absolute bias** (D-021, `modules/lidar_bias.md` §5.6).

- Failing number: a ceiling at −1.92 cm against the laser (gate 1.5 cm).
- Five hypotheses were tested. Four were rejected (range scale, trajectory scale, registration, the laser); our
  estimator was a minor contributor.
- Root cause: a constant inward surface offset of 0.68 cm.
- The prediction was **pre-registered** before the hold-out rooms were analysed: 0.0 ± 0.7 cm. Measured: −1.25 →
  **+0.11 cm**.
- What fell short: the spread was 1.05 cm against 0.7 cm predicted. The plan-level re-test then found the
  reflex-corner error (D-027).

**A recorded failed prediction.** The phase vote was predicted at about 17/87 walls in gate. It measured 11/94 and
was kept off (§3.2).

**Candidates for the declared loop** (`FIX_LOOP.md` Part 2):

- **C2, I-004.** The ceiling gate is at 3/4 on the laser. One measured cause: 0 revisit loops, so floor anchors tilt
  the ceiling. The prediction is 4/4, with the LiDAR sample plans bit-identical.
- **C3.** Video footprint intervals that miss when rooms are missing.
- **C4.** Photo room sizes: dimensions within 8% are 0/6, 3/12 and 0/14.

The worst gate is chosen on the own-capture benchmark by a rule committed in advance: the largest relative shortfall
from its threshold.

**[PENDING: the declaration (gate, failing number, root cause, predicted number), committed before the fix; the
shipped fix; before/after runs at tags `before-fix` / `after-fix`; the diff.]**

## 7. Benchmark results (REPORT, final code)

| Gate | LiDAR | Video (vs LiDAR plan) | Photo v3 (vs LiDAR plan) |
|---|---|---|---|
| Wall lengths / room dimensions (video ±3%, photo ±8%) | laser: room dimensions 8/8 within 1.5 cm, walls 84% | **FAIL** 2/6, 3/10, 2/6 | **FAIL** 0/6, 3/12, 0/14 |
| Opening widths ≤ 2 cm on ≥ 85% | no GT; repeat-pair proxy **FAIL** 2/12 | – | – |
| Ceiling ≤ 1.5 cm per room | **FAIL** 3/4 (laser), max 2.16 cm | – | – |
| Ceiling spread across captures ≤ 1 cm | not verifiable: only with_ceiling sees ceilings | – | – |
| Repeatability, within max(1 cm, 0.5%) per wall | **FAIL** 15/87 = 17.2% [10.7, 26.5]; same-topology 14/30, median 1.8 cm; 34 unmatched | **[PENDING: own takes]** | **[PENDING: own repeat room]** |
| Drift accountability | **PASS** (§3.1) | – | – |
| One stitched plan, no overlaps | **PASS**: 1 piece each, 0 m² | PASS, PASS, **FAIL** (4 pieces) | **FAIL**: 3 / 4 / 2 pieces, 0 m² overlap |
| Correct adjacency | internal only | **FAIL**: 2/3, 5/6 and 0/1 reported pairs correct | **FAIL**: 0/2, 1/11 and 4/10 reference pairs found |
| Footprint ±8% (the photo gate; shown for video) | – | **FAIL** +33.2%, −33.9%, −69.0% | **FAIL** +16.5%, −41.0%, −26.9% |
| Calibrated intervals | §5 | **FAIL** 70% / 88% / 60% | PASS 100% (at ±79%) / **FAIL** 58% / 77% |
| Damage false positives on clean captures | **PASS**: 0 confirmed on all 3 | – | – |
| Same command twice | identical (70/70 walls) | **not identical** (I-007) | not tested |
| PASS / FAIL count (REPORT §0) | 3 / 3 | 2 / 13 | 1 / 16 |

The video single_room adjacency verdict depends on D-033. Dropping the 1 cm passage also removed LiDAR's R1–R2 link,
which the video finds independently (BF-14).

**Timing** of the one command (REPORT §4, wall time / peak RSS):

| Tier | single_room | floor_only | with_ceiling |
|---|---|---|---|
| LiDAR | 52 s / 2.0 GB | 146 s / 3.3 GB | 238 s / 4.9 GB |
| Video | 272 s / 5.9 GB | 794 s / 5.9 GB | 1737 s / 7.6 GB |
| Photo (warm depth cache, not cold-start) | 9 s / 1.4 GB | 37 s / 1.7 GB | 45 s / 1.9 GB |

**Damage** (D-023..D-026, D-031; `damage.md` v3).

- Detection is sensitive and quoting is conservative. Only "confirmed" detections create scope items: stains at 0.35
  or more, cracks at 0.65 or more.
- Two precision fixes came from real false positives:
  - a dominant-line test rejects straight, axis-aligned bars as cracks (D-024);
  - a stain must lie **on** the surface (D-031). The false stain was a jar lid 2.4 cm proud of the wall.
- Injection recall: stains 11/18, cracks 9/15. Median error: 2.0% (stain area), 5.0% (crack length).

**[PENDING: own-capture benchmark against tape at the photo and video tiers (`scripts/process_own_capture.py` →
`outputs/own/eval/own_eval.md`, then `bench_final.py --skip-lidar` → REPORT §6); photo repeat room; staged damage;
head-to-head against <app, version> on 2 rooms (REPORT §3; at the video or photo tier, because my phone
has no LiDAR).]**

## 8. Known failure modes

| Condition | What happens | What we do | Evidence |
|---|---|---|---|
| Mirror | depth sees a reflected room; a mirror panel was read as a window (O9) | confidence filter; reflection test; regions not enclosed by measured walls dropped; stain relief | `FAILURE_MODES.md`, `damage.md` v3.2 |
| Glass | no returns → rooms merge | glazing found by the sill test; `not_observed`, never free space | plan_beta I-8 |
| Wet-look floor | phantom points below the floor (≤ 0.7%) | excluded; robust floor peak | `failure_modes.md` §5 |
| Low light | LiDAR unaffected; the RGB tiers lose matches earlier | sharpest-frame choice; flags | `arkitscenes_validation.md` §6 |
| Furniture against walls | wardrobe fronts measured as walls; objects on shelves look like stains | vertical-extent rule; stain relief check | judge J-I-5, `damage.md` v3 |
| Capture never enters a room | no room produced | `coverage_warning` | `benchmark_arkitscenes_accuracy.md` §5 |
| Small capture, no revisit loop | drift correction tilts the ceiling (−2.16 cm) | open; fix-loop candidate C2 | I-004 |
| Non-Manhattan wall | a 60° wall is squared off | not handled | same as above |
| Blank walls in video | VO restarts; a wrong segment can pass the self-check (+338%) | segments, self-check, `reliability = low`, σ floor (D-034) | `video_tier.md` V2-I11 |
| Long video, GPU non-determinism | same command: 5 rooms / 40.91 m², then 3 rooms / 18.95 m² | disclosed; a video before/after needs ≥ 2 runs per state | I-007, `benchmark_final.md` BF-12 |
| Video door leak | a room carved through a door (7.74 vs 1.93 m²) | open | plan_beta T-I3 |
| Partial spins, L-shaped rooms, hallways | photo rooms 9–24% off, median | wide per-side intervals; corridor and L-shape rules are next | `photo_tier.md` v3.6 |

Hardware-forced deviations (recruiter-approved, `OPEN_QUESTIONS.md`): there is no LiDAR device, so the LiDAR tier is
benchmarked on the provided captures and on ARKitScenes, and the head-to-head runs at the photo or video tier. The
licences of every model and dataset are in `DISCLOSURES.md`.
