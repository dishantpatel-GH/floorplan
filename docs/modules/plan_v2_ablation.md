# plan extractor v2: ablation, one row per change

Code: `floorplan/plan/beta/` (v2, live) against `floorplan/plan/beta_v1/` (v1, frozen copy of the judge's chosen
extractor, imports fixed so it runs standalone; never edited after the copy). Harness: `scripts/plan_v2_eval.py`.
Evidence: `outputs/plan_v2/` (`scores.json`, `plausibility.json`, `registration.json`, `repeat_<tag>.png`,
`runs/<tag>/<capture>/{plan.json, run.json, debug.png}`). Final plans: `outputs/plan_v2/<capture>/`.

## How every number here is produced

- **Scenes.** The canonical LiDAR scenes `outputs/scenes_v2/<capture>/` (drift correction ON, pixel-centre depth
  intrinsics D-011, per-raw-point `raw_frame`). The judge scored the older `outputs/drift/*/scene_on` scenes, so
  v1 was re-run here on scenes_v2 to get a fair "before" (row 0). The judge's own beta number on the old scenes was
  11/98; on scenes_v2 the same code gives 7/99. Same code, different scene build: the scene alone moves the gate
  by 4 walls, which is the size of noise on ~95 walls.
- **Registration.** floor_only is put into with_ceiling's frame with `floorplan/eval/register.py` on the two
  scenes_v2 scenes (from geometry, never from a plan; judge J-2): yaw 89.08°, residual 0.98 cm, inliers 0.79,
  local tile disagreement median **2.0 cm**, p90 4.1 cm (`registration.json`).
- **Scoring.** `judge_common.compare` unchanged (floorplan/eval matching, the gate |Δ| ≤ max(1 cm, 0.5% L), unmatched
  = fail, topology split, Wilson 95% interval), plus the judge's floor recall (`judge_plausibility.floor_recall`)
  for both captures, interval coverage on all matched walls and on same-topology walls only, opening widths and
  the invariance tests (shift 1.3 × 0.7 cm, rot90, thin90), each scored as "plan vs itself".
- **Reproduce.** `python scripts/plan_v2_eval.py register`, then for each row
  `python scripts/plan_v2_eval.py run --capture <c> --tag <tag> [--set KEY=VALUE]` on both captures and
  `python scripts/plan_v2_eval.py score <tag>`. The job lists are `outputs/plan_v2/jobs_*.txt`; 
  `jobs_ablation.txt` regenerates every row with the FINAL code by switching the later changes off. A row's
  `--set` values are in each `run.json`.
- **Regenerability check.** Row 1 re-run with the final code and the switches (tag `b1_regen`) reproduces
  14 / 95, 9.0 cm, same-topology 31 / 13 / 2.0 cm exactly.

Abbreviations: A = with_ceiling, B = floor_only. "Pass" = walls passing the gate / walls judged. "ST" = same-
topology walls (n / pass / median |Δ|): both corners built from matched neighbours, so their disagreement is
*measurement*; the rest is *segmentation*. Cov = interval coverage (share of |z| ≤ 1.96); target ~0.95.

## The table (repeat pair, scenes_v2, drift ON)

Rows are in the order the changes were made; each row is measured with all kept changes above it switched on.
**Kept** rows are in the final v2; **rejected** rows stay in the code only as switches (ablation) or are reverted.

| # | Change (what) | Why | Rooms A / B / matched | Pass (rate) [Wilson 95%] | Median \|Δ\| | ST n / pass / median | Footprint A / B (m²) | Floor recall A / B | Cov all / ST | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **v1 frozen** (`beta_v1`), tag `v1` | baseline | 7 / 6 / 6 | 7 / 99 (7.1%) [3.5, 13.9] | 15.0 cm | 17 / 7 / 1.9 cm | 62.6 / 61.9 | 0.953 / 0.939 | 0.42 / 0.94 | before |
| 1a | B-1 cuts = every long line run (`b1_nopair`) | alpha's lines as segmentation barriers, raw | 9 / 10 / 9 | 8 / 113 (7.1%) [3.6, 13.4] | 10.1 cm | 27 / 8 / 2.9 cm | **54.1** / 56.9 | 0.939 / 0.933 | 0.49 / – | **rejected**: footprint −14% (beds, sofas, counters cut rooms: J-I-5 again) |
| 1b | B-1 cuts only where an opposite face lies 5–40 cm behind (`b1`, first version) | a partition is seen from both rooms; furniture from one side | 7 / 6 / 6 | 8 / 101 (7.9%) [4.1, 14.9] | 14.2 cm | 16 / 7 / 1.8 cm | 62.6 / 61.9 | 0.953 / 0.939 | 0.47 / – | **rejected**: the merged partition was seen from one side only, nothing changed |
| 1 | **B-1 cuts where paired OR the face covers ≥ 0.8 m of the 0.3–2.0 m band** (`b1`) | a wall rises from the floor past furniture height; beds, sofas, desks, counters stop ≤ ~1 m (≤ 0.7 m of extent) | **8 / 8 / 8** | **14 / 95 (14.7%)** [9.0, 23.2] | **9.0 cm** | 31 / 13 / 2.0 cm | 62.7 / 61.9 | 0.953 / 0.940 | 0.58 / 0.90 | **kept**: the floor_only merge is gone, footprint and recall unchanged |
| 2 | X-1 single visit per room + X-2 + B-4 (`x_all`) | measure all walls of a room from ONE contiguous visit (my idea) | 8 / 8 / 8 | 9 / 95 (9.5%) [5.1, 17.0] | 10.7 cm | 31 / 9 / **3.2 cm** | 62.7 / 61.9 | 0.953 / 0.941 | 0.58 / 0.90 | **rejected** (see X-1 below) |
| 2b | X-1 on, X-2 off (`x_noweights`) | isolate X-1 | 8 / 8 / 8 | 10 / 95 (10.5%) | 10.8 cm | 31 / 10 / 3.2 cm | 62.7 / 62.0 | 0.953 / 0.941 | 0.58 / – | confirms X-1 is the cause |
| 3 | **X-2 noise weights 1/σ_lidar(range)², X-1 off** (`x_novisit`) | near points are less noisy (D-011 table) | 8 / 8 / 8 | **15 / 95 (15.8%)** [9.8, 24.4] | 9.0 cm | 31 / 14 / **1.9 cm** | 62.7 / 61.8 | 0.953 / 0.940 | 0.60 / 0.90 | **kept** (small gain, within noise; principled) |
| 4 | **B-4 calibration**: drift σ 1 cm in one visit / 2 cm across passes (drift.md 5.5), inconsistency √(rmse² − sensor²) | intervals that cover the truth | same as row 3 | same | same | same | same | same | 0.60 / 0.90 (median \|z\| 0.26) | **kept**; with B-4 off (`b4_off`): 0.60 / 0.90, median \|z\| 0.29. ST coverage was already ~0.9: the all-wall coverage is held down by topology errors, which no σ can cover |
| 5a | X-3 band 0.8–1.6 m for everything (`x_band816`) | measure where the tape is held | 7 / 7 / 7 | 9 / 61 (14.8%) | 4.4 cm | 25 / 9 / 2.1 cm | 61.0 / 60.3 | **0.911 / 0.897** | 0.67 / 0.92 | **rejected**: fewer points → walls become inferred → a room is dropped (recall −4%); the lower median is fewer, easier walls |
| 5b | X-3 band 1.0–2.0 m (`x_band1020`) | same, above furniture | 7 / 7 / 7 | 8 / 62 (12.9%) | 4.5 cm | 24 / 8 / 2.8 cm | 61.1 / 60.3 | 0.909 / 0.897 | 0.62 / 0.92 | **rejected**, same reason |
| 5 | **X-3 tape band for the POSITION only** (existence and status keep 0.3–2.0 m) (`x3_tape`) | the own-capture ground truth is taped at ~1 m, above skirting and below coving | 8 / 8 / 8 | 15 / 96 (15.6%) [9.7, 24.2] | 9.3 cm | 28 / 14 / **1.6 cm** | 62.7 / 61.8 | 0.953 / 0.940 | 0.59 / 0.93 | **kept**: repeatability not worse (ST median 1.9 → 1.6 cm), and it measures what the tape measures |
| 6 | **B-3 alpha's tall-face jamb rule** (v1 slice rule as fallback) (`b3` vs `b3_v1jamb`) | 9 / 11 widths at ±1.4 cm in alpha | walls unchanged | 15 / 96 | 9.3 cm | 28 / 14 / 1.6 cm | – | – | – | **kept**: matched doors within 2 cm across captures **1 / 7 → 2 / 7** (+2 at 2.2 and 2.4 cm), median \|Δ width\| **5.0 → 2.4 cm**; widths measured on jambs A 4 → 8, B 2 → 5 |
| 7 | **B-6 ceiling double layer** (alpha A-14) | ±1.7 cm claimed on an 11 cm drift error (J-I-7) | no change on drift-ON scenes (all 8 ceilings within 0.1 cm of v1) | | | | | | | | **kept**: on the drift-OFF with_ceiling scene the bedroom goes from 2.958 m `measured` [2.941, 2.975] to `inferred` [2.941, **3.068**], which now covers the drift-ON value 3.065 m (`runs/{v1,v2}_driftoff`) |
| 8a | B-2 coplanar merge only (`b2_coplanar_only`) | merge parallel measured walls ≤ 2 cm apart | identical to row 6 | | | | | | | | no merge fired: after refinement, parallel lines are never that close; kept as a guard |
| 8 | **B-2 canonical outline**: remove shallow (≤ 10 cm) steps/bumps whose connecting or base edge has no raw-point support; merged wall re-measured (`b2`) | jogs split long walls differently in each capture (topology errors) | 8 / 8 / 8 | 13 / 86 (15.1%) [9.1, 24.2] | **8.6 cm** | 25 / 12 / 1.7 cm | 62.7 / 61.9 | 0.952 / 0.940 | **0.64 / 0.96** | **kept, honestly neutral on the gate** (rate 15.6 → 15.1%, 10 fewer judged walls, median and coverage better). Unsupported geometry is removed from the plan, which is what B-2 is for |
| 9 | **B-1b** an unentered region split off by a cut goes back to its entered neighbour; **B-2 overlap guard** (`v2b` = final `v2`) | an overlap of 0.13 m² appeared on A after row 8; plans must not overlap | 8 / 8 / 8 | **13 / 87 (14.9%)** [8.9, 23.9] | **8.2 cm** | 26 / 12 / **1.7 cm** | 62.7 / 61.9 | 0.952 / 0.940 | **0.63 / 0.96** | **kept**: overlap 0.133 → 0.0 m² |
| 10 | B-5 raster on a world lattice (`v2lat`) | make rot90 / thin90 exact | 8 / 8 / 8 | 8 / 89 (9.0%) [4.6, 16.7] | 8.2 cm | 24 / 8 / 2.5 cm | 62.6 / **57.6** | 0.950 / 0.942 | 0.57 / 0.96 | **rejected**, but it is the most important *finding* of this round (see B-5) |
| 11 | X-4 photo-folder hints (`v2_photo` vs `v2_photo_nohints`) | one room per folder | photo scene with_ceiling: 2 rooms either way; the 3 regions voted for 3 different folders, so nothing merged | | | | | | | | **kept as plumbing, no measured effect** (the photo tier builds its own plan; this only matters if it calls beta) |
| 12 | B-8 coverage warning | a degenerate input must not look like a result | 1.5 × 1.5 m crop of single_room: `meta.coverage_warning = "plan covers 1.7 m2 (< 4.0 m2) ..."` | | | | | | | | **kept** |

**Before → after (row 0 → row 9, the final v2):** rooms matched 6 → 8 (every room of both captures matched),
pass 7 / 99 (7.1%) → 13 / 87 (14.9%), median |Δ| 15.0 → 8.2 cm, same-topology walls 17 → 26 at a median of
1.9 → 1.7 cm, interval coverage 0.42 → 0.63 (same-topology 0.94 → 0.96), footprints 62.6 / 61.9 → 62.7 / 61.9 m²,
floor recall unchanged (0.953 / 0.939 → 0.952 / 0.940). The gain did **not** come from shrinking the plan.

**The gate is still failed.** The pass rate's Wilson interval [8.9, 23.9]% overlaps the v1 interval [3.5, 13.9]%
a little, so the gain is real but modest on 87 walls. Same-topology walls pass 12 / 26 = 46%: that is the
ceiling for this pair even with perfect segmentation, because their median disagreement (1.7 cm) equals the
residual misalignment between the two drift-corrected captures (registration tiles: 2.0 cm median).

## Other captures (`outputs/plan_v2/plausibility.json`, judge's checks)

| Capture | Rooms v1 → v2 | Footprint m² | Floor recall | Overlap m² | Widths measured on jambs | Runtime s |
|---|---|---|---|---|---|---|
| single_room | 2 → 3 | 19.41 → **17.61** | 0.603 → 0.591 | 0 → 0 | 0 → 2 | 3.9 → 3.9 |
| floor_only | 6 → 8 | 61.93 → 61.90 | 0.939 → 0.940 | 0.024 → 0 | 2 → 5 | 18.4 → 19.6 |
| with_ceiling | 7 → 8 | 62.64 → 62.65 | 0.953 → 0.952 | 0 → 0 | 4 → 8 | 29.7 → 34.9 |

single_room is the one red flag: −1.8 m² (−9%). The observed floor inside rooms only drops by 1.2 points of
recall, so the lost area is mostly *unobserved* floor that v1's arrangement rectangle used to include (a strip
behind a wardrobe line, 50% covered). B-1's cuts also changed the watershed there and split off a 1.9 m² entered
alcove as its own room. Without ground truth I cannot say which is right; it is listed as an open problem.

## Invariance (each extractor against itself)

| Test | v1 A | v1 B | v2 A | v2 B | v2 lattice A | v2 lattice B |
|---|---|---|---|---|---|---|
| shift 1.3 × 0.7 cm | 71 / 82 (87%) | 72 / 72 (100%) | 61 / 71 (86%) | 70 / 70 (100%) | **46 / 78 (59%)** | **43 / 82 (52%)** |
| rot90 | 65 / 81 (80%) | 54 / 78 (69%) | 60 / 70 (86%) | 37 / 76 (49%, rooms 8 → 7) | **69 / 70 (99%)** | 59 / 72 (82%) |
| thin90 | 60 / 81 (74%) | 58 / 76 (76%) | 51 / 71 (72%) | 61 / 74 (82%) | 59 / 74 (80%) | 63 / 70 (90%) |

## Notes per change (the reasoning a reviewer will ask for)

**B-1 (room merges).** Root cause from the judge (J-I-4): the floor_only capture saw the partition between two
bedrooms only below 1.3 m, so v1's 1.0–2.0 m band missed it. Evidence for the fix (`outputs/plan_v2/sep_lines_floor_only.txt`,
every long line run with its height profile; `cuts_floor_only.png` / `cuts_with_ceiling.png`, left = kept rule,
right = any run): that partition (u = 2.70, normal −u) is a 2.30 m single-sided line with heights 0.17–1.29 m
(p5–max), only 21% of its surface above 1.0 m, so "paired faces" alone misses it and "any long line" also cuts at beds
and counters (row 1a: footprint −14%). The kept rule is "paired OR ≥ 0.8 m of vertical extent". After it,
floor_only's merged 16.62 m² room splits into 9.17 + 7.60 m²; with_ceiling's two rooms are 9.25 + 7.59 m² (v1:
9.17 + 7.60). The cut is used
only for the watershed; the cut cells are given back to the nearest room, so the plan's free space never shrinks.

**X-1 (single visit) — rejected by evidence.** Hypothesis: mixing two passes whose residual drift is 1.9–2.5 cm
blurs each wall, while one visit is locally consistent (ICP noise ~0.7 cm). Measured: same-topology median
1.9 → 3.2 cm, pass 15 → 9 (rows 2, 2b, 3). Why the hypothesis fails: (1) averaging two passes with independent
drift errors *reduces* the error by about √2; one visit keeps its whole drift error; (2) one visit sees fewer
patches of each wall, so the fit is noisier; (3) which visit wins can differ between captures. Kept as the
switch `single_visit` (default off) and the per-visit tools (`room_visits`, `choose_visit`) for a future
per-pass diagnostic.

**X-2 (noise weights).** Weighted least squares with w = 1/σ_lidar(range)² from the measured D-011 table. A small
gain (14 → 15 passes, ST 2.0 → 1.9 cm) that is within noise; kept because it is the statistically right
estimator when noise grows with range, and it costs nothing.

**X-3 (measurement height band).** The full-band fit and the tape-band position differ in what they measure: a
wall is not perfectly vertical and has skirting at the bottom and coving at the top. Restricting the *whole*
measurement to 0.8–1.6 m throws away points that decide whether a wall is measured at all, and a room is dropped
(rows 5a/5b: recall −4%). Restricting only the *position* keeps the plan and moves each offset to where the tape
is held (row 5). The ARKitScenes laser check (arkitscenes_validation.md) measures plane-to-plane distances and
does not run this extractor, so it could not arbitrate this within the time box: that is listed as open.

**B-4 (calibration).** Same-topology walls were already covered (0.90 → 0.96 after B-2), with a median |z| of
0.24–0.27, i.e. the intervals are slightly *wide* on typical walls. All-wall coverage (0.63) is low because
walls with different corners differ by decimetres: a segmentation error, which the interval must not try to
cover. The drift term now follows the drift module's measured values (1 cm within a visit, 2 cm between passes),
and the inconsistency term widens smeared walls.

**B-5 (raster phase) — the finding.** Anchoring the raster to a world lattice makes rot90 and thin90 almost exact
(99% / 80–90%), but then a 1.3 cm sub-cell shift changes 41–48% of the walls. So v1 and v2 are strongly
raster-phase sensitive, and v1's data-anchored raster *hid* this in the shift test (the anchor moves with the
data, J-I-2). Two different captures always have different phases: part of the cross-capture topology
disagreement is the extractor's own noise. That makes the phase the next fix (Part 4 candidate below).

**B-9 (runtime).** No speed-up was ported: v2 runs in 3.9 / 19.6 / 34.9 s (single_room / floor_only /
with_ceiling) on the loaded machine, inside the 90 s budget; the jamb rule and the cut lines add 1–5 s (machine shared with other agents).

**Not done in this round (time box):** B-7 (glass flag), B-10 (partial rooms kept outside the footprint), the
CLI exit code for B-8 (`scripts/run_plan_beta.py` is not part of this task), the per-wall variant of X-1.

## Part 4 candidate (after this round)

- **Worst failing gate:** wall repeatability, **13 / 87 = 14.9% [8.9, 23.9]** of walls within max(1 cm, 0.5%)
  (the gate needs every wall). Opening widths are second: 2 of 7 matched doors within 2 cm (29%, gate 85%).
- **Root cause, with evidence:**
  - 36 of 87 walls are unmatched and 25 have different neighbours (median |Δ| 32 cm; 1 of them passes): segmentation /
    topology, not measurement (v1: 54 unmatched, 28 different-topology at 51 cm);
  - the extractor itself changes 41–48% of walls under a 1.3 cm sub-cell shift when its raster phase is pinned
    (row 10): the outline (arrangement rectangles at the 50% coverage threshold, mask lines from the raster
    border, jog and bump rules) depends on where walls fall inside 2 cm cells;
  - same-topology walls agree to 1.7 cm median, the same as the captures' residual misalignment (2.0 cm), so
    measurement is at the drift floor.
- **Next fix to ship:** phase-averaged outlines. Run the raster stages (free space, segmentation, arrangement) at
  4 sub-cell phases (0 / 1 cm in u and v) and keep, per room, the outline whose edge set is the majority (the
  medoid by topology); the raw-point measurement is unchanged. Cost: ~4 × the raster stages, ≈ 60 s on
  with_ceiling, still inside 90 s.
- **Predicted number:** the shift self-test from 52–59% to ≥ 85% of walls, and on the repeat pair about a third
  of the 25 different-topology walls becoming same-topology. At the same-topology pass rate (46%) that is about
  +4 passes: **≈ 17 / 87 ≈ 20%** [Wilson ≈ 13, 29]. Not the gate: the gate also needs the drift module to bring
  the captures' residual misalignment from 2.0 cm towards 1 cm.

## Part S (stability round, 2026-10-04): the "Part 4 candidate" above, tested

Harness: this file's `scripts/plan_v2_eval.py`, re-pointed with `PLAN_V2_OUT=outputs/stability/plan_eval`
(job script `outputs/stability/plan_eval/run_tag.sh <tag> [--set K=V]`; `PHASE_SET=0.25,0.25` adds the new
**phase** test, which moves the raster by a quarter cell on the same scene). Rows continue the table above. Full
write-up: `docs/modules/plan_beta.md` Part v3.

| # | Change | Rooms A / B / m. | Pass [Wilson 95%] | Median \|Δ\| | ST n / pass / med | Footprint A / B | Recall A / B | Cov all / ST | Invariance (A, B): shift / rot90 / thin90 / phase | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 9 (re-run, `s0`) | final v2 as found | 8 / 8 / 8 | 13 / 87 (14.9%) [8.9, 23.9] | 8.2 cm | 26 / 12 / 1.7 cm | 62.7 / 61.9 | 0.952 / 0.940 | 0.63 / 0.96 | 61/71, 70/70 / 60/70, 37/76 / 51/71, 61/74 / 52/72, 44/83 | reproduces row 9 exactly |
| 13 | S-2 phase vote, 4 phases, medoid by raw agreement count (`s4`) | 8 / 8 / 8 | 11 / 95 (11.6%) [6.6, 19.6] | 6.1 cm | 29 / 11 / 1.8 cm | 62.7 / **59.6** | 0.952 / 0.944 | 0.64 / 0.97 | 63/73, 77/81 / 64/73, 55/83 / 54/74, 65/87 / – | **rejected**: footprint B −2.3 m² (the count score favours fragmented builds) |
| 14 | S-3 + S-2 with Dice score (`s5`) | 8 / 8 / 8 | 11 / 94 (11.7%) [6.7, 19.8] | 7.4 cm | 27 / 11 / 1.9 cm | 62.7 / **59.6** | 0.952 / 0.944 | 0.59 / 0.96 | 63/73, 78/78 / 66/70, 58/79 / 54/74, 60/80 / 57/72, 46/87 | **rejected**: better self-invariance, worse repeat pair, plan shrinks, real flip persists (57.56 vs 59.61 m²) |
| 15 | **S-3 enclosed-cell rule** (`s3`) | 8 / 8 / 8 | 13 / 87 (14.9%) [8.9, 23.9] | 8.2 cm | 26 / 12 / 1.7 cm | 62.7 / 61.9 | 0.952 / 0.940 | 0.63 / 0.96 | 61/71, 70/70 / 60/70, 37/76 / 51/71, **63/72** / 52/72, **46/81** | **kept**: neutral on the gate, removes the 2.5 m² run-to-run flip (3/3 identical plans on the benchmark's scenes, was 2/3) |

**Prediction check.** The Part 4 candidate above predicted the shift self-test going to ≥ 85% and the gate to ≈ 17/87. Measured
with the vote: the gate fell to 11/94, and floor_only shrank. The data-anchored shift test was already 86–100%,
because it does not move the phase. On the new phase test, 52–57 of 72 walls agree on A and 44–46 of 81–87 on B,
with or without the vote. Pairwise agreement between the four phase builds is only 0.59–0.73 (mean Dice). That is
too low for a majority vote to be stable, so the prediction failed. The phase sensitivity is the open problem
(`plan_beta.md` v3.5).
