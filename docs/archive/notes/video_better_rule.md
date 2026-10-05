# Video: what "better" means, and the baseline it is measured against

5 Oct, 09:40 IST, for the request "make sure video works better than it did". Written and committed before any
change to the video default; the only video commits before it (`1de73d1`, `1de73d1`) add an experimental switch that
is off. It records what the video tier does today and when a change may replace today's default. I do not edit the
rule after a candidate has been replayed.

Today = the default video settings (scale step "depth_agreement", D-076) of the code at `e70039d`, whose `floorplan/`
and `scripts/` are those of `b0b4a34`. Commits after it, up to this note, add room names, segmenter doors and windows
and an experimental `--video-rooms` switch (off). They change nothing else in `floorplan/video/` and nothing in
`floorplan/plan/beta/` or `floorplan/recon/`. Names and segmenter openings only add names and openings, and the
replays turn them off; after r1's cache replayed at `1de73d1` gives the same plan as at `e70039d`, wall for wall.

## Why every comparison is a replay of the same DPVO runs

- The same video gives different plans on different DPVO runs (I-007). At today's defaults the nine cached take1 runs
  give 1 to 9 rooms and 10.0 to 40.3 m² (the house is about 29–31 m²). Both live take1 runs at the default this
  morning (`outputs/presentable/video_default_r1`, `_r2`) gave 1 room.
- So a candidate is never judged on fresh runs. Each cached run below is replayed with the candidate's code on the
  run's own DPVO, MoGe-2, GeoCalib and SfM arrays (CPU; a cache miss stops the replay) and compared with the baseline
  replay of the same cache. The difference is then the change, not the DPVO run.
- Tools, in `outputs/video_better/` (git-ignored, like all outputs): `baseline/replay.py` (one replay from a
  git-archive export; `VIDEO_PARAMS='{...}'` sets a switch), `run_arm.sh` (every cache in `baseline/RUNS.txt`, one at
  a time), `baseline/summarize.py` (the scores below), `rule_check.py` ((a) and (b) below, as code).
- The baseline is the candidate's parent. If the code that makes the video plan's rooms and walls moves before a
  candidate is judged, I replay the baseline again at the candidate's parent first. The thresholds stay.

## The rule

A video change becomes the default only if (a), (b) and (c) all hold. Otherwise it ships behind a switch that is off.

**(a) My own video, take1** (tape ground truth: 14 walls in the bedroom, hall and kitchen, scored by
`scripts/eval_own_capture.py --gt-json`; a wall of a room the plan lacks is a miss). On the N = 9 take1 caches of the
baseline table, with k = 7 (three quarters of N, rounded up):

1. The footprint error, |footprint / 30 m² − 1|, is more than 6 points below the baseline's on at least k caches.
   A cache whose baseline and candidate footprints are both within 29–31 m² counts as better.
2. More of the 14 tape walls are found than at the baseline on at least k caches (14 and 14 counts as better).
3. Walls within 10%, summed over the nine caches, are not fewer than the baseline's 18, and no cache loses more than
   one.

**(b) The recruiters' sample videos** (video tier only; the reference is the LiDAR plan of the same capture,
`outputs/benchmark/final/lidar/<capture>/drift_on/plan.json`, scored with `scripts/bench_tier_ref.py`):

1. single_room: on each d069 cache (r1, r2, r3) the footprint error stays within the d069 spread, at most 5.05%
   (16.72–18.50 m²). On the after run's cache it may rise by the spread's width, 6.3 points, to at most 26.2%
   (baseline 19.9%).
2. floor_only (after run, d066) and with_ceiling (after re-run): on each cache, no fewer plan rooms matched to a LiDAR
   room, and a footprint error at most 6 points above the baseline's. Today that means d066 keeps 3 matched rooms and
   at most 55.2%; with_ceiling keeps 1 and at most 19.2%; the floor_only after run (0 rooms) sets no floor.

**(c) The other tiers do not move.** The video tier shares the plan step (`floorplan/plan/`) and fusion
(`floorplan/recon/`) with the LiDAR and photo tiers. A change there is switched on for the video tier only, or the plan
step re-run on the saved LiDAR scenes of the three samples (`drift_on/scene`) gives the same rooms and footprints with
and without it.

## Why these thresholds (from the baseline's own spread)

- **6 points of footprint.** The 12 fix-loop caches were each replayed twice with today's code: this baseline and the
  D-076 follow-up's "depth_agreement" arm (`outputs/fixloop/followup/replay/*_old`). Those were made before `b0b4a34`
  was committed, so I replayed instead of reusing them. Before r1's cache was replayed a third time
  (`followup/ablation/own_b1_repeat_old`). 11 of 12 caches gave the same footprint each time. Before r1's gave 24.15,
  24.11 and 22.39 m² (errors 19.5, 19.6 and 25.4%; Open3D's fusion is multi-threaded): 5.9 points apart. A smaller
  margin would let that noise count as an improvement. floor_only and with_ceiling use the same margin; their own
  replays did not differ.
- **Walls within 10%: one per cache.** Before r1's three replays gave 5, 4 and 4.
- **Walls found: strictly more.** Those replays gave 12, 12 and 12, and every other take1 cache gave the same count
  both times, so one more wall is beyond the replay noise.
- **k of N.** The request's "3 of 4" assumed four take1 caches; there are nine. I keep its share, three quarters,
  rounded up: 7 of 9. The 29–31 m² and 14-of-14 exceptions are there because a cache that is already right cannot get
  measurably better: two caches give 29.9 and 29.6 m² at the baseline, and 30 m² is itself known to about ±1 m².
  (a) counts footprint and walls found separately: each must improve on 7 caches, not necessarily the same ones.
- **The d069 spread.** single_room's d069 r1–r3 are three DPVO runs of the same video, so their spread is that
  capture's run-to-run noise: +1.2, −2.9 and −5.05% at the baseline. Within the spread means at most 5.05%; its width
  is 6.3 points. The after run's cache (+19.9%) is outside it already, so there the width is the allowance.

## Baseline: today's defaults, each cache replayed on CPU

Code: a git-archive export of `e70039d`, default `VideoParams`, `run_capture.py --tier video --no-damage`. Replays
08:39–09:32 IST, one at a time (`outputs/video_better/baseline/`: `run_baseline.log`, `replay/`, `summary.json`).
"What each room holds": the spaces the keyframe cameras inside the room were filmed from (all take1 runs share the
same 228 keyframes; the keyframe ranges per space are in `summarize.py`, read off
`outputs/own_house/diag/video/keyframes_every4.jpg`). "Bedroom room": the plan room the tape scorer pairs with the
bedroom.

Caches: 23:28 = `own_house/runs/video_take1`; before / after r1, r2 = `fixloop/{before,after}/video_take1_r{1,2}`;
presentable = `presentable/video_{default,pnp}_r{1,2}` (live runs this morning; "pnp" names the run's own scale step,
the replay uses the default); d069 = `benchmark/final/video/single_room/d069_r{1,2,3}`; single_room / floor_only after
run = `fixloop/after/sample_*`; d066 = `benchmark/final/video/floor_only/d066`; with_ceiling after re-run =
`fixloop/after/replay/with_ceiling_new_fresh` (the after run's own cache lacks a DPVO re-run, FIX_LOOP.md).

**Own take1, tape ground truth.**

| Run | Rooms | What each room holds (m²) | Footprint (error vs 30 m²) | Walls ≤3% / ≤10% / found, of 14 | Bedroom room, m² (tape 11.09) | Keyframes kept |
|---|---|---|---|---|---|---|
| 23:28 | 9 | hall 14.9; passage+kitchen 6.5; hall 3.7; bedroom (no camera inside) 3.0; bedroom 2.4; bedroom 2.2; passage 1.8; hall 1.6; hall 1.5 | 37.6 m² (25.2%) | 0 / 1 / 8 | 14.9 (hall) | 228 / 228 |
| before r1 | 4 | bedroom 12.2; hall 6.4; passage+kitchen 3.8; bedroom 1.7 | 24.1 m² (19.5%) | 2 / 5 / 12 | 12.2 (bedroom) | 177 / 228 |
| before r2 | 4 | hall 18.0; hall 16.2; bedroom+passage+kitchen 4.8; bedroom (no camera inside) 1.2 | 40.3 m² (34.5%) | 0 / 1 / 8 | 18.0 (hall) | 228 / 228 |
| after r1 | 2 | bedroom+passage+hall+kitchen 27.8; hall 10.0 | 37.8 m² (26.0%) | 1 / 1 / 6 | 27.8 (all four) | 228 / 228 |
| after r2 | 2 | bedroom 11.9; passage 6.8 | 18.7 m² (37.7%) | 1 / 1 / 8 | 11.9 (bedroom) | 121 / 228 |
| presentable default r1 | 1 | bedroom 10.0 | 10.0 m² (66.8%) | 0 / 1 / 6 | 10.0 (bedroom) | 122 / 228 |
| presentable pnp r1 | 3 | bedroom+hall 14.5; passage+hall 12.5; kitchen+passage 2.8 | 29.9 m² (0.4%) | 2 / 2 / 13 | 14.5 (bedroom+hall) | 228 / 228 |
| presentable default r2 | 1 | hall+passage 19.2 | 19.2 m² (35.9%) | 0 / 0 / 3 | 19.2 (hall+passage) | 71 / 228 |
| presentable pnp r2 | 2 | passage+hall+kitchen 16.8; bedroom 12.8 | 29.6 m² (1.3%) | 3 / 6 / 11 | 12.8 (bedroom) | 209 / 228 |
| **all nine** | 1–9 | | median error 26.0% | sums 9 / 18 / 75 (medians 1 / 1 / 8) | | |

**Sample videos, against the LiDAR plan of the same capture** (single_room 3 rooms, 17.61 m²; floor_only 8 rooms,
61.90 m²; with_ceiling 8 rooms, 62.50 m²).

| Run | Rooms (LiDAR) | Footprint vs LiDAR | Matched to a LiDAR room: plan vs LiDAR m² | Unmatched plan rooms, m² | Keyframes kept |
|---|---|---|---|---|---|
| single_room d069 r1 | 3 (3) | 17.83 m² (+1.2%) | 2: 9.29 vs 8.31 (+12%); 7.16 vs 7.56 (−5%) | 1.39 | 164 / 164 |
| single_room d069 r2 | 3 (3) | 17.10 m² (−2.9%) | 2: 9.05 vs 8.31 (+9%); 6.38 vs 7.56 (−16%) | 1.67 | 164 / 164 |
| single_room d069 r3 | 3 (3) | 16.72 m² (−5.05%) | 2: 9.11 vs 8.31 (+10%); 6.22 vs 7.56 (−18%) | 1.40 | 164 / 164 |
| single_room after run | 3 (3) | 21.12 m² (+19.9%) | 2: 10.21 vs 8.31 (+23%); 6.65 vs 7.56 (−12%) | 4.26 | 164 / 164 |
| floor_only after run | 0 (8) | 0 (−100%) | 0 | – | 372 / 432 |
| floor_only d066 | 5 (8) | 31.45 m² (−49.2%) | 3: 8.08 vs 13.99 (−42%); 5.65 vs 9.26 (−39%); 5.36 vs 3.62 (+48%) | 10.87, 1.49 | 211 / 432 |
| with_ceiling after re-run | 3 (8) | 54.28 m² (−13.2%) | 1: 48.35 vs 11.17 (+333%) | 4.67, 1.27 | 211 / 836 |

What the baseline says:

- On take1 the DPVO run decides most of the plan. The two caches from this morning's pnp live runs give 29.9 and
  29.6 m² with today's default scale step; the two default live runs' caches give 10.0 and 19.2 m².
- No cache gives the hall, kitchen, passage and bedroom as four rooms. The kitchen never gets a room of its own (at
  best kitchen+passage, 2.8–6.5 m²). The hall comes out more than once on 23:28 (four pieces) and before r2 (18.0 and
  16.2 m²). The bedroom is split on 23:28 (three pieces) and before r1 (12.2 and 1.7 m²). After r1 puts all four
  spaces into one 27.8 m² room.
- Three caches leave out a large part of the walk: segments that fail the self-check are dropped (121, 122 and 71 of
  228 keyframes kept). Presentable default r1 keeps only the bedroom, default r2 only the hall and passage.
- Where the scorer's bedroom room is the bedroom alone it measures 10.0–12.8 m² against 11.09 m².
- single_room: −5.05 to +1.2% on d069, +19.9% on the after run. The closet (LiDAR 1.93 m²) is never matched:
  1.39–1.67 m² on d069 and 4.26 m² on the after run (I-014). floor_only: 0 rooms, or 5 of 8 at −49%. with_ceiling:
  3 of 8 rooms, one of them 48 m² on an 11 m² LiDAR room.

## Reported, not judged

Rooms and the spaces they hold, walls within 3%, the bedroom room's area, matched room areas and unmatched rooms on the
samples, keyframes kept. They explain a result; they do not decide it. The 3% wall gate, ceilings and door widths are
not part of "better" here; today 0–3 of 14 walls are within 3%, 1 of 20 ceiling rows within 1.5 cm and 0 of 26 door
widths within 2 cm over the nine caches (`baseline/replay/eval/own_eval.md`).
