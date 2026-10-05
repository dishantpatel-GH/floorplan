# Benchmark results

Fresh runs of every benchmark capture with the code at **`525a70a`** (HEAD on 5 Oct 2026, "photo: door stitching on,
judged on k65"), made on 5 Oct 2026 between 16:00 and 16:38 IST, scored by `scripts/bench_results.py` at 16:44.

| File | What |
|---|---|
| `tables.md` | the tables the report uses: gates against ground truth, repeatability, sample captures, drift |
| `summary.json` | every number behind the tables: per wall value, 95% interval, truth, error; openings; ceilings; footprint; timing; the run folder of each number |
| `plans/` | each capture's homeowner drawing (`*_presentation.png`) and technical drawing (`*_technical.png`), max 1600 px wide |
| `overlays/` | plan over the ground truth: own house over the tape outline, k65 over the simulator rooms, sample video over the LiDAR plan of the same capture, the LiDAR repeat pair; and our own-house plan next to CubiCasa's |

The run folder is `outputs/benchmark/fresh_1600/` (git-ignored; logs, `time.log` of every run, the exact code as a
git archive in `code/`, `CODE.txt`).

## Headline

| Capture, tier | Result |
|---|---|
| own house photo, dim (live) | 5 of 6 walls within ±8% (median error 3.6%; the 12.6 cm step missed); footprint +0.6%; ceiling 2.56 m vs 2.6289 m tape (−7 cm); openings 3 of 3 found, widths off by 2.7–6.3 cm |
| own house photo, lit (live) | 2 of 6 walls within ±8% (median 8.9%); footprint +4.0%; ceiling 2.51 m (−11 cm); openings 2 of 3 |
| repeatability, dim vs lit (same room) | walls: median |dim − lit| 24.9 cm (10.3%), max 47 cm, 1 of 6 within 8% of each other; ceiling 4.8 cm apart; door D2 and the window repeat within 0.2 cm |
| own house video take1 (9 replays of cached tracks) | 0–3 of 14 walls within ±3% per run; median wall error 14.8–31.0%; footprint 27.4–31.3 m² vs 28.0 m² tape outline (−2.3% to +11.9%) |
| k65 photo (live, stitched) | 11 of 26 walls within ±8% (median 6.6%, 8 missed); footprint −3.8%; plan IoU 0.41; 8 of 9 openings found |
| k65 LiDAR (live) | median wall error 1.8%, 4 of 26 within max(1 cm, 0.5%); footprint −3.0%; IoU 0.93; ceilings +1 cm; opening widths median 0.6 cm |
| k65 video (replan of the 4 Oct run, older front end) | 0 of 26 within ±3%; footprint +127%; IoU 0.30. Indicative only |
| sample LiDAR (live) | the three plans are identical to the 10:04 runs of the same command (deterministic); repeat pair 15 of 87 walls within max(1 cm, 0.5%) (Wilson 11–27%), same-topology walls 14 of 30, median 1.8 cm |
| sample video vs LiDAR (replays) | room dimensions within ±3%: single_room 0–2 of 4 (4 caches), floor_only 1 of 14 and 1 of 8, with_ceiling 0 of 6 |
| drift off vs on (LiDAR) | floor_only footprint 52.6 vs 61.9 m² (7 vs 8 rooms); with_ceiling 61.9 vs 62.5 m² (9 vs 8 rooms) |
| time (pipeline runtime) | photo 36–41 s (6–7 photos), 150 s (k65, 31 photos); LiDAR 77 / 234 / 472 s (samples), 440 s (k65); video live runs behind the caches 559–951 s (take1), replan alone 43–72 s |

All rows and their 95% intervals are in `tables.md` and `summary.json`.

## What was run, and how

| Capture | Tier | Input | Run | Ground truth |
|---|---|---|---|---|
| own house, one room, dim light | photo | 6 photos (`photos/room_take2`) | live, no cache (new photo work folder) | tape (`gt/ground_truth.csv`), ceiling 2.6289 m |
| own house, same room, lit | photo | 7 photos (`photos/room`) | live, no cache | tape |
| own house | video | `take1.mp4` | **replay of cached tracks**: the 9 cached DPVO/MoGe-2 runs of take1, plan step and everything after it re-run with the HEAD code | tape |
| k65 (simulated flat) | photo | 31 photos in 5 room folders, 1x lens | live, no cache, stitched | simulator geometry (`sim_gt.json`) |
| k65 | LiDAR | `lidar/take1` | live | simulator geometry |
| k65 | video | `video/take1.MOV` | **replan of a cached run** (see below) | simulator geometry |
| sample single_room, floor_only, with_ceiling | LiDAR | Stray Scanner folders | live | none; the LiDAR plan is the reference for the video runs |
| sample floor_only, with_ceiling | LiDAR, `--no-drift` | the same folders | live (drift ablation) | the drift-on plan of the same capture |
| sample captures | video | the Stray folders' `rgb.mp4` | **replay of cached tracks** (single_room 4 caches, floor_only 2, with_ceiling 1) | the LiDAR plan of the same capture (REFERENCE-BASED, not ground truth) |

Why the video numbers come from replays: a live video run is 9–12 min on the 8 GB GPU per take (k65: about 1 h),
and the brief accepts deterministic caches when the live path also runs. The video front end has not changed since
the 'auto' replays of 5 Oct (`outputs/video_auto/replay/`, D-087; `git diff 20b8ece 525a70a -- floorplan/video` is
the default flip only), so each number here is the HEAD plan step, openings, room names and drawings on the scene
that the HEAD front end makes from that cache. The 9 take1 caches come from 9 live runs of the same video, so their
spread is the tracker's run-to-run spread.

k65 video: the HEAD front end picks 1597 keyframes where the cached run of 4 Oct 17:58 picked 1823, so that cache
cannot be replayed ("cache miss: keyframe export would run"), and a live run takes about an hour. The k65 video row
is the HEAD plan step on the scene of that 4 Oct run (older video front end). Treat it as indicative only.

No live take1 video run was added this time: the GPU queue reached it at 16:32, and a live run takes 9–16 min, so
it would have ended after the 16:45 cut-off. The live path is the one that made the 9 caches (live runs of 4–5 Oct,
559–951 s each, times in `summary.json`).

Photo and LiDAR runs are live. Two runs (k65 photo, floor_only LiDAR) lost their `time.log` and `stdout.log`
(the wrapper script was edited while they ran: `WRAPPER_FAULT.txt` in their folders); their plans are complete and
their times come from `run_report.json` (`runtime_s`), as for every other row.

Timing caveat: the machine was shared with other jobs (two of ours side by side, plus other work on the same
machine; CPU at 95–100 °C). Times are upper bounds. The with_ceiling LiDAR run took 474 s here against 238–374 s
in the earlier final runs.

Not re-run (cited): the fix-loop before/after numbers (`docs/FIX_LOOP.md`, tags `before-fix` and `after-fix`), the
ARKitScenes laser comparison (`docs/modules/benchmark_arkitscenes_accuracy.md`).

## Regenerate

From the repo root (GPU steps take `outputs/.gpu.lock`):

```bash
scripts/bench_fresh.sh                      # every run above into outputs/benchmark/fresh_<HHMM>/, then results/
VIDEO=live scripts/bench_fresh.sh           # same, video tier live (the caches exist only on the machine that made them)
env -u PYTHONPATH python scripts/bench_results.py outputs/benchmark/fresh_1600   # re-score only (about 1 min)
```

One number at a time:

| Number | Command |
|---|---|
| own house photo, dim | `python scripts/run_capture.py <dir with room/ -> photos/room_take2> --tier photo --out X` then `python scripts/eval_own_capture.py --gt data/own_house/gt/ground_truth.csv --gt-json data/own_house/gt/gt_polygons.json --plan photo_dim=X/plan.json` |
| own house photo, lit | the same with `photos/room` |
| own house video (one cache) | `python scripts/replan_run.py outputs/video_auto/replay/<cache> --out X`, scored as above with `--plan video_<cache>=X/plan.json` |
| k65 photo | `python scripts/run_capture.py <k65 1x photo folders> --tier photo --out X`; `python scripts/eval_own_capture.py --gt data/k65/gt/ground_truth.csv --gt-json data/k65/gt/sim_gt.json --plan photo_k65=X/plan.json`; IoU and overlay: `python scripts/eval_door_stitch.py --run X --gt data/k65/gt/ground_truth.csv --png X.png` |
| k65 LiDAR | `python scripts/run_capture.py data/k65/lidar/take1 --tier lidar --out X`, scored as k65 photo with `--plan lidar_k65=X/plan.json` |
| sample LiDAR | `python scripts/run_capture.py data/sample/<capture>/<scan> --tier lidar --out X` |
| sample video vs LiDAR | `python scripts/bench_tier_ref.py <video plan.json> <lidar plan.json> --tier video --ref-scene <lidar run>/scene` |
| LiDAR repeat pair, drift on/off | `scripts/bench_results.py` (calls `scripts/bench_lidar_eval.py`: registration from the scene geometry, then the max(1 cm, 0.5%) wall gate) |
