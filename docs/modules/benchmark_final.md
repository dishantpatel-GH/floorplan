# Benchmark: final report generator (Deliverable 5), all three tiers

What this is, in plain language: one script reads every benchmark run folder and writes the benchmark report
(`outputs/benchmark/final/REPORT.md`): the gates at all three tiers, the repeatability table, the head-to-head table
and timing. It never makes up a number: a run that has not happened yet becomes a "not run yet" or "pending" row,
and a run that crashed becomes a FAIL row with its error. It is safe to re-run at any time. After the photo v3 lands,
and again after the 08:00 own-home capture has been processed, re-running it refreshes the report.

Sections 1–8 quote runs made on 2026-10-04 between 03:43 and 05:02 IST, scored by the 05:05 `bench_final.py` run.
**Section 9 is the final run on the final code (05:08–06:28 IST, scored at 06:28) and supersedes sections 4 and 5
wherever they differ.** The 05:05 report is kept as `outputs/benchmark/final/REPORT_0505.md` (and `results_0505.json`). `REPORT.md` always shows the newest numbers. Where it and this page disagree, REPORT.md wins,
and this page says which run it quotes.

## 1. What was run

| Script (new) | What it does |
|---|---|
| `scripts/bench_final_runs.sh lidar` | The one command, `run_capture.py --tier lidar`, default settings, on the 3 sample captures, one after another → `final/lidar/<capture>/drift_on/` |
| `scripts/bench_final_runs.sh lidar_rerun` | The same command on floor_only a second time → `drift_on_rerun/` (determinism check, D-029) |
| `scripts/bench_final_runs.sh lidar_off` | `--no-drift` on floor_only and with_ceiling → `drift_off/` (the drift ablation) |
| `scripts/bench_final_runs.sh video [capture variant]` | `--tier video` on the 3 captures (their `rgb.mp4`) → `final/video/<capture>/default/`, or one capture again → `<variant>/` (`rerun` = run-to-run check, `current` = newer code) |
| `scripts/bench_final_runs.sh photo <label> [capture]` | `--tier photo` on the rotate-protocol folders → `final/photo/<capture>/<label>/` |
| `scripts/bench_final_arkit.py` | ARKitScenes against a Faro laser: the build and measure stages of `bench_arkitscenes.py`, unchanged, with the current code → `final/arkitscenes/` |
| `scripts/bench_tier_ref.py` | Scores a video or photo plan against the LiDAR plan of the same capture (reference-based, section 3) |
| `scripts/bench_final.py` | Scores everything and writes `results.json`, `REPORT.md`, `tier_scores/*.json` and `*.png` |

Every run folder records:

- `time.log`: `/usr/bin/time -v` output, plus stderr, which holds the traceback when a run crashes;
- `stdout.log` and `exit_code.txt`;
- `started.txt` and `finished.txt`;
- `load_before.txt` and `load_after.txt`;
- `code_fingerprint.txt`: md5 hashes of the pipeline code;
- `attempts.txt`: the number of tries after a CUDA out-of-memory retry.

**Regenerate everything** (from the repo root):

```
scripts/bench_final_runs.sh lidar; scripts/bench_final_runs.sh lidar_rerun; scripts/bench_final_runs.sh lidar_off
scripts/bench_final_runs.sh video
python scripts/make_photo_folders.py ../TakeHome/Dataset/single_room/c00a170fe1 \
    outputs/benchmark/final/lidar/single_room/drift_on/plan.json outputs/benchmark/final/lidar/single_room/drift_on/scene \
    --protocol rotate --out outputs/benchmark/final/photo_inputs
scripts/bench_final_runs.sh photo interim         # and later: photo v3
env -u PYTHONPATH python scripts/bench_final_arkit.py
env -u PYTHONPATH python scripts/bench_final.py   # ~3 min (LiDAR registration); --skip-lidar reuses lidar/results.json
```

When photo v3 lands (another agent, about 06:00). v3 also edits `plan/beta/extract.py`, so the LiDAR runs are
re-made too, to keep the reference current (about 9 min of CPU):

```
scripts/bench_final_runs.sh photo v3 && scripts/bench_final_runs.sh lidar && python scripts/bench_final.py
```

The gate table always shows the newest run per capture. Older code states move to the "earlier code states" table,
so the pre_v3 → interim → v3 trend stays visible.

After the morning capture:

```
python scripts/process_own_capture.py <capture root>   # writes outputs/own/eval/own_eval.json
python scripts/bench_final.py --skip-lidar             # fills sections 2, 3 and 6 of REPORT.md
```

## 2. Which earlier results were reused, and which were re-run (and why)

| Earlier evidence | Still valid? | What was done |
|---|---|---|
| `outputs/benchmark/lidar/*` (01:46–02:19) | **No.** After it was made, other agents changed `recon/drift.py` (D-029, 02:47), `recon/fusion.py` (03:10), `plan/beta/{grid,walls,params,tiers,extract}.py` (D-030 and the tier work, 02:46–03:34), `damage/__init__.py` (D-031, 02:56) and `run_capture.py` (03:36) | All LiDAR runs re-made, twice: a first pass at 03:43–03:56 (kept as `drift_on_0343/` plus `drift_on_rerun/`), and a final pass on the code of 04:32–04:52 (`drift_on/`, `drift_off/`) |
| `outputs/benchmark/arkitscenes/summary.json` (01:55) | **No.** It was written before D-027 (`bias.py`, 02:09). Its `plan_corrected.json` files carry the old "+2b on every wall" correction. Re-running only the `measure` stage would have re-measured those stale plans | Both stages, `build` and `measure`, re-run into `final/arkitscenes/` (drift on, 5 rooms, about 2 min) |
| Video and photo runs in `outputs/runs/plan_v4_*` | Made for development, not with the benchmark settings | New end-to-end runs |

Two checks show that the ARKitScenes re-run does reflect the current code:

1. The summary still prints the old offline "corner_aware" column: the raw error plus b × (s_start + s_end), as
   proposed in benchmark_arkitscenes_accuracy.md. It is now **identical** to the pipeline's "corrected" column on
   every row. Example, walls with convex ends: median 1.07 cm, 91% within 1.5 cm, in both columns. So the shipped
   D-027 code does exactly what was proposed.
2. The raw (uncorrected) numbers are unchanged from the 01:55 run: walls median 0.92 cm, room dimensions 1.35 cm.
   So D-029 and D-030 did not move the geometry of these 5 rooms. Only the correction changed.

## 3. How each tier is scored

### 3.1 LiDAR (sample captures): reused scoring, re-pointed at the new runs

`bench_final.py` imports `bench_lidar_eval.py` unchanged and points its folder constant at `final/lidar`. That
covers:

- the repeat pair (with_ceiling against floor_only): registration from scene geometry, then `judge_common.compare`
  with the gate max(1 cm, 0.5%), unmatched walls counted as failures, and a Wilson interval;
- the same-topology split and coverage;
- the opening table;
- single_room against each big capture;
- the drift on/off block: plan geometry, plausibility, the drift module's own report, the overlay figures;
- the per-room ceilings;
- damage confirmed and review counts.

Two things are new:

- `internal`: one stitched plan (connected components of the adjacency graph) and pairwise room-overlap area, for
  every LiDAR plan;
- `rerun_floor_only`: the same command run twice. Is the plan JSON identical apart from `meta` (timings)?

### 3.2 LiDAR (absolute): ARKitScenes against a Faro laser

These are the only absolute LiDAR numbers available, because the sample captures have no ground truth. The method is
in `docs/modules/benchmark_arkitscenes_accuracy.md`. GT walls are fitted on the laser, GT lengths run corner to
corner, GT ceilings are plane to plane. Nothing in that method was changed.

### 3.3 Video and photo: REFERENCE-BASED against the LiDAR plan of the same capture

**Problem.** The video and photo tiers need a truth to be scored against, and the samples have none.

**Options.**

| Option | For | Against | Verdict |
|---|---|---|---|
| A. LiDAR plan of the same capture as reference | Same rooms and the same moment. LiDAR room dimensions are within 1.2 cm of the laser on ARKitScenes (8/8 within 1.5 cm) | It is not ground truth. It carries LiDAR plan errors, e.g. a room that LiDAR splits differently | **chosen**, and every row is labelled "REFERENCE-BASED, not GT" |
| B. Video trajectory against ARKit poses only | Already done in video_tier.md | Says nothing about the plan, adjacency or overlaps | not enough |
| C. Wait for the own capture (tape) | Real ground truth | Not before 08:00, and the defense needs sample numbers too | done as well (section 6 of REPORT) |

**Room matching.** The video plan and the reference live in different frames.

- **Video:** the two footprints are registered rigidly. Both plans are Manhattan-aligned, so the yaw is tested in
  90° steps (4 yaws). For each yaw, an FFT cross-correlation on a 5 cm raster finds the translation; the best
  footprint IoU wins. Rooms are then matched with Hungarian assignment on polygon IoU (at least 0.2). Scale is
  deliberately **not** fitted: a scale error is part of what is being scored.
- **Photo:** the photo plan names each room after its folder (plan_beta T-3). Each folder is mapped to the reference
  room that contains the folder's spin-camera centre. The camera positions come from the simulator's truth file and
  are moved into the reference plan's frame by the reference scene's `T_align`. On all 3 captures every folder
  centre fell inside a reference room (distance 0.0 m). The photo geometry being scored is never used for the match.
  The footprint registration still runs, but only to draw the overlay figure.

**Metrics** (per matched room):

- the two bounding-box dimensions, sorted, and the floor area;
- for each: relative error, within tolerance (video 3%, photo 8%), and whether the tier's 95% interval covers the
  reference value;
- the footprint, scored the same way;
- adjacency, mapped through the room match: precision, and recall over all reference pairs;
- overlaps between the tier's own room polygons (more than 0.01 m²);
- one stitched plan: the tier's adjacency graph has exactly one connected component.

**The wall-length gate is scored on room dimensions.** Matching individual walls across a learned-depth plan and a
LiDAR plan without a shared frame is fragile. For a rectangular room the two bounding-box dimensions are the wall
lengths, so the dimensions are used as a proxy. The proxy is exact for rectangles and does not judge short jogs. The
own capture is scored per wall against the tape (`gt_eval.py`).

**Photo folders do not cover every room.** The simulator only makes a folder for a room with a usable spin:
floor_only has 6 folders for 8 reference rooms, with_ceiling 7 for 8. A whole-flat footprint can then never reach the
reference. So each photo run also gets a **fair subset** row: the footprint of the reference rooms that have a folder,
with room and adjacency recall counted over those rooms. Both rows are shown, and the gate verdict uses the
whole-flat row.

**single_room photo inputs.** `outputs/photo_inputs/single_room__c00a170fe1` is the old protocol: no doorway pairs,
no capture times. On it the photo tier gives **0 rooms** (pre_v3 row). A rotate-protocol set did not exist, so the
benchmark made one with the existing simulator (`make_photo_folders.py --protocol rotate`) from the final LiDAR run,
into `outputs/benchmark/final/photo_inputs/`. It has 3 folders and 9 photos. It has **no doorway pairs**, because the
37 s walk-through never stands in a door looking both ways.

## 4. Results

The quotes below come from `REPORT.md` as generated at the time stated in its header. The full tables, including
calibration and timing, are in REPORT.md.

### 4.1 LiDAR tier

| Gate | Verdict | Number (source run) |
|---|---|---|
| Repeatability, every wall within max(1 cm, 0.5%), with_ceiling vs floor_only | **FAIL** | 15/87 walls = 17.2% (Wilson 10.7–26.5%); 34 unmatched; same-topology 14/30, median 1.8 cm; rooms matched 8/8 |
| Same capture, same command, twice (floor_only) | **PASS** | plan.json identical apart from `meta` (footprint 61.9009 both times, 70/70 walls). Only the DXF header date differs. The earlier benchmark saw a 13.99 → 11.46 m² room flip; D-029 and D-030 removed it |
| Same capture, run at 03:43–03:47 and again at 04:32–04:35 (other agents edited `plan/beta/extract.py` and `tiers.py` in between) | floor_only and with_ceiling **identical**; single_room **changed** | single_room lost opening O2, a 1 cm "passage", to a new rule ("narrower than 0.45 m: not a real passage", `meta.dropped_openings`). With it went the R1–R2 adjacency (3 → 2 pairs). Areas, walls and footprint are identical (BF-9) |
| Opening widths ≤ 2 cm on ≥ 85% | not verifiable (no GT); repeat-pair proxy FAIL | 2/12 opening slots agree within 2 cm; matched median \|Δ\| 6.0 cm |
| Ceiling height ≤ 1.5 cm per room (ARKitScenes) | **FAIL** (3 of 4) | median \|e\| 0.53 cm. Room 47895909 is −2.16 cm (raw −3.52), but its interval (±2.69 cm) covers the laser value. The 5th room, 47430003, produced no room (plan covers too little floor) |
| Ceiling spread across captures ≤ 1 cm | not verifiable | only with_ceiling observes ceilings (8/8 rooms); floor_only 0/8, single_room 0/3 |
| Wall lengths (ARKitScenes, laser) | reported | median \|e\| 0.86 cm, 84% within 1.5 cm, max 8.81 cm (n 19). Before D-027: 1.15 cm, 74%. Hold-out walls: 1.03 cm, 5/5 within 1.5 cm |
| Room dimensions (ARKitScenes, laser) | reported | median 0.43 cm, 8/8 within 1.5 cm, max 1.18 cm |
| Drift accountability (method + on/off) | **PASS** | floor_only ON 8 rooms / 61.90 m², connected, floor recall 0.94; OFF 7 rooms / 52.59 m², not connected, 0.86. with_ceiling ON 8 / 62.50; OFF 8 / 62.16 with 82 against 70 walls. Loop surface separation (revisit loops, in-sample): floor_only 14.3 → 2.3 cm, with_ceiling 2.8 → 1.2 cm |
| One stitched plan, no overlaps (internal) | **PASS** | 3 / 8 / 8 rooms, each 1 component, 0 m² overlap |
| Damage false positives on the clean captures (0 confirmed) | **PASS** | 0 confirmed and 0 scope items on all three default runs; single_room has 2 review-only cracks (0.608, 0.148). The earlier floor_only "water stain" false positive is gone (D-031). Under drift-off, floor_only shows 1 review crack (0.557), also 0 confirmed |
| Calibration | reported | ARKitScenes lengths: 93% coverage (n 27, median half-width 6.4 cm). Repeat pair: same-topology walls 97%, all matched walls 68% |

**Timing** (one command, wall clock, final runs at load 1.4–9.8, other agents sharing the machine):
single_room **60 s**, floor_only **166 s**, with_ceiling **265 s**. Peak RSS 2.0 / 3.2 / 4.9 GB. Drift correction takes
2.9 / 18.9 / 35.7 s of that. ARKitScenes rooms: 3–34 s (pipeline only).

### 4.2 Video tier (reference-based)

| Capture [run] | Result |
|---|---|
| single_room [current, 04:58, current code] | 3 rooms. Footprint 16.43 m² [11.65, 21.20] against 17.61 (−6.7%, covered). Dimensions within 3%: **2/4** (median 2.5%, max 3.6%). The 3rd room does not match the LiDAR R3 (the T-I3 door leak), so only 2/3 rooms match. Adjacency: 0/3 reported pairs are in the reference; the reference has only 2 pairs since BF-9. No overlaps, 1 component. Interval coverage 7/7, median half-width 15.6%. **270 s**, 5.9 GB |
| single_room [default, 03:43, earlier code] | 3 rooms, 17.31 m² (−1.7%), dimensions 2/4 within 3% (R2 +0.2% / +1.1%, R1 +12.4% / +12.6%), coverage 7/7. 334 s |
| floor_only [default, 03:48] | **0 rooms.** The run finished (exit 0, 1130 s). It produced 4 DPVO segments, 0/4 bridges verified at the restarts, `whole_scene_consistent=False`, and a floor at −3.31 m (the rerun found −1.45 m). The levelling/floor went wrong, so no room survived |
| floor_only [rerun, 04:41, same command] | **8 rooms**, 38.43 m² against 61.90 (−37.9%), not covered. Dimensions within 3%: 0/8. Adjacency 0/5 correct. 3 components. Coverage **31%** (n 13) with a median half-width of only 12.8%: **confident garbage**. 955 s |
| with_ceiling [default, 04:08] | **No crash this time.** The task expected the missing-floor crash (plan_beta T-I9), but this run found a floor and produced 8 rooms. Footprint 39.49 m² [24.90, 54.08] against 62.50 (−36.8%), not covered. Dimensions within 3%: 2/8 (median 26.2%). 4/8 rooms match. Adjacency 2/6 correct, 2/10 found. 2 components. Coverage **38%** (n 13), half-width 21%. The front end itself said `whole_scene_consistent=False`, with 11 VO restarts not joined by verified geometry. 1996 s, 7.3 GB |

Same command, same video, run twice: floor_only gave **0 rooms, then 8 rooms** (video front-end non-determinism, BF-5).
### 4.3 Photo tier (reference-based)

Three code states were run:

- **pre_v3** (03:51): the code while the v3 agent was editing it;
- **interim** (03:58): after its `_tilts` guard;
- **v3**: when it lands (`scripts/bench_final_runs.sh photo v3`).

| Capture [label] | Rooms (matched / folders) | Footprint vs ref (fair subset) | Dims within 8% | Adjacency correct / reported; found / ref | Stitched | Coverage |
|---|---|---|---|---|---|---|
| single_room [interim] | 2 / 3 | 13.29 vs 17.61 m² (−24.5%), covered | 0/4 (median 8.7%) | 0/0; 0/2 (reference pairs after BF-9) | no (2 components) | 100% (half-width 33%) |
| floor_only [interim] | 5 / 6 | 34.44 vs 61.90 (−44.4%); subset 57.47 (−40.1%), not covered | 1/10 (median 26.4%) | 1/2; 1/11 | no (3) | 50% |
| with_ceiling [interim] | 7 / 7 | 54.16 vs 62.50 (−13.3%); subset 60.08 (−9.8%), covered | 0/14 (median 24.2%) | 4/5; 4/10 | no (2) | 77% (half-width 64%) |
| with_ceiling [pre_v3] | crash | `KeyError: 'R1-W7'` in `plan/beta/extract.py::_tilts` | | | | |
| single_room [pre_v3] | 0 rooms | old-protocol input (no doorway pairs, no times) | | | | |

Every photo gate fails on the current code. The footprint and adjacency trend from pre_v3 to interim is in REPORT.md.

## 5. Issues log (found while benchmarking)

| # | Issue | Root cause (evidence) | Status |
|---|---|---|---|
| BF-1 | Earlier LiDAR and ARKitScenes benchmark numbers no longer described the code | 11 pipeline files changed after them (section 2). The ARKitScenes `corrected` column still applied +2b to every wall | All re-run. Every run now stores a code fingerprint, and REPORT lists pipeline files changed after each run |
| BF-2 | Photo pre_v3 with_ceiling crashed: `KeyError: 'R1-W7'` in `_tilts` | The photo v3 work replaces a room's walls, so a wall id built from the room's edge lines no longer exists. Seen in the code state of 03:51 | The v3 agent's guard (`if ... in by_id`) is already in `extract.py`. The interim run passes (7 rooms) |
| BF-3 | Photo single_room gives 0 rooms | The input is the old protocol: 2 folders, no doorway pairs, no times. Log: "0 doorway pair(s) … 1 linked block, 1 fallback, 5 unplaceable fragment(s)" | Benchmark input regenerated with the rotate protocol (section 3.3). On it: 2 rooms, still not stitched (0 pairs: the 37 s video has no doorway pair) |
| BF-4 | Photo interim run 1 died with CUDA OOM | The video SfM held 6.45 GB of the shared 8 GB GPU | `bench_final_runs.sh` now waits for ≥ 3 GB free and retries an OOM up to 3 times. Failed attempts' stderr is kept |
| BF-5 | Video floor_only: 0 rooms on one run, 8 rooms on the next (same command, same input) | Video front-end non-determinism (video_tier.md 13c: GPU feature matching and DPVO are not bit-deterministic). Run 1: 4 segments, floor −3.31 m, so the levelling was wrong. Run 2: floor −1.45 m | Open (video front end). Reported as a repeatability failure of the video tier |
| BF-6 | Editing `bench_final_runs.sh` while it ran could have made bash execute part of the new text | bash reads a running script from the file by byte offset; an in-place rewrite changes what it reads next | The original bytes were restored in place; new versions are installed by `mv`, which gives a new inode. Rule: never edit a running shell script in place |
| BF-7 | `pkill -f "<pattern>"` killed the shell that ran it | The pattern also matched the invoking command line | Kill by PID |
| BF-9 | LiDAR single_room plan changed between two runs of the same command 50 min apart | Not non-determinism: a rule added to `plan/beta/extract.py` at 04:18 by another agent drops openings narrower than 0.45 m. O2 was a 1 cm phantom "passage" (width 0.010 m [−0.015, 0.035], first noted in plan_beta v4.5). Adjacency is reported "via" an opening, so R1–R2 disappeared with it. As a side effect, the video single_room adjacency score went from 1/3 to 0/3 correct without any change in the video. floor_only and with_ceiling are unchanged (`code_change_check`) | Reported. Probably a correct fix: a 1 cm passage is not a door. But "adjacent" should also cover rooms that share a wall with no door, otherwise adjacency recall depends on door detection. Owner: plan_beta |
| BF-10 | Video plans with `whole_scene_consistent=False` carry narrow intervals (floor_only rerun: 31% coverage, half-width 13%; with_ceiling: 38%, 21%) | The front end detects that the scene is unreliable (unjoined restarts, untrusted segments), but the plan's intervals only get the scale term (worst trusted segment, 7–9%) and the surface noise. Room placement across unjoined segments carries no uncertainty at all | Open. Proposed fix (video and uncertainty owners): when `whole_scene_consistent` is False, mark the plan `unreliable`, and widen room positions and the footprint by the spread across segments, or refuse the stitched plan and return per-segment plans |
| BF-11 | Video with_ceiling did NOT crash on a missing floor this time | The front end is not repeatable from run to run (BF-5): this run found a floor. The T-I9 crash path still exists for runs that find none | Recorded. The earlier crash is documented in plan_beta T-I9 |
| BF-8 | ARKitScenes ceiling gate 3/4 (room 47895909 −2.16 cm) | The earlier "all 5 within 0.7 cm" was a room-box plane-to-plane measure. This is the plan extractor's per-room ceiling against laser planes. The raw value is −3.52 cm, and D-021's +1.36 cm correction brings it to −2.16 | Reported. Its 95% interval (±2.69 cm) covers it, so it is not confident garbage |

## 6. Limitations

1. Video and photo scores are **reference-based**. A LiDAR mistake, such as a room split differently, shows up as a
   tier error. Section 4.1 bounds LiDAR room dimensions at about 1 cm (ARKitScenes, 8/8 within 1.5 cm); room
   segmentation has no such bound.
2. The wall-length gate for video and photo is judged on room bounding-box dimensions (section 3.3).
3. Photo inputs are simulated from walk-through frames (D-014, D-017). Their doorway pairs and spins are imperfect,
   and single_room has no doorway pair at all. The own capture is the real test.
4. The timing runs shared the CPU and GPU with other agents (load up to about 23). Wall times are upper bounds.
5. The video tier is not repeatable run to run (BF-5; video_tier.md 13c). One run per capture is a sample, not a
   distribution.

## 7. Proposed commits (replay order)

| # | Message | Files |
|---|---|---|
| 1 | `bench: one runner for the final benchmark runs (fingerprint, load, OOM retry)` | `scripts/bench_final_runs.sh` |
| 2 | `bench: re-run ARKitScenes build+measure on current code (D-027 now in the plans)` | `scripts/bench_final_arkit.py` |
| 3 | `bench: reference-based scoring of video/photo plans against the LiDAR plan of the same capture` | `scripts/bench_tier_ref.py` |
| 4 | `bench: final report generator (gates x 3 tiers, repeatability, head-to-head, timing, calibration)` | `scripts/bench_final.py` |
| 5 | `docs: final benchmark module + generated REPORT.md` | `docs/modules/benchmark_final.md`, `outputs/benchmark/final/REPORT.md` |

## 8. Concepts for the defense

- **Reference-based is not ground truth.** Video and photo are scored against LiDAR because it is the best measured
  thing in the same rooms. Its absolute error is known separately, from ARKitScenes against a laser. Every such row is
  labelled.
- **Why the dimensions proxy.** Matching walls one by one needs a shared frame and the same wall split, and
  learned-depth plans have neither. For a rectangle, the two room dimensions are the wall lengths.
- **Fair subset.** Do not blame the pipeline for rooms nobody photographed. Do not hide it either: both rows are
  shown, and the gate uses the whole flat.
- **Calibration has two numbers.** Coverage near 95% means the intervals are honest. Coverage near 100% with ±60%
  half-widths is honest but useless. Coverage of 50% means the tier is confidently wrong.
- **Determinism is a precondition for repeatability.** The LiDAR command now gives the same plan twice (D-029). The
  video front end does not, so its failures (BF-5) are partly luck. That is stated, not hidden.
- **Idempotent report.** It was smoke-tested on a synthetic `own_eval.json` (GT made from the LiDAR plan, a fake app
  export, in the scratchpad, never in `outputs/own`): sections 2, 3 and 6 fill in, and with no file they show
  "pending". The report is a function of the run folders: re-running the script never changes a number
  unless a run changed. Missing runs show as pending, never as zero.
- **Staleness tracking.** In a multi-agent repo the code moves under the benchmark. Fingerprints and a "changed
  since" column make a stale number visible instead of silently wrong.

## 9. Final run (2026-10-04, 05:08–06:28 IST, final code)

All runs in this section share one code fingerprint (`57bec9dc`, LiDAR path `f6e659b2`, in each run's
`code_fingerprint.txt`); no pipeline file changed during or after them (REPORT section 4, last column "none").
Report: `outputs/benchmark/final/REPORT.md`, generated 06:28. The previous report is kept as `REPORT_0505.md`.

### 9.1 What was run

One sequential chain on the free GPU, so that no two heavy jobs overlapped (log: `final/runs_final_chain.log`):

```
scripts/bench_final_runs.sh video "" final              # 05:08-05:55  video/<capture>/final/
scripts/bench_final_runs.sh video floor_only final_rerun # 05:55-06:10  I-007 check: same command, second time
scripts/bench_final_runs.sh photo v3                     # 06:10-06:12  photo/<capture>/v3/ (rotate-protocol folders)
scripts/bench_final_runs.sh lidar; ... lidar_rerun; ... lidar_off   # 06:12-06:26
env -u PYTHONPATH python scripts/bench_final_arkit.py   # 06:26-06:28
env -u PYTHONPATH python scripts/bench_final.py         # 06:28, about 2 min
```

Before the LiDAR runs overwrote `drift_on/` and `drift_off/`, those plans were copied (without `scene/`) to
`drift_on_0432/`, `drift_off_0445/` and `drift_on_rerun_0353/`, and the ARKitScenes summary to `arkitscenes_0346/`.

`bench_final.py` changes (scoring only, no pipeline code):

- `X_rerun` pairs with `X` in the repeatability table, like `rerun` pairs with `default`. The row now also shows
  segments, `floor_y`, `whole_scene_consistent` and footprint coverage for both runs.
- Every video and photo capture gets a "plan self-report" row: the D-034 reliability flag, the tier scale sigma, the
  front end's segments, `whole_scene_consistent` and `floor_y`, the D-032 floor fallback and the D-033 dropped
  openings.
- The LiDAR code-change check also compares against the 04:32 and 04:45 snapshots.
- The LiDAR internal row shows adjacency pairs and the openings that D-033 dropped.

### 9.2 What changed in the code since the 05:05 report, and its measured effect

| Change | Effect in this run | Evidence |
|---|---|---|
| Photo v3 (`floorplan/photo/layout.py`): room sizes from each room's own spin | single_room 2 → 3 rooms. Footprint −24.5% → +16.5%. floor_only: dimensions within 8% go from 1/10 to 3/12, and 6/6 rooms are matched (was 5/6). with_ceiling: footprint −13.3% → −26.9%, dimensions still 0/14. The numbers match photo_tier.md v3.5 (45.7 and 36.5 m²) | REPORT §1, "earlier code states" |
| D-032 floor fallback | **Not exercised.** Every video and photo run found a floor (`floor fallback: not used` in each self-report row). The with_ceiling crash it guards against (T-I9) did not reproduce, so this run neither confirms nor refutes the fallback path | REPORT §1 self-report rows |
| D-033 implausible-opening filter (moved into `run_capture.py`) | LiDAR plans are bit-identical to the 04:32/04:45 runs (all 5 code-change checks: identical geometry True). single_room still drops O2 (passage, 0.010 m). With it goes the R1–R2 adjacency, so the reference has 2 pairs (R1–R3 via O1, R2–R3 via O3), not 3. **Effect on single_room video adjacency:** the final video plan reports R1–R2, R1–R3 and R2–R3. R1–R2 maps to reference R1–R2, the pair D-033 removed, so it is scored wrong: 2/3 correct and FAIL. Against the 03:43 reference (3 pairs, O2 present) the same video plan would be 3/3. The video finds that connection independently of LiDAR. That is weak evidence that R1 and R2 really connect and that LiDAR measured the passage as 1 cm wide. D-033 then deleted the adjacency along with the phantom width (BF-9 / BF-14) | `lidar/single_room/drift_on{,_0343}/plan.json`, `tier_scores/video__single_room__final.json` → adjacency.pairs |
| D-034 video sigma floor (20%) + `meta.reliability = "low: ..."` when `whole_scene_consistent` is False | Fired on floor_only (both runs) and with_ceiling. It did not fire on single_room (1 segment, consistent). Long-video interval coverage: floor_only 31% (rerun at 04:41) → **88%** (n 16, median half-width 41%). with_ceiling 38% → **60%** (n 10, half-width 34%). floor_only second run **70%**. The footprint interval now covers LiDAR on floor_only/final (40.91 [8.82, 73.01] against 61.90). It still misses on with_ceiling (19.40 [4.16, 34.64] against 62.50, −69%) and on floor_only/final_rerun (18.95 [4.07, 33.83], −69%) | REPORT §1 and §5 |
| cuDNN determinism in `dpvo_runner.py` (I-007) | **The floor_only outcome is still not stable.** Run 1: 5 rooms, 40.91 m², 2 segments. Run 2: 3 rooms, 18.95 m², 4 segments (one untrusted). Root cause in 9.4 | REPORT §2, `video/floor_only/final{,_rerun}/stdout.log` |

### 9.3 Numbers before → after (05:05 report → 06:28 report)

Gate counts (REPORT §0): LiDAR 3 PASS / 3 FAIL, unchanged. Video 3 / 8 → 2 / 13. Photo 1 / 16, unchanged. The video
count rose partly because floor_only is now scored on a run with rooms; the 05:05 gate table used its 0-room
`default` run as one FAIL row. The 3 "reported" rows per tier are the new self-report rows.

| Tier / capture | 05:05 report (run) | 06:28 report (final run) |
|---|---|---|
| LiDAR, all | repeat pair 15/87 (17.2%); ARKitScenes walls 0.86 cm median, ceilings 3/4; drift ON/OFF 61.90/52.59 and 62.50/62.16 m²; 0 confirmed damage; rerun identical | **identical** (same plans, bit for bit; ARKitScenes summary table identical to the 03:46 run) |
| LiDAR timing (s) single_room / floor_only / with_ceiling | 60 / 166 / 265 | 52 / 146 / 238 (lower load: 1-min load 3.5–5.3, against up to 9.8) |
| video single_room | [current 04:58] 3 rooms, 16.43 m² (−6.7%, PASS), dimensions 2/4, adjacency 0/3, coverage 100% | [final] 3 rooms, 23.46 m² (+33.2%, FAIL), dimensions 2/6 (median 9.2%), adjacency 2/3 (2/2 reference pairs found), coverage 70% (n 10). Now all 3 rooms match, and the T-I3 door-leak room (7.74 m² against the reference R3, 1.93 m²) is scored instead of left out. R1 is +9% (was −0.1% / +3.4%) |
| video floor_only | [default 03:49] 0 rooms; [rerun 04:41] 8 rooms, 38.43 m² (−37.9%), coverage 31% | [final] 5 rooms, 40.91 m² (−33.9%, covered), dimensions 3/10, adjacency 5/6 correct (5/11 found), 1 component, coverage 88%; [final_rerun] 3 rooms, 18.95 m² (−69.4%, not covered), dimensions 0/6, coverage 70% |
| video with_ceiling | [default 04:08] 8 rooms, 39.49 m² (−36.8%), 4/8 rooms matched, adjacency 2/6, 2 components, coverage 38% | [final] 5 rooms, 19.40 m² (−69.0%, not covered), 3/8 rooms matched, adjacency 0/1, 4 components, coverage 60%, flagged low reliability |
| photo single_room | [interim] 2 rooms, 13.29 m² (−24.5%), 2 components | [v3] 3 rooms, 20.51 m² (+16.5%, covered), dimensions 0/6, 0 pairs (no doorway pair in the input), 3 components, coverage 100% at a half-width of 79% |
| photo floor_only | [interim] 5/6 rooms, 34.44 m² (−44.4%), dimensions 1/10, coverage 50% | [v3] 6/6, 36.53 m² (−41.0%), dimensions 3/12, adjacency 1/2, 4 components, coverage 58% |
| photo with_ceiling | [interim] 7/7, 54.16 m² (−13.3%), dimensions 0/14, adjacency 4/5, coverage 77% | [v3] 7/7, 45.70 m² (−26.9%, covered), dimensions 0/14, adjacency 4/5, 2 components, coverage 77% |
| video wall time (s) | 270 (single_room current) / 1130 (floor_only default) / 1996 | 272 / 794 / 1737 (second floor_only run 895) |
| photo wall time (s) | 27 / 76 / 127 (interim) | 9 / 37 / 45. **Warm cache:** "depth + gravity ... loaded from cache" in every v3 log (the interim runs also hit the cache on with_ceiling). These are not cold-start times |

### 9.4 Residual failures and root causes

| # | Failure | Root cause (evidence) | Owner / next step |
|---|---|---|---|
| BF-12 | Video floor_only: the same command twice gives 5 rooms / 40.91 m² and then 3 rooms / 18.95 m², even with cuDNN deterministic (I-007 **not** fixed) | Two independent sources, measured: **(a)** SfM features and matches are bit-identical across the two runs (all 480 feature and 2290 match datasets in `sfm_calib/*.h5`), and the imported matches table in `database.db` is identical. But the **two-view geometry verification** table differs (md5 `aab95d7a` against `97e8b7c0`). This is pycolmap's RANSAC match verification, run by hloc; it is not covered by `random_seed`/`num_threads=1` in `mapper_options`. So SfM registers 979 against 987 points, the self-calibrated focal is 1564.6 against 1566.0 px, and DPVO gets different intrinsics. **(b)** DPVO itself is not deterministic. A controlled test (`final/dpvo_determinism/`: `run.sh`, `run{1,2}.npz`; `dpvo_runner.py` twice with identical arguments, frames 0–1200 of the same video, cuDNN flag on) gave poses that differ from frame 1 on. After a similarity alignment the shape agrees to a median 0.05% of the path diagonal (max 0.24%), and the arbitrary monocular scale differs by 2× (0.51). The likely source is the custom CUDA kernels (atomics) and `autocast` mixed precision (`dpvo.py:332`). Small differences then flip discrete decisions downstream: restart detection (fresh DPVO runs 1 against 2), segments 2 against 4, a segment failing the self-check, levelling (median camera pitch −22.2° against −27.4°) | Video front end. Seed or serialise the two-view verification (pycolmap `TwoViewGeometryOptions` RANSAC seed / one thread), or cache `calib.json` per video hash. DPVO: try `MIXED_PRECISION` off for the fresh runs. Neither change is made here: this is a pipeline change |
| BF-13 | D-034 widens the intervals, but the footprint still misses LiDAR on 2 of 3 long-video runs (−69%: upper bounds 34.64 and 33.83 m², about 28 m² below the reference) | The sigma floor is a **scale** term. The large errors are **missing rooms**: with_ceiling 5 rooms for 8 (3 matched), floor_only rerun 3 for 8. A scale sigma cannot cover area that is not in the plan | Video and uncertainty owners: when unreliable, report the footprint as a lower bound (or `coverage_warning` with the unmatched floor fraction), not as a symmetric interval |
| BF-14 | single_room adjacency verdicts depend on D-033 | D-033 removes O2 (1 cm) and, with it, the R1–R2 adjacency. Video and the 03:43 LiDAR plan both connect R1–R2 (9.2) | plan_beta: when a sub-threshold opening is dropped, keep a "shares a wall / connected" adjacency instead of deleting the link (already proposed in BF-9) |
| BF-15 | Video single_room got worse (footprint −6.7% → +33.2%) with no relevant pipeline change | The 3rd room (the T-I3 door leak) now registers onto reference R3 and is scored: 7.74 against 1.93 m². R1's scale moved +9%. The run between them differs only by the cuDNN flag and by the run-to-run variation of BF-12. Both runs are single-segment and `whole_scene_consistent=True`, so D-034 does not widen them | Video (T-I3 door leak); repeatability per BF-12 |
| — | Photo: every size gate fails; with_ceiling footprint is worse under v3 (−26.9%) than at the interim state (−13.3%) | Simulated spins see part of each room; layout sides over- or under-reach (photo_tier.md v3.6). The interim with_ceiling footprint was closer by luck of fragments (its dimensions were 0/14 as well) | Photo owner; the 08:00 own capture (tape) is the real test |
| — | LiDAR repeat pair 17.2%, ceilings 3/4 (ARKitScenes) | Unchanged from section 5 (BF-8; repeat pair: different wall splits between captures) | Unchanged |

### 9.5 Verdict for the defense

- **LiDAR** is stable and deterministic on the final code: identical plans across the 04:32 and 06:12 code states and
  across reruns.
- **Video** is now *honest* on the long walkthroughs:
  - the flag fires;
  - coverage is 60–88%, against 31–38% before;
  - it is still **not repeatable** (BF-12), and on two of three long runs the footprint is wrong by more than its
    widened interval (BF-13).
- **Photo v3** gives every folder a room. It fails every size gate on simulated spins.
