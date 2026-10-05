# Fix loop (Part 4): declaration, shipped fix, before/after

Status, 5 Oct 04:50 IST. **Declared: the video wall gate. Shipped: scale votes from PnP (`fbad1b3`, tag `after-fix`).
The gate still fails, and the prediction was wrong.** Part 1 was filled in from the scored before runs and committed
**before** any fix code (tag `before-fix`). The after runs, the replays and the verdict are appended under it; the
declaration itself is unchanged. Part 2 lists the candidates as they stood on 4 Oct. Part 3 is the procedure; its
ranking rule (step 3) is word for word the one committed in `6a04490`. Every number here is quoted from the file named
next to it.

Final state, 5 Oct (afternoon). The shipped fix lives on in the default. D-076 made the PnP scale opt-in again after
it lost on the sample videos; since 14:55 the video scale step is `"auto"` (D-087), which keeps PnP's scale on every
segment whose own votes cover at least half of it and takes depth agreement elsewhere: the per-segment rule that
D-076's "revisit if" named. The declaration, the after runs and the verdict below are unchanged. What the final code
gives on the declared gate is in `docs/BENCHMARK_REPORT.md`.

Correction, 5 Oct (review). Part 1, item 1 says the rule was "committed before the own capture was scored". That is
wrong. `6a04490` went into git at 00:03 on 5 Oct, with the replayed history (`README.md`, History). The own capture was
first scored before that: the photos at 23:30 and photos plus video at 23:38 on 4 Oct (`outputs/own_house/eval_photo/`,
`outputs/own_house/eval_all/`). The rule's text is older than the capture: the scratch copy of this file was last saved
at 21:17 on 4 Oct, its step 3 is the same as in `6a04490`, and the photos were taken at 22:38–22:40 (their file names).
That evidence is a file time on my laptop. Git does not show it.

---

## Part 1. Fix declaration (one page; fill in, commit, then fix)

Filled in on 5 Oct at 02:31 IST, before any fix code. The ranking behind it is in `fixloop_ranking.md`.

> **1. Worst gate and its failing number.**
> Gate: walls within ±3% at tier video, on my own capture (`take1.mp4`, 117 s, 14 tape walls in the bedroom, hall
> and kitchen). Measured: **0 of 14 walls within 3%** in both before runs, against 14 of 14 needed. The better run,
> r1, finds 9 of 14 walls with a median error of 20.1%. r2 finds 9 with 28.6%
> (`outputs/fixloop/before/eval/own_eval.md`, scored at 02:24 on 5 Oct, code `4f9d0e0`). It is the worst gate under
> the ranking rule of Part 3, step 3, which was committed before the own capture was scored (commit `6a04490`). Its
> shortfall is 1.00, and its median error / tolerance (6.7) is the largest of the eight rows at 1.00. Caveat: the
> committed snippet computes that tie-break for wall gates only. Applied to every row, as the text reads, it would put
> the photo repeat room (40.4) and the video ceilings (20.7) first. I follow the snippet; `fixloop_ranking.md` says
> why.
>
> **2. Root-cause hypothesis and the evidence for it.**
> Hypothesis: the scale step cannot follow DPVO's scale on this walk. Its depth-agreement votes are too few to see
> DPVO's scale change, so stretches of the walk get a scale that is far off.
> Evidence:
> - Part 3, step 4 on the before runs: 7 of the 9 found walls are short in both runs (78%), with a median signed error
>   of −19.7% (r1) and −17.7% (r2). The rule calls that scale
>   (`outputs/fixloop/before/eval/step4_scale_check.json`).
> - Only 31 (r1) and 25 (r2) of 902 keyframe pairs vote on scale. r1 cuts a segment at kf 51 (t = 34.8 s). Its first
>   segment (kf 0–50, the hall's first pass) has 1 vote and a 268 m path, fails the self-check and is dropped.
> - The 23:28 run of the same video (`outputs/own_house/diag/video/workflow_result.json`): DPVO's scale drops about
>   13× at t = 34.5 s. 30 pairs vote, none between t 12 and 37 s, so the drop is not seen. The ±1.5× clamp hides it:
>   the local scale 24.742–55.670 is exactly 37.114 / 1.5 to 37.114 × 1.5, with 133 of 228 keyframes on the clamp.
> - In that run, PnP on MoGe-2 depth over 177 keyframe pairs gives the scale each stretch needs. The run's scale is
>   1.1× to 14.7× too large between t 8 and 101 s. kf1 and kf42 are 0.50 m apart; the scene puts them 9.96 m apart.
> - One global scale does not repair it: dividing out the median scale leaves 2 of 14 (r1) and 1 of 14 (r2) within 3%.
>
> Rejected alternatives, measured on the 23:28 run (the last four are replays with one step changed):
> - Depth model: MoGe-2 sizes inside one frame are within 6% of the tape (five sizes, −5.8% to +0.9%).
> - Plan step: on the PnP-scaled scene the same plan step gives 3 rooms, not 8.
> - D-066 gate back to 0.005: the hall's first pass fails the self-check and is dropped (6 rooms, 31.7 m²).
> - No ±1.5× clamp: 3 rooms but 16.5 m², with the camera at a median 3.8 m above the floor.
> - PnP votes plus a segment cut at the jump: the spread test drops kf 53–227, which leaves 1 room.
>
> **3. The fix and the predicted number.**
> Fix: scale votes from PnP on MoGe-2 depth. It is one change for one root cause, the scale signal.
> - `floorplan/video/scale.py`: for keyframe pairs 2, 4 and 6 apart, SIFT matches, the MoGe-2 depth of the first
>   frame, `solvePnPRansac` and an LM refine. A pair votes when its metric step is at least 8 cm and it agrees with
>   DPVO within 35° in direction and 4° in rotation. The vote is metric step / DPVO step. In `estimate_scales` the
>   segment cut stays. The local scale and its ±1.5× clamp become a running median of the votes within ±8 keyframes
>   (at least 3 votes, log-interpolated in between), and the bootstrap resamples the votes.
> - `floorplan/video/frontend.py`, step 8: the keyframe images go to both `estimate_scales` calls.
> - `floorplan/video/params.py`: the four thresholds.
> - Unchanged: segment cuts, DPVO re-runs, the self-check, the pose graph and the plan step.
>
> Predicted after the fix: **0 of 14 walls within 3%, range 0–2. The gate still fails.**
> How: I replayed the 23:28 run from its cached DPVO, MoGe-2, GeoCalib and SfM arrays, with only the scale step
> swapped for the 177 PnP votes the fix computes. Four more replays resample the votes. All five were scored with the
> scorer at `4f9d0e0` (`outputs/fixloop/before/predict/summary.json`). The step 4 counterfactual above agrees:
> 1–2 of 14.
> What else should change. These are replay numbers, the 23:28 run as scored now → the fix, with the range over 5:
> - Walls within 10%: 2 → 6 of 14 (3–6). Found: 8 → 13 of 14 (all 5). Median error of the found walls:
>   23.0% → 10.5% (10.5–40.0%).
> - Rooms: 8 → 3 (3–4). Footprint: 35.4 → 29.5 m² (29.5–31.4); the house is about 29–31 m².
> - 95% interval coverage: 0.75 → 0.82 (0.65–0.82). kf1/kf42: 9.96 → 0.47 m (truly 0.50 m).
> - Not fixed: the ceilings stay 0 of 3 in all 5 (bedroom 3.54–3.99 m against 2.629 m; tracking is lost while the
>   ceiling is filmed at the end). Door widths stay 0 of 3–4. The self-check still fails (spread 7.9 > 2.0), so the
>   plan keeps its low-reliability flag.
> - Photo and LiDAR plans: no change expected (bit-identical); only the video front end uses `scale.py`. The three
>   sample videos must not get worse.
> - The two before runs disagree (I-007), and the after runs will too. Expected verdict: meaningful movement short of
>   the gate.
>
> **4. Regeneration.** Before = tag `before-fix`, after = tag `after-fix`. Each state runs the video twice (I-007):
>
> ```bash
> S=before   # S=after at tag after-fix
> for r in r1 r2; do
>   flock outputs/.gpu.lock env -u PYTHONPATH python scripts/run_capture.py outputs/own_house/capture/video/take1.mp4 \
>       --tier video --no-damage --out outputs/fixloop/$S/video_take1_$r
> done
> env -u PYTHONPATH python scripts/eval_own_capture.py --gt outputs/own_house/capture/gt/ground_truth.csv \
>     --gt-json outputs/own_house/capture/gt/gt_polygons.json \
>     --plan photo_lit=outputs/fixes/photo_lit/plan.json --plan photo_dim=outputs/fixes/photo_dim/plan.json \
>     --plan video_take1_r1=outputs/fixloop/$S/video_take1_r1/plan.json \
>     --plan video_take1_r2=outputs/fixloop/$S/video_take1_r2/plan.json --out outputs/fixloop/$S/eval
> git diff before-fix after-fix -- floorplan scripts/run_capture.py > docs/fixloop.diff
> ```

After the fix, appended under the declaration and never edited into it.

### After the fix: runs, replays and verdict (5 Oct, 02:59–04:47 IST)

**What ran.**
- Code `fbad1b3` ("video: scale votes from pnp on moge depth"), tag `after-fix`, clean tree
  (`outputs/fixloop/after/CODE_COMMIT.txt`).
- Item 4 with `S=after` (`outputs/fixloop/after/run_after.sh`): take1 r1 at 02:59–03:09 and r2 at 03:10–03:21, one
  after the other on the GPU, then the scoring.
- Same scorer and tape GT as the before runs. `scripts/eval_own_capture.py` and `floorplan/benchmark/` have no change
  since `4f9d0e0`, and the two photo plans score exactly as before.
- `outputs/fixloop/after/summarize.py` reads the four video runs (`summary.json`). All four have the same 228
  keyframes, so kf1 / kf42 is the same pair of frames in each.

**How walls are counted.** The scorer writes no rows for a tape room that has no plan room. After r1 has no kitchen
and after r2 has one room, so the scorer's own fraction is over 11 and 6 walls. The gate counts all 14 tape walls, as
declared. A wall of a missing room is a failure.

| | Before (`before-fix`), r1 / r2 | Predicted | After (`after-fix`), r1 / r2 |
|---|---|---|---|
| **Gate: walls within ±3%** | **0 / 0 of 14** | **0 of 14 (0–2)** | **3 / 0 of 14** |
| Walls within 10% | 1 / 3 of 14 | 6 of 14 (3–6) | 4 / 1 of 14 |
| Walls found | 9 / 9 of 14 | 13 of 14 | 10 / 3 of 14 |
| Median error of the found walls | 20.1% / 28.6% | 10.5% (10.5–40.0%) | 15.9% / 34.9% |
| Rooms | 4 / 4 | 3 (3–4) | 2 / 1 |
| Footprint (the house is about 29–31 m²) | 22.9 / 37.8 m² | 29.5 m² (29.5–31.4) | 28.8 / 6.4 m² |
| 95% interval coverage | 9 / 6 of 14 | 0.82 (0.65–0.82) | 11 of 14 / 1 of 3 |
| Ceilings within 1.5 cm | 0 / 0 of 3 | 0 of 3 | 0 / 0 of 3 |
| Door widths within 2 cm | 0 of 3 / 1 of 4 | 0 of 3–4 | 1 of 4 / 0 of 2 |
| kf1 to kf42 in the scene (0.50 m by PnP) | 127.0 m (kf1 left out) / 9.39 m | 0.47 m | 0.70 m / both left out |
| Keyframes in the scene | 177 / 228 of 228 | 228 | 228 / 20 |
| Scale votes | 31 / 25 depth-agreement pairs | 177 PnP votes | 147 / 192 PnP votes |
| Self-check | r1: kf 0–50 fail on residual; r2: pass | fails (spread 7.9 > 2.0) | r1: fails (spread 102.5), all kept; r2: 3 of 4 segments fail on spread |
| Low-reliability flag | yes / no | yes | yes / yes |

Sources: `outputs/fixloop/after/eval/own_eval.md`, `summary.json`, `eval/step4_scale_check.json` and the run logs.
The kf distances "by PnP" come from PnP on MoGe-2 depth (`outputs/own_house/diag/video/revisit_check.json`), the kind
of measurement the fix votes with. They are not tape. The tape checks are the walls and the room areas.

**After r1.**
- The bedroom (R2) is 11.02 m² against 11.09 m² on the tape outline. W1 (+1.7%), W4 (+0.5%) and W6 (−2.1%) are within
  3%; W5 is −5.7%. The short walls W2 (0.126 m) and W3 (0.44 m) are off by +3.3 and −9.0 cm. Door D2 is 0.731 m against
  0.737 m.
- R1 (17.82 m²) is the hall, the kitchen and the passage in one room: the camera is inside R1 for all of t 0–76 s.
  It is paired with the hall (12.71 m² on the tape outline): W1 −82.7%, W2 −49.9%, W5 −61.3%, W4 +11.4%, W3 missed.
- Places filmed twice now sit where they should (true distance by PnP / in the scene): kf1–kf42 0.50 / 0.70 m,
  kf68–kf129 0.44 / 0.58, kf69–kf124 0.64 / 0.60, kf156–kf200 0.32 / 0.29, kf156–kf225 0.31 / 0.51,
  kf158–kf205 0.06 / 0.60. The pose graph accepts 6 loops (3 revisits); each before run accepted 1.
- The local scale runs from 0.27 to 47.8 inside one segment (174×). That is DPVO's own drift on this run, now
  followed. The self-check fails on it (spread 102.5 > 2.0). No segment passes, so the front end keeps them all.
- Ceilings: bedroom 1.918 m (−71.1 cm), hall 2.924 m (+29.5 cm), kitchen none.
- Part 3, step 4: 6 short and 4 long of 10 found walls (60% one sign), median signed −3.9%. The rule no longer calls
  it scale.

**After r2.**
- DPVO's first run was cut at kf 51, 71 and 127, and DPVO ran again from frame 2215. Only kf 51–70 passes the
  self-check (spread 1.05). The other three segments fail on spread (11.16, 2.14, 10.37 > 2.0), so 208 of 228
  keyframes are left out.
- The one room (6.35 m²) is built from kf 51–70 (t 34.8–46.2 s), the walk from the hall into the kitchen. The scorer
  pairs it with the bedroom by its label "room", so its 3 "found" walls are not bedroom walls.

**The same DPVO run with the old and the new scale step.** The two after runs differ in their DPVO runs, as the before
runs did (I-007). To separate the fix from that, I replayed each after run on its own cached DPVO (first and fresh
runs), MoGe-2, GeoCalib and SfM arrays, once with the before-fix code and once with the after-fix code. CPU only, one
job at a time (`outputs/fixloop/after/replay.py`, `run_replays.sh`, results in `replay/`).
- Check: the after-fix replays are close to the real runs but not bit-identical. The cache holds the depth as float16,
  the live run used float32. They give the same segments and the same self-check verdicts. r1: 2 rooms, 29.24 m² (real
  28.84), kf1–kf42 0.68 m (real 0.70), but 1 wall within 3%, not 3. r2: 1 room, 6.27 m² (real 6.35).

| take1 (`replay/eval/own_eval.md`) | r1's DPVO run: old → new | r2's DPVO run: old → new |
|---|---|---|
| Walls within 3% | 1 → 1 of 14 | 1 → 0 of 14 |
| Walls within 10% / found | 1 / 6 → 3 / 8 | 1 / 8 → 1 / 3 |
| Rooms, footprint | 2, 37.79 m² → 2, 29.24 m² | 2, 18.70 m² → 1, 6.27 m² |
| Keyframes in the scene | 228 → 228 | 121 → 20 |
| kf1–kf42 (0.50 m by PnP) | 10.47 → 0.68 m | left out in both |
| kf68–kf129 (0.44 m by PnP) | 0.09 → 0.58 m | 26.62 m → left out |

- On r1's DPVO run the fix puts the walk together: the footprint goes from 37.8 to 29.2 m², and kf1–kf42 from 10.47 to
  0.68 m. The gate does not move: 1 → 1 of 14.
- On r2's DPVO run the fix loses most of the plan. With the old step, segments 1 and 3 pass the spread test (1.00 and
  1.50; the old clamp kept the spread at 2.25 or less), and 121 keyframes stay. With the new step their spreads are
  1.05 and 10.60, so 20 keyframes stay.
- So the 3 of 14 of the real r1 is within the noise: the same code on the same DPVO run, with float16 depth, gives 1.

**Side effects.**
- Photo and LiDAR plans cannot change. The diff touches `floorplan/video/{scale,frontend,params}.py` and adds a test.
  `floorplan/video/scale.py` is imported only by `floorplan/video/frontend.py` and `tests/test_scale_votes.py` (grep
  over the repo, `outputs/` left out). The photo tier's `floorplan/photo/frontend.py` imports `floorplan/photo/scale.py`,
  a different module. Importing the LiDAR and photo entry points loads no `floorplan.video` module. The two photo plans
  score exactly as before.
- The recruiters' sample videos, video tier only (never the photo tier): one after run each, 03:22–04:07, in
  `outputs/fixloop/after/sample_<capture>`. They ran with `--no-damage`; damage only adds annotations to a finished
  plan. Each plan is scored against the LiDAR plan of the same capture with `scripts/bench_tier_ref.py`
  (`compare_samples.py`, `samples.json`; the replays in `replay/samples_replay_vs_lidar.json`). "Sides" are the two
  box dimensions of each matched room.

| Sample (LiDAR plan) | Before | After, one run | Same DPVO run as the after run (with_ceiling: a re-run, see below): old → new scale step |
|---|---|---|---|
| single_room (3 rooms, 17.61 m²) | d069 r1–r3: 3 rooms, −1.6 / −4.5 / −2.7%; sides within 3%: 2, 3, 2 of 4 | 3 rooms, 22.07 m² (+25.3%); sides 0 of 6 | 3 rooms each, +19.9% → +29.3%; sides 0 of 4 → 2 of 6 |
| floor_only (8 rooms, 61.90 m²) | d066: 5 rooms, 30.11 m² (−51.4%); sides 0 of 8 | 0 rooms | 0 rooms → 1 room, 7.62 m² |
| with_ceiling (8 rooms, 62.50 m²) | final (4 Oct 05:55, older code): 5 rooms, 19.40 m² (−69.0%); sides 2 of 6 | 1 room, 5.20 m² (−91.7%) | 3 rooms, 54.28 m² → 2 rooms, 54.45 m² |

- All three after runs are worse than the before runs. On the same DPVO runs the old scale step is also far off, so
  the drop comes mainly from these DPVO runs (I-007), not from the fix.
  - single_room: both steps give a plan 20–29% too large. In the after run the closet is 5.85 m² against 1.93 m² on
    LiDAR (the I-014 symptom); the two main rooms are within 2.5% in area.
  - floor_only: 4 of 6 segments pass the self-check. Their joins are not verified, and the first four segments end up
    with the camera 4.5 m above the floor the plan uses. Both steps give 0 or 1 room. The before run d066 kept only its
    first segment.
  - with_ceiling: on the cached float16 depth one cut moved, so this replay pair ran one fresh DPVO run again (frames
    214–1968, `run_replay_wc.sh`; GPU). Both steps then fail the same two long segments on spread (old 2.01 and 2.25,
    new 3.47 and 10.99) and keep 211 of 836 keyframes. Both plans have one 48–50 m² room. The real after run kept 26.

**Verdict: prediction wrong, and why.**
- The gate fails, as predicted. After: 3 and 0 of 14 within 3%, against 14 needed. The rule's number (the better run)
  goes from 0 to 3 of 14, a shortfall of 1.00 → 0.79. 3 is above the predicted range (0–2); r2's 0 is inside it. But on
  the same DPVO runs the old and the new scale step give 1 and 1 (r1) and 1 and 0 (r2). So the gate did not move beyond
  run-to-run noise.
- The plan did not come out as predicted: 2 and 1 rooms against 3 (3–4); 10 and 3 walls found against 13; 28.8 and
  6.4 m² against 29.5 (29.5–31.4).
- Why the prediction missed: it came from replays of one run (23:28) with one segment. That segment failed the
  self-check, and when no segment passes, the front end keeps everything. The fix left the spread test at 2.0. That
  limit was set when the clamp kept the spread at 2.25 or less. The PnP scale follows DPVO's real drift, with spreads
  of 2 to 107 on these runs. Once one segment passes, the drifting ones are dropped: on r2's DPVO run 20 keyframes stay
  instead of 121. I did not replay a run with several segments before declaring.
- What the fix does, measured on the same DPVO run: the walk holds together. On r1's run kf1–kf42 goes from 10.47 to
  0.68 m (0.50 m by PnP), and the footprint from 37.8 to 29.2 m² (the house is about 29–31 m²).
- Not done here: a self-check that fits the PnP scale, for example the votes' scatter around the running median
  instead of the spread of the scale itself. The after runs would then have to be run again.

- Readable diff: `git diff before-fix after-fix -- floorplan/ tests/`, saved as `docs/fixloop.diff`.
  `scripts/run_capture.py`, named in item 4, has no change.

### Follow-up after the loop (5 Oct, from 05:20 IST)

This is not part of the loop. The declaration, the after runs and the verdict above stay as they are. The question:
should the shipped video tier use the PnP scale (D-075) or the old depth-agreement scale? Notes and scripts are in
`outputs/fixloop/followup/`.

**Why before r2's DPVO run collapses with the PnP scale** (no segment dropped, 1 room, camera 3.25 m above the plan
floor; `cause_b.py`, `probe.py`):
- The camera path drops 2.2 m. 1.47 m of it is two keyframe steps, kf 135 to 137 (t 78-80 s).
- DPVO's step kf 136 to 137 points mostly down (0.0259 units down, 0.0078 across). The local scale there is 44.2, so
  it becomes 1.14 m. With the old scale (5.44) it is 0.14 m.
- PnP solves no pair between kf 131 and 137. The one pair over the step, kf 136 to 138, says 0.95 m with 0.30 m down;
  DPVO says 1.48 m with 1.23 m down. They differ by 38° in direction and 16.5° in rotation. So DPVO glitches where
  PnP finds no matches.
- Each keyframe's own depth puts its camera 1.33 m (median) above the floor it sees. The floor's height in the scene
  is about -1.4 m before kf 137 and about -3.2 m after it.
- The pose graph anchors only floors within 0.15 m of the lowest ones (3 fragments), so it does not level this. The
  plan takes the lowest floor, and the hall and bedroom, filmed before t 80 s, sit 2 m above it.

**Follow-up.** For the PnP scale only: (1) the self-check judges a segment by its vote coverage and vote residual,
not by the spread of its scale; (2) before the pose graph, the camera path is moved vertically so that every
keyframe puts the floor it sees at one height. `scale_method = "pnp" | "depth_agreement"` selects the scale step;
"depth_agreement" is the code before D-075.

**Decision rule, written before any follow-up replay was run** (`outputs/fixloop/followup/NOTES.md`):
- Arms on each cached DPVO run: old = "depth_agreement"; new = "pnp" with the follow-up. Runs: take1 before r1, r2,
  after r1, r2 and 23:28; single_room d069 r1-r3 and the after run; floor_only and with_ceiling (after runs).
- Errors per run. Room count: own, distance from 3-4 rooms; samples, |rooms − LiDAR rooms|. Footprint:
  |footprint / reference − 1|, reference 30 m² (own) or the LiDAR plan.
- New is worse on a run if it has a larger room-count error, or a footprint error more than 5 points larger. It is
  better if it is not worse and has a smaller room-count error or a footprint error more than 5 points smaller.
- "pnp" becomes the default only if (A) it is not worse on more than half of all runs, and (B) on the 5 own runs it
  has a lower median footprint error and a lower total room-count error. Otherwise "depth_agreement" is the default
  again and "pnp" stays behind the parameter.

**Result** (replays 06:13–07:28 on each run's own cache, CPU, one at a time; `outputs/fixloop/followup/summary.json`;
the full table is in `DECISIONS.md` D-076). Old → new:
- take1, before r1 / before r2 / after r1 / after r2 / 23:28. Footprint 24.1 / 40.3 / 37.8 / 18.7 / 37.6 m² →
  29.2 / 31.3 / 27.4 / 30.8 / 29.8 m² (the house is about 29–31 m²). Rooms 4 / 4 / 2 / 2 / 9 → 2 / 3 / 2 / 2 / 3.
  Walls within 3%: 2 / 0 / 1 / 1 / 0 → 1 / 3 / 0 / 2 / 1 of 14.
- Before r2's run, which had 1 room with D-075, now has 3 rooms: the floor levelling moves the path up to 2.31 m.
- Samples, footprint against the LiDAR plan: single_room d069 r1 / r2 / r3 +1 / −3 / −5% → +21 / +28 / −16%, its
  after run +20 → +8%; floor_only's after run 0 → 3 rooms (−63%), d066 −49 → −88% (5 → 1 room); with_ceiling's
  after re-run −13 → −66% (3 → 2 rooms).
- Rule: (B) holds, take1's median footprint error 26.0% → 2.7% and room-count error 7 → 3. (A) fails: not worse on
  6 of 12 runs. **"depth_agreement" is the default again; "pnp" is opt-in** (`--video-scale pnp`, commit `b0b4a34`).
  d066 is not named in the rule above but `NOTES.md` counts it when its cache replays; without it, 6 of 11 would pass.
- On the sample videos SIFT finds few features on white walls (single_room: 22 votes over 164 keyframes), so the PnP
  scale is interpolated there. On my own take1 it is the better scale.

---

## Part 2. Candidates, with their evidence (as of the 06:28 final benchmark)

Context: how far each failing gate on the **sample** benchmark is from its threshold
(`outputs/benchmark/final/REPORT.md` §1, generated 06:28). Video and photo rows are scored against the LiDAR plan of the
same capture, not against ground truth. "Relative shortfall" = (needed − measured) / needed, the ranking rule of
Part 3.

| Tier | Gate (threshold) | Measured | Relative shortfall |
|---|---|---|---|
| Photo | room dimensions ±8%, single_room / floor_only / with_ceiling | 0/6, 3/12, 0/14 | 1.00 / 0.75 / 1.00 |
| LiDAR | repeatability, every wall within max(1 cm, 0.5%) | 15/87 = 17.2% | 0.83 |
| LiDAR | opening widths ≤ 2 cm on ≥ 85% (repeat-pair proxy) | 2/12 slots | 0.80 |
| Video | room dimensions ±3%, single_room / floor_only / with_ceiling | 2/6, 3/10, 2/6 | 0.67 / 0.70 / 0.67 |
| Photo | 95% interval coverage | 100% (at ±79%) / 58% / 77% | 0 / 0.39 / 0.19 |
| Video | 95% interval coverage | 70% / 88% / 60% | 0.26 / 0.07 / 0.37 |
| LiDAR | ceiling ≤ 1.5 cm per room (ARKitScenes, laser) | 3/4 rooms, max 2.16 cm | 0.25 |

The own capture replaces the photo and video rows with tape ground truth. It has no LiDAR, because my
phone has none (`OPEN_QUESTIONS.md`). So the LiDAR rows stay as they are.

### C1. LiDAR bias: the completed rehearsal (D-021, D-027). Template already filled, not the declared loop

- **Gate and failing number.** Ceiling ≤ 1.5 cm against a Faro laser: a ceiling at −1.92 cm; room boxes −1.37 cm on
  average (`modules/lidar_bias.md` §5.6).
- **Root cause, with evidence.** Five hypotheses were tested. Range-dependent scale, trajectory scale, registration
  and the laser were all rejected; our estimator was a minor contributor. Apple's LiDAR surfaces sit 0.68 cm into the
  room (D-021).
- **Pre-registered prediction.** Fixed on 3 development rooms before 2 hold-out rooms were analysed:
  0.0 ± 0.7 cm. **Measured: −1.25 → +0.11 cm**; ceilings 5/5 within 0.7 cm.
- **What it missed.** The spread after the fix was 1.05 cm, not the 0.7 cm predicted. The plan-level re-test then
  found the reflex-corner error (D-027): corner-aware shifts take walls within 1.5 cm from 82% to 94%.
- **Final-code state.** ARKitScenes, final run: room dimensions 8/8 within 1.5 cm (median 0.43 cm); wall lengths
  median 0.86 cm, 84% within 1.5 cm (n 19) (`REPORT.md` §1).
- **Role.** This shows the method works. It is not the declared loop: it was not chosen as the worst gate of the
  final benchmark.

### C2. I-004: drift correction tilts the ceiling of a capture with no revisit loops (LiDAR, laser GT)

- **Gate and failing number.** Ceiling ≤ 1.5 cm per room: **3/4 rooms**. Room 47895909 is at **−2.16 cm**
  (corrected, drift ON). The other three are −0.82, +0.23 and +0.04 cm
  (`outputs/benchmark/final/arkitscenes/items.csv`, ceiling_height rows; identical to the 03:46 run).
- **Root cause, with evidence.**
  - In 47895909 the ceiling's spread across fragments grows from **0.06 cm (ARKit) to 0.55 cm (corrected)**
    (`final/arkitscenes/47895909/build_report.json` → `drift.anchors.{arkit,corrected}.ceiling_mad_sigma_cm`).
  - The room has **0 revisit loops** accepted (1 local loop), **0 Manhattan anchors and 8 floor anchors** (same
    file, `drift.loops`, `drift.anchors`). So floor anchors are the only vertical constraint in the solve. They
    level each fragment's floor and tilt the ceiling, which was consistent before (I-004).
  - With drift OFF the same room read −0.99 cm, and all 4 rooms passed (`modules/benchmark_arkitscenes_accuracy.md`
    §7; earlier code state, so the number must be re-measured at `before-fix`).
- **Fix.** I-004 option (a). In `floorplan/recon/drift.py` (`run_graph`), when no revisit loop is accepted after the
  loop rounds, re-solve without the floor (vertical) anchor terms. Yaw and position corrections stay.
- **Predicted.**
  - 47895909 ceiling: −2.16 → about −1.0 cm (−1.3 to −0.7), so the gate goes **3/4 → 4/4**. Its ceiling spread:
    0.55 → ≤ 0.1 cm.
  - The other ARKitScenes rooms are unchanged: they have 1, 13 and 9 revisit loops (their `build_report.json`).
  - The three LiDAR sample plans stay **bit-identical**: they have 1 / 10 / 25 revisit loops
    (`final/lidar/<capture>/drift_on/run_report.json` → `front_end.drift.loops.revisit_accepted`).
- **Before/after.** Before: `env -u PYTHONPATH python scripts/bench_final_arkit.py`, then copy
  `outputs/benchmark/final/arkitscenes/` to `outputs/fixloop/before/arkitscenes/`. After: the same at `after-fix`.
  Side-effect check: `scripts/bench_final_runs.sh lidar`, then compare the `plan.json` hashes. About 10 min in total.
- **Why it is strong.** One gate, one root cause measured in the drift report, laser ground truth, a prediction with
  a range, and a side-effect check that can fail.
- **Caveats.** n = 4 rooms. And it is a LiDAR gate, while the own capture cannot test LiDAR.

### C3. D-034 / BF-13: video intervals still miss when rooms are missing (video, calibration)

- **Gate and failing number.** Calibration at every tier. Video 95% coverage is 70% / 88% / 60% (single_room /
  floor_only / with_ceiling, `REPORT.md` §1). The footprint interval misses LiDAR on 2 of 3 long runs:
  - with_ceiling: 19.40 [4.16, 34.64] against 62.50 m²;
  - floor_only, second run: 18.95 [4.07, 33.83] against 61.90 m².
- **Root cause, with evidence.**
  - D-034's 20% floor is a **scale** term. The big errors are **missing rooms**: with_ceiling has 5 rooms for 8
    (3 matched), and the floor_only rerun has 3 for 8 (`modules/benchmark_final.md` §7.4, BF-13).
  - Coverage rose with D-034 (floor_only 31% → 88%, with_ceiling 38% → 60%; §7.2). But the upper bounds sit about
    28 m² below the reference, so no scale sigma can reach it.
- **Fix.** When `meta.reliability` is low, widen the footprint's upper bound by an area the run measures itself. For
  example, the floor area swept by the camera path that lies outside every room. Or report the footprint as a lower
  bound.
- **Predicted.** Footprint coverage on long videos 1/3 → 3/3; room-dimension coverage unchanged.
- **Caveats.**
  - The area term is not designed yet.
  - The video tier is **not repeatable** (I-007, BF-12): floor_only gave 5 rooms / 40.91 m² and then
    3 rooms / 18.95 m² on the same command (`REPORT.md` §2). So any video before/after must run each state at least
    twice and report every run.
  - It is a calibration fix, not an accuracy fix.

### C4. Photo room sizes (photo, wall lengths ±8%)

- **Gate and failing number.** Photo room dimensions within ±8%: 0/6, 3/12 and 0/14, median |error| 9.5%, 15.3% and
  24.3%. Footprint +16.5% / −41.0% / −26.9% (`REPORT.md` §1, photo v3, simulated folders against LiDAR).
- **Root causes, with evidence.**
  - Simulated "spins" are walk-through frames covering 85–200°, so whole walls are never seen (PT-16).
  - MoGe-2 depth is short on these frames: depth ratio median 0.905 (photo_tier v2, quoted in PT-16).
  - L-shaped rooms are fitted as one rectangle (T-I11).
  - Hallways over-reach through their doors: +257% area (T-I10).
  - With true poses and a full turn, still only **5/16** dimensions are within 8% (median 15.3%; `photo_tier.md` v3.4).
- **Fix candidates, by root cause.**
  - Common scale: if the errors share a sign, fix the depth scale (no reference object, D-067).
  - L-shape: a two-rectangle fit where a side has two strong peaks ≥ 1 m apart (v3.6 item 2; the rival ratio is
    already stored).
  - Corridor: a folder with ≥ 2 doorway pairs and narrow free space gets the corridor rule (v3.6 item 6).
- **Predicted.** Computed on the before run (Part 3, step 4): the pass fraction after removing the common scale, or
  after excluding the L-shaped or corridor rooms, is the prediction for that fix.
- **Why it is likely the morning pick.** The own capture scores photos against the tape. Photo sizes are the
  furthest gate from threshold on the sample. The real spins are what PT-16 could not test.

### Not a candidate this morning: I-007 (video run-to-run)

`ISSUES.md` lists I-007 as mitigated. The 06:28 runs contradict that: same command, 5 rooms / 40.91 m² against
3 rooms / 18.95 m² (`REPORT.md` §2). There are two independent causes (`modules/benchmark_final.md` BF-12):

1. pycolmap's two-view match verification differs between runs although features and matches are bit-identical.
   The self-calibrated focal comes out at 1564.6 against 1566.0 px.
2. DPVO differs from frame 1 even with identical inputs and cuDNN deterministic: the arbitrary scale differs 2×.

Two root causes are not one fix, and the second sits inside DPVO's CUDA kernels. It stays a disclosed limitation.
Its consequence for the fix loop: **any video before/after must be run at least twice per state.**

### Already measured: option A (B-1, repeatability)

The judge predicted "about 15% of walls, 7/7 rooms" before the code existed. B-1 measured 7/99 → 14/95 = 14.7% and
6 → 8/8 rooms (`modules/plan_v2_ablation.md` row 1). The gate is still
failed: the final run gives 15/87 = 17.2% (`REPORT.md` §1). Use it only if the morning loop cannot be shipped. If so,
say that the prediction was the judge's.

---

## Part 3. Morning procedure (own capture → worst gate → declare → fix → before/after)

**0. Before scoring anything.** Commit this file unchanged. Done in commit `6a04490` ("fix loop: write the ranking
rule before the own capture is scored"). That commit came with the replayed history at 00:03 on 5 Oct, after the first
scoring at 23:38 on 4 Oct, so git alone does not prove the rule came first (see the correction at the top).

**1. Process and score the own capture** (`OWN_CAPTURE_RUNBOOK.md`):

```bash
O=../TakeHome/OwnCaptures
env -u PYTHONPATH python scripts/process_own_capture.py $O --out outputs/own     # photo, photo repeat, video; tape scoring
env -u PYTHONPATH python scripts/bench_final.py --skip-lidar                     # REPORT.md §2 (photo repeat), §3, §6
```

**2. Read** `outputs/own/eval/own_eval.md` (per-run gates and rows), then `outputs/benchmark/final/REPORT.md` §2 for
the photo-repeat row.

**3. Rank the gates by distance from their threshold** (this rule is fixed in advance):

- Relative shortfall = (needed − measured) / needed.
  - Wall gates need all walls within tolerance (needed 1.0).
  - Openings need 0.85.
  - Ceilings need 1.0.
  - Calibration needs 0.95 coverage (over-coverage is not a shortfall).
  - Head-to-head needs 0.70.
  - The photo repeat room needs 1.0 of its walls.
- Ties are broken by median |error| / tolerance.
- The worst gate is the top row.

```bash
env -u PYTHONPATH python - outputs/own/eval/own_eval.json <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
REQ = {"wall_gate": 1.0, "opening_gate": 0.85, "ceiling_gate": 1.0}   # floorplan/benchmark/gt_eval.py gate_summary
rows = []
for run, res in d.items():
    if run == "head_to_head":
        rows.append((run, "beat_or_tie >= 0.70", res["beat_or_tie_frac"], 0.70, None)); continue
    g = res.get("gates", {})
    for k, req in REQ.items():
        if k in g:
            x = g.get("wall_median_rel_err", 0) / g[k]["tol"] if k == "wall_gate" else None
            rows.append((run, k, g[k]["pass_frac"], req, x))
    if "calibration" in g:
        rows.append((run, "calibration (95% coverage)", g["calibration"]["coverage_95"], 0.95, None))
rows = [(r, k, m, q, max(0.0, (q - m) / q), x) for r, k, m, q, x in rows]
rows.sort(key=lambda t: (-t[4], -(t[5] or 0)))
for r, k, m, q, s, x in rows:
    print(f"{r:22s} {k:28s} measured {m:5.2f} needed {q:4.2f} shortfall {s:4.2f}" + ("" if x is None else f"  err/tol {x:.2f}"))
EOF
```

(Tested on the dry-run `outputs/own_test/out/eval/own_eval.json`. Add the photo-repeat row from REPORT §2 by hand.)

Rules for edge cases, written down now so they cannot be bent later:

- **A gate a tier does not measure by design.** Example: photo door widths are `not_observed`, because doorway pairs
  give position, not width (plan_beta v4.10, step 6'd). It still ranks; missed openings count as misses. If it comes
  out on top, the declaration names it, and the fix is to measure it.
- **No root-cause evidence within 30 minutes.** Write down what was tried, then move to the next row. The
  declaration says so.
- **Video gates.** Because of I-007, the "before" is two runs, and the failing number is the better of the two.

**4. Root cause and prediction, computed on the before run.**

- Wall gates: compute the signed errors from the `own_eval.md` rows. If ≥ 75% share a sign and the median signed
  error is more than half the tolerance, the hypothesis is **scale**. The prediction is the pass fraction after
  dividing out the median scale (the counterfactual).
- Otherwise, list the failing walls by room. If the failures cluster in L-shaped rooms, corridors or rooms seen
  through doors (C4), the prediction is the pass fraction with that cluster fixed to its median-room error.
- Calibration: the prediction is the coverage after the proposed widening, applied to the before run's intervals.
- Write the declaration (Part 1, items 1–3), then commit it on its own, before any fix code:
  `fix loop: declare <gate> = <number>, root cause <...>, predicted <number>`.

**5. Tag "before" and run it.**

```bash
git tag -a before-fix -m "fix loop: before (declared in <declaration commit>)"
# photo-tier fix (the photo folders were already gathered by process_own_capture.py)
env -u PYTHONPATH python scripts/run_capture.py outputs/own/_photo_inputs_main --tier photo --out outputs/fixloop/before/photo
env -u PYTHONPATH python scripts/eval_own_capture.py --gt $O/gt/ground_truth.csv \
    --plan photo=outputs/fixloop/before/photo/plan.json --out outputs/fixloop/before/eval
# video-tier fix: the same with  $O/video/take1.mp4 --tier video  -> before/video_take1_run1, _run2 (I-007)
# LiDAR fix (C2): scripts/bench_final_arkit.py, then cp -r outputs/benchmark/final/arkitscenes outputs/fixloop/before/
```

Run the photo "before" twice and check that the two `plan.json` files are identical. Photo determinism has not been
tested, and the photo tier uses the same pycolmap verification as the video SfM (BF-12a). If they differ, report both
runs.

**6. Fix, tag "after", run the same commands.**

```bash
# one commit per root cause: <module>: <the declared fix>
git tag -a after-fix -m "Fix loop: after"
# the same commands as step 5, with --out outputs/fixloop/after/...
git diff before-fix after-fix -- floorplan scripts/run_capture.py > docs/fixloop.diff   # the readable diff
```

Anyone can regenerate both runs with `git worktree add ../before before-fix` and `git worktree add ../after after-fix`,
then the step 5 commands in each.

**7. Report.** Fill in the before / predicted / after table in Part 1. Then update `TECHNICAL_REPORT.md` §6 and
`COMPLIANCE.md` row 4, and commit them: before vs after, predicted vs measured, and why it fell short (if it did).
Then refresh the report: `bench_final.py --skip-lidar`, or the full run if the fix touched the LiDAR path.

**Time budget.** Photo run plus scoring is about 1–2 min per state (`REPORT.md` §4: v3 photo 9–45 s on the sample,
warm cache). A video run is 272–1737 s per state on the sample. Two runs per state make a video fix the most
expensive option.
