# Fix loop (Part 4): declaration, shipped fix, before/after

Status, 4 Oct 06:45 IST. **No fix is declared yet.** Part 1 is the one-page declaration template. It gets filled in
after the own capture has been scored, and is committed **before** any fix code (commit J1 in `COMMIT_PLAN.md`).
Part 2 lists the strongest current candidates with their evidence. Part 3 is the exact morning procedure. Every
number here is quoted from the file named next to it. Numbers in `<angle brackets>` do not exist yet.

---

## Part 1. Fix declaration (one page; fill in, commit, then fix)

> **1. Worst gate and its failing number.**
> Gate: `<gate>` at tier `<tier>`, on `<capture>`. Measured: `<number>` against the threshold `<threshold>`
> (`outputs/own/eval/own_eval.md`, run at `<time>`). It is the worst gate under the ranking rule of Part 3, step 3,
> which was committed before the own capture was scored (`<commit id of J0>`).
>
> **2. Root-cause hypothesis and the evidence for it.**
> Hypothesis: `<one sentence>`. Evidence: `<the measurement that shows it; file and row>`. Rejected alternatives:
> `<each with the number that rejects it>`.
>
> **3. The fix and the predicted number.**
> Fix: `<one change, one root cause; the files it touches>`. Predicted after the fix: `<gate number with a range>`.
> How the prediction was made: `<counterfactual computed on the before run, e.g. "remove the common scale and
> recount">`. What else is expected to change: `<e.g. "LiDAR sample plans bit-identical">`.
>
> **4. Regeneration.** Before = tag `before-fix`, after = tag `after-fix`. Commands: `<copied from Part 3, step 6>`.

After the fix (commit J4), appended under the declaration and never edited into it:

| | Before (`before-fix`) | Predicted | After (`after-fix`) |
|---|---|---|---|
| `<gate>` | `<number>` | `<number ± range>` | `<number>` |
| Side effects checked | `<e.g. LiDAR plan hashes>` | unchanged | `<result>` |

- Readable diff: `git diff before-fix after-fix -- <paths>` (saved as `docs/fixloop.diff`).
- Verdict: `<pass / meaningful movement short of the gate, and why / prediction wrong, and why>`.

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

The own capture replaces the photo and video rows with tape ground truth. It has no LiDAR, because the candidate's
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
    (3 matched), and the floor_only rerun has 3 for 8 (`modules/benchmark_final.md` §9.4, BF-13).
  - Coverage rose with D-034 (floor_only 31% → 88%, with_ceiling 38% → 60%; §9.2). But the upper bounds sit about
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
6 → 8/8 rooms (`modules/plan_v2_ablation.md` row 1; `COMMIT_PLAN.md` phase F, tags on F50 / F52). The gate is still
failed: the final run gives 15/87 = 17.2% (`REPORT.md` §1). Use it only if the morning loop cannot be shipped. If so,
say that the prediction was the judge's.

---

## Part 3. Morning procedure (own capture → worst gate → declare → fix → before/after)

**0. Before scoring anything.** In the real repo, commit this file unchanged:
`docs(fixloop): ranking rule and candidates, committed before the own capture is scored` (J0). The rule in step 3 is
then provably fixed in advance.

**1. Process and score the own capture** (README §7.1):

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
- Write the declaration (Part 1, items 1–3), then commit it:
  `docs(fixloop): fix declaration - worst gate <gate> = <number>, root cause <...>, predicted <number>` (J1).

**5. Tag "before" and run it.** In the real repo:

```bash
git tag -a before-fix -m "Fix loop: before (declared in <J1 commit>)"
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
# one commit per root cause: fix(<module>): <the declared fix>   (J3..)
git tag -a after-fix -m "Fix loop: after"
# the same commands as step 5, with --out outputs/fixloop/after/...
git diff before-fix after-fix -- floorplan scripts/run_capture.py > docs/fixloop.diff   # the readable diff
```

Anyone can regenerate both runs with `git worktree add ../before before-fix` and `git worktree add ../after after-fix`,
then the step 5 commands in each.

**7. Report.** Fill in the before / predicted / after table in Part 1. Then update `TECHNICAL_REPORT.md` §6 and
`COMPLIANCE.md` row 4, and commit
`docs(fixloop): before vs after, predicted vs measured, why it fell short (if it did)` (J4). Then refresh the
report: `bench_final.py --skip-lidar`, or the full run if the fix touched the LiDAR path.

**Time budget.** Photo run plus scoring is about 1–2 min per state (`REPORT.md` §4: v3 photo 9–45 s on the sample,
warm cache). A video run is 272–1737 s per state on the sample. Two runs per state make a video fix the most
expensive option.
