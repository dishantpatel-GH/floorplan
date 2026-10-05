# Fix loop: the ranking that picks the declared gate

This is the ranking rule of `FIX_LOOP.md` (Part 3, step 3) applied to my own capture, at the code tagged `before-fix`.
The rule was committed in `e042ca5`, before the own capture was scored. I scored the runs on 5 Oct at 02:24 IST.

Correction, 5 Oct (review): `e042ca5` came into git with the replayed history at 00:03 on 5 Oct. The own capture was
first scored before that: the photos at 23:30 and photos plus video at 23:38 on 4 Oct (`outputs/own_house/eval_photo/`,
`eval_all/`). The four pre-fixes below came after that scoring. The rule's text is older than the capture;
`FIX_LOOP.md` (top) gives the evidence, which is a file time, not git.

## What was scored

- **Code:** `7b1a40f`, with a clean tree. It has the four pre-fixes: D-074 (the photo ceiling cut-off), the MoGe CPU
  device, the geometric pairing of repeated room labels (D-072), and `--gt-json` in `process_own_capture.py`. The
  declaration commit adds docs only.
- **Photo, bedroom ("room"):** the lit take and the dim take (take 2). The plans are `outputs/fixes/photo_lit/plan.json`
  and `outputs/fixes/photo_dim/plan.json`. They ran on 5 Oct at 01:46 with the D-074 code, on the cached depth and
  features. The later commits do not change the photo tier on the GPU.
- **Video, `take1.mp4` (117 s):** run twice at `7b1a40f`, one run after the other on the GPU. r1 ran 02:01–02:13 and
  r2 ran 02:14–02:24 (`outputs/fixloop/before/video_take1_r1`, `_r2`).
- **Scorer:** `scripts/eval_own_capture.py --gt-json`, so walls are paired on the tape outlines (D-072). The report is
  `outputs/fixloop/before/eval/own_eval.md`.
- **No head-to-head row:** `app_export/` holds no app file yet.

## How the rule was applied

1. The step 3 snippet ran unchanged on `own_eval.json`. I cut it from `git show e042ca5:docs/FIX_LOOP.md`. Its output
   is in `outputs/fixloop/before/rank_raw.txt`.
2. I did two steps by hand, as the rule says.
   - **Video:** the before is two runs (I-007). Each video gate counts the better run: more items passing, then the
     lower median error / tolerance.
   - **Photo repeat room:** the lit take is take 1 and the dim take is take 2. A tape wall passes when both takes
     measure it and they agree within max(1 cm, 0.5%), as in `scripts/bench_final.py`. The room needs all 6 walls.
3. The sort puts the largest relative shortfall first. Ties go by median error / tolerance, which the snippet computes
   for wall gates only.

`outputs/fixloop/before/rank.py` does steps 1 to 3.

## Ranking

| # | Run | Gate | Measured | Needed | Shortfall | Median error / tolerance |
|---|---|---|---|---|---|---|
| 1 | video take1, better of r1 and r2 | walls within ±3% | 0 of 14 (9 found) | 1.00 | 1.00 | 6.7 |
| 2 | photo, dim take | walls within ±8% | 0 of 6 (2 found) | 1.00 | 1.00 | 2.8 |
| 3 | photo, lit take | door widths within 2 cm | 0 of 2 (2 missed) | 0.85 | 1.00 | – |
| 4 | photo, lit take | ceiling within 1.5 cm | 0 of 1 | 1.00 | 1.00 | – |
| 5 | photo, dim take | door widths within 2 cm | 0 of 2 (2 missed) | 0.85 | 1.00 | – |
| 6 | photo, dim take | ceiling within 1.5 cm | 0 of 1 | 1.00 | 1.00 | – |
| 7 | video take1, better of r1 and r2 | ceilings within 1.5 cm | 0 of 3 | 1.00 | 1.00 | – |
| 8 | photo repeat room, lit vs dim | every wall within max(1 cm, 0.5%) | 0 of 6 (1 measured in both) | 1.00 | 1.00 | – |
| 9 | photo, lit take | walls within ±8% | 1 of 6 (4 found) | 1.00 | 0.83 | 1.1 |
| 10 | video take1, better of r1 and r2 | door widths within 2 cm | 1 of 4 (r2; 1 missed, 1 phantom) | 0.85 | 0.71 | – |
| 11 | video take1, better of r1 and r2 | 95% interval coverage | 9 of 14 (r1) | 0.95 | 0.32 | – |
| 12 | photo, lit take | 95% interval coverage | 4 of 5 | 0.95 | 0.16 | – |
| 13 | photo, dim take | 95% interval coverage | 3 of 3 | 0.95 | 0.00 | – |

**The top row is the video wall gate.** Both runs have 0 of 14 walls within 3%. The better run, r1, finds 9 of 14 walls
with a median error of 20.1%, which is 6.7 times the tolerance. Next comes the dim photo take: 0 of 6 within 8%, with a
median of 22.5%, or 2.8 times the tolerance. Rows 3 to 8 get no tie-break value from the snippet, so their order
among themselves means nothing.

## The two video runs

Same command, same code, same video:

| | r1 | r2 |
|---|---|---|
| Walls within 3% / within 10% / found | 0 / 1 / 9 of 14 | 0 / 3 / 9 of 14 |
| Median error of the found walls | 20.1% | 28.6% |
| Door widths within 2 cm | 0 of 3 | 1 of 4 |
| Ceilings within 1.5 cm (tape 2.629 m) | 0 of 3: +63, +71, +61 cm | 0 of 3: −6 and −56 cm, one not seen |
| 95% interval coverage | 9 of 14 | 6 of 14 |
| Rooms and footprint (the house is about 29–31 m²) | 4 rooms, 22.9 m² | 4 rooms, 37.8 m² |
| Keyframe pairs that voted on scale | 31 of 902 | 25 of 902 |
| Scale segments | 3; the first (kf 0–50, the hall's first pass) had 1 vote, failed the self-check and was dropped | 1, passed the self-check |
| Run time | 747 s | 562 s |

The 23:28 run of 4 Oct (same video, same video-tier code) gave a third result: 8 rooms and 35.4 m² from a single
untrusted segment.

## Photo repeat room

| Tape wall | Tape (m) | Lit (m) | Dim (m) | Difference (m) | Tolerance (m) | Pass |
|---|---|---|---|---|---|---|
| room W5 | 2.997 | 2.739 | 2.134 | 0.605 | 0.015 | no |

Only W5 is measured in both takes. W3, W4 and W6 are found only in the lit take, W1 only in the dim take, and W2 in
neither. So 0 of 6 walls pass.

## Tie-break check: the code and the text differ

The text of step 3 says "ties are broken by median |error| / tolerance". The committed snippet computes that value for
wall gates only (`... if k == "wall_gate" else None`), and every other row gets none. Eight rows tie at shortfall 1.00,
so this decides the top row. If the text is applied to every row where an error can be computed, the order changes:

| # | Row | Shortfall | Median error / tolerance |
|---|---|---|---|
| 1 | photo repeat room: W5 differs by 60.5 cm between the takes (tolerance 1.5 cm) | 1.00 | 40.4 |
| 2 | video ceilings, better run r2: −6 and −56 cm, one not seen | 1.00 | 20.7 |
| 3 | photo ceiling, lit take: −11.5 cm | 1.00 | 7.6 |
| 4 | video walls, better run r1 | 1.00 | 6.7 |

I follow the committed snippet.
- It is the only form of the rule that gives an order without new choices. The text leaves choices open: whether a
  take-to-take difference counts as an "error", what to do with ceilings not seen and doors missed, and whether a
  1.5 cm tolerance can be compared with a 3% one. Making those choices now, after the numbers are known, is what the
  rule was written to prevent.
- The snippet was tested on the dry run (`outputs/own_test/`) before the capture existed, so its behaviour was fixed
  in advance.

What this choice leaves out:
- The video ceilings are not declared. The declared fix changes the video scale, and the ceiling heights depend on it.
  The declaration states the predicted ceiling numbers: the replays keep 0 of 3.
- The photo repeat room is not declared. The one wall both takes measure, W5, spans the W4 and W6 sides: 2.74 m lit
  and 2.13 m dim against 2.997 m. The photo diagnosis puts those sides on the side rule, which takes furniture faces
  for walls (cause 1 in `outputs/own_house/diag/photo_workflow_result.json`).

## Files

All under `outputs/fixloop/before/`:
- `eval/own_eval.md`, `eval/own_eval.json`: the scored runs.
- `rank_rule_e042ca5.py`: the committed snippet. `rank_raw.txt`: its output. `rank.py`, `rank.json`,
  `rank_tables.md`: the steps by hand and the tables above.
- `eval/step4_scale_check.json`: Part 3, step 4 on the video runs.
- `predict/summary.json`: the replays of the fix, scored with the scorer at `7b1a40f`.
- `run_video_before.sh` and its log, `video_take1_r1.log`, `video_take1_r2.log`, `CODE_COMMIT.txt`.
