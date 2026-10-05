# Own house video at HEAD: default vs PnP scale, and the sample videos

5 Oct, 08:15–10:26 IST. Code: `5fe6ae6` for every run (a git-archive snapshot, so commits made during the runs could
not mix in). Runs, logs and scripts: `outputs/presentable/` (git-ignored): `run.sh`, `run_pnp_samples.sh`,
`summarize.py`, `label_by_walk.py`, `samples_table.py`. One GPU job at a time, under `outputs/.gpu.lock`.

## 1. My video, take1, against the tape

Four fresh runs of `run_capture.py take1.mp4 --tier video --no-damage`, one after the other: the default scale step
(depth agreement) twice and `--video-scale pnp` twice. Each run makes its own DPVO run. Scored with
`scripts/eval_own_capture.py` and the tape outlines (`gt_polygons.json`): `outputs/presentable/eval/` as is,
`eval_walk/` with the room names below.

Tape: bedroom 11.09 m²; hall 12.71 m² with the entrance; kitchen about 4.2 m²; passage about 1 m²; house about
29–31 m². 14 walls (bedroom 6, hall 5, kitchen 3). 7 openings: 4 doors (bedroom 0.787 and 0.737 m, the hall
entrance 1.04 m, the bathroom 0.70 m) and 3 windows (bedroom 1.473, hall 1.753, kitchen 1.155 m). Ceiling 2.629 m.

Room pairing. Every video plan room is labelled `room`, and the tape calls the bedroom `room`. The scorer pairs by
label first, so a plan with one room is scored as the bedroom whatever it holds. I name each plan room by the
keyframe cameras inside it (`label_by_walk.py`; bedroom = keyframes 138–227, t 81–117 s) and score that copy. This
changes one run: default r2, whose one room holds t 0–46 s (hall and passage). Scored as the bedroom it gets 0 / 0 / 3
walls; scored as the hall, 1 / 1 / 4.

| Run | Scale | Keyframes kept | Rooms | Footprint | Bedroom (11.09 m²) | Other room | Walls ≤3% / ≤10% / found, of 14 | Openings in the plan | Ceiling (2.629 m) |
|---|---|---|---|---|---|---|---|---|---|
| default r1 | depth agreement | 122 of 228 | 1 | 9.99 m² (−67%) | 9.99 m² (−10%) | none (hall, kitchen, passage missing) | 0 / 2 / 5 | none | bedroom 2.605 m (−2.4 cm) |
| default r2 | depth agreement | 71 of 228 | 1 | 19.40 m² (−35%) | missing | 19.40 m²: hall and passage | 1 / 1 / 4 | door 0.77 m to unmapped space, window 1.20 m (not checked which) | not observed |
| pnp r1 | pnp | 228 of 228 | 2 | 31.12 m² (+4%) | 9.02 m² (−19%) | 22.10 m²: hall, kitchen, passage | 0 / 2 / 9 | door 0.68 m, bedroom to hall side (passage door 0.787 m: −10.6 cm) | bedroom 2.38 m (−25 cm); hall side 2.48 m (−14 cm) |
| pnp r2 | pnp | 209 of 228 | 2 | 28.99 m² (−3%) | 10.03 m² (−10%) | 18.97 m²: hall, kitchen, passage | 4 / 5 / 11 | door 0.74 m, bedroom to hall side (passage door 0.787 m: −5.0 cm); window 0.68 m on the hall side (hall window 1.753 m) | bedroom 2.46 m (−17 cm); hall side 2.43 m (−20 cm) |

- Hall, kitchen and passage together are about 17.9 m² on tape. No run makes the kitchen or the passage a room of
  its own.
- The PnP doors join the bedroom to the hall-side room, so they are the bedroom's door to the passage, D1 (0.787 m,
  on wall W1). D2 (0.737 m) is on the bedroom's outer wall W4, the side with the hall and kitchen windows. The scorer
  pairs doors by width only: it pairs the PnP doors with D2 (−5.6 cm and 0.0 cm), and again with the hall entrance,
  because the door joins the two plan rooms. The table counts each door once, at D1.
- pnp r2's walls within 3%: bedroom 3.376 vs 3.480 m (−3.0%) and 2.966 vs 2.997 m (−1.0%); hall 4.535 vs 4.469 m
  (+1.5%) and 2.869 vs 2.883 m (−0.5%).

The scale self-check per run (`scene/scene_info.json`):
- default r1: DPVO restarted 3 times; 2 of 4 segments fail the depth residual (0.099 and 0.100, limit 0.058), and
  their walk lengths are 117.5 and 115.5 m. 106 keyframes are dropped; the bedroom is what is left.
- default r2: 1 restart; the segment from keyframe 71 fails the scale spread (2.25, limit 2.0). 157 keyframes are
  dropped, the bedroom with them.
- pnp r1: one segment, 181 votes, vote coverage 0.89, whole scene consistent; floor levelled.
- pnp r2: 2 restarts; keyframes 0–18 have no votes and are dropped (the first 16 s of the hall). The two restarts are
  not joined by verified geometry, so the plan says reliability low.

The plans (`plan.png` in each run folder):
- default r1: one notched room, the bedroom. No doors, no windows.
- default r2: one 19.4 m² room with one door and one window. No bedroom.
- pnp r1: the bedroom and one large hall-side room, one door between them. No windows.
- pnp r2: the bedroom and a hall-side room, the door between them and one window. Its long hall wall is 4.53 m
  (tape 4.469 m). It is the closest of the four to the tape sketch.

None of them is close to CubiCasa's plan of the same flat: 4 named spaces, every door with its swing, the windows,
the stove and the sinks. The best video plan has 2 spaces, 1 door and 1 window.

## 2. Which scale step for my video

PnP is better on the footprint, the rooms, the walls found and the doors, in both runs:
- footprint within 4% in both (31.12 and 28.99 m²), against −67% and −35%;
- 2 rooms with the bedroom in both, against 1 room each, one of them without the bedroom;
- walls found 9 and 11 of 14, against 5 and 4; within 3%: 0 and 4, against 0 and 1;
- the bedroom's door to the passage in both runs, 10.6 and 5.0 cm narrow. The default runs do not show it: r1 has no
  openings, r2 has no bedroom.

The default is as good or better on two counts, both from r1: its bedroom ceiling is 2.4 cm off (PnP 14–25 cm low),
and its bedroom area is −10% (PnP −19% and −10%).
This matches the D-076 replays on cached DPVO runs: on take1 the median footprint error is 2.7% with PnP and 26%
with the default.

Recommendation: make the plan of my house with `--video-scale pnp` and show pnp r2
(`outputs/presentable/video_pnp_r2/plan.png`), saying it is the better of two runs. Keep the default as it is: on the
recruiters' sample videos PnP is not better as a rule (D-076: not worse on only 6 of 12 replays). The fresh runs of
section 3 split the same way: PnP is better on single_room (−2% against +37%) and worse on floor_only (4 rooms and
−64%, 1 room matched, against 7 rooms and +11%, 4 matched). They are one run each on different DPVO runs (I-007), not
replays of the same one. I changed no default.

## 3. The recruiters' sample videos against their LiDAR plans

Reference-based: the LiDAR plan of the same capture (`outputs/benchmark/final/lidar/<capture>/drift_on/plan.json`),
scored with `scripts/bench_tier_ref.py` (unchanged since `dced951`), not ground truth. Video and LiDAR tiers only.
"Matched room areas" pairs rooms by overlap after a rigid fit of the two footprints. "HEAD default" and "HEAD pnp" are
the runs made for this note (09:34–09:56 and 10:03–10:26).

| Capture (LiDAR plan) | Run | Code, scale step | Rooms | Footprint vs LiDAR | Rooms matched | Matched room areas, video vs LiDAR (m²) |
|---|---|---|---|---|---|---|
| single_room (3 rooms, 17.61 m²) | d066 | 4 Oct 20:58, depth agreement | 3 | 23.85 m² (+35%) | 3 of 3 | 8.82 vs 8.31 (+6%); 7.96 vs 1.93 (+312%); 7.06 vs 7.56 (−7%) |
|  | d069 r1 | 4 Oct 22:07, depth agreement | 3 | 17.32 m² (−2%) | 2 of 3 | 9.33 vs 8.31 (+12%); 6.38 vs 7.56 (−16%) |
|  | d069 r2 | 4 Oct 22:11, depth agreement | 3 | 16.82 m² (−4%) | 2 of 3 | 8.94 vs 8.31 (+8%); 6.45 vs 7.56 (−15%) |
|  | d069 r3 | 4 Oct 22:16, depth agreement | 3 | 17.14 m² (−3%) | 2 of 3 | 9.02 vs 8.31 (+9%); 6.88 vs 7.56 (−9%) |
|  | fix loop after | dced951, PnP (D-075) | 3 | 22.07 m² (+25%) | 3 of 3 | 8.51 vs 8.31 (+2%); 7.71 vs 7.56 (+2%); 5.85 vs 1.93 (+202%) |
|  | HEAD default | 5fe6ae6, depth agreement | 3 | 24.15 m² (+37%) | 3 of 3 | 9.09 vs 8.31 (+9%); 7.79 vs 1.93 (+303%); 7.27 vs 7.56 (−4%) |
|  | HEAD pnp | 5fe6ae6, `--video-scale pnp` | 3 | 17.23 m² (−2%) | 2 of 3 | 8.47 vs 8.31 (+2%); 7.46 vs 7.56 (−1%) |
| floor_only (8 rooms, 61.90 m²) | d066 | 4 Oct 21:16, depth agreement | 5 | 30.11 m² (−51%) | 4 of 8 | 8.03 vs 9.26 (−13%); 6.26 vs 2.84 (+121%); 5.47 vs 7.68 (−29%); 1.62 vs 1.68 (−4%) |
|  | fix loop after | dced951, PnP (D-075) | 0 | 0.00 m² (−100%) | 0 of 8 | none |
|  | HEAD default | 5fe6ae6, depth agreement | 7 | 68.64 m² (+11%) | 4 of 8 | 27.14 vs 13.99 (+94%); 16.35 vs 9.26 (+77%); 5.78 vs 12.12 (−52%); 5.14 vs 7.68 (−33%) |
|  | HEAD pnp | 5fe6ae6, `--video-scale pnp` | 4 | 22.31 m² (−64%) | 1 of 8 | 3.33 vs 13.99 (−76%) |
| with_ceiling (8 rooms, 62.50 m²) | final | 4 Oct 05:55, depth agreement (older code) | 5 | 19.40 m² (−69%) | 3 of 8 | 9.73 vs 12.81 (−24%); 3.35 vs 3.39 (−1%); 2.71 vs 11.17 (−76%) |
|  | fix loop after | dced951, PnP (D-075) | 1 | 5.20 m² (−92%) | 1 of 8 | 5.20 vs 9.33 (−44%) |

Run folders: d066, d069 and final in `outputs/benchmark/final/video/<capture>/`; fix loop after in
`outputs/fixloop/after/sample_<capture>`; HEAD default and pnp in `outputs/presentable/sample_<capture>` and
`sample_<capture>_pnp`.

- single_room, HEAD default: like d066. The two main rooms are within +9% and −4% of LiDAR, but the closet (LiDAR
  1.93 m²) grows into a 4.26 × 1.83 m room of 7.79 m², so the footprint is +37% (I-014, D-070). The d069 runs, the
  same scale step on 4 Oct, gave −2% to −4%, with the closet not matched.
- single_room, HEAD pnp: the closest video plan of the sample so far. The two main rooms are +2% and −1%, 3 of 4
  room sides within 3%, footprint −2%. The closet comes out as a 4.12 × 0.31 m strip and is not matched. The
  self-check fails the run's only segment (vote coverage 0.31, limit 0.5) and keeps it with the 25% sigma floor.
- floor_only, HEAD default: 7 rooms and +11% footprint, but the rooms are wrong. The largest room is 27.14 m² against
  LiDAR's 13.99 m², the four matched rooms are −52% to +94% off, no room side is within 3%, and one room sits apart
  from the rest. DPVO restarted twice; neither restart is joined by verified geometry.
- floor_only, HEAD pnp: 4 rooms, −64%, 1 room matched. 2 of 4 segments fail the vote coverage, and 181 of 432
  keyframes are dropped with them.
- with_ceiling: not run again (about 27 min a run).
- So the video plans do not match the LiDAR plans of the sample videos. The closest is single_room: its two main
  rooms are 1–16% off on every run, and the closet is missed, too big or a strip. On floor_only and with_ceiling
  (8 rooms each) at most 4 rooms match per run, and 2 of the 13 matched room areas are within 10%.
- PnP on the samples (D-076 replays on the same DPVO runs, not new runs): worse on single_room d069 r1–r3 (+21, +28,
  −16% against +1, −3, −5%), floor_only d066 (1 room against 5) and with_ceiling's after run (2 rooms, −66%,
  against 3 rooms, −13%); better on single_room's after run (+8% against +20%) and floor_only's after run (3 rooms
  against 0).

## 4. A scorer caveat

The label pairing of section 1 also touches the fix loop's after r2 run (`outputs/fixloop/after/video_take1_r2`).
Its one room (6.35 m²) holds 11 passage and 2 hall keyframes, no bedroom ones, and was scored as the bedroom. Its
walls within 3% are 0 either way. Its one wall within 10% (2.361 m against the bedroom's 2.545 m) belongs to that
room, not to the bedroom.
