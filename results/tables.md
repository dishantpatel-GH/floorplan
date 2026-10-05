# Benchmark tables (HEAD 0c97b0e1f334e8047e9ee7361fff284be6fbbc93 (2026-10-05 15:24:21 +0530 defaults judged on k65: auto video scale, door stitching, see-through doors); code = git archive of HEAD in code/, exported 2026-10-05T16:00:31+05:30)

Numbers from `results/summary.json` (scripts/bench_results.py). Wall gate: photo ±8%, video ±3% of the tape / simulator length; MISSED walls count as failures. Footprint against the tape outline (own house) or the simulator's rooms (k65). Openings: GT openings in the rooms the plan has, found = placed on the right wall; width error over the found ones that carry a width. Time = pipeline runtime of the run (s).

## 1. Gates against ground truth

| Capture | Tier | Run | Walls within tol | Median wall err | Missed walls | GT inside 95% CI | Footprint | Ceiling (tape 2.6289 m own house) | Openings found | Width err median | Time s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| own house (dim) | photo | live | 5/6 (±8%) | 3.6% | 1 | 5/5 | 11.15 vs 11.09 m² (+0.6%) | 2.56 (-7 cm) | 3/3, 0 phantom | 5.9 cm (0 within 2 cm) | 36 |
| own house (lit) | photo | live | 2/6 (±8%) | 8.9% | 1 | 5/5 | 11.54 vs 11.09 m² (+4.0%) | 2.51 (-11 cm) | 2/3, 0 phantom | 6.0 cm (0 within 2 cm) | 41 |
| own house take1 | video | replay own_2328 | 1/14 (±3%) | 31.0% | 2 | 2/12 | 29.82 vs 28.02 m² (+6.4%) | 3.60 (+97 cm), 2.46 (-17 cm), 2.07 (-56 cm) | 3/6, 3 phantom | 24.4 cm (0 within 2 cm) | 54 replan (live run 566) |
| own house take1 | video | replay own_b1 | 1/14 (±3%) | 18.5% | 5 | 3/9 | 29.23 vs 28.02 m² (+4.3%) | 2.48 (-15 cm), 2.43 (-19 cm) | 3/5, 1 phantom | 31.1 cm (0 within 2 cm) | 72 replan (live run 747) |
| own house take1 | video | replay own_b2 | 3/14 (±3%) | 18.6% | 4 | 4/10 | 31.34 vs 28.02 m² (+11.9%) | 2.51 (-11 cm), 2.43 (-19 cm) | 2/6, 9 phantom | 42.6 cm (0 within 2 cm) | 70 replan (live run 562) |
| own house take1 | video | replay own_a1 | 1/14 (±3%) | 16.7% | 10 | 1/4 | 27.38 vs 28.02 m² (-2.3%) | 3.06 (+43 cm), 2.87 (+24 cm) | 0/4, 6 phantom | — (0 within 2 cm) | 51 replan (live run 559) |
| own house take1 | video | replay own_a2 | 2/14 (±3%) | 24.9% | 4 | 3/10 | 30.80 vs 28.02 m² (+9.9%) | 2.43 (-19 cm), 2.52 (-11 cm) | 3/5, 3 phantom | 22.4 cm (0 within 2 cm) | 52 replan (live run 681) |
| own house take1 | video | replay own_p1 | 1/14 (±3%) | 15.5% | 4 | 5/10 | 31.00 vs 28.02 m² (+10.7%) | 2.43 (-20 cm), 2.42 (-20 cm) | 3/5, 3 phantom | 22.5 cm (0 within 2 cm) | 48 replan (live run 713) |
| own house take1 | video | replay own_pp1 | 0/14 (±3%) | 24.6% | 7 | 1/7 | 30.96 vs 28.02 m² (+10.5%) | 2.34 (-29 cm), 2.48 (-15 cm) | 4/5, 0 phantom | 19.3 cm (0 within 2 cm) | 48 replan (live run 585) |
| own house take1 | video | replay own_p2 | 2/14 (±3%) | 25.9% | 4 | 4/10 | 29.50 vs 28.02 m² (+5.3%) | 2.36 (-26 cm), 2.55 (-7 cm) | 3/5, 1 phantom | 5.1 cm (0 within 2 cm) | 47 replan (live run 863) |
| own house take1 | video | replay own_pp2 | 3/14 (±3%) | 14.8% | 4 | 7/10 | 28.59 vs 28.02 m² (+2.1%) | 2.46 (-17 cm), 2.41 (-22 cm) | 4/5, 0 phantom | 25.8 cm (0 within 2 cm) | 43 replan (live run 951) |
| k65 (sim), IoU 0.411 | photo | live | 11/26 (±8%) | 6.6% | 8 | 18/18 | 56.94 vs 59.20 m² (-3.8%) | 2.60 (-20 cm), 2.69 (-11 cm), 2.85 (+30 cm), 2.79 (+20 cm), 2.60 (+0 cm) | 8/9, 3 phantom | 8.2 cm (0 within 2 cm) | 150 |
| k65 (sim), IoU 0.932 | lidar | live | 4/26 (±max(1 cm, 0.5%)) | 1.8% | 1 | 15/25 | 57.42 vs 59.20 m² (-3.0%) | 2.81 (+1 cm), 2.81 (+1 cm), 2.57 (+1 cm), 2.61 (+1 cm), 2.61 (+1 cm) | 5/9, 2 phantom | 0.6 cm (4 within 2 cm) | 440 |
| k65 (sim), IoU 0.302 | video | replan of the cached 4 Oct run (older video front end) | 0/26 (±3%) | 20.9% | 10 | 10/16 | 134.35 vs 59.20 m² (+126.9%) | 4.01 (+146 cm), 1.92 (-68 cm), 1.92 (-68 cm) | 3/9, 20 phantom | 128.9 cm (0 within 2 cm) | 333 replan (live run 3693) |

## 2. Repeatability

### Own house, photo: the same room captured twice (dim light vs lit)

| Item | Dim | Lit | Tape | Dim − lit | |dim − lit| / mean |
|---|---|---|---|---|---|
| room W1 | 2.701 | 2.477 | 2.545 | +22.4 cm | 8.7% |
| room W3 | 0.448 | 0.399 | 0.440 | +4.9 cm | 11.6% |
| room W4 | 3.474 | 3.946 | 3.480 | -47.2 cm | 12.7% |
| room W5 | 3.149 | 2.876 | 2.997 | +27.3 cm | 9.1% |
| room W6 | 3.598 | 4.068 | 3.734 | -46.9 cm | 12.2% |
| room C1 | 2.562 | 2.514 | 2.629 | +4.8 cm | 1.9% |
| room D1 | 0.760 | — | 0.787 | — | — |
| room D2 | 0.800 | 0.800 | 0.737 | +0.0 cm | 0.0% |
| room Win1 | 1.532 | 1.530 | 1.473 | +0.2 cm | 0.1% |

Walls in both: 6; median |dim − lit| 24.9 cm, max 47.2 cm, median relative 10.3%; 1 of 6 within 8% of each other.

### Own house, video take1: 9 cached DPVO/MoGe-2 runs replayed with the HEAD code

| Item | Tape | Found in | Mean | SD | Min | Max | Range |
|---|---|---|---|---|---|---|---|
| room W1 | 2.545 | 8/9 | 1.703 | 0.392 | 1.018 | 2.176 | 1.159 |
| room W2 | 0.126 | 2/9 | 0.384 | 0.161 | 0.270 | 0.498 | 0.228 |
| room W3 | 0.440 | 5/9 | 0.827 | 0.958 | 0.218 | 2.505 | 2.287 |
| room W4 | 3.480 | 6/9 | 3.488 | 0.203 | 3.223 | 3.820 | 0.597 |
| room W5 | 2.997 | 8/9 | 2.829 | 0.216 | 2.490 | 2.999 | 0.509 |
| room W6 | 3.734 | 9/9 | 2.572 | 0.749 | 0.913 | 3.310 | 2.397 |
| room C1 | 2.629 | 9/9 | 2.629 | 0.421 | 2.335 | 3.596 | 1.261 |
| hall W1 | 4.469 | 8/8 | 3.470 | 0.409 | 3.002 | 4.380 | 1.378 |
| hall W2 | 2.883 | 8/8 | 2.646 | 0.475 | 1.897 | 3.300 | 1.404 |
| hall W3 | 0.705 | 5/8 | 0.340 | 0.262 | 0.094 | 0.740 | 0.646 |
| hall W4 | 2.388 | 8/8 | 1.193 | 0.668 | 0.492 | 2.404 | 1.912 |
| hall W5 | 2.845 | 8/8 | 2.558 | 0.322 | 2.098 | 3.058 | 0.960 |
| hall C1 | 2.629 | 8/8 | 2.463 | 0.051 | 2.406 | 2.555 | 0.148 |
| kitchen W1 | 1.600 | 1/3 | 1.794 | — | 1.794 | 1.794 | 0.000 |
| kitchen W2 | 2.210 | 3/3 | 1.886 | 0.859 | 1.212 | 2.853 | 1.641 |
| kitchen W3 | 2.210 | 3/3 | 1.420 | 1.000 | 0.287 | 2.183 | 1.896 |
| kitchen C1 | 2.629 | 2/3 | 2.469 | 0.561 | 2.072 | 2.866 | 0.794 |
| room D1 | 0.787 | 7/9 | 0.687 | 0.183 | 0.476 | 1.031 | 0.556 |
| room D2 | 0.737 | 1/9 | 1.230 | — | 1.230 | 1.230 | 0.000 |
| hall D1 | 1.040 | 3/9 | 0.667 | 0.087 | 0.574 | 0.748 | 0.173 |
| passage D1 | 0.700 | 8/9 | 0.699 | 0.012 | 0.691 | 0.727 | 0.036 |
| room Win1 | 1.473 | 7/9 | 1.525 | 0.004 | 1.522 | 1.534 | 0.012 |
| hall Win1 | 1.753 | 7/9 | 1.060 | 0.145 | 0.740 | 1.155 | 0.415 |
| kitchen Win1 | 1.155 | 6/9 | 0.918 | 0.194 | 0.574 | 1.085 | 0.511 |

Median wall error per run: 14.8% to 31.0%
Footprint per run: 27.38 to 31.34 m²

## 3. Sample captures (no ground truth)

LiDAR runs live; video = replays of cached tracks scored against the LiDAR plan of the same capture (REFERENCE-BASED, not ground truth; room dimensions within ±3%).

| Capture | LiDAR rooms / walls / openings | LiDAR footprint m² | LiDAR time s | Same plan as the 10:04 run | Video run | Rooms matched | Room dims within 3% | Median dim err | Footprint vs LiDAR | One stitched plan | Video plan s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| single_room | 3 / 18 / 2 | 17.61 | 77 | yes (max wall diff 0.0 cm) | sr_r1 | 2/3 | 1/4 | 4.6% | +1.3% | True | 42 replan (live run 261) |
| | | | | | sr_r2 | 2/3 | 2/4 | 2.5% | -2.9% | True | 56 replan (live run 267) |
| | | | | | sr_r3 | 2/3 | 2/4 | 2.9% | -5.1% | True | 66 replan (live run 267) |
| | | | | | sr_a | 2/3 | 0/4 | 11.5% | +19.9% | True | 72 replan (live run 235) |
| floor_only | 8 / 70 / 8 | 61.90 | 234 | yes (max wall diff 0.0 cm) | fo_a | 7/8 | 1/14 | 16.9% | +13.3% | True | 104 replan (live run 781) |
| | | | | | fo_d066 | 4/8 | 1/8 | 16.6% | -49.0% | True | 94 replan (live run 989) |
| with_ceiling | 8 / 70 / 12 | 62.50 | 472 | yes (max wall diff 0.0 cm) | wc_a | 3/8 | 0/6 | 25.6% | +11.1% | False | 175 replan (live run 435) |

LiDAR repeat pair (with_ceiling vs floor_only: two scans of the same flat, registered from the scene geometry; a wall passes when the two lengths agree within max(1 cm, 0.5%), unmatched walls fail):

- walls judged 87, pass 15, fail 38, unmatched 34; pass rate 17.2% (Wilson 95% [0.107, 0.265]); median |delta| 6.39 cm; within 2 cm 18
- rooms 8 / 8, matched 8; footprint 62.5 / 61.9 m²; same-topology walls 14/30 pass (median 1.8 cm)
- openings matched 8, widths within 2 cm 2/12, median |delta| 6.0 cm

## 4. Drift on / off (LiDAR, stitched footprint, same capture)

| Capture | Footprint on m² | Footprint off m² | Rooms on / off | Walls same within gate | Median wall delta cm |
|---|---|---|---|---|---|
| floor_only | 61.9 | 52.59 | 8 / 7 | 11/89 | 4.0 |
| with_ceiling | 62.5 | 61.9 | 8 / 9 | 8/92 | 7.5 |
