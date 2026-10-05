# Compliance matrix (Deliverable 1)

Every requirement of the brief (`TakeHome/Applied_AI_Case_Study.pdf`, Parts 1–5, deliverables 1–8, constraints, the
walk-in test) → where it lives → what the artifact is → status. Statuses that depend on numbers are filled from
the fresh runs of the final code (`0c97b0e`, 5 Oct 16:00–16:38 IST, `results/summary.json`); `docs/BENCHMARK_REPORT.md`
has the full gate rows. Restructured on 5 Oct with the final defaults (photo `pillar_face_side` on, D-086; video scale step
`"auto"`, D-087; photo door stitching on, D-088; see-through doorways on; `remove_jogs` stops when a snap changes
nothing).

Status key: **pass**; **partial** (done in part, reason given); **fail** (not met, reason given); **deviation**
(met differently from the brief, disclosed, reason given).

## Part 1: capture route, tiers, device matrix

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 1.1 | Route 2: name the stock capture tools; one-page protocol a non-engineer follows (install, walk, how long, avoid, hand-over) | `docs/CAPTURE_PROTOCOL.md` | one page: Camera app (photos, video), Stray Scanner (LiDAR); one folder per room; hand-over into `data/<capture>/` and one command per tier | **pass**. Followed on my own flat (OnePlus Nord, 4 Oct) and by the scripted walker in the simulator; revised from that evidence (D-051, D-055, D-064, D-068, D-074) |
| 1.2 | Photo tier: 2–8 stills per room, no depth or poses, one folder per room → the same stitched whole-property plan, intervals widened | `floorplan/photo/`, `scripts/run_capture.py --tier photo` | room folders → metric scene → plan; rooms joined at shared doors (door stitching on, D-088) | runs end to end: **pass**. Stitch on k65: **partial** (5/5 rooms, adjacency 4/4, no overlaps, footprint −3.8%, but one room drawn about 9 m from its place, IoU 0.41). Walls ±8%: **fail** (own dim 5/6, lit 2/6, k65 11/26); `BENCHMARK_REPORT.md` §2 |
| 1.3 | Video tier: handheld walkthrough clip | `floorplan/video/`, `--tier video` | plan from the clip alone: DPVO path, scale step `"auto"` (D-087) | runs end to end: **pass**. Accuracy: **fail** (own `take1` 0–3 of 14 walls within 3% over 9 runs, footprint −2.3% to +11.9%; sample clips 0–2 of 4, 1 of 14 and 1 of 8, 0 of 6 room dimensions within 3% of the LiDAR plan); `BENCHMARK_REPORT.md` §2–3 |
| 1.4 | LiDAR tier: depth, poses, intrinsics on Pro-class devices | `floorplan/pipeline/scene.py`, `floorplan/recon/drift.py`, `floorplan/plan/beta/`, `--tier lidar` | plan from a Stray Scanner folder | runs end to end: **pass**. Accuracy: **partial** (k65 median wall error 1.8%, ceilings +1.1 to +1.4 cm, IoU 0.93; laser rooms 8/8 within 1.5 cm, cited; repeat pair 15/87 walls within max(1 cm, 0.5%)); `BENCHMARK_REPORT.md` §2 |
| 1.5 | Same output contract from every tier; intervals widen as sensor data thins | `floorplan/model.py`, `schema/plan.schema.json`, `floorplan/uncertainty/tier_budget.py` | one JSON schema; a tier scale term per tier (D-016) and the σ floor for unverified video joins (D-034) | contract: **pass**. Calibration: **partial** (photo 1.00 / 1.00 / 0.97; LiDAR k65 0.70; video 0.32–0.76 per run); `BENCHMARK_REPORT.md` §2 |
| 1.6 | Photos and video from any iPhone 15 or newer | `docs/DEVICE_MATRIX.md`, `sim/` | own data from a OnePlus Nord; iPhone 15-format files from the simulator (k65); phone-format robustness copies (HEIC, rotation tags, no EXIF) | **deviation**: no iPhone 15 was available to me. The formats were emulated and tested; a real iPhone 15 capture has not been run |
| 1.7 | Device matrix: tier → hardware → honest accuracy | `docs/DEVICE_MATRIX.md` | table + scale and pose sources per tier + limits | **pass**; its accuracy cells point to BENCHMARK_REPORT |

## Part 2: output contract (per capture)

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 2.1 | Dimensioned per-room plan: walls, ceiling height, floor area, openings | `floorplan/plan/beta/`, `floorplan/openings/` | `plan.json` rooms, walls, ceiling heights, areas, openings; doors from priors where the door itself is not seen (D-085, rule d); see-through doorways (photo tier, `docs/modules/openings.md`) | present at every tier: **pass**. Accuracy: **partial**, per tier in `BENCHMARK_REPORT.md` §3 |
| 2.2 | Stitched multi-room plan with correct adjacency | `floorplan/plan/beta/`, `floorplan/photo/` (door stitching) | one `plan.json` per capture with adjacency; `plan.png` | LiDAR and video: one continuous capture. Photo: rooms joined at shared doors and doorway photos. Adjacency: **pass** for photo k65 (4/4) and LiDAR k65 (4/4, 1 extra pair); **fail** for video (own take1 misses a room in 7 of 9 runs; k65 1/4) |
| 2.3 | Per-surface damage regions with class and metric extent | `floorplan/damage/` | `plan.json` damage regions (class, area or length, surface) | **partial**: implemented and run in every `run_capture` call; scored on simulated decals only (`docs/modules/damage.md`), because no staged damage was captured (row B2) |
| 2.4 | Concealed-damage flags with the rule that fired | `floorplan/damage/rules.py`, `rules.json` | `plan.json` damage flags with the rule id | **pass** (`WALL_BASE_WICKING` fires on injected wall-base stains); no real damage to fire on |
| 2.5 | Scope line items keyed to surfaces | `floorplan/damage/scope.py` | `plan.json` scope items | **pass** |
| 2.6 | A confidence interval on every measurement | `floorplan/model.py` (Measurement), `floorplan/uncertainty/` | `ci95` on every number; `not_observed` where a value was not seen | present: **pass**. Coverage: **partial** (photo 0.97–1.00; LiDAR k65 0.70; video 0.32–0.76). Topology errors and missing rooms are not inside any interval (BF-13) |
| 2.7 | One command per capture | `scripts/run_capture.py` | `run_capture.py <input> --tier lidar\|video\|photo` | **pass** |
| 2.8 | JSON to the published schema | `schema/plan.schema.json`, `floorplan/export/` | schema v1.0.0, every run validated on save | **deviation**: no schema was published with the brief (`docs/OPEN_QUESTIONS.md`); ours is modelled on RoomPlan and magicplan's exports |
| 2.9 | Rendered plan a homeowner would recognise | `floorplan/export/render.py`, `floorplan/export/dxf.py` | `plan.svg`, `plan.png`, `plan.dxf` | **pass** (`docs/modules/export.md`); pictures in the reports |
| 2.10 | The stitched plan from every tier, photos included | the three front-ends + the shared back-end | three plans of the same home | **partial**: all three tiers run on k65 and give one plan each (photo stitched, LiDAR, video); the own flat has photo plans of the bedroom only and video plans of the flat; `BENCHMARK_REPORT.md` §3 |

## Part 2: benchmark set

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| B1 | One multi-room capture, 3+ rooms plus a connector | `data/sample/` (floor_only, with_ceiling), `data/own_house/video/`, `data/k65/` | sample flat (LiDAR, video); own flat walkthrough (video); simulated k65 flat (photos per room) | **partial**: real multi-room captures at LiDAR and video; the own photo set is one room, so the multi-room photo set is the simulated k65 flat |
| B2 | One furnished room with staged damage, two damage classes | `docs/HOUSE_CAPTURE_GUIDE.md` (step 3), `docs/modules/damage.md` | planned paper decals (water stain, crack) | **fail**: the decals were not staged in the own capture. The detector was rehearsed on simulated decals only |
| B3 | The same rooms at all three tiers, the multi-room set included; photos as per-room folders that must stitch | `data/own_house/`, `data/sample/`, `data/k65/` | own flat: video + photos; sample flat: LiDAR + video; k65: photos, video and LiDAR files | **deviation**: no LiDAR device for my own rooms (the recruiters allowed the sample data for LiDAR, `docs/OPEN_QUESTIONS.md` Q2). The sample runs video and LiDAR only, never photo; the photo tier is benchmarked on the own bedroom and k65 |
| B4 | At least one room captured twice at the same tier | `data/own_house/photos/room`, `room_take2`; `data/sample/` floor_only vs with_ceiling | own bedroom photographed lit and dim; sample LiDAR repeat pair | **pass** (data); the gate itself: G3 |
| B5 | Laser or tape ground truth on everything; raw sensor data and measurements submitted | `data/own_house/gt/` (tape), `data/own_house/sketch/`, `benchmark/ground_truth.csv` (the empty template), ARKitScenes (laser, fetched), `data/k65/gt/` (exact) | tape on the own flat, laser on 5 ARKitScenes rooms, exact geometry on k65 | **partial**: the recruiters' sample captures have no ground truth, so on them video is scored against the LiDAR plan; everything of mine has tape |

## Part 2: gates

| # | Gate | File path | Artifact | Status |
|---|---|---|---|---|
| G1 | Opening widths ≤ 2 cm on ≥ 85%; missed and phantom openings count as misses | `floorplan/eval/gates.py`, `floorplan/benchmark/gt_eval.py` | openings matched by position (D-085) | **fail** at every tier: LiDAR k65 4 of 9 within 2 cm (median 0.6 cm, 2 phantom); photo own dim 3/3 found, 2.7–6.3 cm off; k65 photo 8/9 found, 3 phantom; video 0 within 2 cm; `BENCHMARK_REPORT.md` §2 |
| G2 | Ceiling height ≤ 1.5 cm per room; spread ≤ 1 cm across captures; say whether biased or unrepeatable | `floorplan/uncertainty/bias.py`, `floorplan/eval/` | ceiling rows per tier | LiDAR **pass** per room on k65 (5/5, +1.1 to +1.4 cm); photo **fail** (own −7 / −11 cm, dim vs lit 4.8 cm; k65 −20 to +30 cm); video **fail** (2.07–3.60 m). Biased, not unrepeatable, at LiDAR (+1 cm in every room); `BENCHMARK_REPORT.md` §2. Of the sample captures only `with_ceiling` sees the ceilings, so the sample gives no LiDAR spread across captures |
| G3 | Repeatability: same room, same tier, within 1 cm or 0.5% per wall | `floorplan/eval/repeatability.py` | repeatability table | **fail**: LiDAR repeat pair 15/87 walls (17.2%, Wilson 10.7–26.5%), same-topology 14/30, median 1.8 cm, and the same command gives the same plan; photo dim vs lit 0/6 walls (median 24.9 cm); video 9 runs, bedroom W6 0.91–3.31 m; `BENCHMARK_REPORT.md` §4 |
| G4 | Drift accountability: method stated, footprint ablation on vs off | `floorplan/recon/drift.py`, `scripts/drift_ablation.py`, `docs/modules/drift.md` | pose graph with loop closures and plane anchors; ablation figures | method and ablation: **pass**. Numbers: floor_only 61.9 vs 52.6 m² (8 vs 7 rooms), with_ceiling 62.5 vs 61.9 m² (8 vs 9 rooms); `BENCHMARK_REPORT.md` §5 |
| G5 | Photo tier: one stitched plan, correct adjacency, no overlaps, footprint ±8% with calibrated stitch intervals | `floorplan/photo/`, `floorplan/eval/` | stitched photo plans (k65, own bedroom) | **partial** on k65: the scored items pass (5/5 rooms, adjacency 4/4 with 0 wrong, 0 overlaps, footprint −3.8% inside its interval, coverage 0.97), but one room is drawn about 9 m from its place (IoU 0.41). Own flat: bedroom only (footprint +0.6% dim, +4.0% lit); `BENCHMARK_REPORT.md` §2 |
| G6 | Photo walls ±8%, video walls ±3%, calibration scored at every tier | `floorplan/benchmark/gt_eval.py` | wall rows and interval coverage per tier | **fail**: photo ±8% own dim 5/6, lit 2/6, k65 11/26; video ±3% own 0–3/14, k65 0/26. Calibration scored at every tier: photo 0.97–1.00, LiDAR 0.70, video 0.32–0.76; `BENCHMARK_REPORT.md` §2 |

## Parts 3–5

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 3 | Head-to-head on 2 rooms: our LiDAR tier vs a consumer app (name, version, its export); beat or tie ≥ 70% of shared dimensions | `docs/BENCHMARK_REPORT.md` (head-to-head table), `data/cubicasa/`, `docs/DISCLOSURES.md` §8 | CubiCasa 3.14.1 (Android, Google Play, checked 5 Oct 2026): home data report PDF and plan images | **deviation**: no LiDAR device, so our video and photo tiers are compared with CubiCasa (itself a video scan) on the own flat, same tape. Beat-or-tie share: photo dim 3 of 3 (**pass**), photo lit 1 of 3, both photo takes 4 of 6 (67%, **fail**), video 1 of 5 (**fail**); `BENCHMARK_REPORT.md` §6 |
| 4 | Fix loop: worst gate with its number, root cause, fix and predicted number; shipped; before and after regenerable; readable diff | `docs/FIX_LOOP.md`, `docs/fixloop_ranking.md`, `docs/fixloop.diff`, tags `before-fix` / `after-fix` | declaration committed before the fix; the shipped fix (PnP scale votes, D-075), now inside the default `"auto"` step (D-087) | fix shipped, declaration and diff: **pass**. Gate: still failing on the declared runs and the prediction was wrong; the post-mortem is in `FIX_LOOP.md`. Final code: video walls within 3% 0–3 of 14 over 9 runs (still failing); `BENCHMARK_REPORT.md` §2 |
| 5 | Process evidence: commit as you work | git history, `README.md` (History) | commits in the order the work was done; tags `before-fix`, `after-fix` | how the history was put together is stated in `README.md` (History) |

## Deliverables

| # | Deliverable | File path | Artifact | Status |
|---|---|---|---|---|
| D1 | Compliance matrix | `docs/COMPLIANCE.md` | this table | **pass** |
| D2 | Capture route + device matrix | `docs/CAPTURE_PROTOCOL.md`, `docs/DEVICE_MATRIX.md` | one-page protocol, matrix | **pass** |
| D3 | README: running on a fresh capture in under 15 minutes on a clean machine, one command per capture | `README.md`, `setup/install.sh`, `scripts/fetch_weights.py`, `scripts/fetch_data.py` | install script, weight and data fetchers, one command per tier | **pass** on a clean clone in a separate folder of the dev machine (5 Oct): `setup/install.sh` 555 s with a warm package cache (weights 338 s), `fetch_data.py` from the Drive links 80 s, then LiDAR, photo and video runs reproduce `results/summary.json`; on another machine the install time depends on bandwidth (CUDA PyTorch is several GB) |
| D4 | Reproduction bundle: regenerate every reported number from raw inputs | `scripts/bench_*`, `results/`, `data/` (fetched) | one command per reported number; cached model outputs replay, the live path also runs | **pass**: `scripts/bench_fresh.sh` reruns everything and `scripts/bench_results.py` rescores; one command per number in `results/README.md`. Video numbers here are replays of cached tracks (the live path made the caches). DPVO is not bit-repeatable (I-007), so a video number can move between runs of the same command |
| D5 | Benchmark report: gates at all three tiers, repeatability, head-to-head, timing | `docs/BENCHMARK_REPORT.md` | report with plan pictures | **pass**: written from the fresh runs of the final code (`results/summary.json`); gates at all three tiers, repeatability, head-to-head, timing, plan pictures |
| D6 | Fix-loop bundle | `docs/FIX_LOOP.md`, `docs/fixloop.diff`, tags | as row 4 | as row 4 |
| D7 | Technical report, at most 6 pages | `docs/TECHNICAL_REPORT.md` | architecture, tiers and device matrix, drift, error budget, calibration, fix loop, failure modes | **pass**: organised by method, from the fresh runs; 6 A4 pages as rendered (`docs/TECHNICAL_REPORT.pdf`), 7 figures |
| D8 | Raw benchmark data: sensor logs, ground truth, app exports | `data/` (`scripts/fetch_data.py`, checked against `data/MANIFEST.json`), `data/cubicasa/` | own flat (photos, video, tape, sketch), k65 (simulated), the sample captures, the CubiCasa export | **pass** once the release archives are published; ARKitScenes is fetched from Apple under its own licence, not redistributed |

## Constraints and the walk-in test

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| C1 | Handheld consumer capture only | `docs/CAPTURE_PROTOCOL.md` | phone only, no rig, no reference object (D-067) | **pass** |
| C2 | Any pretrained model, dataset or API, with disclosure | `docs/DISCLOSURES.md` | every model, dataset, library and app with its licence | **pass** |
| C3 | Runs without calling our infrastructure | `floorplan/`, `README.md` | no API at run time; `HF_HUB_OFFLINE=1` after the fetch | **pass** |
| C4 | Weights and large binaries fetched by script | `scripts/fetch_weights.py`, `scripts/fetch_data.py`, `setup/` | pinned revisions and SHA-256 | **pass** |
| C5 | Mirrors, glass, wet-look surfaces and low light covered | `docs/FAILURE_MODES.md`, `floorplan/qa/`, `docs/CAPTURE_PROTOCOL.md` (Avoid) | analysis, detectors, handling, capture rules | **partial**: detectors and handling measured on the sample captures (mirror, glass, glossy floor views); low light measured on the own dim photo set (5/6 walls within 8% in dim light vs 2/6 lit; `BENCHMARK_REPORT.md` §3); no real large mirror, glass partition or wet floor was captured |
| W | Walk-in test: all three tiers ready to run cold on an unseen space, on the recruiters' iPhone 15 or newer | `scripts/run_capture.py`, `docs/CAPTURE_PROTOCOL.md` | one command per tier, time limits on the slow steps (D-058, D-061), offline after setup | **partial**: all three tiers run cold on unseen captures (the simulator's flats, the sample); never yet on a real iPhone 15 capture |
