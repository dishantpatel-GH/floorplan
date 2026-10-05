# Compliance matrix (Deliverable 1)

Every requirement of the brief (`TakeHome/Applied_AI_Case_Study.pdf`, Parts 1–5, deliverables 1–8, constraints, the
walk-in test) → where it lives → what the artifact is → status. Statuses that depend on numbers say "see
BENCHMARK_REPORT": `docs/BENCHMARK_REPORT.md` is regenerated from fresh runs of the final code, and its gate rows set
those statuses. Restructured on 5 Oct with the final defaults (photo `pillar_face_side` on, D-086; video scale step
`"auto"`, D-087; photo door stitching on, D-088; see-through doorways on; `remove_jogs` stops when a snap changes
nothing).

Status key: **pass**; **partial** (done in part, reason given); **fail** (not met, reason given); **deviation**
(met differently from the brief, disclosed, reason given); **see BENCHMARK_REPORT** (the number decides).

## Part 1: capture route, tiers, device matrix

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 1.1 | Route 2: name the stock capture tools; one-page protocol a non-engineer follows (install, walk, how long, avoid, hand-over) | `docs/CAPTURE_PROTOCOL.md` | one page: Camera app (photos, video), Stray Scanner (LiDAR); one folder per room; hand-over into `data/<capture>/` and one command per tier | **pass**. Followed on my own flat (OnePlus Nord, 4 Oct) and by the scripted walker in the simulator; revised from that evidence (D-051, D-055, D-064, D-068, D-074) |
| 1.2 | Photo tier: 2–8 stills per room, no depth or poses, one folder per room → the same stitched whole-property plan, intervals widened | `floorplan/photo/`, `scripts/run_capture.py --tier photo` | room folders → metric scene → plan; rooms joined at shared doors (door stitching on, D-088) | runs end to end: **pass**. Stitch and accuracy: **see BENCHMARK_REPORT** (own bedroom lit and dim, simulated k65 flat) |
| 1.3 | Video tier: handheld walkthrough clip | `floorplan/video/`, `--tier video` | plan from the clip alone: DPVO path, scale step `"auto"` (D-087) | runs end to end: **pass**. Accuracy: **see BENCHMARK_REPORT** (own `take1`, the 3 sample clips) |
| 1.4 | LiDAR tier: depth, poses, intrinsics on Pro-class devices | `floorplan/pipeline/scene.py`, `floorplan/recon/drift.py`, `floorplan/plan/beta/`, `--tier lidar` | plan from a Stray Scanner folder | runs end to end: **pass**. Accuracy: **see BENCHMARK_REPORT** (ARKitScenes laser, sample repeat pair) |
| 1.5 | Same output contract from every tier; intervals widen as sensor data thins | `floorplan/model.py`, `schema/plan.schema.json`, `floorplan/uncertainty/tier_budget.py` | one JSON schema; a tier scale term per tier (D-016) and the σ floor for unverified video joins (D-034) | contract: **pass**. Whether the intervals are calibrated: **see BENCHMARK_REPORT** (coverage per tier) |
| 1.6 | Photos and video from any iPhone 15 or newer | `docs/DEVICE_MATRIX.md`, `sim/` | own data from a OnePlus Nord; iPhone 15-format files from the simulator (k65); phone-format robustness copies (HEIC, rotation tags, no EXIF) | **deviation**: no iPhone 15 was available to me. The formats were emulated and tested; a real iPhone 15 capture has not been run |
| 1.7 | Device matrix: tier → hardware → honest accuracy | `docs/DEVICE_MATRIX.md` | table + scale and pose sources per tier + limits | **pass**; its accuracy cells point to BENCHMARK_REPORT |

## Part 2: output contract (per capture)

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 2.1 | Dimensioned per-room plan: walls, ceiling height, floor area, openings | `floorplan/plan/beta/`, `floorplan/openings/` | `plan.json` rooms, walls, ceiling heights, areas, openings; doors from priors where the door itself is not seen (D-085, rule d); see-through doorways (photo tier, `docs/modules/openings.md`) | present at every tier: **pass**. Accuracy: **see BENCHMARK_REPORT** |
| 2.2 | Stitched multi-room plan with correct adjacency | `floorplan/plan/beta/`, `floorplan/photo/` (door stitching) | one `plan.json` per capture with adjacency; `plan.png` | LiDAR and video: one continuous capture. Photo: rooms joined at shared doors and doorway photos. Adjacency scores: **see BENCHMARK_REPORT** |
| 2.3 | Per-surface damage regions with class and metric extent | `floorplan/damage/` | `plan.json` damage regions (class, area or length, surface) | **partial**: implemented and run in every `run_capture` call; scored on simulated decals only (`docs/modules/damage.md`), because no staged damage was captured (row B2) |
| 2.4 | Concealed-damage flags with the rule that fired | `floorplan/damage/rules.py`, `rules.json` | `plan.json` damage flags with the rule id | **pass** (`WALL_BASE_WICKING` fires on injected wall-base stains); no real damage to fire on |
| 2.5 | Scope line items keyed to surfaces | `floorplan/damage/scope.py` | `plan.json` scope items | **pass** |
| 2.6 | A confidence interval on every measurement | `floorplan/model.py` (Measurement), `floorplan/uncertainty/` | `ci95` on every number; `not_observed` where a value was not seen | present: **pass**. Coverage: **see BENCHMARK_REPORT**. Topology errors and missing rooms are not inside any interval (BF-13) |
| 2.7 | One command per capture | `scripts/run_capture.py` | `run_capture.py <input> --tier lidar\|video\|photo` | **pass** |
| 2.8 | JSON to the published schema | `schema/plan.schema.json`, `floorplan/export/` | schema v1.0.0, every run validated on save | **deviation**: no schema was published with the brief (`docs/OPEN_QUESTIONS.md`); ours is modelled on RoomPlan and magicplan's exports |
| 2.9 | Rendered plan a homeowner would recognise | `floorplan/export/render.py`, `floorplan/export/dxf.py` | `plan.svg`, `plan.png`, `plan.dxf` | **pass** (`docs/modules/export.md`); pictures in the reports |
| 2.10 | The stitched plan from every tier, photos included | the three front-ends + the shared back-end | three plans of the same home | **see BENCHMARK_REPORT** |

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
| G1 | Opening widths ≤ 2 cm on ≥ 85%; missed and phantom openings count as misses | `floorplan/eval/gates.py`, `floorplan/benchmark/gt_eval.py` | openings matched by position (D-085) | **see BENCHMARK_REPORT** |
| G2 | Ceiling height ≤ 1.5 cm per room; spread ≤ 1 cm across captures; say whether biased or unrepeatable | `floorplan/uncertainty/bias.py`, `floorplan/eval/` | ceiling rows per tier | **see BENCHMARK_REPORT**. Of the sample captures only `with_ceiling` sees the ceilings, so the sample gives no LiDAR spread across captures |
| G3 | Repeatability: same room, same tier, within 1 cm or 0.5% per wall | `floorplan/eval/repeatability.py` | repeatability table | **see BENCHMARK_REPORT** |
| G4 | Drift accountability: method stated, footprint ablation on vs off | `floorplan/recon/drift.py`, `scripts/drift_ablation.py`, `docs/modules/drift.md` | pose graph with loop closures and plane anchors; ablation figures | method and ablation: **pass**. Numbers: **see BENCHMARK_REPORT** |
| G5 | Photo tier: one stitched plan, correct adjacency, no overlaps, footprint ±8% with calibrated stitch intervals | `floorplan/photo/`, `floorplan/eval/` | stitched photo plans (k65, own bedroom) | **see BENCHMARK_REPORT** |
| G6 | Photo walls ±8%, video walls ±3%, calibration scored at every tier | `floorplan/benchmark/gt_eval.py` | wall rows and interval coverage per tier | **see BENCHMARK_REPORT** |

## Parts 3–5

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 3 | Head-to-head on 2 rooms: our LiDAR tier vs a consumer app (name, version, its export); beat or tie ≥ 70% of shared dimensions | `docs/BENCHMARK_REPORT.md` (head-to-head table), `data/cubicasa/`, `docs/DISCLOSURES.md` §8 | CubiCasa 3.14.1 (Android, Google Play, checked 5 Oct 2026): home data report PDF and plan images | **deviation**: no LiDAR device, so our video and photo tiers are compared with CubiCasa (itself a video scan) on the own flat, same tape. Beat-or-tie share: **see BENCHMARK_REPORT** |
| 4 | Fix loop: worst gate with its number, root cause, fix and predicted number; shipped; before and after regenerable; readable diff | `docs/FIX_LOOP.md`, `docs/fixloop_ranking.md`, `docs/fixloop.diff`, tags `before-fix` / `after-fix` | declaration committed before the fix; the shipped fix (PnP scale votes, D-075), now inside the default `"auto"` step (D-087) | fix shipped, declaration and diff: **pass**. Gate: still failing on the declared runs and the prediction was wrong; the post-mortem is in `FIX_LOOP.md`. Final-code numbers: **see BENCHMARK_REPORT** |
| 5 | Process evidence: commit as you work | git history, `README.md` (History) | commits in the order the work was done; tags `before-fix`, `after-fix` | how the history was put together is stated in `README.md` (History) |

## Deliverables

| # | Deliverable | File path | Artifact | Status |
|---|---|---|---|---|
| D1 | Compliance matrix | `docs/COMPLIANCE.md` | this table | **pass** |
| D2 | Capture route + device matrix | `docs/CAPTURE_PROTOCOL.md`, `docs/DEVICE_MATRIX.md` | one-page protocol, matrix | **pass** |
| D3 | README: running on a fresh capture in under 15 minutes on a clean machine, one command per capture | `README.md`, `setup/install.sh`, `scripts/fetch_weights.py`, `scripts/fetch_data.py` | install script, weight and data fetchers, one command per tier | **partial** until the clean-clone check on another machine has been run |
| D4 | Reproduction bundle: regenerate every reported number from raw inputs | `scripts/bench_*`, `results/`, `data/` (fetched) | one command per reported number; cached model outputs replay, the live path also runs | **see BENCHMARK_REPORT** (how each number is regenerated). DPVO is not bit-repeatable (I-007), so a video number can move between runs of the same command |
| D5 | Benchmark report: gates at all three tiers, repeatability, head-to-head, timing | `docs/BENCHMARK_REPORT.md` | report with plan pictures | written from the fresh runs |
| D6 | Fix-loop bundle | `docs/FIX_LOOP.md`, `docs/fixloop.diff`, tags | as row 4 | as row 4 |
| D7 | Technical report, at most 6 pages | `docs/TECHNICAL_REPORT.md` | architecture, tiers and device matrix, drift, error budget, calibration, fix loop, failure modes | written from the fresh runs |
| D8 | Raw benchmark data: sensor logs, ground truth, app exports | `data/` (`scripts/fetch_data.py`, checked against `data/MANIFEST.json`), `data/cubicasa/` | own flat (photos, video, tape, sketch), k65 (simulated), the sample captures, the CubiCasa export | **pass** once the release archives are published; ARKitScenes is fetched from Apple under its own licence, not redistributed |

## Constraints and the walk-in test

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| C1 | Handheld consumer capture only | `docs/CAPTURE_PROTOCOL.md` | phone only, no rig, no reference object (D-067) | **pass** |
| C2 | Any pretrained model, dataset or API, with disclosure | `docs/DISCLOSURES.md` | every model, dataset, library and app with its licence | **pass** |
| C3 | Runs without calling our infrastructure | `floorplan/`, `README.md` | no API at run time; `HF_HUB_OFFLINE=1` after the fetch | **pass** |
| C4 | Weights and large binaries fetched by script | `scripts/fetch_weights.py`, `scripts/fetch_data.py`, `setup/` | pinned revisions and SHA-256 | **pass** |
| C5 | Mirrors, glass, wet-look surfaces and low light covered | `docs/FAILURE_MODES.md`, `floorplan/qa/`, `docs/CAPTURE_PROTOCOL.md` (Avoid) | analysis, detectors, handling, capture rules | **partial**: detectors and handling measured on the sample captures (mirror, glass, glossy floor views); low light measured on the own dim photo set (see BENCHMARK_REPORT); no real large mirror, glass partition or wet floor was captured |
| W | Walk-in test: all three tiers ready to run cold on an unseen space, on the recruiters' iPhone 15 or newer | `scripts/run_capture.py`, `docs/CAPTURE_PROTOCOL.md` | one command per tier, time limits on the slow steps (D-058, D-061), offline after setup | **partial**: all three tiers run cold on unseen captures (the simulator's flats, the sample); never yet on a real iPhone 15 capture |
