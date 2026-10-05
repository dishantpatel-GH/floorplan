# Documents: what each one is for, in reading order

How to install and run the pipeline is in the top-level `README.md`. The pages here explain how to capture, what the
pipeline delivers, how it was measured and why it is built the way it is.

## 1. Capture and hardware

| Document | What it is for |
|---|---|
| [CAPTURE_PROTOCOL.md](CAPTURE_PROTOCOL.md) | **The one page a non-engineer follows** (Part 1, Route 2): what to install, how to walk, what to avoid, where the files go in `data/` and the one command per tier |
| [DEVICE_MATRIX.md](DEVICE_MATRIX.md) | Which tier (photos, video, LiDAR) runs on which phone, where its scale and poses come from, and where its measured accuracy is |

## 2. Results

| Document | What it is for |
|---|---|
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | The ≤ 6-page report (Deliverable 7): architecture, tiers, drift, error budget, calibration, fix loop, failure modes |
| [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) | The benchmark (Deliverable 5): gates at all three tiers, repeatability, head-to-head with CubiCasa, timing, plan pictures; regenerated from fresh runs of the final code |
| [FIX_LOOP.md](FIX_LOOP.md) | Part 4: the declaration (worst gate, root cause, predicted number), the shipped fix, before and after, the post-mortem. With [fixloop_ranking.md](fixloop_ranking.md) (how the worst gate was picked) and [fixloop.diff](fixloop.diff) (the readable diff between the tags `before-fix` and `after-fix`) |
| [COMPLIANCE.md](COMPLIANCE.md) | Deliverable 1: every requirement of the brief → file → artifact → status, deviations included |

## 3. Honesty pages

| Document | What it is for |
|---|---|
| [DISCLOSURES.md](DISCLOSURES.md) | Every pretrained model, dataset, library and app used, with licences and commercial-use flags; the head-to-head app (CubiCasa 3.14.1) and the Part 3 deviation |
| [FAILURE_MODES.md](FAILURE_MODES.md) | Mirrors, glass, glossy floors, low light: what goes wrong per tier, how it is detected and handled, what risk remains |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | The recruiters' answers (3 Oct) that set the scope: own captures are mandatory, the sample data stands in for LiDAR |
| [DATA_NOTES.md](DATA_NOTES.md) | What the recruiters' sample captures are (Stray Scanner format, conventions checked) and what they are not |

## 4. Decision log

| Document | What it is for |
|---|---|
| [DECISIONS.md](DECISIONS.md) | Every design decision, D-001 to D-088, in the order it was made: context, decision, evidence, limits. The final defaults are D-086 (photo: a room side sits on the wall, not on a pillar face), D-087 (video: scale step `"auto"`) and D-088 (photo: door stitching on, judged on k65) |
| [ISSUES.md](ISSUES.md) | Integration issues: symptom → root cause → fix → status |

## 5. Module notes (`modules/`)

How each part works, the options tried and the evidence, one note per module. They grew with the work, so the newest
section of a note is at its end; a decision in `DECISIONS.md` overrides an older section.

| Note | Module |
|---|---|
| [video_tier.md](modules/video_tier.md) | video front-end: frames, DPVO path, scale step, segments |
| [photo_tier.md](modules/photo_tier.md) | photo front-end: per-room spins, ceiling photos, room boxes, door stitching |
| [plan_beta.md](modules/plan_beta.md), [plan_alpha.md](modules/plan_alpha.md) | plan extraction (beta is the default; alpha is the fallback when a plan step times out) |
| [openings.md](modules/openings.md) | doors and windows from the segmenter, door priors, see-through doorways |
| [drift.md](modules/drift.md) | drift correction for the LiDAR tier (pose graph, loop closures, plane anchors) |
| [lidar_bias.md](modules/lidar_bias.md), [arkitscenes_validation.md](modules/arkitscenes_validation.md) | LiDAR wall bias and the laser check on ARKitScenes |
| [damage.md](modules/damage.md) | damage regions, concealed-damage rules, scope items |
| [eval.md](modules/eval.md), [benchmark_final.md](modules/benchmark_final.md) | registration, matching, repeatability, gates, the report generator |
| [export.md](modules/export.md) | JSON contract, rendered plan, DXF |
| [failure_modes.md](modules/failure_modes.md) | the detectors behind `FAILURE_MODES.md` |
| [scale_reference.md](modules/scale_reference.md) | the paper-sheet scale cue (built, off by default since D-067) |
| [sim.md](modules/sim.md) | the simulator (more in `sim/README.md`) |
| [plan_v2_ablation.md](modules/plan_v2_ablation.md), [judge_plan_extractor.md](modules/judge_plan_extractor.md), [benchmark_lidar_sample.md](modules/benchmark_lidar_sample.md), [benchmark_arkitscenes_accuracy.md](modules/benchmark_arkitscenes_accuracy.md) | dated evidence (4 Oct) behind D-018 and the LiDAR results; the current numbers are in `BENCHMARK_REPORT.md` |

## 6. The own benchmark capture

| Document | What it is for |
|---|---|
| [HOUSE_CAPTURE_GUIDE.md](HOUSE_CAPTURE_GUIDE.md) | The checklist used on the phone on 4 Oct: the capture protocol plus tape measuring and the app scan |
| [OWN_CAPTURE_RUNBOOK.md](OWN_CAPTURE_RUNBOOK.md) | How the own capture was checked, run and scored |
| `../benchmark/ground_truth.csv` | The empty template for tape ground truth; the own flat's filled copy is `data/own_house/gt/ground_truth.csv` |

## 7. Archive (`archive/`)

Working notes a reviewer does not need, kept because the decision log links to them: the build plan
(`PLAN.md`), the first benchmark-capture plan (`BENCHMARK_CAPTURE_PLAN.md`, now inside `COMPLIANCE.md`), the
research behind the capture protocol (`CAPTURE_PRACTICES_RESEARCH.md`) and the layout methods
(`LAYOUT_METHODS_RESEARCH.md`), and two dated video notes from 5 Oct (`notes/`), whose conclusions are D-087.
