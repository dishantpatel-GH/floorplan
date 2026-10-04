# Build plan and status board

Deadline: **5 Oct 2026, 19:00 IST**. The build in a scratch folder should be complete by about 4 Oct afternoon.
That leaves time to move it into this repo (README, History) and write the 6-page report.

## Architecture: one back-end, three front-ends

```text
 LiDAR tier (Stray Scanner)     Video tier (rgb.mp4 only)          Photo tier (2-8 stills per room folder)
  depth + confidence + poses     SfM / 3D foundation model +         per-room metric reconstruction from photos
        |                        metric depth -> poses + depth        (learned metric depth / multi-view model)
        v                                |                                   |
  [1] SCENE: keyframes -> drift correction (pose graph) -> TSDF fusion -> raw points
            -> gravity levelling + Manhattan alignment                <- every front-end produces this
        |
  [2] PLAN EXTRACTION: wall evidence -> rooms + walls (fitted on raw points) -> openings (doors, windows)
            -> per-room ceiling height -> wall thickness -> adjacency
        |
  [3] STITCH: rooms in one frame (LiDAR/video: one continuous capture; photo: cross-room registration)
        |
  [4] UNCERTAINTY: an error budget per measurement -> 95% intervals (wider for thinner input)
        |
  [5] DAMAGE + SCOPE: damage regions on surfaces (class, metric extent) -> concealed-damage rules -> line items
        |
  [6] EXPORT: JSON (schema/plan.schema.json) + rendered plan (SVG/PNG) + DXF
        |
  [7] EVALUATION: repeatability, drift ablation, cross-tier agreement, ARKitScenes laser check -> gates table
```

## Status (updated 4 Oct, 01:15)

| Milestone | What | Status |
|---|---|---|
| M0 | Data exploration, conventions verified | ✅ |
| M1 | Stage-1 scene builder; measured noise model; pixel-centre intrinsics | ✅ |
| M2 | Plan extraction: two approaches + judge → plan_beta hybrid | ✅ judge done (D-018); v2 in progress (wave 4) |
| M3 | Drift correction + ablation | ✅ (revisit separation 14 → 1.4 cm on floor_only; canonical scenes v2 drift ON) |
| M4 | Export: JSON schema, render, DXF | ✅ |
| M5 | Evaluation harness: registration, repeatability, gates | ✅ |
| M6 | ARKitScenes laser check | ✅ (surface 0.96 cm; wall pairs short-biased, investigated in wave 3) |
| M7 | Video tier | 🟡 v1 fine on short clips, fails on long multi-room clips → v2 in progress (wave 5) |
| M8 | Integration: one command per capture | 🟡 LiDAR end to end ✅; video/photo/damage hooks being wired |
| M9 | Photo tier | 🟡 v1 links 3–4 of 9 rooms → protocol revised (D-017) + v2 in progress (wave 3) |
| M10 | Damage, rules, scope | 🟡 v1: 0 false positives on LiDAR, stains good, cracks short → v2 (wave 3) |
| M11 | Benchmark + fix loop (Part 4) | ⏳ after plan v2; candidate: repeatability (B-1) |
| M12 | README, disclosures, commit plan, report draft | 🟡 docs agent (wave 4) |
| M13 | Own capture (OnePlus Nord, tape GT) → photo/video benchmark + head-to-head | ⏳ 4 Oct morning |

## Scope after the recruiters' reply (D-012)

- **Part 1:** Route 2 protocol (`docs/CAPTURE_PROTOCOL.md`), revised by evidence (D-017).
- **Own benchmark:** photo and video tiers on my home (OnePlus Nord, tape ground truth) on 4 Oct
  morning. LiDAR uses the sample data, with the recruiters' permission.
- **Part 3:** a consumer app (magicplan) on 2 own rooms, compared at the photo/video tier. This is a disclosed
  deviation: no LiDAR device is available for own rooms.
