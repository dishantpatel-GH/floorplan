# Device matrix (Part 1): which tier runs on which hardware, and how accurate it honestly is

The accuracy column points to `docs/BENCHMARK_REPORT.md`, which is regenerated from fresh runs of the final code;
this page does not repeat numbers that a re-run can change. Every tier writes the same `plan.json` contract, and its
intervals widen as the input thins (`floorplan/uncertainty/tier_budget.py`).

| Tier | Capture hardware | Capture app | What the pipeline gets | Benchmarked on (ground truth) | Honest accuracy |
|---|---|---|---|---|---|
| **LiDAR** | iPhone 12 Pro or newer (Pro / Pro Max), iPad Pro 2020 or newer | Stray Scanner (free, App Store) | metric depth 256×192 + confidence, ARKit poses, intrinsics, `rgb.mp4` | the recruiters' 3 Stray Scanner captures (repeat pair: `floor_only` vs `with_ceiling`); 5 ARKitScenes rooms (FARO laser) | `BENCHMARK_REPORT.md`, LiDAR rows: laser-scored walls and ceilings, repeatability, drift on/off |
| **Video** | any phone camera: iPhone 15 or newer (walk-in); OnePlus Nord (own benchmark); Android works | built-in Camera, 1×, 1080p 30 fps | RGB frames and the file's lens metadata only | own flat, `take1` (tape); the 3 sample `rgb.mp4` files against the sample LiDAR plan | `BENCHMARK_REPORT.md`, video rows (gate: walls within ±3%) |
| **Photos** | any phone camera: iPhone 15 or newer (walk-in); OnePlus Nord (own benchmark) | built-in Camera, 1×, 4:3, EXIF kept | 2–8 stills per room folder, EXIF focal length and capture times | own bedroom, lit and dim sets (tape); simulated k65 flat in iPhone 15 format (exact geometry) | `BENCHMARK_REPORT.md`, photo rows (gate: walls within ±8%, stitched footprint within ±8%) |

**How each tier gets its scale and its poses** (the main error terms):
- **LiDAR:** scale and poses from the sensor. Residual drift is corrected by a pose graph with loop closures and
  plane anchors (`docs/modules/drift.md`); a measured inward bias of the wall surfaces is removed (D-021, D-027,
  `docs/modules/lidar_bias.md`).
- **Video:** an up-to-scale camera path from DPVO (the default `path_source`; a global SfM path is an option, used
  only when one model holds 80% of the walk). Scale per segment: step `"auto"` (D-087), PnP votes on MoGe-2 metric
  depth where a segment's own votes cover half of it, else agreement with MoGe-2 depth. No reference object (D-067), so the
  learned-depth bias and the focal-length error set the scale term, and the intervals carry it.
- **Photos:** each room is fitted from its own spin (MoGe-2 depth, the EXIF focal length, a ceiling prior where the
  ceiling photo is weak, D-057, D-074); a room side sits on the wall, not on a pillar face (D-086); rooms are joined at
  the doors they share (door stitching on, D-088) and by the doorway photos. Doorways seen through are openings
  (`docs/modules/openings.md`).

**Limits that hold whatever the numbers:**
- **Android cannot run the LiDAR tier:** Stray Scanner is iOS only, and no Android phone has a comparable LiDAR.
- **Processing hardware:** all benchmark runs used one 8 GB NVIDIA GPU. The video tier's DPVO path needs CUDA
  (compiled extensions, `setup/dpvo_setup.sh`); the timings per tier are in `BENCHMARK_REPORT.md`.
- **Lens:** the 1× main lens only. The 0.5× ultra-wide was tried and dropped (D-065, D-068).
- **The photo tier is benchmarked on the own flat and the simulated k65 flat only.** The recruiters' sample has no
  photo capture, so it runs only the video and LiDAR tiers. Simulated numbers are labelled as such and are not real
  accuracy.
