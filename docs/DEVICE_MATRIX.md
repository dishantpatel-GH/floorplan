# Device matrix (Part 1 deliverable): which tier runs on which hardware, and what accuracy it honestly delivers

Every accuracy cell comes from a measurement in this repo; "pending" cells are filled from `outputs/benchmark` and
`outputs/own_eval`. Each figure is labelled with the evidence it comes from.

| Tier | Hardware it runs on | Capture app | What the pipeline gets | Tested on | Honest accuracy (evidence) |
|---|---|---|---|---|---|
| **LiDAR** | iPhone 12 Pro or newer (Pro / Pro Max), iPad Pro 2020 or newer | Stray Scanner (free, App Store) | metric depth 256x192 + confidence + ARKit poses + intrinsics | Provided sample (iPhone, Stray Scanner, 3 captures); ARKitScenes 47895909 (iPad Pro + FARO laser GT) | ARKitScenes vs laser: surface median 0.96 cm; room box +0.25 / −1.52 cm, height −1.92 cm; wall-pair median 1.5 cm. Repeatability on the sample: pending |
| **Video** | Any phone with a camera: iPhone 15 or newer (walk-in), OnePlus Nord (own benchmark), any Android | Native Camera app (1080p30, stabilisation off where possible) | RGB frames only; scale from learned metric depth + focal metadata + priors (no reference object, D-067) | Own home (OnePlus Nord, tape GT); sample `rgb.mp4` vs the LiDAR reference | pending (gate ±3%) |
| **Photos** | Any phone: iPhone 15 or newer (walk-in), OnePlus Nord (own benchmark) | Native Camera app (1×, JPEG/HEIC with EXIF) | 2–8 stills per room folder + EXIF focal length; scale from learned metric depth + priors (no reference object, D-067) | Own home (OnePlus Nord, tape GT); simulated folders from the sample vs the LiDAR reference | pending (gate ±8%) |

Notes:
- **Android cannot run the LiDAR tier.** Stray Scanner is iOS-only, and no Android phone has a comparable LiDAR.
  Android phones run the photo and video tiers.
- **What drives accuracy per tier:**
  - LiDAR: sensor noise (measured: 7 mm per point at 1–2 m) and residual drift. Scale is metric.
  - Video and photos: the scale estimate dominates. No reference object is used (D-067), so the learned-depth bias
    (1σ 4%) and the focal-length error set the scale term.
