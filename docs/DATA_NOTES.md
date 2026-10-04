# What the sample data is (and what it is not)

Source: `TakeHome/Dataset/`, provided with the take-home email ("Please run your code on this Sample Data").
Evidence for every statement below comes from `scripts/explore_capture.py`. The outputs are in
`outputs/explore/<capture>/`: `overview.png`, `contact_sheet.jpg`, `height_hist.png` and `stats.json`.

## Format: a Stray Scanner export (iOS LiDAR app)

Each capture folder contains:

| File | Content | Notes |
|---|---|---|
| `rgb.mp4` | 1920x1440 H.264 colour video, recorded in portrait (frames appear rotated 90°) | ~46 fps on average; the frame count equals the depth frame count |
| `depth/NNNNNN.png` | 256x192 uint16 depth in **millimetres**, one per frame | LiDAR depth from ARKit `sceneDepth` |
| `confidence/NNNNNN.png` | 256x192 uint8, 0 = low, 1 = medium, 2 = high | ARKit's per-pixel depth confidence |
| `camera_matrix.csv` | 3x3 RGB intrinsics (fx = fy ≈ 1600 px, cx ≈ 955, cy ≈ 718) | one matrix per capture |
| `odometry.csv` | per frame: timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy | ARKit camera pose plus **per-frame** intrinsics |
| `imu.csv` | ~100 Hz accelerometer (a_x..a_z, in g) and gyroscope (alpha_x..alpha_z, rad/s) | not needed while ARKit poses exist |

Facts established by measurement, not assumption:

1. **Pose convention: camera-to-world, OpenCV camera axes** (x right, y down, z forward).
   - Test: re-project frame *i*'s depth into frame *i+15* and *i+45*, then measure the depth disagreement.
   - Under the OpenCV interpretation the median disagreement is **6.4 / 6.9 / 7.0 mm** across the three captures.
   - Under the ARKit-axes interpretation (y up, z back) it is 22–32 cm.
   - So Stray Scanner already converted ARKit's axes, and **no flip is needed**.
2. **World frame is gravity-aligned with +y up.**
   - The floor-plane normal is within 0.6° / 0.2° / 0.02° of the y axis.
   - The origin is wherever the phone started.
   - The heading (yaw) is arbitrary per capture, so different captures of the same rooms are rotated relative to each other.
3. **Depth intrinsics are the RGB intrinsics scaled by 256/1920 = 0.1333.** The depth map is the LiDAR map registered to the RGB camera.
4. **Depth confidence is mostly high:** 89–94% of pixels are "high" (2). We fuse only high-confidence depth (see D-006).
5. **Tracking is continuous.**
   - Median frame interval is 1/60 s.
   - The only gaps are single 50 ms gaps (2 dropped frames), with no relocalisation jumps.

## The three captures

The three were recorded back-to-back on the same device within about 6 minutes; the device-uptime timestamps are
65575 s, 65621 s and 65764 s. They show the **same apartment**.

| Capture | Duration | Camera path | Reconstructed extent | Ceiling seen? | What it shows |
|---|---|---|---|---|---|
| `single_room/c00a170fe1` | 37 s | 14.5 m | 9.2 x 6.9 m (the LiDAR sees through doors) | no | Living area (sofa, fridge, wardrobe), white cabinets, a bathroom with a glass shower screen, a dark glass window and tiles. Despite the name, it spans several connected spaces. |
| `single_scan_floor_only/1a8384c3f6` | 115 s | 54 m | 13.0 x 13.4 m | no (phone aimed down; median depth 1.1 m) | Whole apartment: a central corridor and about 6–8 rooms. Starts and ends at the same spot, which gives a loop closure. |
| `single_scan_with_ceiling/c7d28f72c6` | 215 s | 100 m | 12.9 x 14.1 m | yes; floor-to-ceiling ≈ 3.08 m (rough, whole capture) | The same apartment, walked roughly **twice**: a first pass and a second pass through most rooms. This gives many loop closures, the best data for the drift ablation. |

## What this means for the design

- **This is LiDAR-tier data**: depth, poses and intrinsics from a Pro-class device. The photo and video tiers have no native
  sample, so they will be **derived** from the same captures.
  - Video tier: `rgb.mp4` alone, ignoring depth and poses.
  - Photo tier: 2–8 stills per room sampled from the video.

  Deriving them means all three tiers see the same rooms, so the LiDAR result can act as a reference for the other two.
  This is disclosed everywhere it is used.
- **There is no ground truth** (no tape or laser measurements). Absolute accuracy cannot be verified on this data. We use:
  - Repeatability: `floor_only` and `with_ceiling` cover the same rooms, so per-wall agreement can be measured.
  - Internal consistency: loop-closure residuals, opposite-wall parallelism, wall-thickness consistency.
  - An external laser-ground-truth check on a public dataset (ARKitScenes, Faro scanner) for the LiDAR pipeline itself.
- **Ceiling height:**
  - Only `with_ceiling` observes the ceiling.
  - For `floor_only` and `single_room` the pipeline must say "not observed" and widen the interval honestly, not invent a value.
- **Glass and mirrors are present:** the shower screen, a dark glass window and the fridge door. Expect missing or wrong LiDAR
  returns there; covered in the failure-modes section.
