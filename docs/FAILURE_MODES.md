# Failure modes: mirrors, glass, glossy floors, low light (and the generic ones)

The brief says: "Real properties contain mirrors, glass, wet-look surfaces and low light. Cover them in your submission."
This page gives, for each condition and each tier, what goes wrong, how we detect it, what we do about it, the evidence
from the sample apartment, and the risk that remains.

- Code: `floorplan/qa/surfaces.py`.
- Evidence: `scripts/analyze_failure_modes.py`, with outputs in `outputs/failure_modes/`.
- Headline numbers: `outputs/failure_modes/summary.json`.
- Module write-up: `docs/modules/failure_modes.md`.

**Data caveat.** The sample apartment was captured with LiDAR only. For the photo and video tiers, the evidence comes
from two sources:
- the RGB frames of the same captures;
- MoGe-2 single-image metric depth, the learned depth that the photo and video front-ends rely on, compared with LiDAR.

My own home capture (OwnCaptures/) will replace this proxy with native phone photos and video.

## Summary table

| Condition | LiDAR tier | Video tier | Photo tier | Detector (surfaces.py) | Handling |
|---|---|---|---|---|---|
| Mirror | Phantom depth at the reflected path length, flagged low confidence | Learned depth reconstructs the reflected room; SfM matches the reflection | Same as video, from fewer views | `classify_wall_gaps` (reflection test with a shifted-plane control), `detect_mirrors` | Exclude the phantom points; the mirror plane is a wall; widen that wall's interval |
| Glass (window, shower screen) | Pass-through or missing returns, low-confidence blobs | Depth lands on the glass or on what is behind it, inconsistently; global scale is wrong | Same as video | `lowconf_blob_mask`, `classify_wall_gaps` (sill test), `occlusion_violations` | Label the gap as glazing, not a door; exclude see-through points |
| Glossy / wet-look floor | A few low-confidence floor pixels; no sub-floor phantom in this data | Learned depth bends the floor near reflections; highlights look like features or stains | Same as video | `floor_reflection_stats`, `below_floor_points` | Exclude points below the floor; mask highlights for damage detection |
| Low light | Depth is unaffected (active sensor); the RGB used for colour and damage gets noisy | Noise and blur break matching and learned depth | Noisy photos; noise also inflates the sharpness score | `frame_quality`, `flag_frames` | Down-weight or skip frames; ask for lights on; widen intervals |
| Motion blur | Depth is unaffected; RGB is blurred | Blurred frames give fewer matches and worse depth | Rare with stills | `frame_quality`, `flag_frames`, `pick_sharpest` | Use the sharpest frame per 0.5 s window |

---

## 1. Mirrors

**What goes wrong.**
- **LiDAR.** The laser bounces off the mirror and returns from whatever the mirror shows. The depth is the length of
  the *reflected* path, so the reflected room appears *behind* the wall.
  - Sample evidence: the hall mirror at `floor_only` frame #2016, where the capturing person is visible in it.
  - The depth in the mirror region has a median of **6.1 m**, while the frame median is 2.5 m.
  - ARKit marks **82%** of those pixels low confidence and 10% medium. Only **7.8%** survive our high-confidence,
    ≤ 4 m filter (D-006).
  - Figure: `outputs/failure_modes/lidar_mirror_and_stairwell.png`.
- **Video and photo.** Learned depth and SfM have no physics that tells them a mirror is a surface. They reconstruct
  the reflected scene as real geometry behind the wall.
  - On the bathroom mirror/glass view (`with_ceiling` #4547), MoGe-2 disagrees with LiDAR by more than 10% on **34%**
    of pixels, even after its global scale is corrected.
  - Figure: `outputs/failure_modes/hard_frames.png`, row 1.

**How we detect it.** The test is geometric and works the same on any tier's point cloud, given each point's camera
centre.
1. **Seen through a wall.** `points_behind_walls` takes the viewing ray of each point and reports which wall line it
   crosses and how far behind the wall the point lies.
2. **Is it a reflection?** `reflection_consistency` reflects those points across the wall plane.
   - A mirror's phantom then lands exactly on real geometry on the camera side.
   - A real room seen through a door or glass does not.
3. **Control.** The same reflection is repeated across the plane shifted by ±25 cm.
   - A cluttered room "matches" almost anything within 5 cm, so a mirror needs a match of at least 0.5 **and** a
     margin of at least 0.25 over the control (`is_mirror`).
   - Without the control, a doorway in `single_room` (wall r0_w4) scored 0.52 and was labelled a mirror. Its control
     scored 0.48.
4. **Fallback.** `detect_mirrors` handles the case where the mirror plane itself was observed.
   - It ray-marches each point through a voxel map of the fused surface and finds points seen *through* an observed
     surface.
   - It clusters them, fits the plane they were seen through, and runs the same reflection test.

**Evidence that the detector works.**
- **Synthetic room with ground truth** (`outputs/failure_modes/synthetic.json`). A 4 x 3 m room contains a physically
  generated mirror, a glass window and a doorway.
  - Mirror: located at 1.0–2.0 m along the wall (truth 1.0–2.0). Reflection match 1.00 against a control of 0.43.
  - Window: classified as glazing, 1.0–2.6 m (truth 1.0–2.5, 0.2 m bins). Sill coverage 0.95.
  - Doorway: classified as an opening, 0.6–1.6 m (truth 0.6–1.5). Sill coverage 0.10. Reflection 0.88 against a
    control of 0.81, so it was correctly **not** called a mirror.
- **Sample apartment.** After the confidence filter, one mirror candidate remains: `floor_only`, wall r0_w8, 250 points
  (0.02% of the sample). It scored a reflection match of 0.82 against a control of 0.47. It is unverified: the frame we
  checked shows a dark glossy panel. Because D-006 already removes the hall mirror, LiDAR-tier mirrors are mostly
  pre-filtered in this data.

**What we do.**
- Exclude the phantom points.
- Treat the mirror plane as the wall surface and measure to it.
- Widen that wall's interval, because fewer real points support it.
- At capture time, the protocol says "mirrors not filling the view".

**Residual risk.**
- A mirror facing an open doorway reflects geometry that may lie out of range, so the reflection test has nothing to
  match. The fallback is the low-confidence blob plus "seen through a wall" flag, which gives a wider interval but no
  exclusion.
- A floor-to-ceiling mirrored wardrobe can be fitted as the wall with no phantom points (correct). If phantom points
  dominate, the wall moves outward. The reflection test catches that case only if the reflected room was also scanned.

## 2. Glass: windows, shower screens, glass doors

**What goes wrong.**
- **LiDAR.** Clear glass transmits the laser. ARKit returns the depth of what is *behind* the glass (the shower wall,
  the outside), often with low confidence.
  - At the shower screen (`single_room` #914), the depth passes through the glass and only the metal rail returns.
    12% of the frame is low-confidence blobs.
  - Over all frames, low-confidence *blobs* (not the thin edge lines) cover on average **4.9% / 8.8% / 7.9%** of
    pixels (`single_room` / `floor_only` / `with_ceiling`), and **16% / 37% / 33%** at the 95th percentile. These
    frames face glass, the balcony and dark glossy panels.
  - Plan maps: `outputs/failure_modes/<capture>/geometry_plan.png`, right panel.
- **Video and photo.** Learned depth is inconsistent on glass: sometimes the pane, sometimes what is behind it. Worse,
  glass and dark windows throw off the **global metric scale** of the whole frame.
  - MoGe-2, given the true field of view, under-estimates depth by a factor of **1.40** on the dark-window frame and
    **1.68** on the shower-screen frame. That is a 28–41% error.
  - After a per-frame scale correction, the error falls to **1.4–1.5%**.
  - So the *shape* survives and the *scale* does not. This is why photo and video scale must never come from a
    single learned-depth frame.
- **Plan.** A glazed section of wall has no points on its plane, so it looks like a gap. A glass shower screen looks
  like free space. A window with a sill must not be reported as a doorway.

**How we detect it.**
- `lowconf_blob_mask` finds low-confidence regions that are blobs rather than edge lines, using a morphological
  opening and a minimum area. These blobs are accumulated into a plan map (`lowconf_plan_map` in the script).
- `classify_wall_gaps` takes the points seen through each wall and groups them by *where their rays cross the wall*,
  which gives the pane or opening extent. It then runs a sill test (`_sill_coverage`):
  - wall surface observed 0.1–0.5 m above the floor beneath the gap → **glazing** (a window);
  - no wall there → **opening** (a door or passage, or possibly a glass door).
- Why a sill test rather than the height at which the rays cross the wall:
  - On the sample, the lowest crossing height was 0.34–1.1 m even for doorways. A chest-height phone rarely sends a
    ray through the bottom of a doorway to a point within LiDAR range.
  - That first version labelled 66 of 69 see-through regions as glazing, which is wrong. The sill test gives 4
    glazing, 64 openings and 1 mirror candidate over the three captures.
- `occlusion_violations` flags glass that other views saw as solid. Example: the shower screen in `floor_only` #4251,
  where 1,274 points were seen through a vertical plane (`phantom_clusters_frames.png`).

**What we do.**
- Label glazing as a window or fixed glass, not a doorway (`label_opening_as_glass`).
- Exclude the see-through points from wall fits.
- A see-through gap down to the floor stays "opening" with a widened interval and the note "door or glass door",
  because LiDAR alone cannot tell an open doorway from a clean glass door.

**Residual risk.**
- Frameless glass partitions with no sill (walk-in showers, glass doors) remain ambiguous.
- Dark windows at night reflect like mirrors (see §1).
- Learned-depth scale on glass-heavy frames is wrong by tens of percent, which the photo and video tiers must absorb
  with multi-view scale fusion.

## 3. Glossy and wet-look floors

**What goes wrong.**
- **LiDAR.** On these matte-glazed tiles, ARKit does *not* produce sub-floor reflections. Floor-bound rays that land
  below the floor plane are **0.03%** in `single_room`.
  - Larger values do appear: 0.56% in `floor_only` and 0.70% in `with_ceiling`, with 0.33–0.35% at high confidence.
  - We traced them: they are the **stairwell** going down (`lidar_mirror_and_stairwell.png`, row 2), not
    reflections.
  - The cost of the glossy floor is a modest **5–6.5%** of floor pixels at low confidence, mostly near reflected lamps.
- **Video and photo.** Specular highlights move with the camera, so feature matchers latch onto them and get wrong
  correspondences, and learned depth bends the floor.
  - On the glossy floor + balcony frame (`with_ceiling` #3897), MoGe-2 is still off by **7.2%** median after scale
    correction, and **44%** of pixels are off by more than 10%.
  - For damage detection, a highlight or a reflected lamp looks like a stain.

**How we detect it.**
- `floor_reflection_stats` works per frame. It casts every pixel ray to the levelled floor plane and counts measured
  points clearly *below* it, the low-confidence fraction on floor pixels, and blown highlights on the floor.
- `below_floor_points` works per scene. It flags points more than 10 cm under the floor.
  - Sample counts: 0 / 23,377 (0.27%) / 35,079 (0.21%) raw points, with a median of 0.31–0.36 m below the floor.
  - These are stairwell points. The flag says "stairwell or reflection", and the plan extractor decides.

**What we do.**
- Exclude below-floor points from floor and wall fits.
- Fit the floor plane robustly; it is unaffected at this level.
- Mask highlights for the damage module (`mask_for_damage_detection`).

**Residual risk.**
- A genuinely wet or polished floor (water, high-gloss marble) behaves like a horizontal mirror and returns a phantom
  room below. `below_floor_points` and the horizontal-plane branch of `detect_mirrors` catch this, but there is no
  sample to verify on.
- A stairwell that is open to the room is legitimately below the floor. It must become an opening, not be excluded
  silently.

## 4. Low light (and motion blur, its consequence)

**What goes wrong.**
- **LiDAR.** Depth is unaffected, because the sensor is active. ARKit tracking uses the camera, though, and the RGB
  that feeds colour, damage detection and the photo and video tiers degrades.
- **Video and photo.**
  - In dim light the phone raises ISO (more noise) and lengthens exposure (more blur).
  - Noise and blur remove features, so matching, pose estimation and learned depth all degrade.
  - Noise also **inflates sharpness metrics**: in our simulation, the Laplacian "sharpness" rises **2.8x** at
    16x less light and **11x** at 64x less light.

**Sample evidence.**
- The apartment was well lit: median luma is 150–154 and the 10th percentile is 117–131. No frame has more than 1%
  blown pixels.
- H.264 removes sensor noise: the estimated σ is about 0.26 grey levels. Real low light is therefore *simulated*
  (`lowlight_simulation.csv`) on sample frames, with photon (Poisson) and read noise:

  | Light | Auto-ISO off: luma | Auto-ISO off: flagged | Auto-ISO on: noise σ | Auto-ISO on: flagged |
  |---|---|---|---|---|
  | 1x | 148 | 5% | 1.4 | 5% |
  | 4x less | 78 | 8% | 2.7 | 8% |
  | 16x less | 41 | 100% | 5.7 | 100% |
  | 64x less | 22 | 100% | 13.4 | 100% |

- **Motion blur is real in the sample**, and it is driven by rotation:

  | Camera rotation | Frames flagged blurry |
  |---|---|
  | < 30°/s | 0.4–1.0% |
  | > 90°/s | 31–49% |

  Spearman correlation of relative sharpness with rotation speed: −0.39 to −0.56. Overall 5–6% of frames are blurry.
  Figures: `outputs/failure_modes/<capture>/frames_summary.png` and `frames_timeseries.png`.

**How we detect it.**
- `frame_quality` computes, per frame:
  - sharpness: Laplacian variance at a fixed 640 px long side;
  - median luma;
  - dark and clipped fractions;
  - Immerkær noise on the flatter half of the pixels.
- `flag_frames`:
  - judges blur *relative to neighbours within ±1 s*, because a white wall is never "sharp";
  - judges low light *absolutely*: luma < 60 or noise > 3 grey levels;
  - raises capture-level flags when more than 30% of frames are blurry or more than 50% are dark.
- `pick_sharpest` keeps the sharpest frame per 0.5 s window, for the video front-end.

**What we do.**
- Down-weight or skip flagged frames.
- When noise is high, do not trust sharpness, because it is inflated.
- At capture level, ask for a re-capture ("turn on all lights", "turn more slowly") and widen the intervals.
- The protocol already says "all lights on", "turn slowly" and "hold still for each shot".

**Residual risk.** The thresholds come from simulation and from physical reasoning, not from real dark captures. They
should be checked on the own-home capture.

## 5. Generic failure modes

| Failure | Tiers | What goes wrong | Detection / handling | Status |
|---|---|---|---|---|
| Textureless walls (white paint) | photo, video | Few features, so SfM and matching fail or drift; learned depth is fine on planes but scale is ambiguous | Feature count per frame (front-ends); LiDAR is unaffected | Partly handled (front-ends) |
| Clutter and furniture occluding walls | all | Wall bases hidden; furniture fronts fitted as walls; area underestimated | Walls are fitted to points spanning the full height (plan extractors); not-observed status; wider intervals where support is low | Handled by plan extractors; flagged via support counts |
| Open-plan spaces | all | No doors, so room segmentation is ambiguous | The plan extractor's segmentation; we report what the geometry supports and do not invent walls | Documented in plan extractor docs |
| Non-Manhattan walls | all | Axis snapping bends angled walls | The Manhattan score in scene_info; off-axis walls are fitted at their own angle (D-009) | Handled in alignment |
| Tiny rooms (WC, closet) | photo, video | Too close to walls: no parallax, near-range depth errors, the room does not fit in a frame | Minimum-distance check (median depth < 0.5 m) → re-capture advice | Advice only |
| Doors closed during capture | all | Adjacency missed; door planes fitted as walls | Two passes that disagree show up as `inconsistent_geometry` (§6); the protocol says doors open | Detection partial |
| People or pets moving | video, LiDAR | Ghost geometry; matches on moving people | `occlusion_violations` (points seen through surfaces seen in other frames) → `inconsistent_geometry`, excluded | Detected (same test) |
| Phone at different heights (photos) | photo | Camera-height scale prior broken; ceiling or floor lines missing | The protocol fixes chest height; scale comes from multi-view learned depth, not from a height prior alone | Protocol |
| Pose drift over long multi-pass captures | LiDAR, video | Two passes give two copies of a wall | `occlusion_violations` cluster rate grows with capture length (§6); the drift module corrects it | Measured here; fixed by drift module |

## 6. Consistency check: points seen through surfaces

`detect_mirrors` / `occlusion_violations` ray-march every point (400k sampled per capture, after the D-006 filter)
through a 5 cm voxel map of the fused surface. Points seen *through* an observed surface are physically impossible and
are flagged.

| Capture | Duration / passes | Points seen through a surface | Clusters |
|---|---|---|---|
| single_room | 37 s, one pass | **1.1%** | 3 small planar see-through (173–580 points, at glossy bathroom and wardrobe panels; not verified individually), 2 non-planar, 1 small horizontal (220 points) |
| floor_only | 115 s, loop | **9.5%** | 7 non-planar, 1 glass shower screen |
| with_ceiling | 215 s, two passes | **5.0%** | 7 non-planar |

- The clusters in the longer captures do **not** lie on a plane: crossing-plane RMS is 0.2–0.4 m against a 5 cm
  threshold. They are therefore not mirrors or glass. They are two observations of the scene that disagree.
- The rate is 1.1% for the single 37 s pass and 5–9.5% for the long multi-pass captures. This is consistent with pose
  drift between passes, which is the drift module's job. We report this as a correlation, not a proven cause.
- These points are tagged `inconsistent_geometry`, to be excluded and the nearby walls' intervals widened.
- Figures: `outputs/failure_modes/<capture>/geometry_plan.png` and `phantom_clusters_frames.png`, where each cluster
  is projected on the frame that produced it.
