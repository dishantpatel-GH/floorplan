# Layout methods for the photo tier: walls behind furniture, L-shaped rooms, glass, scale, ceiling height

Research date 2026-10-04. Web and code reading only: nothing was installed and no pipeline code was changed.

**Status note (D-067, 4 Oct evening).** No reference object is used in any capture. The AprilTag and A4-sheet items
below (Pick 3, §1.3) are background only; the view-type-weighted scale fusion still applies.

**Scope.** The five failure modes measured on the simulated houses (Isaac Sim, InteriorAgent scenes) for the photo tier:
- input: 2–8 iPhone photos per room;
- per photo: MoGe-2 depth and normals at 518 px, gravity from GeoCalib refined by the floor;
- poses: ALIKED + LightGlue pose graph;
- room shape: one Manhattan rectangle per room (`floorplan/photo/layout.py`).

**Legend.**
- **(U)**: not confirmed on a fetched primary page.
- **[AUTH]**: reported by the method's own authors.
- **(derived)**: our own arithmetic.

**Related documents.** This builds on `../../floorplan_capture_landscape.md` (§4 layout, §5.D depth, §5.G scale) and `CAPTURE_PRACTICES_RESEARCH.md`. Numbers repeated here were re-checked against the primary source.

## 0. The short answer

**How the ranking was made.** Expected error removed on the measured failures, times confidence, divided by integration cost.

**Shared rule.** Every pick keeps the repo's rule "models propose, geometry measures" (D-004). A learned model only decides *which* points belong to a wall; every dimension is still fitted to MoGe points.

| # | Method | Fixes | Effort | Weights / licence | First test |
|---|---|---|---|---|---|
| 1 | **Semantic wall gating.** EoMT-L (ADE20K) labels each photo. Only `wall` pixels feed the side fit. Furniture, glass, mirror and door classes are removed | 1 (furniture fronts), 3 (mirrors, windows), doors | ~0.5 day (own env) | `tue-mps/ade20k_semantic_eomt_large_512`, MIT, HF | Oracle run with the simulator's per-pixel classes, then the model |
| 2 | **Measure walls above the clutter.** <br>• Wall band up to the ceiling. <br>• Ceiling-reach test. <br>• Ceiling-extent bound. <br>• Spin with the 0.5× lens so both wall junctions are in view. <br>• Ceiling height from the floor/ceiling ratio inside one photo | 1, 5 (ceiling height) | ~0.5 day + protocol change | none | Simulator A/B: 0.5× vs 1× spin |
| 3 | **Exact scale wherever possible.** (D-067: the AprilTag and tag-photo items are withdrawn; only the view-type fusion stays.) <br>• AprilTag printed on the A4 sheet, detected at full resolution. <br>• Scale fusion weighted by view type. <br>• Camera height calibrated from the tag photos | 4 (scale), 5 | ~0.5 day | `pupil-apriltags` (already installed) | Detection rate on light floors |
| 4 | **Rectilinear room polygon from free space and wall evidence** (Cabral & Furukawa). <br>• Unseen sides are flagged, never mirrored. <br>• A second spin in L-shaped rooms. <br>• PolyLayout (installed) as a cross-check | 2 (L-shapes), 1 | ~1 day | none (PolyLayout Apache-2.0) | The two L-shaped living-dining rooms |
| 5 | **Glass and mirror handling.** <br>• Masks from the ADE mirror/window classes plus SAM 3 text prompts ("glass shower screen", "glass door", "mirror"). <br>• Masked points never vote. <br>• A ring around each mirror or window frame gives the wall plane. <br>• Shower screens become interior partitions | 3 (glass, mirrors) | ~0.5–1 day | SAM 3 (HF, gated, SAM licence) or GEM (HF, Apache-2.0); the rules themselves need no weights | Oracle masks from the simulator's glass and mirror flags |

**Not worth the time now** (evidence in §2):
- **Multi-view feed-forward models as the scale source.** On ScanNet, zero-shot relative depth error is 22.2% for MapAnything (multi-view, images only) against 10.6% for MoGe-2 on a single image.
- **Cuboid perspective layout nets** (LSUN-style). An LSUN-trained net (Lin et al. 2018) reaches about 26% pixel error on generic real rooms (RealEstate10K), against 6–7% on LSUN/Hedau.
- **SpatialLM on reconstructed clouds**: zero-shot layout F1@0.25 is 47–69%.
- **Plane-DUSt3R** out of domain: Chamfer distance 23 m on ASE.
- **Pano2Room**: a view-synthesis method, not a layout method.

## 1. Ranked shortlist

### 1.1 Semantic wall gating (EoMT-L or OneFormer, ADE20K)

**Fixes.**
- Failure 1: wardrobe, cabinet, counter and vanity fronts taken as walls.
- Part of failure 3: the mirror and windowpane classes.
- Far walls seen through doors.

**Key idea.**
- Today a wall point is "the farthest point per column with a horizontal normal in the 1.0–2.0 m band", and a nearer long peak wins. A wardrobe front passes all three tests.
- A per-pixel class decides instead: only `wall` pixels can vote for a wall, and occluder classes are removed.
- Glass, mirror, window and door pixels are removed **and** excluded from "farthest per column". In those columns the view passes through, or into a reflection, so they make rooms too *big*.

**Evidence.**
- **EoMT-L** (DINOv2, 512 px) scores 58.4 mIoU on ADE20K val, with 319M parameters at 92 FPS on an H100. ViT-Adapter-L + Mask2Former scores 58.9 at 21 FPS ([EoMT, CVPR 2025, Table 5](https://arxiv.org/html/2503.19108)) [AUTH].
- **OneFormer Swin-L** scores 57.0 / 57.7 (single-scale / multi-scale) ([repo](https://github.com/SHI-Labs/OneFormer)).
- **Per-class IoU.** None has been published for these checkpoints. The closest public evaluation log is ViT-Adapter-L + Mask2Former at 58.3 mIoU ([log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beit_adapter_large_640_160k_ade20k_ss.log)):
  - usable: wall 82.2, floor 85.7, ceiling 87.6, cabinet 66.2, wardrobe 57.9, door 59.1, windowpane 66.6, mirror 78.1, countertop 67.6;
  - weak: counter 54.9, shower 12.3.
  - In a related run (ViT-Adapter-L BEiTv2, 61.2 mIoU), rare classes swing by up to 20 points across its 8 evaluation checkpoints (shower 3.6–24.4), while wall stays at 82.6–83.7.
- **Glass is not a usable class.** ADE20K's "glass" means a drinking glass, and there is no glass-partition or shower-screen class ([objectInfo150.csv](https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv)).
- **Clutter filtering helps layout.**
  - HouseLayout3D splits OneFormer labels into structural, "geometrically inaccurate" (windows and mirrors: "these often suffer from poor depth estimates and are therefore excluded from the skeleton") and objects. Its structural "layout skeleton" doubles Avg F1 over the raw mesh, from 0.109 to 0.223 ([arXiv 2512.02450](https://arxiv.org/html/2512.02450)).
  - Coarse Semantic Injection raises Structured3D layout F1@0.5 from 0.498 to 0.599, but with ground-truth semantics ([arXiv 2605.16832](https://arxiv.org/html/2605.16832)).
  - Apple RoomPlan detects walls from a *semantically labelled* point cloud ([Apple ML research](https://machinelearning.apple.com/research/roomplan)).
- **Known weakness.** "Off-the-shelf semantic segmentation often fails in distinguishing free-standing and built-in structures like wardrobes" ([Matterport](https://matterport.com/blog/all-we-want-is-an-empty-room-panorama-inpainting)). Pick 2's ceiling-reach test is the backstop for this.

**Code, weights, licence.**
- **Use:** `tue-mps/ade20k_semantic_eomt_large_512` (HF, MIT, 1.26 GB in fp32), loaded with `EomtForUniversalSegmentation` + `AutoImageProcessor` from transformers ≥ 4.54 ([docs](https://huggingface.co/docs/transformers/main/en/model_doc/eomt)).
- **Without `transformers` (U, not tried).** The official EoMT repo builds the DINOv2 backbone with `timm` (`vit_large_patch14_reg4_dinov2`), and timm is already in the main env. Copying `models/vit.py`, `eomt.py` and `scale_block.py` plus the three sliding-window helpers would avoid the extra env. Whether the HF `pytorch_model.bin` matches the repo's key format is unchecked.
- **Fallback:** `shi-labs/oneformer_ade20k_swin_large` (MIT; `OneFormerForUniversalSegmentation` with the task "semantic").
- **Avoid:**
  - Mask2Former ADE weights (CC BY-NC).
  - SegFormer (NVIDIA research-only licence).
  - DINOv3-based EoMT, 59.5 mIoU: deltas on top of gated DINOv3 weights.

**Compute.** Not measured on an 8 GB GPU. Estimate 0.05–0.2 s per image and under 3 GB in fp16 (U). A room has at most 8 photos, so seconds per room.

**Integration.**
1. **Own environment.** Create `envs/seg` with torch from the PyTorch index plus `transformers>=4.54`. The main env must not get `transformers` (requirements.txt, I-002).
   - A worker `scripts/seg_worker.py <photos…>` writes one uint8 ADE label PNG per photo.
   - It labels the same EXIF-rotated image that MoGe sees.
2. **Labels on the view.** `PhotoView` gains `sem`: the label map nearest-resampled to the 518 px depth grid.
3. **Class groups** (HF 0-based ids). Map by index: label strings differ between repos, e.g. OneFormer's "window " with a trailing space.

   | Group | Classes (id) |
   |---|---|
   | WALL | wall 0 |
   | FLOOR | floor 3, rug 28 |
   | CEILING | ceiling 5 |
   | WALL-ATTACHED (optional) | painting 22, poster 100, bulletin board 144 |
   | GLASS / MIRROR | mirror 27, windowpane 8, screen door 58, shower 145, glass 147 |
   | DOOR / OPENING | door 14, stairs 53, stairway 59, railing 38, bannister 95 |
   | OCCLUDER | everything else: cabinet 10, wardrobe 35, chest of drawers 44, counter 45, countertop 70, kitchen island 73, shelf 24, bookcase 62, refrigerator 50, stove 71, oven 118, dishwasher 129, washer 107, sink 47, bathtub 37, toilet 65, curtain 18, blind 63, bed 7, sofa 23, … |
4. **In `fit_room_layout`.**
   - Set `wall &= sem ∈ WALL`, after eroding the wall mask by 2 px wherever it touches another class (edge pixels mix depths).
   - Run `_farthest_per_column` on WALL points only. A column where furniture hides the whole band then contributes nothing, instead of voting at the furniture front.
5. **In `_fit_side`.**
   - A peak whose inliers are less than 80% WALL is not a wall.
   - `layout_prefer_near_ratio` (the through-door rule) compares WALL peaks only. DOOR pixels are already out.
6. **Fallback.** Report the WALL fraction per side. Below about 30% of candidate points, mark the side "semantics unreliable" and fall back to today's rule with a wider sigma.
7. **Later, other tiers.** The same labels, voted per voxel across keyframes, can gate video and LiDAR walls. The LiDAR benchmark also reads bathroom cabinet fronts as walls (`docs/modules/arkitscenes_validation.md`).

**Test first.**
- **Oracle.** The simulator knows each triangle's class (`sim/scene_geom.py`) and already ray-casts every photo (`sim/emulate.py`). Write the class of the first hit for every pixel next to each photo, and run the fit with that oracle mask. That is the ceiling of what Pick 1 can deliver, with no model risk.
- **Then the model.** Swap in EoMT and score it against the oracle: wall precision and recall per photo, and the room-dimension error.

### 1.2 Measure walls above the clutter (upper wall band, ceiling-reach test, both junctions in view)

**Fixes.**
- Failure 1: the wall's position where Pick 1 has only removed the wrong surface.
- Failure 5: ceiling height without the tilted-up photo's own depth scale.
- Helps failure 4: fewer blank ceiling-only photos.

**Key idea.** The ceiling-wall junction is rarely occluded. Layout methods measure walls there:
- HorizonNet projects the ceiling-wall boundary onto the ceiling plane and lets each point vote for wall planes within 0.16 m ([paper](https://arxiv.org/abs/1901.03861)).
- LayoutNet down-weights wall-floor corners because they "are often occluded" ([paper](https://arxiv.org/abs/1803.08999)).

**What we do today, and why it fails.**
- Our band (1.0–2.0 m) discards exactly that region, so the farthest point per column lands on the wardrobe front.
- Vanities, counters, partitions and many wardrobes stop below the ceiling, while the wall continues to it.
- Full-height built-ins are the exception; Pick 1 and the inter-room gap check (Pick 4) cover them.

**Four rules.**

| Rule | What it does | Precedent |
|---|---|---|
| **Upper band** | Wall points (Pick 1 labels) from 2.0 m up to 5 cm below the ceiling | Apple RoomPlan keeps 12 height slices of 30 cm ([Apple ML](https://machinelearning.apple.com/research/roomplan)); FloorSAM keeps the highest points per cell "to filter out low objects (e.g., furniture)" ([arXiv 2509.15750](https://arxiv.org/html/2509.15750)) |
| **Ceiling-reach test** | A candidate vertical surface whose top stops more than about 25 cm below the ceiling, with wall or ceiling visible beyond it, is furniture or a partition | Apple patent: a wall is rejected as "not a floor-to-ceiling wall (e.g., cubicle wall)" when its height "does not meet a height threshold" relative to the ceiling ([US20210225043A1](https://patents.google.com/patent/US20210225043A1/en)) |
| **Ceiling-extent bound** | No wall can be nearer than the farthest observed ceiling point (minus 5 cm) | — |
| **Read depth, not angles** | Take the MoGe points at the junction, not the junction angle intersected with a plane. That intersection costs 16 cm per degree of pitch at 3 m (derived) | — |

**Capture change that makes it work** (derived, camera at 1.40 m, ceiling at 2.50 m):

| Lens and orientation | Floor-wall line visible for walls beyond | Ceiling-wall line visible for walls beyond |
|---|---|---|
| Today: 1× landscape, tilted 10° down | 1.89 m | **3.71 m** |
| 1× landscape, level | 2.80 m | 2.20 m |
| 1× portrait, level | 2.10 m | 1.65 m |
| **0.5× landscape, level** | 1.40 m | 1.10 m |
| 0.5× landscape, tilted 10° down | 0.98 m | 1.57 m |

- **Lens geometry.** The 0.5× lens on an iPhone 15 is 13 mm equivalent with a 120° field of view ([Apple specs](https://support.apple.com/en-us/111831)). On a 4:3 photo that is 106° horizontal × 90° vertical (derived).
- **Fewer photos.** It needs about 4 photos for 360° instead of 5–6, which frees slots in the 8-photo budget.
- **D-017 needs a test, not a re-read.** It rejected 0.5× with "strong distortion and lower quality hurt metric depth". That has not been tested; the simulator can.

**Ceiling height (failure 5).** Prefer methods that cancel the per-photo depth scale:

| Method | Formula | Error behaviour |
|---|---|---|
| (d) **in-photo ratio** | r = (ceiling plane above camera) / (camera above floor plane), from the same MoGe depth map. Then H = h·(1 + r), with h from the AprilTag (Pick 3) or the fused scale | Scale-free. The residual error is MoGe's shape error, which is not measured on our data: test it |
| (a) **angle ratio**, HorizonNet's per-column rule | H = h·(1 + tan φc / tan φf) ([post_proc.py](https://github.com/sunset1995/HorizonNet/blob/master/misc/post_proc.py)) | 8–17 cm per degree of pitch for walls at 2–5 m, but pitch errors cancel to first order when opposite walls of a spin are averaged (derived) |
| (c) **door cross-ratio** | H = D·(tan φc + tan φf) / (tan φt + tan φf), with a door of known height D on the same wall | Under 1 cm per degree of pitch; 1% error in D gives 2.5 cm |
| (b) **tilted-up photo**, fallback | H = h + d·tan φc, with d from the plan | Keep only as the fallback |

**Code, compute.** Geometry only, no weights; milliseconds.

**Integration.**
1. **Two passes in `fit_room_layout`.**
   - Pass 1, as today: get the ceiling from `ceil_y`, now restricted to Pick 1's CEILING class.
   - Pass 2: set `layout_band_hi_m` to H − 0.05 for this room and keep `layout_band_lo_m` at 1.0.
2. **Ceiling-reach test in `_fit_side`.** For every candidate peak, take the 95th-percentile height of its inliers. If it is below H − 0.25 m and a farther peak or ceiling points exist on that side, drop the candidate as furniture. Door headers and soffits are the known exceptions: keep them when Pick 1 labels them wall.
3. **Lower bound per side.** The 98th-percentile offset of CEILING points, minus 5 cm, is the lower bound. Combine it with the existing `tan_extent` lower bound for unseen sides.
   - Ceiling seen through a doorway belongs to the next room. Ignore ceiling pixels in image columns that contain DOOR/OPENING labels, or that lie beyond a detected header.
4. **Height per spin photo.**
   - Compute r_i with rule (d) and with rule (a), and take the median over the spin.
   - H = h_tag·(1 + r) when a tag photo exists in the room; otherwise use the fused scale.
   - The 40° ceiling photo moves to rule (b), with d taken from the plan.
5. **Capture protocol.** `CAPTURE_PROTOCOL.md` spin uses the 0.5× lens, or 1× portrait if the 0.5× test fails. The EXIF focal length (13 mm) still sets MoGe's field of view (D-048).

**Test first.**
- Simulator A/B on the same houses, as in D-050, comparing a 0.5× spin with a 1× spin:
  - share of spin photos that see the ceiling line;
  - per-photo MoGe scale spread;
  - room-size error;
  - ceiling-height error of rules (d) and (a) against today's prior fusion.

### 1.3 Exact scale wherever possible: AprilTag on the sheet, fusion weighted by view type

**Status (D-067).** The AprilTag and sheet parts below are withdrawn: no capture uses a reference object. Only the
fusion weighted by view type still applies.

**Fixes.**
- Failure 4: per-photo scale with MAD 5.6%, range 0.81–1.31, and 0.59–1.18 on ceiling photos.
- Failure 5, through h.

**Key idea.** The sheet already gives exact scale (±0.2%) when it is found. The problem is detection: the D-017 detector finds 84% of sheets that are brighter than the floor, but only 20% on near-white floors.
- **Detection.** A black-bordered fiducial printed on the same A4 sheet is found on any floor. It also gives the camera's height above the floor directly.
- **Fusion.** For photos without the tag, stop letting ceiling-heavy photos vote for scale, and weight each photo by view type.

**Evidence.**
- **The error is mostly per-image scale** (MoGe-2 paper, Table B.4, metric AbsRel %, no alignment):

  | MoGe-2 variant | NYUv2 | iBims-1 |
  |---|---|---|
  | Predicted intrinsics | 7.33 | 13.6 |
  | Ground-truth intrinsics | 6.46 | 9.92 |
  | Scale-invariant (per-image scale aligned) | 3.44 | 3.16 |

  So most of the metric error is per-image scale ([MoGe-2](https://arxiv.org/html/2507.02546)). The pipeline already passes the EXIF focal (`fov_x` in `floorplan/photo/depth.py`).
- **Multi-view models do not fix scale.** On zero-shot ScanNet, MapAnything multi-view with images only reaches 22.23% relative error, and 24.94% with intrinsics, against 10.57% for MoGe-2 on one image. MapAnything only becomes good (5.58%) when metric poses are fed in ([MapAnything, Table 4](https://arxiv.org/html/2509.13414)).
  - The authors call its ScanNet metric scale "sub-optimal" and blame "lower benchmark dataset quality". Either way, it is no fix for our photos.
- **No newer model clearly wins indoors.** MoGe-3, MetricAnything Student-PointMap, FoundationGeo, UniK3D and DA3 each win on some indoor sets and lose on others (§2.8).
- **AprilTag detection range.** Tags of 0.167 m are detected about 100% of the time out to about 6–7 m on a 1296×964 camera. Orientation error is about 0.2–1° at 10–70° off-axis (read from figures; [AprilTag 2](https://april.eecs.umich.edu/media/pdfs/wang2016iros.pdf)).
- **Do not chain per-photo corrections.** In this repo the pose graph's per-photo scale corrections had MAD 11%, against a raw spread of 5.6% (D-056 comment in `floorplan/photo/scale.py`). Use the tag as a direct cue, as the sheet is used today.
- **Precedent.** Commercial camera-only apps get scale from a user measurement or reference: Polycam Rescale, Hover's hand-measured door, magicplan's laser. Zillow uses a calibration target that "allow[s] camera height to be computed" ([ZInD](https://github.com/zillow/zind)).

**Code, weights, licence.**
- Detector: `pupil-apriltags` (installed in `.venv`, but not in `requirements.txt` and imported nowhere). OpenCV ArUco is an alternative.
- No weights.

**Compute.** CPU; tens to hundreds of ms per 12 MP photo (U).

**Integration.**
1. **Print.** An AprilTag `tag36h11`, 180 mm black square, centred on the A4 sheet. Add a 100 mm check bar and the instruction "print at 100%": "fit to page" on Letter shrinks it to 0.941 (279.4 / 297, derived).
2. **Detector.** In `floorplan/scale` `observe_image(...)`, use the `rgb_full`/`K_full` arguments that `sheet_cues` already passes. Never detect at 518 px.
   - Sub-pixel corners, then homography/IPPE with the EXIF K, give the tag plane's distance and the camera height above the floor.
   - Scale = tag-plane distance / MoGe floor-plane distance at the tag pixels.
   - For IPPE's two-fold ambiguity at grazing angles, keep the solution whose normal agrees with gravity.
3. **Fusion in `scale.fuse`.**
   - Give each MoGe photo cue its own sigma by view type: pitch, floor fraction, ceiling fraction. Calibrate the sigmas on the simulator.
   - Photos pitched up by more than about 20° get no scale vote.
   - Use a Huber estimate (k = 1.345) and keep the chi-square inflation.
   - Do not correct the 0.975 median bias until it is confirmed on real photos.
4. **Per-user camera height.** The median camera height over tag photos replaces the 1.35 m chest-height prior in rooms without a tag.
   - Remaining spread: the shot-to-shot hold height, assumed to be 5–10 cm (U), i.e. 3.5–7% of a 1.4 m camera height (derived).
   - Without calibration, the person-to-person spread adds to it: adult standing eye height has an SD of 6.2–6.7 cm within each sex (ANSUR II).
   - Today's prior has a sigma of 11% (`CAMERA_HEIGHT_PRIOR_M`).
5. **Optional A/B: MetricAnything Student-PointMap.**
   - Id `yjh001/metricanything_student_pointmap`, Apache-2.0, a fine-tune of MoGe-2 ViT-L ([HF](https://huggingface.co/yjh001/metricanything_student_pointmap)).
   - Per the project README it loads with `moge.model.v2.MoGeModel` (U), so only the HF id changes.
   - It scores iBims-1 9.47 vs MoGe-2's 9.92 and DIODE 14.1 vs 16.2 with ground-truth intrinsics ([MetricAnything, Table 6](https://arxiv.org/html/2601.22054)).

**Test first.** Detection rate and scale error of the tag on light floors at 1–4 m, at full resolution vs 518 px, in the simulator and on one real print.

### 1.4 Rectilinear room polygon from free space and wall evidence; unseen sides flagged

**Fixes.**
- Failure 2: L-shaped and open-plan living-dining rooms (8–10 vertices, 78–81% of their bounding box) fitted as one rectangle, with the unseen end mirrored (−31% area).
- A second defence for failure 1: floor seen beyond a furniture front proves the wall is farther away.

**Key idea.** A wall can never be nearer than space that has been seen to be free. The room outline is the cheapest closed Manhattan path around the "core free space":
- the path runs through cells with wall evidence;
- each corner costs a constant penalty, so an L-shape (6 vertices) wins only when the data pay for its two extra corners.

This is Cabral & Furukawa's formulation, built for exactly our input (panoramas, sparse points, heavy clutter).
- **Grid and evidence.** Free-space evidence counts how often camera-to-point rays cross a cell; wall evidence counts the points in it.
- **Edge cost.** The sum of ρ(E_W) over the edge's cells + α. ρ is 1 for cells with too little wall evidence, and α = 5 is the per-edge penalty.
- **Solver.** Long Manhattan edges, Dijkstra, then dynamic programming for the best path at each vertex count. Anchor points remove the shrinking bias.

**Unseen sides.** An unseen side is reported, not mirrored. Zillow's capture rule is the precedent: extra panoramas for "Large rooms, corridors and L-shaped spaces (Ex: If you're more than 10 feet away from a wall, take another panorama)" and "Rooms where you cannot see all corners from one panorama" ([Zillow capture guide](https://wp-tid.zillowstatic.com/bedrock/app/uploads/sites/2/2022/03/Capture-Guide_Zillow-3D-Home-Interactive-Floor-Plans_2021-3c0104-1.pdf)).

**Evidence.**
- **Cabral & Furukawa, CVPR 2014** ([paper](https://openaccess.thecvf.com/content_cvpr_2014/html/Cabral_Piecewise_Planar_and_2014_CVPR_paper.html)).
  - On 7 real scenes with 5–17 panoramas each, area error ((over + under) / ground-truth area) is 0.006–0.156.
  - It is best in about half the scenes against volumetric graph cuts, and never worst.
  - Its floor / wall / ceiling labels "allow for recovery of thin walls and missing rooms".
- **HorizonNet on MatterportLayout** ([Zou et al., IJCV](https://arxiv.org/abs/1910.04099)): L-shapes are as easy as boxes, larger polygons are not.

  | Corners | 4 | 6 (L) | 8 | 10+ |
  |---|---|---|---|---|
  | 3D IoU | 81.88 | 82.26 | 71.78 | 68.32 |
- **PolyLayout** finds topology by splitting walls (ScanNet++ IoU 87.4). Its perimeter cost gives "slight shrinking bias" on unobserved walls ([paper](https://arxiv.org/abs/2608.03323)).

**Code, compute.** Geometry only, about 200 new lines, milliseconds.

**What the repo already has.**
- plan_beta's free-space code: `freespace.py` with `ray_carved`, `floor_observed`, `wall_evidence` and `fill_furniture_holes`.
- The photo tier v3 already swaps each folder's free-space room for the spin rectangle (`_layout_rooms` in `extract.py`).

**Integration.**
1. **Candidate lines.** `_fit_side` returns up to 3 peaks per side, not only the best one.
2. **Rasterise** the room's photos on a 5 cm grid in the room's Manhattan frame:
   - free cells: those crossed by camera-to-point rays, stopping 10 cm short, plus FLOOR-class points;
   - rays through a DOOR/OPENING pixel carve nothing, because what lies beyond a doorway is the next room;
   - wall evidence: Pick 1 wall points plus Pick 2's upper band.
3. **Search.**
   - Enumerate rectilinear polygons of 4, 6 or 8 vertices built from the candidate lines: thousands at most.
   - Cost: boundary cells without wall evidence, plus α per vertex, plus a large penalty for free cells outside the polygon.
   - A notch is accepted only if both of its edges have wall evidence and no free space lies inside it.
4. **Unseen sides.** A side with no evidence is reported as `unseen`, with a wide sigma and a QA message: "take a second spin from the other end of this room". It is not mirrored. The existing lower bound (neighbouring walls' extent) stays.
5. **Cross-check with PolyLayout** (`envs/polylayout`, installed). Run it when a room has 6 or more photos that see the ceiling junction, and flag topology disagreements.
   - It needs about 10 room-covering views and an explicit up axis (SETUP.md).
   - Its numbers are decimetre-level (ScanNet++ Chamfer distance 0.20 m), so it gives topology, not dimensions.
6. **Multi-room check.** If the facing walls of two adjacent rooms are more than about 0.35 m apart (more than a wall thickness), the nearer wall is a suspected furniture front, and the result is flagged.
7. **Protocol.** L-shaped and open-plan rooms get a second spin in the other arm, in the same folder.

**Test first.** The two L-shaped living-dining rooms (8–10 vertices):
- polygon area vs the rectangle;
- vertex count;
- the cost of a second spin in the capture time.

### 1.5 Glass and mirror handling (masks, ring planes, partitions)

**Fixes.** Failure 3:
- shower glass partitions, glass doors and windows taken as surfaces;
- mirrors adding a "room behind the mirror";
- glass that MoGe sees through, which makes the farthest-point rule overshoot.

**Key idea.** Three layers:
1. **Find the glass and mirrors.** Pick 1's ADE classes cover mirrors (IoU about 78%) and windows (about 67%). ADE20K has **no** class for shower screens or glass partitions, so add open-vocabulary SAM 3 prompts for those.
2. **Never let masked depth vote.** Measure what is *around* each mask instead:
   - A mirror or window sits in a wall. A ring of wall pixels around it gives that wall's plane. Mirror3DNet places the mirror plane from a 25 px border in the same way.
   - A shower screen or glass door is not a room boundary. Its foot line on the floor plane locates it without using any glass depth.
3. **Geometric safety net** for what the detector misses:
   - Points well behind the fitted wall plane, inside the wall's extent, are flagged as glass, mirror or opening. This is the same test as Locometric's opening patent.

**Evidence: what depth models do on glass.**
- **Surface or "through" depends on the model** (MD-3k, 3,161 GDD glass images; α > 0 means the model predicts what is behind the glass) ([One Scene, Two Depths](https://arxiv.org/html/2606.29600)):

  | Prefers the glass surface | α | Prefers what is behind | α |
  |---|---|---|---|
  | Depth Anything V2-L | −30.6 | UniDepthV2-L | +45.4 |
  | Depth Pro | −35.2 | UniK3D-L | +56.4 |
  | | | Marigold | +20.8 |

  MoGe was not tested.
- **MoGe mixes the layers more often.** On LayeredDepth's first-layer (glass-surface) ordering, MoGe v1 reaches 76.76% pair accuracy, against 85.34% for Depth Anything V2 and 87.39% for Depth Pro ([LayeredDepth](https://arxiv.org/html/2503.11633)). We have to measure MoGe-2 ourselves.
- **Layout methods are confused by mirrors.** HorizonNet "could also be confused by the reflected room boundaries in the mirror" ([Zou et al.](https://arxiv.org/abs/1910.04099)).
- **Others exclude glass from the geometry.**
  - HouseLayout3D excludes windows and mirrors because they "often suffer from poor depth estimates" ([arXiv 2512.02450](https://arxiv.org/html/2512.02450)).
  - Matterport: "when the camera processes a reflection, it believes there is a room on the other side of the mirror". Its users mark mirrors by hand so that "the app is told to ignore what it sees in the mirror" ([scanning guidelines 2020](https://assets.ctfassets.net/icnj41gkyohw/3KQeAKpcELDEzkWOQ6Nnrs/2308325ca2430bea93637bb10bd5a62c/Matterport-Capture_Service-Scanning-Guidelines.pdf)).
  - Polycam says frameless shower enclosures "are transparent to LiDAR" ([help](https://learn.poly.cam/hc/en-us/articles/34295265957140-Improving-Scan-Accuracy-Best-Practices-for-Interior-Spaces)).
- **Ring planes work.** Mirror3DNet's ring-plane refinement cuts mirror-pixel depth RMSE from 1.272 m to 0.766 m on sensor depth ([Mirror3D](https://ar5iv.labs.arxiv.org/html/2106.06629)).

**Evidence: detectors.**

| Detector | Score | Source |
|---|---|---|
| SAM 3, zero-shot, glass | GSD-S IoU 0.713, precision 0.764, recall 0.915 (prompt not disclosed) | [GlassGuard, Table I](https://arxiv.org/html/2610.02110) |
| GEM (trained on GSD-S) | IoU 0.770 (Tiny) / 0.774 (Base) | [GEM](https://arxiv.org/html/2401.15282) |
| GlassSemNet | IoU 0.753 | same |
| SAM 1 | GDD IoU 0.485 (fails on glass) | [L+GNet](https://arxiv.org/html/2603.03718) |
| Frozen SAM 3 + small adapters (SAM3-UNet), mirrors | MSD IoU 0.943, PMD 0.804, the best mirror numbers found | [SAM3-UNet](https://arxiv.org/html/2512.01789) |

**Code, weights, licence.**
- **SAM 3**: `facebook/sam3` on HF, gated. Accept the licence once, then download by script with a token.
  - Access is approved manually; how long that takes is unknown (U), so request it before relying on it. GEM, below, needs no approval.
  - 848M parameters, about 1.7 GB in bf16. Needs torch ≥ 2.7 and CUDA ≥ 12.6; ours is torch 2.14 with cu130.
  - SAM licence: commercial use allowed; military and ITAR uses barred; the licence must ship with the weights.
  - It runs in `envs/seg` next to Pick 1 (`Sam3Model` / `Sam3Processor`), or from the official repo.
- **Alternative glass detector: GEM.** Weights on HF, tagged Apache-2.0: Tiny 260 MB, Base 1.23 GB ([HF](https://huggingface.co/Bryceee/GEM-Glass-Segment-model)). It needs detectron2 and MaskDINO's CUDA ops, which is the integration risk.
- **Mirror planes: Mirror3DNet.** MIT; checkpoints over plain HTTP from SFU; needs detectron2 ([repo](https://github.com/3dlg-hcvc/mirror3d)).
- **Out of reach on this network.** Every dedicated mirror detector found (HetNet, SATNet, CSFwinformer, SAM2/3-UNet adapters) hosts its weights on Google Drive, which is blocked. HetNet is also non-commercial.

**Compute.** SAM 3 latency and VRAM on an 8 GB laptop GPU are not published (U). Its image embedding is computed once and reused across prompts, so 5 prompts cost little more than 1. GEM-Tiny runs at 16 FPS at 384 px [AUTH].

**Integration.**
1. **Worker.** The `envs/seg` worker (Pick 1) also runs SAM 3 once per photo, with the prompts "mirror", "window", "glass door", "glass shower screen" and "glass partition". It saves one mask PNG per prompt with its score.
2. **Mask on the view.** `PhotoView.glass_mask` is the union of the ADE classes {27 mirror, 8 windowpane, 58 screen door, 145 shower} and the SAM 3 masks above a score threshold, dilated by 5–8 px at 518 px.
3. **In `fit_room_layout`.**
   - Masked pixels are neither wall candidates nor "farthest per column" candidates.
   - If more than 50% of a column's band is masked, that column contributes nothing. Do not fall back to the next-farthest point.
4. **Ring plane per mirror or window.**
   - Take MoGe points in a 25 px ring around the mask that have horizontal normals and Pick 1's WALL label, and fit a plane with RANSAC.
   - If the plane is vertical and Manhattan-aligned, its points are wall evidence for that side.
5. **Shower screens and glass doors.** The mask's lowest pixels, intersected with the floor plane, give the partition's line on the plan. It is reported as an interior partition or door, never as the room's outer wall. The wall behind it must come from wall pixels: the shower's back wall, seen through the open doorway or above the screen.
6. **Safety net.** After the fit, a compact image patch of points more than 0.2 m beyond the fitted room is flagged "possible mirror, glass or opening" in the report.
   - A multi-view consistency check will **not** catch mirrors: the virtual room behind a mirror is consistent from every view.
   - The test that would work is to reflect suspect points across the candidate wall plane and look for real points there (idea, untested).

**Test first.**
- **Oracle masks.** The simulator marks glass (translucent or OmniGlass materials) and mirrors per mesh (`sim/scene_geom.py`), so exact masks per photo are free.
- **Measure MoGe-2 inside the masks.** On the bathrooms and glass doors, classify depth inside each mask against its ring plane: surface (within 5 cm), through/phantom (more than 20 cm beyond), or mixed.
- **Masking ablation.** Room error with and without oracle masks.
- **Then SAM 3 against the oracle.** Look for false positives on glossy tiles, TV screens and glass-front cabinets.

### 1.6 Next tier (worth a timed experiment, not a first pick)

| Method | Why it might help | Why not first | Cost |
|---|---|---|---|
| [Room Envelopes](https://arxiv.org/abs/2511.03970) (AJCAI 2025) | MoGe v1 fine-tuned to predict the **layout pointmap**, the first structural surface with furniture removed. This is the only model found that predicts walls *behind* a full-height wardrobe. Occluded layout pixels: Chamfer 0.753 m and F@0.1 0.512, vs MoGe v1 1.037 / 0.114 | Synthetic test only, scored after scale-shift alignment, with no real-image numbers. Licence CC BY-NC-SA 3.0 | Zero install: `moge infer --version v1 --pretrained hugsam/room_envelopes_model`. Align it to MoGe-2 on pixels Pick 1 labels as structure |
| [uLayout](https://github.com/JonathanLee112/uLayout) (WACV 2025) | Amodal floor-wall and ceiling-wall boundary per column for perspective photos; needs pitch, which we have. LSUN floor/ceiling 2D IoU about 80 / 83 | Turning a boundary angle into a distance costs 14–16 cm per degree of pitch or boundary error at 3 m (derived). Weights on Google Drive, which is blocked here | MIT; download in a browser |
| [HorizonNet](https://github.com/sunset1995/HorizonNet) on a stitched spin | Clutter-robust, handles missing top and bottom bands (Stanford 2D-3D), and its weights are on HF | No evidence for hand-held spins with horizontal gaps; assumes a 1.6 m camera | MIT; HF `sunset1995/HorizonNet` |
| [PolyLayout](https://github.com/ghanning/PolyLayout) as the main fitter | Joint multi-room Manhattan polygons and a shared ceiling height | About 10 views per room; decimetre accuracy; shrinking bias on unobserved walls | Installed |
| [SAM 3](https://github.com/facebookresearch/sam3) prompts for furniture ("wardrobe", "kitchen cabinet") as a second opinion to Pick 1. Pick 5 already uses SAM 3 for glass | Instance masks for built-ins that ADE confuses with walls | Open-vocabulary ADE accuracy is 20–30 mIoU below closed-set models; no benchmark for these concepts | SAM licence, gated |
| [Bi-Layout](https://github.com/LIAGM/Bi_Layout) | "Enclosed" vs "extended" layout settles open-plan room ends | Panorama only | Apache-2.0, HF weights |

## 2. Everything relevant (with links)

### 2.1 Semantic segmentation (wall vs furniture; glass and mirror classes)

ADE20K val mIoU is single-scale / multi-scale where both are known.

| Model | ADE20K mIoU | Params | Licence (code / weights) | Weights | Notes |
|---|---|---|---|---|---|
| **[EoMT-L, DINOv2](https://github.com/tue-mps/eomt)** (CVPR 2025) | 58.4 | 316–319M | MIT / MIT | HF `tue-mps/ade20k_semantic_eomt_large_512` | transformers ≥ 4.54; 512 px tiles |
| [OneFormer Swin-L](https://github.com/SHI-Labs/OneFormer) (CVPR 2023) | 57.0 / 57.7 | 219M | MIT / MIT | HF `shi-labs/oneformer_ade20k_swin_large` | used by HouseLayout3D for structural filtering |
| OneFormer DiNAT-L | 58.3 / 58.4 | 223M | MIT | HF | needs NATTEN |
| [Mask2Former Swin-L](https://github.com/facebookresearch/Mask2Former) | 56.1 / 57.3 | 216M | MIT / **CC BY-NC** | HF `facebook/mask2former-swin-large-ade-semantic` | NC weights |
| [BEiT-L + UperNet](https://huggingface.co/microsoft/beit-large-finetuned-ade-640-640) | 57.0 (MS) | 441M | Apache-2.0 | HF | heavy |
| EoMT-L, DINOv3 | 59.5 | ~0.3B | MIT code; DINOv3 licence, gated | HF deltas | upgrade path |
| [SegFormer-B5](https://github.com/NVlabs/SegFormer) | 51.0 / 51.8 | 85M | NVIDIA research-only | HF | 7 points lower |
| [InternImage-H + M2F](https://github.com/OpenGVLab/InternImage) | 62.6 / 62.9 | 1.31B | MIT | HF | mmseg 0.27 + DCNv3 build; NYUv2-40 variant 67.1 mIoU (wall 89.6, cabinet 71.8, mirror 73.0 in-domain) |
| [SAM 3](https://github.com/facebookresearch/sam3) | ADE-847 13.8 (open-vocab) | 848M | SAM licence, gated | HF | text prompts |
| [CAT-Seg L](https://github.com/cvlab-kaist/CAT-Seg) | 37.9 (A-150) / 16.0 (A-847) | 434M | MIT | HF | best open-vocab ADE-150 |
| Grounding DINO + SAM 2.1 | open-vocabulary box detector + promptable masks (no ADE number) | ~0.4B | Apache-2.0 | HF | prompts |
| [DFormerv2](https://github.com/VCIP-RGBD/DFormer) (RGB-D) | NYU 58.4 / SUN 53.3 (L) | 96M | non-commercial | HF | MoGe depth as input untested |

Class-level evidence, from a public log of ViT-Adapter-L + Mask2Former at 58.3 mIoU ([log](https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beit_adapter_large_640_160k_ade20k_ss.log)), IoU %:

| Group | Classes |
|---|---|
| Structure | wall 82.2, floor 85.7, ceiling 87.6 |
| Storage and counters | cabinet 66.2, wardrobe 57.9, countertop 67.6, counter 54.9, chest of drawers 39.9 |
| Openings | door 59.1, windowpane 66.6, screen door 77.3 |
| Mirror and glass | mirror 78.1, shower 12.3, glass (drinking) 26.4 |

ADE annotator self-consistency is 82.4% of pixels ([ADE20K paper](https://ar5iv.labs.arxiv.org/html/1608.05442)).

### 2.2 Single-image perspective layout nets (amodal floor-wall boundary)

| Method | Venue | What it predicts | Evidence | Code / weights / licence | Fit for us |
|---|---|---|---|---|---|
| [LayoutNet](https://arxiv.org/abs/1803.08999) | CVPR 2018 | Corner and boundary maps "for both visible and occluded boundaries"; perspective mode has no 3D | LSUN / Hedau era | [PyTorch port](https://github.com/sunset1995/pytorch-layoutnet) MIT, panorama only, Google Drive | Low |
| [RoomNet](https://arxiv.org/abs/1703.06241) | ICCV 2017 | 11 LSUN room types + keypoints | LSUN keypoint error 6.30%, pixel error 9.86% [AUTH] | No official code | Low |
| [Flat2Layout](https://arxiv.org/abs/1905.12571) | arXiv 2019 | Per-column row vectors + DP, general room types | not checked | No code found (U) | Low |
| [lsun-room](https://github.com/leVirve/lsun-room) | ICPR 2018 | Per-pixel layout planes | About 26% pixel error on generic RealEstate10K rooms vs 6–7% on LSUN/Hedau ([cad-estate](https://arxiv.org/abs/2306.09077)) | MIT, Google Drive | Low: the cuboid prior breaks on real rooms |
| [ST-RoomNet](https://github.com/lab231/ST-RoomNet) | CVPRW 2023 | Cuboid via a spatial transformer | LSUN pixel error 5.24%, Hedau 7.10% [AUTH] | TF 2.9; licence not stated | Low |
| [GeoLayout](https://arxiv.org/abs/2008.06286) | ECCV 2020 | Depth of dominant planes; layout by intersection | not checked | No official code found (U) | Low |
| [NonCuboidRoom](https://github.com/CYang0515/NonCuboidRoom) | **WACV 2022** (not CVPR) | Wall / floor / ceiling planes + vertical lines → non-cuboid 3D layout | Structured3D IoU 81.40, pixel error 5.87%, layout-depth RMSE 0.29 m; NYUv2-303 pixel error 10.61%. Own failure case: "mistakenly detects the foreground furniture as the wall" when most of the wall is occluded ([paper](https://arxiv.org/abs/2104.07986)) | MIT; Google Drive; PyTorch 1.5 / CUDA 10.1 | Medium-low |
| [Layout Anything](https://arxiv.org/abs/2512.02952) | WACV 2026 | OneFormer adapted to layout | LSUN pixel error 5.43% / corner error 4.02%; MP3D-Layout 4.03 / 3.15 [AUTH] | No code | Not usable |
| [uLayout](https://github.com/JonathanLee112/uLayout) | WACV 2025 | One model for perspective and 360 images; a perspective photo is placed at the latitude of its pitch; per-column boundaries | LSUN floor/ceiling 2D IoU 80.1–80.3 / 83.1–83.6; 360 3D IoU 86.04 (PanoContext), 86.90 (Stanford), 81.84 (MP3D) [AUTH] ([paper](https://arxiv.org/html/2503.21562)) | MIT; 3 checkpoints on Google Drive | Medium (§1.6) |
| [Polygon HGT + wireframes](https://arxiv.org/abs/2306.12203) | ICCVW 2023 | Semantic plane polygons | Structured3D 2D metrics | No code link | Low |
| [Room Envelopes](https://arxiv.org/abs/2511.03970) | AJCAI 2025 | Layout pointmap (furniture removed), fine-tuned from MoGe v1 | Occluded-layout Chamfer 0.753 m, F@0.1 0.512 vs MoGe v1 1.037 / 0.114, synthetic only | HF `hugsam/room_envelopes_model`, CC BY-NC-SA 3.0 | Medium (§1.6) |
| [cad-estate](https://github.com/google-research/cad-estate) | 2023 | 2,246 RealEstate10K rooms with amodal wall/floor/ceiling masks + 3D planes | Reconstruction vs ScanNet depth: reprojection IoU 0.90, depth error 0.22 m | Data CC BY 4.0 | Real-photo evaluation set for generic rooms |

### 2.3 Panorama layout nets, and whether they work on a photo spin

| Method | Venue | Evidence | Code / weights / licence |
|---|---|---|---|
| [HorizonNet](https://github.com/sunset1995/HorizonNet) | CVPR 2019 | See note below the table | MIT; HF `sunset1995/HorizonNet` (st3d, mp3d, zind, panos2d3d) |
| [LGT-Net](https://github.com/zhigangjiang/LGT-Net) | CVPR 2022 | MP3D 3D IoU 81.11; ZInD 89.95 | MIT; Google Drive |
| [LED2-Net](https://github.com/fuenwang/LED2-Net) | CVPR 2021 | MP3D 80.14; ZInD 88.49 (re-run in LGT-Net) | MIT |
| [HoHoNet](https://github.com/sunset1995/HoHoNet) | CVPR 2021 | MP3D 79.88 | MIT |
| [DOPNet](https://github.com/zhijieshen-bjtu/DOPNet) | CVPR 2023 | MP3D 2D / 3D IoU 84.11 / 81.70 | MIT |
| [Bi-Layout](https://github.com/LIAGM/Bi_Layout) | CVPR 2024 | Enclosed vs extended layouts; MP3D 3D IoU 82.57 (mild oracle); high-ambiguity subset 59.97 vs 54.80 | Apache-2.0; HF `LIAGM/Bi_Layout_Model` |
| [AtlantaNet](https://github.com/crs4/AtlantaNet) | ECCV 2020 | MP3D 80.02; beyond Manhattan | MIT |
| [PanoTPS-Net](https://arxiv.org/abs/2510.11992) | 2025 | 3D IoU 85.49 / 86.16 / 81.76 / 91.98 (PanoContext / S2D3D / MP3D / ZInD) [AUTH] | no licence file |
| [Pano2Room](https://github.com/TrickyGo/Pano2Room) | SIGGRAPH Asia 2024 | Novel-view synthesis (3DGS) from one panorama; **not** a layout method | — |

HorizonNet in detail:
- **Benchmarks.**
  - PanoContext/Stanford 2D-3D 3D IoU 83.87%.
  - Stanford 2D-3D, whose panoramas have a "black missing polar region caused by smaller camera V-FOV": 79.79, or 83.51 with PanoContext added.
  - MatterportLayout by corner count: 81.88 (4), 82.26 (6), 71.78 (8), 68.32 (10+).
- **Mirrors.** "could also be confused by the reflected room boundaries in the mirror" ([Zou et al.](https://arxiv.org/abs/1910.04099)).
- **How it gets walls.** From the ceiling boundary, voting within 0.16 m; it assumes h = 1.6 m.

**On spins.** Our D-017 spin is a hand-held panorama with gaps, sway and a limited vertical field. HorizonNet tolerates missing top and bottom bands, and uLayout handles single perspective photos by design. No paper was found that runs a panorama layout net on a hand-held spin of 4–6 stills with horizontal gaps, so treat it as untested.

### 2.4 Multi-view and sparse-view room layout from perspective photos

| Method | Venue | Input | Evidence | Code / weights / licence |
|---|---|---|---|---|
| **[PolyLayout](https://github.com/ghanning/PolyLayout)** | ECCV 2026 | Posed perspective photos, about 10 per room; the assignment of images to rooms is known (our folders) | See notes below the table ([paper](https://arxiv.org/abs/2608.03323)) | Apache-2.0; ScanNet++-trained weights (research-only data terms); **installed** |
| [PixCuboid](https://github.com/ghanning/PixCuboid) | ICCVW 2025 | Posed photos; cuboid rooms only | ScanNet++ IoU 87.2, Chamfer 0.22 m, depth RMSE 0.09 m; 2D-3D-S 89.0 / 0.18 / 0.10. Largest gain going from 2 to 3 views; learned features "ignore clutter" ([paper](https://arxiv.org/abs/2508.04659)) | Apache-2.0; installed |
| [Plane-DUSt3R](https://github.com/justacar/Plane-DUSt3R) | ICLR 2025 | Unposed sparse views | Structured3D 3D precision 52.63%, recall 48.37%; ASE Chamfer 23.16 m (PolyLayout's table) | MIT code; DUSt3R base CC BY-NC-SA |
| [SpatialLM 1.1](https://github.com/manycore-research/SpatialLM) | 2025 | Dense z-up point cloud | Structured3D layout F1@0.25 94.3; zero-shot on MASt3R-SLAM clouds 47.4–68.9 | Qwen-0.5B Apache-2.0; Sonata encoder CC BY-NC |
| [HouseLayout3D](https://arxiv.org/abs/2512.02450) | arXiv 2025 | Images + COLMAP poses + Metric3D depth → mesh | Avg F1 0.109 (mesh) → 0.223 (OneFormer structural skeleton) → 0.381 (full) | (U) |
| [GRIHA](https://arxiv.org/abs/2103.08297) | MTAP 2022 | Phone photos + ARCore poses | no dimension errors verified | no code |
| [ProClosure](https://github.com/ClarityLab-Org/ProClosure) | arXiv 2026 | MASt3R-SLAM cloud + SAM 3 tracks → rooms | HM3D room F1@0.25 0.741 → 0.890 | NC dependencies |

PolyLayout results ([paper](https://arxiv.org/abs/2608.03323)):

| Dataset (ground-truth poses) | IoU | Chamfer | Other |
|---|---|---|---|
| ScanNet++ | 87.4 | 0.20 m | wall recall 69.3%, room recall 36.3% |
| ASE | 94.3 | 0.12 m | — |
| 2D-3D-S | 90.0 | 0.15 m | — |

- **Predicted poses.** With π³-predicted poses on ASE, IoU drops to 89.4 and Chamfer rises to 0.21 m.
- **Runtime.** 3.5–5.5 s per scene.
- **Unobserved walls.** A "slight shrinking bias".

### 2.5 Floor plans from point clouds and density maps (for completeness)

These need a dense top-down density map. A room shot with 2–8 photos gives only a partial one, so they serve as cross-checks, not fixes.

Scores are Structured3D Room / Corner / Angle F1. The corner tolerance is 10 px on a 256 px map, i.e. decimetres.

| Method | Venue | R / C / A F1 | Code / licence |
|---|---|---|---|
| [Floor-SP](https://github.com/woodfrog/floor-sp) | ICCV 2019 | Room-wise shortest path on density and normal maps (no F1 checked) | MIT |
| [MonteFloor](https://arxiv.org/abs/2103.11161) | ICCV 2021 | 95.0 / 82.5 / 80.5; about 71 s per scene | no code |
| [RoomFormer](https://github.com/ywyue/RoomFormer) | CVPR 2023 | 97.3 / 87.2 / 81.2 | MIT (installed) |
| [SLIBO-Net](https://proceedings.neurips.cc/paper_files/paper/2023/hash/987bed997ab668f91c822a09bce3ea12-Abstract-Conference.html) | NeurIPS 2023 | 98.4 / 85.4 / 84.4 | repository returns 404 |
| [PolyRoom](https://github.com/3dv-casia/PolyRoom) | ECCV 2024 | 98.3 / 90.2 / 85.2 | no licence |
| [FRI-Net](https://github.com/Daisy-1227/FRI-Net) | ECCV 2024 | 99.1 / 87.8 / 86.9 | no licence |
| [Raster2Seq](https://github.com/Cornell-VAILab/Raster2Seq) | SIGGRAPH 2026 | 99.6 / 98.3 / 92.7 (Structured3D-B) | MIT (installed) |
| [CAGE](https://github.com/ee-Liu/CAGE) | 2025 | 99.1 / 91.7 / 89.3 | MIT + Commons Clause |

### 2.6 Non-rectangular rooms from sparse views

| Method | Key idea | Evidence | Code |
|---|---|---|---|
| **[Cabral & Furukawa](https://openaccess.thecvf.com/content_cvpr_2014/html/Cabral_Piecewise_Planar_and_2014_CVPR_paper.html)** (CVPR 2014) | Free space + wall evidence on a grid. Shortest closed path around the core free space, with long Manhattan edges and a per-edge penalty. Anchor points against shrinkage. Floor/wall/ceiling labels turn each column's boundary into a 3D point via the floor height | Area error 0.006–0.156 on 7 scenes (5–17 panoramas) | Paper only |
| PolyLayout wall split / merge | Topology from iterative wall splitting (up to 32 planes) | §2.4 | Apache-2.0 |
| [ZInD](https://github.com/zillow/zind) | "visible" geometry is clamped to what is observed; "complete" geometry is merged. "An opening is an artificial construct that divides a large room into multiple parts" | Dataset definitions | Academic-only data |
| [Zillow capture guide](https://wp-tid.zillowstatic.com/bedrock/app/uploads/sites/2/2022/03/Capture-Guide_Zillow-3D-Home-Interactive-Floor-Plans_2021-3c0104-1.pdf) | Extra panoramas for L-shaped spaces, or when not all corners are visible; "If the room is furnished, imagine the corners without furniture" | Production protocol | — |
| [Beyond the Frontier](https://arxiv.org/abs/2406.09160) (RA-L 2024) | Predicts unseen walls from partial occupancy grids (robot exploration) | Gains in information gain on office plans | Paper CC BY 4.0 |
| magicplan open plan ([help](https://help.magicplan.app/magicplan-floor-plan-editor-faq)) | Each area becomes its own closed room; the shared walls are then deleted | Product practice | — |

### 2.7 Glass and mirror detection; depth on transparent and reflective surfaces

Metrics are IoU / F-beta / MAE / BER where given (lower is better for MAE and BER). Papers retrain each other's methods, so compare numbers only within one source. GD = Google Drive, which is blocked on this network.

**Glass detectors**

| Method | Benchmark numbers | Code; licence | Weights |
|---|---|---|---|
| [GDNet](https://github.com/Mhaiyang/CVPR2020_GDNet) (CVPR 2020) | GDD: 87.63 / 0.937 / 0.063 / 5.62 ([paper](https://www.cs.cityu.edu.hk/~rynson/papers/cvpr20d.pdf)). GDD has 3,916 images, 2,827 of them indoor | licence terms (U) | GD |
| [GSD](https://www.cs.cityu.edu.hk/~rynson/papers/cvpr21.pdf) (CVPR 2021) | GSD: 83.64 / 0.903 / 0.055 / 6.12 | code zip; licence (U) | (U) |
| [GlassSemNet](https://github.com/xavhl/GlassSemNet) (NeurIPS 2022) | GSD-S IoU 0.753; GDD 0.908 | (U) | GD |
| [EBLNet](https://github.com/hehao13/EBLNet) (ICCV 2021) | GDD IoU 88.16–88.72 | not stated | GD, Baidu |
| [GhostingNet](https://github.com/YT3DVision/GhostingNet) (TPAMI 2025) | GDD 91.19; GSD 86.69 | not stated | GD, Baidu, MEGA |
| **[GEM](https://github.com/isbrycee/GEM-Glass-Segmentor)** (2024) | GSD-S 0.770–0.774 / 0.865 / 0.029–0.032; 11–16 FPS at 384 px ([paper](https://arxiv.org/html/2401.15282)) | detectron2 + MaskDINO | **HF**, Apache-2.0 tag |
| [GlassWizard](https://github.com/wxliii/GlassWizard) (ICCV 2025) | GDD 93.30; GSD 90.40 (Stable Diffusion 2 based, 177 ms on an RTX 4090) | not stated | UNet on HF; text embeddings on GD |
| [L+GNet](https://github.com/ojalar/lgnet) (2026) | GDD 0.948, BER 2.50; GSD 0.931; 0.76–2.06 GB VRAM at 512 px ([paper](https://arxiv.org/html/2603.03718)) | Apache-2.0 | GD + gated DINOv3 |
| [VGGT-Glass](https://github.com/YT3DVision/VGGT_GLASS) (2026) | GDD 95.56 / 0.981; GSD 91.91 ([paper](https://arxiv.org/html/2608.26752)) | not stated | GD, MEGA |
| SAM 3, zero-shot | GSD-S IoU 0.713, recall 0.915 ([GlassGuard](https://arxiv.org/html/2610.02110)) | SAM licence | **HF, gated** |
| [MonoGlass3D](https://arxiv.org/abs/2509.05599) (2025) | Glass mask + plane from one RGB image; metrics not in the abstract | no code found | — |

**Mirror detectors**

| Method | Benchmark numbers | Code; licence | Weights |
|---|---|---|---|
| [MirrorNet](https://github.com/Mhaiyang/ICCV2019_MirrorNet) (ICCV 2019) | MSD 0.790; PMD 0.585 | license.txt | GD |
| [PMD](https://jiaying.link/cvpr2020-pgd/pmd_release.zip) (CVPR 2020) | MSD 0.815; PMD 0.660 | zip | (U) |
| [HetNet](https://github.com/Catherine-R-He/HetNet) (AAAI 2023) | MSD 0.828; PMD 0.690; 49 FPS on a 2080Ti ([paper](https://arxiv.org/pdf/2211.15644)) | BSD-3 + non-commercial clause | GD |
| [SATNet](https://github.com/tyhuang0428/SATNet) (AAAI 2023) | MSD 85.41; PMD 69.38 | Apache-2.0 | GD |
| [CSFwinformer](https://github.com/wangsen99/CSFwinformer) (TIP 2024) | MSD 82.13; PMD 69.84 | Apache-2.0 | GD |
| [SAM2-UNet](https://github.com/WZH0120/SAM2-UNet) | MSD 0.918; PMD 0.728 | Apache-2.0 | GD |
| [SAM3-UNet](https://github.com/WZH0120/SAM3-UNet) (2025) | MSD 0.943 / 0.972 / 0.014; PMD 0.804 ([paper](https://arxiv.org/html/2512.01789)) | Apache-2.0 | GD + gated SAM 3 |
| [MirrorSAM](https://winter-flow.github.io/project/MirrorSAM) (AAAI 2026) | MSD 0.913; PMD 0.759 | no code found | — |
| **[Mirror3D / Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d)** (CVPR 2021) | Mask + mirror plane + depth fix; mirror-pixel RMSE 1.272 → 0.766 m (sensor depth) and 0.903 → 0.567 m (BTS) | MIT; detectron2 | **SFU server over HTTP** |
| VMD / MG-VMD | Video mirror detection | — | Needs video; not applicable |

**Depth models on glass and mirrors**

| Study | Finding |
|---|---|
| [MD-3k](https://arxiv.org/html/2606.29600) (2026) | Which layer a model predicts on glass (α, as in §1.5): Depth Pro −35.2 and DAv2-L −30.6 prefer the surface; UniDepthV2-L +45.4, UniK3D-L +56.4 and DAv1-L +90.4 prefer what is behind. MoGe not tested |
| [LayeredDepth](https://arxiv.org/html/2503.11633) (2025) | First-layer (glass surface) pair / quadruplet accuracy: MoGe 76.76 / 58.92; DAv2 85.34 / 70.43; Depth Pro 87.39 / 69.46 |
| [Booster](https://ar5iv.labs.arxiv.org/html/2206.04671) | Ground truth is the closest surface (glass painted before scanning). Scale-and-shift aligned δ<1.05 on glass and mirror pixels: DAv2 0.644, Metric3D 0.296 ([arXiv 2408.06083](https://arxiv.org/html/2408.06083)). Mostly close-up objects, not room-scale glass |
| [Depth4ToM](https://ar5iv.labs.arxiv.org/html/2307.15052) (ICCV 2023) | Fine-tuning for transparent and mirror surfaces: DPT δ<1.05 37.70% → 45.97%, MAE 113 → 57 mm on Booster; weights on OneDrive |
| [MoGe-2](https://arxiv.org/html/2507.02546) | No glass or mirror evaluation. HAMMER (table-top, includes transparent objects; whole-image score) 26.9 / 65.6 metric AbsRel / δ1. Notes that "SfM might miss … reflective surfaces" |

**Geometric and process ideas**

| Idea | Source |
|---|---|
| Ring-plane fit around a mirror | [Mirror3D](https://ar5iv.labs.arxiv.org/html/2106.06629) |
| Depth beyond the expected wall = opening | [Locometric patent US11269060B1](https://patents.google.com/patent/US11269060B1/en) |
| Mirror detection from flipped matches between two images | [Apple patent CN112509040A](https://patents.google.com/patent/CN112509040A/en) |
| Manual mirror and window marking | Matterport guidelines (§2.11) |
| SAM 3 masks + LiDAR planes, deliberately avoiding monocular depth on glass | [GlassGuard](https://arxiv.org/html/2610.02110) |

### 2.8 Metric depth models indoors, without per-image alignment (AbsRel % / δ1 %)

All rows use the same evaluation suite (MoGe papers). DIODE and ETH3D mix indoor and outdoor scenes; HAMMER is close-range table-top.

| Model (intrinsics) | NYUv2 | iBims-1 | HAMMER | DIODE | ETH3D | Source |
|---|---|---|---|---|---|---|
| MoGe-2 (predicted K) | 7.33 / 96.1 | 13.6 / 83.0 | 26.9 / 65.6 | 17.5 / 66.4 | 10.4 / 90.8 | [MoGe-2, Table B.4](https://arxiv.org/html/2507.02546) |
| **MoGe-2 (GT K)**: what we run, via EXIF fx | **6.46 / 96.9** | 9.92 / 92.4 | 30.4 / 74.2 | 16.2 / 77.1 | 10.5 / 92.2 | same |
| MoGe-2 scale-invariant (per-image scale aligned) | 3.44 / 98.2 | 3.16 / 98.2 | 3.96 / 99.2 | 5.30 / 94.6 | 3.55 / 98.7 | same |
| UniDepthV2-L (GT K) | 7.81 / 96.0 | **7.71 / 95.5** | 37.7 / 47.1 | 41.0 / 67.1 | 15.0 / 85.2 | same |
| Metric3D v2 (GT K) | 7.16 / 96.5 | 9.96 / 94.1 | 35.7 / 44.3 | 49.1 / 1.98 | 11.8 / 88.8 | same |
| Depth Pro (predicted K) | 10.7 / 91.9 | 15.9 / 81.5 | 39.1 / 63.0 | 31.9 / 37.7 | 38.5 / 32.8 | same |
| MoGe-3 ViT-L | 8.43 / 95.4 | 11.7 / 88.3 | 23.5 / 74.2 | 15.2 / 77.3 | 14.8 / 85.3 | [MoGe-3, Table C.1](https://arxiv.org/html/2607.17967v2) |
| Depth Anything 3 (variant U) | 7.74 / 94.4 | 10.8 / 86.7 | 19.1 / 74.7 | 19.0 / 64.3 | 15.0 / 79.7 | same |
| UniK3D | 9.88 / 92.8 | 9.35 / 93.6 | 30.5 / 58.3 | 18.1 / 73.0 | 14.9 / 83.7 | same |
| MetricAnything Student-PointMap (GT K) | – | 9.47 / 92.0 | – | 14.1 / 74.3 | 9.75 / 92.4 | [MetricAnything, Table 6](https://arxiv.org/html/2601.22054) |
| FoundationGeo (predicted K) | 10.2 / 93.0 | 10.4 / 90.0 | 22.4 / 69.6 | 17.5 / 69.7 | 17.8 / 74.0 | [FoundationGeo, Table 2](https://arxiv.org/html/2607.11588) |

ScanNet, zero-shot relative error % ([MapAnything, Table 4](https://arxiv.org/html/2509.13414)):

| Model and inputs | rel % |
|---|---|
| MoGe-2, one image | 10.57 |
| MapAnything, one image | 27.77 |
| MapAnything, multi-view, images only | 22.23 |
| MapAnything, multi-view + K | 24.94 |
| MapAnything, multi-view + K + poses | 5.58 |

Licences and ids:

| Model | HF id | Licence | Note |
|---|---|---|---|
| MoGe-2 | `Ruicheng/moge-2-vitl-normal` | MIT | — |
| MoGe-3 | `Ruicheng/moge-3-vitl` | MIT | needs FlexGEMM, not installed |
| MetricAnything Student-PointMap | `yjh001/metricanything_student_pointmap` | Apache-2.0 | — |
| UniDepthV2 | — | CC BY-NC 4.0 | — |
| UniK3D | — | CC BY-NC-SA 4.0 | — |
| Depth Pro | — | Apple custom (apple-amlr) | — |
| Metric3D v2 | — | BSD-2 | — |
| DA3-Metric-Large | — | Apache-2.0 | metric depth = focal × output / 300 |
| DA3-Nested | — | CC BY-NC 4.0 | its metric scale is one pooled least-squares fit to DA3-Metric |

### 2.9 Scale anchors

| Anchor | Evidence | Expected scale error |
|---|---|---|
| **AprilTag 36h11 / ArUco 4×4, 180 mm, printed on A4** (rejected: D-067) | 0.167 m tags detected about 100% of the time to about 6–7 m; orientation error 0.2–1° at 10–70° off-axis ([AprilTag 2](https://april.eecs.umich.edu/media/pdfs/wang2016iros.pdf)) | About 0.2–1% within about 3 m (derived); works on any floor colour |
| Plain A4 sheet (D-013; opt-in, off by default since D-067) | D-017: detected 84% of the time when brighter than the floor, 20% on near-white floors | ±0.2% when found |
| Camera-height prior | ANSUR II standing eye height 151.95 ± 6.17 cm (women), 164.24 ± 6.65 cm (men); no study of how high phones are held | 7–9% SD for an unknown user (derived, assuming a 5–10 cm hold spread); about 3.5–7% once calibrated per user from tag photos |
| Door height | DIN 18101 rebated leaf 1985 mm; US 80 in, UK 1981 mm, India about 2100 mm (U); leaf vs opening is ambiguous | 2–6%, region-dependent |
| Language / size priors | Language-calibrated DPT on SUN-RGBD: AbsRel 0.147 vs 0.139 for an oracle least-squares fit ([arXiv 2601.01457](https://arxiv.org/html/2601.01457)) | Coarse |
| User tape measurement | Hover, Polycam Rescale, magicplan laser, Locometric | As good as the tape |

### 2.10 Ceiling height and single-view metrology

Sensitivities are derived for a camera at h = 1.5 m and a ceiling at H = 2.5 m.

| Formula | Inputs | Sensitivity |
|---|---|---|
| (a) H = h(1 + tan φc / tan φf) (HorizonNet, averaged over columns) | Floor-wall and ceiling-wall junctions of the same wall; camera height h | 8.3 / 10.6 / 16.5 cm per 1° of pitch at 2 / 3 / 5 m; 0.6–1.3 cm per pixel at 518 px. Pitch error cancels to first order between opposite walls of a spin (derived) |
| (b) H = h + d·tan φc | Tilted-up photo; d from the plan; h from the pose graph | 4.4–9.1 cm per 1°; a 3% error in d gives 3 cm |
| (c) H = D(tan φc + tan φf) / (tan φt + tan φf) | A door of known height D on the same wall, floor line visible | 0.4–1.0 cm per 1°; a 1% error in D gives 2.5 cm |
| (d) In-photo MoGe ratio (ceiling-plane distance / floor-plane distance) | One photo that sees both floor and ceiling | Independent of that photo's scale; the error is MoGe's shape error, unmeasured on our data |
| [Criminisi et al.](https://www.robots.ox.ac.uk/~lav/Papers/criminisi_etal_ijcv2000/criminisi_etal_ijcv2000.html) single-view metrology | Vertical vanishing point + vanishing line + one reference height | People's heights: 190.4 ± 3.27 cm (3σ) vs 190 cm true, with vanishing points at σ = 0.1 px |

Gravity accuracy is the limiting input for (a). GeoCalib's median errors on LaMAR phone images are pitch 0.87° and roll 0.28° ([GeoCalib](https://arxiv.org/html/2409.06704)). The floor-plane refinement should do better; measure it.

### 2.11 How commercial apps handle it (public sources)

| Product | Furniture occlusion | L-shaped / open plan | Glass / mirrors | Scale (camera-only) | Ceiling height | Humans in the loop |
|---|---|---|---|---|---|---|
| **Apple RoomPlan** (LiDAR) | See notes below the table | iOS 17: polygon floors, `sections` ([WWDC23](https://developer.apple.com/videos/play/wwdc2023/10192/)) | Glass and mirrored walls named as a challenge ([WWDC22](https://developer.apple.com/videos/play/wwdc2022/10127/)); the patent uses RGB for transparent windows | LiDAR | Lifted with the estimated wall height; up to 3.6 m | No |
| magicplan | The user aims at floor corners "through furniture and other obstacles" on an AR grid ([help](https://help.magicplan.app/scan-a-room-with-the-camera-of-your-mobile-device-ios)) | Corner mode handles any polygon; open plan = closed rooms with the shared walls then deleted | Glass walls: use Manual Scan | AR tracking; patent d = h / tan(tilt) with the user's height ([US9041796](https://patents.google.com/patent/US9041796)) | Grid dragged from floor to ceiling; default 2.44 m | No |
| Polycam | "Scan behind furniture where possible" | Editor | "Cover or avoid large mirrors and glass"; frameless shower glass shows up as open space ([help](https://learn.poly.cam/hc/en-us/articles/34295265957140-Improving-Scan-Accuracy-Best-Practices-for-Interior-Spaces)) | Photo models need the Rescale tool (two points + a known distance) | LiDAR | No |
| CubiCasa (video) | Keep baseboards in frame; furniture hiding the floor line lowers accuracy ([help](https://help.cubi.casa/en/articles/6662584-how-accurate-are-your-plans)) | Whole home in one scan | "scanning mirrors is perfectly okay" | Not documented | Shown, not editable | **Yes**: human QA "verifies these predictions and corrects any errors" |
| Hover (interiors) | Walk around the furniture | One room at a time; hallways captured separately | — | The user hand-measures a door or wall section ([help](https://help.hover.to/en/articles/9264961-how-to-scan-an-interior-space-universal)) | Minimum and maximum per room | Not stated |
| DocuSketch (360) | "see at least two sides of a corner"; floor and ceiling corners "work equally well" ([help](https://help.docusketch.com/docs/best-practices-for-quality-360-images-and-accurate-sketches)) | Extra 360° shots for offsets and blind corners | — | Fixed lens height | The user measures the lowest ceiling height and enters it | **Yes** |
| Matterport | De-Furnish: segment the furniture, "plane extension" behind it, then inpaint. Weakness: segmentation fails on wardrobes ([blog](https://matterport.com/blog/all-we-want-is-an-empty-room-panorama-inpainting)) | "treat each small area as its own room" | **Manual mirror and window marking** in the Capture app: "the app is told to ignore what it sees in the mirror" ([guidelines, 2020](https://assets.ctfassets.net/icnj41gkyohw/3KQeAKpcELDEzkWOQ6Nnrs/2308325ca2430bea93637bb10bd5a62c/Matterport-Capture_Service-Scanning-Guidelines.pdf)) | Depth camera, LiDAR or AI depth | Automated (announced 2023) | Schematic plans are a paid service |
| Zillow 3D Home / ZInD | Annotators correct the layouts | "An opening is an artificial construct that divides a large room"; visible vs complete layouts | Windows, doors and openings as boxes | Calibration target → camera height; fixed tripod height | From the layout + h | Annotators |

Apple RoomPlan, furniture occlusion ([Apple ML](https://machinelearning.apple.com/research/roomplan), [patent US20210225043A1](https://patents.google.com/patent/US20210225043A1/en)):
- **Pipeline.** A top-down semantic map plus a 512×512×12 z-slicing map (3 cm × 30 cm cells) feed corner and edge maps, then a line-verification step. The article claims it is "capable of accounting for wall occlusions".
- **Detection accuracy.** Precision and recall of 95% for walls and windows and 90% for doors. No dimensional accuracy is published.
- **Partitions.** The patent rejects walls that are not floor-to-ceiling.
- **Furniture.** Cabinets are the `storage` object class.

Vendor accuracy claims are not cm-level for camera-only capture:
- CubiCasa: 95–97% on average, "within 3%" with LiDAR.
- Polycam (LiDAR): ±½ inch.
- Matterport (U): 2–3% wall to wall.
- magicplan Corner Mode: about 10 cm error in a small room, the one independent number found ([MAVRiC 2020](https://mavricresearch.com/2020/11/10/magicplan-now-with-added-lidar-%F0%9F%A7%82/)).

## 3. Risks and what to test first

### Test order

Each step reuses the simulator's exact ground truth.

1. **Oracle semantics.** Ray-cast per-pixel classes from the USD meshes, and run the side fit with WALL-only points.
   - If the bathroom (1.49 vs 3.54 m) and bedroom (2.69 vs 3.67 m) do not recover, the problem is visibility, not classification. Then Picks 2 and 4 matter more than the model.
2. **EoMT on the failing photos.** Score it against the oracle (wall precision and recall, plus the confusions wardrobe→wall and cabinet→wall).
   - Then repeat with OneFormer, and with "both agree".
3. **0.5× spin vs 1× spin**, as a D-050-style A/B on the same seeds. Measure:
   - ceiling-line visibility;
   - per-photo MoGe scale MAD;
   - room error;
   - ceiling height by rules (a) and (d) against today's prior.
4. **Tag detection on light floors** at 1–4 m, at full resolution vs 518 px, and the resulting scale error.
   - Then the fusion with view-type sigmas: does the per-photo MAD of 5.6% shrink after dropping tilted-up photos?
5. **Polygon fit on the two L-shaped living-dining rooms**, with and without a second spin. Also run PolyLayout on the same posed photos.
6. **Glass and mirror rooms** (shower partitions, glass doors). See §1.5.

### Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Built-in wardrobes and fitted kitchens, flush with the wall and painted the same colour, get labelled "wall" | Matterport reports exactly this failure; wardrobe IoU is about 58% | Ceiling-reach test (Pick 2); inter-room gap check (Pick 4); a top surface seen below the ceiling marks furniture |
| Rare classes are noisy | Shower 3.6–24.4% IoU, counter 26–55%; no class exists for shower glass | Treat shower glass with Pick 5's detector, not with ADE |
| Removing furniture leaves a side with no wall points | The side goes "unseen" | Pick 2's upper band; Pick 4 flags it instead of mirroring |
| Simulator-to-real gap for the segmentation model | Isaac renders may be easier or harder than iPhone photos | Check 10 real photos by eye before trusting simulator scores |
| The 0.5× lens hurts MoGe scale or matching | D-017's untested claim; softer images; distortion correction | It is test 3; fallback 1× portrait |
| Tag problems | Printer scaling, IPPE flip at grazing angles, feet over the tag | Check bar; gravity-consistent IPPE solution; tag in at least 2 photos (existing rule) |
| False notches in the polygon fit where corners are hidden | Turns a rectangle into an L-shape | A notch needs wall evidence on both edges and no free space inside; tune α on simulator rooms with known shape |
| Ceiling ratio (a) is pitch-sensitive (about 10 cm per degree at 3 m) | Ceiling-height gate is 1.5 cm | Average opposite walls of a spin; prefer (d) and (c); use the floor-refined gravity |
| Licences | Room Envelopes NC; SAM 3 and DINOv3 gated; Mask2Former ADE weights NC | First picks are MIT, Apache-2.0 or geometry-only; disclose NC items if used for a test |
| `transformers` breaks the main env | It forces huggingface-hub < 2 (I-002) | Separate `envs/seg` and a worker process, as with PolyLayout |

## 4. Sources

Primary sources used above, by topic. Links in the tables point to the same pages.

**Layout and floor plans**
- Cabral & Furukawa, CVPR 2014: https://openaccess.thecvf.com/content_cvpr_2014/html/Cabral_Piecewise_Planar_and_2014_CVPR_paper.html
- HorizonNet: https://arxiv.org/abs/1901.03861, code https://github.com/sunset1995/HorizonNet
- LayoutNet: https://arxiv.org/abs/1803.08999
- Zou et al., comparative study: https://arxiv.org/abs/1910.04099
- NonCuboidRoom: https://arxiv.org/abs/2104.07986, code https://github.com/CYang0515/NonCuboidRoom
- uLayout: https://arxiv.org/html/2503.21562, code https://github.com/JonathanLee112/uLayout
- PolyLayout: https://arxiv.org/abs/2608.03323, code https://github.com/ghanning/PolyLayout
- PixCuboid: https://arxiv.org/abs/2508.04659
- Plane-DUSt3R: https://arxiv.org/abs/2502.16779
- SpatialLM: https://github.com/manycore-research/SpatialLM
- Room Envelopes: https://arxiv.org/abs/2511.03970, model https://huggingface.co/hugsam/room_envelopes_model
- cad-estate: https://arxiv.org/abs/2306.09077
- HouseLayout3D: https://arxiv.org/abs/2512.02450
- Coarse Semantic Injection: https://arxiv.org/abs/2605.16832
- FloorSAM: https://arxiv.org/abs/2509.15750
- Bi-Layout: https://github.com/LIAGM/Bi_Layout
- LGT-Net: https://arxiv.org/abs/2203.01824
- ZInD: https://github.com/zillow/zind
- Zillow capture guide: https://wp-tid.zillowstatic.com/bedrock/app/uploads/sites/2/2022/03/Capture-Guide_Zillow-3D-Home-Interactive-Floor-Plans_2021-3c0104-1.pdf

**Segmentation**
- EoMT: https://arxiv.org/abs/2503.19108, https://huggingface.co/tue-mps/ade20k_semantic_eomt_large_512
- OneFormer: https://github.com/SHI-Labs/OneFormer
- Mask2Former model zoo: https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md
- ADE20K class list: https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv
- Per-class log: https://huggingface.co/czczup/ViT-Adapter/raw/main/mask2former_beit_adapter_large_640_160k_ade20k_ss.log
- SAM 3: https://github.com/facebookresearch/sam3
- CAT-Seg: https://github.com/cvlab-kaist/CAT-Seg

**Depth, scale, metrology**
- MoGe-2: https://arxiv.org/abs/2507.02546
- MoGe-3: https://arxiv.org/abs/2607.17967
- MapAnything: https://arxiv.org/abs/2509.13414
- MetricAnything: https://arxiv.org/abs/2601.22054
- FoundationGeo: https://arxiv.org/abs/2607.11588
- UniDepthV2: https://arxiv.org/abs/2502.20110
- Depth Pro: https://arxiv.org/abs/2410.02073
- Metric3D v2: https://arxiv.org/abs/2404.15506
- Depth Anything 3: https://arxiv.org/abs/2511.10647
- GeoCalib: https://arxiv.org/abs/2409.06704
- Criminisi et al.: https://www.robots.ox.ac.uk/~lav/Papers/criminisi_etal_ijcv2000/criminisi_etal_ijcv2000.html
- AprilTag 2: https://april.eecs.umich.edu/media/pdfs/wang2016iros.pdf
- ANSUR II: https://www.abbottaerospace.com/downloads/natick-tr-15-007-2012-anthropometric-survey-of-us-army-personnel-methods-and-summary-statistics/
- iPhone 15 specs: https://support.apple.com/en-us/111831

**Glass and mirrors**
- MD-3k ("One Scene, Two Depths"): https://arxiv.org/abs/2606.29600
- LayeredDepth: https://arxiv.org/abs/2503.11633
- GlassGuard (SAM 3 on GSD-S): https://arxiv.org/abs/2610.02110
- GEM: https://arxiv.org/abs/2401.15282, weights https://huggingface.co/Bryceee/GEM-Glass-Segment-model
- L+GNet: https://arxiv.org/abs/2603.03718
- VGGT-Glass: https://arxiv.org/abs/2608.26752
- GDNet: https://www.cs.cityu.edu.hk/~rynson/papers/cvpr20d.pdf
- HetNet: https://arxiv.org/abs/2211.15644
- SAM3-UNet: https://arxiv.org/abs/2512.01789
- Mirror3D: https://arxiv.org/abs/2106.06629, code https://github.com/3dlg-hcvc/mirror3d
- Booster: https://arxiv.org/abs/2206.04671
- Depth4ToM: https://arxiv.org/abs/2307.15052

**Commercial**
- Apple RoomPlan: https://machinelearning.apple.com/research/roomplan
- Apple patent US20210225043A1: https://patents.google.com/patent/US20210225043A1/en
- WWDC22: https://developer.apple.com/videos/play/wwdc2022/10127/
- WWDC23: https://developer.apple.com/videos/play/wwdc2023/10192/
- magicplan: https://help.magicplan.app/scan-a-room-with-the-camera-of-your-mobile-device-ios, https://help.magicplan.app/magicplan-floor-plan-editor-faq
- Polycam: https://learn.poly.cam/hc/en-us/articles/34295265957140-Improving-Scan-Accuracy-Best-Practices-for-Interior-Spaces
- CubiCasa: https://help.cubi.casa/en/articles/6662584-how-accurate-are-your-plans
- Hover: https://help.hover.to/en/articles/9264961-how-to-scan-an-interior-space-universal
- DocuSketch: https://help.docusketch.com/docs/best-practices-for-quality-360-images-and-accurate-sketches
- Matterport: https://matterport.com/blog/all-we-want-is-an-empty-room-panorama-inpainting and the 2020 scanning guidelines PDF (linked in §2.11)
- MAVRiC magicplan test: https://mavricresearch.com/2020/11/10/magicplan-now-with-added-lidar-%F0%9F%A7%82/
