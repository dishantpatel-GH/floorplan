# Damage module (`floorplan/damage`)

> **5 Oct.** No staged damage was captured in the own flat, so the detector is scored on simulated decals only
> (`docs/COMPLIANCE.md`, row B2). It runs by default in `run_capture.py` (`--no-damage` turns it off).

> **v2 (2026-10-04) is at the end of this file** ("Damage v2"). It fixes crack length (I-6), replaces the saturating
> confidence (I-9), adds the photo-tier adapter, the `run_damage_on_plan` hook used by `scripts/run_capture.py`, and
> checks the staged paper/tape decals. Sections 1-8 below are v1, kept unchanged as history.
>
> **v3 (damage_precision, 2026-10-04) is the last part** ("Damage v3"). It removes a confirmed false water stain on
> clean floor_only: the gold lid of a jar on a shelf, now rejected because a stain must lie *on* the surface. It also
> makes the log report counts after two-tier filtering, and it corrects D-026's "no false stain" evidence.

Status (v1): working end-to-end on the LiDAR and video tiers. The photo tier is supported through `views.json` but is untested,
because no photo front-end output existed while this was built. Every number below comes from code that was run. The
evidence is in `outputs/damage/`.

## 1. Purpose and where it sits

Stage [5] of `docs/archive/PLAN.md`. It takes posed images from any tier, the aligned scene and the plan, and produces the
Part 2 output contract:

- **damage regions per surface.** Each region has a class (`water_stain` or `crack`), an area, a width × height and a
  position in surface coordinates (s = metres along the wall from `p0`; t = height above the room floor). Every one of
  these has a 95% interval, and each region also carries a detection confidence.
- **concealed-damage flags.** Each flag records the rule that fired, the measured evidence, and the probability that
  the geometric condition holds.
- **scope line items keyed to surface ids.** Quantities come from the plan's surface areas and the damage extents,
  with their intervals propagated.

Interface:

```python
from floorplan.damage import analyse_damage, views_from_stray, views_from_video
from floorplan.damage.project import Views
views = views_from_stray(capture_dir, scene)            # LiDAR tier
views = views_from_video(video_out_dir, scene, info)    # video tier (floorplan.video output)
views = Views.load(dir)                                  # any front-end that writes views.json + images/ (photo tier)
res = analyse_damage(views, scene, info, plan=plan_dict, out_dir=Path(...))
plan_dict["damage"], plan_dict["scope_items"] = res["damage"], res["scope_items"]   # + res["concealed_flags"]
```

`views.json` holds, per image, `{name, K (3x3, at that image's resolution), T_wc (4x4 camera-to-ALIGNED-world, OpenCV
axes)}`, plus `tier`, `pose_sigma_m` and `scale_sigma_rel`. The photo front-end only needs to write this file.

CLI: `python scripts/run_damage.py --capture <stray dir> | --video-scene <dir> | --views <dir> [--scene] [--plan]`.

## 2. How it works, step by step

1. **Surfaces** (`project.surfaces_from_plan`).
   - Each plan wall becomes a rectangle: wall length × room height, with t = 0 at the room floor.
   - Each room whose ceiling was *measured* gets a ceiling polygon. Floors are optional (`--floor`).
   - Door and window rectangles are cut out of the walls.
   - *Why the cut-outs:* a closed wooden door lies in the wall plane and is brown. Without the cut-out it looks exactly
     like a huge water stain.
   - With no plan, `surfaces_from_scene` fits axis-aligned wall planes straight from the Manhattan-aligned TSDF (ids
     `P1..Pn`). Walls longer than 4 m are tiled to bound memory.
2. **Views.**
   - All tiers reduce to images + K + camera-to-aligned-world poses.
   - LiDAR: up to 160 keyframes at 960 px wide, which gives 2.5 mm/px at 2 m, finer than the ortho grid.
   - Video: the video tier's upright keyframes, with its focal length and its scale uncertainty (`sigma_rel_total`).
3. **Occlusion** (`project.zbuffer`).
   - Each view gets a z-buffer of the fused scene points at 1/8 resolution, closed with a 3×3 min-filter. Holes count
     as occluded.
   - This needs no depth sensor, so it works the same way for all tiers.
   - *Without it,* sofa fabric was painted onto the wall behind it (first test, I-1).
4. **Orthophoto per surface** (`project.build_ortho`). The output is a rectified texture of the surface at 5 mm/px:
   - The 16 best views are chosen by coverage × frontality ÷ distance.
   - Every ortho pixel is projected into each view and sampled. A sample is rejected when the point is occluded, at a
     grazing angle (< ~15°), or more than 4 m away.
   - Exposure is equalised with one gain per view, and the views are fused by a **per-pixel median**.
   - *Why:* one pixel is a fixed 25 mm², so extents are pixel counts. The median removes things that are not on the
     surface (glare, people). The per-view stack is kept for the consistency checks.
5. **Background colour model** (`detect.background`).
   - A masked Gaussian (σ = 0.2 m) of L\*a\*b\*, re-weighted 3 times so that outliers (the damage) do not leak in.
   - Slow lighting gradients across a wall are therefore not anomalies.
6. **Water stains** (`detect.detect_stains`), *chroma-first*:
   - **Seeds:** pixels **yellower** (b\*) than the background by more than 4 robust σ and more than 3 b\* units, and
     darker.
   - **Growth:** hysteresis down to 2σ.
   - **Outline:** re-drawn at **half the stain's own median b\* shift** (a half-maximum edge).
   - **Rejection rules**, each one logged:
     - darkening without a yellow shift (shadow or grime; b\* shift < max(2, 1.5σ, 0.25·|ΔL|));
     - yellower but not darker;
     - thin strip (short side < 4 cm or aspect > 5: frames, corner shading);
     - rectangle with straight edges (picture frames, switches, panels);
     - larger than 3 m² or half the surface (a material change);
     - hugging an unseen patch (occluder halo);
     - fewer than 4 agreeing views, or support < 0.7;
     - views disagree around it (median cross-view spread > 8 L\*: mirror or glass).
7. **Cracks** (`detect.detect_cracks`):
   - **Ridge filter:** multi-scale Sato filter for dark ridges, σ = 1–2 px. Thresholds are noise-relative with absolute
     floors: 3.0 for seeds and 1.5 for growth, about a line 7–8 L\* darker.
   - **Pre-processing:** straight horizontal/vertical runs of 5 cm or more are cut away *before* labelling, so a crack
     touching a frame is not merged with the frame. Fragments within 1 cm are then grouped and skeletonised.
   - **Rejection rules:**
     - straight and axis-aligned (an edge or a tile joint);
     - more than 40% on axis runs (an outline);
     - wider than 2 cm;
     - occluder halo;
     - not darker than **both** sides by 1.5 L\* (a step edge is darker on one side only);
     - fewer than 4 agreeing views, or a view disagreement;
     - lying on the outline of a detected stain (a tide line);
     - on a textured surface (more than 1 m of passing lines per m² and at least 4 of them): flagged as unreliable, not
       silently cleared.
8. **Multi-view consistency** (`_view_support`).
   - A view supports a region if its own contrast (region vs a 3 cm ring) is at least 40% of the consensus contrast.
   - Paint is seen from every viewpoint; glare and reflections move with the camera.
   - The photo tier lowers the required number of agreeing views to 2 (it has only 2–8 photos per room). The weaker
     evidence then shows up in the confidence, which scales with the view count.
9. **Metric description with intervals** (`detect.describe`). The 1σ error budget has these terms:
   - boundary: sqrt((res/2)² + pose_sigma²), with pose_sigma = 5 mm for LiDAR and 20 mm for video;
   - threshold sensitivity: half the area change between a 0.7× and a 1.4× growth threshold, also spread over the
     perimeter for width and height;
   - scale: scale_sigma_rel × size, which is 0 for LiDAR and the video front-end's measured value otherwise (area takes
     2× because it scales with length²).

   For cracks, the length interval's upper bound is 2× the measured skeleton (see I-6).
10. **Rules** (`rules.py` + `rules.json`, which is data that a restorer can edit). Flag confidence = detection confidence
    × p_condition, where p_condition is the normal-CDF probability that the threshold holds given the position interval.
11. **Scope** (`scope.py` + `scope_catalog.json`).
    - Line items per surface. Wall area = plan wall length × room height, with the plan's intervals propagated.
    - Additive items (crack length, cut-outs) are summed.
    - The repaint is de-duplicated: one per wall.

## 3. Decisions

**D-dmg-1: detector = classical anomaly on orthophotos + multi-view verification. No zero-shot model and no VLM in
the default path.**

| Option | For | Against | Verdict |
|---|---|---|---|
| (1) Classical anomaly on orthophotos | Metric by construction (pixel = 25 mm²); every rejection is an explicit, logged rule; runs cold and offline; deterministic | Hand-set thresholds; no semantics (cannot name mould) | **chosen** |
| (2) OWLv2 / Grounding DINO + SAM | Semantics, open vocabulary | Needs `transformers` → separate env (I-002); ~3–6 GB of GPU memory on a shared 8 GB GPU; per-image boxes still need the same projection, consistency and metric machinery; zero-shot "water stain" prompts tend to fire on shadows and wood grain in clean rooms (not measured here for lack of time) | not built; possible proposer later |
| (3) VLM API | Best semantics | Network + key at the walk-in; non-deterministic; no metric output; disclosure burden | rejected as a default |

- **Evidence that (1) is enough for the two staged classes:**
  - **specificity:** 0 false positives on the three clean LiDAR captures (114 m² analysed);
  - **injection:** stain recall 8/12 with 100% class accuracy, a median area error of 1.9% and 100% interval coverage;
    crack recall 7/9 (section 5).
- The time box (2.5 h) also ruled out building and fairly evaluating (2). Every rule is cheap to explain.

**D-dmg-2: chroma-first stain segmentation.** Segmenting on darkness failed in the first injection run:
- stains merged with neighbouring shadows, and pale stains on noisy surfaces were missed;
- stain recall was 3/12.

The yellow/brown b\* shift is the physical signature of a water stain (tannins and minerals carried to the surface),
and shadows do not have it. After the switch, stain recall rose to 8/12 (I-4).

**D-dmg-3: half-maximum outline.** The hysteresis mask on the 1 cm-smoothed map overstated the stain:
- width +2 cm, area +10–28%, and 0/8 width/height intervals covered the truth.

With the half-maximum outline:
- median area error 1.9%, signed error +0.6%;
- area interval coverage 8/8, width/height coverage 13/16 (I-5).

**D-dmg-4: z-buffer of the fused scene for occlusion, not sensor depth.** It is tier-agnostic: the video and photo tiers
have no sensor depth, and it uses the same code path for all three tiers.

**D-dmg-5: rules and scope as JSON data.** They can be reviewed by a domain expert, and each rule carries its rationale
and reference.

**D-dmg-6: pin the plan in injection runs.** Plan extractors rewrite `plan.json` and its wall ids while they work, so
`inject_damage.py` copies the plan it used next to the truth (I-3).

**D-dmg-7: thresholds were tuned on the clean sample (specificity) and checked on injection.** The thresholds were *not*
tuned to the injected shapes; the decal parameters were fixed before any tuning. Disclosed caveat: the rejection rules
were added by looking at false positives on `single_room` and then checked on `floor_only` and `with_ceiling`, which
are partly the same apartment.

## 4. Issues log

- **I-1. Sofa texture on the wall orthophoto.**
  - **Root cause:** z-buffer gaps between 2 cm voxels at 1/4 resolution, with holes treated as visible. The fabric
    appeared speckled in the first test sheet.
  - **Fix:** 1/8 resolution + 3×3 min-filter + holes treated as occluded.
- **I-2. Crack detector fired everywhere** (5–19 m of "cracks" per m² on clean walls).
  - **Root cause:** the threshold was relative to the robust σ of the ridge response, which is about 0 on flat paint,
    so any edge passed.
  - **Fix:** absolute floors, calibrated on a synthetic line (Sato response ≈ 0.35–0.5 × contrast); a two-sided contrast
    test against step edges; the outline (axis-run) rule.
  - **Result:** clean `single_room` went from 13 detections to 1, then to 0 after I-7.
- **I-3. All injected truths "missed" and many false positives in one run.**
  - **Root cause:** the plan_alpha agent rewrote `plan.json` between injection and detection, and the wall ids changed
    from `r0_w3` to `R1-W6`.
  - **Fix:** the plan is pinned into the injection folder and the clean reference uses a pinned copy
    (`outputs/damage/_plans/`).
- **I-4. Stains merged with shadows, or were missed** (first injection run: 3/12).
  - **Fix:** chroma-first segmentation (D-dmg-2), giving 8/12.
- **I-5. Stain extents biased +2 cm and intervals did not cover the truth.**
  - **Fix:** half-maximum outline + a threshold term in the width/height interval (D-dmg-3).
- **I-6. Crack length underestimated** (median signed error −54%; interval coverage 3/7).
  - **Root cause:** a 1.5–3 mm crack at 5 mm/px, median-fused over slightly misregistered views, drops below the ridge
    threshold along faint stretches, so only part of the crack survives.
  - **Possible fixes:**
    - (a) orthophoto at 2–3 mm/px for crack search (about 3× the cost);
    - (b) per-view ridge detection, then vote on the surface;
    - (c) widen the interval.
  - **Chosen for now:** (c), honestly: the upper bound is 2× the skeleton length. (a) is the next step.
  - Width and height of crack boxes are still underestimated (1/14 intervals covered), for the same reason.
- **I-7. One crack on a dark tile area near the floor accepted with support 0.67** (clean `single_room`).
  - **Fix:** `min_support` raised from 0.6 to 0.7.
  - **Effect:** clean false positives went to 0. Injection recall was unchanged for cracks; one stain that was only
    matched through a crack fragment dropped out.
- **I-8. Stain tide lines detected as 2–6 "cracks" around every stain.** Seen in `figures/injection_detections.jpg`.
  - The scorer at first hid these because it matched any overlapping class.
  - **Fix:** cracks lying on an accepted stain's outline are rejected, and the scorer now counts wrong-class overlaps as
    false positives.
  - **After the fix:** 0 false positives on the injected sets.
- **I-9 (open). Video tier: 1 crack false positive with confidence 1.0** (0.10 m, `single_room` video, surface `P9`).
  - **Root cause (likely):** lower-resolution keyframes (768 px) and noisier fused geometry, while the confidence
    saturates because the crack z-score is relative to a tiny ridge noise.
  - **Next:** base crack confidence on the two-sided L\* contrast and length instead of the z-score, and add a tier
    factor.

## 5. Results

**Specificity on the clean sample (all three LiDAR captures, plan surfaces).**

| Capture | Tier | Surfaces | Analysed m² | Accepted regions (all false) | Candidates rejected by rules |
|---|---|---|---|---|---|
| single_room | LiDAR | 19 | 23.0 | **0** | 26 |
| single_scan_floor_only | LiDAR | 57 | 37.5 | **0** | 79 |
| single_scan_with_ceiling | LiDAR (walls + measured ceilings) | 85 | 53.4 | **0** | 92 |
| single_room | video (scene-fallback planes) | 9 | 11.7 | **1 crack (conf 1.0)** | 23 |
| single_scan_floor_only | video (scene-fallback planes) | 9 | 6.7 | **0** | 9 |

- **Files:** `outputs/damage/<capture>[__video]/damage.json` (it includes `rejected_detail`, which lists every rejected
  candidate with its rule) and `surfaces/*.jpg`.
- **Analysed area is the area actually seen by at least 3 views,** not the wall area. High walls and ceilings in
  `floor_only` were not filmed.

**Injection** (`single_room`, 3 seeds × (4 stains + 3 cracks), painted into 160 LiDAR keyframes, detector run blind).
Results are in `outputs/damage/injection_summary.json`, with per-seed `inject_eval/*/score.json`.

| | Water stain | Crack |
|---|---|---|
| Recall | **8/12 (67%)** | **7/9 (78%)** |
| Class accuracy (detected) | 100% | 100% |
| Median abs. error (area / length) | **1.9%** (signed +0.6%) | 54% (signed −54%, I-6) |
| Truth inside the 95% interval (area / length) | **8/8** | 3/7 |
| Truth inside the width/height intervals | 13/16 | 1/14 |
| Median width error | 3 mm | 87 mm |
| Median centre error | 9 mm | 53 mm |
| Mean confidence of true positives | 0.93 | 0.98 |
| False positives on the injected sets | 0 | 0 |

- **Misses:**
  - stains on a dark tile wall (low contrast), behind the glass shower screen (a transparent surface), and on the dark
    fridge door;
  - cracks on a surface seen by too few views.
- **Figure:** `outputs/damage/figures/injection_detections.jpg`. Red = detected stain, green = detected crack, thin
  blue/magenta = rejected candidates.
- **Rules and scope on injected data:**
  - `WALL_BASE_WICKING` fired on both wall-base stains (seeds 0 and 2), with bottoms 6 cm and 3 cm above the floor.
  - Scope example (seed 0):
    - `r0_w3: stain-block primer + 2 coats paint, whole wall 1.62 m2 [1.46-1.79]`
    - `r0_w3: flood cut: remove and replace drywall to 0.6 m along the wall, dry cavity 0.61 m2 [0.60-0.63]`
    - `r0_w3: remove and refit skirting board 1.02 m [1.00-1.05]`
- **Runtime:** 1–4 min per capture on CPU (no GPU used).

## 6. Limitations, failure modes and next steps

- **Glass and mirrors.**
  - The orthophoto of a mirror shows the reflected room. The view-spread rule and the support rule reject most of it.
  - Stains *behind* glass are missed (an injected one was).
  - LiDAR returns through glass can put a plan wall in the wrong place.
- **Dark or patterned surfaces** (tiles, wood, a dark fridge): low contrast, so stains are missed. Tile grout is
  removed by the axis-run rules. Marbled textures trigger the "textured, unreliable" flag.
- **Low light:** noise raises the robust σ, so faint stains fall below threshold (fewer detections, not false ones).
  Exposure differences are handled by per-view gains.
- **Clutter:** occluded wall is not analysed. The `surfaces[].analysed_m2` field reports what was actually checked, so
  "no damage" always means "none on the X m² we saw".
- **Classes:** only water stains and cracks. Mould, peeling paint and holes are not detected. The rules for mould and wet
  rooms exist, but need a mould class or room labels; the plan currently labels every room "room".
- **Thin cracks:** hairline cracks (< 1 mm) are below the 5 mm ortho resolution, and crack length is underestimated
  (I-6).
- **Photo tier:** untested (no photo front-end output yet). Fewer views mean weaker consistency evidence. With 2–8
  photos per room, many wall patches are seen by only 1–2 photos and are reported as not analysed.
- **Staged paper decals (tomorrow):**
  - A printed stain on white paper is a yellow/brown blob, which is exactly what the stain detector looks for.
  - The paper edge is near-invisible on a white wall but can produce a thin rectangle on a coloured wall; the
    rectangle and strip rules reject the edge, not the stain.
  - A printed crack is usually darker and wider than a real hairline crack, which helps detection.
- **Next steps:**
  - crack search at 2.5 mm/px;
  - contrast-based crack confidence (I-9);
  - plane-deviation geometry for holes and bulges from LiDAR;
  - an optional OWLv2+SAM proposer in a separate env, used only as an extra proposer that goes through the same metric
    and consistency checks.

---

# Damage v2 (2026-10-04)

Everything below was added in v2. Every number comes from code that was run; the evidence is in `outputs/damage/v2/`
(runner `run_all.sh`, ablation `ablate.sh`, hook test `hook_test.py`, logs in `logs/`, pooled metrics and the
confidence sweep in `injection_summary_v2.json`, figure `figures/staged_and_injected_detections.jpg`).

## v2.1 Purpose of v2

Six open problems from v1:

1. Crack length was underestimated by 54% (I-6), and its interval was widened by hand (×2 upper bound).
2. Confidence saturated at 1.0, so the video tier reported a false crack with confidence 1.0 (I-9).
3. The photo tier was untested.
4. `scripts/run_capture.py` needs a one-call hook.
5. Tomorrow's real staged decals must survive the rejection rules:
   - a brown/yellow blotch on white paper taped to a wall;
   - a jagged line drawn on beige masking tape.
6. Full re-validation.

## v2.2 How it works now (only what changed), step by step

**Cracks**

1. **Coarse search (5 mm/px, whole surface), as in v1, with three changes:**
   - **Step edges are removed pixel by pixel** (`two_sided_map`).
     - Test: a ridge pixel must be at least 1.5 L\* darker than *both* sides, 1 cm away, in its best of 4 directions.
     - Why: the Sato filter also fires on the dark side of a step edge (tape border, paint/tile boundary). In v1 such
       an edge merged with a real line, and the merged region was rejected as an outline.
   - **Only straight horizontal/vertical runs of 12 cm or more are cut before grouping** (v1 cut 5 cm runs).
     - Why: the 5 cm cut removed near-horizontal crack *branches*.
   - **Fragments are grouped by gap bridging** (`link_fragments`), not by a 1 cm dilation. A fragment's end is
     linked to another fragment:
     - in any direction if it is within 1 cm (junctions, where Sato dips);
     - within 5 cm if it lies inside a ±35° cone of the direction the end points to (collinear gaps).
     - Only fragments with at least 1 cm of line take part (no specks or tape corners).
     - The bridges are drawn into the mask, so the skeleton runs continuously across the gap.
2. **Fine re-tracing at 2.5 mm/px** (`trace_crack`).
   - **Window:** every coarse candidate that is not an obvious straight axis-aligned edge gets its own orthophoto,
     covering its bbox plus a margin of 5 cm + half its length (at most 30 cm). Building it uses the same views,
     z-buffer and median as the coarse one.
   - **Ridge and noise:** Sato σ = 1–3 px on L\*, plus the same pixel two-sided test. The noise level is measured in
     the window, away from the candidate.
   - **Growth:** hysteresis from the coarse skeleton, down to max(4σ, 1.0), with gap bridging.
   - **Threshold ladder:** the trace must not "explode". It explodes if it grows longer than 1.6 × the coarse length
     + 10 cm, or its straightness drops below 0.6 × the coarse value. If it explodes, the threshold is raised
     (×1.4, ×2, ×2.8).
   - **Skeleton:** spurs shorter than 1 cm are pruned.
   - **Length:** the **minimum spanning tree** of the 8-connected skeleton graph. Branches are included.
   - **Threshold sensitivity:** the length is measured again at 0.7× and 1.4× the threshold; variants that explode
     are ignored.
3. **Rules on the refined measurement** (`_crack_rule`). These are the v1 rules plus:
   - **tortuosity:** span/length must be at least 0.70;
   - the **outline** (axis-run) rule counts runs of 8 cm or more on the fine skeleton.

   Two questions are asked on the **coarse surface grid**:
   - **multi-view support**, because 5 mm is larger than the registration error;
   - **occluder halo**, because the fine window chooses its own views, which can see round an occluder.
4. **Length interval:**
   σ² = 2·σ_boundary² + (threshold sensitivity)² + (0.10·L)² + (scale·L)².
   - The 10% term is *trace completeness*: faint ends and branches are found or missed as whole pieces.
   - It was set from the v2 injection residuals, which is disclosed as calibrated on injection.
   - v1's hand-made "×2 upper bound" is gone.

**Confidence** (`detect.confidence`) is a product of terms in physical units:
- **crack:** s(two-sided contrast; 4 L\*, 1.5) × s(length; 10 cm, 2.5 cm) × straightness penalty above 0.97 × agreement;
- **stain:** s(local b\* shift; 3, 1) × agreement;
- **agreement** = support × (1 − exp(−n_agreeing/3)).

Regions with confidence below `min_confidence` = **0.10** are rejected and logged as "low confidence".

**Stains**

- Colour shift is measured against the stain's **immediate 3 cm ring**, not the 0.2 m background model.
- The half-maximum outline is drawn half-way between the core and that ring.
- *Why:* a stain on white paper must be judged against the paper.
- A second, *local* (5 cm) colour model for **seeds** exists (`stain_local_sigma_m`) but is **off** (I-18).
- The tide-line rule now uses only the stain's **outline band**, so a crack drawn *inside* a discoloured patch (a line
  on tape) is not rejected as a tide line.

**Tiers and the hook**

- `views_from_video_file`: decodes the keyframes again from the video file at 1280 px using the scene's `kf` (which
  are frame indices) and the final upright rotation. It needs no front-end work directory.
  - Validated against the front-end's own frames: mean absolute difference < 1 grey level, same poses, same
    focal ratio.
- `views_from_photos`:
  - **Images:** the original photos (EXIF-upright, HEIC ok, ≤ 2048 px).
  - **Poses:** the photo scene's `T_wc`, which are **already in the aligned frame** (unlike LiDAR and video).
  - **K:** EXIF 35 mm focal on the long side, principal point at the centre (the front-end's own model).
  - **Validated:** scene points projected into the photos have colours that correlate 0.92–0.995 with the photo
    pixels.
  - **Photo-tier settings:** `min_views = 2`, `min_support_views = 2`, `pose_sigma = 3 cm`, and the front-end's scale
    σ.
- `run_damage_on_plan(plan, scene, info, input_path, tier, log=print, out_dir=None)`:
  - It fills `plan.damage` and `plan.scope_items` in the schema shape (`to_plan_items`):
    - damage regions carry `class`, `extent{area,width,height,length,center_s,center_t}` and `confidence`;
    - concealed-damage flags become damage entries with `concealed: true` and the `rule` that fired;
    - scope items carry `description`, `quantity` and `damage_ids`.
  - It **never raises.** `plan.meta["damage"]` records either the coverage summary (analysed m², rejected candidates
    by rule, unanalysed surfaces) or the error string.

**Simulation of tomorrow's staged decals** (`scripts/inject_damage.py make --paper-stains N --tape-cracks N`):
- **Paper stain:**
  - A4 sheet, tilted ±5° (hand-taped), 2–15% brighter and bluer than the wall (optical brighteners).
  - A 3 mm contact shadow along its lower and right edges.
  - A blotch from v1's stain generator inside it.
  - Only placed on light surfaces: the decal is rendered as a reflectance ratio (I-19).
- **Tape crack:**
  - A strip of masking tape 18–50 mm wide and 25–45 cm long, horizontal, vertical or oblique, beige (yellower and
    darker than the wall).
  - On it, a zigzag pen line 1–2.5 mm wide, segments 1–2.5 cm long, swinging up to 35% of the tape width.
  - The truth is the line, not the tape.

## v2.3 Decisions

**D-dmg-8: refine cracks locally at 2.5 mm/px, not whole surfaces at 2.5 mm.**

| Option | For | Against | Verdict |
|---|---|---|---|
| (a) Whole-surface orthophoto at 2.5 mm | simple | 4× memory and time for every wall, mostly blank paint | rejected |
| (b) Per-view ridge detection + vote | uses native pixels | needs per-view ridge registration onto the surface; more code | not built |
| (c) Coarse search, then fine window per candidate | cost ∝ candidates; same projection code | a crack missed entirely at 5 mm is not recovered | **chosen** |

- Evidence on injected crack s0-T6:
  - v1 measured two fragments of 0.203 + 0.209 m and missed the branch;
  - v2 measures 0.541 m against a true 0.530 m.
- The fine orthophoto *alone* was not the fix: the crack was fully visible in the 5 mm ridge map. The losses came from
  grouping (I-10).

**D-dmg-9: gap bridging with direction cones instead of dilation.**
- A 1 cm dilation split cracks at Y-junctions. A larger dilation would chain parallel texture lines (wood grain).
- The cone (±35°, 5 cm) links only ends that point at each other. The 1 cm "any direction" radius handles junctions.

**D-dmg-10: MST skeleton length** instead of counting neighbour pairs. Counting double-counts staircase corners; the MST
keeps one path per pixel and includes branches.

**D-dmg-11: confidence from physical terms, threshold from a precision/recall sweep.**

The sweep (`inject_damage.py summary`) pools 6 injected sets (15 cracks, 18 stains) and the 9 clean tier runs. It
drops detections below τ:

| τ | stain recall | crack recall | FP on injected sets | FP on clean captures | precision |
|---|---|---|---|---|---|
| 0.00 | 0.61 | 0.60 | 0 | 1 (video, conf 0.067) | 0.95 |
| **0.10 (chosen)** | **0.61** | **0.60** | **0** | **0** | **1.00** |
| 0.20 | 0.61 | 0.53 | 0 | 0 | 1.00 |
| 0.30 | 0.61 | 0.47 | 0 | 0 | 1.00 |
| 0.50 | 0.50 | 0.40 | 0 | 0 | 1.00 |
| 0.70 | 0.39 | 0.20 | 0 | 0 | 1.00 |

- **Why τ = 0.10:** it is the smallest τ with zero false positives. Recall is the same as at τ = 0.
- **Calibration status:** with 20 true positives and 1 false positive the scores **cannot** be calibrated into
  probabilities (no reliability curve is possible). The confidence is a monotone *evidence score*, and it is
  reported as such.
- **What it fixed:** v1 gave 1.0 to everything. In v2:
  - true-positive cracks span 0.18–0.82 and stains 0.37–0.98;
  - the v1 video false crack (1.0) is gone.

**D-dmg-12: pixel-level two-sided test.** It suppresses step edges *before* grouping. Without it, a jagged line drawn on
masking tape merged with the tape's straight borders and was rejected as a rectilinear outline (I-13).

**D-dmg-13: tortuosity rule (span/length ≥ 0.70).**
- **Ablation** (`outputs/damage/v2/ablation/`): v2's extra sensitivity — the two-sided pixel test, 5 cm bridging and
  the 12 cm cut — brought back 2–5 clean-wall false cracks per capture.
- **Feature table** (all clean-run false positives vs all injected true positives, pre-rule run): straightness was the
  separating feature.
  - false positives: 0.49–0.69 (object outlines, magnets, cables at occlusion borders; two at 0.89 and 0.97);
  - true positives: 0.69–0.95.
- **Cost:**
  - one branched injected crack (s0-T6, straightness 0.69) is now rejected;
  - map-pattern crazing would be rejected too (limitation).

**D-dmg-14: local stain seeds off.**
- With them on, one more paper stain was found, and 4 clean-sample false stains appeared (conf 0.20–0.67).
- With them off, there are 0 false stains, but paper-stain recall is 3/6 in simulation.
- Accuracy-first rule: no confident garbage.

**D-dmg-15: video views decoded from the video file.** The hook only gets `input_path` and not the run's work directory.
The scene's `kf` are frame indices, so decoding them again is exact (checked) and gives 1280 px instead of 1024 px.

**D-dmg-16: photo tier needs 2 views per pixel**, not 3. With 6–8 photos per room (D-017) most wall patches are seen by
two photos. The agreement term in the confidence carries the weaker evidence.

## v2.4 Issues log (v2)

- **I-10 (root cause of I-6). Crack split at its junction, branch lost.**
  - **Symptom:** s0-T6 measured as 2 regions of 0.20 m each (truth 0.53).
  - **Root cause (evidence: diagnostic ridge-map crops of s0-T6; not kept in outputs):**
    - the 5 mm ridge map showed the whole crack;
    - Sato dips at the Y-junction, leaving a gap of about 2 cm, larger than the 1 cm grouping;
    - the near-horizontal branch was removed by the 5 cm axis-run cut.
  - **Fix:** gap bridging (D-dmg-9) + a 12 cm cut. **Result:** 0.541 m.
- **I-11. The fine trace leaked into texture** (s2-T5: 1.25 m traced for a 0.39 m crack; fridge magnets: 1.6 m).
  - **Fix:** threshold ladder with an "explosion" test (v2.2 step 2). **Result:** 0.362 m.
- **I-12. A crack crossing a paint/tile step edge was broken in two** (s1-T5: 0.29 + 0.20 m).
  - **Root cause:** the step edge's ridge band (about 2 cm thick) is cut away, so the gap is about 3.6 cm, more than
    the 3 cm link.
  - **Fix:** a 5 cm collinear link. **Result:** one region, 0.525 m (truth 0.51).
- **I-13. Tape decal never refined.** The zigzag merged with the tape borders, and at 5 mm/px with a 3 px tolerance it
  looked 59–92% "rectilinear".
  - **Fix:** pixel two-sided test + the outline rule decided on the fine skeleton (8 cm runs).
  - **Result:** staged_s0-T4 measured 0.341 m (truth 0.340).
- **I-14. Tape corners and specks were bridged onto the trace** (+10 cm). **Fix:** only fragments of 1 cm or more
  are linked.
- **I-15. False cracks at occlusion borders.** The fine window picked views that see round the occluder.
  **Fix:** the halo is judged on the coarse, surface-wide visibility.
- **I-16. Real tape lines had support 0.4–0.7 at 2.5 mm.** A 1–2 mm line drifts in and out of a 1 px band across
  views. **Fix:** support is measured on the coarse grid.
- **I-17. v2 sensitivity produced 16 clean-wall false cracks** (all tiers). **Fix:** tortuosity rule (D-dmg-13) +
  τ = 0.10. **Result:** 0.
- **I-18. Local stain seeds produced 4 clean false stains.** **Fix:** off (D-dmg-14).
- **I-19. Paper rendered as a brightness ratio is wrong on dark surfaces** (paper on a dark cabinet stayed dark). This is
  a simulation-only issue. **Fix:** paper decals are placed only where the scene colour median is 120/255 or more.
- **I-20 (open). Video `with_ceiling`:** the scene-fallback plane finder found only 1 wall plane (0.54 m² analysed).
  Run with a plan, this does not arise. The fallback's band (floor + 0.15 … 2.2 m) probably misses the walls of that
  scene (floor_y = −3.53, ceiling = −0.49).
- **I-9 (v1) closed.** The single_room video run has 0 accepted regions (v1: 1 crack at 1.0).

## v2.5 Results

**Specificity on the clean sample** (final code; evaluation run with τ = 0 so that every candidate is visible; the
reported numbers use τ = 0.10).

| Capture | Tier | Analysed m² | Accepted (all false) v1 → v2 | Rejected candidates |
|---|---|---|---|---|
| single_room | LiDAR | 23.0 | 0 → **0** | 81 |
| single_scan_floor_only | LiDAR | 37.5 | 0 → **0** | 163 |
| single_scan_with_ceiling | LiDAR | 53.4 | 0 → **0** | 186 |
| single_room | video (1280 px frames from the video file) | 10.0 | **1 (conf 1.0) → 0** | 25 |
| single_scan_floor_only | video | 15.9 | 0 → **0** (1 at conf 0.067, below τ) | 97 |
| single_scan_with_ceiling | video (fallback planes, I-20) | 0.5 | – → 0 | 3 |
| single_room | photo (2 placed photos, 1 per room) | **0.0** (every wall seen by 1 photo) | – → 0 | 0 |
| single_scan_floor_only | photo (11 placed) | 5.0 | – → **0** | 31 |
| single_scan_with_ceiling | photo (14 placed) | 9.3 | – → **0** | 24 |

**Injection** (6 sets, painted into 160 LiDAR keyframes, detector run blind):
- the 3 v1 sets (`single_room_s0-2`: 12 stains, 9 cracks);
- 3 *staged* sets (`staged_s0-2`: 6 paper stains, 6 tape cracks).

| | Water stain v1 → v2 | Crack v1 → v2 |
|---|---|---|
| Recall (all 6 sets) | – → **11/18** (v1 sets 8/12, paper 3/6) | – → **9/15** (v1 sets 5/9, tape 4/6) |
| Recall on the v1 sets | 8/12 → 8/12 | 7/9 nominal (6/9 real: one "hit" was a fridge magnet matched by bbox tolerance) → 5/9 |
| Median abs. error (area / length) | 1.9% → **2.0%** | **54% → 5.0%** |
| Median signed error | +0.6% → +0.9% | **−54% → −4.5%** |
| Truth inside the 95% interval (area / length) | 8/8 → **11/11** | 3/7 → **7/9 (78%)** |
| Truth inside width/height intervals | 81% → 82% | **7% → 83%** |
| Median centre error | 9 mm → 9 mm | **53 mm → 15 mm** |
| False positives on the injected sets | 0 → 0 | 0 → 0 |
| Confidence of true positives | 0.93 (saturating) → 0.37–0.98 | 0.98 (saturating) → 0.18–0.82 |

Per-item details for the staged decals:

| Decal | Truth | Measured | Notes |
|---|---|---|---|
| paper stain staged_s2-T1 | 0.0124 m² | 0.0126 m² | paper not reported, conf 0.80 |
| paper stain staged_s1-T2 | 0.0164 m² | 0.0166 m² | conf 0.98 |
| paper stain staged_s0-T2 | 0.0165 m² | 0.0126 m² | −24%, still inside its interval |
| paper stain × 3 | – | missed | no seed formed (local seeds off, D-dmg-14) |
| tape crack staged_s2-T3 | 0.465 m | 0.445 m | |
| tape crack staged_s2-T4 | 0.455 m | 0.435 m | |
| tape crack staged_s0-T4 | 0.278 m | 0.264 m | |
| tape crack staged_s0-T3 | 0.328 m | 0.222 m | interval missed |
| tape crack staged_s1-T3, -T4 | – | missed | one on the dark fridge, rejected as tortuous after leaking |

In every staged set **the tape strip itself was rejected as a stain** ("thin strip").

**Hook** (`outputs/damage/v2/hook_test.py` + `logs/hook_test.log`):
- `run_damage_on_plan` on a `Plan` object from `outputs/runs/single_room__c00a170fe1/lidar` → `save_plan_json`
  reports **0 schema problems**;
- a missing photo folder gives `meta.damage = {ok: false, error: "FileNotFoundError: ..."}`, `damage: []`, and no
  exception.

**Runtime** (CPU, 22 cores shared by 6–9 parallel runs): LiDAR 110–341 s per capture, video 25–45 s, photo 17–42 s.

## v2.6 Limitations and next steps

- **Crack recall on the v1 set dropped by one** (5/9 vs 6/9 real). This is the deliberate price of the tortuosity
  rule (D-dmg-13): branched or crazed cracks with span/length below 0.70 are rejected.
  - Next: decide tortuosity on the main path, the longest geodesic, not on the whole skeleton with its branches.
- **Crack length interval coverage is 78%,** not 90%. The two misses are a −19% faint end (s1-T7) and a −32% tape
  line (staged_s0-T3).
  - The 10% completeness term was set from the same injections (disclosed). It needs fresh seeds before it is trusted.
- **Paper-stain recall is 3/6 in simulation** (simulated blotches are pale: 22–38% tint over 8–17 cm).
  - **Advice for tomorrow's staging:** paint the blotch clearly brown/yellow. A darker, more saturated blotch seeds on
    the wall model.
  - The local-seed option is there (`--param stain_local_sigma_m=0.05`) if the real decal is missed, at the cost of
    false stains seen on the sample.
- **Photo tier:**
  - The `single_room` simulation placed only 1 photo per room, so nothing could be analysed. That is a front-end
    output; under D-017 there are 6–8 photos per room.
  - Two-view evidence is weak, and confidences are lower by design.
- **Video `with_ceiling` fallback planes (I-20)**; with a plan this does not arise.
- **The whole crack pipeline is tuned and evaluated on one apartment** (the same caveat as D-dmg-7), now with more
  rules. Fresh captures (tomorrow's home) are the real test.

# Damage v3: damage_precision (2026-10-04)

Everything below was added in v3; sections 1-8 (v1) and "Damage v2" above are kept unchanged as history. Every number
comes from runs executed for this part. Evidence: `outputs/damage_precision/` (one-command runs `before/` and
`after/`, diagnosis `diag/` with the script `diag_stain.py`, run table script `summarise_runs.py`, held-out runs
`heldout_drift_off/`, a frozen copy of the v2 sources in `src_before/`) and `outputs/damage/v3/` (the v2 injection
and clean-capture benchmark re-run with the v3 code: `run_all.sh`, `logs/`, `injection_summary_v3.json`).
Tests: `tests/test_damage_precision.py` (6 tests; the full suite passes, 42 tests).

## v3.1 Problem

The LiDAR-sample benchmark (`docs/modules/benchmark_lidar_sample.md` section 6) ran the one command
(`scripts/run_capture.py <capture> --tier lidar`) on the clean `single_scan_floor_only` capture and got:

- a **confirmed** water stain, confidence 0.733, on wall R6-W10 (12 x 6 cm, 0.92-0.98 m above the floor);
- a concealed-damage flag on it, `BELOW_WINDOW_SEAL_LEAK` ("stain directly below a window: possible window seal leak");
- **2 scope items**: stain-block primer + 2 coats on the whole wall (2.04 m²) and "inspect, water-test and reseal
  window perimeter".

The sample apartment has no damage, so this is confident garbage with a money-relevant line on the invoice. It also
contradicts D-026, which said that no false *stain* had appeared at any tier.

Second, smaller problem: the damage log line counted flags and scope items **before** the D-023/D-026 two-tier
filtering. On `single_room` it printed `[damage] 2 regions, 0 flags, 4 scope items`, while the plan contains
0 scope items (both detections are review-tier).

## v3.2 Root cause, with evidence

**Reproduced** with the one command (`outputs/damage_precision/before/floor_only/`, current code before v3):
`D1 water_stain R6-W10 confidence 0.732, confirmed`, the `BELOW_WINDOW_SEAL_LEAK` flag and 2 scope items. The
confidence is 0.732, not 0.733; the 0.001 difference is the drift solve's run-to-run jitter
(benchmark_lidar_sample.md section 4.1).

**What it is.** `outputs/damage_precision/diag_stain.py` re-builds the R6-W10 orthophoto and writes three pictures:

- `diag/views_R6-W10.jpg`: crops of the 7 raw camera frames that fed the orthophoto, centred on the candidate. In
  every one of them the red circle sits on the **brass/gold lid of a black jar** (a soap or toothbrush holder),
  standing on a bathroom shelf next to a green-labelled pump bottle and a tap.
- `diag/ortho_R6-W10.png`: the orthophoto. The shelf objects (grey figure, green bottle, black jar) are painted onto
  the wall texture, and the "stain" outline is the lid at the top of the jar.
- `diag/depth_R6-W10.png`: LiDAR raw points coloured by their offset from the wall plane (−10…+10 cm). The
  candidate sits in the region that stands proud of the wall.

It is neither a shadow nor the window frame. The existing chroma rule already rejects shadows (16 shadow/grime
candidates on this capture: "darkening without a yellow/brown shift").

LiDAR raw points inside the candidate and in rings around it (`diag/relief_R6-W10.json`; offset from the plan's wall
plane, + = in front of the wall):

| Where | Points | Median | 10th pct | 90th pct | Share > 1.5 cm proud |
|---|---|---|---|---|---|
| inside the candidate | 1,008 | **+3.1 cm** | +0.2 | +8.9 | 0.68 |
| ring 0–2 cm | 1,353 | +3.2 cm | +0.3 | +9.3 | 0.68 |
| ring 2–5 cm | 3,786 | +6.6 cm | +0.2 | +49.8 | 0.76 |
| ring 5–10 cm | 14,841 | +28.2 cm | +2.8 | +54.0 | 0.93 |

**Why each guard let it through.**

1. **The orthophoto treats anything within about 10 cm of the wall as wall.** A view sample is rejected as occluded
   only if the scene z-buffer is in front of the wall by more than 5 cm + 3% of the range (`build_ortho`,
   `occl_tol`, `occl_rel`): about 9.5 cm at 1.5 m. A jar standing 3–9 cm in front of the wall on a shelf passes, so
   its texture is fused into the wall orthophoto. The tolerance exists because the z-buffer comes from 2 cm voxel
   points and the poses carry error.
2. **The stain rules are colour and shape only.** The lid is yellower than the wall (+7.4 b\*) and slightly darker
   (−1.3 L\*, just over the 1.0 minimum). It is a blob, not a strip or a rectangle. It is seen consistently by 7 views
   (support 0.86), because it is a real object. Every rule passes, and the confidence (b\* contrast × view agreement)
   is high.
3. **D-025's geometric check was applied to cracks only.** The relative form, line versus beside it, would not have
   caught this case either: the object is larger than its yellow part, so the 0–2 cm ring is the jar too
   (+3.2 cm against +3.1 cm inside).
4. **The clean validation never saw this surface.** `outputs/damage/v2/run_all.sh` runs the clean LiDAR captures on
   the pre-integration scenes (`outputs/<capture>/`) and plans (`outputs/damage/_plans/`). It does not run the
   one-command path (drift ON, plan_beta v2), which creates wall R6-W10 across the bathroom shelf. So the D-026
   evidence ("no false stain at any tier") was true for the validation inputs, but not for what the one command
   produces.
5. **The concealed flag is a consequence.** The plan reports a "window" O9 (0.69 × 0.70 m, sill at 1.05 m, found as
   a see-through region in the wall profile) 7 cm above the jar. The view crops suggest it is the mirror or glazed
   panel above the shelf; LiDAR sees "through" mirrors. That is a plan-module question and is not changed here. Once
   the false stain is gone, the flag and both scope items go with it.

## v3.3 Options

| Option | For | Against | Verdict |
|---|---|---|---|
| (a) Raise the stain confirm threshold (0.35 → 0.75) | one constant | true stains score 0.37–0.98, so about half of them would drop to review (v2d sweep: stain recall at τ 0.7 = 0.39); and a slightly yellower lid would still pass. It treats the symptom | rejected |
| (b) Tighten the orthophoto occlusion tolerance (5 cm + 3%) | removes object texture at its source, for every detector | the tolerance absorbs pose error and the 2 cm voxel z-buffer; tightening it drops real wall pixels everywhere, and it changes every measurement and the whole benchmark. Not measurable in the time box | not built; noted as next step |
| (c) Ring-relative relief (D-025 form: inside vs ring) | consistent with D-025 | measured on this case: inside +3.1 cm, ring 0–2 cm +3.2 cm, so relative relief ≈ 0. The object is bigger than its yellow part, so it **fails on the motivating case** | rejected |
| (d) Shadow test | asked for in the brief | it is not a shadow (raw frames); shadows are already rejected by the chroma rule | not needed |
| (e) **Absolute region relief against the local wall: a stain must lie ON the surface** | physical, not tuned to the instance; tier-aware (LiDAR 1.5 cm, learned depth 4 cm, as D-025); abstains when there are too few points; two-sided, so it also covers see-through regions (glass, mirror) | needs scene points (all three tiers have `raw_points`); an object flush with the wall (< 1.5 cm, e.g. a sticker) is not caught | **chosen** |
| (f) Semantic "object vs wall" classifier | would name the jar | new model and environment (I-002), no time to validate | not built |

## v3.4 The fix (`floorplan/damage/__init__.py`)

`_region_relief(o, region, raw_points)` runs for every stain that passed the colour and shape rules:

1. Take the scene's raw points (LiDAR returns; at the video and photo tiers, learned-depth points) in a window of
   the candidate's bbox ± 30 cm, from 5 cm behind to 25 cm in front of the wall plane. The 5 cm limit keeps the far
   face of the wall (walls are 7 cm or more thick) out.
2. **Inside:** points whose (s, t) falls on a pixel of the candidate.
3. **Local wall level:** the median offset of points more than 2 cm outside the candidate and within ±3 cm of the
   plane. This makes the test immune to a wall that bows away from the fitted plane (unit test: a 1 cm offset wall
   gives |relief| < 0.3 cm). With fewer than 50 wall points, the fitted plane (0) is used, and the note says so.
4. **relief = median(inside) − wall level.** Reject when it is above the tier threshold ("protrudes … object in front
   of the wall") or below minus the threshold ("lies … behind the surface: see-through"). The thresholds are D-025's:
   `RELIEF_MAX_M` = LiDAR 1.5 cm, video/photo 4 cm.
5. **Abstain** (keep the stain, record the reason) when fewer than 30 points fall inside. On the sparse photo tier
   that is the usual case.

Every accepted stain now carries `relief = {relief_m, n_in, n_wall, frac_proud_1p5cm, wall_level_m, note}` in
damage.json and in `plan.damage[].detail`. Rejected ones appear in `rejected_detail` with the rule, and in
`rejected_candidates` as `water_stain: protrudes from the surface` or `water_stain: behind the surface`.

Why absolute rather than D-025's relative form: a crack is thin, so "beside the line" is the wall itself, and the
relative test also cancels a wall that is locally off the plan plane. A stain is an area, and an object that causes
a false stain is usually larger than its discoloured part, so the ring would be on the object too. The local wall
level, taken outside a 2 cm gap and only from near-plane points, keeps the immunity to plane offsets without that
failure.

**Logging (c).** `analyse_damage` now writes `reported = {confirmed, confirmed_by_class, review, concealed_flags,
scope_items}` to damage.json. These are the counts after the D-023/D-026 two-tier filtering, the same function
(`to_plan_items`) that fills the plan. It logs two lines, so the raw counts can no longer be mistaken for what is
reported:

```
[damage] detector: 2 candidate regions, 0 raw flags, 4 raw scope items (before two-tier filtering); analysed ...
[damage] reported: 0 confirmed {}, 2 review, 0 concealed flags, 0 scope items
```

`run_damage_on_plan` stores `confirmed` and `review` in `plan.meta.damage`, and `flags` and `scope_items` there are
now post-tier. `scripts/run_damage.py` marks each detection `[confirmed]` or `[review]` and prints review-only scope
lines as "(review only, not quoted)". It also gains `--no-stain-relief` (ablation) and can re-run damage on a
one-command run's own `plan.json`: a small adapter, `_lohi`, reads the published `{value, ci95}` form, which used to
raise `KeyError: 'hi'`.

**Learned-depth tiers (video, photo): review, not reject.** At those tiers the relief measurement is too noisy to
decide on its own (v3.5, null distribution). A stain whose relief is beyond ±4 cm there is **kept but forced to
`review`** (`detail.relief_review` says why): it is listed for a human, and it is never quoted and never raises a
concealed flag (`to_plan_items`). At the LiDAR tier the measurement is decisive, so the stain is rejected. This
follows the D-023 rule: a decision we cannot stand behind goes to a human, not onto the invoice.

## v3.5 Results (before → after)

**Integrated one-command runs, LiDAR tier, clean captures.** Before = the code before v3 (`before/`, sources in
`src_before/`). After = v3 (`after/`, `run_after.sh`). The plan columns count `plan.json` after two-tier filtering.

| Capture | Confirmed | Review | Concealed flags | Scope items | Damage log line(s) |
|---|---|---|---|---|---|
| floor_only, before | **1** (water stain 0.732, R6-W10) | 0 | **1** (window seal leak) | **2** | `1 regions, 1 flags, 2 scope items` |
| floor_only, after | **0** | 0 | **0** | **0** | `detector: 0 candidate regions …` / `reported: 0 confirmed, 0 review, 0 concealed flags, 0 scope items` |
| single_room, before | 0 | 2 (cracks 0.608, 0.148) | 0 | 0 | `2 regions, 0 flags, 4 scope items` (**misleading**: the plan has 0) |
| single_room, after | 0 | 2 (same) | 0 | 0 | `detector: 2 candidate regions, 0 raw flags, 4 raw scope items (before two-tier filtering)` / `reported: 0 confirmed, 2 review, 0 concealed flags, 0 scope items` |
| with_ceiling, before | 0 | 0 | 0 | 0 | `0 regions, 0 flags, 0 scope items` |
| with_ceiling, after | 0 | 0 | 0 | 0 | `reported: 0 confirmed, 0 review, 0 concealed flags, 0 scope items` |

**Target met on all three clean captures: zero confirmed false detections and zero scope items.** On floor_only
the jar lid is now in `rejected_detail`: `protrudes 2.4 cm from the surface: object in front of the wall (on a
shelf/sill), not a stain on it` (n_in 997 points, local wall level +0.74 cm, 58% of points more than 1.5 cm proud). It
is drawn as a thin rejected outline in `after/floor_only/damage/surfaces/R6-W10.jpg`, and
`diag/after_R6-W10_crop.png` is a zoomed copy. The same re-run on the identical before-scene and plan
(`same_scene/floor_only`, via `run_damage.py`) gives relief +2.38 cm and the same rejection, so the fix does not
depend on run-to-run jitter. Analysed area is unchanged (43.0 / 18.7 / 50.8 m²). The single_room 0.608 crack stays
review only because of D-026; it is a wardrobe-front line and not part of this fix.

**Injection benchmark** (`outputs/damage/v3/run_all.sh` = v2's runner with `O=outputs/damage/v3`; 6 injected sets
plus 9 clean tier runs; evaluation mode τ = 0, reported at τ = 0.10). The ablation arm is the same runner with
`EXTRA=--no-stain-relief` (`outputs/damage/v3_ablation_no_stain_relief/`). v2d is the previous published run.

| | v2d (before, earlier run) | v3 ablation, check off (this run) | **v3 (this run)** |
|---|---|---|---|
| Stain recall | 11/18 | 11/18 | **11/18** (floor 11/18) |
| Crack recall | 9/15 | 9/15 | **9/15** (floor 9/15) |
| Stain area: truth inside the 95% interval | 1.00 (11/11) | 1.00 | **1.00** |
| Stain width/height interval coverage | 0.82 | 0.82 | **0.82** |
| Crack length interval coverage | 0.78 (7/9) | 0.78 | **0.78** |
| Crack width/height interval coverage | 0.83 | 0.83 | **0.83** |
| Median abs. error, stain area / crack length | 2.0% / 5.0% | 2.0% / 5.0% | **2.0% / 5.0%** |
| False positives on injected sets | 0 | 0 | **0** |
| Clean false positives, τ = 0.10 (9 tier runs) | 0 | 0 | **0** |
| Clean, τ = 0 | 1 (video crack 0.067) | 1 | 1 (same; review, 0 scope) |

The relief of the 11 detected injected stains (real single_room walls with paint rendered into the frames) ranges
from **−0.93 to +0.75 cm**: all inside ±1.5 cm, so none were lost. On the v2 runner's clean inputs the check never
fires. Those inputs never contained the false stain (v3.2, point 4), which is why the integrated runs above are the
real test.

**Clean false positives at all tiers** (v3, all at τ = 0.10): LiDAR 0/0/0, video 0/0/0, photo 0/0/0 (single_room /
floor_only / with_ceiling). Reported after tiering: 0 confirmed and 0 scope items in all 9. One video crack at 0.067
is listed as review in evaluation mode only (below τ). The video and photo runs were repeated with the final code
(`rerun_learned_depth_clean.sh`) after the learned-depth review rule was added. The LiDAR path is unchanged by that
rule.

**Null distribution of the measurement** (`relief_null.py` → `relief_null.json`, `relief_null_video_band6cm.json`):
random 10 × 10 cm patches on the clean sample's walls (about 2 per m²). "All patches" includes patches that land on
furniture or objects near the wall, so its exceedance rate is an upper bound on how often a real stain at a random
spot would be rejected. "Flat neighbourhood" means at least 80% of the 30 cm window lies within ±3 cm of the plane
(±6 cm for video): that is the measurement's noise floor.

| Tier / capture | Patches measured (abstained) | All: median \|relief\| | All: share over tier threshold | Flat: n | Flat: median / 95th pct \|relief\| | Flat: share over threshold |
|---|---|---|---|---|---|---|
| LiDAR single_room (thr 1.5 cm) | 72 (32) | 0.41 cm | 0.22 | 32 | 0.23 / 0.87 cm | 0.03 |
| LiDAR floor_only | 198 (64) | 0.26 cm | 0.24 | 100 | 0.11 / 0.87 cm | 0.02 |
| video single_room (thr 4 cm) | 34 (3) | 1.58 cm | 0.35 | 7 | 0.31 / 1.24 cm | 0.00 |
| video floor_only | 61 (14) | 1.73 cm | 0.16 | 18 | 1.05 / 3.05 cm | 0.00 |
| photo floor_only (thr 4 cm) | 51 (24) | 0.39 cm | 0.18 | 22 | 0.18 / 0.75 cm | 0.00 |
| photo with_ceiling | 74 (51) | 0.78 cm | 0.22 | 22 | 0.41 / 1.64 cm | 0.00 |

How to read it:

- **LiDAR.** On flat wall the 95th percentile is 0.87 cm, below the 1.5 cm threshold. The jar measured +2.4 cm and
  the injected stains at most 0.93 cm. 75–79% of the exceedances on random patches are *positive* (something stands in
  front of the wall), which is what the test is for.
- **Video.** Learned depth gives 1.4–1.7 cm median |relief| even on random patches, flat walls reach 3 cm at the
  95th percentile, and the random-patch exceedance changes 0.10 → 0.35 between two random draws (small n). That is
  not decisive, hence review instead of reject at that tier.
- **Photo.** It abstains on 30–40% of patches (sparse points).

**Held-out check of the measurement.** In the benchmark's `with_ceiling --no-drift` run there was a second false
stain, R7-W2 at 0.412, "object on a shelf above a glass screen". It played no part in designing the check.

- Our damage re-run on that run's scene and plan (`heldout_drift_off/with_ceiling`) does not reproduce the candidate:
  it now fails view support (0.60 of 5), so the relief stage is never reached. 0 confirmed, 0 scope.
- `heldout_relief_bbox.py` measures the relief over that candidate's bounding box: **−4.4 cm** (104 points, wall level
  +0.04 cm). The returns lie *behind* the wall plane, which is what glass gives. The two-sided test would reject it.
  The one-sided first version would not, and that is why the check was made two-sided. The first, one-sided pass is
  kept in `outputs/damage/v3a_one_sided/` (identical benchmark numbers) and `after_one_sided/floor_only`.
- `heldout_drift_off/floor_only` gives 0 confirmed, 1 review (crack 0.557, below D-026's 0.65) and 0 scope.

## v3.6 D-026 corrected (wording to carry into DECISIONS.md)

> D-026's evidence line "No false *stain* has appeared at any tier" is **wrong for the one-command path**. On
> 2026-10-04 the LiDAR-sample benchmark found a confirmed false stain (0.733) on clean floor_only, with a
> window-seal-leak flag and 2 scope items. It was the gold lid of a black jar on a bathroom shelf, 2–4 cm in front of
> the wall and inside the orthophoto's occlusion tolerance. The statement held only for the v2 validation inputs
> (pre-integration scenes and plans). **Fixed in damage v3:** a stain must lie on the surface. Region relief against
> the local wall level is rejected beyond ±1.5 cm at LiDAR and sent to review beyond ±4 cm at learned-depth tiers.
> After the fix: 0 confirmed detections and 0 scope items on all three clean captures; injection recall unchanged
> (11/18 stains, 9/15 cracks). The per-class thresholds (stain 0.35, crack 0.65) are unchanged. Stains still confirm
> at 0.35, because the false stain is now removed by evidence about *what it is*, not by a higher bar for everything.

## v3.7 Limitations and next steps

- **Objects flush with the wall are not caught.** Anything under 1.5 cm proud (a sticker, a thin label, a poster) is
  still judged by colour alone. Paper decals are the intended staging, and they *should* pass.
- **The orthophoto still paints near-wall objects onto the wall** (occlusion tolerance 5 cm + 3%). The relief check
  removes stains that come from them. A tier-aware occlusion tolerance (tighter at LiDAR) would remove such texture at
  its source, for cracks too. That changes every measurement, so it needs a full re-benchmark. Not done.
- **Video and photo relief is review-only,** and the 4 cm threshold there rests on 7 + 18 flat-wall patches. No
  stain candidate appeared at those tiers on the clean sample, so the review rule is unit-tested but not exercised on
  real data.
- **A real stain partly behind an object within 10 cm** (for example behind a bottle on a shelf) is rejected at LiDAR.
  We accept this: the camera cannot see the wall there either.
- **One motivating instance.** The design was fixed on R6-W10. The thresholds are D-025's (not re-tuned), the
  injected stains are the recall check, and R7-W2 (bbox measurement only) is the only held-out false stain. The walk-in
  capture is the real test.
- **Plan-side:** O9 (a likely mirror or glazed panel read as a window) is not addressed here. Concealed-flag rules
  that trust opening types inherit plan errors.

## v3.8 Regenerate

```
# before (run on the v2 sources, now in outputs/damage_precision/src_before/):
python scripts/run_capture.py <DS>/single_scan_floor_only/1a8384c3f6 --tier lidar --out outputs/damage_precision/before/floor_only
#   (same for single_room/c00a170fe1 -> before/single_room, single_scan_with_ceiling/c7d28f72c6 -> before/with_ceiling)
env -u PYTHONPATH python outputs/damage_precision/diag_stain.py outputs/damage_precision/before/floor_only \
    <DS>/single_scan_floor_only/1a8384c3f6 R6-W10 outputs/damage_precision/diag
# after:
outputs/damage_precision/run_after.sh
python outputs/damage_precision/summarise_runs.py outputs/damage_precision/{before,after}/{floor_only,single_room,with_ceiling}
outputs/damage/v3/run_all.sh && outputs/damage/v3/rerun_learned_depth_clean.sh
O=outputs/damage/v3_ablation_no_stain_relief EXTRA=--no-stain-relief outputs/damage/v3/run_all.sh
python scripts/inject_damage.py summary --inject outputs/damage/_inject/{single_room_s0,single_room_s1,single_room_s2,staged_s0,staged_s1,staged_s2} \
    --eval outputs/damage/v3/inject_eval/{single_room_s0,single_room_s1,single_room_s2,staged_s0,staged_s1,staged_s2} --clean outputs/damage/v3/clean/* --out outputs/damage/v3/injection_summary_v3.json
env -u PYTHONPATH python outputs/damage_precision/relief_null.py
FLAT_BAND=0.06 TIERS=video OUT=outputs/damage_precision/relief_null_video_band6cm.json env -u PYTHONPATH python outputs/damage_precision/relief_null.py
env -u PYTHONPATH python outputs/damage_precision/heldout_relief_bbox.py
```
