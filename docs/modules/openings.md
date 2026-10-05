# Doors and windows from the segmenter (video and photo tiers)

Code: `floorplan/openings/semantic.py`. Step in `scripts/run_capture.py` after the plan step, on by default for the
video and photo tiers (`--no-semantic-openings` turns it off). For runs made before it:
`scripts/add_openings.py <run_dir> --out DIR`. Tests: `tests/test_semantic_openings.py`.

## Why

The plan step finds an opening only where the geometry has a gap: a free-space neck between two rooms, or rays that
crossed a wall plane. Phone captures without LiDAR rarely leave such a gap. The own-house plans showed almost no
doors and no windows. The segmenter the photo tier already runs (SegFormer-B5, ADE20K) labels "door", "double door",
"screen door" and "windowpane" pixels. Each photo and each video keyframe has metric depth and a pose in the plan
frame, so those pixels can be put on the plan's walls and measured there.

## How

Per view (a photo, or a video keyframe):

1. The view is levelled on the floor it sees. A plane is fitted to its floor pixels within 3.5 m. The view is turned
   about its camera centre until that plane is level (up to 50°). Its floor height is that plane's.
2. Door and window pixels form blobs. Blobs under 0.2% of the image are dropped.
3. A blob goes on the plan wall that holds most of its pixels: on the wall line (within 0.35 m), or behind it. For a
   door, behind means at most 1 m behind (a leaf swung the other way). The camera must be on the wall's room side.
4. The wall's surface in this view is a line fitted to the view's own "wall" pixels within 1 m of the blob.
5. Each blob pixel gives a position along that line, where its ray crosses it. Pixels on the surface count. Pixels
   behind it count too: they were seen through the opening. Pixels in front of it do not (an open leaf, a cupboard,
   a curtain). Pixel centres stop half a pixel short of the blob's edge, so half a pixel is added at each end.
6. A blob that is mostly in front of the wall is an open leaf. Its length in plan is the door's width. The hinge is
   the end nearest the wall. The doorway is on the side where the view sees through the wall.
7. The interval = the longest run of 2 cm bins with support. An end that touches the image's left or right border
   is "cut": it does not measure the opening's end. Blobs on one wall less than 0.3 m apart are one opening (two
   leaves, a grille door, a window with a mullion).

Across views: intervals on parallel walls within 0.45 m that overlap by 30% are one opening. A door seen from both
rooms lands on both rooms' walls. Width = median width of the views that saw both ends. A pose error moves both ends
of a view together, so pooling ends over views would put the pose errors into the width. With fewer than 2 such
views, each end is the median of the views that saw it.

Rejected:

- Doors that do not reach within 0.35 m of the floor, or whose top is under 1.45 m (kitchen cupboards). Wardrobe
  fronts stand 0.4-0.6 m in front of the wall, so their pixels are never on its surface. 1.45 m, not 1.8 m: take1's
  entrance door tops measured 1.5-1.8 m on levelled keyframes.
- A view's blob less than 0.3 m (window) or 0.6 m (door) tall on its wall, not cut at the top or bottom: it was seen
  at a grazing angle through a wall it is not on.
- Photo tier: openings seen in one view, unless both ends are seen and the blob covers 1% of the image, or one end
  is seen and the blob covers 5% (a photo taken close to a door: the door is there, its width is a lower bound).
  Video tier: openings seen in fewer than 3 keyframes. Keyframes come about 1 s apart, so a real opening is in
  several; on take1 the one-view detections were pieces of windows already found, put on other walls.
- Openings with an end that no view saw, unless seen in 4 views (or the 5% photo case above).
- Widths outside 0.5-2.6 m (doors) or 0.3-4.0 m (windows). A lower bound only has to reach 0.3 m.

Into the plan: where the geometry already has an opening on the same wall, the geometric one stays and keeps its
width. It becomes a door when the segmenter saw a door there. A geometric opening without a width takes the
segmenter's. The rest are added with `"source": "segmentation"` (schema: optional `source` on each opening). A door
whose wall has a parallel wall of another room within 0.45 m connects both rooms. Width interval: the spread of the
views' widths (floor 3 cm per end), then the tier's scale term like every other width. An end never seen gives a
lower bound: status "inferred", interval at least ±0.45 m, confidence 0.45 or less (drawn dashed). `plan.meta.semantic_openings` lists every opening and every
rejection with its reason.

## Measured

Own house, take1 and the bedroom photos (`outputs/presentable/openings/`, `run_measure.sh`). Each detection is named
by the keyframes that saw it. Tape widths from `ground_truth.csv`. Error in cm. The video runs are the saved fix-loop
runs and the 5fe6ae6 reruns in `outputs/presentable/`.

| Opening (tape, m) | Photo lit | Photo dim | after r1 | before r1 | before r2 | pnp r1 | default r1 | default r2 |
|---|---|---|---|---|---|---|---|---|
| Bedroom window 1.473 | 1.530 (+5.7) | 1.532 (+5.9) | 1.524 (+5.1) | 1.524 (+5.1) | 0.398 (-108) | 1.523 (+5.0) | 1.523 (+5.0) | no |
| Hall window 1.753 | - | - | 1.583 (-17.0) | no | 1.593 (-16.0) | 1.072 (-68.1) | no | 1.588 (-16.5) |
| Kitchen window 1.155 | - | - | 0.529 (-62.6) | 0.837 (-31.8) | 0.848 (-30.7) | 1.055 (-10.0) | no | no |
| Bathroom door 0.700 | - | - | 0.692 (-0.8) | 0.689 (-1.1) | no | 0.696 (-0.4) | no | no |
| Bedroom doors 0.787, 0.737 | no | no | no | no | no | no | no | no |

"-": not in the photos. "no": not found.

- Bedroom window: +5.0 to +5.9 cm on 6 runs of both tiers (7 with pnp r2: 1.523 m). before r2 puts it on a wall it
  is not on.
- Geometry wins where both exist, also when it is worse: on pnp r2 the plan keeps its geometric kitchen window,
  0.677 m (-47.8 cm), where the segmenter measured 1.060 m (-9.5 cm) from 9 keyframes.
- Bathroom door: within 1.1 cm on the 3 runs that found it.
- Hall window: 16-17 cm short on 3 of the 4 runs that found it.
- Kitchen window: a wall cupboard hides part of it (keyframes 1446-1513), so the glass seen is narrower.
- Bedroom doors: the walk-through sees them close up, cut by the frame. Views give 0.1-0.5 m pieces, too narrow
  for a door. Where the geometry has a passage there, the pieces make it a door (after r1: O1).
- Entrance door (not in the tape GT): 0.659 and 0.875 m (after r1, two clusters), 0.730 m (pnp r1), 0.806 m
  (default r2).
- Extra windows: a run whose poses disagree puts one window on 2 walls. after r1 has 5 windows for 3: the bedroom
  window seen from the passage lands on two more walls.
- Photo tier doors: the spin photos see them cut by the frame. One door per set is kept as a lower bound: lit
  0.542 m, dim 0.519 m (the scorer pairs both with D2, 0.737 m; both intervals cover it). Drawn dashed.
- Scorer (`eval_own_capture.py`, then Hungarian on width per room, ±2 cm gate): the bathroom door is in the passage,
  which has no GT polygon, so it was not scored. No own-house opening passes (`eval_own/`). Since D-085 openings are
  paired by the wall `gt_polygons.json` names for them; a door of a room without an outline (the bathroom door)
  pairs by width within 10 cm with a prediction that is at no known door's place.

Simulator, k65 photo set (`sim_gt.json`, exact): kitchen window 1.283 m for 1.200 (+8.3 cm), bathroom window 0.771 m
for 0.800 (-2.9 cm), balcony window 0.936 m for 3.473 (curtains cover it). Doors 1.062 and 0.827 m. The scorer then
paired them with 0.710 and 0.645 by width only. Paired by position (D-085), the 0.825 m door is the entrance
(0.830, -0.5 cm, 7 cm from its centre) and the 1.062 m one is a phantom. Two extra windows in the living room
(0.951, 1.404 m): the glass of the 2.1 m kitchen and 2.3 m balcony openings.

Ray-cast synthetic views (exact geometry): a 0.80 m door and a 1.20 m window come back within 1.3 mm.

Recruiters' sample videos (video tier only; the saved fix-loop runs `outputs/fixloop/after/sample_*`), against the
LiDAR plan of the same capture (`compare_lidar.json`; a reference, not ground truth):

- single_room: one segmenter door, 0.970 m from 12 keyframes. It is the bathroom door (a toilet behind it in
  keyframes 604-850). LiDAR has it at 0.709 m, 0.6 m away on the next wall. The video plan's geometric door O3
  (0.649 m) is the same door, so the plan shows it twice. The leaf stands ajar beyond the wall; seen at an angle,
  its pixels cross the wall line past the jamb, so the width comes out 26 cm wide. The geometric openings of that
  plan: O1 0.955 m for LiDAR 0.921, O3 0.649 m for 0.709.
- The 5fe6ae6 rerun of single_room (`outputs/presentable/sample_single_room`) gives the same door: 0.972 m.
- floor_only: that run's plan has no rooms, so there is nothing to put openings on.
- with_ceiling: 26 keyframes in trusted segments, one room; no blob lands on a wall.

## Door priors (D-085)

Code: `floorplan/openings/priors.py`. Step in `scripts/run_capture.py` and `scripts/replan_run.py` after the room
names (`--door-priors RULES`, default `d`; `none` turns it off); `scripts/add_openings.py <run> --priors-only --rules
abcd` adds it to a finished plan. Tests: `tests/test_door_priors.py`.

Why: the user (5 Oct): where the door itself is not identified, use door priors and, from the gap, assume a door.
The segmenter finds a door only when its leaf is in frame; a doorway photo or a walk-through stands in the doorway.

Rules. Each door they make or change has `"source": "prior"`, its evidence, a confidence under 0.5 (dashed in the
technical drawing, a door in the presentation drawing):

- a. A plan-step passage 0.55-1.15 m wide with wall on both sides is a door; width = the gap (measured). Wider gaps
  stay passages, narrower ones are not doors.
- b. Photo tier: each doorway-pair photo stands on a threshold. The wall of its own room within 0.6 m of the camera,
  the camera looking away from it into the room (60°), gets a door at the camera's projection. A camera more than
  0.2 m from that wall line keeps its own position (the plan lacks the jog or alcove the door is in; the drawing still
  puts it on the wall). The pair's two doors on the two faces of one partition are one door. It replaces the plan
  step's widthless pair door.
- c. Video tier: where the camera path goes from one plan room into the next (a stay of at least 0.5 m of path on
  each side, no pose jump over 0.3 m between frames), a door at the crossing; crossings within 0.6 m are one door.
- d. A segmenter door cut by the image frame in every view (its width a lower bound): the prior width, laid from
  the end that was seen, never narrower than what was seen.

Prior width: 0.80 m, 95% interval ±0.20 m (sigma 0.10); 0.70 m when a room's name says bathroom or toilet. A measured
door or passage on the same wall (or the other face of the partition) within max(0.5 m, half its width) wins: the
prior only adds its room link and a note.

Measured (`outputs/presentable/door_priors/`: `run_measure.sh` runs all four rules on the saved plans, `score_all.py`
scores before and after by position; `eval/summary.md`). Doors found / missed / phantom (of which duplicates):

| Plan | Before | After (rules a-d) | Door widths after, cm | What changed |
|---|---|---|---|---|
| k65 photo | 5 / 0 / 1 (0) of 5 | 5 / 0 / 1 (0) | -0.5, -131.9, -149.1, +15.5, -1.0 | b: the 4 pair doors get prior widths |
| k65 LiDAR | 4 / 1 / 1 (1) | 4 / 1 / 1 (1) | +0.1, -0.6, -2.0, -0.1 | nothing: no passage in 0.55-1.15 m |
| k65 video | 2 / 3 / 1 (0) | 2 / 3 / 2 (0) | -128.9, -18.0 | c: one phantom, one door on the balcony door's other side |
| own photo lit | 1 / 3 / 0 (0) of 4 | 1 / 3 / 0 (0) | +6.3 (was -19.7) | d: the bedroom door seen cut, 0.54 -> 0.80 m |
| own photo dim | 1 / 3 / 0 (0) | 1 / 3 / 0 (0) | +6.3 (was -22.4) | d: 0.51 -> 0.80 m |
| own video pnp r1 | 2 / 2 / 1 (0) | 2 / 2 / 1 (0) | -10.6, -0.4 | c: the one walk-through lands on the measured door |

- b on k65 photo: the bathroom door 0.70 for 0.71 m (-1.0 cm), the bedroom door 0.80 for 0.645 (+15.5); the kitchen
  and balcony "doors" are 2.12 and 2.29 m glazed openings, and the 0.80 m prior misses them by 1.3-1.5 m (outside its
  interval). A doorway photo looks along the room; it cannot tell a door from a wide opening.
- c on clean camera paths is precise: 7 walk-through doors (the k65 LiDAR path's four, each crossed twice, checked by
  calling `path_crossing_doors` on that plan and scene; own video pnp r1, default r2 and pnp r2 one each) all land on
  doors the plan step had measured. On after/before r1-r2 and default r1 the path makes no new crossing. The k65 video path is broken (2080 steps over 0.3 m between frames, 23.7 km of path), and
  its plan is two unjoined parts of 134 m² for 59 m²; there c adds a phantom.
- a never fires on these plans.
- Default: d only. No rule found a door the plans lacked, so a, b and c stay off (c adds a phantom on a broken path;
  b claims 0.80 m where the opening is 2.1-2.3 m). d changes no count and both of its widths get better.
- The doors still missed have no evidence at all: k65 LiDAR's entrance (the walk starts and ends in the living room),
  the own bedroom's second door in the photos (no pair photo, no gap, no door pixels) and in the video (the walk never
  goes through it). A prior needs a place.

## Open doorways seen through (photo tier)

Code: `floorplan/openings/seethrough.py`, run inside the openings step (`add_semantic_openings`, which `run_capture.py`
and `replan_run.py` call) for the photo tier. Switch: `SemParams.see_through_doors`, on (off for video);
`scripts/add_openings.py --see-through on|off`. Tests: `tests/test_see_through_doors.py` (ray-cast views: a 0.80 m
doorway seen straight and 32 deg off comes back within 2 cm; a window with a 0.9 m sill, to the outside or to a room,
and a 0.8 m deep alcove give nothing).

Why: the bedroom's door to the kitchen side (D1, 0.787 m) stands open, its leaf against the wardrobe, and no plan had
it. What the pipeline sees in the dim photo 224001 (camera 1.80 m in front of the plan's W1, 1.51 m up, looking 25 deg
down and 30 deg along the wall): image columns 0.65-0.98 cross W1 1.11-1.89 m from its start and land 0.5-5.5 m beyond
it, on the next room's floor and far wall (past 6 m the photo depth is empty). The segmenter calls those pixels floor
and wall. There are no door pixels: the leaf (columns 0.49-0.63) is labelled wall and stands 0.1-0.6 m in front of
the line at 1.12 m; the wardrobe's orange fronts are labelled door, 0.7 m in front of it. Jambs: wall points on the
line at 1.05-1.12 m on the hinge side (the next room's side wall recedes from 0.98 m) and the latch-side frame (wall,
railing) on the line from 1.85 m. The head is above the frame: the top row crosses W1 1.61 m up. 223938 sees the same
gap from 1.44 m (cut by the left border) to the wall at 1.89 m. The evidence is in the depth, not in the classes.

How: per photo and plan wall, each ray is crossed with the wall plane. A 2 cm bin is open when, of its rays crossing
0.3-1.8 m up and not hidden by something in front, at least 60% land 0.5 m or more beyond (or have no depth) and at
most 10% end on the wall. A span of open bins needs the floor seen beyond it, no wall under it (a sill), something
farther than 1.0 m behind it (else an alcove or a step, D-082) and wall over it where the photo sees 2.2-2.5 m up.
Jambs: the wall points beside each end; the jamb is their inner edge on the wall line, so an oblique photo's reveal
does not narrow the opening. An end without a jamb counts as cut only at the image border or behind furniture.
Across photos, width and centre come from the photos that saw both jambs. Without one, the widest part one photo saw
is a lower bound (0.4-0.95 m) and rule d gives it the prior width. A door or passage with a width on that wall wins,
and so does a window. A doorway-pair door without a width takes the measured one.

Measured (`outputs/own_house/diag/door_gap/run_measure.py`: each saved plan without its segmenter and prior openings,
then the openings step with the rule off and on, priors rule d; scored by position. Runs in
`outputs/own_house/diag/door_gap/` and `outputs/sim/<flat>/door_gap/`). Doors found / missed / phantom (duplicates):

| Photo plan | Before | After | What changed | Windows (same before and after) |
|---|---|---|---|---|
| own dim | 1 / 3 / 0 (0) of 4 | 2 / 2 / 0 (0) | D1 0.760 m for 0.787 (-2.7 cm), on W1 1.12-1.88 m from its start (tape 1.168-1.955); D2 0.80 m (+6.3) as before | 1 / 2 / 0, bedroom window +5.9 cm |
| own lit | 1 / 3 / 0 (0) of 4 | 2 / 2 / 0 (0) | D1 0.886 m (+9.9 cm), a lower bound (no photo sees both jambs; dashed), at 0.85-1.74 m on a W1 the plan has 2.26 m long (tape 2.545); D2 as before | 1 / 2 / 0, +5.7 cm |
| k65 | 5 / 0 / 1 (0) of 5 | 5 / 0 / 1 (0) | nothing: 2 doorways, both rejected (1.41 and 1.66 m seen past one jamb) | 3 / 1 / 2 (0) |
| k22 | 5 / 1 / 0 (0) of 6 | 5 / 1 / 0 (0) | the bedroom-bathroom door (a doorway-pair door, no width) gets 0.826 m for 0.799 (+2.7 cm); 3 doorways lie on walls where the segmenter has a window (phantom windows in the score): the window wins | 1 / 4 / 6 (0) |
| k38 | 3 / 2 / 7 (0) of 5 | 3 / 2 / 7 (0) | nothing: of 3 doorways, 2 land on openings that have widths and 1 on a window; those win | 4 / 0 / 4 (1) |

- Default on: D1 is found on dim on W1 within 3 cm, and no plan gains a phantom or loses a found opening.
- Letting the doorway win over the window adds a duplicate door on k22 (5 / 1 / 1 (1)), so the window wins.
- Lit: 223840, the only photo that sees the hinge jamb, gives 0.886 m, 10 cm over the tape. Taken with the latch jamb
  from 223846 the width was 1.09 m (+30 cm), which is why ends from two photos are not combined.
- k22's black, glossy bathroom tiles reflect the room, and the depth sees "through" them. Those spans end with no
  wall, image border or furniture beside them, and are dropped; one 0.98 m lower bound is over the 0.95 m limit.
- D1 on dim lies 5-8 cm short of the tape positions along W1 (the plan's W1 is 3.2 cm longer than the tape's).
- The thresholds were set on these five plans; there is no held-out set.

## Pose check found on the way

The levelling in step 1 measures each keyframe's pose against the floor in its own depth. On take1 the two differ by
15-45° (after r1) and 5-30° (before r2). The floor gives camera heights of 1.0-1.7 m and pitches of 6-37° down,
which a hand-held walk-through has. So the video poses are tilted, not the depth. The plan's walls are built with
those poses. That is one reason video plans of take1 differ so much between runs.

## Limits

- Widths are only as good as each frame's MoGe-2 metric depth. The hall window is 9-10% short on 3 runs.
- An opening is placed with the run's poses. On runs whose poses disagree it can land on the wrong wall, or on two.
- Door swing is not measured. The drawing uses the existing convention.
- The thresholds were set on take1, the bedroom photos and k65. There is no held-out set.
- A door ajar beyond its wall, seen at an angle, comes out too wide (single_room: +26 cm).
- One opening can appear twice: a geometric one and a segmenter one on a perpendicular wall are not merged.
- Open doorways seen through: a mirror or a glossy surface looks like an opening in the depth. The rule drops spans
  whose see-through stops with nothing beside it, and a mirror framed by wall could still pass.
- Door priors: a standard-door width is not a measurement. A doorway photo cannot tell a 0.8 m door from a 2.2 m
  opening, and a path crossing is only as good as the poses.
