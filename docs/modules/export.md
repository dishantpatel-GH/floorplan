# Module: export (JSON contract, rendered plan, DXF)

Files:

| File | What it is |
|---|---|
| `schema/plan.schema.json` | Our published output schema (JSON Schema draft 2020-12), version 1.0.0 |
| `floorplan/export/json_export.py` | `Plan` → JSON (and back), two-layer validation |
| `floorplan/export/render.py` | `Plan` → `plan.svg` + `plan.png` (homeowner-style floor plan); also the shared drawing geometry |
| `floorplan/export/dxf.py` | `Plan` → `plan.dxf` (CAD, metres, real DIMENSION entities) + an independent read-back preview |
| `scripts/render_plan.py` | CLI: `plan.json` → validate → svg/png/dxf + `render_report.json` |
| `tests/fixtures/make_synthetic_plan.py` | Generator for the synthetic apartment fixture and the stress plan |
| `tests/fixtures/synthetic_plan.json` | The fixture (6 rooms, 27 walls, 11 openings, damage, scope items) |
| `tests/test_export.py` | 14 tests: schema, round trip, validator catches mistakes, determinism, robustness, DXF |

How to run (from the repo root):

```bash
PY=../.venv/bin/python
$PY tests/fixtures/make_synthetic_plan.py                         # (re)generate the fixture
$PY scripts/render_plan.py tests/fixtures/synthetic_plan.json --out outputs/export/synthetic --dxf-preview
PYTHONPATH=. $PY -m pytest tests/test_export.py -q                 # 14 passed in ~13 s
```

---

## 1. Purpose and where it sits in the pipeline

```
capture → scene (fusion + alignment) → plan extraction (rooms, walls, openings, intervals)
        → damage / scope → Plan object ──► EXPORT ──► plan.json   (the contract: "JSON to the published schema")
                                                  ├─► plan.svg / plan.png  ("rendered plan", the product surface)
                                                  └─► plan.dxf   (CAD hand-off, like magicplan / poly.cam)
```

Export is the **last step** of every tier (LiDAR, video, photo). It does no measuring. It turns the in-memory `Plan`
(`floorplan/model.py`) into the three things a customer and a grader actually see:

1. **`plan.json`**: the machine-readable contract (Part 2: "JSON to the published schema", "a confidence interval
   on every measurement"). The evaluation harness (`floorplan/eval/match.py`) already reads this format.
2. **`plan.svg` / `plan.png`**: the stitched floor plan "a homeowner would recognise from poly.cam or magicplan".
3. **`plan.dxf`**: the CAD export contractors and estimators expect from those apps.

Why it matters for the score:
- The walk-in test (30%) is scored from what we output, so a correct number that is badly presented, or presented
  without its interval, is lost value.
- Calibration is scored at every tier, so the export must keep "measured", "inferred" and "not observed" visibly
  distinct and must never print a number we do not have.

## 2. How it works, step by step

### 2.1 JSON export (`json_export.py`)

1. **Header first.** Every file starts with `schema`, `schema_version`, `units`, `coordinate_frame`, `capture_id`
   and `tier`, so anyone who opens it knows what it is and how to read the coordinates.
   *Without it:* a consumer has to guess the units and the axes. A mirrored or rotated plan is the classic result.
2. **One uniform measurement object.** Every number becomes
   `{"value": 3.42, "ci95": [3.412, 3.428], "unit": "m", "method": "...", "status": "measured"}`.
   - `status` is one of `measured`, `inferred` or `not_observed`.
   - A not-observed value is `value: null, ci95: null`.
   - If a producer hands us a `Measurement` with `value=None` but status "measured", we publish it as
     `not_observed`, because a number-less "measured" value is a contradiction.

   *Without it:* intervals would be optional, and someone would eventually ship a bare number.
3. **Rounding to 0.1 mm** (4 decimals), with NaN/inf mapped to `null`.
   *Why:* 0.1 mm is 100× finer than the tightest gate (1 cm), so nothing is lost. Repeated runs are byte-identical,
   which Part 4 needs ("before and after runs, both regenerable"). Without it, `json.dumps` writes `NaN`, which is
   invalid JSON.
4. **Readable formatting.** The JSON is indented, but number pairs such as points and intervals stay on one line
   (`[1.2, 4.0]`), so a polygon reads as a list of points.
5. **Validation in two layers**:
   - **Schema** (`jsonschema` 4.26, already installed): types, required keys, enums, and the conditional rule
     "not_observed ⇒ value and ci95 are null; otherwise both are present".
   - **Semantic** checks that JSON Schema cannot express:
     - `lo ≤ value ≤ hi`;
     - every referenced id exists (wall→room, room→walls, opening→walls/rooms, adjacency→rooms/openings);
     - no duplicate ids.

   *Without it:* a dangling `wall_id` or an inverted interval reaches the customer silently.
6. **Inverse (`plan_from_json`).** JSON → `Plan`, so the renderer CLI and the tests can work from the file alone.
   The round trip is exact (tested).

### 2.2 Rendered plan (`render.py`)

1. **View transform: draw (u, −v).** The aligned frame is right-handed with +y up. Seen from above, with +x to the
   right, +z points **down** the page. Drawing v = z upward (what the diagnostic figures in
   `scripts/prepare_scene.py` do) mirrors the apartment.
   *Without it:* the homeowner's kitchen appears on the wrong side, which they notice instantly.
2. **Fixed drawing scale.** The scale (inches per metre) is chosen from the plan size: 14 in for the longest side,
   clamped to 0.35–1.4 in/m. Because the scale is known, a 7.5 pt font is a known number of metres, so we can test
   whether a label fits along a wall or inside a room *before* committing to it.
3. **Walls as mitred rings.** For each room:
   - push each edge of its interior polygon outward by that wall's thickness;
   - intersect consecutive offset lines to get clean corners at any angle (the 45° bay wall works);
   - take ring = outer polygon − room.

   All rings are unioned. Then every room interior is subtracted, so a ring never intrudes into a neighbour.
   - Thickness is the measured `Wall.thickness` when present. Otherwise it is a **nominal 0.10 m**, drawn
     **hatched**, and the legend says "thickness not observed (nominal 0.10 m)".
   - A polygon edge is matched to its `Wall` when the wall is parallel and its midpoint lies within 5 cm of the edge.
4. **Openings.** Each opening is resolved to its host wall. If the wall links are missing, the nearest wall is used.
   The opening's centre is projected onto the wall face, and a rectangle is cut through the whole partition (both
   rooms' rings). Then the symbol is drawn:
   - **door**: leaf plus a 90° swing arc;
   - **window**: thin double glazing lines plus jamb lines;
   - **passage**: dashed thresholds on both faces.

   Openings with detection confidence below 0.5, or without an observed width, are drawn orange and dashed.
   - **Swing side is not measured** by anything in the pipeline. The convention is "the door opens into the room
     with fewer doors" (into the bedroom, not the corridor), and the subtitle says so.
5. **Dimensions.** Every wall at least 25 cm long gets an aligned dimension line, with 45° architectural ticks,
   extension lines and the text `3.42 m ±0.8 cm`, where ± is the half-width of the 95% interval.
   - **Envelope walls** (the outward side is outside every room) get their dimension outside the building, beyond
     the wall's outer face.
   - **Partition walls** get their dimension inside their own room. Otherwise the two rooms' dimensions for the same
     partition would pile up on one side.
   - Inferred values are shown grey italic with "≈". Not-observed values show "n/o".
6. **Room labels.** Each label shows the name (bold), the area ± CI and the ceiling height ± CI, or "ceiling not
   observed".
   - The anchor is the **pole of inaccessibility**: the interior point farthest from every wall. For L-shaped rooms
     it is better than the centroid, which can fall outside the room.
   - If that spot is taken, grid points ordered by distance to the walls are tried next.
7. **Collision-aware label placement (`Labeler`).** Every label has a list of candidates, from the best to the most
   compact:
   - offsets from the wall;
   - three font sizes;
   - one-line versus two-line text;
   - full versus shortened wording.

   The first candidate that overlaps nothing, and (for room labels) lies inside the room, wins. Overlap is tested
   with the real rendered text extents from matplotlib, not estimates.
   - Placement priority: dimensions, then room labels, then opening widths.
   - Door swings are **soft** obstacles: text avoids them if possible, but a dimension may cover a door arc rather
     than move to the wrong side of its wall (see Issue 3).
   - Whatever could not be placed cleanly is counted in `render_report.json`: `dims_forced`,
     `room_labels_reduced` / `forced`, `opening_labels_dropped`. Crowding is therefore measured, not hidden.
8. **Frame.** Title (capture, tier, room count, footprint ± CI), a subtitle that explains the notation, a scale bar
   and a legend.
9. **One figure, two files.** The same matplotlib figure is saved as SVG (vector, text kept as text) and PNG
   (200 dpi). Output is deterministic: SVG ids use a fixed salt and no timestamps are written. A test checks the
   SVG is byte-identical across runs.

### 2.3 DXF (`dxf.py`)

1. The file is AutoCAD R2010 with `$INSUNITS = 6` (metres), `$MEASUREMENT = 1` (metric) and decimal units, so a
   distance measured in CAD equals the real distance.
2. It has five layers: **WALLS** (outlines + SOLID hatch where measured, ANSI31 hatch where nominal), **ROOMS**
   (interior polygons), **DIMENSIONS**, **OPENINGS** (leaf + arc, glazing lines, dashed passages) and **TEXT**
   (room labels, opening widths, title).
3. **Real `DIMENSION` entities.** They are created with `add_aligned_dim(...)` and then `dim.render()`. Without
   `render()`, many viewers show nothing, because the dimension's display block is never built.
   - The text is overridden with our measured value and interval.
   - A custom dimstyle `PLAN_M` uses ticks instead of arrows and puts the text above the line.
   - Endpoints are ordered so that the text always reads left-to-right or bottom-to-top.
4. The geometry is **the same** as in the rendered plan (`build_wall_geometry`, `to_view`), so the SVG and the DXF
   cannot disagree.
5. **Read-back check.** `preview_dxf` re-opens the saved file with ezdxf's own drawing add-on and rasterises it.
   This check shares no code with `render.py`.

## 3. Decisions

### E-1 Define our own schema, modelled on RoomPlan and magicplan

- **Context.** Part 2 asks for "JSON to the published schema", but no schema was provided (OPEN_QUESTIONS.md).
- **Options.**
  - (a) Dump the dataclasses with `asdict`.
  - (b) Copy Apple RoomPlan's `CapturedRoom` JSON verbatim.
  - (c) Write our own versioned JSON Schema, borrowing RoomPlan's and magicplan's structure and adding uncertainty.
- **Choice.** (c).
- **Why.**
  - (a) would make every internal rename a breaking change for consumers, and it has no place for rules such as
    "no number without an interval".
  - (b) has no intervals, no measurement status, no rooms-with-polygons on older iOS, and no damage or scope.
  - (c) keeps the familiar shape and adds what the case study scores.
  - The schema is versioned (`1.x`), so a later change to the company's real schema becomes a mapping layer, not a
    rewrite.
- **Mapping.** RoomPlan fields are from Apple's RoomPlan API (`CapturedRoom`, `CapturedRoom.Surface`). The magicplan
  structure is described at concept level from its public export features; field names are not copied, so verify
  them before claiming exact compatibility.

| Ours | Apple RoomPlan `CapturedRoom` | magicplan export | Note |
|---|---|---|---|
| `rooms[]` (id, label, polygon) | `sections[]` (label, centre; iOS 17), `floors[].polygonCorners` | floor → rooms (name, shape) | RoomPlan has no per-room polygon before iOS 17 |
| `walls[]` p0, p1 | `walls[]` Surface: `transform` (4x4) + `dimensions.x` (width) | room walls (corner points) | we store 2-D endpoints in plan metres |
| `walls[].thickness` | `dimensions.z` (depth) | wall thickness | RoomPlan often reports ~0; we measure it or publish null |
| `openings[]` kind door / window / passage | `doors[]` (`.door(isOpen:)`), `windows[]`, `openings[]` | doors / windows attached to a wall | same three kinds |
| `openings[].wall_ids` | `parentIdentifier` (iOS 17) | the wall item's host wall | |
| `openings[].confidence` (0–1) | `confidence` (.low / .medium / .high) | none | ours is numeric; the size interval is separate |
| every `Measurement` (value, ci95, status, method) | none: bare floats | bare values | **our addition**, required by Part 2 |
| `ceiling_height`, `floor_area`, `perimeter` | derived by the app, not in `CapturedRoom` | room properties | |
| `adjacency[]` | none | implicit (shared walls / doors) | explicit, so stitching can be scored |
| `damage[]`, `scope_items[]` | none | none (magicplan has estimate add-ons) | **our addition** (Part 2 contract) |
| `coordinate_frame` | ARKit world: right-handed, y up | plan 2-D | ours is the same frame, aligned to the walls |

### E-2 The measurement object: `ci95: [lo, hi]` plus `status`

- **Options.**
  - (a) `value ± sigma`.
  - (b) `[lo, hi]` 95% interval.
  - (c) Both.
- **Choice.** (b), plus `status` and `method`.
- **Why.**
  - The gates are stated as "within X" and "calibrated intervals". A 95% interval is directly checkable: the true
    value should fall inside about 95% of the time.
  - (b) also allows asymmetric intervals, for example the photo-tier scale error.
  - A sigma implies Gaussian noise, which is not true for every tier.
  - `status` separates "measured", "inferred" (from a prior or neighbouring evidence) and "not observed"
    (no number).
- **Evidence.** The schema's conditional rule rejects an invented number on a not-observed field (test
  `test_validator_catches_mistakes[...-schema]`, the case that sets a WC ceiling of 2.5 m on a not-observed ceiling).

### E-3 null vs "not observed"

- **Choice.**
  - A `null` measurement field means "not applicable or not produced by this tier", for example `sill_height` of a
    door.
  - A measurement object with `status: not_observed` means "applies, but we have no data", for example the ceiling
    of a room that was scanned with the phone pointing down.
- **Why.** A consumer must be able to tell "no ceiling" (impossible) from "ceiling not seen" (a capture gap) from
  "doors have no sill".

### E-4 True top-down view: draw (u, −v); DXF uses the same transform

- **Options.**
  - (a) Draw v up, the matplotlib default and what the diagnostic figures do.
  - (b) Draw v down.
- **Choice.** (b), in both the SVG/PNG and the DXF.
- **Why and evidence.**
  - `det(T_align) = +1.000000` for all three captures (`outputs/export/handedness_check.json`), so the aligned
    frame is right-handed like ARKit's.
  - Looking down from +y with +x to the right, +z = x × y points down the page.
  - So (a) shows the apartment's mirror image.
  - The JSON stores (u, v) unchanged and documents this in `coordinate_frame.view_from_above = {u: right, v: down}`.

### E-5 One rendering path: matplotlib → SVG and PNG

- **Options.**
  - (a) Write SVG by hand and convert it to PNG with cairosvg.
  - (b) Use matplotlib for both.
  - (c) Use a web canvas.
- **Choice.** (b).
- **Why.**
  - cairosvg is not installed, and PyPI is slow here (about 0.4 MB/s).
  - matplotlib is installed, gives exact text extents for collision checks, writes both formats from one figure
    (so they cannot disagree), and with `svg.fonttype = none` keeps SVG text as real, searchable text.
- **Cost.** The SVG is about 65–125 KB. Text uses DejaVu Sans; a viewer without it substitutes a similar font.

### E-6 Walls: per-room mitred rings, measured vs nominal thickness

- **Options.**
  - (a) Draw each wall as a rectangle (gaps or overlaps at corners).
  - (b) Buffer the room polygon outward by one uniform thickness (no per-wall thickness).
  - (c) Offset each edge by its own thickness and intersect neighbouring offset lines (mitre), with a bevel fallback.
- **Choice.** (c), then: union of all rooms' rings, minus all room interiors, minus the opening cuts.
- **Why.** It gives clean corners at any angle, including non-Manhattan walls, and uses the measured thickness
  wherever the extractor provides one. Subtracting the interiors guarantees that no wall is drawn inside a room.
- **Nominal thickness.**
  - 0.10 m is a typical interior partition: stud wall with plasterboard, roughly 0.075–0.125 m.
  - Envelope walls are usually thicker (0.2–0.3 m), but their outer face is never observed from inside, so any
    value would be a guess.
  - We draw 0.10 m **hatched** and say "nominal" in the legend and the DXF title. This is a drawing convention, not a
    measurement: the JSON keeps `thickness: null`.
- **Evidence.** `outputs/export/synthetic/plan.png`: 11 of 27 walls nominal, all corners clean, including the 45°
  bay.

### E-7 Dimension placement: envelope walls outside, partitions inside their own room

- **Options.**
  - (a) Always outside each wall, as literally specified.
  - (b) Always inside each room (magicplan's "interior dimensions" style).
  - (c) Outside for envelope walls, inside for partitions.
- **Choice.** (c).
- **Why.**
  - For a partition, "outside" is the neighbouring room, where that room's own dimension for the other face of the
    same partition already sits. With (a) the two lines collide, and a reader cannot tell which number belongs to
    which room.
  - (c) matches the spec where it is unambiguous (envelope walls) and stays readable elsewhere.
- **Fallback order** (see Issue 3):
  1. own side, no overlaps;
  2. own side, allowed over a door swing;
  3. the other side;
  4. forced placement, counted in `dims_forced`.

### E-8 Greedy label placement with measured text extents

- **Options.**
  - (a) Fixed positions, accepting overlaps.
  - (b) A label-placement optimiser (simulated annealing, or the adjustText package, which is not installed).
  - (c) A greedy candidate list per label, with real text bounding boxes.
- **Choice.** (c).
- **Why.**
  - It is deterministic, fast (the whole fixture renders in 0.6 s warm, 1.2 s from the CLI) and explainable in one sentence.
  - It is good enough in practice: 0 forced dimensions and 0 forced room labels on the fixture.
  - (b) adds a dependency and run-to-run variation for marginal gain.
  - Every compromise is counted in `render_report.json`, so crowding shows up as a number.

### E-9 Door swing by convention

- **Context.** Nothing in the pipeline measures hinge side or swing direction. (It could, from the door leaf's
  LiDAR points, but that does not affect any gated number.)
- **Choice.** Draw the conventional symbol swinging into the room with fewer doors (bedroom rather than corridor,
  WC rather than bathroom), and state on the plan that the swing is drawn by convention.
- **Why.** Homeowners expect door symbols. Omitting the arc makes a door look like a passage, and presenting the arc
  as measured would be dishonest.

### E-10 DXF with real DIMENSION entities and our text

- **Options.** (a) Lines plus text. (b) Real `DIMENSION` entities with `<>` (CAD computes the length). (c) Real
  entities with override text.
- **Choice.** (c).
- **Why.**
  - (a) is not a dimension to a CAD user: it cannot be restyled or re-scaled.
  - (b) would show the endpoint distance and drop our interval.
  - (c) keeps the entity semantics and shows our measured value and 95% interval.
- **Evidence.** The test `test_dxf_is_cad_native` re-reads the file and checks: `$INSUNITS == 6`, audit errors 0,
  27 DIMENSION entities all on layer DIMENSIONS, and each text containing "±".

### E-11 Synthetic fixture, derived not typed

- **Context.** The plan extractors (`floorplan/plan/alpha`, `beta`) were being built in parallel, so no real
  `plan.json` existed while this module was built.
- **Choice.** A hand-designed apartment in which walls come from polygon edges, wall thickness from the gap to the
  neighbouring room's parallel wall, and opening-to-wall links from distance.
- **Why.** Self-consistent by construction, with known answers, and it covers every case the real data brings:
  - an L-shaped corridor (the connector) and a tiny 1.1 m WC;
  - a 45° wall;
  - measured and nominal thickness;
  - doors, a passage, windows, and a low-confidence frosted window;
  - an inferred wall length and a not-observed ceiling;
  - damage, a concealed-damage flag and scope items.

  A second `stress_plan()` (16 rooms of 0.8–4 m, rotated 30°, missing lengths) checks robustness, not realism.
- **Evidence.** The fixture validates. `floorplan/eval/match.py` matches it to itself with 6/6 rooms, 27/27 walls
  and 11/11 openings, which shows the JSON format interoperates with the evaluation harness.

## 4. Issues log

| # | Symptom | Root cause (evidence) | Possible fixes | Chosen fix and why |
|---|---|---|---|---|
| 1 | First render: the corners where two **nominal** walls meet were drawn solid black, so they looked measured | `nominal = solid ∩ nominal quads`; a mitred corner square belongs to no edge's quad, so it fell into "measured" (visible at the corridor's top-left corner, first render) | (a) mitre the nominal bands themselves; (b) define nominal as `solid − measured quads`; (c) ignore it | (b): one line, and it is correct by definition: "nominal = everything not backed by a measurement" |
| 2 | The window width label was printed on top of the glazing lines | The first label candidate for every opening was "inside the cut-out gap", which is empty for doors but holds the glazing symbol for windows | (a) a white background box behind the text; (b) skip the gap candidate for windows | (b): a box would hide the window symbol; for windows the label goes just inside the room |
| 3 | WC: the dimension and the room label sat on the door swing. Reserving the swings as hard obstacles then **forced 4 dimensions and the WC label** (`dims_forced` 0 → 4). Allowing a flip to the other side fixed the count but put the bedroom's "3.40 m" line in the corridor, right on the corridor's "5.00 m" line, which is ambiguous | Small rooms have no free space on the own side once the swing is blocked; flipping sides breaks the "dimension sits next to its wall" rule | (a) swings as hard obstacles + flip; (b) ignore swings; (c) **soft** obstacles with staged preference: own side clean → own side over a swing → other side → forced | (c): which wall a number belongs to matters more than a little overlap with a thin arc line. Result: `dims_forced = 0`, `room_labels_forced = 0`, and no dimension on the wrong side in the fixture |
| 4 | Stress plan: small hatched ("nominal") squares at the 4-way junctions of measured partitions | After fix 1, junction squares were outside every measured quad (quads stop at the wall ends) | (a) extend measured quads by their thickness when subtracting; (b) mitre-aware attribution | (a): squares next to a measured wall become measured; a corner between two nominal walls stays hatched. Verified visually on a crop of `outputs/export/stress/plan.png` |
| 5 | `room_labels_full = 0` on the stress plan although some labels were complete | The stat counted "candidate index 0" as full; a full label at a fallback *anchor* has index > 0 | Compare the text and font size instead of the index | Done; the stats now mean what they say (4 full / 12 reduced on the stress plan) |
| 6 | DXF preview: some dimension texts upside down (living room's bottom wall, kitchen's right wall) | An ezdxf aligned dimension orients its text along p1→p2; for walls whose view direction points left or down, the text flips | (a) set the text rotation explicitly; (b) order the endpoints so p1→p2 points right (or up) and recompute the offset side | (b): keeps the default dimstyle behaviour; verified in `plan_dxf_preview.png` |
| 7 | A validator test expected "no 95% interval", but the error came from the schema | The schema already requires `ci95` to be an array when status ≠ not_observed, and semantic checks run only after the schema passes, so the semantic check could never fire | (a) keep both; (b) remove the dead semantic check | (b): no dead code; the test now asserts the schema error |
| 8 | The fixture JSON listed every coordinate on its own line, unreadable for a reviewer | `json.dumps(indent=2)` puts every list element on a new line | (a) a custom encoder; (b) a regex that collapses number pairs | (b): 3 lines of code, output still standard JSON (`dumps` in `json_export.py`) |
| 9 | The diagnostic figures from `scripts/prepare_scene.py` are mirror images of the apartment | They plot v = z upward; the frame is right-handed (det = +1), so v should point down | Change the shared script (not ours) | Requested below; our renderer and DXF already use the correct view |
| 10 | No real plan to render yet (`outputs/plan_alpha`, `outputs/plan_beta` did not exist at the end of this work) | The extractors are being built in parallel | Wait; or render a synthetic plan | Built and verified on the synthetic fixture and the stress plan. Command to run on real plans, once they exist, is in §5 |

## 5. Results

All numbers below come from `scripts/render_plan.py` and `pytest`, run on 2026-10-03. Evidence is in `outputs/export/`.

| Plan | Rooms / walls / openings | Valid | Dimensions drawn (forced) | Room labels full / reduced / forced | Opening labels placed / dropped | Nominal walls | SVG+PNG render time |
|---|---|---|---|---|---|---|---|
| Synthetic apartment (`synthetic/`) | 6 / 27 / 11 | yes | 27 (0) | 5 / 1 / 0 | 10 / 1 | 11 | 1.23 s |
| Stress plan, 16 rooms, rotated 30° (`stress/`) | 16 / 64 / 12 | yes | 64 (4) | 4 / 12 / 0 | 0 / 12 | 16 | 2.49 s |

- Figures:
  - `outputs/export/synthetic/plan.png` (+ `.svg`, `.dxf`, `plan_dxf_preview.png`, `render_report.json`);
  - `outputs/export/stress/plan.png` (+ the same set);
  - `outputs/export/handedness_check.json`.
- DXF (synthetic): 27 DIMENSION entities; per-layer entity counts WALLS 32, ROOMS 6, TEXT 18, DIMENSIONS 27,
  OPENINGS 32; 0 audit errors; `$INSUNITS = 6`.
- Tests: `14 passed`. They cover:
  - the fixture is valid and up to date;
  - the round trip is lossless;
  - 7 kinds of broken documents are each rejected;
  - the SVG is byte-identical across two runs, for the fixture and for a 25-room stress plan rotated 17°;
  - degenerate and empty plans render without crashing;
  - the DXF is CAD-native.
- Interop: `floorplan/eval/match.match_plans(fixture, fixture)` matches 6/6 rooms, 27/27 walls and 11/11 openings.
- The stress plan is deliberately extreme (0.8 m rooms at 0.35 in/m), and the stats show it honestly:
  - 4 dimensions had to overlap something;
  - all 12 opening-width labels were dropped (the widths remain in the JSON).
- **Real captures:** not rendered yet, because no extractor output existed. Once `outputs/plan_alpha/<capture>/plan.json`
  or `outputs/plan_beta/<capture>/plan.json` exist, run:
  `python scripts/render_plan.py outputs/plan_alpha/<capture>/plan.json --out outputs/export/alpha__<capture> --dxf-preview`.

## 6. Limitations and failure modes

**Mirrors, glass, wet-look surfaces and low light** (the case study's constraints). The export layer cannot fix bad
geometry, but it must *show* uncertainty honestly instead of hiding it:

- **Glass (windows, shower screens) and low light.** LiDAR returns are missing or unreliable here (D-006). The
  extractor should then:
  - lower the opening's detection `confidence`; the renderer draws such openings orange and dashed (the fixture's
    frosted WC window, confidence 0.4);
  - mark lengths it could not see as `inferred` (grey italic "≈") or `not_observed` ("n/o").

  Wider intervals in low light simply show up as a larger ±. Nothing is invented in the export.
- **Mirrors.** A mirror shows a phantom "room" behind the wall. The renderer will faithfully draw whatever room the
  plan contains, so the defence has to happen upstream: the extractor must reject regions seen only through a
  planar specular surface. The schema's `evidence` and `meta.warnings` fields are the place to record that.
- **Wet-look floors.** These are a damage and false-positive question. The schema carries `damage[].confidence`,
  `concealed` and `rule`, and floor damage with a `plan_polygon` is drawn hatched red, so a false positive is
  visible and auditable on the plan.

Other limitations, with what we would do next:

1. **Nominal thickness for envelope walls.** We draw 0.10 m because the outer face is never observed. With a window
   reveal (the depth of the window opening) the extractor could measure envelope thickness; the renderer already
   uses any measured value.
2. **The door swing is conventional**, not measured (E-9). Next: detect the leaf plane at the door and add a
   `swing` field to `Opening` (schema 1.1, a backward-compatible addition).
3. **Very dense plans.** The greedy placer degrades gracefully but does not optimise globally: 4 forced dimensions
   on the 16-room stress plan. Next: a second pass that tries moving already-placed labels, or a leader-line callout
   for walls under 0.6 m.
4. **The DXF has no label-collision handling.** CAD users rearrange text, and fixed metric text heights overflow
   small rooms (visible in the stress preview). Next: reuse the Labeler positions.
5. **Single storey.** `Room.floor_level` is exported, but all rooms are drawn in one plane. A multi-level property
   would need one sheet per level.
6. **Curved walls** are drawn as polylines; RoomPlan represents them as curves. There is no curve type in our model
   yet.
7. **The SVG depends on the viewer's fonts** (`svg.fonttype = none`). Embedding glyph paths would make it
   self-contained but larger and not searchable.
8. **The schema is ours, not the company's.** If they share their published schema, we add a mapping function that
   turns our `Plan` into their JSON; the internal model does not change.

## Requested changes to shared code

1. **`scripts/prepare_scene.py` (figures):** plot with an inverted v axis (`ax.invert_yaxis()`) or plot `-z`, so the
   diagnostic overviews show the apartment unmirrored, consistent with the rendered plan (Issue 9 / E-4).
2. **`floorplan/model.py` (optional, schema 1.1):**
   - add `Opening.swing: Optional[str]` (`"left_in"`, `"right_out"`, …) once swing is measured;
   - add `Plan.meta["warnings"]` as a documented convention for mirror and glass flags.

   Both are backward-compatible additions.
3. **Damage and scope producers:** emit dicts with at least `id`, `class` and `surface_id` (damage), and `id`,
   `surface_id`, `description` and `quantity` (scope), with `Measurement` objects for metric extents. The exporter
   converts the `Measurement` objects; the schema rejects records missing these keys. A concealed flag must carry
   `concealed: true` and the `rule` that fired.
