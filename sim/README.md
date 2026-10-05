# Simulator: phone captures of a virtual house, with exact ground truth

**Why it exists.** 30% of the score is the walk-in test: someone else captures an unseen space with their iPhone,
following our protocol, and our pipeline runs cold. The simulator rehearses exactly that, as often as we like:
- many houses;
- many "people" (seeds that change how carefully the protocol is followed);
- all three tiers;
- millimetre-exact ground truth.

It also gives the fix loop (25%) cheap, exact before/after numbers. It does **not** replace the real captures: the
benchmark (15%) must be measured on real data. Sim numbers are always labelled as simulated.

**Real-data parity.** The simulator writes the same folder layout and file formats a phone hands over
(`photos/<room>/*.JPG` with EXIF, `video/*.MOV`, Stray Scanner `lidar/<take>/`, `gt/ground_truth.csv`). The same
command processes both: `scripts/process_own_capture.py <capture folder>`.

**What the submission uses (5 Oct).** One simulated flat, k65 (InteriorAgent `kujiale_0065`, 1 BHK), rendered in
iPhone 15 format; it ships as `data/k65/` (`scripts/fetch_data.py`), so the photo tier can be re-scored without Isaac
Sim. The photo tier is judged on k65 only; k22 and k38 are notes, not gates (D-088).

## The pieces

| Step | Script | Interpreter | What it does |
|---|---|---|---|
| 0a | `isaac_omap.py <scene.usda>` | Isaac* | Isaac Sim occupancy map (PhysX colliders) at body/phone height, 0.10–1.90 m |
| 0b | `scene_gt.py <scene.usda>` | `.venv` | Ground truth from the visible surfaces (ray casting): walls, ceilings, doors, windows. Walkable map = Isaac occupancy ∪ every object's triangles at 0.10–1.90 m, with 25 cm margins. A4 sheet spots |
| 1a | `teleop.py --session <dir>` | Isaac* | **You** drive the phone (WASD + arrows) and record walks and photos |
| 1b | `auto_capture.py --session <dir>` | `.venv` | A scripted person follows `docs/CAPTURE_PROTOCOL.md` literally |
| 1c | `door_photos.py plan\|assemble` | `.venv` | Adds door photos to a photo session: one out through each door from 1.0-1.5 m inside the smaller room, one in from the neighbour; spot and heading chosen by ray-cast labels (the next room's textured objects seen through the opening). Renders with `render.py`, emulates like the rest |
| 2 | `render.py <dir>` | Isaac* | RTX colour frames of the walk and photos, with human imperfection added |
| 3 | `emulate.py <dir> --profile iphone15` | `.venv` | Phone files: JPEG+EXIF, MOV, Stray Scanner folder with LiDAR noise, confidence and drift |
| 4 | `scripts/process_own_capture.py <capture>` | `.venv` | The real pipeline on all tiers, scored against the GT |
| all | `experiment.py --name <x>` | `.venv` | Runs 0→4, skipping finished stages |

\* Isaac = `envs/isaacsim_np126/bin/python`. This is a small venv that reuses `~/env_isaaclab` with numpy 1.26.
Isaac Sim 5.1 needs numpy 1.26; `env_isaaclab` has numpy 2.4.2, which crashes Replicator (issue I-008). Your
`env_isaaclab` is untouched.

## Drive it yourself

```bash
cd ..        # the folder that holds the repo, .venv/ and envs/
envs/isaacsim_np126/bin/python floorplan-capture/sim/teleop.py --session floorplan-capture/outputs/sim/my_walk
```

Click the viewport once, so it has keyboard focus.

| Key | Action |
|---|---|
| W / S | walk forward / back |
| A / D | step left / right |
| Q / E | lower / raise the phone |
| arrows | turn left/right, tilt down/up |
| Shift | 3× faster (repositioning only) |
| **R** | start / stop a recording (video + LiDAR walk) |
| **P** | take a photo. It goes to the room in front of the camera, so doorway pairs sort themselves |
| 1-9 / 0 | force the photo folder to room N / back to automatic |
| T | photos go to `<room>_take2` (the repeat set) |
| Backspace | delete the last photo |
| O | portrait / landscape |
| L | lighting day / dim |
| H | back to the start |

- The HUD shows a minimap, your room, the photo folder, warnings ("wall", "furniture", "turning fast") and REC.
- Walls block you. Furniture only warns.
- Space is **not** used: in Isaac Sim it plays the timeline, and the house's furniture has rigid-body physics. The
  tool keeps the timeline stopped.
- Everything is saved to `session.json` after every take and photo.

Then render and process it (this skips the scripted session, because yours exists):

```bash
.venv/bin/python floorplan-capture/sim/experiment.py --name my_walk --session floorplan-capture/outputs/sim/my_walk
```

Your keyboard path is the intended path. `render.py` adds what a hand does:
- height drifting about ±7 cm;
- tilt about ±5°;
- roll about ±2°;
- aim wobble, walking bob and tremor.

So even a perfectly level keyboard walk renders like a real handheld one.

## Scripted sessions (protocol v2, D-051: designed once from published practice + user review)

The scripted person follows `docs/CAPTURE_PROTOCOL.md` (`sim/protocol_v2.py`).
- **Photos.** Normal rooms get a spin from the open middle plus 1 ceiling photo. Small rooms get one shot from the
  doorway, one from the far end looking back, plus 1 ceiling photo. Doorway pairs once per door.
- **Walk.** Loops along each room's walls, walking forward, with corner glances. Small rooms: step in, pan, step out.
  Pass 1 is aimed at the floor line, pass 2 repeats the route aimed at the ceiling line.
- **Checks.** Human imperfection is applied at path time, then every pose is checked in 3D: no camera closer than
  25 cm to any surface (`session.json` → `meta.clearance`).

## Older scripted sessions (v1; kept for comparison)

```bash
.venv/bin/python floorplan-capture/sim/experiment.py --name k65_s1 --seed 1          # another "person"
.venv/bin/python floorplan-capture/sim/experiment.py --name k65_corners --photo-protocol corners
.venv/bin/python floorplan-capture/sim/experiment.py --name k65_s1 --seed 1 --drift bad --tag badvio --emu-seed 3
```

**Photos, following the page.** For each room:
- **Spin.** N = 8 − doors photos, clockwise. Consecutive photos overlap by a different amount each time (10–50% of
  the view), and the height and tilt drift from shot to shot.
- **Doorway pair**, the first time through each door: one photo back into the room being left, then one into the
  next room, 2.5–4.5 s apart.
- **Repeat set.** One room is shot again into `<room>_take2`.

**Walk (video and LiDAR share it).** The route visits every room. In each room, the scripted person:
1. views the A4 sheet on the way in (1.5 m away);
2. walks to the open middle;
3. makes one slow (≤ 20°/s) full turn while the tilt swings between the floor line and the ceiling line;
4. views the sheet again from a second spot;
5. pauses 2 s in every doorway, facing into the next room.

The walk finishes at the start and holds the first view for 3 s.

## What the phone emulation does (emulate.py), and why it is believable

- **LiDAR depth.** Depth is ray-cast against the same triangles Isaac renders. Checked against Isaac's own depth
  output: **0.00 mm difference at the 99th percentile**, with principal point (W/2 − 0.5, H/2 − 0.5).
  - Each depth pixel averages 3×3 sub-rays, so depth edges blend like the real 256×192 sensor's.
  - **Noise** is the σ(range) we measured on the real Stray Scanner sample (`lidar_noise.json`), spatially
    correlated, and ×4 at medium confidence.
  - **Glass** lets the pulse through. **Mirrors** reflect it: the depth is the path into the "room behind the
    mirror", with high confidence. That is how a real iPhone is fooled.
  - **Check:** high-confidence points land 5–9 mm (median) from the true surfaces.
- **Poses.** ARKit-like world: Y up, origin at the first frame, OpenCV camera axes, as verified on the real sample.
  Visual-inertial drift is added:
  - yaw random walk 0.25°/√m;
  - position random walk 0.6 cm/√m;
  - 0.4% scale error.

  Together these give revisit gaps of about 10–15 cm over a whole flat; the real sample had 14 cm.
- **Images.**
  - **Auto exposure**: up to 8× gain in dim scenes, with shot and read noise that grow with the gain.
  - **Motion blur** from the camera's rotation during a 1/60 s exposure (1/30 s when dim).
  - **Compression**: JPEG q92, H.264, HEVC.
- **Metadata.**
  - **Photo EXIF.** `FocalLengthIn35mmFilm` is written as an integer, by the standard diagonal definition, and
    `DateTimeOriginal` / `SubSecTimeOriginal` come from the session clock.
  - **iPhone video.** Carries `com.apple.quicktime.camera.focal_length.35mm_equivalent`.
  - **Android video.** Carries none, as on a real Android phone.
- **Cameras.**
  - Walks use 1920×1440 with fx = 1598 px, the real sample's iPhone ARKit format.
  - Videos are its 16:9 centre crop.
  - Photos use the iPhone 15 main camera's 26 mm equivalent (2000×1500 render).

## Known limits (honest list)

- **Rendering.** RTX real-time with 1 subframe per video frame, so a little temporal smoothing (like mild motion
  blur). Photos use 12 subframes. There is no rolling shutter.
- **Furniture.** Scenes are furnished and tidy: no people, no clutter piles, no open wardrobes.
- **Drift model.** Random-walk VIO drift; ARKit's relocalisation jumps are not modelled.
- **Speed.** Rendering takes ~0.4 s per 1920×1440 frame on the RTX 2000 Ada (8 GB), so walks are rendered at 10 fps
  (the video and LiDAR tiers subsample anyway). A 7-minute protocol walk of a 5-room flat renders in ~27 min and its
  45–50 photos in ~2 min. Emulation encodes with NVENC (D-043) and casts the LiDAR rays in 6 worker processes.
- **Scenes.** InteriorAgent scenes (kujiale_*) are under their own terms of use (docs/DISCLOSURES.md). They are
  fetched by script and never committed.
