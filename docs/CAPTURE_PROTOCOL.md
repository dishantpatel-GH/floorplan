# Capture protocol (Route 2: stock apps only, one page)

Follow it literally. Background: `docs/archive/CAPTURE_PRACTICES_RESEARCH.md` (published practice) and D-051, D-055,
D-064, D-067, D-068, D-074 in `docs/DECISIONS.md` (tests in the simulator and on my own flat).

**Install and prepare (5 min)**
- Photos and video: any iPhone 15 or newer (any Android phone works too). The built-in **Camera** app; nothing to install.
- LiDAR: an iPhone Pro / Pro Max (12 Pro or newer) or an iPad Pro with LiDAR, and the free **Stray Scanner** app (App Store).
- Nothing else. No tape, no paper sheet, no reference object (D-067).
- iPhone: Settings → Camera → Formats → **Most Compatible**. Camera app: the **1×** lens for everything (never 0.5×,
  never zoom); Photo mode in the default **4:3**, flash off, no Portrait, Pano or Night mode; Video at **1080p, 30 fps**,
  with Action Mode, Cinematic and Macro Control off. Leave the camera data (EXIF) on, which is the default.
- In the home: **all lights on**, curtains open, **every inner door fully open**, nobody walking through.

**How long:** photos about 1 minute per room. Video and LiDAR about 8 minutes for a 1–2-bedroom flat (two loops per room).

## Photos: one folder per room, at most 8 photos

Phone held **sideways** (landscape), chest height, held still for each shot.
1. **Normal room: turn on the spot.** Stand in the open middle, at least 1.5 m from every wall, phone tilted a little
   down so the line where the wall meets the floor is in the picture. Start **facing the door you came in by**, then
   turn **clockwise**, each photo overlapping the last by about a quarter: 4 photos in a room under 10 m², 5 up to 18 m²,
   6 above. **Last: one ceiling photo**, phone clearly tilted up (about 40°) along the room's long side.
   Count: turning photos + 1 ceiling photo + 1 doorway photo per door (step 3) must stay within 8; in a room with many
   doors take fewer turning photos (3 at least).
2. **Small room** (bathroom, small kitchen, balcony, passage) **or a room too full to stand in the middle** (less than
   1 m free to the walls): 5 photos. From the doorway, looking in, turned a little left (this is also the doorway-pair
   photo into the room). From the far end, looking back at the door: one turned a little left of it, one a little
   right. One ceiling photo, clearly tilted up. From the doorway on your way out: one turned a little right. Balcony
   (or any room under about 2 m deep): also one photo through its door from 2 m back in the room next to it; it goes
   in the balcony's folder.
3. **Doorway pair at every door between two rooms**, including doors you do not walk through. Stand on the threshold:
   one photo back into the room you are leaving, aimed at its open middle; then turn round and, within 5 s, one photo
   into the next room. Each photo belongs to the room it looks into. **Keep the door leaf out of the picture:** if the
   open door would fill the frame, step 1–1.5 m back and aim through the doorway at the next room's furniture.
4. Before leaving a room, check that **every wall is in at least one photo**; if one is missing, add a photo of it
   (stay within 8).
5. Sort the photos into one folder per room, named after the room (`kitchen`, `bedroom1`, ...). Doorway photos go in
   the folder of the room they look into. Do not rename, rotate, crop or edit the files.

## Video: one clip, the whole home

1. Record **once**. Walk **forward**, slowly; never sideways or backwards, never spinning on the spot.
2. In each room, two loops about 1 m in from the walls: first with the phone tilted **down** (about 25°) at the floor
   line, then again tilted **up** (about 25°) at the ceiling line, phone turned a little into the room so the walls
   across the room pass through the picture. Small rooms: walk in to the far end, look left and right at the floor
   line, then at the ceiling line, walk back out.
3. In every doorway, **pause 2 s** facing into the next room with both door frames in view.
4. **Finish where you started** and film the first view again for about 3 s.

## LiDAR: Stray Scanner

Tap record and walk **the video route above** (floor-line loop, then ceiling-line loop in each room); without the
ceiling in view there is no ceiling height. Stay within about 3 m of the walls. Stop, then export the recording folder
(Stray Scanner → the recording → Share → Files or AirDrop).

## Avoid (all tiers)

- **Mirrors:** never film or photograph a mirror head-on. Stand to the side so a mirror fills less than a quarter of
  the picture; seen head-on, a mirror shows the pipeline a room behind the wall that is not there.
- **Glass** (glass doors, partitions, large windows): aim at the frame and the wall around the glass, not straight
  through it; the LiDAR and the depth model see through a pane, so its frame is what places it.
- **Wet-look or glossy floors:** dry wet floors first, and keep the wall-floor line in the picture; reflections on a
  shiny floor are not walls.
- **Low light:** all lights on, capture by day if you can. If the picture looks grainy or smears when you move, add
  light or move more slowly.
- Fast turns, close-ups of a blank wall, doors opening or closing, people moving, zoom, filters, editing.

## Hand the files over

1. Copy by **USB cable**, AirDrop or the Files app. **Never** WhatsApp, Telegram or e-mail: they shrink the files and
   remove the camera data (capture time and focal length, which the pipeline uses).
2. Put them in the repo in a new folder `data/<capture>/` (any name), laid out like `data/own_house/` and `data/k65/`:
   - `photos/<room>/` (one folder per room, as above)
   - `video/<name>.mov` (or `.mp4`, as the phone saved it)
   - `lidar/<scan>/` (the Stray Scanner folder, as exported)
3. Run one command per tier (environment from `README.md`); the first line is an optional check of the folder:
   ```bash
   python scripts/check_own_capture.py data/<capture>     # seconds: PASS / WARN / FAIL per item
   python scripts/run_capture.py data/<capture>/photos --tier photo
   python scripts/run_capture.py data/<capture>/video/<name>.mov --tier video
   python scripts/run_capture.py data/<capture>/lidar/<scan> --tier lidar
   ```
   Each run prints its output folder (under `outputs/runs/`): `plan.png` is the drawing, `plan.json` the dimensioned
   plan with intervals, `plan.dxf` for CAD, `run_report.json` what the run did and what it could not measure.

Scale comes from the files themselves (learned metric depth, the lens data in each file and priors), and the
intervals include that uncertainty (D-067).
