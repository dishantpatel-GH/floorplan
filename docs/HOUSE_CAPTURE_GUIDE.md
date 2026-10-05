# House capture guide: what to capture and measure

**Goal:** a benchmark of your own home at the **photo** and **video** tiers, with **tape-measured ground truth**. The
case study requires it, and the recruiters confirmed: "capturing your data is mandatory". The LiDAR tier uses the sample
data (allowed by the recruiters).

**Time needed:** about 90 minutes. Do the steps **in this order**, and tick each box as you go.

---

## 0. What you will hand back (the folder layout)

```text
OwnCaptures/
  photos/
    01_<room>/          2-8 photos of that room (one folder per room)
    02_<room>/
    ...
    03_<room>_take2/    the same room shot a second time (repeatability)
  video/
    take1.mp4           whole-house walkthrough, first time
    take2.mp4           the same walkthrough, second time
    lowlight.mp4        optional: one take with fewer lights on
  app_export/           consumer app screenshots or exports (Matterport, CubiCasa; step 7) + app versions
  gt/
    ground_truth.csv    your measurements (template: benchmark/ground_truth.csv)
    sketch_<room>.jpg   photo of each hand sketch
```

---

## 1. Prepare (10 min)

- [ ] Charge the phone. Check there is at least 5 GB free.
- [ ] Clean the camera lens with a soft cloth.
- [ ] Have ready: tape measure (or laser measurer), pen and paper, masking tape, and **one plain A4 sheet** for the
      staged water stain (step 3).
- [ ] Choose the rooms: **at least 3 rooms plus the hallway/passage connecting them**. Include, if you can:
  - the bathroom (it has a **mirror**, which the brief asks us to cover);
  - a room with a **window** (glass).
- [ ] Choose **one furnished room** (bedroom or living room) for the staged damage.
- [ ] Open all internal doors fully, turn on all lights, and open curtains.
- [ ] Ask anyone at home not to walk through the rooms during capture.

## 2. Phone camera settings (2 min)

**Your OnePlus Nord** (OnePlus Camera app; menu names vary slightly with the OxygenOS version)
- **Photo mode:**
  - Use the **1×** main lens for photos and video. Don't use 0.5×/0.6× (ultra-wide) or any zoom.
  - Keep the normal resolution (12 MP). Don't use the 48 MP or "Pro" mode.
  - Keep the **4:3** photo format (the default; not 16:9, not 1:1). Videos are 16:9; that is fine.
- **Settings (gear icon):**
  - Turn **off** "Scene detection / AI scene enhancement" and "Super stable".
  - Keep "HDR" on Auto.
  - Video resolution: **1080p 30 fps**.
  - If there is a "Video encoding" option, choose **H.264** (not H.265/HEVC).
  - Turn **off** video stabilisation and "Nightscape" for video, if offered.
- **Flash off.** No Portrait, Panorama, Nightscape photo, filters or beauty mode.
- **Leave location/EXIF data on.** That is the default; the focal length stored in each photo helps the pipeline.
- **To test ARCore** (CubiCasa in step 7 needs it): install "Google Play Services for AR" from the Play Store. If it
  installs, your phone supports ARCore.

**iPhone**
- Settings → Camera → Formats → **Most Compatible**.
- Settings → Camera → Record Video → **1080p at 30 fps**.
- Turn off Action Mode and Cinematic.

**Android**
- Camera app → Settings: picture format **JPEG** (not RAW-only).
- Video **1080p, 30 fps**.
- **Video stabilisation OFF**, if your camera offers the switch.
- **Turn off "scene optimiser" / AI beautify.**

**Both phones**
- The main **1× lens** for everything. Never 0.5×/0.6×, never zoom.
- No Portrait or Panorama mode. Flash off. Don't edit any photo.

## 3. Stage the damage (10 min). Nothing goes directly on the wall

Use two **different damage classes**, in the furnished room:

- [ ] **Class A, water stain.**
  - Paint an irregular, **clearly brown/yellow (not pale)** blotch, 20–40 cm across, on an A4 sheet. Tea, coffee or watercolour works; make
    the edge uneven, like a real stain.
  - Let it dry. Tape it **flat** on a wall, low (near the skirting) or high (near the ceiling).
- [ ] **Class B, crack.**
  - Draw a thin, jagged dark line 30–60 cm long on a strip of masking tape or paper.
  - Tape it to a **different** wall, running diagonally from the corner of a door or window, as real cracks do.
- [ ] Take one close-up photo of each and keep it **outside** the room folders (`gt/damage_closeups/`).

## 4. No scale sheet (changed 4 Oct)

Nothing to place: skip this step.

No reference object is needed. The software works out sizes from the photos and video themselves: it estimates depth,
reads the lens data stored in each file and, for photos, assumes the usual height a phone is held at. The A4 sheet
from step 1 is only for the staged water stain (step 3).

## 5. Photo tier (about 1 minute per room). Revised 4 Oct, 20:15 (protocol v2.2: every wall, data audit; 1× lens)

**Trick for sorting later:** before each room, take one photo of a paper with the room name written on it. These
"slate" photos tell us where each room starts. They are labels, not a scale reference. We delete them afterwards.

For each room:
1. [ ] Slate photo (room name).
2. [ ] Hold the phone **horizontally** (landscape, 4:3) at chest height, on the **1× main lens** (not 0.5×/0.6×, no zoom). Hold still for each shot.
Every room gets photos **looking in** from each of its doors and **looking out** through them, so neighbouring
folders share views and the rooms can be joined.

3. [ ] **Normal rooms (living room, bedrooms):**
   - **Stand in the open middle, as far from every wall as you can** (ideally 1.5 m or more). Tilt the phone
     slightly down so the floor line is visible, and **keep this same tilt** for the turning photos. (Shots tilted
     differently matched only 24% of the time in tests, against 100% with the same tilt.)
   - **Start facing the door you came in through**, then **turn on the spot, always clockwise (to your right).**
     Bigger rooms get more photos: about **4** under 10 m², **5** up to 18 m², **6** above. Each overlaps the previous
     by about a quarter. Keep the room's folder within **8** photos in total (turning + 1 ceiling + 1 doorway photo per
     door); a room with 4 doors therefore gets 3 turning photos.
   - Then **1 ceiling photo**: tilt the phone up (about 40°) along the room's long side, so the ceiling and the line
     where it meets the walls are in the picture. The ceiling photo is always the last turning photo.
4. [ ] **Small rooms (bathroom, small kitchen, balcony): 5 photos (balcony 6).**
   - From the **doorway**, looking in, turned a little to the **left**, door frame visible (this is the doorway pair's
     photo into the room).
   - From the **far end**, looking back at the door: **2 photos**, turned a little left of the door, then a little
     right of it. The door is in both.
   - **1 ceiling photo**, as above.
   - On the way out, from the **doorway**: **1 photo** into the room turned a little to the **right**.
   - **Balcony** (or any room under about 2 m deep): last, step back about 2 m into the room next to it and take
     **1 photo of the balcony through the door** (it goes in the balcony folder).
5. [ ] **Doorway pair at every door between two rooms** (once per door, also doors you don't walk through):
   - Stand **on the threshold**.
   - Take **1 photo back into the room you are leaving**, aimed at its middle.
   - Turn around and take **1 photo into the next room**, aimed at its middle (it goes into the next room's folder).
   - Take both **within about 5 seconds**, with nothing in between.
   - **Keep the door leaf out of the picture.** If the open door would fill the frame, step 1-1.5 m back and aim
     straight through the doorway, so the next room's furniture fills the opening.
6. [ ] Check that **every wall** of the room appears in at least one of its photos (if not, take one more of it). A room too full to stand in the middle (bed, table): use the small-room method.
7. [ ] Total per room: **at most 8 photos** (slate excluded).

Then:
- [ ] Shoot **one room a second time**: leave, come back, repeat its full set. This is the repeatability take.
- [ ] Hold still for each shot (no blur). Keep yourself out of mirrors as much as possible.
- [ ] **Never send the photos through WhatsApp** or similar. Their capture times are what pair the doorway shots, and
      their focal length sets the scale.

## 6. Video tier (one walk, room by room, about 1 min per room). Revised 4 Oct, 20:00 (protocol v2.2)

Settings:
- **1080p at 60 fps** if offered (30 fps is acceptable).
- **Landscape**, main **1×** lens.
- Turn **off** automatic lens switching / macro and "Super stable" / "Ultra steady". A lens switch or crop in the middle
  of a clip breaks the measurement.

- [ ] **Take 1**, with all lights on. Press record **once** and don't stop until the end. Walk **forward**, slowly,
      never sideways or backwards, and **never turn on the spot** (walk while you turn).
  1. **Room by room: first the floor line, then the ceiling line, then move on.**
     - **Normal rooms:** walk once round the room, about 1 m in from the walls, phone tilted **down** (about 25°) and
       turned a little **into the room** (about 30°), so the walls across the room pass through the picture. Glance
       into the corners as you pass them. Then walk the same loop again with the phone tilted **up** (about 25°),
       ceiling line in view.
     - **Small rooms (bathroom, kitchen, balcony):** walk **in** to the far end, look across the room to the left and
       right at the floor line, then left and right at the ceiling line, and walk back out.
     - **Doorways:** in **every doorway, pause 2 seconds** facing into the next room, with both door frames in view.
  2. Don't pan slowly across bright windows.
  3. **Finish where you started**, then film the starting view again for about 3 seconds. Stop recording.
- [ ] **Take 2:** the same walk again, as similar as possible. This is the repeatability take.
- [ ] *(Optional, recommended)* **Low-light take:** switch off about half the lights and repeat once. The brief asks us
      to cover low light.

## 7. Head-to-head with a consumer app (Part 3, about 30 min). Revised 4 Oct, 22:10

**magicplan cannot be used on Android.** It removed camera scanning from Android in version 2024.24.0 ("Android
devices are not supported for magicplan's scan features", https://help.magicplan.app/supported-devices); on Android it
only draws rooms from lengths you type in, which would just repeat the tape numbers. Use these free apps instead. Do
**Matterport first**: its cloud processing takes from under an hour to about 8 hours.

**A. Matterport** (free plan, no card; Android 9+ on a Google-certified phone; OnePlus is on its supported list)
- [ ] Install **Matterport** (`com.matterport.android.capture`) and sign up for the Free plan. Screenshot the app version.
- [ ] Create ONE space for both rooms (the free plan allows one). Choose this phone's camera as the capture device.
- [ ] Room 1: stand near the middle, tap scan and turn slowly through a full 360°. Move 1.5–2.5 m and scan again:
      3–5 spots per room. Then scan in the doorway, go into room 2 and do the same.
- [ ] Upload over Wi-Fi. When the space is ready, open **Measurement Mode** (app or my.matterport.com) and measure
      each wall corner to corner along the floor line, floor to ceiling, and each door and window frame to frame.
      **Screenshot every reading.** Also save the space's share link.

**B. CubiCasa** (one free scan, no card; **needs Android 12 or newer**: check Settings → About phone)
- [ ] Install **CubiCasa** and sign up (country India). Turn on all the lights.
- [ ] Do ONE scan that covers both rooms: walk slowly with the phone aimed at the line where the floor meets the walls,
      all the way round room 1, through the door, all the way round room 2. Submit it.
- [ ] Save the plan image when it arrives (6-hour delivery is included, not guaranteed). It gives each room's
      width × length and area, not every wall. Only one free order: if it is rejected, skip it.

**For both apps**
- [ ] Use **2 of the rooms you tape-measure** (step 8), the same 2 for both apps.
- [ ] Name the walls exactly as in step 8 (W1 = the wall with the room's main door, then clockwise) when you note
      which reading belongs to which wall.
- [ ] Put the screenshots in `app_export/` and write the app names and versions in `app_export/app_version.txt`.

## 8. Measure the ground truth (30–40 min). The most important part

Draw a quick sketch of each room. Name the walls **W1, W2, W3, …, going clockwise, starting with the wall that has the
main door**. Measure in **metres, to the millimetre** (e.g. 3.412). Write everything into
`gt/ground_truth.csv`, or on the sketches.

| What | How to measure it | Per room |
|---|---|---|
| **Every wall length** | Corner to corner along the wall, at **about 1 m above the floor** (above the skirting board). Keep the tape straight and taut. | all walls |
| **Ceiling height** | Floor to ceiling **near the middle of the room**. A second person helps hold the tape. With no helper, measure up a corner. | 1 (2 if the ceiling is uneven) |
| **Door / opening width** | The **clear width between the two side frames (jambs)**, at about 1 m height. | every door |
| **Door / opening height** | Floor to the top frame. | every door |
| **Window** | Width, height, and sill height (floor to the bottom edge of the glass/frame). | every window |
| **Diagonal** | One corner to the opposite corner. A check on the sketch. | 1 |
| **Damage regions** | Width and height of each staged stain/crack, plus its distance from the floor and from the nearest corner. | each damage |
| **Hallway** | The same as a room: its walls, its openings and its ceiling. | — |

Then:
- [ ] Measure each wall **twice**. If the two readings differ by more than 3 mm, measure a third time and write down the
      middle value.
- [ ] Photograph each sketch (`gt/sketch_<room>.jpg`).

## 9. Transfer (5 min)

- [ ] Copy everything with a **USB cable** (or AirDrop from an iPhone). **Never** use WhatsApp, Telegram or email: they
      compress the files and delete the camera data.
- [ ] Put the files in the folder layout from step 0, inside `TakeHome/OwnCaptures/<date>/` next to the repo
      (one folder per capture date, as in `OWN_CAPTURE_RUNBOOK.md`).
- [ ] Remove the slate photos from the room folders. Keep them in `gt/slates/` instead.

## Final checklist

- [ ] At least 3 rooms + the hallway: photo folders (2–8 each) and both video takes.
- [ ] One room shot twice (photos); the whole walk recorded twice (video).
- [ ] Staged damage (water stain + crack) in a furnished room, measured.
- [ ] Every wall, ceiling height, door, window and damage region measured.
- [ ] Consumer app: 2 rooms, export or screenshots, version noted.
