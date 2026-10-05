# Capture protocol (Route 2: stock apps only; one page)

*Why it looks like this: docs/CAPTURE_PRACTICES_RESEARCH.md (published practice) and D-051 (simulation tests).*

**You need:**
- the phone;
- for the LiDAR tier only, an iPhone Pro / Pro Max (12 Pro or newer) with the free app **Stray Scanner** (App Store);
- all lights on and all doors fully open;
- nothing else: no paper, marker or other reference object (D-067).

**Set up once (2 min):**
1. iPhone: Settings → Camera → Formats → **Most Compatible**.
2. Camera app:
   - **Lens:** the **1×** main lens for everything (photos, video, LiDAR). Never zoom, never 0.5×.
   - **Photo** mode, not Portrait or Pano. Flash off. Keep the **default 4:3** photo format (not 16:9, not square):
     it sees more floor and ceiling.
   - **Video** mode at 1080p or 4K, 30 fps. Action Mode and Cinematic off.

---

### Tier 1: Photos (at most 8 per room, about 1 minute per room)

Phone **horizontal** (landscape, default 4:3), chest height, **1× main lens**. Hold still for each shot. (The 0.5× ultra-wide was tried on 4 Oct: better coverage, but it sees through wide openings and the photo tier mistook the next room's walls for this room's, footprint −49% vs −8% with 1× on the simulated 1 BHK. Not used until the pipeline is adapted; D-068.)

Every room is photographed **looking in** from each of its doors and **looking out** through them, so that
neighbouring rooms' folders share views and the rooms can be joined (v2.1, 4 Oct 14:45).

1. **Normal rooms: turn on the spot, then one ceiling photo.**
   - Stand in the **open middle** of the room, **as far from every wall as you can** (ideally 1.5 m or more; not next
     to a wall or a big piece of furniture). Tilt the phone slightly down so the floor line is visible.
   - **Start facing the door you came in through** (the first photo looks out through it into the previous room).
   - Turn **clockwise (to your right)**. **Bigger rooms get more turning photos**: about 4 in a room under 10 m²
     (e.g. 3 × 3 m), 5 up to 18 m², 6 above. Each should overlap the previous by about a quarter.
   - The room's folder must stay within **8 photos**: turning photos + 1 ceiling photo + 1 doorway photo per door. In
     a room with many doors (a hall or living room with 4 doors) that leaves 3 turning photos. Its doorway photos look
     in from every side and complete the coverage.
   - Then **one ceiling photo**: tilt the phone up (about 40°) and aim along the room's long side, so the ceiling and
     where it meets the walls are visible. **The ceiling photo ends the turning photos.**
2. **Small rooms (bathroom, small kitchen, balcony, narrow passage): 5 photos (6 for a balcony).**
   - From the **doorway**, looking in, turned a little to the **left** side of the room, door frame in the picture.
     (This is the doorway pair's photo into the room, below.)
   - From the **far end**, looking back at the door: **two photos**, one turned a little left of the door, then one
     turned a little right of it. The door is in both; you can see the next room through it.
   - One **ceiling photo**, as above.
   - On your way out, from the **doorway** again: one photo into the room turned a little to the **right** side.
   - **Balcony** (or any room under about 2 m deep): finally, step back about 2 m into the room next to it and take
     one photo of the balcony through the door. It goes in the balcony's folder.
2b. **A room too full to stand in the middle** (a bed or table leaves less than about 1 m to the walls): photograph it like a small room (doorway left/right, far end left/right, ceiling).
2c. **Before leaving a room, check every wall appears in at least one of its photos.** If one is missing, take one more photo of it from where you can see most of it (stay within 8 photos).
3. **Doorway pair at every door between two rooms**, once per door, including doors you do not walk through (when
   you are in one of the two rooms, step onto that threshold and take the pair).
   - Stand **on the threshold**.
   - Take one photo **back into the room you are leaving**, aimed at its open middle, not at a nearby wall.
   - Then turn round and take one photo **into the next room**, aimed at its open middle, within about 5 s.
   - Each photo goes in the folder of the room it looks into.
   - **Keep the door leaf out of the picture.** If the open door would fill the frame, step 1-1.5 m back into the room
     you are leaving and aim straight through the doorway, so the next room's furniture fills the opening. A photo of
     a door leaf or a bare wall matches nothing in the next room, and the two rooms cannot be joined (k65 simulator,
     5 Oct: the bedroom and bathroom stayed unjoined for exactly this reason).
4. **No reference object:** nothing is placed in the rooms and nothing extra is photographed (changed 4 Oct, D-067).
5. **Keep the photos' EXIF data:** no messaging apps. Capture times pair the doorway shots; the focal length sets the
   scale.

### Tier 2: Video (one clip, two passes, about 4 minutes per pass for a 2-bedroom flat)

1. **Settings:** 1080p at 30 or 60 fps, landscape, 1× lens. Automatic lens switching (Macro Control), Action Mode and
   Cinematic all off.
2. Record **once**. Walk **forward**, slowly, never sideways or backwards, never turning on the spot.
3. **Room by room: floor line, then ceiling line, before moving on** (v2.2).
   - **Normal rooms:** walk once round the room about 1 m in from the walls, phone tilted **down** (about 25°) and
     turned a little **into the room** (about 30°), so the walls across the room pass through the picture; glance
     into the corners. Then walk the same loop again with the phone tilted **up** (about 25°) at the ceiling line.
   - **Small rooms (bathroom, kitchen, balcony):** walk **in** to the far end, look across the room left and right at
     the floor line, then left and right at the ceiling line, and walk back out.
   - **Doorways:** in every doorway, **pause 2 s** facing into the next room, with both door frames in view.
4. (The whole house is covered in one pass: each room's ceiling is filmed right after its floor.)
5. **Finish where you started**, then film the first view again for about 3 s.

### Tier 3: LiDAR (iPhone Pro + Stray Scanner)

1. Open Stray Scanner and tap record. Walk **the video's route, both passes**: pass 1 tilted down at the floor line,
   pass 2 tilted up at the ceiling line. Without the ceiling in view, ceiling height cannot be measured.
2. Stay within about 3 m of the walls.
3. Stop recording. Export the recording folder (Stray Scanner → recording → share → Files/AirDrop).

### Avoid (all tiers)

- Fast turns.
- Pointing straight at a blank wall up close.
- Mirrors filling the view.
- Opening or closing doors, or people moving, during capture.
- Dark rooms.
- Zoom, filters or editing.

### Hand the files over

1. Copy everything by USB cable or AirDrop. **Never** use WhatsApp, Telegram or email: they compress the files and
   strip the camera data.
2. Use this layout:
   - `photos/<room-name>/` (one folder per room)
   - `video/walkthrough.mov`
   - `lidar/<Stray Scanner folder>`
3. Run one command per capture: `python scripts/run_capture.py <folder> --tier photo|video|lidar`

No reference object is needed: scale comes from the photos and video themselves (learned metric depth, the lens data
stored in each file and, for photos, a camera-height prior). The photo and video intervals include this scale
uncertainty (D-067). A door-height cue was built but stays off (D-069).
