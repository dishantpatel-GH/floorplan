#!/usr/bin/env python
"""Pre-flight check of an own-home capture folder, before scripts/process_own_capture.py is run on it.

    python scripts/check_own_capture.py <root> [--photos DIR] [--report FILE] [--ceiling] [--runs OUT]

Prints a PASS / WARN / FAIL report and writes <root>/capture_check.md (or --report). It reads only file headers
(EXIF, ffprobe), so it takes seconds, needs no GPU and never changes the capture. It never stops on a missing part:
the missing part becomes a row of the report.

  FAIL  the pipeline would crash, or silently measure wrong, or a tier cannot be scored. Fix before running.
  WARN  it runs, but accuracy drops or a deliverable (repeatability, head-to-head) is missing. Decide, then run.
  PASS  as the capture guide asks (docs/HOUSE_CAPTURE_GUIDE.md: layout step 0, settings 2, photos 5, video 6,
        ground truth 8).

What is checked, and against which code:
  layout   guide step 0: photos/<room>/, photos/<room>_take2/, video/take1|take2|lowlight, gt/ground_truth.csv,
           app_export/ (process_own_capture.py reads exactly these names)
  photos   per room: count 2-8, file type, size, EXIF Orientation, FocalLengthIn35mmFilm and FocalLength, digital
           zoom, capture time. The values are read the way the pipeline reads them (floorplan/photo/images.py), so
           "missing" here means "missing for the pipeline". Doorway pairs are found with the pipeline's own rule
           (floorplan/photo/protocol.py doorway_pairs).
  ceiling  optional (--ceiling): one photo per room must look up at the ceiling. Uses the ADE20K segmenter of
           envs/seg (CPU, a few seconds per photo). Rule: ceiling covers >= 30% of the photo (k65: ceiling photos
           0.39-0.86, all others <= 0.17).
  video    container, codec, size, rotation tag, fps, duration, bitrate (ffprobe), and a one-frame decode with OpenCV
           (the decoder the video tier uses, floorplan/video/frames.py)
  gt       ground_truth.csv parses the way floorplan/benchmark/gt_eval.py read_gt parses it; ids; metres; ranges
  app      app_export/app_dimensions.csv ids exist in the ground truth (scripts/eval_own_capture.py head_to_head)
  --runs   after processing: reads <OUT>/photo/scene/scene_info.json and <OUT>/video_*/scene/scene_info.json and
           reports what the pipeline decided (focal source, doorway pairs, ceiling photos, video rotation and focal)

Exit code: 0 = no FAIL, 1 = at least one FAIL.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}      # floorplan/photo/images.py IMAGE_EXT
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".3gp", ".mkv"}         # scripts/process_own_capture.py VIDEO_EXT
OTHER_IMAGE_EXT = {".dng", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".raw"}   # images the pipeline ignores
PHOTOS_MIN, PHOTOS_MAX = 2, 8                                # guide step 5
F35_ULTRAWIDE, F35_MAIN_MAX = 20.0, 32.0                     # 1x main lenses are 23-28 mm; ultra-wide 13-16 mm
CEILING_SHARE = 0.30                                         # see the module docstring
GT_COLUMNS = ("room_id", "item_type", "item_id", "value_m")
GT_RANGE = {                                                 # plausible metres per item type (default 0.3-15)
    "wall": (0.3, 15.0), "ceiling_height": (2.0, 4.0), "door_width": (0.5, 3.0), "door_height": (1.7, 2.6),
    "window_width": (0.3, 5.0), "window_height": (0.3, 3.0), "window_sill": (0.0, 2.5), "diagonal": (0.5, 20.0),
    "damage_width": (0.01, 5.0), "damage_height": (0.01, 5.0), "damage_from_floor": (0.0, 4.0),
    "damage_from_corner": (0.0, 15.0), "area": (0.5, 300.0)}
H2H_TYPES = {"wall", "ceiling_height", "door_width", "window_width", "area"}   # rows gt_eval.evaluate produces
EDITORS = ("snapseed", "photoshop", "lightroom", "gimp", "picsart", "whatsapp", "instagram", "google photos", "canva")
_TAG = dict(make=0x010F, model=0x0110, orientation=0x0112, software=0x0131, focal=0x920A, f35=0xA405, zoom=0xA404)

# what to do for every finding the checker can report (the runbook, docs/OWN_CAPTURE_RUNBOOK.md, has the same table)
ACTIONS = {
    "L-ROOT": "Give the folder that holds photos/, video/, gt/ and app_export/.",
    "L-PHOTOS": "Put one folder per room under <root>/photos/. Without it the photo tier is skipped.",
    "L-ROOMS": "The guide asks for at least 3 rooms plus the hallway. Fewer still runs; say so in the report.",
    "L-TAKE2": "No <room>_take2 folder: photo repeatability cannot be reported. Shoot one room again, or state it.",
    "L-TAKE2-PARTNER": "Rename the repeat folder to <first-take folder name>_take2 (for example 02_bedroom_take2).",
    "L-LOOSE": "Move loose files into a room folder or out of photos/. The pipeline ignores them.",
    "L-HIDDEN": "Delete hidden or trashed image files (.trashed-*, ._*): the pipeline would read them as photos.",
    "L-STEM": "Two files of one room share a name (IMG_1.jpg and IMG_1.heic): keep one, they overwrite each other.",
    "L-VIDEO": "Put the walkthrough clips in <root>/video/. Without it the video tier is skipped.",
    "L-TAKE-NAMES": "Name the clips take1.mp4, take2.mp4 (and lowlight.mp4). Only take1 is scored as 'video' and "
                    "used in the head-to-head; take2 gives repeatability.",
    "L-GT": "Write gt/ground_truth.csv (template: benchmark/ground_truth.csv in the repo). Without it nothing is scored.",
    "L-APP": "Transcribe the consumer app's dimensions into app_export/app_dimensions.csv "
             "(room_id,item_type,item_id,value_m; same ids as the ground truth). Without it there is no head-to-head.",
    "P-COUNT": "Guide: 2-8 photos per room. Remove slate photos (gt/slates/). An empty folder gives no room.",
    "P-UNREADABLE": "Copy the file again by USB or delete it. The photo tier stops on a file it cannot open.",
    "P-EXIF-NONE": "The photo went through a messenger, a screenshot or an editor. Copy the ORIGINAL by USB. Without "
                   "EXIF the tier guesses a 70 deg field of view (measured scale shift +22%) and loses the photo order.",
    "P-F35-MISSING": "No 35 mm focal length: the tier guesses a 70 deg field of view and widens the scale to 20%. If "
                     "FocalLength is present, pass the 1x lens' true value for all photos (see the runbook, 'focal').",
    "P-ULTRAWIDE": "Ultra-wide lens (0.5x/0.6x). Decision: 1x main lens only. Re-shoot the photo, or drop it.",
    "P-ZOOM": "Digital zoom or a tele lens. Re-shoot at 1x, or drop the photo: the focal length no longer matches "
              "the picture.",
    "P-LENS-MIX": "The photos do not share one lens/focal length. Find the odd ones in the table and re-shoot or drop.",
    "P-ASPECT": "Not 4:3. A 16:9 or full-screen photo is a crop of the 4:3 sensor but keeps its focal tag: the focal "
                "in pixels comes out 8% (16:9) to 12% (20:9) short. Re-shoot in 4:3, or drop the photo.",
    "P-PORTRAIT": "Portrait photo. The guide asks for landscape; it runs (EXIF Orientation is applied), but the "
                  "horizontal view is 53 deg, not 67: neighbours in a spin no longer overlap.",
    "P-MIXED-SIZE": "Photos differ in size or orientation: SfM then cannot share one camera (no focal prior). Runs, "
                    "but weaker links. Keep one format.",
    "P-RES-HIGH": "48/50 MP photos. They run (the tier resizes to 1024 px) but load slowly (150 MB of RAM each).",
    "P-RES-LOW": "Under 5 MP: a messenger copy or a screenshot. Copy the original.",
    "P-TIME-MISSING": "No capture time (EXIF DateTimeOriginal). The room loses its turning-order prior and its "
                      "doorway pairs. Copy the originals; never edit or re-save them.",
    "P-TIME-DUP": "Photos with the same capture time: their order falls back to the file name. Check that the names "
                  "are in shooting order.",
    "P-TIME-ORDER": "Capture-time order differs from file-name order. The pipeline uses the capture time; check that "
                    "the clock was not changed during the capture.",
    "P-DUP-FILE": "The same photo is in two places. Keep a doorway photo only in the folder of the room it shows.",
    "P-PAIRS": "Rooms without a doorway pair can only be joined by image matches or the 'beside the block' fallback. "
               "A pair is two photos of different rooms taken within 15 s with nothing in between (guide 5.5).",
    "P-CEILING": "No photo of this room looks up at the ceiling: its ceiling height will be 'not observed'.",
    "P-EDITED": "The Software tag names an editor. Use the original file.",
    "P-MIRROR": "Mirrored photo (front camera?). Use the back 1x camera.",
    "P-HEIC": "HEIC needs pillow_heif in the main .venv (it is installed). Nothing to do unless this row says FAIL.",
    "V-PROBE": "ffprobe could not read the file. Copy it again by USB.",
    "V-DECODE": "OpenCV could not decode a frame: the video tier cannot run. Re-encode: "
                "ffmpeg -i in.mp4 -c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p -map_metadata 0 out.mp4",
    "V-HDR": "10-bit / HDR video. Colours are tone-mapped badly by OpenCV. Re-record with HDR off, or re-encode to "
             "8-bit as under V-DECODE.",
    "V-RES": "Guide: 1080p. 4K runs but decoding is 4x slower; below 720p loses accuracy.",
    "V-PORTRAIT": "Portrait video. It runs (the rotation tag is applied) but walls leave the picture sooner.",
    "V-ROT": "Rotation tag 0 or absent: the tier then votes on the rotation with GeoCalib. After the run check "
             "info.rotation in scene/scene_info.json (python scripts/check_own_capture.py <root> --runs <out>).",
    "V-FPS": "Guide: 30 or 60 fps. Slow motion or under 24 fps breaks the visual odometry's frame spacing.",
    "V-DUR": "Very short clips have too few keyframes; very long ones take long (about 6-7 s of compute per second "
             "of video on the shared GPU).",
    "V-BITRATE": "Low bitrate: the clip was compressed by a messenger or a cloud service. Copy the original by USB.",
    "V-FOCAL": "Android writes no lens data into the video. The tier calibrates the focal itself (1-sigma 4-5%, the "
               "scale follows it 1:1). Compare its focal with the photo-based one: --runs <out>.",
    "G-HEADER": "The CSV header must contain room_id,item_type,item_id,value_m (template: benchmark/ground_truth.csv).",
    "G-VALUE": "value_m must be a plain number in metres with a decimal point (3.412). No units, no comma.",
    "G-UNITS": "The value looks like centimetres or millimetres. Write metres.",
    "G-RANGE": "Outside the plausible range for this item. Re-measure or fix the typo.",
    "G-IDS": "Wall ids must be W1, W2, ... without gaps, clockwise from the wall with the main door (guide step 8).",
    "G-DUP": "The same room_id + item_id appears twice. Keep one row.",
    "G-ROOMS": "Use the photo folder name as room_id (for example 02_bedroom). Rooms are paired by name first; other "
               "rooms are paired by their wall-length profile, which can swap two similar rooms.",
    "G-TEMPLATE": "The template's example rows are still in the file. Delete them.",
    "G-DIAG": "Walls and diagonal of this room do not close (rectangle test). Re-measure, unless the room is not a "
              "rectangle.",
    "G-INCOMPLETE": "The guide asks for every wall, one ceiling height and every door per room.",
    "A-HEADER": "app_dimensions.csv needs the header room_id,item_type,item_id,value_m.",
    "A-VALUE": "value_m must be a plain number in metres.",
    "A-IDS": "These ids are not in the ground truth, so they are left out of the head-to-head. Use the same room_id "
             "and item_id as gt/ground_truth.csv.",
    "A-KIND": "Only walls, ceiling heights, door and window widths and areas are scored (gt_eval.evaluate). Other "
              "rows are ignored in the head-to-head.",
    "A-VERSION": "Write the app name and version into app_export/app_version.txt (guide step 7).",
    "E-FFPROBE": "Install ffmpeg (ffprobe). Without it the video checks and the video tier's metadata are blind.",
    "E-DISK": "Free disk space. A whole-house video run writes 1-3 GB of work files.",
    "E-SEG": "envs/seg is missing (build it with setup/seg_env.sh): the photo tier runs without semantic wall masks "
             "(D-060) and --ceiling cannot run.",
    "X-CHECKER": "The checker itself failed on this part (a bug in scripts/check_own_capture.py). The capture may be "
                 "fine; look at the part by hand.",
    "R-RUN": "Read the named scene_info.json / log; see the runbook section 'After the run'.",
}


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str, str, str]] = []     # status, code, section, where, message
        self.tables: list[tuple[str, list[str], list[list]]] = []

    def add(self, status: str, code: str, section: str, where: str, msg: str) -> None:
        self.rows.append((status, code, section, where, " ".join(str(msg).split())))   # one line per finding

    def ok(self, section, where, msg):
        self.add("PASS", "", section, where, msg)

    def count(self, status):
        return sum(r[0] == status for r in self.rows)


# ---------------------------------------------------------------------------------------------------- photos
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_photo(path: Path) -> dict:
    """One photo as the pipeline will see it (images.py load_photo), plus the tags the pipeline does not read."""
    d = dict(path=path, file=path.name, error=None)
    try:
        from PIL import Image
        from floorplan.photo import images as I
        I._register_heif()
        im = Image.open(path)
        exif = im.getexif()
        get = lambda k: I._exif_value(exif, _TAG[k])  # noqa: E731  (IFD0, then the Exif sub-IFD: as the pipeline)
        w, h = im.size
        ori = int(_num(get("orientation")) or 1)
        f35 = get("f35")
        d.update(format=im.format, stored=(w, h), orientation=ori,
                 upright=(h, w) if ori in (5, 6, 7, 8) else (w, h),
                 f35=float(f35) if f35 not in (None, 0) and _num(f35) else None,      # images.py:57-58
                 f35_raw=f35, focal_mm=_num(get("focal")), zoom=_num(get("zoom")),
                 make=str(get("make") or "").strip("\x00 "), model=str(get("model") or "").strip("\x00 "),
                 software=str(get("software") or "").strip("\x00 "),
                 time_s=I.capture_time(exif),                                         # images.py:65-79
                 subsec=bool(str(I._exif_value(exif, I._EXIF_SUBSEC) or "").strip("\x00 ")),
                 n_tags=len(exif), bytes=path.stat().st_size)
        if im.format in ("JPEG", "MPO"):
            im.draft("RGB", (640, 640))          # decode test at reduced size (fast); a truncated file fails here
        im.load()
    except Exception as e:                       # unreadable for the checker = unreadable for the pipeline
        d["error"] = f"{type(e).__name__}: {e}"
    try:
        with open(path, "rb") as f:
            d["digest"] = hashlib.md5(f.read(1 << 18)).hexdigest() + f":{path.stat().st_size}"
    except OSError:
        d["digest"] = None
    m = re.search(r"(20\d{6})[_-]?(\d{6})", path.stem)       # Android names: IMG_20261004_101502 / IMG20261004101502
    d["name_time"] = None
    if m:
        try:
            d["name_time"] = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
                tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            pass
    return d


def focal_px_hfov(f35: float, upright: tuple[int, int]) -> tuple[float, float]:
    """Focal in pixels at full size and the field of view across the long side, by the pipeline's rule
    (images.py focal_px, f35_rule 'diagonal': f35 * image diagonal / 43.27 mm)."""
    w, h = upright
    f = f35 * math.hypot(w, h) / 43.27
    return f, math.degrees(2 * math.atan(max(w, h) / (2 * f)))


def take1_folder(take2_name: str, names: list[str]) -> str | None:
    """First-take folder of '<x>_take2' (same rule as scripts/process_own_capture.py take1_folder)."""
    try:
        from process_own_capture import take1_folder as f
        return f(take2_name, names)
    except Exception:
        base = take2_name[: -len("_take2")]
        return base if base in names else None


def _doorway_pairs(photos: list[SimpleNamespace]) -> tuple[list[dict], str]:
    """Doorway pairs by the pipeline's rule; falls back to a local copy of it if the module cannot be imported."""
    max_dt, rhythm = 15.0, 2.0
    try:
        from floorplan.photo.params import PhotoParams
        p = PhotoParams()
        max_dt, rhythm = p.doorway_pair_max_dt_s, p.doorway_pair_rhythm
        from floorplan.photo.protocol import doorway_pairs
        return doorway_pairs(photos, max_dt, rhythm), "floorplan/photo/protocol.py"
    except Exception:
        timed = sorted((p for p in photos if p.time_s is not None), key=lambda p: (p.time_s, p.name))
        gaps = sorted(y.time_s - x.time_s for x, y in zip(timed, timed[1:])
                      if x.room == y.room and y.time_s - x.time_s < 60)
        thr = max_dt if not gaps else min(max_dt, max(4.0, rhythm * gaps[len(gaps) // 2]))
        cand = sorted(((y.time_s - x.time_s, x, y) for x, y in zip(timed, timed[1:])
                       if x.room != y.room and y.time_s - x.time_s <= thr), key=lambda c: c[0])
        used, out = set(), []
        for d, x, y in cand:
            if x.name in used or y.name in used:
                continue
            used |= {x.name, y.name}
            out.append(dict(a=x.name, b=y.name, rooms=[x.room, y.room], dt_s=round(d, 2), limit_s=round(thr, 1)))
        return out, "local copy of the rule"


def check_photos(rep: Report, photos_dir: Path, ceiling: bool, tmp: Path) -> dict:
    S = "photos"
    out = dict(f35=None, upright=None, rooms=[])
    if not photos_dir.is_dir():
        rep.add("FAIL", "L-PHOTOS", "layout", str(photos_dir), "no photos/ folder: the photo tier cannot run")
        return out
    loose = [f.name for f in sorted(photos_dir.iterdir()) if f.is_file()]
    if loose:
        rep.add("WARN", "L-LOOSE", "layout", "photos/", f"{len(loose)} file(s) outside the room folders: "
                f"{', '.join(loose[:5])}{' ...' if len(loose) > 5 else ''}")
    room_dirs = sorted(p for p in photos_dir.iterdir() if p.is_dir())
    names = [d.name for d in room_dirs]
    out["rooms"] = names
    if not room_dirs:
        rep.add("FAIL", "L-PHOTOS", "layout", "photos/", "no room folders")
        return out
    main = [n for n in names if not n.endswith("_take2")]
    take2 = [n for n in names if n.endswith("_take2")]
    odd = [n for n in names if not re.match(r"^\d{2}_.+", n)]
    rep.add("WARN" if len(main) < 4 else "PASS", "L-ROOMS" if len(main) < 4 else "", "layout", "photos/",
            f"{len(main)} room folder(s) (guide: at least 3 rooms + the hallway)"
            + (f"; not named NN_<room>: {', '.join(odd)}" if odd else ""))
    if not take2:
        rep.add("WARN", "L-TAKE2", "layout", "photos/", "no <room>_take2 folder (photo repeatability)")
    for n in take2:
        partner = take1_folder(n, names)
        if partner is None:
            rep.add("FAIL", "L-TAKE2-PARTNER", "layout", f"photos/{n}", "no first-take folder found for this repeat "
                    f"take (expected photos/{n[:-len('_take2')]}/)")
        else:
            rep.ok("layout", f"photos/{n}", f"repeat take of photos/{partner}")

    all_rows, per_room, table = [], {}, []
    for d in room_dirs:
        files = sorted(d.iterdir())
        imgs = [f for f in files if f.is_file() and f.suffix.lower() in IMAGE_EXT]
        hidden = [f.name for f in imgs if f.name.startswith(".")]
        ignored = [f.name for f in files if f.is_file() and f.suffix.lower() not in IMAGE_EXT]
        subdirs = [f.name for f in files if f.is_dir()]
        where = f"photos/{d.name}"
        if hidden:
            rep.add("FAIL", "L-HIDDEN", "layout", where, f"hidden image file(s) the pipeline would read: {hidden}")
        if ignored or subdirs:
            raw = [n for n in ignored if Path(n).suffix.lower() in OTHER_IMAGE_EXT | VIDEO_EXT]
            rep.add("WARN" if raw or subdirs else "PASS", "L-LOOSE" if raw or subdirs else "", "layout", where,
                    f"ignored by the pipeline: {ignored + [s + '/' for s in subdirs]}")
        stems = Counter(f.stem for f in imgs)
        if any(c > 1 for c in stems.values()):
            rep.add("FAIL", "L-STEM", "layout", where, f"same file name, different extension: "
                    f"{[s for s, c in stems.items() if c > 1]}")
        n = len(imgs)
        if n == 0:
            rep.add("FAIL", "P-COUNT", S, where, "no photos: this room will be missing from the plan")
        elif n < PHOTOS_MIN or n > PHOTOS_MAX:
            rep.add("WARN", "P-COUNT", S, where, f"{n} photo(s) (guide: {PHOTOS_MIN}-{PHOTOS_MAX}; slate photos?)")
        else:
            rep.ok(S, where, f"{n} photos")
        rows = []
        for f in imgs:
            r = read_photo(f)
            r["room"], r["name"] = d.name, f"{d.name}/{f.stem}.jpg"          # the pipeline's working name
            rows.append(r)
        per_room[d.name] = rows
        all_rows += rows

    good = [r for r in all_rows if not r["error"]]
    for r in all_rows:
        where = f"photos/{r['room']}/{r['file']}"
        if r["error"]:
            rep.add("FAIL", "P-UNREADABLE", S, where, r["error"])
            table.append([r["room"], r["file"], "UNREADABLE", "", "", "", "", "", "", ""])
            continue
        w, h = r["upright"]
        mp, aspect = w * h / 1e6, max(w, h) / min(w, h)
        hf = focal_px_hfov(r["f35"], r["upright"])[1] if r["f35"] else None
        t = dt.datetime.fromtimestamp(r["time_s"], dt.timezone.utc).strftime("%H:%M:%S.%f")[:-4] \
            if r["time_s"] is not None else "-"
        table.append([r["room"], r["file"], r["format"], f"{w}x{h}", f"{mp:.1f}", r["orientation"],
                      "-" if r["f35"] is None else f"{r['f35']:g}",
                      "-" if r["focal_mm"] is None else f"{r['focal_mm']:.2f}", "-" if hf is None else f"{hf:.0f}", t])
        wa = re.match(r"^IMG-\d{8}-WA\d+", r["file"], re.I)
        if wa or (r["f35"] is None and r["time_s"] is None and not r["make"]):
            rep.add("FAIL", "P-EXIF-NONE", S, where, "no camera EXIF at all" + (" (WhatsApp file name)" if wa else "")
                    + ": focal length and capture time are lost")
            continue
        if r["f35"] is None and r["focal_mm"]:      # the photo tier can still use FocalLength + a known sensor
            from floorplan.photo.images import SENSOR_DIAG_MM
            diag = SENSOR_DIAG_MM.get((str(r.get("make") or "").lower(), str(r.get("model") or "").lower()))
            if diag:
                f35 = r["focal_mm"] * 43.27 / diag
                rep.add("PASS", "", S, where, f"FocalLengthIn35mmFilm is {r['f35_raw']!r}; using FocalLength "
                        f"{r['focal_mm']:.3f} mm and the {r['make']} {r.get('model', '')} sensor ({diag:g} mm diagonal): f35 {f35:.2f} mm")
                continue
        if r["f35"] is None:
            est = ""
            if r["focal_mm"]:
                est = f"; FocalLength {r['focal_mm']:.2f} mm is present" + (
                    " and is under 3 mm: this looks like the ultra-wide lens" if r["focal_mm"] < 3.0 else "")
            rep.add("FAIL", "P-F35-MISSING", S, where, f"FocalLengthIn35mmFilm is {r['f35_raw']!r}{est}")
        elif r["f35"] < F35_ULTRAWIDE:
            rep.add("FAIL", "P-ULTRAWIDE", S, where, f"35 mm focal {r['f35']:g} mm (< {F35_ULTRAWIDE:g}): field of "
                    f"view {hf:.0f} deg")
        elif r["f35"] > F35_MAIN_MAX:
            rep.add("WARN", "P-ZOOM", S, where, f"35 mm focal {r['f35']:g} mm (> {F35_MAIN_MAX:g}): tele lens or zoom")
        if r["zoom"] and r["zoom"] > 1.01:
            rep.add("FAIL", "P-ZOOM", S, where, f"DigitalZoomRatio {r['zoom']:.2f}")
        if r["time_s"] is None:
            rep.add("FAIL", "P-TIME-MISSING", S, where, "no DateTimeOriginal"
                    + ("; the file name carries a time, but the pipeline does not read it" if r["name_time"] else ""))
        if abs(aspect - 4 / 3) > 0.02:
            short = (1 - math.hypot(1, 1 / aspect) / 1.25) * 100
            rep.add("FAIL", "P-ASPECT", S, where, f"{w}x{h} is {aspect:.2f}:1, not 4:3"
                    + (f"; focal in pixels about {short:.0f}% short if this is a sensor crop" if aspect > 1.4 else ""))
        if r["orientation"] in (2, 4, 5, 7):
            rep.add("WARN", "P-MIRROR", S, where, f"EXIF Orientation {r['orientation']} (mirrored)")
        elif h > w:
            rep.add("WARN", "P-PORTRAIT", S, where, f"portrait ({w}x{h}, EXIF Orientation {r['orientation']})")
        if mp > 30:
            rep.add("WARN", "P-RES-HIGH", S, where, f"{mp:.0f} MP")
        elif mp < 5:
            rep.add("WARN", "P-RES-LOW", S, where, f"{mp:.1f} MP")
        if any(e in r["software"].lower() for e in EDITORS):
            rep.add("WARN", "P-EDITED", S, where, f"Software tag: {r['software']}")
        if r["format"] in ("HEIF", "HEIC"):
            rep.ok(S, where, "HEIC opens (pillow_heif)")
    rep.tables.append(("Photos (as the pipeline reads them)",
                       ["room", "file", "type", "upright size", "MP", "EXIF orient.", "f35 mm", "focal mm",
                        "FOV long side deg", "capture time"], table))

    # ---- capture-wide consistency: one lens, one format
    f35s = Counter(r["f35"] for r in good if r["f35"])
    sizes = Counter(r["upright"] for r in good)
    cams = Counter((r["make"], r["model"]) for r in good if r["make"] or r["model"])
    fmm = Counter(round(r["focal_mm"], 2) for r in good if r["focal_mm"])
    if len(f35s) > 1 or len(fmm) > 1:
        rep.add("FAIL", "P-LENS-MIX", S, "photos/", f"35 mm focal values {dict(f35s)}, physical focal values "
                f"{dict(fmm)}: more than one lens or zoom setting")
    elif f35s:
        f35 = next(iter(f35s))
        up = sizes.most_common(1)[0][0]
        f, hf = focal_px_hfov(f35, up)
        crop = f" (crop factor {f35 / next(iter(fmm)):.2f} from FocalLength {next(iter(fmm)):.2f} mm)" if fmm else ""
        rep.ok(S, "photos/", f"one lens: 35 mm focal {f35:g} mm{crop} -> {f:.0f} px at {up[0]}x{up[1]}, field of "
               f"view {hf:.0f} deg across the long side")
        out.update(f35=f35, upright=up)
    if len(sizes) > 1:
        rep.add("WARN", "P-MIXED-SIZE", S, "photos/", "photo sizes differ: "
                + ", ".join(f"{w}x{h} x{c}" for (w, h), c in sizes.most_common()))
    if len(cams) > 1:
        rep.add("WARN", "P-LENS-MIX", S, "photos/", f"more than one camera: {dict(cams)}")
    elif cams:
        rep.ok(S, "photos/", f"camera: {' '.join(next(iter(cams))).strip()}")
    dig = defaultdict(list)
    for r in all_rows:
        if r.get("digest"):
            dig[r["digest"]].append(f"{r['room']}/{r['file']}")
    for k, v in dig.items():
        # a repeat take is a separate photo set: only copies inside one set confuse the pairing
        sets = Counter(x.split("/")[0].endswith("_take2") for x in v)
        if max(sets.values()) > 1:
            rep.add("FAIL", "P-DUP-FILE", S, "photos/", f"identical files: {v}")

    # ---- capture times: order inside a room, doorway pairs between rooms (main set only, as process_own_capture)
    for room, rows in per_room.items():
        ok = [r for r in rows if not r["error"] and r["time_s"] is not None]
        where = f"photos/{room}"
        if not ok:
            continue
        times = Counter(r["time_s"] for r in ok)
        dup = {t: c for t, c in times.items() if c > 1}
        if dup:
            who = [r["file"] for r in ok if r["time_s"] in dup]
            rep.add("FAIL" if max(dup.values()) > 2 else "WARN", "P-TIME-DUP", S, where,
                    f"identical capture times: {who}" + ("" if any(r["subsec"] for r in ok) else
                                                         " (the phone writes no sub-second time)"))
        by_time = [r["file"] for r in sorted(ok, key=lambda r: (r["time_s"], r["name"]))]
        by_name = [r["file"] for r in sorted(ok, key=lambda r: r["name"])]
        if by_time != by_name:
            rep.add("WARN", "P-TIME-ORDER", S, where, f"time order {by_time} differs from name order")
        span = max(times) - min(times)
        if span > 900:
            rep.add("WARN", "P-TIME-ORDER", S, where, f"the room's photos span {span / 60:.0f} min")
    ns = [SimpleNamespace(room=r["room"], name=r["name"], time_s=r["time_s"]) for r in good if r["room"] in main]
    if ns and any(p.time_s is not None for p in ns):
        pairs, src = _doorway_pairs(ns)
        comp = {m: m for m in main}

        def find(x):
            while comp[x] != x:
                x = comp[x]
            return x
        for p in pairs:
            comp[find(p["rooms"][0])] = find(p["rooms"][1])
        groups = defaultdict(list)
        for m in main:
            groups[find(m)].append(m)
        big = max(groups.values(), key=len)
        alone = sorted(set(main) - set(big))
        txt = "; ".join(f"{p['rooms'][0]} ~ {p['rooms'][1]} ({p['dt_s']:.1f} s)" for p in pairs) or "none"
        lim = f", limit {pairs[0]['limit_s']} s" if pairs else ""
        if alone and len(main) > 1:
            rep.add("WARN", "P-PAIRS", S, "photos/", f"{len(pairs)} doorway pair(s) ({src}{lim}): {txt}. Rooms not "
                    f"joined to the others by a pair: {alone}")
        elif len(main) > 1:
            rep.ok(S, "photos/", f"{len(pairs)} doorway pair(s) ({src}{lim}) join all {len(main)} rooms: {txt}")
        out["pairs"] = pairs

    if ceiling:
        _check_ceiling(rep, per_room, tmp)
    else:
        rep.add("INFO", "", S, "photos/", "ceiling photo per room: not checked (add --ceiling: a few seconds per "
                "photo on the CPU; or check after the run with --runs)")
    return out


def _check_ceiling(rep: Report, per_room: dict, tmp: Path) -> None:
    """One photo per room must look up at the ceiling (guide 5.3). ADE20K 'ceiling' share per photo from envs/seg."""
    S = "photos"
    try:
        import numpy as np
        from floorplan.photo.semantic import _seg_python
        py = _seg_python()
        if py is None:
            rep.add("WARN", "E-SEG", S, "photos/", "ceiling check skipped: no envs/seg")
            return
        lst, raw = tmp / "ceiling_list.txt", tmp / "ceiling_sem.npz"
        rows = [r for rs in per_room.values() for r in rs if not r["error"]]
        lst.write_text("".join(f"{r['name']}\t{Path(r['path']).resolve()}\n" for r in rows))
        import os
        env = dict(os.environ)
        env.setdefault("HF_HOME", str(py.parents[3] / "weights" / "hf_seg"))     # as floorplan/photo/semantic.py
        r = subprocess.run([str(py), str(ROOT / "scripts" / "seg_walls.py"), "--list", str(lst), "--out", str(raw)],
                           env=env, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not raw.exists():
            rep.add("WARN", "E-SEG", S, "photos/", f"ceiling check skipped: segmenter failed ({r.stderr[-200:]})")
            return
        z = np.load(raw)
        ci = [str(x).strip() for x in z["names"]].index("ceiling")
        share = {str(k): float((z[f"m{i}"] == ci).mean()) for i, k in enumerate(z["keys"])}
        for room, rs in per_room.items():
            up = {r["file"]: share.get(r["name"], 0.0) for r in rs if not r["error"]}
            hit = [f"{f} ({s:.2f})" for f, s in up.items() if s >= CEILING_SHARE]
            if hit:
                rep.ok(S, f"photos/{room}", f"ceiling photo: {', '.join(hit)}")
            elif up:
                best = max(up, key=up.get)
                rep.add("WARN", "P-CEILING", S, f"photos/{room}", f"no ceiling photo (largest ceiling share "
                        f"{up[best]:.2f} in {best}; needs >= {CEILING_SHARE})")
    except Exception as e:
        rep.add("WARN", "X-CHECKER", S, "photos/", f"ceiling check failed: {type(e).__name__}: {e}")


# ----------------------------------------------------------------------------------------------------- video
def _frac(s) -> float | None:
    try:
        a, b = str(s).split("/")
        return float(a) / float(b) if float(b) else None
    except (ValueError, ZeroDivisionError):
        return _num(s)


def probe_video(path: Path) -> dict:
    if not shutil.which("ffprobe"):
        return dict(error="ffprobe not installed")
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                            str(path)], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return dict(error=(r.stderr.strip() or "ffprobe failed")[-200:])
        meta = json.loads(r.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return dict(error=f"{type(e).__name__}: {e}")
    v = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
    if v is None:
        return dict(error="no video stream")
    fmt = meta.get("format", {})
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if not rot and v.get("tags", {}).get("rotate"):
        rot = -int(v["tags"]["rotate"])
    tags = {k.lower(): str(x) for k, x in {**fmt.get("tags", {}), **v.get("tags", {})}.items()}
    dur = _num(v.get("duration")) or _num(fmt.get("duration"))
    return dict(error=None, container=fmt.get("format_name", ""), codec=v.get("codec_name"),
                pix_fmt=v.get("pix_fmt", ""), transfer=v.get("color_transfer", ""),
                size=(int(v.get("width", 0)), int(v.get("height", 0))), rotation=rot,
                fps=_frac(v.get("avg_frame_rate")), fps_r=_frac(v.get("r_frame_rate")), duration=dur,
                bitrate=_num(v.get("bit_rate")) or _num(fmt.get("bit_rate")), tags=tags,
                has_rot_side_data=any("rotation" in sd for sd in v.get("side_data_list", [])))


def decode_one_frame(path: Path) -> dict:
    """First frame with OpenCV, exactly as floorplan/video/frames.py opens the file."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        rot = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META))) % 360
        ok, bgr = cap.read()
        cap.release()
        return dict(ok=bool(ok), rotation_meta=rot, shape=None if not ok else bgr.shape[:2],
                    mean=None if not ok else float(bgr.mean()))
    except Exception as e:
        return dict(ok=False, error=f"{type(e).__name__}: {e}")


def check_videos(rep: Report, video_dir: Path, photo: dict) -> None:
    S = "video"
    if not video_dir.is_dir():
        rep.add("FAIL", "L-VIDEO", "layout", str(video_dir), "no video/ folder: the video tier cannot run")
        return
    clips = sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    other = [p.name for p in sorted(video_dir.iterdir()) if p not in clips and not p.name.startswith("_")]
    if not clips:
        rep.add("FAIL", "L-VIDEO", "layout", "video/", f"no video file ({sorted(VIDEO_EXT)})"
                + (f"; other files: {other}" if other else ""))
        return
    stems = [p.stem for p in clips]
    if "take1" not in stems:
        rep.add("FAIL", "L-TAKE-NAMES", "layout", "video/", f"no take1.* (found {[p.name for p in clips]}): no clip "
                f"would be scored as the 'video' tier")
    if "take2" not in stems:
        rep.add("WARN", "L-TAKE-NAMES", "layout", "video/", "no take2.* (video repeatability)")
    extra = [p.name for p in clips if p.stem not in ("take1", "take2", "lowlight")]
    if extra:
        rep.add("WARN", "L-TAKE-NAMES", "layout", "video/", f"clips with other names are run as video_<name> and "
                f"scored separately: {extra}")
    if not shutil.which("ffprobe"):
        rep.add("WARN", "E-FFPROBE", S, "video/", "ffprobe not installed")
    table = []
    for p in clips:
        where = f"video/{p.name}"
        m = probe_video(p)
        d = decode_one_frame(p)
        if m.get("error"):
            rep.add("FAIL" if not d.get("ok") else "WARN", "V-PROBE", S, where, m["error"])
        if not d.get("ok"):
            rep.add("FAIL", "V-DECODE", S, where, f"OpenCV cannot decode the first frame ({d.get('error', 'read failed')})")
        if m.get("error"):
            table.append([p.name, "?", "?", "?", "?", "?", "?", "?", "decodes" if d.get("ok") else "NO DECODE"])
            continue
        w, h = m["size"]
        up = (h, w) if abs(m["rotation"]) in (90, 270) else (w, h)
        mbps = (m["bitrate"] or 0) / 1e6
        table.append([p.name, m["container"].split(",")[0], f"{m['codec']} {m['pix_fmt']}", f"{w}x{h}",
                      m["rotation"], f"{m['fps']:.2f}" if m["fps"] else "?",
                      f"{m['duration']:.0f}" if m["duration"] else "?", f"{mbps:.1f}",
                      "decodes" if d.get("ok") else "NO DECODE"])
        if m["codec"] not in ("h264", "hevc"):
            rep.add("WARN", "V-DECODE", S, where, f"codec {m['codec']} (expected h264 or hevc)")
        if "10" in m["pix_fmt"] or m["transfer"] in ("smpte2084", "arib-std-b67"):
            rep.add("WARN", "V-HDR", S, where, f"pixel format {m['pix_fmt']}, transfer {m['transfer'] or '?'}")
        if max(up) > 2000 or max(up) < 1280:
            rep.add("WARN", "V-RES", S, where, f"{up[0]}x{up[1]} (guide: 1920x1080)")
        if up[1] > up[0]:
            rep.add("WARN", "V-PORTRAIT", S, where, f"upright size {up[0]}x{up[1]} (rotation tag {m['rotation']})")
        if d.get("ok") and (-m["rotation"]) % 360 != d["rotation_meta"]:
            rep.add("WARN", "V-ROT", S, where, f"ffprobe rotation {m['rotation']} but OpenCV reports "
                    f"{d['rotation_meta']} deg clockwise: the tier uses OpenCV's value")
        elif m["rotation"] == 0:
            rep.add("INFO", "V-ROT", S, where, "rotation tag 0 or absent: the tier votes on the rotation (GeoCalib)")
        else:
            rep.ok(S, where, f"rotation tag {m['rotation']}: used as is")
        fps = m["fps"] or 0
        if fps < 24 or fps > 65:
            rep.add("WARN", "V-FPS", S, where, f"{fps:.1f} fps (guide: 30 or 60)")
        elif m["fps_r"] and abs(m["fps_r"] - fps) > 1.0:
            rep.add("INFO", "V-FPS", S, where, f"variable frame rate (average {fps:.2f}, nominal {m['fps_r']:.0f}): "
                    "fine, the tier reads frame timestamps")
        if m["duration"] and (m["duration"] < 20 or m["duration"] > 900):
            rep.add("WARN", "V-DUR", S, where, f"{m['duration']:.0f} s long")
        px_rate = mbps * 1e6 / max(w * h * max(fps, 1), 1)             # bits per pixel per frame
        if m["bitrate"] and px_rate < 0.06:
            rep.add("WARN", "V-BITRATE", S, where, f"{mbps:.1f} Mbit/s for {w}x{h} at {fps:.0f} fps "
                    f"({px_rate:.3f} bit/pixel; phone originals are 0.15-0.35)")
        f35_keys = [k for k in m["tags"] if "35mm" in k or "focal" in k]
        phone = [f"{k}={v}" for k, v in m["tags"].items() if k.startswith(("com.android", "com.apple.quicktime.m"))]
        note = f"lens metadata: {f35_keys or 'none'}; phone tags: {phone or 'none'}"
        if not f35_keys:
            if photo.get("f35"):
                # a 16:9 video that uses the full sensor WIDTH has the photo's horizontal field of view
                f_ref = photo["f35"] * 1.25 * max(up) / 43.27
                note += (f". Reference from the photos' EXIF ({photo['f35']:g} mm): {f_ref:.0f} px at {max(up)} px "
                         f"wide if the video is not cropped; stabilisation crops raise it by 10-25%")
            rep.add("INFO", "V-FOCAL", S, where, note)
        else:
            rep.ok(S, where, note)
        if d.get("ok") and not m.get("error") and m["codec"] in ("h264", "hevc") and fps >= 24:
            rep.ok(S, where, f"{m['codec']} {w}x{h} {fps:.0f} fps, {m['duration'] or 0:.0f} s, {mbps:.1f} Mbit/s; "
                   f"first frame decodes")
    rep.tables.append(("Videos", ["file", "container", "codec", "stored size", "rotation tag", "fps", "seconds",
                                  "Mbit/s", "OpenCV"], table))


# ---------------------------------------------------------------------------------------------- ground truth
def _read_csv(path: Path) -> tuple[list[str], list[tuple[int, dict]]]:
    """Header and (line number, row dict) the way gt_eval.read_gt reads the file ('#' lines are comments)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        raw = [(i + 1, r) for i, r in enumerate(csv.reader(f))]
    rows = [(i, r) for i, r in raw if r and any(c.strip() for c in r) and not r[0].lstrip().startswith("#")]
    if not rows:
        return [], []
    head = [h.strip() for h in rows[0][1]]
    return head, [(i, dict(zip(head, [c.strip() for c in r]))) for i, r in rows[1:]]


def check_gt(rep: Report, gt_csv: Path, photo_rooms: list[str]) -> dict:
    S = "ground truth"
    if not gt_csv.is_file():
        rep.add("FAIL", "L-GT", "layout", str(gt_csv), "no ground_truth.csv: nothing can be scored")
        return {}
    try:
        raw = gt_csv.read_bytes()
        if raw[:3] == b"\xef\xbb\xbf":
            rep.add("FAIL", "G-HEADER", S, "gt/ground_truth.csv", "the file starts with a byte-order mark (saved by "
                    "Excel as 'CSV UTF-8'): the scorer then reads the first column as '\\ufeffroom_id'. Save as plain CSV")
        head, rows = _read_csv(gt_csv)
    except Exception as e:
        rep.add("FAIL", "G-HEADER", S, "gt/ground_truth.csv", f"cannot read the file: {type(e).__name__}: {e}")
        return {}
    miss = [c for c in GT_COLUMNS if c not in head]
    if miss:
        rep.add("FAIL", "G-HEADER", S, "gt/ground_truth.csv", f"header {head} lacks {miss}")
        return {}
    items, seen, empty = {}, Counter(), 0
    for line, d in rows:
        where = f"gt/ground_truth.csv:{line}"
        room, kind, item, val = d.get("room_id", ""), d.get("item_type", ""), d.get("item_id", ""), d.get("value_m", "")
        if not val:
            empty += 1
            continue
        try:
            v = float(val)
        except ValueError:
            rep.add("FAIL", "G-VALUE", S, where, f"{room} {item}: value_m {val!r} is not a number (the scorer stops)")
            continue
        if not room or not item:
            rep.add("FAIL", "G-IDS", S, where, f"empty room_id or item_id ({room!r}, {item!r})")
            continue
        seen[(room, kind, item)] += 1                 # one id may carry several types (door D1: width and height)
        items[(room, kind, item)] = v
        lo, hi = GT_RANGE.get(kind, (0.3, 15.0))
        if kind != "area" and v > 30:
            rep.add("FAIL", "G-UNITS", S, where, f"{room} {item} ({kind}) = {v:g}: not metres")
        elif not lo <= v <= hi:
            rep.add("WARN", "G-RANGE", S, where, f"{room} {item} ({kind}) = {v:g} m, outside {lo:g}-{hi:g} m")
        if kind not in GT_RANGE:
            rep.add("WARN", "G-IDS", S, where, f"unknown item_type {kind!r} (known: {sorted(GT_RANGE)})")
        if (room, item, val) in (("bedroom1", "W1", "3.412"), ("bedroom1", "W2", "2.987")):
            rep.add("WARN", "G-TEMPLATE", S, where, "template example row")
    for (room, kind, item), c in seen.items():
        if c > 1:
            rep.add("FAIL", "G-DUP", S, "gt/ground_truth.csv", f"{room} {kind} {item} appears {c} times")
    rooms = sorted({r for r, _, _ in items})
    n_by = Counter(k for _, k, _ in items)
    if not items:
        rep.add("FAIL", "G-VALUE", S, "gt/ground_truth.csv", "no measurement rows")
        return {}
    rep.ok(S, "gt/ground_truth.csv", f"{len(items)} measurements in {len(rooms)} rooms: "
           + ", ".join(f"{k} x{c}" for k, c in sorted(n_by.items())) + (f"; {empty} row(s) without a value" if empty else ""))
    for room in rooms:
        walls = {i: v for (r, k, i), v in items.items() if r == room and k == "wall"}
        nums = sorted(int(m.group(1)) for m in (re.fullmatch(r"W(\d+)", i) for i in walls) if m)
        bad = [i for i in walls if not re.fullmatch(r"W\d+", i)]
        where = f"gt: {room}"
        if bad or nums != list(range(1, len(nums) + 1)):
            rep.add("WARN", "G-IDS", S, where, f"wall ids {sorted(walls)}: expected W1..W{len(walls)} without gaps")
        kinds = {k for (r, k, _) in items if r == room}
        lack = [k for k in ("wall", "ceiling_height", "door_width") if k not in kinds]
        if len(walls) < 3 and "wall" in kinds:
            lack.append("at least 3 walls")
        if lack:
            rep.add("WARN", "G-INCOMPLETE", S, where, f"missing: {', '.join(lack)}")
        diag = [v for (r, k, _), v in items.items() if r == room and k == "diagonal"]
        if len(walls) == 4 and diag and not bad:
            a, b = (walls["W1"] + walls["W3"]) / 2, (walls["W2"] + walls["W4"]) / 2
            exp = math.hypot(a, b)
            if abs(walls["W1"] - walls["W3"]) < 0.05 and abs(walls["W2"] - walls["W4"]) < 0.05:
                if abs(diag[0] - exp) > max(0.02, 0.01 * exp):
                    rep.add("WARN", "G-DIAG", S, where, f"diagonal {diag[0]:.3f} m, but the walls give {exp:.3f} m")
                else:
                    rep.ok(S, where, f"rectangle closes: diagonal {diag[0]:.3f} m vs {exp:.3f} m from the walls")
    if photo_rooms:
        main = [n for n in photo_rooms if not n.endswith("_take2")]
        unmatched = [r for r in rooms if r not in main]
        no_gt = [n for n in main if n not in rooms]
        if unmatched or no_gt:
            rep.add("WARN", "G-ROOMS", S, "gt/ground_truth.csv", f"room ids that are not photo folder names: "
                    f"{unmatched or 'none'}; photo folders without ground truth: {no_gt or 'none'}")
        else:
            rep.ok(S, "gt/ground_truth.csv", "every room_id is a photo folder name (rooms are paired by name)")
    geo = gt_csv.with_name("sim_gt.json").is_file()
    rep.add("INFO", "", S, "gt/", "sim_gt.json present: walls paired by geometry" if geo else
            "no room polygons (tape ground truth): walls are paired by clockwise order from W1")
    if not list(gt_csv.parent.glob("sketch_*")):
        rep.add("INFO", "", S, "gt/", "no sketch_<room>.jpg photos")
    return items


def check_app(rep: Report, app_dir: Path, gt: dict) -> None:
    S = "app export"
    if not app_dir.is_dir():
        rep.add("WARN", "L-APP", "layout", str(app_dir), "no app_export/ folder: no head-to-head")
        return
    if not (app_dir / "app_version.txt").is_file():
        rep.add("WARN", "A-VERSION", S, "app_export/", "no app_version.txt")
    f = app_dir / "app_dimensions.csv"
    if not f.is_file():
        rep.add("WARN", "L-APP", "layout", "app_export/", f"no app_dimensions.csv ({len(list(app_dir.iterdir()))} "
                f"other file(s) in the folder): no head-to-head")
        return
    try:
        head, rows = _read_csv(f)
    except Exception as e:
        rep.add("FAIL", "A-HEADER", S, "app_export/app_dimensions.csv", f"cannot read: {type(e).__name__}: {e}")
        return
    miss = [c for c in ("room_id", "item_id", "value_m") if c not in head]
    if miss:
        rep.add("FAIL", "A-HEADER", S, "app_export/app_dimensions.csv", f"header {head} lacks {miss}")
        return
    hit, unknown, kinds, unscored = 0, [], Counter(), []
    for line, d in rows:
        room, kind, item = d.get("room_id", ""), d.get("item_type", ""), d.get("item_id", "")
        where = f"app_export/app_dimensions.csv:{line}"
        try:
            v = float(d.get("value_m", ""))
        except ValueError:
            rep.add("FAIL", "A-VALUE", S, where, f"{room} {item}: value_m {d.get('value_m')!r} is not a number")
            continue
        # same key as eval_own_capture.head_to_head: (room, item type, id); without a type the id must be unique
        cand = [k for k in gt if k[0] == room and k[2] == item and (not kind or k[1] == kind)]
        if len(cand) != 1:
            unknown.append(f"{room}/{kind or '?'}/{item}" + (" (ambiguous: add item_type)" if len(cand) > 1 else ""))
            continue
        hit += 1
        kinds[cand[0][1]] += 1
        if cand[0][1] not in H2H_TYPES:
            unscored.append(f"{room}/{cand[0][1]}/{item}")
        if abs(v - gt[cand[0]]) > 0.5 * gt[cand[0]]:
            rep.add("WARN", "A-VALUE", S, where, f"{room} {item}: app {v:g} m vs ground truth {gt[cand[0]]:g} m "
                    f"(more than 50% apart: wrong id or units?)")
    if unscored:
        rep.add("WARN", "A-KIND", S, "app_export/app_dimensions.csv", f"{len(unscored)} row(s) of a type our plans "
                f"are not scored on, left out of the head-to-head: {unscored[:8]}")
    if unknown:
        rep.add("FAIL" if not hit else "WARN", "A-IDS", S, "app_export/app_dimensions.csv",
                f"{len(unknown)} id(s) not in the ground truth: {unknown[:8]}")
    if hit:
        rep.ok(S, "app_export/app_dimensions.csv", f"{hit} dimension(s) match ground-truth ids ("
               + ", ".join(f"{k} x{c}" for k, c in kinds.items()) + ")")
    elif not unknown:
        rep.add("WARN", "L-APP", S, "app_export/app_dimensions.csv", "no rows")


# ------------------------------------------------------------------------------------------------- after the run
def check_runs(rep: Report, out: Path, photo: dict) -> None:
    """What the pipeline decided on this capture (read from the run outputs; nothing is recomputed)."""
    S = "after the run"
    if not out.is_dir():
        rep.add("WARN", "R-RUN", S, str(out), "no such output folder")
        return
    for d in sorted(p for p in out.iterdir() if p.is_dir() and p.name.startswith(("photo", "video_"))):
        si = d / "scene" / "scene_info.json"
        if not (d / "plan.json").exists():
            if not d.name.startswith("photo_work"):
                rep.add("FAIL", "R-RUN", S, d.name, f"no plan.json: the run failed (log: {d}.log)")
            continue
        if not si.exists():
            continue
        try:
            info = json.loads(si.read_text())
        except Exception as e:
            rep.add("WARN", "R-RUN", S, d.name, f"cannot read scene_info.json: {e}")
            continue
        if d.name.startswith("photo"):
            src = Counter((info.get("focal_source") or {}).values())
            sc = info.get("scale") or {}
            proto = info.get("protocol_v2") or {}             # keys written by floorplan/photo/frontend.py
            pairs = proto.get("doorway_pairs") or []
            ceil = {r: l for r, l in (info.get("room_layouts") or {}).items()
                    if isinstance(l, dict) and (l.get("height") or {}).get("measured_value") is not None}
            rooms = sorted({str(n).split("/")[0] for n in (info.get("focal_source") or {})})
            no_ceil = [r for r in rooms if r not in ceil]
            bad = src.get("default_hfov", 0)
            rep.add("FAIL" if bad else "PASS", "P-F35-MISSING" if bad else "", S, d.name,
                    f"focal source {dict(src)}; scale x{sc.get('factor', float('nan')):.3f}, 1-sigma "
                    f"{100 * (sc.get('sigma_rel') or info.get('scale_sigma_rel') or float('nan')):.1f}%; cues "
                    f"{[c.get('cue') for c in sc.get('cues', []) if isinstance(c, dict)]}")
            rep.add("WARN" if (len(rooms) > 1 and not pairs) else "PASS", "P-PAIRS" if (len(rooms) > 1 and not pairs)
                    else "", S, d.name, f"{len(pairs)} doorway pair(s) used; placement "
                    f"{dict(Counter(r.get('placement') for r in (info.get('rooms') or {}).values() if isinstance(r, dict)))}")
            rep.add("WARN" if no_ceil else "PASS", "P-CEILING" if no_ceil else "", S, d.name,
                    f"ceiling photos found in {sorted(ceil)}" + (f"; none in {no_ceil}" if no_ceil else ""))
        else:
            rot = info.get("rotation") or {}
            cal = info.get("calibration") or info.get("calib") or {}
            cj = d / "work" / "calib.json"
            if not cal and cj.exists():
                cal = json.loads(cj.read_text())
            flip = rot.get("flipped_by_pitch_check")
            rep.add("WARN" if flip or rot.get("source") != "container metadata" else "PASS",
                    "V-ROT" if flip or rot.get("source") != "container metadata" else "", S, d.name,
                    f"rotation {rot.get('final_rotation', rot.get('rotation'))} deg from {rot.get('source')}; "
                    f"flipped by the pitch check: {flip}; median camera pitch {rot.get('median_pitch_deg')}. "
                    "Open work/frames_r*/sfm/ and look at one frame: it must be upright")
            f, names = cal.get("f"), None
            vm = info.get("video_metadata") or {}
            size = vm.get("size") or [None, None]
            if f and photo.get("f35") and size[0]:
                long_up = max(size)
                f_full = f * long_up / 1024.0                     # calib.json f is at the 1024 px SfM size
                f_ref = photo["f35"] * 1.25 * long_up / 43.27
                ratio = f_full / f_ref
                st = "PASS" if 0.95 <= ratio <= 1.08 else "WARN"
                rep.add(st, "" if st == "PASS" else "V-FOCAL", S, d.name,
                        f"video focal {f_full:.0f} px ({cal.get('source')}, 1-sigma {100 * cal.get('sigma_rel', 0):.1f}%)"
                        f" vs {f_ref:.0f} px from the photos' EXIF: ratio {ratio:.3f} (1.00-1.08 = no or little crop; "
                        f"above = stabilisation crop or a calibration error; the video scale follows the focal 1:1)")
            elif f:
                rep.add("INFO", "V-FOCAL", S, d.name, f"video focal {f:.0f} px at the SfM size ({cal.get('source')})")
            if info.get("whole_scene_consistent") is False:
                rep.add("WARN", "R-RUN", S, d.name, "segments not joined by verified geometry: relative sigma floored "
                        "at 20% (run_capture.py D-034)")


# ------------------------------------------------------------------------------------------------------ report
def to_markdown(rep: Report, root: Path, took: float) -> str:
    n = {s: rep.count(s) for s in ("FAIL", "WARN", "PASS", "INFO")}
    verdict = "FAIL: fix the FAIL rows before running" if n["FAIL"] else (
        "WARN: runs; read the warnings first" if n["WARN"] else "PASS: ready to run")
    md = [f"# Capture check: {root}", "",
          f"Checked {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} in {took:.0f} s by scripts/check_own_capture.py.",
          "", f"**{verdict}** ({n['FAIL']} fail, {n['WARN']} warn, {n['PASS']} pass, {n['INFO']} info)", ""]
    for status, title in (("FAIL", "Fail"), ("WARN", "Warn"), ("INFO", "Info"), ("PASS", "Pass")):
        rows = [r for r in rep.rows if r[0] == status]
        if not rows:
            continue
        md += [f"## {title} ({len(rows)})", "", "| Code | Part | Where | Finding |", "|---|---|---|---|"]
        md += [f"| {c or '-'} | {s} | {w} | {m.replace('|', '/')} |" for _, c, s, w, m in rows]
        md.append("")
    codes = sorted({r[1] for r in rep.rows if r[1] and r[0] in ("FAIL", "WARN", "INFO")})
    if codes:
        md += ["## What to do", "", "| Code | Action |", "|---|---|"]
        md += [f"| {c} | {ACTIONS.get(c, '')} |" for c in codes] + [""]
    for title, head, rows in rep.tables:
        if rows:
            md += [f"## {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]
    return "\n".join(md)


def _section(rep: Report, name: str, fn, *args):
    """Run one part; a bug in the checker becomes a report row, never a traceback."""
    try:
        return fn(rep, *args)
    except Exception as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        rep.add("FAIL", "X-CHECKER", name, "-", f"{type(e).__name__}: {e} (check_own_capture.py:{tb.lineno})")
        return {}


def main() -> int:
    import time
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--photos", type=Path, help="photo folder to check instead of <root>/photos")
    ap.add_argument("--report", type=Path, help="where to write the report (default <root>/capture_check.md)")
    ap.add_argument("--ceiling", action="store_true", help="also look for each room's ceiling photo (segmenter, slow)")
    ap.add_argument("--runs", type=Path, help="output folder of process_own_capture.py: report what the runs decided")
    a = ap.parse_args()
    t0 = time.time()
    rep = Report()
    root = a.root
    if not root.is_dir():
        rep.add("FAIL", "L-ROOT", "layout", str(root), "the capture folder does not exist")
    else:
        free = shutil.disk_usage(ROOT).free / 1e9
        rep.add("WARN" if free < 5 else "PASS", "E-DISK" if free < 5 else "", "machine", str(ROOT),
                f"{free:.0f} GB free on the output disk")
        with tempfile.TemporaryDirectory(prefix="capture_check_") as tmp:
            photo = _section(rep, "photos", check_photos, a.photos or root / "photos", a.ceiling, Path(tmp)) or {}
        _section(rep, "video", check_videos, root / "video", photo)
        gt = _section(rep, "ground truth", check_gt, root / "gt" / "ground_truth.csv", photo.get("rooms", [])) or {}
        _section(rep, "app export", check_app, root / "app_export", gt)
        if a.runs:
            _section(rep, "after the run", check_runs, a.runs, photo)
    md = to_markdown(rep, root, time.time() - t0)
    order = {"FAIL": 0, "WARN": 1, "INFO": 2, "PASS": 3}
    for status, code, sec, where, msg in sorted(rep.rows, key=lambda r: order[r[0]]):
        print(f"{status:4s} {code:16s} {where}: {msg}")
    n = {s: rep.count(s) for s in order}
    print(f"\n{n['FAIL']} fail, {n['WARN']} warn, {n['PASS']} pass, {n['INFO']} info  ({time.time() - t0:.0f} s)")
    target = a.report or (root / "capture_check.md" if root.is_dir() else None)
    if target is not None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(md)
            print(f"report: {target}")
        except OSError as e:
            print(f"could not write the report to {target}: {e}")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
