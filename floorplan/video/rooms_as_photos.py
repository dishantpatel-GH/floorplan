"""Video as photos: the turning frames of each room of a walkthrough video, written as phone photos for the photo tier.

Why (own home, 5 Oct): the photo tier measured my bedroom within 4% of its area (D-077), the video tier failed on the
same house. Its camera path broke: DPVO's scale dropped about 13x at a blank pillar (t 34.5 s) and tracking collapsed
while the ceiling was filmed at the end (outputs/own_house/diag/video/workflow_result.json). Single video frames
measure well (MoGe-2, bedroom window 1.487 m against 1.473 m). The photo tier never chains a camera path: every photo
is measured on its own, and photos are joined by matches and by the turning (spin) prior. So: cut the video into
rooms, take a turning series of sharp frames in each room and hand them to the photo tier as photos.

What a frame must carry to be read like a photo (floorplan/photo/images.py, protocol.py, layout.py):
  * EXIF FocalLengthIn35mmFilm. The photo tier converts it with the diagonal rule f_px = f35 * diag_px / 43.27
    (D-048). The tag is an integer, so the frame is centre-cropped by a few pixels until an integer f35 gives the
    video's own focal length (1280 x 720, f 945.5 px -> 1274 x 716 with f35 = 28 mm, f 945.7 px). Nothing is scaled.
  * EXIF DateTimeOriginal + SubSecTimeOriginal = the frame's time in the video. It orders the turning series and
    puts the ceiling photo last (D-074).
  * Per folder, frames from ONE spot that turn in one direction: the layout of a room is fitted with all its turning
    photos at one centre, at the Manhattan-snapped spin steps (layout.room_layouts), and a room whose placed turning
    photos stand more than 1 m apart loses its layout. A frame from another spot (walking through a door) does not
    belong in a folder.
  * No doorway pairs: the photo tier reads two photos of different folders taken seconds apart as "same spot, turned
    round" (relative yaw 180 deg). Video frames at a room change look the same way, 1-2 m apart: the opposite. Rooms
    are joined by feature matches between their frames instead (run the photo tier with doorway_pair_max_dt_s=0).

The camera orientation (yaw, pitch per frame) comes from a video-tier run (its DPVO rotations, levelled): rotations
stay usable where the path's scale broke. On take1, three of four runs agree within a few degrees over the whole walk;
the fourth (fix loop, after r1) is about 100 deg off from t 78 s, so check the run before using it.

Oracle test on take1 (5 Oct; rooms cut at known times; outputs/video_as_photos/; tape GT; 14 walls):
  * All turning frames (v1): 4 rooms, 0 of 14 walls within 3%, 2 within 10%. Hall 11.28 m2 (tape 12.71), kitchen
    4.69 (4.21), bedroom 2.83 (11.09): one frame looking out through the bedroom door put that side 6.2 m away, and
    the neighbouring rooms then cut the box down to a 0.36 m strip.
  * Without the frames that look out of their room (v2, picked by eye): 4 rooms, 29.8 m2, again 0 and 2 of 14. Hall
    12.81 m2 (+1%, walls -9% and +9%). Kitchen 4.93 m2 (+17%): the camera stood in its doorway, so the side behind
    it is a mirrored guess. Bedroom 5.70 m2 (-49%): the wardrobe front, 0.55 m from the camera, is taken as the wall.
  * Why frames fall short of the stills: a 16:9 frame (42 deg tall) tilted 27 deg down sees a 15 cm strip of the
    1.0-2.0 m wall band on a wall 2 m away, a 4:3 still (54 deg tall) 36 cm. Bedroom: 4.6k wall points per frame
    against 11.5k per still. The camera also moved up to 0.46 m while turning. With pose-graph positions instead of
    one centre (CPU refit) the bedroom box is 7.30 m2 and the hall's long side 4.48 m (tape 4.469 m).
  Verdict: not a replacement for the video tier's path. It would need automatic room cuts, depth-aware frame choice,
  per-frame positions in the box fit and a doorway box for rooms filmed from their door.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DIAG_35MM = 43.27            # full-frame diagonal (mm): the definition of the 35 mm equivalent focal length
_EXIF_IFD = 0x8769
_TAG_F35, _TAG_DTO, _TAG_SUBSEC, _TAG_MAKE, _TAG_MODEL = 0xA405, 0x9003, 0x9291, 0x010F, 0x0110


@dataclass
class RoomSpec:
    """One room of the walk: the time window where the camera turns on one spot, and optionally where it looks up."""
    name: str                                   # folder name (room label in the plan)
    turn: tuple[float, float]                   # seconds: the turning window (camera on one spot)
    ceiling: tuple[float, float] | None = None  # seconds: a window where the camera looks up at the ceiling
    step_deg: float = 55.0                      # yaw step between turning frames (protocol: about 60 deg)
    max_frames: int = 7                         # turning frames (protocol: at most 8 photos per folder)
    exclude: list = field(default_factory=list)  # [t0, t1] windows never used (e.g. frames looking out of the room)
    pitch_pref_deg: float | None = None         # prefer frames tilted down less than this (None: no preference)


@dataclass
class Picked:
    room: str
    file: str
    frame: int
    t_s: float
    yaw_deg: float
    pitch_deg: float
    sharpness: float
    role: str                                   # "turn" or "ceiling"
    pos_xz: list = field(default_factory=list)  # camera position from the video run (top view), for the record only


def orientation_from_run(run_dir: Path) -> dict:
    """Per-frame yaw (deg, unwrapped, + = turning right seen from above), pitch (deg, + = up) and top-view position
    from a video-tier run (scene/scene.npz: T_align, T_wc, timestamps)."""
    z = np.load(Path(run_dir) / "scene" / "scene.npz", allow_pickle=True)
    T = np.einsum("ij,njk->nik", z["T_align"], z["T_wc"])
    fwd = T[:, :3, 2]                                   # optical axis in the aligned frame (+y up)
    pitch = np.degrees(np.arcsin(np.clip(fwd[:, 1], -1.0, 1.0)))
    # right-handed, y up: atan2(z, x) grows when the camera turns clockwise seen from above (to the right)
    yaw = np.degrees(np.unwrap(np.arctan2(fwd[:, 2], fwd[:, 0])))
    return dict(t=np.asarray(z["timestamps"], float), yaw=yaw, pitch=pitch, pos=T[:, :3, 3][:, [0, 2]])


def crop_for_integer_f35(w: int, h: int, f_px: float, max_crop_frac: float = 0.02) -> tuple[int, int, int, float]:
    """(f35, crop_w, crop_h, f_px_after): the smallest centre crop (aspect kept, even pixel counts per side) for which
    an integer FocalLengthIn35mmFilm gives f_px under the diagonal rule. The crop changes no pixel scale."""
    best = None
    full = math.hypot(w, h)
    f35 = max(1, math.ceil(f_px * DIAG_35MM / full - 1e-9))   # a larger f35 needs a smaller diagonal (more crop)
    while True:
        diag = f_px * DIAG_35MM / f35                       # image diagonal (px) that this f35 needs
        if 1 - diag / full > max_crop_frac:
            break
        for cw in {int(math.floor(diag * w / full / 2)) * 2, int(math.ceil(diag * w / full / 2)) * 2}:
            for ch in {int(math.floor(diag * h / full / 2)) * 2, int(math.ceil(diag * h / full / 2)) * 2}:
                if cw > w or ch > h or cw <= 0 or ch <= 0:
                    continue
                f_after = f35 * math.hypot(cw, ch) / DIAG_35MM
                cand = (abs(f_after - f_px), f35, cw, ch, f_after)
                if best is None or cand < best:
                    best = cand
        f35 += 1
    if best is None:
        raise ValueError(f"no centre crop within {max_crop_frac:.0%} gives an integer f35 for f={f_px:.1f} px")
    _, f35, cw, ch, f_after = best
    return f35, cw, ch, f_after


def write_photo(rgb: np.ndarray, path: Path, f35: int, t_epoch: float, crop_wh: tuple[int, int] | None = None,
                make: str = "video-frame", model: str = "") -> None:
    """JPEG with the EXIF the photo tier reads: FocalLengthIn35mmFilm, DateTimeOriginal + SubSecTimeOriginal (UTC;
    only differences matter). rgb is upright; crop_wh is the centre crop from crop_for_integer_f35."""
    from PIL import Image
    if crop_wh is not None:
        h, w = rgb.shape[:2]
        cw, ch = crop_wh
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        rgb = rgb[y0:y0 + ch, x0:x0 + cw]
    im = Image.fromarray(np.ascontiguousarray(rgb))
    ex = Image.Exif()
    ex[_TAG_MAKE] = make
    if model:
        ex[_TAG_MODEL] = model
    sub = ex.get_ifd(_EXIF_IFD)
    dt = _dt.datetime.fromtimestamp(t_epoch, tz=_dt.timezone.utc)
    sub[_TAG_F35] = int(f35)
    sub[_TAG_DTO] = dt.strftime("%Y:%m:%d %H:%M:%S")
    sub[_TAG_SUBSEC] = f"{dt.microsecond // 1000:03d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=95, exif=ex)


def _sharp_in(t: np.ndarray, sharp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sharpness relative to the local median (1 s window), so a dim stretch is not ruled out by a bright one."""
    rel = np.zeros_like(sharp)
    for i in np.flatnonzero(mask):
        w = np.abs(t - t[i]) <= 0.5
        rel[i] = sharp[i] / max(float(np.median(sharp[w])), 1e-9)
    return rel


def select_turning(t, yaw, pitch, sharp, spec: RoomSpec, pitch_lo: float = -40.0, pitch_hi: float = 8.0,
                   tol_deg: float = 10.0, min_sep_frac: float = 0.6) -> list[int]:
    """Turning frames inside spec.turn, about spec.step_deg apart in yaw, in time order, all in the window's main
    turning direction. Each target yaw takes the sharpest frame within tol_deg (pitch inside [pitch_lo, pitch_hi]:
    steep frames see no wall band, frames tilted up would be read as ceiling photos).

    Only the turning 'frontier' counts: when the camera turns back, frames it already passed are skipped, so the
    series keeps one direction (the spin prior assumes it)."""
    a, b = spec.turn
    win = (t >= a) & (t <= b)
    idx = np.flatnonzero(win)
    if len(idx) < 2:
        return []
    direction = 1.0 if yaw[idx[-1]] - yaw[idx[0]] >= 0 else -1.0
    prog = direction * (yaw - yaw[idx[0]])
    frontier = np.maximum.accumulate(np.where(win, prog, -np.inf))
    ok = win & (prog >= frontier - tol_deg) & (pitch >= pitch_lo) & (pitch <= pitch_hi)
    for x0, x1 in spec.exclude:
        ok &= ~((t >= x0) & (t <= x1))
    rel = _sharp_in(t, sharp, ok)
    span = float(prog[idx].max())
    n_max = int(min(spec.max_frames, np.floor((min(span, 360.0 - spec.step_deg / 2)) / spec.step_deg) + 1))
    picked, last_t = [], -np.inf
    for k in range(n_max):
        target = k * spec.step_deg
        cand = np.flatnonzero(ok & (np.abs(prog - target) <= tol_deg) & (t > last_t))
        if len(cand) == 0:
            continue
        score = rel[cand] * (1.0 - 0.5 * ((prog[cand] - target) / tol_deg) ** 2)   # sharp AND near the target
        if spec.pitch_pref_deg is not None:   # a 16:9 frame tilted down 30 deg sees walls > 2 m away only below 1 m
            score = score * np.exp(-(np.maximum(0.0, spec.pitch_pref_deg - pitch[cand]) / 12.0) ** 2)
        i = int(cand[np.argmax(score)])
        if picked and abs(prog[i] - prog[picked[-1]]) < min_sep_frac * spec.step_deg:
            continue
        picked.append(i)
        last_t = t[i]
    return picked


def select_ceiling(t, pitch, sharp, window: tuple[float, float], pitch_lo: float = 25.0,
                   pitch_hi: float = 50.0) -> int | None:
    """The sharpest frame tilted up pitch_lo..pitch_hi deg (protocol: about 40 deg, wall-ceiling line in view)."""
    a, b = window
    ok = (t >= a) & (t <= b) & (pitch >= pitch_lo) & (pitch <= pitch_hi)
    if not ok.any():
        return None
    rel = _sharp_in(t, sharp, ok)
    return int(np.flatnonzero(ok)[np.argmax(rel[ok])])


def read_frames(video: Path, frames: list[int]) -> dict[int, np.ndarray]:
    """Upright RGB frames by index (sequential decode, same order as floorplan.video.frames.scan_video)."""
    import cv2
    want, out = set(frames), {}
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    i = 0
    while want - set(out):
        ok, bgr = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out


def export_rooms(video: Path, rooms: list[RoomSpec], out_dir: Path, f_px: float, orient: dict,
                 t0_epoch: float | None = None, scan_width: int = 320, log=print) -> dict:
    """Write out_dir/<room>/<frame>.jpg for every room and out_dir/../<out_dir.name>_manifest.json (what was picked
    and why). orient: orientation_from_run(); its frame order must be the video's decode order."""
    from floorplan.video.frames import scan_video
    from floorplan.video.metadata import probe
    video, out_dir = Path(video), Path(out_dir)
    scan = scan_video(video, scan_width)
    t, sharp = np.asarray(scan.timestamps, float), np.asarray(scan.sharpness, float)
    if len(t) != len(orient["t"]):
        raise ValueError(f"orientation has {len(orient['t'])} frames, the video {len(t)}")
    w, h = scan.size
    f35, cw, ch, f_after = crop_for_integer_f35(w, h, f_px)
    if t0_epoch is None:
        tag = (probe(video).get("format", {}).get("tags", {}) or {}).get("creation_time")
        t0_epoch = (_dt.datetime.fromisoformat(tag.replace("Z", "+00:00")).timestamp() if tag else 0.0)
    picks: list[Picked] = []
    for spec in rooms:
        idx = select_turning(t, orient["yaw"], orient["pitch"], sharp, spec)
        roles = ["turn"] * len(idx)
        if spec.ceiling:
            c = select_ceiling(t, orient["pitch"], sharp, spec.ceiling)
            if c is not None:
                idx, roles = idx + [c], roles + ["ceiling"]
        for i, role in zip(idx, roles):
            picks.append(Picked(spec.name, f"{spec.name}/f{i:05d}_t{t[i]:06.2f}.jpg", int(i), float(t[i]),
                                float(orient["yaw"][i]), float(orient["pitch"][i]), float(sharp[i]), role,
                                [round(float(x), 3) for x in orient["pos"][i]]))
        log(f"[rooms] {spec.name}: {len(idx)} frames at t " + ", ".join(f"{t[i]:.1f}" for i in idx))
    rgb = read_frames(video, [p.frame for p in picks])
    for p in picks:
        write_photo(rgb[p.frame], out_dir / p.file, f35, t0_epoch + p.t_s, (cw, ch), model=video.name)
    manifest = dict(video=str(video), size=[w, h], f_px=f_px, f35_mm=f35, crop=[cw, ch], f_px_after_crop=f_after,
                    t0_epoch=t0_epoch, rooms=[asdict(r) for r in rooms], frames=[asdict(p) for p in picks])
    (out_dir.parent / f"{out_dir.name}_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


PHOTO_PARAMS = {"spin_direction": "auto",    # a room's turning direction is whatever the camera did (kitchen: left)
                "doorway_pair_max_dt_s": 0}  # frames at a room change are not a turn-round pair (module docstring)


def load_spec(spec_path: Path) -> tuple[list[RoomSpec], dict]:
    """{"orient_run": <video-tier run dir>, "f_px": <optional>, "rooms": [{"name", "turn": [t0, t1], ...}]}, or a
    bare list of rooms. Returns (rooms, the rest)."""
    spec = json.loads(Path(spec_path).read_text())
    rest, rooms = ({}, spec) if isinstance(spec, list) else ({k: v for k, v in spec.items() if k != "rooms"},
                                                              spec["rooms"])
    out = []
    for r in rooms:
        r = dict(r)
        if r.pop("skip", False):
            continue
        out.append(RoomSpec(**{k: (tuple(v) if k in ("turn", "ceiling") and v else v) for k, v in r.items()}))
    return out, rest


def prepare_photo_input(video: Path, spec_path: Path, out_dir: Path, log=print) -> tuple[Path, dict]:
    """scripts/run_capture.py --video-rooms: write the photo folders and return (folder, photo-tier overrides).
    The focal length is the video tier's calibrated one (orient run's work/dpvo_meta.json) unless the spec gives
    f_px."""
    rooms, rest = load_spec(spec_path)
    run = Path(rest["orient_run"])
    f_px = rest.get("f_px") or json.loads((run / "work" / "dpvo_meta.json").read_text())["f"]
    export_rooms(Path(video), rooms, Path(out_dir), float(f_px), orientation_from_run(run), log=log)
    return Path(out_dir), dict(PHOTO_PARAMS)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="write each room's turning frames of a video as photos (EXIF focal "
                                             "and capture time) for the photo tier")
    ap.add_argument("video", type=Path)
    ap.add_argument("--rooms", type=Path, required=True,
                    help='JSON list of {"name", "turn": [t0, t1], "ceiling": [t0, t1] (optional)}')
    ap.add_argument("--orient-run", type=Path, required=True, help="video-tier run dir (scene/scene.npz)")
    ap.add_argument("--f-px", type=float, required=True, help="focal length in pixels at the video's size")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rooms, _ = load_spec(a.rooms)
    export_rooms(a.video, rooms, a.out, a.f_px, orientation_from_run(a.orient_run))


if __name__ == "__main__":
    main()
