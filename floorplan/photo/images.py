"""Load native phone photos: upright orientation, HEIC, EXIF focal length -> pinhole intrinsics prior.

Why this matters:
  * Phones store pixels in sensor order and record the display rotation in EXIF Orientation. Ignoring it hands the
    models sideways images: learned depth degrades and "up" is wrong. PIL's exif_transpose applies it.
  * The focal length is the single most important intrinsic for metric depth (MoGe-2 takes the field of view as
    input) and seeds bundle adjustment. Phones write FocalLengthIn35mmFilm. We convert it with the long image side
    (f_px = f35 / 36 mm * long_side), the convention for a 4:3 phone frame filling the 36 mm width. When it is absent
    we fall back to GeoCalib (learned single-image calibration), then to a 70 deg field-of-view default; the source
    is recorded so the interval can widen.
  * Some Android phones write FocalLengthIn35mmFilm = 0 but keep the physical FocalLength (my OnePlus
    Nord: 4.745 mm, f35 0). For phones whose main sensor we know, f35 = FocalLength * 43.27 mm / sensor diagonal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
_EXIF_F35, _EXIF_FOCAL, _EXIF_MAKE, _EXIF_MODEL = 0xA405, 0x920A, 0x010F, 0x0110
_EXIF_DTO, _EXIF_SUBSEC = 0x9003, 0x9291            # DateTimeOriginal, SubSecTimeOriginal (Exif sub-IFD)
# main-camera sensor diagonal (mm) for phones that leave FocalLengthIn35mmFilm at 0 (maker spec sheets)
SENSOR_DIAG_MM = {
    ("oneplus", "ac2001"): 8.0,   # OnePlus Nord: Sony IMX586, 1/2.0", 8000x6000 at 0.8 um = 6.4 x 4.8 mm
    ("oneplus", "ac2003"): 8.0,   # OnePlus Nord (EU/IN variant), same module
}


@dataclass
class Photo:
    room: str                     # folder name = room label
    path: Path                    # original file
    name: str                     # unique working name "<room>/<file>.jpg"
    rgb: np.ndarray               # upright RGB at full resolution (uint8)
    f35: float | None             # EXIF 35 mm-equivalent focal length (mm) if present
    make: str
    time_s: float | None = None   # EXIF capture time (DateTimeOriginal + SubSec), seconds since epoch; None if absent
    f35_src: str = "exif_f35"     # "exif_f35", or "exif_focal_sensor" (physical focal + known sensor size)


def _register_heif() -> None:
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:          # HEIC will then fail to open with a clear PIL error
        pass


def _exif_value(exif, tag):
    v = exif.get(tag)
    if v is None:
        try:
            v = exif.get_ifd(0x8769).get(tag)   # Exif sub-IFD (where real phones put focal lengths)
        except Exception:
            v = None
    return v


def load_photo(path: Path, room: str) -> Photo:
    im = Image.open(path)
    exif = im.getexif()
    f35 = _exif_value(exif, _EXIF_F35)
    f35 = float(f35) if f35 not in (None, 0) else None
    maker, model = str(_exif_value(exif, _EXIF_MAKE) or "").strip(), str(_exif_value(exif, _EXIF_MODEL) or "").strip()
    src = "exif_f35"
    if f35 is None:
        focal, diag = _exif_value(exif, _EXIF_FOCAL), SENSOR_DIAG_MM.get((maker.lower(), model.lower()))
        if focal and diag:
            f35, src = float(focal) * 43.27 / diag, "exif_focal_sensor"
    rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return Photo(room=room, path=path, name=f"{room}/{path.stem}.jpg", rgb=rgb, f35=f35, make=f"{maker} {model}".strip(),
                 time_s=capture_time(exif), f35_src=src)


def capture_time(exif) -> float | None:
    """EXIF DateTimeOriginal (+ SubSecTimeOriginal) as seconds. Why: the doorway-pair rule (D-017) pairs two photos
    of different rooms taken seconds apart. Only the ORIGINAL capture time is used (DateTime is the edit time).
    The time zone is irrelevant: only differences between photos of one capture session are used."""
    import datetime as _dt
    v = _exif_value(exif, _EXIF_DTO)
    if not v:
        return None
    try:
        t = _dt.datetime.strptime(str(v).strip("\x00 ")[:19], "%Y:%m:%d %H:%M:%S").replace(tzinfo=_dt.timezone.utc)
    except ValueError:
        return None
    sub = str(_exif_value(exif, _EXIF_SUBSEC) or "").strip("\x00 ")
    frac = float("0." + sub) if sub.isdigit() else 0.0
    return t.timestamp() + frac


def list_rooms(photo_root: Path) -> dict[str, list[Path]]:
    """Room name -> sorted image paths. Each sub-folder of photo_root is one room (capture protocol)."""
    _register_heif()
    rooms = {}
    for d in sorted(p for p in Path(photo_root).iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if files:
            rooms[d.name] = files
    return rooms


def load_all(photo_root: Path) -> list[Photo]:
    return [load_photo(f, room) for room, files in list_rooms(photo_root).items() for f in files]


def focal_px(photo: Photo, long_side: int, default_hfov_deg: float, f35_rule: str = "36mm") -> tuple[float, str]:
    """Focal length in pixels at an image whose long side is `long_side`, and where it came from.
    f35_rule: "36mm" = f35 / 36 mm * long side (v1); "diagonal" = f35 * image diagonal / 43.27 mm, the standard
    definition of the 35 mm equivalent (issue I-009)."""
    if photo.f35:
        if f35_rule == "diagonal":
            h, w = photo.rgb.shape[:2]
            diag = long_side * (h * h + w * w) ** 0.5 / max(h, w)
            return photo.f35 * diag / 43.27, photo.f35_src
        return photo.f35 / 36.0 * long_side, photo.f35_src
    return long_side / (2 * np.tan(np.radians(default_hfov_deg) / 2)), "default_hfov"


def resize_long(rgb: np.ndarray, long_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = long_side / max(h, w)
    if abs(s - 1) < 1e-6:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize((round(w * s), round(h * s)), Image.LANCZOS))
