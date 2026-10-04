"""Container metadata of a phone video, read with ffprobe: lens focal length (35 mm equivalent), make/model, rotation.

Why: the focal length is the largest single error source of the video tier on a short capture (video_tier.md, Issue
13e: the scale error follows the focal error about 1:1). iPhone Camera-app videos (iOS 17+) carry QuickTime keys
such as `com.apple.quicktime.camera.focal_length.35mm_equivalent` and `com.apple.quicktime.camera.lens_model`.
Android MP4s usually carry nothing (sometimes `com.android.manufacturer`/`model`). The Stray Scanner sample has none.

The 35 mm equivalent focal length is defined on the 43.27 mm full-frame DIAGONAL, so f_px = f35 * diag_px / 43.27
(same convention as floorplan.scale.sheet.load_image_with_intrinsics for photos). For VIDEO it is less exact than for
photos: video modes crop the sensor (16:9) and electronic stabilisation crops a few % more, and it is not documented
whether the stored value accounts for that. Hence a wide 1-sigma (FOCAL_META_SIGMA) and a cross-check against the
image-based estimate before it is trusted (frontend._focal).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

FOCAL_META_SIGMA = 0.05      # 1-sigma relative: rounding to whole mm (~2%) + unknown video/stabilisation crop (~4%)
_F35_KEYS = ("focal_length.35mm_equivalent", "focallengthin35mmformat", "focal_length_35mm", "35mm")
_LENS_FOCAL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm")


def probe(path: str | Path) -> dict:
    """All format and stream tags plus side data (display matrix) as one dict; {} if ffprobe is unavailable."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                            str(path)], capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout) if r.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}


def _all_tags(meta: dict) -> dict[str, str]:
    tags = {k.lower(): str(v) for k, v in meta.get("format", {}).get("tags", {}).items()}
    for s in meta.get("streams", []):
        tags.update({k.lower(): str(v) for k, v in s.get("tags", {}).items()})
    return tags


def video_metadata(path: str | Path) -> dict:
    """{'f35_mm': float|None, 'lens_model', 'make', 'model', 'codec', 'rotation_deg', 'keys': [...]}.

    f35_mm comes only from an explicit 35 mm-equivalent key. A lens_model string such as 'iPhone 15 back camera
    6.24mm f/1.6' gives the PHYSICAL focal length, which needs the sensor size; it is reported but not converted."""
    meta = probe(path)
    tags = _all_tags(meta)
    f35 = None
    for k, v in tags.items():
        if any(key in k for key in _F35_KEYS):
            try:
                f35 = float(re.findall(r"\d+(?:\.\d+)?", v)[0])
                break
            except (IndexError, ValueError):
                pass
    video = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), {})
    rot = 0
    for sd in video.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    lens = next((v for k, v in tags.items() if k.endswith("lens_model")), None)
    return dict(f35_mm=f35, lens_model=lens,
                lens_physical_mm=float(_LENS_FOCAL_RE.search(lens).group(1)) if lens and _LENS_FOCAL_RE.search(lens)
                else None,
                make=next((v for k, v in tags.items() if k.endswith(".make") or k == "make"
                           or k.endswith("manufacturer")), None),
                model=next((v for k, v in tags.items() if k.endswith(".model") or k == "model"), None),
                codec=video.get("codec_name"), size=[video.get("width"), video.get("height")],
                rotation_deg=rot, keys=sorted(tags))


def focal_px_from_f35(f35_mm: float, width: int, height: int) -> float:
    """35 mm equivalent focal length -> pixels for a frame of (width, height) pixels (diagonal convention)."""
    return float(f35_mm) * (width ** 2 + height ** 2) ** 0.5 / 43.27
