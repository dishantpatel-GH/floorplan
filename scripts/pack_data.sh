#!/usr/bin/env bash
# Packs the benchmark data in data/ into the archives of the GitHub release "data-v1" and writes their size and
# SHA-256 into data/MANIFEST.json, which scripts/fetch_data.py checks after each download.
#
#   own_house.zip  data/own_house: one room, 7 lit + 6 dim photos, a 1x walkthrough video, the hand sketch
#   k65.zip        data/k65: the simulated k65 flat (photos per room, video, LiDAR, exact ground truth)
#   sample.zip     data/sample: the three sample captures from the case study (Stray Scanner folders)
#
# Zip, not tar.zst, so that fetch_data.py needs nothing but Python. Photos, video and depth PNGs are stored as they
# are (already compressed), text is deflated. Fixed timestamps and sorted entries: the same files give the same
# SHA-256. Symlinks are followed. Each archive must stay under 2 GB (the GitHub limit per release asset).
#
# Usage: bash scripts/pack_data.sh [OUT_DIR]      (default: release_assets/ next to the repo folder)
set -euo pipefail
unset PYTHONPATH

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-$(dirname "$REPO")/release_assets}
PY=${PY:-$REPO/.venv/bin/python}
[ -x "$PY" ] || PY=python3
mkdir -p "$OUT"

"$PY" - "$REPO" "$OUT" <<'EOF'
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
ARCHIVES = [  # archive, folder under data/, what it holds
    ("own_house.zip", "own_house", "own house, one room: 7 lit + 6 dim photos and a walkthrough video (OnePlus Nord, "
     "1x lens), the hand sketch; tape ground truth in gt/ (also in git)"),
    ("k65.zip", "k65", "simulated k65 flat (Isaac Sim, iPhone 15 emulation): photos per room, video, LiDAR "
     "(Stray Scanner format), exact ground truth"),
    ("sample.zip", "sample", "the case study's sample captures: single_room, single_scan_floor_only, "
     "single_scan_with_ceiling (Stray Scanner folders)"),
]
STORED = {".mp4", ".mov", ".jpg", ".jpeg", ".png", ".heic", ".npz", ".pdf", ".zip"}
SKIP = {"__pycache__", ".DS_Store"}
LIMIT = 2 * 1024 ** 3 - 1


def files_of(root: Path):
    for d, dirs, names in os.walk(root, followlinks=True):
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        for n in sorted(names):
            if n not in SKIP:
                yield Path(d) / n


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []
for name, sub, what in ARCHIVES:
    src = repo / "data" / sub
    if not src.is_dir():
        sys.exit(f"{src} is missing")
    t0, n, raw = time.time(), 0, 0
    dst = out / name
    tmp = dst.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for f in files_of(src):
            arc = f"{sub}/{f.relative_to(src).as_posix()}"
            info = zipfile.ZipInfo(arc, date_time=(2026, 10, 5, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_STORED if f.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            info.file_size = f.stat().st_size                   # zip64 headers only where a file needs them
            with open(f, "rb") as fi, z.open(info, "w") as fo:
                while chunk := fi.read(1 << 22):
                    fo.write(chunk)
            n += 1
            raw += f.stat().st_size
    tmp.replace(dst)
    size = dst.stat().st_size
    if size > LIMIT:
        sys.exit(f"{name}: {size / 1e9:.2f} GB, over the 2 GB limit of a release asset")
    digest = sha256(dst)
    rows.append(dict(name=name, unpacks_to=f"data/{sub}", bytes=size, sha256=digest, files=n, what=what))
    print(f"{name:14s} {size / 1e6:8.1f} MB  {n:6d} files  {digest}  ({time.time() - t0:.0f} s)")

manifest = dict(release="data-v1", built_by="scripts/pack_data.sh", fetched_by="scripts/fetch_data.py",
                archives=rows)
(repo / "data" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"wrote {repo / 'data' / 'MANIFEST.json'}; archives in {out}")
EOF
