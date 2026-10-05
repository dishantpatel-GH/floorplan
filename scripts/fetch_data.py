#!/usr/bin/env python3
"""Download the benchmark data into data/ and check each archive's SHA-256 against data/MANIFEST.json.

The archives are assets of the GitHub release "data-v1" of this repo (built by scripts/pack_data.sh):
own_house.zip -> data/own_house, k65.zip -> data/k65. The case study's sample captures are not re-hosted: put
your copy in data/sample/ (data/README.md). The small ground-truth files are in git already. Standard library only, so it runs before any env exists.

Usage:
  python scripts/fetch_data.py                    # every archive in data/MANIFEST.json
  python scripts/fetch_data.py own_house k65      # some of them
  python scripts/fetch_data.py --list             # what there is, with sizes
  python scripts/fetch_data.py --url URL          # another source: a release download URL, a mirror, file:///dir/
The default URL is https://github.com/<owner>/<repo>/releases/download/data-v1/ from `git remote get-url origin`.
An archive whose SHA-256 is already recorded in data/.fetched/ is skipped; --force fetches it again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "MANIFEST.json"


def github_repo() -> str:
    """owner/name of the GitHub repo this clone came from."""
    try:
        origin = subprocess.run(["git", "-C", str(REPO), "remote", "get-url", "origin"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sys.exit("no git remote 'origin' to find the release from: pass --url")
    m = re.match(r"^(?:https://|ssh://git@|git@)github\.com[:/]([^/]+)/(.+?)(?:\.git)?/?$", origin)
    if not m:
        sys.exit(f"origin is {origin}, not a GitHub repo: pass --url (a release download URL or file:///dir/)")
    return f"{m[1]}/{m[2]}"


def gh_download(repo: str, release: str, name: str, dst: Path) -> str:
    """A private repo's release needs a login: let the GitHub CLI fetch the asset. Returns the SHA-256."""
    subprocess.run(["gh", "release", "download", release, "--repo", repo, "--pattern", name, "--dir",
                    str(dst.parent), "--clobber"], check=True)
    (dst.parent / name).replace(dst)
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path) -> str:
    """Stream url to dst; return the SHA-256."""
    h, done, t0 = hashlib.sha256(), 0, time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        while chunk := r.read(1 << 22):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total and sys.stdout.isatty():
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\r  {done / 1e6:8.1f} MB in {time.time() - t0:.0f} s" + " " * 20)
    return h.hexdigest()


def unpack(archive: Path, dest: Path) -> int:
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for n in names:                             # no absolute paths, no way out of dest
            if n.startswith("/") or ".." in Path(n).parts:
                sys.exit(f"{archive.name}: unsafe path {n}")
        z.extractall(dest)
    return len(names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="archives to fetch (own_house, k65); default all")
    ap.add_argument("--url", help="base URL of the archives (ends with /); default: the repo's release")
    ap.add_argument("--dest", type=Path, default=REPO / "data", help="where to unpack (default data/)")
    ap.add_argument("--list", action="store_true", help="list the archives and stop")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded zip files in <dest>/.download/")
    ap.add_argument("--force", action="store_true", help="fetch even if already fetched")
    a = ap.parse_args()

    man = json.loads(MANIFEST.read_text())
    rows = {r["name"].removesuffix(".zip"): r for r in man["archives"]}
    if a.list:
        for k, r in rows.items():
            print(f"{k:10s} {r['bytes'] / 1e6:7.0f} MB -> {r['unpacks_to']}: {r['what']}")
        return
    local = {r["name"]: r for r in man.get("not_in_release", [])}
    for n in [n for n in a.names if n in local]:
        print(f"{n}: {local[n]['what']}")
    want = [n for n in a.names if n not in local] or ([] if a.names else list(rows))
    bad = [n for n in want if n not in rows]
    if bad:
        sys.exit(f"unknown archive(s) {bad}; there are {list(rows)}")
    repo = None if a.url else github_repo()
    base = a.url or f"https://github.com/{repo}/releases/download/{man['release']}/"
    base = base if base.endswith("/") else base + "/"
    dl, mark = a.dest / ".download", a.dest / ".fetched"
    dl.mkdir(parents=True, exist_ok=True)
    mark.mkdir(parents=True, exist_ok=True)
    failed = []
    for n in want:
        r = rows[n]
        m = mark / f"{r['name']}.sha256"
        if not a.force and m.exists() and m.read_text().strip() == r["sha256"]:
            print(f"{r['name']}: already in {a.dest / n}")
            continue
        print(f"{r['name']} ({r['bytes'] / 1e6:.0f} MB) from {base}")
        part = dl / (r["name"] + ".part")
        try:
            try:
                digest = download(base + r["name"], part)
            except urllib.error.HTTPError:
                if not (repo and shutil.which("gh")):
                    raise
                print("  not public; trying the GitHub CLI (gh auth login first if it asks)")
                digest = gh_download(repo, man["release"], r["name"], part)
        except (urllib.error.URLError, OSError, subprocess.CalledProcessError) as e:
            part.unlink(missing_ok=True)
            print(f"  could not download it: {e}")
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                print(f"  Is release {man['release']} published with this file? For a private repo install the GitHub "
                      "CLI (gh auth login) or pass --url.")
            if n == "sample":
                print("  The sample captures are the case study's own data: if the release does not carry them, "
                      "download them from the link in the case-study brief and put the three folders in "
                      "data/sample/ (data/sample/single_room/c00a170fe1/rgb.mp4 and so on).")
            failed.append(r["name"])
            continue
        if digest != r["sha256"]:
            part.unlink()
            print(f"  SHA-256 {digest} does not match {r['sha256']}: removed")
            failed.append(r["name"])
            continue
        print(f"  SHA-256 ok; {unpack(part, a.dest)} files into {a.dest / n}")
        m.write_text(r["sha256"] + "\n")
        if a.keep:
            part.replace(dl / r["name"])
        else:
            part.unlink()
    if not any(dl.iterdir()):
        shutil.rmtree(dl)
    if failed:
        sys.exit(f"not fetched: {', '.join(failed)}")
    print("done")


if __name__ == "__main__":
    main()
