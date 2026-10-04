#!/usr/bin/env python
"""Download every model weight the video tier needs, at pinned revisions, with checksums.

The case study requires "weights and large binaries fetched by script or volume": nothing is committed to the repo,
and a fresh machine gets exactly the versions this code was evaluated with.

  * Hugging Face models are pinned by commit hash (a later push to the model repo cannot change our results).
  * GitHub-release files (GeoCalib, ALIKED, LightGlue) have no commit hash, so they are pinned by SHA-256 instead.
  * Files go to the standard caches (HF_HOME, torch.hub), which is where the libraries look for them, so the
    pipeline then runs fully offline (HF_HUB_OFFLINE=1).

Usage: python scripts/fetch_weights.py [--experiments] [--dry-run]
  --experiments  also fetch the models used only by the option comparison (DA3METRIC, DA3-BASE, MapAnything)

Licences (all allow commercial use): MoGe-2 MIT; DPVO MIT repo (weights ship without a separate licence);
GeoCalib CC-BY-4.0 (attribution); ALIKED BSD-3; LightGlue Apache-2.0; DA3METRIC-LARGE / DA3-BASE Apache-2.0;
MapAnything facebook/map-anything-apache Apache-2.0. Do NOT swap in facebook/map-anything (CC-BY-NC) or
DA3-GIANT/LARGE (CC-BY-NC).
"""
import argparse
import hashlib
import sys
from pathlib import Path

HF_FILES = [  # (repo, revision, files or None for the whole snapshot, needed by)
    ("Ruicheng/moge-2-vitl-normal", "cb0e8bbd6b1e243589717c78e750b1ba4c093acf", ["model.pt"], "core"),
    ("pablovela5620/dpvo", "c998d3b57bf47c619f851d37dff0aa1fa43e1c34", ["dpvo.pth"], "core"),
    ("depth-anything/DA3METRIC-LARGE", "4010e39f3634a45bc60553321fb49fb760bd594e", None, "experiments"),
    ("depth-anything/DA3-BASE", "f4a6c9b3c95e41c82048423d3493a81ec3fa810e", None, "experiments"),
    ("facebook/map-anything-apache", "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a", None, "experiments"),
]

URL_FILES = [  # (url, path relative to torch.hub dir, sha256)
    ("https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar", "geocalib/pinhole.tar",
     "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac"),
    ("https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth", "checkpoints/aliked-n16.pth",
     "5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2"),
    ("https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth",
     "checkpoints/aliked_lightglue_v0-1_arxiv.pth",
     "d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_hf(experiments: bool, dry: bool) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download
    for repo, rev, files, group in HF_FILES:
        if group == "experiments" and not experiments:
            continue
        print(f"[hf] {repo}@{rev[:8]} {files or 'snapshot'}")
        if dry:
            continue
        if files:
            for f in files:
                hf_hub_download(repo, f, revision=rev)
        else:
            snapshot_download(repo, revision=rev)


def fetch_urls(dry: bool) -> None:
    import torch
    hub = Path(torch.hub.get_dir())
    for url, rel, digest in URL_FILES:
        dst = hub / rel
        if dst.exists() and sha256(dst) == digest:
            print(f"[url] ok   {dst}")
            continue
        print(f"[url] get  {url} -> {dst}")
        if dry:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(dst), progress=True)
        if sha256(dst) != digest:
            dst.unlink()
            sys.exit(f"checksum mismatch for {url}; file removed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    fetch_hf(a.experiments, a.dry_run)
    fetch_urls(a.dry_run)
    print("done")


if __name__ == "__main__":
    main()
