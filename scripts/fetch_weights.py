#!/usr/bin/env python
"""Download every model weight the video and photo tiers need, at pinned revisions, with checksums.

The case study requires "weights and large binaries fetched by script or volume": nothing is committed to the repo,
and a fresh machine gets exactly the versions this code was evaluated with.

  * Hugging Face models are pinned by commit hash (a later push to the model repo cannot change our results).
  * GitHub-release files (GeoCalib, ALIKED, LightGlue) have no commit hash, so they are pinned by SHA-256 instead.
  * The large files are also checked by SHA-256 after the download (WEIGHT_SHA256).
  * Files go to weights/ in the repo: weights/hf (the Hugging Face cache, HF_HOME) and weights/torch (torch.hub,
    TORCH_HOME). floorplan/paths.py points both libraries there when the folders exist, and the DPVO and segmenter
    subprocesses inherit that, so the pipeline then runs fully offline (HF_HUB_OFFLINE=1). HF_HOME / TORCH_HOME set
    in the shell win; --system-cache uses the libraries' default caches (~/.cache) instead.

Usage: python scripts/fetch_weights.py [--experiments] [--dry-run] [--system-cache]
  --experiments  also fetch the models used only by the option comparison (DA3METRIC, DA3-BASE, MapAnything)

Licences: MoGe-2 MIT; DPVO MIT repo (weights ship without a separate licence); GeoCalib CC-BY-4.0 (attribution);
ALIKED BSD-3; LightGlue Apache-2.0; DA3METRIC-LARGE / DA3-BASE Apache-2.0; MapAnything
facebook/map-anything-apache Apache-2.0. All of these allow commercial use. SegFormer-B5 does not: the NVIDIA Source
Code License allows research or evaluation only (docs/DISCLOSURES.md). Do NOT swap in facebook/map-anything
(CC-BY-NC) or DA3-GIANT/LARGE (CC-BY-NC).
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HF_FILES = [  # (repo, revision, files or None for the whole snapshot, needed by)
    ("Ruicheng/moge-2-vitl-normal", "cb0e8bbd6b1e243589717c78e750b1ba4c093acf", ["model.pt"], "core"),
    ("pablovela5620/dpvo", "c998d3b57bf47c619f851d37dff0aa1fa43e1c34", ["dpvo.pth"], "core"),
    ("nvidia/segformer-b5-finetuned-ade-640-640", "739f5d4692954e4a185eac280dec1ba5a7d52f1d",
     ["config.json", "preprocessor_config.json", "pytorch_model.bin"], "seg"),
    ("depth-anything/DA3METRIC-LARGE", "4010e39f3634a45bc60553321fb49fb760bd594e", None, "experiments"),
    ("depth-anything/DA3-BASE", "f4a6c9b3c95e41c82048423d3493a81ec3fa810e", None, "experiments"),
    ("facebook/map-anything-apache", "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a", None, "experiments"),
]

WEIGHT_SHA256 = {  # (repo, file): SHA-256 of the file at the pinned revision
    ("Ruicheng/moge-2-vitl-normal", "model.pt"): "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01",
    ("pablovela5620/dpvo", "dpvo.pth"): "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12",
    ("nvidia/segformer-b5-finetuned-ade-640-640", "pytorch_model.bin"):
        "a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df",
}

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


def seg_cache_dir() -> str | None:
    """The hub cache the segmenter env reads. None: the library default, which the segmenter inherits too."""
    if os.environ.get("HF_HUB_CACHE"):
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from floorplan.photo.semantic import seg_hf_home
    return str(seg_hf_home() / "hub")


def fetch_hf(experiments: bool, dry: bool) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download
    for repo, rev, files, group in HF_FILES:
        if group == "experiments" and not experiments:
            continue
        cache = seg_cache_dir() if group == "seg" else None
        print(f"[hf] {repo}@{rev[:8]} {files or 'snapshot'}" + (f" -> {cache}" if cache else ""))
        if dry:
            continue
        if files:
            for f in files:
                path = hf_hub_download(repo, f, revision=rev, cache_dir=cache)
                want = WEIGHT_SHA256.get((repo, f))
                if want and sha256(Path(path)) != want:
                    sys.exit(f"checksum mismatch for {repo}/{f} at {path}: expected {want}")
        else:
            snapshot_download(repo, revision=rev, cache_dir=cache)


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
    ap.add_argument("--system-cache", action="store_true", help="the libraries' default caches, not weights/")
    a = ap.parse_args()
    if not a.system_cache:                          # before huggingface_hub and torch are imported
        for var, sub in (("HF_HOME", "hf"), ("TORCH_HOME", "torch")):
            if var not in os.environ:
                (REPO / "weights" / sub).mkdir(parents=True, exist_ok=True)
                os.environ[var] = str(REPO / "weights" / sub)
    print(f"HF_HOME={os.environ.get('HF_HOME', '(default)')} TORCH_HOME={os.environ.get('TORCH_HOME', '(default)')}")
    fetch_hf(a.experiments, a.dry_run)
    fetch_urls(a.dry_run)
    print("done")


if __name__ == "__main__":
    main()
