"""Where the environments, third-party code and weights live.

setup/install.sh puts everything inside the repo folder, git-ignored: .venv/, envs/dpvo/, envs/seg/, third_party/,
weights/hf/ (Hugging Face cache) and weights/torch/ (torch.hub). The FLOORPLAN_* variables override single paths and
FLOORPLAN_THIRD_PARTY_ROOT the folder they are looked up in. The folder that holds the repo is searched second: the
dev machine keeps its envs there.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def find(rel: str, env: str | None = None) -> Path:
    """$env if set; else rel under FLOORPLAN_THIRD_PARTY_ROOT if set; else the first of <repo>/rel and <repo>/../rel
    that exists; else <repo>/rel, where setup/install.sh puts it."""
    if env and os.environ.get(env):
        return Path(os.environ[env])
    if os.environ.get("FLOORPLAN_THIRD_PARTY_ROOT"):
        return Path(os.environ["FLOORPLAN_THIRD_PARTY_ROOT"]) / rel
    for base in (REPO, REPO.parent):
        if (base / rel).exists():
            return base / rel
    return REPO / rel


def use_repo_weights() -> None:
    """Read model weights from weights/ in the repo when scripts/fetch_weights.py put them there, unless HF_HOME or
    TORCH_HOME are set. Runs on import of the package, before huggingface_hub reads HF_HOME; subprocesses (DPVO, the
    segmenter) inherit the variables."""
    for var, sub in (("HF_HOME", "hf"), ("TORCH_HOME", "torch")):
        d = REPO / "weights" / sub
        if var not in os.environ and d.is_dir():
            os.environ[var] = str(d)
