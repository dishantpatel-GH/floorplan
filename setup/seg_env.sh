#!/usr/bin/env bash
# Builds the env for the photo tier's wall masks (SegFormer-B5 on ADE20K, D-060). It needs transformers, which must
# stay out of the main env (docs/ISSUES.md I-002), so it gets its own env with its own torch. No sudo, no compiling.
#
# The env goes to envs/seg in the repo folder (git-ignored), where floorplan/photo/semantic.py looks first; set
# FLOORPLAN_SEG_PYTHON (for both) to put it elsewhere. Versions: setup/seg_constraints.txt. The model itself is
# fetched by scripts/fetch_weights.py. Without this env the photo tier still runs, from geometry only, and says so in
# run_report.json and plan.json.
#
# Usage: bash setup/seg_env.sh
set -euo pipefail
unset PYTHONPATH          # packages on it (from a sourced ROS install, for example) leak into the env

SETUP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$SETUP")
PY=${FLOORPLAN_SEG_PYTHON:-$REPO/envs/seg/bin/python}
ENV=$(dirname "$(dirname "$PY")")
C=$SETUP/seg_constraints.txt

[ -x "$PY" ] || uv venv --python 3.11.14 "$ENV"
# PyPI part first: the filelock and fsspec versions transformers was used with are newer than the ones on the
# PyTorch index, and the torch step below keeps them.
uv pip install --python "$PY" -c "$C" transformers scipy numpy pillow
# torch only from the cu130 index: with PyPI as an extra index, uv takes the CUDA wheels from PyPI, which is much
# slower here.
uv pip install --python "$PY" -c "$C" torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Check the imports, and whether the pinned model is in the cache the segmenter reads (the same rule as
# floorplan/photo/semantic.py: $HF_HOME, which floorplan/paths.py sets to weights/hf in the repo when that exists,
# else weights/hf_seg next to envs/).
if [ -z "${HF_HOME:-}" ] && [ -d "$REPO/weights/hf" ]; then export HF_HOME=$REPO/weights/hf; fi
export HF_HOME=${HF_HOME:-$(dirname "$(dirname "$ENV")")/weights/hf_seg}
HF_HUB_OFFLINE=1 "$PY" - "$SETUP/../scripts" <<'EOF'
import sys

import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor  # noqa: F401

sys.path.insert(0, sys.argv[1])
from seg_walls import MODEL, REVISION  # noqa: E402

print(f"seg env ok: torch {torch.__version__} (CUDA: {torch.cuda.is_available()}), "
      f"transformers {transformers.__version__}")
try:
    for f in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        hf_hub_download(MODEL, f, revision=REVISION)
    print(f"model {MODEL}@{REVISION[:8]} is in the cache")
except Exception:
    print(f"model {MODEL}@{REVISION[:8]} is not in the cache yet: run python scripts/fetch_weights.py (main env)")
EOF
echo "python for the wall masks: $PY"
