#!/usr/bin/env bash
# Builds the env for the video tier's camera trajectory (DPVO). DPVO needs compiled CUDA ops, so it runs in its own
# env as a subprocess (floorplan/video/vo.py). No sudo. About 5 min on the dev machine.
#
# Steps (docs/DISCLOSURES.md 5b): DPVO @ 0ac95b6, setup/dpvo.patch (a build fix for torch 2.14), torch 2.14.1+cu130
# and DPVO's dependencies pinned by setup/dpvo_constraints.txt, Eigen 3.4.0 headers, a build of DPVO's CUDA ops, the
# torch_scatter shim (setup/dpvo_shims), and the weights dpvo.pth.
#
# Paths default to what vo.py looks for, in the repo folder (git-ignored; setup/install.sh calls this script):
#   envs/dpvo                the env       (FLOORPLAN_DPVO_PYTHON = envs/dpvo/bin/python)
#   third_party/dpvo         DPVO source   (FLOORPLAN_DPVO_REPO)
#   third_party/dpvo_shims   the shim      (FLOORPLAN_DPVO_SHIMS)
# vo.py reads the same variables (and FLOORPLAN_THIRD_PARTY_ROOT for another folder), so set them for both.
# The checkpoint goes to weights/hf in the repo when that folder exists (scripts/fetch_weights.py), else to $HF_HOME.
#
# Needs uv, git, curl, an NVIDIA GPU and the CUDA toolkit (built with nvcc 13.0; CUDA_HOME, default /usr/local/cuda).
# TORCH_CUDA_ARCH_LIST defaults to 8.9, the GPU it was built on (RTX 2000 Ada); set it for another GPU.
#
# Usage: bash setup/dpvo_setup.sh
set -euo pipefail
unset PYTHONPATH          # packages on it (from a sourced ROS install, for example) leak into the build

SETUP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$SETUP")
ROOT=${FLOORPLAN_THIRD_PARTY_ROOT:-$REPO}
PY=${FLOORPLAN_DPVO_PYTHON:-$ROOT/envs/dpvo/bin/python}
ENV=$(dirname "$(dirname "$PY")")
DPVO=${FLOORPLAN_DPVO_REPO:-$ROOT/third_party/dpvo}
SHIMS=${FLOORPLAN_DPVO_SHIMS:-$ROOT/third_party/dpvo_shims}
C=$SETUP/dpvo_constraints.txt
DPVO_COMMIT=0ac95b656d1fda91c271d2a106460d19ad966fc7
CKPT_REVISION=c998d3b57bf47c619f851d37dff0aa1fa43e1c34     # as in scripts/fetch_weights.py
CKPT_SHA256=30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12
if [ -z "${HF_HOME:-}" ] && [ -d "$REPO/weights/hf" ]; then export HF_HOME=$REPO/weights/hf; fi
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9} MAX_JOBS=${MAX_JOBS:-6}

# DPVO source at the pinned commit (shallow fetch)
if [ ! -d "$DPVO/.git" ]; then
  mkdir -p "$DPVO"
  git -C "$DPVO" init -q
  git -C "$DPVO" fetch -q --depth 1 https://github.com/princeton-vl/DPVO "$DPVO_COMMIT"
  git -C "$DPVO" checkout -q FETCH_HEAD
fi
if [ "$(git -C "$DPVO" rev-parse HEAD)" != "$DPVO_COMMIT" ]; then
  echo "$DPVO is not at DPVO commit $DPVO_COMMIT" >&2
  exit 1
fi

# The env. torch comes only from the cu130 index: with PyPI as an extra index, uv takes the CUDA wheels from PyPI,
# which is much slower here.
[ -x "$PY" ] || uv venv --python 3.11.14 "$ENV"
uv pip install --python "$PY" -c "$C" torch==2.14.1+cu130 torchvision==0.29.1+cu130 \
    --index-url https://download.pytorch.org/whl/cu130
# numba and pypose are required: dpvo/patchgraph.py imports loop_closure/optim_utils at module level.
# evo (GPL-3.0) is only imported by DPVO's demo and evaluation scripts, never by this repo.
uv pip install --python "$PY" -c "$C" numpy opencv-python einops yacs plyfile tqdm matplotlib scipy ninja wheel \
    huggingface_hub kornia numba==0.68.0 pypose pytest evo

# Eigen headers (a step of DPVO's README; no system Eigen needed)
if [ ! -d "$DPVO/thirdparty/eigen-3.4.0" ]; then
  mkdir -p "$DPVO/thirdparty"
  curl -sSfL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar xz -C "$DPVO/thirdparty"
fi

# Build fix for torch >= 2.14: Tensor.type() -> Tensor.scalar_type() in the dispatch macros; no behaviour change
if git -C "$DPVO" apply --reverse --check "$SETUP/dpvo.patch" 2>/dev/null; then
  echo "dpvo.patch is already applied"
else
  git -C "$DPVO" apply "$SETUP/dpvo.patch"
fi

# Build cuda_corr, cuda_ba and lietorch_backends, and install the torch_scatter shim. Both are editable installs, so
# the .so files land in the DPVO folder. The shim is copied next to DPVO first: vo.py also puts that folder on
# PYTHONPATH.
if [ ! "$SHIMS" -ef "$SETUP/dpvo_shims" ]; then
  mkdir -p "$SHIMS"
  cp -r "$SETUP/dpvo_shims/." "$SHIMS/"
fi
export PATH=$ENV/bin:$CUDA_HOME/bin:$PATH
uv pip install --python "$PY" -c "$C" --no-build-isolation --no-deps -e "$SHIMS" -e "$DPVO"
rm -rf "$DPVO/build"      # intermediate objects only (about 256 MB)

# Weights at the pinned revision, checked by SHA-256. The link lets upstream `python demo.py` (default
# --network dpvo.pth) run from the DPVO folder.
CK=$("$PY" - "$CKPT_REVISION" "$CKPT_SHA256" <<'EOF' | tail -n 1
import hashlib
import sys

from huggingface_hub import hf_hub_download

path = hf_hub_download("pablovela5620/dpvo", "dpvo.pth", revision=sys.argv[1])
digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
if digest != sys.argv[2]:
    sys.exit(f"dpvo.pth: sha256 {digest}, expected {sys.argv[2]}")
print(path)
EOF
)
ln -sfn "$CK" "$DPVO/dpvo.pth"

# Quick check: the compiled ops load, lietorch gives the same results on the GPU and the CPU, and the shim matches a
# plain softmax. The first video run is the full test.
"$PY" - <<'EOF'
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)   # upstream deprecation notices (autocast, jit.script)
import torch
import cuda_ba, cuda_corr, lietorch_backends  # noqa: F401  (the compiled ops)
import torch_scatter
from dpvo.dpvo import DPVO  # noqa: F401  (the model and everything it imports: numba, pypose, ...)
from dpvo.lietorch import SE3, Sim3

assert torch.cuda.is_available(), "torch sees no CUDA device"
g = torch.Generator().manual_seed(0)
for G in (SE3, Sim3):
    xi = torch.randn(256, G.manifold_dim, dtype=torch.float64, generator=g) * 0.5
    err = (G.exp(xi).log() - G.exp(xi.cuda()).log().cpu()).abs().max().item()
    assert err < 1e-9, f"lietorch {G.__name__}: GPU and CPU differ by {err:.1e}"
x, idx = torch.randn(2000, generator=g), torch.randint(0, 50, (2000,), generator=g)
ref = torch.zeros_like(x)
for k in idx.unique():
    ref[idx == k] = torch.softmax(x[idx == k], 0)
err = (torch_scatter.scatter_softmax(x, idx, dim=0) - ref).abs().max().item()
assert err < 1e-5, f"torch_scatter shim: scatter_softmax differs by {err:.1e}"
print(f"DPVO env ok: torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
EOF
echo "python for the video tier: $PY"
