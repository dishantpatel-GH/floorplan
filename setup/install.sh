#!/usr/bin/env bash
# One-command setup. Everything goes inside the repo folder and is git-ignored:
#   .venv/           main env: Python 3.11, torch 2.14.1 (cu130, or cpu), requirements.txt with --no-deps
#   weights/         model weights at pinned revisions, checked by SHA-256 (scripts/fetch_weights.py)
#   envs/dpvo/, third_party/dpvo*   the video tier's DPVO env, built from source (setup/dpvo_setup.sh; needs nvcc)
#   envs/seg/        the photo tier's wall segmenter env (setup/seg_env.sh)
# Then it runs the tests. Data is separate: python scripts/fetch_data.py (data/README.md).
#
# Usage: bash setup/install.sh [--cpu-only] [--no-video] [--no-seg]
#   --cpu-only   CPU torch, no weights, no DPVO, no segmenter: the LiDAR tier, the tests and the scoring scripts
#   --no-video   skip the DPVO env (the video tier will not run)
#   --no-seg     skip the segmenter env (the photo tier runs from geometry only and says so in plan.json)
#
# Needs: Linux x86_64, uv (https://docs.astral.sh/uv/; it fetches Python 3.11 itself), git, curl; for the video tier
# an NVIDIA GPU and the CUDA toolkit (nvcc, CUDA_HOME, default /usr/local/cuda). No sudo. Re-running it is safe: done
# steps are quick.
set -euo pipefail
unset PYTHONPATH          # packages on it (a sourced ROS install, for example) leak into the envs and the tests
unset HF_HUB_OFFLINE      # the weights are downloaded here; runs set it afterwards

CPU=0 VIDEO=1 SEG=1
for a in "$@"; do
  case $a in
    --cpu-only) CPU=1 VIDEO=0 SEG=0 ;;
    --no-video) VIDEO=0 ;;
    --no-seg) SEG=0 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option $a (see --help)" >&2; exit 2 ;;
  esac
done

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=$REPO/.venv/bin/python
TIMES=()
step() {  # step "name" cmd...: run, time, remember
  local name=$1; shift
  local t0=$SECONDS
  echo; echo "=== $name"
  "$@"
  TIMES+=("$(printf '%-34s %5d s' "$name" $((SECONDS - t0)))")
}
skip() { TIMES+=("$(printf '%-34s %7s' "$1" skipped) ($2)"); }

for tool in uv git curl; do
  command -v "$tool" >/dev/null || { echo "$tool is needed (uv: curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2; exit 1; }
done
command -v ffprobe >/dev/null || echo "note: no ffprobe; the video tier then reads rotation and focal length less reliably"

main_env() {
  [ -x "$PY" ] || uv venv --python 3.11 "$REPO/.venv"
  if [ $CPU = 1 ]; then
    uv pip install --python "$PY" torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
  else
    uv pip install --python "$PY" torch==2.14.1+cu130 torchvision==0.29.1+cu130 \
        --index-url https://download.pytorch.org/whl/cu130
  fi
  uv pip install --python "$PY" --no-deps -r requirements.txt     # the full pinned closure: nothing else comes in
}
step "main env (.venv)" main_env

# weights/hf and weights/torch: the DPVO and segmenter setups below read the same caches
export HF_HOME=${HF_HOME:-$REPO/weights/hf} TORCH_HOME=${TORCH_HOME:-$REPO/weights/torch}
if [ $CPU = 1 ]; then skip "weights" "--cpu-only: only the GPU tiers use them"
else step "weights (weights/)" "$PY" scripts/fetch_weights.py; fi

if [ $VIDEO = 1 ]; then
  if [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]; then step "video tier: DPVO env (envs/dpvo)" bash setup/dpvo_setup.sh
  else skip "video tier: DPVO env" "no nvcc in ${CUDA_HOME:-/usr/local/cuda}/bin: set CUDA_HOME"; fi
else skip "video tier: DPVO env" "--no-video"; fi

if [ $SEG = 1 ]; then step "photo tier: segmenter env (envs/seg)" bash setup/seg_env.sh
else skip "photo tier: segmenter env" "--no-seg"; fi

step "tests" "$PY" -m pytest -q tests

echo; echo "=== done; time per step"
printf '%s\n' "${TIMES[@]}"
echo "total                              $SECONDS s"
echo
echo "Next: python scripts/fetch_data.py   (benchmark data into data/), then the commands in README.md."
echo "Activate the env with: source .venv/bin/activate; runs work offline: export HF_HUB_OFFLINE=1"
