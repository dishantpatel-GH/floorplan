#!/usr/bin/env bash
# Final benchmark (Deliverable 5): run the ONE command (scripts/run_capture.py) on the sample captures with default
# settings and the CURRENT code. Every run is wrapped in /usr/bin/time -v and records a code fingerprint, so the
# report can say which code produced which number.
#
#   scripts/bench_final_runs.sh lidar        # drift_on (default) on the 3 captures, one at a time (clean timing)
#   scripts/bench_final_runs.sh lidar_rerun  # default on floor_only a second time (determinism, D-029)
#   scripts/bench_final_runs.sh lidar_off    # --no-drift on floor_only and with_ceiling (drift ablation)
#   scripts/bench_final_runs.sh video        # --tier video on the 3 captures (rgb.mp4 of the Stray folder)
#   scripts/bench_final_runs.sh photo LABEL [CAPTURE]  # --tier photo on <capture>__rotate/photos
#
# Output: outputs/benchmark/final/<tier>/<capture>/<variant>/{plan.json, run_report.json, stdout.log, time.log,
#         exit_code.txt, code_fingerprint.txt, load_before.txt, load_after.txt}
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/../.venv/bin/python}
D=${DATASET:-$ROOT/../TakeHome/Dataset}
F=$ROOT/outputs/benchmark/final
CAPS=("single_room/c00a170fe1 single_room" "single_scan_floor_only/1a8384c3f6 floor_only" "single_scan_with_ceiling/c7d28f72c6 with_ceiling")
PHOTO_IN=("single_room__c00a170fe1__rotate" "single_scan_floor_only__1a8384c3f6__rotate" "single_scan_with_ceiling__c7d28f72c6__rotate")

fingerprint() {  # md5 over the pipeline code: identical fingerprint = identical code
  # line 1: all of floorplan/ + run_capture.py; line 2: without floorplan/photo and floorplan/video (the LiDAR path);
  # then one line per file (md5, path) so a later change can be traced to the file
  ( cd "$ROOT"
    find floorplan scripts/run_capture.py -name '*.py' -print0 | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1
    find floorplan scripts/run_capture.py -name '*.py' -not -path 'floorplan/photo/*' -not -path 'floorplan/video/*' -print0 \
      | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1
    find floorplan scripts/run_capture.py -name '*.py' -print0 | sort -z | xargs -0 md5sum )
}

gpu_free_mb() { local f; f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1); echo "${f:-99999}"; }

one() {  # one <input> <tier> <out> [extra args...]
  # The 8 GB GPU is shared with other agents: video/photo wait for >= 3 GB free (up to 20 min), and a CUDA
  # out-of-memory crash is retried up to 3 times after 60 s. Each failed attempt's stderr is kept (time.log.attemptN).
  local inp=$1 tier=$2 out=$3; shift 3
  mkdir -p "$out"; rm -f "$out/plan.json" "$out/run_report.json" "$out"/time.log.attempt*
  local attempt rc
  for attempt in 1 2 3; do
    if [ "$tier" != lidar ]; then
      for _ in $(seq 1 120); do [ "$(gpu_free_mb)" -ge 3000 ] && break; sleep 10; done
    fi
    fingerprint > "$out/code_fingerprint.txt"; uptime > "$out/load_before.txt"; date -Is > "$out/started.txt"
    ( cd "$ROOT" && env -u PYTHONPATH /usr/bin/time -v "$PY" scripts/run_capture.py "$inp" --tier "$tier" --out "$out" "$@" ) \
        > "$out/stdout.log" 2> "$out/time.log"
    rc=$?
    echo "$attempt" > "$out/attempts.txt"
    grep -q -e "OutOfMemoryError" -e "CUDA out of memory" "$out/time.log" || break
    cp "$out/time.log" "$out/time.log.attempt$attempt"; echo "$tier $out: CUDA OOM on attempt $attempt, retrying"; sleep 60
  done
  echo "$rc" > "$out/exit_code.txt"; uptime > "$out/load_after.txt"; date -Is > "$out/finished.txt"
  echo "$tier $out exit=$rc"
}

case "${1:-}" in
  lidar)
    for p in "${CAPS[@]}"; do set -- $p; one "$D/$1" lidar "$F/lidar/$2/drift_on"; done ;;
  lidar_rerun)   # same command, same capture, second time: is the plan bit-identical? (D-029 determinism)
    set -- ${CAPS[1]}; one "$D/$1" lidar "$F/lidar/$2/drift_on_rerun" ;;
  lidar_off)
    for p in "${CAPS[@]:1}"; do set -- $p; one "$D/$1" lidar "$F/lidar/$2/drift_off" --no-drift; done ;;
  video)   # video [CAPTURE VARIANT]: all 3 captures -> default/, or one capture again -> <VARIANT>/ (run-to-run check)
    only=${2:-}; var=${3:-default}
    for p in "${CAPS[@]}"; do set -- $p; [ -n "$only" ] && [ "$only" != "$2" ] && continue
      one "$D/$1" video "$F/video/$2/$var"; done ;;
  photo)
    label=${2:?label, e.g. pre_v3 or v3}
    only=${3:-}   # optional: run one capture only (single_room | floor_only | with_ceiling)
    caps=(single_room floor_only with_ceiling)
    for i in 0 1 2; do
      [ -n "$only" ] && [ "$only" != "${caps[$i]}" ] && continue
      # single_room has no rotate-protocol folders in outputs/photo_inputs (only the old 2-folder set without doorway
      # pairs, which gives 0 rooms); the benchmark makes them with scripts/make_photo_folders.py from the final
      # LiDAR run, into outputs/benchmark/final/photo_inputs/ (see docs/modules/benchmark_final.md)
      inp=$F/photo_inputs/${PHOTO_IN[$i]}/photos
      [ -d "$inp" ] || inp=$ROOT/outputs/photo_inputs/${PHOTO_IN[$i]}/photos
      one "$inp" photo "$F/photo/${caps[$i]}/$label"
    done ;;
  *) echo "usage: $0 lidar|lidar_rerun|lidar_off|video|photo LABEL"; exit 2 ;;
esac
