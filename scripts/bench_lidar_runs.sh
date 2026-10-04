#!/usr/bin/env bash
# Benchmark (lidar_sample part): runs the ONE command per capture for the LiDAR tier, default settings,
# plus --no-drift on the two multi-room captures (plan-level drift ablation). Times each run (/usr/bin/time -v).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/../.venv/bin/python}
D=${DATASET:-$ROOT/../TakeHome/Dataset}
B=$ROOT/outputs/benchmark/lidar
run() { # capture_dir capture_name variant [flags]
  local inp=$1 name=$2 var=$3; shift 3
  local out=$B/$name/$var; mkdir -p "$out"
  ( cd "$ROOT" && env -u PYTHONPATH /usr/bin/time -v "$PY" scripts/run_capture.py "$inp" --tier lidar --out "$out" "$@" ) \
     > "$out/stdout.log" 2> "$out/time.log"
  echo "$name/$var exit=$?"
}
run $D/single_room/c00a170fe1              single_room      drift_on &
run $D/single_scan_floor_only/1a8384c3f6   floor_only       drift_on &
run $D/single_scan_with_ceiling/c7d28f72c6 with_ceiling     drift_on &
run $D/single_scan_floor_only/1a8384c3f6   floor_only       drift_off --no-drift &
run $D/single_scan_with_ceiling/c7d28f72c6 with_ceiling     drift_off --no-drift &
wait
