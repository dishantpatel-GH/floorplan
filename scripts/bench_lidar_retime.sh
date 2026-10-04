#!/usr/bin/env bash
# Benchmark (lidar_sample): second, SEQUENTIAL run of the default command on the three captures (one at a time,
# for cleaner timing than the 5-way parallel first pass, and as a run-to-run determinism check).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/../.venv/bin/python}
D=${DATASET:-$ROOT/../TakeHome/Dataset}
B=$ROOT/outputs/benchmark/lidar
for pair in "single_room/c00a170fe1 single_room" "single_scan_floor_only/1a8384c3f6 floor_only" "single_scan_with_ceiling/c7d28f72c6 with_ceiling"; do
  set -- $pair; out=$B/$2/drift_on_rerun2; mkdir -p "$out"
  uptime > "$out/load_before.txt"
  ( cd "$ROOT" && env -u PYTHONPATH /usr/bin/time -v "$PY" scripts/run_capture.py "$D/$1" --tier lidar --out "$out" ) > "$out/stdout.log" 2> "$out/time.log"
  echo "$2 exit=$?"; uptime > "$out/load_after.txt"
done
