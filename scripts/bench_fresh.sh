#!/usr/bin/env bash
# Fresh benchmark runs of every benchmark capture with the code at HEAD, then results/ (scripts/bench_results.py).
#
#   scripts/bench_fresh.sh            # all steps, one after another, into outputs/benchmark/fresh_<HHMM>/
#   VIDEO=live scripts/bench_fresh.sh # video tier live (DPVO + MoGe-2 on the GPU, ~10 min a take) instead of replays
#
# The code is a git archive of HEAD (code/ in the run folder), so edits in the working tree cannot leak in; its hash
# is in CODE.txt. Every step is timed with /usr/bin/time -v and leaves started/finished/exit_code/load files.
# GPU steps take the shared lock outputs/.gpu.lock (one GPU job at a time on the 8 GB card).
#
# Video default (VIDEO=replay): the plan step and everything after it, with the HEAD code, on the 'auto' replays of
# cached DPVO/MoGe-2 runs (outputs/video_auto/replay/<cache>, D-087; the video front end has not changed since those
# replays, only the plan step has). Those caches exist only on the machine that made them; on a clone use VIDEO=live.
# The 16:00 run on 5 Oct did the same steps in two lanes (GPU steps and CPU steps side by side).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}; [ -x "$PY" ] || PY=$ROOT/../.venv/bin/python
O=$ROOT/outputs
DATA=${DATA:-$ROOT/data}
OWN_PHOTOS=${OWN_PHOTOS:-$O/own_house/capture/photos}          # room/ (lit, 7) and room_take2/ (dim, 6)
OWN_VIDEO=${OWN_VIDEO:-$O/own_house/capture/video/take1.mp4}
K65_PHOTOS=${K65_PHOTOS:-$O/fixes/k65_inputs_1x}               # 31 photos, 5 room folders, 1x lens (the k65 benchmark)
K65=${K65:-$O/sim/k65v2/capture_iphone15}                      # lidar/take1, video/take1.MOV, gt/
SAMPLE=${SAMPLE:-$DATA/sample}; [ -d "$SAMPLE/single_room" ] || SAMPLE=$ROOT/../TakeHome/Dataset
VIDEO=${VIDEO:-replay}
F=${F:-$O/benchmark/fresh_$(date +%H%M)}
mkdir -p "$F/code" "$F/in/own_dim" "$F/in/own_lit" "$F/in/k65"
git -C "$ROOT" archive HEAD | tar -x -C "$F/code"
ln -sfn "$(readlink -f "$O")" "$F/code/outputs"
echo "HEAD $(git -C "$ROOT" rev-parse HEAD) ($(git -C "$ROOT" log -1 --format='%cd %s' --date=iso HEAD)); code = git archive of HEAD in code/, exported $(date -Is)" > "$F/CODE.txt"
ln -sfn "$OWN_PHOTOS/room_take2" "$F/in/own_dim/room"
ln -sfn "$OWN_PHOTOS/room" "$F/in/own_lit/room"
for r in "$K65_PHOTOS"/*/; do ln -sfn "$(readlink -f "$r")" "$F/in/k65/$(basename "$r")"; done
# the photo work folder sits next to the input (photo_work__<name>): a new input folder = no cached depth/features
export FLOORPLAN_SEG_PYTHON=${FLOORPLAN_SEG_PYTHON:-$ROOT/../envs/seg/bin/python}
export FLOORPLAN_THIRD_PARTY_ROOT=${FLOORPLAN_THIRD_PARTY_ROOT:-$(cd "$ROOT/.." && pwd)}

one() {  # one <out> <gpu|cpu> <cmd...>
  local out=$1 kind=$2; shift 2
  mkdir -p "$out"
  uptime > "$out/load_before.txt"; date -Is > "$out/started.txt"; cp "$F/CODE.txt" "$out/"
  if [ "$kind" = gpu ]; then
    ( cd "$F/code" && env -u PYTHONPATH flock "$O/.gpu.lock" /usr/bin/time -v "$@" ) > "$out/stdout.log" 2> "$out/time.log"
  else
    ( cd "$F/code" && env -u PYTHONPATH /usr/bin/time -v "$@" ) > "$out/stdout.log" 2> "$out/time.log"
  fi
  echo $? > "$out/exit_code.txt"; uptime > "$out/load_after.txt"; date -Is > "$out/finished.txt"
  echo "$(date +%T) $out exit=$(cat "$out/exit_code.txt")"
}

# photo: own house dim and lit (the repeatability pair), k65 (stitched, per-room folders)
one "$F/photo/own_dim" gpu "$PY" scripts/run_capture.py "$F/in/own_dim" --tier photo --out "$F/photo/own_dim"
one "$F/photo/own_lit" gpu "$PY" scripts/run_capture.py "$F/in/own_lit" --tier photo --out "$F/photo/own_lit"
one "$F/photo/k65" gpu "$PY" scripts/run_capture.py "$F/in/k65" --tier photo --out "$F/photo/k65"
# LiDAR: the three sample captures (the reference for their video runs), k65, and the drift ablation
one "$F/lidar/single_room" cpu "$PY" scripts/run_capture.py "$SAMPLE/single_room/c00a170fe1" --tier lidar --out "$F/lidar/single_room"
one "$F/lidar/floor_only" cpu "$PY" scripts/run_capture.py "$SAMPLE/single_scan_floor_only/1a8384c3f6" --tier lidar --out "$F/lidar/floor_only"
one "$F/lidar/with_ceiling" cpu "$PY" scripts/run_capture.py "$SAMPLE/single_scan_with_ceiling/c7d28f72c6" --tier lidar --out "$F/lidar/with_ceiling"
one "$F/lidar/k65" cpu "$PY" scripts/run_capture.py "$K65/lidar/take1" --tier lidar --out "$F/lidar/k65"
one "$F/lidar/floor_only_drift_off" cpu "$PY" scripts/run_capture.py "$SAMPLE/single_scan_floor_only/1a8384c3f6" --tier lidar --no-drift --out "$F/lidar/floor_only_drift_off"
one "$F/lidar/with_ceiling_drift_off" cpu "$PY" scripts/run_capture.py "$SAMPLE/single_scan_with_ceiling/c7d28f72c6" --tier lidar --no-drift --out "$F/lidar/with_ceiling_drift_off"
# video
if [ "$VIDEO" = live ]; then
  one "$F/video/own_live" gpu "$PY" scripts/run_capture.py "$OWN_VIDEO" --tier video --out "$F/video/own_live"
  one "$F/video/sr_live" gpu "$PY" scripts/run_capture.py "$SAMPLE/single_room/c00a170fe1" --tier video --out "$F/video/sr_live"
  one "$F/video/fo_live" gpu "$PY" scripts/run_capture.py "$SAMPLE/single_scan_floor_only/1a8384c3f6" --tier video --out "$F/video/fo_live"
  one "$F/video/wc_live" gpu "$PY" scripts/run_capture.py "$SAMPLE/single_scan_with_ceiling/c7d28f72c6" --tier video --out "$F/video/wc_live"
  one "$F/video/k65" gpu "$PY" scripts/run_capture.py "$K65/video/take1.MOV" --tier video --out "$F/video/k65"
else
  VA=$O/video_auto/replay
  for n in own_2328 own_b1 own_b2 own_a1 own_a2 own_p1 own_pp1 own_p2 own_pp2 sr_r1 sr_r2 sr_r3 sr_a fo_a fo_d066 wc_a; do
    one "$F/video/$n" gpu "$PY" scripts/replan_run.py "$VA/$n" --out "$F/video/$n"
  done
  # k65: its cached run predates the 'auto' replays, so the whole front end is replayed from its caches first
  one "$F/video_replay/k65" cpu "$PY" "$O/video_better/baseline/replay.py" "$F/code" "$O/sim/k65v2/runs_iphone15/video_take1" \
      "$K65/video/take1.MOV" "$F/video_replay/k65"
  one "$F/video/k65" gpu "$PY" scripts/replan_run.py "$F/video_replay/k65" --out "$F/video/k65"
fi
env -u PYTHONPATH "$PY" "$ROOT/scripts/bench_results.py" "$F"
