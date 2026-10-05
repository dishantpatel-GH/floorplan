# Phone captures to dimensioned floor plans

This takes what a phone records (photos, a video walkthrough or a LiDAR scan) and turns it into one floor plan of
the whole home: rooms, walls, ceiling heights, floor areas, doors and windows, plus damage regions and scope items.
**Every number comes with a 95% interval**, and with a status (`measured`, `inferred` or `not_observed`), never a
silent guess.

## Three tiers, one output

| Tier | Input | Where the metric geometry comes from |
|---|---|---|
| `lidar` | Stray Scanner folder from an iPhone or iPad Pro | LiDAR depth; ARKit poses with drift corrected by a pose graph over LiDAR fragments |
| `video` | walkthrough `.mov` / `.mp4` from any phone | DPVO trajectory, MoGe-2 depth, scale from depth agreement and the focal length in the file |
| `photo` | one folder of 2-8 photos per room | MoGe-2 depth and GeoCalib gravity per photo, a spin per room, doorway pairs to join rooms |

All three tiers write the same scene format, so everything after it is shared:

```text
lidar | video | photo front-end -> scene.npz -> rooms, walls, openings, ceilings (fitted on raw points)
  -> 95% intervals (fit + sensor + drift + tier scale) -> damage and scope -> plan.json, plan.svg/png, plan.dxf
```

The rule throughout: models propose, geometry measures. Every reported length comes from lines and planes fitted
to measured points (D-004). Learned models are used only where the sensor gives no metric geometry.

## Quick start on a clean machine

Linux x86_64 with [uv](https://docs.astral.sh/uv/), git and curl. The LiDAR tier, the tests and the scoring run on
the CPU; the video and photo tiers need an NVIDIA GPU (tested on 8 GB), and the video tier also the CUDA toolkit
(`nvcc`, for DPVO's ops).

```bash
git clone <this repo> floorplan-capture && cd floorplan-capture
bash setup/install.sh            # envs, weights and tests; prints the time of each step (--cpu-only: LiDAR tier only)
python3 scripts/fetch_data.py    # own house + k65 into data/, checked by SHA-256; the case study's sample captures: your copy in data/sample/
source .venv/bin/activate && export HF_HUB_OFFLINE=1     # runs need no network from here
python scripts/run_capture.py data/sample/single_room/c00a170fe1 --tier lidar
```

Everything is created inside the repo folder and is git-ignored: `.venv/` (main env), `envs/dpvo/` and
`envs/seg/` (the DPVO and segmenter envs), `third_party/` (DPVO source), `weights/` (models at pinned revisions and
SHA-256), `data/` (captures) and `outputs/` (runs). `setup/install.sh` takes `--no-video` and `--no-seg` to skip an
env. Each env script also runs alone (`setup/dpvo_setup.sh`, `setup/seg_env.sh`, `scripts/fetch_weights.py`), and
`FLOORPLAN_DPVO_PYTHON`, `FLOORPLAN_DPVO_REPO`, `FLOORPLAN_DPVO_SHIMS`, `FLOORPLAN_SEG_PYTHON`, `HF_HOME` and
`TORCH_HOME` point the code at envs and weights kept elsewhere (`floorplan/paths.py`).

Notes:
- DPVO is built for the GPU it was tested on (`TORCH_CUDA_ARCH_LIST=8.9`); set that variable for another GPU. Without
  `nvcc`, `install.sh` skips the DPVO env and says so. The video tier reads rotation and focal length with `ffprobe`
  when it is installed.
- The segmenter (SegFormer-B5, D-060) keeps wardrobes and cabinets from being taken as walls in the photo tier. It
  needs `transformers`, which must stay out of the main env (`docs/ISSUES.md` I-002), hence its own env. Without it
  the photo tier works from geometry only and says so in the log, `run_report.json` and `plan.json`.
- `env -u PYTHONPATH` in front of a command helps if your shell sets `PYTHONPATH` (a sourced ROS install, for
  example): foreign packages on that path break the envs and pytest.

## One command per capture

```bash
python scripts/run_capture.py <input> --tier lidar|video|photo [--out DIR]
```

| Tier | `<input>` | Example |
|---|---|---|
| `lidar` | Stray Scanner folder (`rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`) | `python scripts/run_capture.py data/sample/single_room/c00a170fe1 --tier lidar` |
| `video` | a video file, or a Stray Scanner folder (only its `rgb.mp4` is read) | `python scripts/run_capture.py data/sample/single_room/c00a170fe1 --tier video` |
| `photo` | a folder with one sub-folder of 2-8 photos per room | `python scripts/run_capture.py data/<capture>/photos --tier photo` |

Output goes to `outputs/runs/<name>/<tier>/` (the run prints it):

- `plan.json`: the output contract (`schema/plan.schema.json`). Every measurement is `{value, ci95, status, method}`.
- `plan.png`, `plan.svg`: the drawing with dimension lines; `plan_presentation.png`: a clean version to show a
  homeowner; `plan.dxf`: CAD, in metres.
- `scene/` and `run_report.json`: the metric scene the plan came from, and what each stage did and how long it took.

Useful flags: `--no-damage`, `--no-bias-correction` (raw LiDAR values), `--extractor alpha` (the other plan
extractor), `--no-drift` (ablation only); `python scripts/run_capture.py --help` lists the rest.

## Run on your own capture

1. Capture with `docs/CAPTURE_PROTOCOL.md` (one page; the step-by-step version with tape measurements is
   `docs/HOUSE_CAPTURE_GUIDE.md`). Use the 1x lens, and copy the files by USB cable: messaging apps strip the camera
   data the pipeline reads.
2. Put the files in `data/<capture>/` (any name; details in `data/README.md`):
   ```text
   data/<capture>/photos/<room>/*.jpg    one folder per room, 2-8 photos; a doorway photo goes in the room it looks into
   data/<capture>/video/<name>.mp4       one walkthrough of the whole home
   data/<capture>/lidar/<scan>/          the Stray Scanner folder, as exported
   data/<capture>/gt/ground_truth.csv    optional tape measurements (template: benchmark/ground_truth.csv)
   ```
3. Check and run (from the repo root, env active):
   ```bash
   python scripts/check_own_capture.py data/<capture>                  # seconds, no GPU: PASS / WARN / FAIL per item
   python scripts/run_capture.py data/<capture>/photos --tier photo
   python scripts/run_capture.py data/<capture>/video/<name>.mp4 --tier video
   python scripts/run_capture.py data/<capture>/lidar/<scan> --tier lidar
   python scripts/process_own_capture.py data/<capture> --out outputs/<capture>   # all tiers, scored against gt/
   ```
4. The plan is `plan.png` (drawing) and `plan.json` (numbers) in the folder each run prints, under `outputs/runs/`.

The full walk-through with run times is `docs/OWN_CAPTURE_RUNBOOK.md`.

## Repo layout

| Folder | What |
|---|---|
| `floorplan/` | the pipeline: `io`, `recon` (keyframes, drift, fusion), `plan` (`beta` is the default extractor), `video`, `photo`, `openings`, `scale`, `uncertainty`, `damage`, `qa`, `export`, `eval`, `benchmark`; `paths.py` says where envs and weights are |
| `scripts/` | `run_capture.py` (one command per capture), the data and weight fetchers, the benchmark (`bench_*`), evaluation and own-capture scripts |
| `setup/` | `install.sh`, and the DPVO and segmenter env scripts with their pins and patch |
| `data/` | captures (fetched, git-ignored) and their ground truth (in git); `data/README.md` |
| `results/` | the benchmark numbers and plan pictures from the fresh runs; `results/README.md` |
| `docs/` | reports, decisions, issues, capture protocol, one note per module; `docs/README.md` |
| `tests/` | unit and regression tests (CPU) |
| `sim/` | simulated homes (Isaac Sim) rendered into phone-format files with exact ground truth (`sim/README.md`) |
| `schema/`, `benchmark/`, `env/` | the plan's JSON schema; the tape ground-truth template; the torch and numpy pins of the main env |

## Results

The numbers and plans of the final benchmark are in `results/README.md`, written by the `scripts/bench_*` scripts
from fresh runs, and discussed in `docs/BENCHMARK_REPORT.md` (gates at all three tiers, repeatability, head-to-head
with CubiCasa, timing) and `docs/TECHNICAL_REPORT.md` (architecture, drift, error budget, calibration, fix loop,
failure modes). The photo tier is gated on the own house (tape) and the simulated k65 flat (exact ground truth); the
sample captures run the LiDAR and video tiers. Simulated numbers are always labelled as simulated.

## Documents

`docs/README.md` maps every document. The ones to start with: `docs/COMPLIANCE.md` (each requirement of the brief,
where it is met and its status), `docs/DECISIONS.md` (why each choice was made), `docs/ISSUES.md` (problems found,
cause and fix), `docs/FIX_LOOP.md`, `docs/DISCLOSURES.md` (every model, dataset, app and licence).

## Tests

```bash
env -u PYTHONPATH python -m pytest -q tests      # CPU, well under a minute
```

## The brief's deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Compliance matrix | `docs/COMPLIANCE.md` |
| 2 | Capture route, device matrix | `docs/CAPTURE_PROTOCOL.md`, `docs/DEVICE_MATRIX.md` |
| 3 | README, under 15 minutes to a fresh capture | this file, `setup/install.sh`, `scripts/fetch_data.py` |
| 4 | Reproduction bundle | `results/README.md` (one command per number), `scripts/bench_*`, `data/` |
| 5 | Benchmark report | `docs/BENCHMARK_REPORT.md` |
| 6 | Fix loop bundle | `docs/FIX_LOOP.md`, `docs/fixloop.diff`, tags `before-fix` and `after-fix` |
| 7 | Technical report | `docs/TECHNICAL_REPORT.md` |
| 8 | Raw benchmark data | `data/` (`scripts/fetch_data.py`, `data/MANIFEST.json`), `data/cubicasa/` |

## History

The work started in a scratch folder on 3 Oct. It was moved into this repo on the night of 4 to 5 Oct, in the order
it was done, by a script, so those commits are a second or two apart. The scratch folder kept only the last version of
each file, so every file went in at that version and an early commit may not import or run; only the latest commit is
tested. Later work was committed as it happened. On the evening of 5 Oct the history was condensed so that each
commit is one solved step: every commit keeps the exact files and the original time of the last step it holds, and the
fix-loop tags `before-fix` and `after-fix` point at the same files as before (`docs/FIX_LOOP.md`).
