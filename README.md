# Phone captures to dimensioned floor plans

This takes what a phone records (photos, a video walkthrough or a LiDAR scan) and turns it into one floor plan of
the whole home. The plan has rooms, walls, ceiling heights, floor areas, doors and windows, plus damage regions and
scope items. **Every number comes with a 95% interval.**

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

## Setup

Linux x86_64 with Python 3.11 and [uv](https://docs.astral.sh/uv/). The LiDAR tier, the tests and the scoring
scripts run on the CPU. The video and photo tiers need an NVIDIA GPU (tested on 8 GB).

Run everything from the repo root. Environments and data sit next to the repo, not in it, so the envs go in the
parent folder too: the benchmark shell scripts and the module notes use `../.venv/bin/python`. Weights go to the
usual Hugging Face and torch caches, except the segmenter's, which go to `../weights/hf_seg`.

The steps, in order:

```bash
# 1. main env
uv venv --python 3.11 ../.venv && source ../.venv/bin/activate
uv pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
#   CPU only: torch==2.14.1+cpu torchvision==0.29.1+cpu from https://download.pytorch.org/whl/cpu
uv pip install --no-deps -r requirements.txt   # the full pinned closure, so nothing else gets pulled in
# 2. weights: pinned revisions and SHA-256 (the LiDAR tier needs none)
python scripts/fetch_weights.py
# 3. video tier: the DPVO env (needs the CUDA toolkit, nvcc)
bash setup/dpvo_setup.sh
# 4. photo tier: the wall segmenter's env
bash setup/seg_env.sh
export HF_HUB_OFFLINE=1                        # everything runs offline from here
env -u PYTHONPATH python -m pytest -q tests    # 42 tests, about 20 s
```

Install times on the dev machine: the main env 55 s from a warm uv cache (`docs/COMPLIANCE.md`, D3), the DPVO env
about 5 min (`docs/DISCLOSURES.md` §5b). A cold install was not timed.

`env -u PYTHONPATH` matters if your shell sets `PYTHONPATH` (a sourced ROS install, for example): foreign pytest
plugins on that path abort the run.

**Video tier.** DPVO needs compiled CUDA ops, so it runs in its own env as a subprocess. `setup/dpvo_setup.sh`
builds DPVO @ `0ac95b6` with a small patch for torch 2.14 (`docs/DISCLOSURES.md` §5b) into `../envs/dpvo` and
`../third_party/`, where `floorplan/video/vo.py` looks. For other paths, set `FLOORPLAN_DPVO_PYTHON`,
`FLOORPLAN_DPVO_REPO` and `FLOORPLAN_DPVO_SHIMS` for both the script and the pipeline. The build targets the GPU
it was tested on (`TORCH_CUDA_ARCH_LIST=8.9`); set that variable for another GPU. The video tier also reads
rotation and focal length with `ffprobe` when it is installed.

**Photo tier, wall masks.** An ADE20K segmenter (SegFormer-B5, D-060) keeps wardrobes and cabinets from being taken
as walls. It needs `transformers`, which must stay out of the main env (`docs/ISSUES.md` I-002), so
`setup/seg_env.sh` builds a second env, `../envs/seg` (or set `FLOORPLAN_SEG_PYTHON`). Without it the photo tier
works from geometry only and says so in the log, in `run_report.json` and in `plan.json` (`meta.warnings`). The k65
result below used it.

## One command per capture

```bash
python scripts/run_capture.py <input> --tier lidar|video|photo [--out DIR]
```

| Tier | `<input>` | Example (`$DATA` = the folder with the sample captures) |
|---|---|---|
| `lidar` | Stray Scanner folder (`rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`) | `python scripts/run_capture.py $DATA/single_room/c00a170fe1 --tier lidar` |
| `video` | a video file, or a Stray Scanner folder (only its `rgb.mp4` is read) | `python scripts/run_capture.py $DATA/single_room/c00a170fe1 --tier video` |
| `photo` | a folder with one sub-folder of 2-8 photos per room | `python scripts/run_capture.py my_home/photos --tier photo` |

Output goes to `outputs/runs/<capture>/<tier>/`:

- `plan.json`: the output contract (`schema/plan.schema.json`). Every measurement is `{value, ci95, status, method}`,
  and `status` is `measured`, `inferred` or `not_observed`, never a silent guess.
- `plan.svg`, `plan.png`: the drawn plan with dimension lines; `plan.dxf`: CAD, in metres.
- `scene/` and `run_report.json`: the metric scene the plan came from, and what each stage did.

Useful flags: `--no-damage`, `--no-bias-correction` (raw LiDAR values), `--extractor alpha` (the other plan
extractor), `--no-drift` (ablation only). On the three sample captures one run took 52-238 s at the LiDAR tier
and 272-1737 s at the video tier.

## Own capture

Capture with `docs/CAPTURE_PROTOCOL.md` (one page) or the step-by-step `docs/HOUSE_CAPTURE_GUIDE.md` (with the tape
measurements), and copy the files by USB cable: messaging apps strip the EXIF data and recompress. Then:

```bash
O=path/to/capture   # photos/<room>/, video/take1.mp4, gt/ground_truth.csv (template: benchmark/ground_truth.csv)
python scripts/check_own_capture.py $O                       # seconds, no GPU: PASS / WARN / FAIL per item
python scripts/process_own_capture.py $O --out outputs/own   # every tier through run_capture.py, scored against tape
python scripts/bench_final.py --skip-lidar                   # adds the own capture to the benchmark report
```

The step-by-step version, with measured run times, is `docs/OWN_CAPTURE_RUNBOOK.md`. Tape ground truth from a hand
sketch: `scripts/sketch_plan_pdf.py`, then `scripts/read_plan_pdf.py`, then `scripts/house_gt.py`. The first two
need `reportlab` and `pypdf`: `uv pip install --no-deps -r requirements-tools.txt`.

`sim/` renders simulated homes (Isaac Sim, InteriorAgent scenes) into phone-format files with exact ground truth.
It is used to rehearse a cold run and to measure fixes (`sim/README.md`). Simulated numbers are always labelled as
simulated.

## Results so far

Sources: `docs/TECHNICAL_REPORT.md` (final benchmark on the sample captures, 4 Oct) and the simulated k65 flat.

| What | Result |
|---|---|
| LiDAR vs a Faro laser (ARKitScenes, 5 rooms, iPad) | room dimensions 8/8 within 1.5 cm; wall lengths median 0.86 cm, 84% within 1.5 cm; ceilings 3 of 4 rooms within 1.5 cm (the miss is −2.16 cm, `docs/ISSUES.md` I-004) |
| LiDAR repeatability (two captures of one flat) | **fails**: 15/87 = 17.2% of walls within max(1 cm, 0.5%) |
| LiDAR, same command twice | identical plan (70/70 walls) |
| LiDAR drift accountability | passes; drift correction is on by default |
| Video vs the LiDAR plan of the same capture | trajectory scale +2.22% on single_room, but plans **fail**: room dimensions within ±3% 2/6, 3/10, 2/6; footprint +33.2%, −33.9%, −69.0% |
| Video, same command twice | **not repeatable** on long walks: 40.91 m², then 18.95 m² (`docs/ISSUES.md` I-007) |
| Photo, simulated k65 flat (exact ground truth) | 5/5 rooms, adjacency 4/4, no overlaps; walls median error 5.6%, 58% within 8%; footprint −9.0%, so the ±8% wall and footprint gates **fail** |
| Damage | 0 confirmed detections on the 3 clean sample captures; injected stains 11/18 and cracks 9/15 found |
| Gate count (final benchmark) | LiDAR 3 pass / 3 fail, video 2 pass / 13 fail, photo 1 pass / 16 fail (photos cut from the sample videos, scored against the LiDAR plan) |

Pending: the own capture against tape (photo and video tiers), the photo repeat room, the head-to-head with a
consumer app, and the declared fix loop (`docs/FIX_LOOP.md`). My phone (OnePlus Nord) has no LiDAR, so the LiDAR
tier is measured on the sample captures and on ARKitScenes only (agreed with the recruiters).

## Where things are

| Path | What |
|---|---|
| `floorplan/` | the pipeline: `io`, `recon` (keyframes, drift, fusion), `plan` (`beta` is the default extractor, `alpha` the other), `video`, `photo`, `scale`, `uncertainty`, `damage`, `qa`, `export`, `eval`, `benchmark` |
| `scripts/` | `run_capture.py`, plus the benchmark, evaluation and own-capture scripts |
| `sim/`, `tests/`, `schema/` | test-data generator, tests, the output schema |
| `setup/` | env scripts for DPVO (with its patch and shim) and the wall segmenter, and their pins |
| `docs/DECISIONS.md` | why each choice was made (D-001 onwards) |
| `docs/ISSUES.md` | problems found, their cause and the fix |
| `docs/CAPTURE_PROTOCOL.md`, `docs/DEVICE_MATRIX.md` | how to capture, and which phone runs which tier |
| `docs/TECHNICAL_REPORT.md` | architecture, accuracy, drift, error budget, calibration |
| `docs/FAILURE_MODES.md` | mirrors, glass, glossy floors, low light |
| `docs/COMPLIANCE.md`, `docs/DISCLOSURES.md` | where each requirement is met; every model, dataset and licence |
| `docs/modules/` | one note per module: what was tried, the numbers, and the scripts that made them |

Not in git: `outputs/` (runs), model weights (`scripts/fetch_weights.py`), environments and data.

## History

The work started in a scratch folder on 3 Oct. It was moved into this repo on the night of 4 to 5 Oct, in the order it
was done, by a script, so those 67 commits are a second or two apart. The scratch folder kept only the last version of
each file, so every file went in at that version and an early commit may not import or run; only the latest commit is
tested. Later work is committed as it happens.
