# Phone captures → dimensioned, stitched floor plans (photos, video, LiDAR)

Turn what a phone records into one whole-property floor plan with walls, ceiling heights, floor areas, openings,
damage regions, concealed-damage flags and scope line items. **Every number carries a 95% interval.**

Three input tiers produce **one output contract** (`schema/plan.schema.json`):

```text
 photos/<room>/*.jpg|heic     video.mov / rgb.mp4               Stray Scanner folder (depth + poses)
 2-8 stills per room          handheld walkthrough              iPhone Pro LiDAR
        |                            |                                  |
 [photo front-end]            [video front-end]                 [LiDAR front-end]
 ALIKED+LightGlue links,      DPVO trajectory, MoGe-2 depth,    keyframes -> drift correction (pose graph)
 MoGe-2 depth, GeoCalib       scale from depth agreement +      -> TSDF fusion + raw points
 gravity, scale-cue fusion    focal metadata, GeoCalib gravity  -> floor levelling + Manhattan alignment
        \____________________________|__________________________________/
                     same metric scene format (scene.npz + scene_info.json)
                                     |
          plan extraction (rooms, walls, openings, ceiling, adjacency; fitted on raw points)
                                     |
          intervals: fit + sensor + drift (+ tier scale term for video/photo)  -> widen as data thins
                                     |
          damage regions -> concealed-damage rules -> scope line items (keyed to surface ids)
                                     |
          plan.json (schema) + plan.svg/png (rendered plan) + plan.dxf (CAD)
```

Design rule: **models propose, geometry measures** (D-004). Every reported dimension comes from planes and lines
fitted to measured points; learned models are used only where the sensor gives no metric geometry (video/photo
depth, scale, gravity, matching). The full reasoning trail is `docs/DECISIONS.md`; the system overview for the
reviewers is `docs/TECHNICAL_REPORT.md`.

---

## 1. What needs a GPU

| Part | Needs | Why |
|---|---|---|
| LiDAR tier (`--tier lidar`), drift correction, plan extraction, damage, export | **CPU only** | Open3D TSDF/ICP, NumPy/SciPy geometry, classical image processing |
| Benchmark, tests, drift ablation, ARKitScenes validation | **CPU only** | Geometry and statistics |
| Video tier (`--tier video`) | **NVIDIA GPU** (tested: RTX 2000 Ada 8 GB, sm_89) + CUDA toolkit to build DPVO | DPVO (compiled CUDA ops), MoGe-2 and GeoCalib are called with `.cuda()` |
| Photo tier (`--tier photo`) | **NVIDIA GPU** | MoGe-2 and GeoCalib on `cuda`; ALIKED/LightGlue matching |

Peak GPU memory: MoGe-2 about 2.0 GB, GeoCalib about 1.0 GB, DPVO about 0.6–0.8 GB; models are loaded one at a time
and freed, so 4 GB of free VRAM is enough (`SETUP.md` §2 measurements; `docs/modules/video_tier.md` §1).
A CPU fallback for the learned models is **not implemented** (known limitation, §9).

---

## 2. Setup (clean Linux x86_64 machine)

Prerequisites: `git`, `curl`, [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
GPU machines additionally need an NVIDIA driver that supports CUDA 13.0 (tested: driver 580) and, for the video
tier only, the CUDA 13.0 toolkit (`nvcc`) and gcc (tested: nvcc 13.0 with gcc 13.3) to compile DPVO.

```bash
git clone <this repo> floorplan && cd floorplan
uv venv --python 3.11 .venv                      # uv downloads Python 3.11 if the machine has none

# 1) torch first, from the index that matches the machine
uv pip install --python .venv/bin/python torch==2.14.1+cu130 torchvision==0.29.1+cu130 \
    --index-url https://download.pytorch.org/whl/cu130          # NVIDIA GPU
#   or, CPU-only machine (LiDAR tier, benchmark, tests):
# uv pip install --python .venv/bin/python torch==2.14.1+cpu torchvision==0.29.1+cpu \
#     --index-url https://download.pytorch.org/whl/cpu

# 2) everything else, exactly as tested (complete pinned closure, so --no-deps)
uv pip install --python .venv/bin/python --no-deps -r requirements.txt

source .venv/bin/activate
python scripts/fetch_weights.py                  # GPU machines: pinned weights (section 3), then:
export HF_HUB_OFFLINE=1                          # run fully offline from here on
python -m pytest -q tests                        # 42 tests (42 passed in 19 s, 4 Oct 03:30)
```

**Why `--no-deps`.** `requirements.txt` is the full closure of what the code imports, copied from the environment
every number was produced with (`env/freeze/main_venv.txt`). MoGe declares dependencies we never use (gradio, a CUDA
`flex-gemm` build) and hloc pulls an unpinned LightGlue; `--no-deps` keeps both out. Do **not** add `transformers`
to this env: it downgrades `huggingface-hub` below 2 and breaks the pinned model cache (`docs/ISSUES.md` I-002).

**Issues found by the clean-install test** (symptom → cause → fix):
- `ModuleNotFoundError: plotly`, then `dash`, on `import open3d` → Open3D 0.20 imports its web visualiser at import
  time → its full dependency closure is now in `requirements.txt` (first attempt had left it out as "unused").
- MoGe's declared dependencies include gradio and a CUDA build of `flex-gemm` that `moge.model.v2` never imports →
  install with `--no-deps` from a complete pinned list.
- pytest aborts with `ModuleNotFoundError: lark` from `/opt/ros/...` → a sourced ROS `PYTHONPATH` injects foreign
  pytest plugins → unset `PYTHONPATH` (below).

**If your shell sets `PYTHONPATH`** (for example a sourced ROS install), unset it for this project
(`env -u PYTHONPATH python -m pytest ...`): foreign pytest plugins on that path abort the test run. Found while timing
the clean install.

### Video tier: the DPVO env (GPU only, about 5 min)

DPVO needs compiled CUDA extensions, so it lives in its own env and is called as a subprocess
(`floorplan/video/vo.py`). Build it with the DPVO setup script (clones DPVO @ `0ac95b6`, applies
`third_party/dpvo.patch` for torch ≥ 2.14, builds `cuda_corr`, `cuda_ba` and `lietorch` for your GPU, links
`dpvo.pth`, runs a smoke test). Then point the pipeline at it:

```bash
export FLOORPLAN_DPVO_PYTHON=$PWD/envs/dpvo/bin/python
export FLOORPLAN_DPVO_REPO=$PWD/third_party/dpvo
export FLOORPLAN_DPVO_SHIMS=$PWD/third_party/dpvo_shims
```

> Scratch-folder note (checked 4 Oct 03:30): `scripts/setup_dpvo.sh` does **not** exist in this folder yet. The script
> is `../envs/dpvo_setup.sh` (next to the repo), with the parent folder's absolute path hard-coded as its
> root, and the patch and shims are in `../third_party/`. Without the `FLOORPLAN_DPVO_*`
> variables, `vo.py` defaults to that layout. Moving the script, patch and shims into the repo (root made relative) is
> commit H3 in `docs/COMMIT_PLAN.md`. Set `TORCH_CUDA_ARCH_LIST` to your GPU (the script uses `8.9`). The video tier
> also reads container metadata with the system `ffprobe` (FFmpeg) when it is installed; without it, rotation falls back
> to OpenCV and the focal length to image estimates.

### Measured install times and honest cold-install estimates

Measured on the development machine (22 CPUs, heavily loaded by other jobs), fresh venv, **warm uv cache**
(evidence: `outputs/submission_docs/`):

| Step | Time |
|---|---|
| `uv venv` | 0.03 s |
| torch + torchvision cu130 (from cache) | 2.6 s |
| `requirements.txt` (`--no-deps`, includes 5 git clones + builds) | 55 s |
| import every `floorplan.*` module | 22 s (first import, cold disk cache) |
| `pytest tests` | 31 passed, 36 s (at that time; the suite now has 42 tests, see §6) |
| `run_capture.py ... single_room --tier lidar` (drift ON, bias correction ON, damage ON) | 53 s (64 s on an earlier code state) |

Cold machine (empty cache): the time is dominated by downloads. Download sizes, measured from the package indexes:

| What | Download |
|---|---|
| torch 2.14.1+cu130 wheel | 555 MB |
| CUDA runtime wheels pulled by torch (cuDNN 553, cuBLAS 423, NCCL 216, cuSPARSELt 170, Triton 226, NVSHMEM 60, NVRTC 43 MB, plus cuFFT/cuSOLVER/cuSPARSE/cuRAND, not measured) | ≈ 1.7 GB measured + ≈ 0.5 GB estimated |
| torch 2.14.1+cpu wheel (CPU machines instead of the two rows above) | 196 MB |
| everything in `requirements.txt` (PyPI wheels; largest: opencv 50, open3d 48, pycolmap 38, scipy 33 MB) | 295 MB |
| model weights, default path (section 3) | ≈ 1.5 GB |

| Machine | Total download | Estimate at 100 Mbit/s | At 50 Mbit/s |
|---|---|---|---|
| CPU-only (LiDAR tier, benchmark, tests) | ≈ 0.5 GB | **≈ 3 min** (download ≈ 40 s + install ≈ 2 min) | ≈ 4 min |
| NVIDIA GPU, all three tiers | ≈ 4.5 GB + DPVO build | **≈ 12 min** (download ≈ 6 min + install 2 min + DPVO build ≈ 5 min, script estimate) | ≈ 18 min: over budget, pre-fetch weights |

So the "< 15 minutes" target holds for a CPU machine and for a GPU machine on a ≥ 100 Mbit/s link. The DPVO build
time is the script's own estimate, not re-timed on a cold machine. PyPI was slow on the development machine
(0.3–0.5 MB/s during setup, `SETUP.md` §2), while Hugging Face and download.pytorch.org were fast.

---

## 3. Weights (fetched by script, never committed)

`python scripts/fetch_weights.py [--experiments] [--dry-run]` downloads every weight at a **pinned revision**
(Hugging Face commit hash) or a **pinned SHA-256** (GitHub release files) into the standard caches (`HF_HOME`,
`torch.hub`), where the libraries look for them. After that the pipeline runs with `HF_HUB_OFFLINE=1`.

| Weights | Used by | Size | Licence |
|---|---|---|---|
| MoGe-2 ViT-L normal (`Ruicheng/moge-2-vitl-normal` @ cb0e8bbd, `model.pt`) | video + photo depth | 1.32 GB | MIT |
| GeoCalib pinhole (`pinhole.tar`, SHA-256 pinned) | video + photo gravity/focal | 111 MiB | CC-BY-4.0 (attribution) |
| LightGlue for ALIKED (v0.1_arxiv) | video + photo matching | 47.6 MB | Apache-2.0 |
| ALIKED-n16 | video + photo features | 2.7 MB | BSD-3 |
| DPVO (`pablovela5620/dpvo` @ c998d3b5, `dpvo.pth`) | video trajectory | 14 MB | MIT repo, no separate weights licence |
| *`--experiments` only:* DA3METRIC-LARGE, DA3-BASE, `facebook/map-anything-apache` | option comparisons, never the default path | 1.34 + 0.54 + 4.91 GB | Apache-2.0 |

The LiDAR tier needs **no weights**. Sizes: `SETUP.md` §4. Full licence review: `docs/DISCLOSURES.md`.

---

## 4. One command per capture

```bash
python scripts/run_capture.py <input> --tier lidar|video|photo [--out DIR]
```

| Tier | `<input>` | Example (`$DATA` = the folder holding the provided sample captures) |
|---|---|---|
| `lidar` | a Stray Scanner folder (`rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`) | `python scripts/run_capture.py $DATA/single_room/c00a170fe1 --tier lidar` |
| `video` | a video file (`.mov`/`.mp4`), or a Stray Scanner folder (only its `rgb.mp4` is read) | `python scripts/run_capture.py $DATA/single_room/c00a170fe1 --tier video` |
| `photo` | a folder with one sub-folder of 2–8 photos per room | `python scripts/run_capture.py captures/home/photos --tier photo` |

Options (checked against `scripts/run_capture.py`, 4 Oct 06:45):

| Flag | Default | What it does |
|---|---|---|
| `--out DIR` | `outputs/runs/<name>/<tier>/` | output folder |
| `--extractor beta\|alpha` | `beta` | plan extractor (beta chosen by the judge, D-018) |
| `--no-drift` | drift ON | LiDAR: use ARKit poses as-is. **Ablation only**: "poses used as-is" fails the drift gate (D-019) |
| `--no-bias-correction` | correction ON | LiDAR: report raw values; the interval is still widened one-sided by the uncorrected bias (D-021, D-027). Also `FLOORPLAN_LIDAR_BIAS=off` |
| `--no-damage` | damage ON | skip damage, rules and scope |
| `--config FILE` | built-in | override pipeline config |

The photo tier writes its working files (features, SfM, depth cache) to `photo_work/` **next to** the input folder.

Runtimes of the one command (all with damage ON; final benchmark, final code, 4 Oct 05:08–06:28,
`outputs/benchmark/final/REPORT.md` §4):

| Tier | single_room | floor_only | with_ceiling |
|---|---|---|---|
| LiDAR (wall time / peak RSS) | 52 s / 2.0 GB | 146 s / 3.3 GB | 238 s / 4.9 GB |
| Video (`rgb.mp4` of the same captures) | 272 s / 5.9 GB | 794 s / 5.9 GB | 1737 s / 7.6 GB |
| Photo (simulated spin folders; depth + gravity **loaded from cache**, so not cold-start) | 9 s / 1.4 GB | 37 s / 1.7 GB | 45 s / 1.9 GB |

A cold photo run is slower: 77 s for 46 phone-format photos in 9 folders
(`outputs/submission_docs/photo_e2e_phone_like/`).

---

## 5. Outputs

Default folder: `outputs/runs/<capture>/<tier>/`.

| File | What it is |
|---|---|
| `plan.json` | The output contract (`schema/plan.schema.json` v1.0.0): rooms, walls, openings, adjacency, footprint, damage, scope items, meta. Every measurement is `{value, ci95: [lo, hi], status, method}`; `status` is `measured`, `inferred` or `not_observed` (never a silent guess). The plan reports on itself in `meta`: `reliability` (`"low: ..."` when video segments were joined without verified geometry, D-034), `floor_fallback` (how a missing floor was estimated, D-032), `dropped_openings` (impossible openings removed, D-033), `coverage_warning`. |
| `plan.svg`, `plan.png` | The rendered plan a homeowner would recognise: rooms, walls, doors with swings, windows, dimension lines with intervals. |
| `plan.dxf` | CAD export in metres with real DIMENSION entities. |
| `scene/` | The aligned metric 3D scene (`scene.npz`, `scene_info.json`) the plan was extracted from, for audit and re-runs. |
| `run_report.json` | Config, per-stage reports (drift loops, alignment, scale), timings and schema-validation problems. |

Coordinates (`docs/ISSUES.md` I-001): plan coordinates are (u, v) = (x, z) of the gravity-aligned world in metres;
viewed from above, v points **down** the page. The renderer and DXF apply that flip, so the drawings are not mirrored.
Lengths, areas and adjacency do not depend on it.

Redraw any plan: `python scripts/render_plan.py <plan.json> [--out DIR] [--strict] [--dxf-preview]`.

---

## 6. Benchmark, tests, ablations (reproduce every reported number)

```bash
python -m pytest -q tests                                    # 42 tests: export, eval, benchmark scorer, stability, damage precision

# canonical LiDAR scenes (drift ON, D-011 intrinsics, per-point source frame), then the benchmark
python scripts/build_canonical_scenes.py                     # -> outputs/scenes_v2/<capture>/ (~2 min, CPU)
python scripts/register_scenes.py --scenes 'outputs/scenes_v2/*/scene.npz' --out outputs/eval/registration_v2
python scripts/benchmark.py [--gt-dir GT/]                   # -> outputs/benchmark/{report.md,gates.csv,...}

# drift accountability: on/off ablation with variants and held-out loop residuals (4-6 min per capture)
python scripts/drift_ablation.py <stray_capture_dir> [--sweep]   # -> outputs/drift/<capture>/

# LiDAR absolute accuracy against a Faro laser scan (ARKitScenes; download commands in data/arkitscenes/README.md)
python scripts/validate_arkitscenes.py --scene <raw/Training/47895909> --laser <faro.ply>

# own captures vs tape ground truth (gates, calibration, head-to-head table)
python scripts/eval_own_capture.py --gt benchmark/ground_truth.csv --plan video=<plan.json> --plan photo=<plan.json> [--app <export>]

# FINAL BENCHMARK (Deliverable 5): the one command at all three tiers on the 3 sample captures, current code
#   (sequential; about 80 min on one 8 GB GPU; details and the exact chain: docs/modules/benchmark_final.md §9.1)
scripts/bench_final_runs.sh video "" final               # -> outputs/benchmark/final/video/<capture>/final/
scripts/bench_final_runs.sh video floor_only final_rerun  # same command again (run-to-run check, I-007)
scripts/bench_final_runs.sh photo v3                      # -> photo/<capture>/v3/ (simulated spin folders)
scripts/bench_final_runs.sh lidar && scripts/bench_final_runs.sh lidar_rerun && scripts/bench_final_runs.sh lidar_off
env -u PYTHONPATH python scripts/bench_final_arkit.py    # LiDAR vs Faro laser on 5 ARKitScenes rooms
env -u PYTHONPATH python scripts/bench_final.py          # -> outputs/benchmark/final/{REPORT.md,results.json}
#   bench_final.py only scores: safe to re-run at any time; --skip-lidar reuses the LiDAR scores

# earlier LiDAR-only gates (01:45-03:10 code state): docs/modules/benchmark_lidar_sample.md
scripts/bench_lidar_runs.sh && scripts/bench_lidar_retime.sh
env -u PYTHONPATH python scripts/bench_lidar_eval.py && env -u PYTHONPATH python scripts/bench_lidar_rerun.py

# whole-plan accuracy against the laser on 5 ARKitScenes rooms (drift on and off)
env -u PYTHONPATH python scripts/bench_arkitscenes.py all     # -> outputs/benchmark/arkitscenes/

# extractor judge (plan_alpha vs plan_beta) and the fix-loop before/after harness
#   judge: the job lists and commands are in docs/modules/judge_plan_extractor.md, "How to reproduce everything"
python scripts/plan_v2_eval.py register && python scripts/plan_v2_eval.py score v1 v2
#   fix loop (Part 4): template, candidates, ranking rule and before/after commands in docs/FIX_LOOP.md
```

Every module document in `docs/modules/` ends with the exact commands that produced its numbers.

---

## 7. Capturing your own space

Follow **`docs/CAPTURE_PROTOCOL.md`** (one page, Route 2: Stray Scanner for LiDAR, the stock Camera app for photos
and video; no paper sheet, marker or other reference object is needed, D-067). Copy files
by cable or AirDrop, never through messaging apps (they strip EXIF and recompress). Which tier runs on which phone,
and with what accuracy: `docs/DEVICE_MATRIX.md`. The step-by-step version for the candidate's home (photos, video,
tape ground truth, staged decals, consumer app) is `TakeHome/HOUSE_CAPTURE_GUIDE.md`.

### 7.1 Morning workflow: own capture → plans → scores (one command)

`scripts/process_own_capture.py` runs every own-capture plan through the one command and scores it against the tape.
Step by step, with every checker code and the measured run times: **`docs/OWN_CAPTURE_RUNBOOK.md`**.

**1. Lay out the files** (by USB cable; never WhatsApp/Telegram/email, which strip EXIF and recompress):

```text
TakeHome/OwnCaptures/
  photos/01_<room>/*.jpg ...        one folder per room, 2-8 photos, slate photos REMOVED (keep them in gt/slates/)
  photos/03_<room>_take2/           the repeat take of room 03_<room> (name = the room folder + "_take2")
  video/take1.mp4  take2.mp4  [lowlight.mp4]
  gt/ground_truth.csv               template: benchmark/ground_truth.csv (walls W1.. clockwise from the main door)
  app_export/app_dimensions.csv     optional, for Part 3: room_id,item_type,item_id,value_m (same ids as the GT)
```

**2. Quick checks before the long runs** (2 min). Photos keep their EXIF time and focal; videos are one lens, no
stabiliser crop:

```bash
source .venv/bin/activate && export HF_HUB_OFFLINE=1      # and run python via `env -u PYTHONPATH` if your shell sets it
O=../TakeHome/OwnCaptures
env -u PYTHONPATH python -c "import sys; sys.path.insert(0,'.'); from floorplan.photo.images import load_all; \
  ps=load_all(sys.argv[1]+'/photos'); print(len(ps),'photos;',sum(p.f35 is not None for p in ps),'with focal;', \
  sum(p.time_s is not None for p in ps),'with capture time')" $O     # uses the pipeline's own EXIF reader
#   every photo needs a capture time (doorway pairs, spin priors) and a focal length (scale; without it σ is 9.7%)
ffprobe -v error -show_entries stream=width,height,r_frame_rate,codec_name:stream_side_data=rotation $O/video/take1.mp4
#   expect 1920x1080 (or 1080x1920 with rotation), 30 or 60 fps; one stream, no lens switch
```

**3. Smoke-test one tier on one room first** (about 1–2 min; catches format problems before the long runs):

```bash
mkdir -p /tmp/one_room && cp -r $O/photos/01_* /tmp/one_room/
env -u PYTHONPATH python scripts/run_capture.py /tmp/one_room --tier photo --out outputs/own/smoke_photo
```

**4. Run everything, score it, refresh the benchmark report** (two commands, in this order):

```bash
env -u PYTHONPATH python scripts/process_own_capture.py $O --out outputs/own
#   photo tier on all room folders (not *_take2)        -> outputs/own/photo/
#   photo tier on the repeat room, take 1 and take 2     -> outputs/own/photo_repeat_take1|take2/
#   video tier on every clip in video/                   -> outputs/own/video_<name>/
#   tape scoring + head-to-head                          -> outputs/own/eval/own_eval.{md,json}
env -u PYTHONPATH python scripts/bench_final.py --skip-lidar
#   -> outputs/benchmark/final/REPORT.md: §6 own capture vs tape, §2 photo repeat room, §3 head-to-head
#      (--skip-lidar reuses the LiDAR scores; the sample-capture runs are not repeated)
env -u PYTHONPATH python scripts/process_own_capture.py $O --out outputs/own --skip-runs   # re-score only
```

Runs are sequential (the 8 GB GPU cannot hold two video runs: `docs/modules/video_tier.md` V2-I6). Measured on the
sample: photo 9–45 s per capture with a warm depth cache, 77 s cold for 46 photos; video 272 s for the 37 s
single_room clip and 1737 s for the longest walkthrough (`outputs/benchmark/final/REPORT.md` §4). Start the script as
soon as the files are copied, and expect the video runs to dominate. Each run's log is `outputs/own/<run>.log`.
The video tier is not run-to-run repeatable on long walks (`docs/ISSUES.md` I-007), so run a take twice before
reading anything into a small difference.

**5. What to look at, in this order:**

1. `outputs/own/eval/own_eval.md`: gates, interval coverage, head-to-head.
2. Each `plan.png`.
3. In each `plan.json` / `run_report.json`: `meta.reliability` (D-034), `meta.floor_fallback` (D-032),
   `meta.dropped_openings` (D-033), `scale_sigma_rel`, `whole_scene_consistent` (video), `meta.coverage_warning` and
   `meta.damage.confirmed/review`.
4. `outputs/benchmark/final/REPORT.md` §6.

Known risk: photos without EXIF capture times get no spin layout. On phone-format copies without times, 4 of 9
folders became rooms (`docs/modules/photo_tier.md` v3.5). With times, every simulated folder became a room. If rooms
are missing on the real photos, their scores are missing rather than wrong; report it as such.

**6. Fix loop (Part 4).** Rank the gates in `own_eval.json` by relative shortfall from the threshold, declare the
worst, then fix, tag and run before/after. The exact commands are in `docs/FIX_LOOP.md` Part 3.

---

## 8. Repository map

| Path | What |
|---|---|
| `floorplan/io`, `recon`, `pipeline`, `plan` | Capture loading, keyframes, drift correction, TSDF, alignment, plan extractors (`plan/beta` default, `plan/alpha` alternative) |
| `floorplan/video`, `photo`, `scale` | Video and photo front-ends; scale-cue fusion (the paper-sheet detector is optional, off by default: D-067) |
| `floorplan/uncertainty` | Measured LiDAR noise model, tier scale term |
| `floorplan/damage`, `qa` | Damage regions, concealed-damage rules, scope; difficult-surface detectors (mirror, glass, glossy, low light) |
| `floorplan/export`, `eval`, `benchmark` | JSON/SVG/DXF; registration, matching, repeatability, gates; scoring against tape ground truth |
| `docs/` | `DECISIONS.md`, `ISSUES.md`, `COMPLIANCE.md`, `TECHNICAL_REPORT.md`, `DISCLOSURES.md`, `FAILURE_MODES.md`, `modules/*.md` |

---

## 9. Known limitations (honest list; details and evidence in `docs/TECHNICAL_REPORT.md`)

- **Repeatability gate fails.** Two LiDAR captures of the same apartment agree within 1 cm / 0.5% on
  15/87 = 17.2% of walls through the one command (`outputs/benchmark/final/REPORT.md` §1).
  - Walls whose corners are built the same way agree to 1.8 cm median.
  - Most failures are topological: 34 walls are unmatched, and the extractor is still raster-phase sensitive.
  - At the LiDAR tier the same command on the same input gives the identical plan (D-029, D-030).
- **Final gate counts** (REPORT §0): LiDAR 3 PASS / 3 FAIL, video 2 / 13, photo 1 / 16. Video and photo are scored
  against the LiDAR plan of the same capture until the own capture adds tape ground truth.
- **LiDAR absolute accuracy.** Apple LiDAR surfaces sit 0.68 cm into the room versus a laser; this is corrected
  (D-021, corner-aware D-027) and its uncertainty is carried (±1.6 cm, ±2.1 cm for an unvalidated device). Against the
  laser (ARKitScenes, iPad only, final code): room dimensions 8/8 within 1.5 cm, wall lengths 84% within 1.5 cm
  (median 0.86 cm), ceilings 3 of 4 rooms within 1.5 cm. In the failing room, which has no revisit loop, drift
  correction spreads the ceiling to −2.16 cm (`docs/ISSUES.md` I-004). The fifth room produced no plan.
- **Video tier.** Trajectories are good: single_room +2.2% scale error, the floor_only trusted path +2.9% (v1: +187%).
  Plans are not.
  - Footprint against the LiDAR plan: +33.2% / −33.9% / −69.0%.
  - Room dimensions within ±3%: 2/6, 3/10 and 2/6, so the ±3% gate fails.
  - The causes are a door leak and missing rooms (REPORT §1).
  - Joins across tracking restarts are not verified, so those plans carry `meta.reliability = "low"` and a 20% σ floor
    (D-034). Interval coverage is 60–88%.
  - **Not repeatable:** the same command gave 40.91 then 18.95 m² on floor_only (pycolmap verification and DPVO are
    not deterministic; `docs/ISSUES.md` I-007, `docs/modules/benchmark_final.md` BF-12).
- **Photo tier.** Every spin folder becomes a room (photo v3), and doorway pairs join rooms with 0 false links. On
  simulated folders, though:
  - the plan is in 2–4 pieces;
  - room dimensions within 8% are 0/6, 3/12 and 0/14;
  - the footprint is +16.5% / −41.0% / −26.9% (REPORT §1).

  The simulated spins never see whole walls (photo_tier PT-16). Native photos against the tape are the real test.
- **No CPU path for the learned models**; the video and photo tiers need an NVIDIA GPU. DPVO paths default to the
  development machine's layout unless the `FLOORPLAN_DPVO_*` variables are set.
- **Ceiling height** is only observable when the capture sweeps the ceiling (protocol step); otherwise it is reported
  `not_observed`, never guessed.
- **Damage:** water stains and cracks only. Detection is reported in two tiers: only "confirmed" detections (stain
  ≥ 0.35, crack ≥ 0.65) create scope items and concealed flags (D-023, D-026). On the three clean sample captures the
  one command reports 0 confirmed detections and 0 scope items, after fixing a false stain (a jar lid on a shelf,
  D-031) and a false crack (a line on a wardrobe front, D-024..D-026). Straight, axis-aligned hairline cracks and
  stains on objects flush with the wall (< 1.5 cm) are known blind spots (`docs/modules/damage.md` v3.7).
- **Ground truth:** the LiDAR tier has laser ground truth only through ARKitScenes (5 rooms, iPad); own-capture tape ground
  truth covers the photo and video tiers. LiDAR on the candidate's own rooms is impossible without an iPhone Pro
  (recruiter-approved deviation, `docs/OPEN_QUESTIONS.md`).
- `scripts/build_canonical_scenes.py` resolves its arguments against a fixed sample-data root (development machine);
  use `run_capture.py` (which takes any folder) on other machines until that is parameterised.

Licences and every pretrained model, dataset and service used: `docs/DISCLOSURES.md`.
