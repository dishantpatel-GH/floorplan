# Disclosures: pretrained models, datasets, libraries, services

The brief allows "any pretrained model, dataset or API with disclosure, everything runs without calling your
infrastructure". This page lists **everything** the pipeline uses, where it is used, its licence, and whether
commercial use needs care. It was built by inspecting the imports of `floorplan/**`, `scripts/**` and `tests/**`
(not from memory), the licence files of the cloned repos in `third_party/`, the wheel metadata of a fresh install
from `requirements.txt`, and `SETUP.md` §3–5.

Flags: **OK** = permissive, commercial use fine (keep notices). **ATTR** = commercial use fine with attribution.
**CHECK** = unclear or conditional; confirm before commercial use. **NC** = non-commercial / research-only.
**EVAL-ONLY** = copyleft or restricted; used only to measure, never shipped or imported by the product code.

---

## 1. Pretrained models on the default path

| Model (pinned) | Where it is used | Code licence | Weights licence | Flag |
|---|---|---|---|---|
| **MoGe-2 ViT-L normal** (`Ruicheng/moge-2-vitl-normal` @ `cb0e8bbd`, `model.pt`) | Metric depth + normals per image: video tier (`floorplan/video/depth.py`), photo tier (`floorplan/photo/depth.py`), hard-frame analysis (`scripts/analyze_failure_modes.py`) | MIT; its bundled DINOv2 module Apache-2.0 | MIT | OK |
| **GeoCalib pinhole** (`pinhole.tar`, SHA-256 pinned) | Gravity (roll/pitch) and focal prior: video upright/flip check and gravity (`floorplan/video/orientation.py`), photo levelling (`floorplan/photo/views.py`) | Apache-2.0 | **CC-BY-4.0** (GeoCalib README; credits the Laval Indoor HDR dataset) | ATTR |
| **ALIKED-n16** | Keypoints for SfM / photo linking (`floorplan/video/sfm.py`, `floorplan/photo/sfm.py` through hloc) | BSD-3 | BSD-3 | OK |
| **LightGlue for ALIKED** (v0.1_arxiv) | Feature matching (same files) | Apache-2.0 | Apache-2.0 | OK |
| **DPVO** (`dpvo.pth` from the HF mirror `pablovela5620/dpvo` @ `c998d3b5`; SHA-256 cross-checked against a second mirror of the official `models.zip`) | Up-to-scale camera trajectory for the video tier (`floorplan/video/dpvo_runner.py`, own env) | MIT (lietorch BSD-3, Eigen MPL-2) | **No separate weights licence**; ships with the MIT repo; trained on synthetic TartanAir | CHECK (probably fine; ask the authors before commercial use) |
| **SegFormer-B5, ADE20K 640** (`nvidia/segformer-b5-finetuned-ade-640-640` @ `739f5d46`) | Semantic wall masks for the photo tier's room boxes (D-060): `scripts/seg_walls.py`, run in its own env `envs/seg` by `floorplan/photo/semantic.py`; without it the photo tier falls back to geometry only and says so | NVIDIA Source Code License for SegFormer: "non-commercially means for research or evaluation purposes only" (https://github.com/NVlabs/SegFormer/blob/master/LICENSE) | Same (Hugging Face card: `license: other`); trained on ADE20K (MIT CSAIL research terms) | **NC**: fine for this evaluation; a product must swap in a permissively licensed segmenter |

Deliberately **not** used, because their weights are non-commercial: SuperPoint / SuperGlue (we use ALIKED +
LightGlue instead), `facebook/map-anything` (CC-BY-NC; we use the Apache variant), Depth Anything 3
GIANT / LARGE / NESTED (CC-BY-NC). `scripts/fetch_weights.py` documents this next to the pins.

## 2. Pretrained models used only in option comparisons (not on the default path)

| Model | Where | Licence | Flag |
|---|---|---|---|
| Depth Anything 3: `DA3METRIC-LARGE`, `DA3-BASE` | `floorplan/video/experiments.py`, optional `--depth-model da3metric` in `floorplan/video/depth.py` | Apache-2.0 (these two checkpoints only) | OK |
| MapAnything, `facebook/map-anything-apache` | `floorplan/photo/recon.py`, `floorplan/photo/intra.py` (experiment, OFF by default), `floorplan/video/experiments.py` | Apache-2.0 | OK |
| DINOv2 torch-hub code (no weights; MapAnything builds its backbone skeleton from it) | MapAnything only | Apache-2.0 | OK |

Fetched only with `python scripts/fetch_weights.py --experiments`.

## 3. Evaluated during setup, NOT used by the submitted pipeline

These were installed and smoke-tested while choosing the stack (`SETUP.md` §3). None is imported by `floorplan/` or
`scripts/`. They are listed because the decision log mentions them (D-004) and the defense may ask.

| Component | Why it was considered | Licence issue | Flag |
|---|---|---|---|
| RoomFormer | Learned room polygons from a density map | MIT code; weights trained on **Structured3D** (terms require a signed agreement) | NC (treat as research-only) |
| Raster2Seq | Learned rooms + doors | MIT tag on HF, but trained on **Structured3D** | NC (treat as research-only) |
| PolyLayout, PixCuboid | Posed images → room polygon | Apache-2.0 code; weights trained on **ScanNet++** (research-only terms) | NC |
| DeepLSD | Line segments + vanishing points (idea for the paper-sheet detector, an option that is off by default; `docs/modules/scale_reference.md` §6) | MIT, but its `pytlsd` dependency bundles the classic LSD core (`src/lsd.cpp`) under **AGPL-3.0** | CHECK (AGPL obligations if ever shipped) |
| Cloud2BIM | Point cloud → IFC | MIT | OK (unused) |
| gtsam, ifcopenshell, laspy, pye57, rerun | General tooling in the dev env | BSD / LGPL / Apache | OK (unused) |

Decision D-004 explains why learned polygon models are not the source of dimensions: their benchmarks score corners
to about 10 px on a 256 px map (decimetres), and their training data is research-only.

## 4. Datasets

| Dataset | How we use it | Licence | Flag |
|---|---|---|---|
| **Provided sample captures** (3 Stray Scanner recordings: `single_room`, `single_scan_floor_only`, `single_scan_with_ceiling`) | LiDAR-tier development and benchmark; video and photo tiers derived from their `rgb.mp4` for cross-tier checks (D-014) | Provided by the company for this assessment | Not redistributed |
| **ARKitScenes** (Apple), 5 rooms with Faro laser scans: 47895909 (scan 191738, first validation room) plus 4 rooms fetched on 4 Oct for the bias study: 42445884 (visit 422009), 47331133 (visit 470348), 47430003 (visit 470537, Validation split), 47333561 (visit 469650) | **Evaluation only**: absolute accuracy of the LiDAR tier against a laser (`scripts/validate_arkitscenes.py`, `scripts/investigate_lidar_bias.py`, `scripts/bench_arkitscenes.py`, `floorplan/io/arkitscenes.py`). **One number was fitted on it**: the LiDAR inward offset b = 0.68 cm (D-021), on the 3 development rooms 47895909, 42445884, 47331133, pre-registered before the 2 hold-out rooms were analysed (`docs/modules/lidar_bias.md`). No model was trained on it. | Apple licence: personal, **non-commercial** by default; its commercial grant applies only to licensees under a 700 M monthly-active-user threshold (see `third_party/ARKitScenes/LICENSE`) | NC / EVAL-ONLY (plus one calibration constant: CHECK, §7) |
| **Phone-format copies of the sample** (`outputs/phone_like/`): 16:9 30 fps H.264/HEVC re-encodes of `single_room/rgb.mp4` with and without rotation metadata; photo folders with EXIF Orientation 6, without EXIF, and as HEIC | Robustness tests of the video and photo front-ends against real phone-file quirks before own captures exist (`docs/modules/video_tier.md` V2-§5, `photo_tier.md` v2.5) | Derived from the provided data; synthetic and labelled as such (`outputs/phone_like/README.md`) | Not redistributed |
| **InteriorAgent** (Hugging Face `spatialverse/InteriorAgent`, Kujiale): 3 of its 25 USD houses, `kujiale_0065` (1 BHK), `kujiale_0038` (1 BHK), `kujiale_0022` (2 BHK), fetched by `sim/fetch_scene.py` | **Simulation only** (`sim/`, D-035): rendered with Isaac Sim into phone-format captures with exact ground truth, to rehearse the walk-in test and measure fixes. No model was trained or tuned on it. Results from it are labelled "simulated" and never reported as the real benchmark | Dataset-specific "InteriorAgent Terms of Use" (https://kloudsim-usa-cos.kujiale.com/InteriorAgent/InteriorAgent_Terms_of_Use.pdf); not gated | CHECK (read the terms before any commercial or redistribution use; we do not redistribute the scenes) |
| **NVIDIA Isaac Sim 5.1** (pip, the user's existing install) | Renderer for the simulator only; not part of the pipeline | NVIDIA Isaac Sim licence (free for individual use) | OK for development; the shipped pipeline does not depend on it |
| **Candidate's own home captures** (photos, video, tape ground truth, staged paper decals) | Own benchmark for the photo and video tiers, head-to-head (Part 3) | Own data, submitted as raw benchmark data | OK |
| Training sets behind the pretrained models | MoGe-2, GeoCalib (Laval Indoor HDR credited), ALIKED, LightGlue, DPVO (TartanAir) | Per the model cards; we only run inference | as in §1 |

## 5. Libraries with notable licences

All other runtime libraries are permissive (MIT / BSD / Apache-2.0 / PSF): NumPy, SciPy, Shapely, pandas,
NetworkX, scikit-image, Matplotlib, Open3D (MIT), OpenCV (Apache-2.0), pycolmap / COLMAP (BSD-3), hloc
(Apache-2.0), Kornia (Apache-2.0), PyTorch (BSD-3), huggingface_hub (Apache-2.0), ezdxf (MIT), trimesh (MIT),
jsonschema (MIT), h5py (BSD-3), utils3d / moderngl (MIT).

| Library | Licence detail | Where | Flag |
|---|---|---|---|
| **pillow-heif** 1.8.0 | Source BSD-3, but its **binary wheels are GPL-2.0** (they bundle x265; libheif and libde265 are LGPL-3.0), per the wheel's `LICENSES_bundled.txt` | HEIC photo loading (`floorplan/photo/images.py`, `floorplan/scale/sheet.py`) | CHECK. The capture protocol asks for JPEG ("Most Compatible"), so HEIC is a fallback. For a commercial build: decode with a system libheif (LGPL) or convert HEIC on the phone. |
| Shapely → GEOS | LGPL-2.1, dynamically linked | polygons everywhere | OK |
| opencv-python wheel | Apache-2.0; bundles FFmpeg and Qt (LGPL), dynamically linked | video decoding, image processing | OK (keep notices) |
| PyTorch CUDA wheels (cuDNN, cuBLAS, NCCL, ...) | NVIDIA redistribution licence | GPU tiers | OK for use; check before redistributing a bundled image |
| **evo** | GPL-3.0 | Installed by the DPVO setup script only because upstream DPVO's demo/eval scripts import it; **not imported by our code** | EVAL-ONLY (drop it from a product env) |
| **CloudCompare** (flatpak) | GPL | Smoke-tested as a separate process for cloud-to-cloud checks; our numbers come from our own Open3D code | EVAL-ONLY (unused) |
| DPVO env extras: numba (BSD-2), pypose (Apache-2.0), yacs (Apache-2.0) | permissive | video trajectory | OK |
| **Eigen 3.4.0** headers | MPL-2.0; downloaded from gitlab.com/libeigen by the DPVO setup script, compiled into DPVO's CUDA extensions | video trajectory (DPVO build only) | OK (MPL-2 file-level copyleft: keep notices, publish changes to Eigen files; none made) |
| **`torch_scatter` shim** (`third_party/dpvo_shims/torch_scatter`) | Our own pure-PyTorch stand-in for the three functions DPVO imports (`scatter_sum`, `scatter_softmax`, `scatter_max`), following torch_scatter 2.1.2 semantics (MIT); written because PyG publishes no `torch_scatter` wheel for torch 2.14 | DPVO env only | OK |
| **FFmpeg `ffprobe`** (system binary, tested 6.1.1 Ubuntu build) | LGPL-2.1+, or GPL when built with GPL parts (the Ubuntu build is); called as a **separate process**, never linked | video metadata: rotation, codec, 35 mm focal (`floorplan/video/metadata.py`); optional: returns `{}` if missing | OK as a separate executable; a product would ship an LGPL build or read the metadata with its own parser |

## 5b. DPVO setup (video tier): what is downloaded and patched

DPVO needs compiled CUDA extensions, so it lives in its own environment and is called as a subprocess
(`floorplan/video/vo.py`, paths from `FLOORPLAN_DPVO_PYTHON`, `FLOORPLAN_DPVO_REPO`, `FLOORPLAN_DPVO_SHIMS`). The setup
script (scratch location `Mapping/envs/dpvo_setup.sh`, about 5 min; moving it into the repo as `scripts/setup_dpvo.sh`
is commit H3 in `docs/COMMIT_PLAN.md`):

| Step | Source | Licence / note |
|---|---|---|
| Clone DPVO @ `0ac95b6` | github.com/princeton-vl/DPVO | MIT; bundles lietorch (BSD-3) |
| Apply `third_party/dpvo.patch` | ours | `Tensor.type()` → `Tensor.scalar_type()` in the dispatch macros, needed for torch ≥ 2.14; no behaviour change |
| torch 2.14.1+cu130, numpy, opencv, einops, yacs, plyfile, tqdm, matplotlib, scipy, ninja, kornia, numba 0.68.0, pypose, pytest | PyPI / download.pytorch.org, pinned by `envs/dpvo_constraints.txt` | permissive |
| **evo** | PyPI | **GPL-3.0**; installed only because DPVO's demo/eval scripts import it; never imported by our code (EVAL-ONLY, drop it in a product env) |
| Eigen 3.4.0 headers | gitlab.com/libeigen | MPL-2.0 |
| `torch_scatter` shim | ours (`third_party/dpvo_shims`) | see §5 |
| Weights `dpvo.pth` | HF mirror `pablovela5620/dpvo` @ `c998d3b5`, SHA-256 cross-checked against the official `models.zip` | see §1 (CHECK) |
| Build `cuda_corr`, `cuda_ba`, `lietorch_backends` with system nvcc 13.0, `TORCH_CUDA_ARCH_LIST=8.9` | local build | needs the CUDA toolkit; no sudo |

## 6. APIs and network services

- **No API is called at run time.** No cloud inference, no LLM/VLM API. A vision-language API was considered for
  damage recognition and rejected (network and key at the walk-in, non-deterministic, no metric output, disclosure
  burden: `docs/modules/damage.md` §3).
- **Network is used only at install time:** PyPI, download.pytorch.org, GitHub (pinned git commits and release
  files), Hugging Face Hub (pinned revisions). After `scripts/fetch_weights.py`, everything runs with
  `HF_HUB_OFFLINE=1`.
- **Capture apps** (not part of the pipeline): Stray Scanner (iOS, free) for the LiDAR tier, the stock Camera app
  for photos and video. Part 3 compares against a consumer scanning app: *name and version to be filled in when the
  head-to-head is run* (placeholder).
- **Development tools:** the code was written with AI coding assistance (allowed by the brief); no AI service is
  part of the pipeline.

## 7. Commercial-use summary

| Item | Status |
|---|---|
| Default pipeline models (MoGe-2, GeoCalib, ALIKED, LightGlue) | Commercial use allowed; **GeoCalib weights need attribution** (CC-BY-4.0) |
| SegFormer-B5 ADE20K (photo-tier wall masks, D-060) | **Non-commercial** (research or evaluation only): swap for a permissively licensed segmenter before any commercial use; the photo tier still runs without it (geometry only) |
| DPVO weights | No explicit licence on the weights: confirm with the authors |
| pillow-heif binary wheel | GPL-2.0 binary: replace for a commercial build, or rely on JPEG capture |
| ARKitScenes | Evaluation, non-commercial terms; no model trained on it. **One calibration constant** (LiDAR inward offset b = 0.68 cm, D-021) was measured on 3 of its rooms: CHECK whether a constant derived from the data counts as use under the licence; for a commercial build, re-measure b on own laser-scanned rooms |
| ffprobe (FFmpeg) | Separate process, optional; ship an LGPL build or own metadata parser in a product |
| DPVO build: evo (GPL-3.0), Eigen (MPL-2.0) | evo never imported (drop it); Eigen notices kept |
| RoomFormer, Raster2Seq, PolyLayout, PixCuboid, DeepLSD's AGPL LSD core, evo, CloudCompare | Not in the shipped pipeline |

## 8. Consumer apps used for the head-to-head (Part 3)

| App | How we use it | Terms | Note |
|---|---|---|---|
| **Matterport** (Android app, free plan) | Scan of 2 tape-measured rooms of the own home; readings from its Measurement Mode | Matterport terms of service, free plan (one active space) | First choice (D-073) |
| **CubiCasa** (Android app, one free scan) | Same 2 rooms; plan with room width × length and area | CubiCasa terms of service | Second choice (D-073) |
| magicplan | **Not used.** It removed the camera scan from Android in 2024.24.0 ("Android devices are not supported for magicplan's scan features", https://help.magicplan.app/supported-devices); on Android it only draws rooms from typed lengths | — | Deviation: the brief names magicplan or poly.cam as examples |

**Deviation from Part 3.** The brief compares our **LiDAR** tier with the app. The candidate's phone (OnePlus Nord) has
no LiDAR, so the table compares our camera tiers (video, photo) with the app on the same rooms and the same tape; the
LiDAR tier's accuracy is shown against laser scans (ARKitScenes) instead.

