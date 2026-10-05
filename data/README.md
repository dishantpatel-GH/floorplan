# Data

Captures live here, one folder per capture. Git holds only this file, `MANIFEST.json`, the ground truth (`gt/`) and
the CubiCasa export; the photos, videos and scans come from `python scripts/fetch_data.py` or from your phone.

## Layout of a capture

```text
data/<capture>/
  photos/<room>/*.jpg      one folder per room, named after it (kitchen, bedroom1, ...); 2-8 photos per room
  video/<name>.mp4         one walkthrough of the whole home (.mov works too)
  lidar/<scan>/            a Stray Scanner folder: rgb.mp4, depth/, confidence/, odometry.csv, camera_matrix.csv
  gt/                      optional tape ground truth: ground_truth.csv (template: benchmark/ground_truth.csv)
```

A capture needs only the parts for the tiers you run. How to take the photos is in `docs/CAPTURE_PROTOCOL.md` (one
page): the 1x lens, phone sideways, a turn on the spot per room plus one ceiling photo, and a doorway pair at every
door between two rooms, each doorway photo in the folder of the room it looks into. Copy the files by USB cable:
messaging apps strip the camera data the pipeline reads.

## Run a capture

From the repo root, with the env active (`source .venv/bin/activate`, `export HF_HUB_OFFLINE=1`):

```bash
python scripts/check_own_capture.py data/<capture>                        # seconds, no GPU: PASS / WARN / FAIL per item
python scripts/run_capture.py data/<capture>/photos --tier photo          # GPU
python scripts/run_capture.py data/<capture>/video/<name>.mp4 --tier video  # GPU
python scripts/run_capture.py data/<capture>/lidar/<scan> --tier lidar    # CPU is enough
```

Each run prints its output folder, `outputs/runs/<name>/<tier>/`: `plan.png` and `plan.svg` are the drawing,
`plan.json` the dimensioned plan with a 95% interval on every number, `plan.dxf` the CAD file. With `gt/` filled in,
`python scripts/process_own_capture.py data/<capture>` runs every tier and scores it against the tape.

## Benchmark data

| Folder | Archive | What | Tiers it is run on |
|---|---|---|---|
| `own_house/` | `own_house.zip` | own house, one room: 7 photos lit (`photos/room`) and 6 dim (`photos/room_take2`), a walkthrough video, the hand sketch; tape ground truth in `gt/` (ceiling 2.6289 m) | photo, video |
| `k65/` | `k65.zip` | simulated k65 flat (Isaac Sim, iPhone 15 emulation): photos per room, video, LiDAR, exact ground truth in `gt/sim_gt.json` | photo (the gated benchmark), video, LiDAR |
| `sample/` | not re-hosted: your copy of the case study's Sample Data | the case study's sample captures: `single_room`, `single_scan_floor_only`, `single_scan_with_ceiling`; put the three folders here as they come | LiDAR, video |
| `cubicasa/` | in git | CubiCasa 3.14.1 (Android, Google Play) export of the own house: report PDF, plan with and without dimensions | head-to-head |

`python scripts/fetch_data.py` downloads the archives from the GitHub release `data-v1` of this repo and checks
each against the SHA-256 in `MANIFEST.json`; `--list` shows them, `--url` takes another source (a mirror, or
`file:///dir/` for a local copy). `scripts/pack_data.sh` builds the archives from this folder and rewrites
`MANIFEST.json`.

Examples on the benchmark data. `process_own_capture.py` runs the tiers through `run_capture.py`, keeps a repeat
take (`<room>_take2/`) out of the plan and scores it as a repeat, and scores everything against `gt/`:

```bash
python scripts/run_capture.py data/sample/single_room/c00a170fe1 --tier lidar
python scripts/run_capture.py data/sample/single_room/c00a170fe1 --tier video     # only its rgb.mp4 is read
python scripts/process_own_capture.py data/k65 --out outputs/k65 --tiers photo
python scripts/process_own_capture.py data/own_house --out outputs/own --tiers photo,video
```
