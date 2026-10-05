# Open questions: answered by the recruiters (2026-10-03, about 21:20 IST)

| # | Our question | Recruiters' answer | Consequence |
|---|---|---|---|
| Q1 | Can we build and evaluate on the sample data only? | **No.** "Capturing your data is mandatory; run on your captures and then sample data." | Own captures with tape ground truth (`data/own_house/`; the benchmark-set rows B1–B5 of `docs/COMPLIANCE.md`) |
| Q2 | Is a written Route 2 protocol enough for Part 1? / LiDAR without a device | "Yes, for LiDAR you can use sample data." | The LiDAR tier is benchmarked on the sample data. Photo and video tiers use our own captures. Part 1 = Route 2 protocol (`docs/CAPTURE_PROTOCOL.md`) |
| Q3 | Can they share native photos/video? | "Don't have native photos." | Photo and video tiers come from our own phone (OnePlus Nord). The sample's `rgb.mp4` runs the video tier against the sample's LiDAR plan; the photo tier is benchmarked on the own bedroom and the simulated k65 flat only (photo folders cut from the sample video were a development aid) |
| Q4 | Can Part 3 (head-to-head) be skipped? | "Complete the scope without deviation as much as possible." | Do Part 3 with a consumer app on our own rooms: CubiCasa 3.14.1 (Android, Google Play, checked 5 Oct 2026), `docs/DISCLOSURES.md` §8. The LiDAR tier is impossible on own rooms, so we compare at the photo/video tier and disclose that |

Remaining assumptions:
- The "published schema" for the JSON was not provided. We define ours (`schema/plan.schema.json`), modelled on RoomPlan and magicplan.
- Damage classes are not specified. We support an open list and stage two classes: water stain and crack.
