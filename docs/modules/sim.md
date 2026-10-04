# Simulator: what it is for, what it found, what it fixed (4 Oct 2026)

All numbers here are **simulated**: Isaac Sim renders of InteriorAgent houses, turned into iPhone-15-format files,
scored against ground truth computed from the scene geometry. They rehearse the walk-in test (30% of the score) and
measure fixes exactly. They do **not** replace the real benchmark (sample captures with laser GT, the own captures
with tape GT). How to run everything: `sim/README.md`.

## 1. Datasets (protocol v2.1, D-051 + D-055: designed once from 31 published sources + the user's review)

The scripted person follows docs/CAPTURE_PROTOCOL.md v2.1.
- **Photos: every room is seen looking in from each of its doors and looking out through them.**
  - Doorway pair at every door between two rooms.
  - Turning photos from the open middle, starting towards the door you came in by, scaled with room size, plus 1
    ceiling photo.
  - Small rooms: doorway photos turned left and right; two far-end photos facing the door; a ceiling photo aimed
    the long way.
  - Balcony: also photographed from the hall.
  - Every folder holds 5–8 photos (the brief allows 2–8).
- **Walk (video and LiDAR).** Loops along the walls, walking forward. Pass 1 aimed at the floor line, pass 2 at the
  ceiling line.
- **Checks.** Isaac Sim occupancy map ∪ mesh obstacles, and a 3D clearance check: no camera closer than 26 cm to any
  surface. Less jitter.
- Each photo's role (doorway pair, far end left/right, ceiling, ...) is printed in red under it on the contact sheets.

| Dataset | House | Seed | Contents | Verification pack |
|---|---|---|---|---|
| `outputs/sim/k65v2` | kujiale_0065, 1 BHK + balcony, 59 m² | 0 | 37 photos (12 MP, 4:3; repeat bedroom included); 8-min two-pass walk → iPhone .MOV (16:9) + Stray Scanner LiDAR | `outputs/sim/k65v2/verify/` |
| `outputs/sim/k22v2` | kujiale_0022, 2 BHK, 61 m² | 2 | 41 photos; 7-min walk | `outputs/sim/k22v2/verify/` |
| `outputs/sim/k38v2` | kujiale_0038, 1 BHK + balcony, 63 m² | 1 | 38 photos; 8-min walk | `outputs/sim/k38v2/verify/` |
| `outputs/sim/k65v2_dim` | kujiale_0065 | 0 | the k65v2 photo set with all lights at 20% (low light) | `outputs/sim/k65v2_dim/verify/` |

The v2 photo sets these replaced are kept in each dataset's `old_v2/`.

**Earlier datasets (v1 protocol), kept for the record.**
- `outputs/sim/k65_s0`: its walk entered furniture (found by the user).
- `outputs/sim/k65_fps30` / `k65_fps10`: the frame-rate check (10 fps: 9 VO segments, 30 fps: 14).
- `outputs/sim/k65_even`: the spin A/B.

## 2. What the simulator found and what was done

| # | Finding (evidence) | Status |
|---|---|---|
| 1 | Capture page ambiguous on photo counts in rooms with 3+ doors, on repeat doorway pairs, and on duration (literal following) | Page fixed (D-040) |
| 2 | Spin photos taken 0.8 m from a wall are close-ups with no usable geometry; doorway back-shots across a hallway show a wall | Page fixed: stand ≥ 1.5 m from walls, aim doorway shots at the open middle (D-044) |
| 3 | Page did not fix the photo aspect; 16:9 photos lose a quarter of the vertical view | Page fixed: keep 4:3 (D-047) |
| 4 | **Photo tier focal-length bug (I-009):** EXIF 35 mm-equivalent converted with the wrong (36 mm width) rule, −3.8% focal → footprint −46.9%, walls 28% | **Fixed (D-048):** footprint −13.2%, walls 16.7%, interval coverage 89% |
| 5 | Photo tier reported cabinet undersides as ceilings (0.9–1.8 m, "measured") | **Fixed (D-049):** plausibility 2–4 m, whole-scene ceiling as inferred fallback (errors −5 to +19 cm, all inside intervals) |
| 6 | **LiDAR tier: rooms joined by wide openings (2.1–2.3 m sliding doors) merge** (3 of 5 rooms) | **Partly fixed (D-046):** header-band cuts. Kitchen separated, its 2.12 m door measured to 4.6 mm, openings ≤ 2 cm 57% → 75%. Identical plans on all 8 real LiDAR scenes. Balcony still merged (header hidden by curtains) |
| 7 | Video tier fragments into 31 VO segments at 10 fps; plan unreliable (flagged low reliability, D-034) | Under test: 10 vs 30 fps on the identical path (D-041) |
| 8 | Photo tier per-room layout: an unseen side of a hub room inferred by symmetry (living room far wall 5.1 m read 3.0 m); furniture fronts and through-door walls taken as walls | Open. Depth is not the cause: MoGe-2 depth vs ray-cast truth has scale 0.998, shape error 1.5% |
| 9 | Protocol A/B: "even circle" spin vs "about a quarter overlap" | Inconclusive on one seed (footprint −13% vs +25%); page unchanged (D-050) |
| 10 | Focal fix on the real benchmark (sample-derived photos re-tagged losslessly with the diagonal rule) | Neutral there, as expected: those photos' EXIF was self-consistent before. Report regenerated (photo label `v4_d048`) |

**Found and fixed on 4 Oct, afternoon (D-055 to D-066):**

| # | Finding (evidence) | Status |
|---|---|---|
| 11 | Photo tier never used the A4 sheet (adapter missing), and the detector's 50-candidate cap hid the sheet in 12 MP photos | Fixed (D-056): sheet distance within 0.1–0.6% of truth. The cue is off by default since D-067 |
| 12 | Ceiling photos' depth scale 0.59–1.18 of truth; kitchen/bathroom ceilings +50% / +33% with narrow intervals | Fixed (D-057): residential prior + honest sigma; all ceilings inside intervals |
| 13 | Video rotation vote picked 180° for upright frames (medians tie) → 0 rooms | Fixed (D-059): per-frame votes |
| 14 | Furniture, partitions and mirrors taken as walls | Fixed in part (D-060): ADE20K wall mask; k65 footprint −14.9% → −8.3%, k38 −8.0% → −3.8% |
| 15 | Data itself was poor: turns covering only 108–260°, door-jamb shots, close-up spins, duplicate take 2 (audit + 3 visual reviews) | Fixed (D-064, D-065): coverage-aware scripted person, 0.5× lens (planned coverage k65 living 47% → 67%, k22 bedroom2 26% → 51%) |
| 16 | Live-run risks: a 3 h plan hang (old code), two crash paths | Guarded (D-058, I-012) |

## 3. Tier results (kujiale_0065, iPhone-15-like capture, scripted person 0)

| Tier | Rooms | Footprint | Walls (median error) | Openings ≤ 2 cm | Ceilings ≤ 1.5 cm | Notes |
|---|---|---|---|---|---|---|
| LiDAR | 3/5 (merge, I-010) | −2.2% | — (merged room) | 57% (normal doors ±5 mm; wide doors missed) | 2/3 seen (1.0, 1.4 cm) | bathroom walls 0.3–1.3 cm |
| Video (10 fps) | 4/5 | −1.7% | 34% | 0% | — | 31 VO segments; low-reliability flag; frame rate under test |
| Photo, after D-048/D-049 | 5/5, adjacency 4/4, no overlaps | −13.2% | 16.7% | — | inferred ±10 cm | before: −46.9% / 28% |

Full table for all runs: `outputs/sim/SIM_REPORT.md` (`python sim/report.py`).
