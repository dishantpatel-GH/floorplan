# Indoor capture practice for floor plans: apps, datasets and SfM guides

Research date 2026-10-04. Bracketed numbers point to the source list at the end. **(U)** means unverified: taken from a search snippet or a secondary page because the primary page was unreachable (JavaScript-only, 401 or 403) or was not found.

## A. Apps

| Source | Orientation / pitch | Path / positions | Distance, speed, time | Doors, small rooms | Ceiling | Avoid |
|---|---|---|---|---|---|---|
| **RoomPlan** [1][2][3] | Not prescribed; live coaching | 1 room up to 9×9 m. Multi-room: keep the ARSession running, merge with StructureBuilder; home up to 2,000 sq ft | Coaching cues `moveCloseToWall`, `moveAwayFromWall`, `slowDown`, `turnOnLight`, `lowTexture` (no numbers given); scan under 5 min | Close doors; open curtains | High ceilings may exceed LiDAR range | Under 50 lux; mirrors/glass; dark surfaces |
| **Polycam** LiDAR [4] | Tilt up for ceiling corners, down for floor-wall joints | Start in a corner, aim at the opposite corner near the floor; walk the perimeter | "Steady and slow"; if tracking is lost, stop and pan | Open doors | Floor to ceiling, not just eye level | Mirrors, glass |
| **Polycam** photo [5] | Slightly down | Perimeter with back to the wall, one step between series; then an inner ring of 1–2 m | 5–6 shots per spot (0°, 45°, 90°); 30–50 % overlap | Start at the door | — | Bright windows, glass |
| **magicplan** [6][7] | Camera Corner mode: calibrate on feet, then ceiling; aim at floor corners | **One spot** per room | — | Aim at each door/window and tap | Set on a height grid | Auto-Scan: open plans, stairwells, glass. One room per scan, doors closed |
| **CubiCasa** video [8][9][10] | **Portrait**, chest height (U), slightly down; baseboards in frame, no more than 1 m of wall | Along walls; one continuous take for the whole home; never pan from the middle of a room | 1.5–3 m from walls; slow; never sideways | Doors open beforehand; small rooms scanned from the doorway; back out of narrow spaces | Only for low/sloped ceilings: tilt slightly up for 5 s or less | — |
| **Matterport** iPhone [11][12] (U) | Rotate in place around the phone; 1 ring (Simple) or 3 rings (Complete) | Spots 1.5–2 m apart, each in line of sight of the last | — | Spots 30–60 cm either side of a threshold, never in the doorway | Upper ring | — |
| **Canvas** LiDAR [13][14] | "Spray paint": floor → wall → ceiling, 1–2 steps sideways, back down | Start at a textured corner; one perimeter loop in one direction | 1.5–3 m from walls; under 20 min per scan | Doors open (U) | No need to cover the whole ceiling | Starting on a blank wall |
| **Hover** video/LiDAR [15][16] | Back to the wall, aim at the far wall with floor and ceiling in frame; paint up and down under high ceilings | One smooth loop per room; hallways as separate spaces | Slow; pause and pan in each corner | Phone up, aimed at the next doorway, while moving between rooms; small rooms scanned from the entry with the whole door frame | Painting motion | Jerky motion |
| **DocuSketch** 360 [17][18] | Tripod on the floor, mid-room, 1.57 m high (U) | +1 shot per ~9 m²; in hallways every ~3 m and at each doorway | 0.9–2.1 m from walls (U) | Doors and windows visible | Ceiling corners count as corners | People hiding corners |

## B. Datasets

| Dataset | Device, height | Positions per room, spacing | Floor/ceiling | Linking rooms |
|---|---|---|---|---|
| **Matterport3D** [19] | Tripod rig tilted up, level and down × 6 headings = 18 images; "approximately the height of a human observer" (no 1.5 m figure in the paper) | Spots about 2.5 m apart; measured 2.25 ± 0.57 m | Both | Panoramas registered together |
| **ZInD** [20] | 360° camera on a tripod; **fixed height per home** plus a calibration target | About 3 panoramas per room (42.8 per home over 14.3 rooms); 1 "primary" per room | Both | Every room and hallway; doors open; closets shot from outside |
| **ARKitScenes** [21] | iPad Pro LiDAR plus a laser scanner | About 4 laser scans per room; up to 3 handheld sequences per room, each with a different motion pattern (details in the supplement, U) | Each sequence covers ceiling, floor and walls | One coordinate system per home |
| **ScanNet** [22] | iPad with depth sensor, novice operators | No prescribed path; a "featurefulness" bar warns before tracking is lost | — | One scan per room |
| **ScanNet++** [23] | Laser scanner, fisheye DSLR, iPhone 13 Pro | About 4 laser scans; about 200 DSLR shots on a dense path at standard height; **about 2 min of iPhone video per medium room** | — | Large spaces as one scene |
| **HorizonNet** [24] | Assumes **1.6 m** camera height, parallel floor and ceiling, square corners | 1 panorama per room | Needs **both** the floor-wall and the ceiling-wall line | — |
| **Structured3D** [25] | Synthetic | Camera at random free spots, not the room centre | — | — |
| **3RScan** [26] (U) | Handheld Tango phone; 2–12 rescans per scene | No pattern found | — | — |

Hypersim was not reviewed (synthetic, low relevance).

## C. SfM / photogrammetry for indoor stills

| Source | Guidance |
|---|---|
| COLMAP [27] | Each object in at least 3 images. "Do not take images from the same location by only rotating the camera". Avoid blank walls and shooting into windows |
| Metashape [28] | "Walk with your back next to the wall and shoot the wall opposite"; "Do not shoot from one point in the center of the room". At least 60–70 % overlap. Adjacent rooms: open the door and shoot from each side with walls of both rooms visible |
| Pix4D [29][30] | Interiors: **90 % overlap**, fisheye lens, back to the wall shooting at 90°. Their test room used 126 images. Doorways: shoot the floor at twice the rate (U) |

## Synthesis: consensus practices

There are two families of method:
- **(a) Rotate in place** (Matterport, DocuSketch, ZInD, magicplan, HorizonNet). The layout is solved from each spot, and scale comes from a **known, fixed camera height**.
- **(b) SfM** (COLMAP, Metashape, Pix4D). This needs **movement between shots** and 60–90 % overlap, and it forbids shooting everything from the room centre.

With 2–8 stills per room, (a) has to be primary and (b) a weak add-on. Getting overlap from so few shots needs the 0.5× lens; that last point is our inference.

**1. Photos (2–8 per room)**
- Shoot level at one recorded height (chest, about 1.5–1.6 m) [20][24].
- Every corner (floor or ceiling), every floor line and ceiling line, and every door and window must appear in at least one shot [17][18][24].
- Shoot from corners or with your back to a wall toward the opposite corner, moving at least one step between shots, with adjacent shots overlapping at least 60 % [27][28].
- Add a spot for every ~9 m² beyond the first and every ~3 m of hallway; keep spots no more than 2–2.5 m apart [17][18][19].
- **Bathrooms and closets**: shoot from the doorway with the whole door frame in view [8][16][20].
- **Linking rooms**: open doors and take one shot each side, 30–60 cm from the threshold, with both rooms' walls visible [12](U)[28].
- Lights on, blinds open, and don't expose against windows [1][27].

**2. Walkthrough video**
- Portrait, chest height, walking forward slowly, never sideways; one continuous take [8][10].
- Walk a perimeter loop 1.5–3 m from the walls; never pan from the middle of a room [8][13].
- Pitch has two proven options:
  - Floor-first: slightly down, baseboards in frame, with brief (5 s or less) upward tilts only under low or sloped ceilings [8][9].
  - Vertical "painting" sweeps from floor to ceiling while moving [13][16].
- **A separate floor pass then ceiling pass appears in no app guide found.** The nearest precedent is ARKitScenes' up to 3 sequences per room with different patterns [21].
- Pause and pan in each corner. Keep the camera aimed at the next doorway while moving between rooms. Scan small rooms from the door [15][16].
- Budget about 2 min per medium room [23].

**3. LiDAR (Stray Scanner)**
- Stray Scanner documents only its data format, with **no capture guidance** [31]: 256×192 depth in mm, confidence 0/1/2, per-frame pose and intrinsics. Capture practice comes from RoomPlan, Polycam and Canvas.
- At least 50 lux, curtains open, mirrors and glass avoided [1][4].
- Start at a textured corner. Walk one loop in one direction, 1.5–3 m from the walls, tilting to catch both ceiling corners and floor joints [4][13].
- Slow. If tracking is lost, stop and pan [4].
- Under 5 min per room and under 20 min per recording [1][14]. Multi-room: one continuous session, walking through each doorway toward the next room [2][16].
- Ceilings above LiDAR range [1]: capture the ceiling line from farther away (inference).

## Sources
1. https://developer.apple.com/videos/play/wwdc2022/10127/
2. https://developer.apple.com/videos/play/wwdc2023/10192/
3. https://developer.apple.com/documentation/roomplan/roomcapturesession/instruction/turnonlight
4. https://poly.cam/blog/how-to-create-floor-plans-using-your-iphone-in-10-minutes
5. https://poly.cam/blog/high-resolution-interior-photogrammetry-full-walkthrough-polycam-tutorial
6. https://help.magicplan.app/scan-a-room-in-seconds-using-lidar
7. https://help.magicplan.app/auto-scan-your-floor-plan
8. https://help.cubi.casa/en/articles/6662269-quick-guide-for-scanning
9. https://kb.crmls.org/knowledgebase/cubicasa-tips-for-scanning/ (secondary)
10. https://kb.crmls.org/knowledgebase/cubicasa-how-to-scan-a-property/ (secondary; "portrait")
11. https://support.matterport.com/s/article/Getting-Started-Matterport-for-iPhone?language=en_US (U)
12. https://support.matterport.com/s/article/How-to-Determine-the-Scan-Path?language=en_US (U)
13. https://support.canvas.io/article/10-tutorial-how-to-scan-a-room-with-canvas
14. https://support.canvas.io/article/11-how-big-of-a-space-can-i-scan
15. https://help.hover.to/en/articles/9264961-how-to-scan-an-interior-space-universal
16. https://help.hover.to/en/articles/9426874-best-practices-for-scanning-interiors
17. https://help.docusketch.com/docs/cameratripod-placement
18. https://help.docusketch.com/docs/best-practices-for-quality-360-images-and-accurate-sketches
19. https://arxiv.org/abs/1709.06158 (Matterport3D §3.1)
20. https://github.com/zillow/zind (README: capture protocol, statistics)
21. https://arxiv.org/abs/2111.08897 (ARKitScenes §3.1)
22. https://arxiv.org/abs/1702.04405 (ScanNet appendix)
23. https://arxiv.org/abs/2308.11417 (ScanNet++)
24. https://arxiv.org/abs/1901.03861 (HorizonNet §3.2)
25. https://arxiv.org/abs/1908.00222 (Structured3D)
26. https://waldjohannau.github.io/RIO/ (3RScan, U)
27. https://colmap.github.io/tutorial.html
28. https://agisoft.freshdesk.com/support/solutions/articles/31000163324-suggested-scenario-for-photo-shooting-an-interior
29. https://www.pix4d.com/blog/indoor-mapping-game-plan
30. https://support.pix4d.com/hc/en-us/articles/202557459
31. https://github.com/strayrobots/scanner/blob/main/docs/format.md
