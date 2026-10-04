# Own benchmark capture plan

The step-by-step guide for the capture is `docs/HOUSE_CAPTURE_GUIDE.md`, a copy of the one used on the phone.

How each case-study benchmark rule (Part 2) is met:

| Requirement | How it is satisfied |
|---|---|
| Multi-room capture, 3+ rooms + connector | Own home, photo + video tiers. The sample data also has a multi-room LiDAR capture |
| Furnished room with staged damage, 2 classes | Paper decals: water stain (class A) and crack (class B), measured with tape |
| Same rooms at all three tiers | Photo + video on our home; LiDAR from the sample data (recruiter-approved deviation); sample video and photos derived for cross-tier checks |
| One room captured twice at the same tier | Video walkthrough ×2; one room's photo set ×2; sample LiDAR `floor_only` vs `with_ceiling` |
| Laser/tape ground truth on everything | Tape: every wall at 1 m height, ceiling heights, doors, windows, damage extents |
| Mirrors, glass, wet-look surfaces, low light | Bathroom mirror and windows included; optional low-light video take |
