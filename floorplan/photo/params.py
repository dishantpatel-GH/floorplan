"""Tunable parameters of the photo tier, each with the reason it has its value (docs/modules/photo_tier.md, "Decisions").

The photo tier has its own parameter object (like the video tier) because its knobs (image resolutions, SfM pairing,
scale-cue weights, stitch fallbacks) have no meaning for the LiDAR tier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class PhotoParams:
    # --- image preparation ---
    sfm_long_side: int = 1024            # ALIKED/LightGlue resolution (hloc default); phone photos are 4000+ px
    depth_long_side: int = 518           # MoGe-2 input; <= 518 px keeps GPU memory small on the shared 8 GB GPU
    default_hfov_deg: float = 70.0       # 1x phone main lens when EXIF has no focal and GeoCalib is unavailable
    f35_rule: str = "diagonal"           # EXIF 35 mm-equivalent -> pixels: "diagonal" (f35*diag/43.27, standard; D-048) or "36mm" (v1)
                                         # (f35*diag/43.27, the standard definition; issue I-009, -3.8% on 4:3 photos)

    # --- joint structure from motion over ALL photos of ALL rooms (decision P-1) ---
    max_keypoints: int = 4096            # stills are few, so we can afford more keypoints than the video tier
    camera_model: str = "SIMPLE_RADIAL"  # one focal + one radial term per phone camera; focal refined by BA
    min_model_size: int = 2              # COLMAP default is 10: a room has only 2-8 photos, keep small models
    min_inliers_pair: int = 15           # verified (epipolar) matches needed before a pair is even considered
    seed: int = 0

    # --- metric edges between photos (link.py, decision P-1) ---
    edge_min_inliers: int = 12           # 3-D inliers needed to trust a photo-photo edge
    edge_min_ratio: float = 0.35         # ... and at least 35% of the lifted matches must agree (repetitive walls)
    edge_rel_thr: float = 0.05           # inlier distance grows 5% with range (learned depth errors are relative)
    edge_abs_thr: float = 0.05           # ... plus 5 cm
    edge_max_scale_ratio: float = 1.33   # two photos' learned depth scales rarely differ > 33% (MoGe p10-p90 0.82-1.03)
    edge_max_violation: float = 0.20     # dense free-space check: > 20% contradicting points = false edge (measured)
    prune_max_dt_m: float = 0.30         # loop consistency: an edge may disagree with the solution by 30 cm ...
    prune_max_dt_rel: float = 0.15       # ... or 15% of its length, whichever is larger ...
    prune_max_dyaw_deg: float = 8.0      # ... and 8 deg of yaw
    bridge_min_inliers: int = 20         # an edge that alone links two photo groups needs >= 20 inliers (measured)

    # --- MapAnything proposals inside a room, verified by depth (intra.py, decision P-2c) ---
    intra_room_proposals: bool = False   # OFF: 3 of 6 verified proposals were still > 0.5 m wrong (P-2c)
    intra_max_tilt_deg: float = 10.0     # proposal must agree with both photos' gravity within 10 deg
    intra_min_overlap_pts: int = 500     # photos must share enough comparable depth to verify the proposal ...
    intra_min_agreement: float = 0.40    # ... and >= 40% of it must agree within 8% (else: unverifiable, dropped)
    intra_edge_weight: int = 10          # weaker than a feature edge in the spanning tree (inlier-count units)

    # --- metric depth (MoGe-2, decision P-3) ---
    depth_max_m: float = 6.0             # photos look across whole rooms (corner to corner): allow up to 6 m
    depth_edge_rel: float = 0.04         # drop pixels whose depth jumps > 4% to a neighbour (flying pixels)

    # --- scale (decision P-4) ---
    scale_model_sigma: float = 0.04      # 1-sigma systematic scale bias of learned metric depth (video tier V-7)
    scale_sigma_no_focal: float = 0.20   # v2: no EXIF focal -> default FOV; measured +22% scale shift (phone_like)
    scale_min_obs: int = 30              # minimum SfM points per image for a depth-ratio scale observation
    use_sheet: bool = False              # D-067: no reference object in the protocol; the paper-sheet cue is opt-in

    # --- dense points / fused surface ---
    voxel_m: float = 0.02                # fused surface grid (like LiDAR TSDF voxel)
    raw_voxel_m: float = 0.01            # raw measurement points: 1 cm de-duplication grid
    raw_stride: int = 2                  # every 2nd depth pixel
    consistency_rel: float = 0.06        # multi-view check: a point must agree with another view within 6% depth

    # --- stitching fallback (decision P-5) ---
    fallback_gap_m: float = 0.15         # wall thickness assumed between rooms placed by the doorway fallback

    # --- protocol v2 (D-017, docs/modules/photo_tier.md part v2) ---
    doorway_pair_max_dt_s: float = 15.0  # two photos of different rooms <= 15 s apart (and consecutive) = one spot
    doorway_pair_rhythm: float = 2.0     # ... and <= 2x the median same-room shot interval (walking between rooms
                                         #     from one spin to the next takes longer than turning round in a door)
    features_need_pair: bool = True      # with doorway pairs present, a cross-room feature edge must join paired rooms
    pair_max_overlap_violation: float = 0.35  # post-solve: rooms joined by a pair must not contradict (gross test,
                                              # 30% and 1.5 m in front). Measured on floor_only R3~R6: right yaw
                                              # 0.002-0.22 (depending on the rest of the graph), wrong yaws 0.50-0.93
    pair_sigma_t_m: float = 0.20         # doorway pair: both photos from the same spot (feet still, phone turned)
    pair_sigma_yaw_deg: float = 6.0      # ~180 deg turn snapped to the photos' own Manhattan axes (normal noise ~2-3 deg)
    pair_sigma_yaw_unmeasured_deg: float = 30.0  # no Manhattan estimate: "turn round" is all we know
    spin_step_deg: float = 60.0          # protocol: about a sixth of a turn between spin photos
    spin_sigma_t_m: float = 0.30         # turning on the spot moves the phone on a ~0.3 m circle (arm + body sway)
    spin_sigma_yaw_deg: float = 10.0     # Manhattan-snapped step (per-photo Manhattan yaw error up to ~16 deg, measured)
    spin_prior: bool = True              # the folders follow the D-017 spin protocol (False for corner-style shots:
                                         # a shared-centre prior on photos taken metres apart is confident garbage)
    spin_direction: str = "clockwise"    # protocol: turn clockwise (to the right). "auto" = choose by Manhattan
                                         # residuals; measured: auto picked the wrong direction in 2 of 13 simulated
                                         # rooms (steps of 40-45 and 90 deg are ambiguous modulo 90 deg)
    spin_preferred_sign: int = 1         # "auto" only: prefer clockwise ...
    spin_direction_margin_deg: float = 20.0  # ... unless the other direction fits better by > 20 deg in total
    spin_sigma_yaw_unmeasured_deg: float = 25.0  # a photo without enough wall: only "about 60 deg"
    prior_sigma_log_s: float = 0.05      # per-photo learned-depth scale spread (MoGe p10-p90 0.88-0.99, ~4%); 0.10
                                         # let two prior-linked groups drift 16% apart (measured, floor_only R3/R6)
    layout_registration: bool = True     # place an unmatched doorway photo in its room by wall alignment
    layout_cell_m: float = 0.05          # top-view grid of the wall correlation
    layout_sigma_m: float = 0.10         # wall map blur: learned-depth wall noise at 2-4 m
    layout_min_score: float = 0.45       # mean wall agreement needed (0..1)
    layout_max_rival_ratio: float = 0.85  # best must beat any other placement/rotation by >= 15% (else ambiguous)
    layout_door_max_wall_dist_m: float = 0.6  # doorway photo camera must lie on the room's wall line
    bridge_max_overlap: float = 0.08     # pair bridge: share of B's 10 cm top-view cells already occupied by A
    layout_sigma_t_m: float = 0.15
    layout_sigma_yaw_deg: float = 4.0
    widen_layout: float = 3.0            # a room joined only through a layout-registered doorway photo
    widen_door_match: float = 3.5        # a room placed by the door-matching fallback (position inferred)

    # --- v3: per-room layout from the room's own spin (layout.py, photo_tier.md v3) ---
    room_layouts: bool = True            # fit a Manhattan rectangle per folder from its spin photos (info.room_layouts)
    layout_stride: int = 2               # every 2nd depth pixel (518 px depth maps: ~40k points per photo)
    layout_cam_height_m: float = 1.35    # camera height when no photo sees the floor (protocol: chest height)
    layout_wall_max_ny: float = 0.3      # wall = MoGe-2 normal within ~17 deg of horizontal
    layout_band_lo_m: float = 1.0        # wall band above the floor: beds, tables, sofas, counters (< 0.95 m) excluded ...
    layout_band_hi_m: float = 2.0        # ... and the ceiling, cornice and door heads above 2.0 m
    layout_farthest_per_column: bool = True  # wall = farthest band point of each image column (furniture in front)
    layout_prefer_near_ratio: float = 0.6  # nearest wall peak at least 60% as long as the longest (through-door)
    layout_far_wall: bool = True         # D-077: a wall seen BESIDE the chosen surface, farther out, replaces it (a
                                         #     wardrobe front or an open door leaf labelled "wall"; layout._far_wall)
    layout_far_wall_min_len_m: float = 0.8   # ... with >= 0.8 m of wall (own Room: 1.0-2.5 m; 1.2 lost 2 of 3 sides)
    layout_far_wall_min_gap_m: float = 0.5   # ... 0.5-1.5 m behind the chosen surface: own Room 0.57-1.01 m (wardrobe,
    layout_far_wall_max_gap_m: float = 1.5   #     door leaf); one wall seen by two photos: 0.30-0.45 m apart (sim)
    layout_far_wall_min_ratio: float = 1.33  # ... and >= 1.33x as far: two photos' depth scales differ < 33% (as
                                             #     edge_max_scale_ratio); own Room 1.51-2.78, same wall 1.15-1.33
    layout_far_wall_max_shadow: float = 0.2  # <= 20% of its rays cross the chosen plane where that surface was seen
                                             #     (sim peaks that must stay: 0.25-1.00; own Room 0.00)
    layout_far_wall_min_beyond: float = 0.8  # >= 80% of its rays cross the chosen plane past an END of that surface,
                                             #     not between two parts of it (an opening in a wall: own lit 0.0)
    layout_far_wall_min_overlap_m: float = 0.3  # ... and it overlaps the room's extent along it by >= 0.3 m (as the
                                             #     polygon's poly_side_min_overlap_m): beyond a perpendicular wall it
                                             #     is the next room through a door (sim k38_s1 bathroom -0.13 m)
    layout_normal_min: float = 0.8       # a point belongs to a side when its normal faces the centre within ~37 deg
    layout_min_dist_m: float = 0.3       # ... and it is at least 0.3 m from the centre on that side
    layout_min_wall_pts: int = 200      # 500 dropped floor_only R5 (329 points); 200 keeps it (6/6 rooms, v3 results)
    layout_min_side_pts: int = 40
    layout_bin_m: float = 0.05           # offset histogram bin
    layout_cell_m: float = 0.10          # wall length counted in distinct 10 cm tangent cells
    layout_min_wall_len_m: float = 0.4   # less than 0.4 m of wall at one offset: the side is not seen
    layout_inlier_m: float = 0.12        # inliers of a wall: within 12 cm of its offset (learned-depth wall noise)
    layout_cam_sigma_m: float = 0.30     # photo position around the spin centre (same as spin_sigma_t_m)
    layout_geom_sigma_m: float = 0.05    # learned-depth surface noise per wall (plan_beta T-2)
    layout_unseen_sigma_rel: float = 0.35  # an unseen side: sigma 35% of its inferred distance ...
    layout_unseen_sigma_min_m: float = 0.40  # ... and at least 0.4 m
    layout_unseen_default_m: float = 1.5   # nothing seen on either side of an axis (never seen in tests)
    layout_min_level_pts: int = 200      # ceiling / floor points needed for a room height
    ceiling_photo_min_pitch_deg: float = 18.0  # protocol v2 (D-052): a photo tilted up more than this is a ceiling photo
    ceiling_photo_last_min_pitch_deg: float = 10.0  # ... from 10 deg for the LAST photo of a room's series (where the
                                         # protocol puts it): own-home ceiling shots were 16.6 and 16.7 deg. Mid-series
                                         # stays 18: turning video frames reached 11.9-16.9 deg (with_ceiling/rotate),
                                         # where 10 for every photo gave 3 different plans in 3 runs
    small_room_from_doorway: bool = True # protocol v2 (D-053): a room without a spin is boxed from its threshold photo
    layout_use_door_shots: bool = True   # D-054: hub rooms add their doorway-pair photos (placed) to the box fit
    door_threshold_inset_m: float = 0.10 # the door wall's room-side face lies ~half a wall thickness ahead of the camera
    door_threshold_sigma_m: float = 0.08
    ceiling_link_sigma_rel: float = 0.05  # D-057: ceiling photo scaled by matches to its room's photos (MoGe spread)
    ceiling_prior_m: tuple = (2.70, 0.25)    # D-057: residential floor-to-ceiling prior (2.4-3.0 m typical; sim 2.55-2.80)
    ceiling_unlinked_sigma_rel: float = 0.25  # ... not linked: its own scale on an upward photo (sim 0.59-1.18)
    semantic_walls: bool = True          # D-060: wall points must be labelled "wall" by the ADE20K segmenter
    semantic_min_pts: int = 150          # ... else (too few) the photo keeps its geometric wall points
    semantic_timeout_s: int = 900        # the segmenter subprocess (GPU ~0.3 s/photo; CPU ~10-35 s/photo)
    threshold_same_spot_unplaced: bool = False  # D-063 (off): try an unplaced 2nd doorway photo at the same spot
    closeup_depth_m: float = 1.0         # D-062: a turning photo whose median depth is below this is a wall close-up
    extra_photos_max: int = 2            # v2.1 (D-055): up to 2 photos after a room's ceiling photo are from other spots
    same_spot_m: float = 0.6             # ... one within this of the threshold photo was taken on the same threshold
    layout_height_min_m: float = 2.0    # a ceiling is 2.0-4.0 m above the floor; outside that the "ceiling" points are
    layout_height_max_m: float = 4.0     # undersides of cabinets/lamps: report not observed (D-049, sim: 0.9-1.8 m)
    layout_height_sigma_m: float = 0.05
    layout_side_sigma_rel: float = 0.5   # extra per-side sigma, relative to the wall distance: calibrated on the
                                         # simulated spins (dims covered 13/20 at 0, 15/20 at 0.35 and 0.5; areas
                                         # 4/12, 6/12, 7/12), v3 results; re-check on the own-home tape
    layout_mc: int = 400                 # Monte Carlo draws for room dimensions / area
    layout_max_spin_spread_m: float = 1.0  # placed spin photos further apart: not one spot, layout not used
    layout_overlap_gap_m: float = 0.10   # rooms pushed apart until they are at least a wall-face gap apart
    layout_polygon: bool = True          # v3-poly (polygon.py): rectangle + evidence-backed steps; plan_beta uses it
    pillars: bool = True                 # D-083 (pillars.py): pillars and wall steps found per photo are cut into
                                         #     the polygon outline as notches; the box sides do not move
    pillar_face_side: bool = True        # D-086: ... except a side that sat on a pillar face: it moves out to the wall
    poly_skip_far_wall_lines: bool = True  # D-083: a surface the far-wall rule set aside as furniture (D-077) bounds
                                         #     no polygon notch (own Room: the false wardrobe-corner notch)

    # --- interval widening (decision P-6) ---
    widen_linked: float = 2.0            # photo intervals vs the LiDAR error budget when rooms are SfM-linked
    widen_unlinked: float = 4.0          # rooms placed by the fallback (position is a guess, sizes are not)

    # --- reference-free scale from standard door heights (door_scale.py, D-069) ---
    door_scale_cue: bool = False         # NOT WIRED YET (D-069): frontend.py never calls door_scale.py, so this
                                         #     has no effect; off until validated on real photos of homes
    door_height_prior_m: float = 2.06    # interior door: US 2.03 m, EU 2.0-2.1 m, India 2.1-2.13 m
    door_height_sigma_rel: float = 0.035  # ... 1-sigma across homes; systematic within one home (all doors alike)

    # --- door-anchored stitching (door_stitch.py, D-081): rooms snap together at the doors they share ---
    door_stitch: bool = True             # D-088: on: k65 +0.035, k22 +0.10 IoU, k38 -0.001 (paired, 3 runs each)
    door_bin_m: float = 0.05             # door intervals are built from 5 cm cells along each wall
    door_stride: int = 2                 # every 2nd depth pixel (518 px photos: ~33k points per photo)
    door_beyond_m: float = 0.25          # a point this far past a wall face was seen through an opening in it
    door_low_top_m: float = 0.8          # below this a window has its sill: a door is open down to the floor
    door_top_m: float = 1.95             # above a door's head the wall continues
    door_min_incidence_deg: float = 15.0  # grazing rays give unstable positions along the wall
    door_min_pts: int = 4                # points per 5 cm cell before it counts
    door_min_out_frac: float = 0.6       # share of a cell's door-height points seen through the wall
    door_gap_bins: int = 2               # gaps up to 10 cm inside a door are closed (a jamb, a stray point)
    door_min_w_m: float = 0.45           # interior doors 0.6-1.0 m, open passages up to ~3 m
    door_max_w_m: float = 2.5            # wider 'openings' were the rest of an L-shaped room (sim k38), not doors
    door_flank_m: float = 0.6            # a door is a gap in a wall: wall seen within this of one of its ends
    stitch_max_move_m: float = 1.0       # a room the pose graph placed with the reference moves at most this much
                                         #     (its errors were 0.15-0.55 m on the 3 flats; k22 bathroom: 1.26 m)
    door_sigma_c_m: float = 0.08         # 1-sigma of a seen door's centre along its wall
    threshold_sigma_c_m: float = 0.25    # a doorway photo's camera: somewhere on the threshold, not at its centre
    first_photo_max_angle_deg: float = 40.0  # the entered room's first turning photo faces its entry door
    threshold_square_deg: float = 12.0   # a threshold photo this square to the wall behind it shows the door's wall
    inferred_slack_m: float = 1.5        # an inferred side may lie this much farther out (sim k65 living: 1.8 m)
    jamb_min_pts: int = 30               # jamb points needed to measure the wall thickness at a door
    wall_thickness_m: float = 0.15       # prior when the jambs are not seen
    wall_thickness_sigma_m: float = 0.05
    pnp_min_inliers: int = 15            # PnP inliers (points also in the next room) for one photo pair to count
    intra_pnp: bool = True               # a room's photo the pose graph left out is placed by PnP on one in its frame
    intra_min_inliers: int = 12          # ... the pose graph's own edges need 12 3-D inliers (edge_min_inliers)
    intra_box_margin_m: float = 0.5      # ... standing in the room or on its threshold
    intra_max_view_deg: float = 75.0     # ... and looking into it
    pnp_px: float = 2.0                  # RANSAC reprojection threshold at the 518 px depth resolution
    pnp_max_depth_m: float = 10.0        # matched points farther than this (outdoors, through windows) are not used
    pnp_max_tilt_deg: float = 3.0        # both photos are levelled: true relative poses had tilt <= 1.9 deg (sim k65)
    pnp_max_dy_m: float = 0.35           # chest-height photos: camera heights within 35 cm
    pnp_min_baseline_m: float = 0.30     # less = a pure rotation (matches on the far scene through windows)
    pnp_max_axis_resid_deg: float = 12.0  # the two rooms' Manhattan axes must agree within this after the PnP
    pnp_min_good_frac: float = 0.5       # share of PnP inliers outside the out-looking room AND inside the other
    out_margin_m: float = 0.15           # "outside a room" = this far past its box
    in_margin_m: float = 0.30            # "inside the other room" = within its box plus this
    max_room_overlap: float = 0.15       # the PnP placement may overlap the two rooms by this share of the smaller
    link_min_inliers: int = 20           # a room pair needs this many verified inliers in total (weak link below)
    link_max_rival: float = 0.5          # a second placement with half the inliers makes the link ambiguous
    cluster_tol_m: float = 0.5           # photo pairs agreeing within this on the room placement are one cluster
    door_assoc_m: float = 0.8            # the looked-through door is within this of where the rays cross the wall
    snap_max_move_m: float = 0.8         # the door snap may move a PnP placement by this much at most
    pnp_sigma_m: float = 0.15            # a PnP room link between photos placed by their room's own fit (or PnP)
    pnp_sigma_pg_m: float = 0.15         # ... through a photo the pose graph placed in its room (0.30 tried, D-081)
    pose_graph_sigma_m: float = 0.0      # the pose graph's placement of a room as a prior; 0 = none (0.35 tried:
                                         #     k22 lost its gain, D-081)
    same_spot_sigma_m: float = 0.30      # doorway pair: both photos on one threshold
    overlap_tol_m: float = 0.0           # stitched rooms must not overlap: with 0.3 m allowed, plan_beta's push-apart
                                         #     moved the k22 rooms afterwards (IoU +0.02 instead of +0.19, 3 runs)
    overlap_sigma_m: float = 0.03        # ... overlaps are penalised at this scale

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> "PhotoParams":
        p = PhotoParams()
        names = {f.name for f in fields(PhotoParams)}
        for k, v in (d or {}).items():
            if k not in names:
                raise KeyError(f"unknown photo param {k}")
            setattr(p, k, v)
        return p
