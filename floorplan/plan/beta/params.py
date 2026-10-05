"""Every tunable of the space-first plan extractor, in one place, with the reason for its value.

None of these values is tuned to a particular capture. They are physical quantities of ordinary homes (door widths,
wall thicknesses, furniture heights) or of the sensor (LiDAR noise), so the same values must work on an unseen
apartment at the walk-in test. docs/modules/plan_beta.md explains how each one was chosen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BetaParams:
    # --- raster (step 1) ---
    cell_m: float = 0.02               # 2 cm cells: same as the TSDF voxel, fine enough for 0.6 m doors (30 cells)
    seed: int = 0                      # all random subsampling and Monte Carlo use this seed
    grid_lattice: bool = False         # v2 B-5: raster corner on a world lattice (multiples of cell_m), see Grid
    grid_phase: tuple = (0.0, 0.0)     # v3 S-2: sub-cell raster offset (u, v) in cells; set by the phase vote
    phase_votes: int = 1               # v3 S-2: build the plan at this many sub-cell raster phases, keep the medoid
    phase_workers: int = 4             # v3 S-2: phases built in parallel (forked processes); 1 = sequential
    phase_agree_tol_m: float = 0.01    # v3 S-2: two walls "agree" if both ends are within max(this, 0.5% L)

    # --- wall evidence: vertical surfaces in a height band that furniture rarely reaches ---
    wall_band_lo_m: float = 1.0        # above sofas (0.9 m), tables (0.75 m) and kitchen counters (0.9 m)
    wall_band_hi_m: float = 2.0        # below door heads (~2.05 m); the floor-only capture sees up to ~2 m
    wall_min_extent_m: float = 0.4     # vertical surface must cover >= 40 cm of the band (a monitor covers ~20 cm)
    wall_dilate_m: float = 0.04        # close 1-2 cell gaps in wall evidence so free space cannot leak

    # --- free space (interior) ---
    floor_tol_m: float = 0.05          # a raw point within 5 cm of the floor level is a floor observation
    carve_hi_m: float = 1.0            # ray parts lower than this (= wall band bottom) carve free space
    carve_stop_m: float = 0.05         # stop carving 5 cm before the hit point (that is the surface itself)
    carve_subsample: int = 8           # every 8th raw ray is enough for a 2 cm raster
    carve_min_hits: int = 2            # two independent rays per cell, so one stray ray cannot open a wall
    traj_radius_m: float = 0.30        # the operator's body occupied this disc around the camera path
    close_m: float = 0.06              # morphological closing that removes speckle between floor samples
    hole_max_m2: float = 6.0           # furniture holes up to the size of a double bed + margin are filled
    hole_min_free_frac: float = 0.5    # ... but only if at least half their border is free space, not wall
    min_component_m2: float = 1.0      # free-space islands smaller than this are noise (e.g. inside thick walls)

    # --- v2 B-1: long wall lines (alpha's detector) as segmentation cuts, see lines.py ---
    use_sep_lines: bool = True         # ablation switch (False = v1 behaviour)
    sep_band_lo_m: float = 0.30        # alpha's band: above skirting ...
    sep_band_hi_m: float = 2.00        # ... below the ceiling junction; includes the part floor_only sees
    sep_peak_min_points: int = 60      # alpha: a peak needs >= 60 TSDF voxels (~0.25 m2 of face)
    sep_occ_min_points: int = 8        # alpha: a 2 cm along-bin is occupied with >= 8 voxels (~16 cm of height)
    sep_min_support_m: float = 0.40    # alpha: a line is kept when >= 40 cm of it is occupied
    sep_min_run_m: float = 0.40        # a cut is a contiguous occupied run of >= 40 cm ...
    sep_max_gap_m: float = 0.06        # ... after closing gaps up to 6 cm (TSDF holes), never a door (>= 50 cm)
    barrier_pair_required: bool = True # cut only where an opposite face lies 5-40 cm behind (both sides of a wall)
    #                                    or where the face is tall (False: any long run, the J-I-5 trap, ablation)
    sep_min_extent_m: float = 0.80     # a one-sided face must cover >= 80 cm of the 0.3-2.0 m band (furniture <= 0.7)
    # --- v5 (D-046): headers over WIDE openings separate rooms (sliding doors, wide doorways; found in simulation) ---
    use_header_cuts: bool = True       # D-046; identical plans on all 8 real LiDAR scenes, fixes sim kitchen merge
    header_band_lo_m: float = 2.05     # above door heads (interior doors ~2.0-2.1 m) ...
    header_band_hi_m: float = 2.75     # ... below most ceilings (the wall above an opening is the header)
    header_occ_min_points: int = 4     # a header can be only ~20 cm tall: fewer voxels per 2 cm bin than a wall
    header_min_open_m: float = 1.30    # only where the band below is open for >= door_max_m (narrow doors: neck rule)
    header_gap_m: float = 0.80         # leave a door-sized gap in the middle: the neck rule then places the opening
    header_coplanar_m: float = 0.12    # the header face lies in the plane of the wall face below it (same facing)
    header_min_wall_m: float = 0.30    # ... and that wall line exists: >= 30 cm of low-band wall somewhere along it
    header_fill_gap_m: float = 0.40    # header faces are patchy (seen from below at a grazing angle): close 40 cm gaps
    header_merge_m: float = 0.20       # parallel header-face peaks this close are one header (frame, track, drift)

    # --- v2 X-4: photo-tier room folders as hints; B-8 coverage warning ---
    use_room_hints: bool = True        # merge touching regions whose floor points belong to the same photo folder
    hint_min_points: int = 200         # a region needs >= 200 labelled floor points to vote
    hint_min_share: float = 0.6        # ... and one folder must hold >= 60% of them
    coverage_min_m2: float = 4.0       # a plan under 4 m2 of rooms gets meta.coverage_warning (judge B-8)

    # --- v4: tier-aware extraction (docs/modules/plan_beta.md "Part v4"); defaults = LiDAR behaviour, unchanged ---
    tier: str = "lidar"                # set by tiers.params_for_tier; recorded in plan.meta / Plan.tier
    surface_from_raw: bool = False     # T-1: add a surface (points + PCA normals) built from raw_points to the TSDF
    #                                    surface. Video: the TSDF (weight >= 5) keeps only ~1/3 of the walls seen in
    #                                    the 1-2 m band, so wall evidence and wall lines were missing (1 room, 23 m2)
    raw_surface_voxel_m: float = 0.03  # T-1: voxel of the raw-point surface (mean of the raw points in it)
    raw_surface_k: int = 24            # T-1: neighbours for the PCA normal (~10-15 cm radius at 3 cm voxels)
    raw_surface_max_curv: float = 0.05 # T-1: keep only planar neighbourhoods (smallest eigenvalue share < 5%)
    raw_surface_min_count: int = 2     # T-1: a voxel needs >= 2 raw points (single stray predictions are noise)
    geom_sigma_m: float = 0.0          # T-2: extra 1-sigma on every wall offset for the tier's surface noise (added
    #                                    in quadrature; LiDAR 0: its noise is already in the fit terms)
    use_folder_seeds: bool = False     # T-3 (photo): one watershed seed per room folder (scene cam_room / traj),
    #                                    no merging across folders, rooms named after folders, adjacency from links
    folder_seed_dt_frac: float = 0.5   # T-3: a folder's seed cameras are those with DT >= 50% of the folder's best
    #                                    camera (spin centre), not doorway photos that stand in the door
    folder_snap_m: float = 0.5         # T-3: a camera outside free space snaps to the nearest free cell within 0.5 m
    folder_hull_fill: bool = False     # T-4 (photo): fill each folder's seen free space to its convex hull minus walls
    use_folder_widen: bool = False     # T-3: multiply a room's interval half-widths by info.rooms[folder].widen

    # --- room segmentation (step 2) ---
    seed_h_m: float = 0.10             # a distance-transform peak must stand 10 cm above its saddle to seed a room
    door_max_m: float = 1.30           # constrictions narrower than this are doors (interior doors 0.6-1.2 m)
    door_min_m: float = 0.50           # ... and wider than this (narrower gaps are between furniture)
    jamb_search_m: float = 0.12        # a door must have wall evidence this close to both ends of its cut
    min_room_m2: float = 1.2           # smaller regions are merged into a neighbour (a WC is ~1.2 m2)
    unvisited_min_enclosure: float = 0.6   # a room never entered is kept only if >= 60% of its outline is measured wall
    brief_entry_s: float = 0.0         # D-079 (video): a region the camera was inside for less than this many seconds
    #                                    is partially observed (a step through a door, a track jump): not a room, not
    #                                    in the footprint, listed in meta.dropped_regions with its outline. 0 = off
    brief_keep_enclosed: bool = True   # D-079: ... unless >= unvisited_min_enclosure of its outline is measured wall
    stay_split: bool = False           # D-079 (video): split a region between camera stays (segment.split_by_stays)
    stay_radius_m: float = 1.0         # D-079: a stay = the camera within 1 m of one place ...
    stay_min_s: float = 4.0            # ... for >= 4 s (a door is passed in 1-2 s)
    stay_open_max_m: float = 2.0       # D-079: between stays, an opening up to 2 m with jambs separates (own hall 1.38 m)

    # --- wall fitting (step 3) ---
    line_min_support_m: float = 0.30   # a candidate wall line needs >= 30 cm of wall evidence along it
    line_merge_m: float = 0.04         # candidate lines closer than 4 cm are the same wall
    room_wall_search_m: float = 0.25   # wall evidence within 25 cm of a room's free space belongs to that room
    jog_max_m: float = 0.06            # steps shorter than this between parallel walls are snapped away
    cell_cover_min: float = 0.5        # an arrangement cell is inside the room if >= 50% covered by the room mask
    enclosed_cells: bool = True        # v3 S-3: also partly covered cells enclosed by the room and its measured walls
    enclosed_min_cover: float = 0.25   # v3 S-3: ... if >= 25% covered (the rest: floor hidden under furniture)
    enclosed_wall_min: float = 0.3     # v3 S-3: an outer side is a wall if it is a measured line and >= 30% of it has
                                       # inward-facing wall-band surface (doors, windows, wardrobes hide the rest)
    enclosed_open_max: float = 0.5     # v3 S-3: an inner side is open if < 50% of it has wall-band vertical surface
    refine_band_m: float = 0.04        # raw points within +-4 cm of the line feed the wall fit (a 1 deg tilt over
    #                                    4 m stays inside; a wider band lets the fit jump to a nearby surface)
    refine_max_shift_m: float = 0.06   # a wall may move <= 6 cm from its free-space position (else: another surface)
    refine_h_lo_m: float = 0.30        # above skirting boards ...
    refine_h_hi_m: float = 2.00        # ... and below the wall-ceiling junction
    refine_corner_gap_m: float = 0.15  # skip 15 cm at each end of a wall (corner rounding, adjacent wall points)
    refine_trim_schedule_m: tuple = (0.05, 0.03, 0.02, 0.015)   # ... trimmed down to +-1.5 cm of the fitted line
    wall_slope_mode: str = "axis"      # "axis": walls on the Manhattan axes, tilt goes into the interval (default);
    #                                    ablations: "room" = one fitted rotation per room, "free" = slope per wall
    yaw_min_wall_m: float = 1.0        # only walls >= 1 m vote for the room rotation
    max_slope_deg: float = 3.0         # walls may deviate this much from the Manhattan axes (drift, real walls)
    patch_m: float = 0.10              # 10 cm patches = independent samples for the standard error
    min_refine_points: int = 200       # fewer raw points -> wall status "inferred" (position from the raster)
    min_patches: int = 10              # ... and the points must cover >= 10 patches of 10x10 cm (0.1 m2 of wall)

    # --- v2 measurement (X-1 single visit, X-2 noise weights) ---
    single_visit: bool = False         # X-1: measure a room's walls from ONE contiguous visit. Evidence says OFF
    #                                    (same-topology median 1.9 -> 3.2 cm when on, plan_v2_ablation.md)
    visit_gap_s: float = 3.0           # camera outside the room for > 3 s ends a visit (a glance through a door
    #                                    while passing is not a new pass)
    visit_min_s: float = 2.0           # visits shorter than 2 s are passes through the room, not scans of it
    inconsistency_term: bool = True    # B-4: add sqrt(rmse^2 - sensor^2) to each wall sigma (alpha A-8)
    measure_h_lo_m: float | None = 0.8   # X-3: position from the tape-height band 0.8-1.6 m (None = full band);
    measure_h_hi_m: float = 1.6          #      existence/status still use 0.3-2.0 m (plan_v2_ablation.md X-3)
    noise_weights: bool = True         # X-2: weight raw points by 1 / sigma_lidar(range)^2 (measured, D-011)

    # --- v2 B-2: canonical outline after refinement (walls.canonicalize) ---
    canonical_outline: bool = True     # ablation switch
    canon_coplanar_m: float = 0.02     # two measured parallel walls <= 2 cm apart are one plane (2 x the ~1 cm
    #                                    within-capture offset sigma; drift between passes is 1.9-2.5 cm)
    canon_min_edge_m: float = 0.30     # an unsupported connecting edge shorter than 30 cm is not a wall ...
    canon_max_jog_m: float = 0.10      # ... and is dropped only if the step it makes is <= 10 cm

    # --- v6 (I-011): outline at openings, after B-2 (walls.outline_at_openings) ---
    opening_outline: bool = True       # ablation switch; LiDAR only (video/photo: off in tiers.TIER_OVERRIDES)
    header_far_min_m: float = 0.20     # a header's far face counts if seen over >= 20 cm of the open span ...
    header_far_reach_m: float = 1.0    # ... extended by 1 m each side (the same wall's far face beside the opening)
    opening_on_cut_m: float = 0.06     # an edge on a header cut: within 6 cm of the header's near face
    opening_jog_min_m: float = 0.05    # shallower excursions are jogs (canonicalize), not wall slabs (thickness_min_m)
    opening_min_cover: float = 0.3     # wide opening: header open spans over >= 30% of the excursion
    opening_sill_band: tuple = (0.30, 0.80)   # sill height: above skirting, below the lowest window sills (~0.8 m)
    opening_head_min: float = 0.3      # window recess: wall above it on the inner plane (header band; patchy, D-046)
    opening_recess_wall_min: float = 0.25  # ... the window seen at the recess back in the wall band (sim 38-42%)
    opening_recess_sill_max: float = 0.2   # ... and the recess back has (almost) no surface at sill height
    opening_max_m: float = 3.5         # an excursion longer than the widest opening (sim 2.9 m) is a wall, not a reveal

    # --- uncertainty (all 1-sigma, metres) ---
    lidar_sigma_m: float = 0.005       # per-surface systematic LiDAR bias (iPad/iPhone LiDAR ~1 cm at 95%)
    drift_sigma_m: float = 0.01        # residual pose drift of a wall measured within ONE visit: bounded by the
    #                                    ICP noise floor of consecutive fragments, 0.7-1.0 cm (drift.md 5.5)
    drift_mixed_sigma_m: float = 0.02  # v2 B-4: a wall whose points come from several visits carries the drift
    #                                    module's held-out residual between passes, 1.9-2.5 cm RMS (drift.md 5.5)
    drift_vertical_sigma_m: float = 0.005  # vertical drift is smaller: gravity is observed by the IMU at all times
    inferred_sigma_m: float = 0.05     # unsupported wall: its position comes from the free-space edge only
    mc_samples: int = 400              # Monte Carlo samples for area / perimeter / footprint intervals

    # --- thickness ---
    thickness_min_m: float = 0.05
    thickness_max_m: float = 0.40

    # --- ceiling ---
    ceiling_min_h_m: float = 1.9       # a ceiling is at least 1.9 m above the floor
    ceiling_shrink_m: float = 0.20     # stay 20 cm away from walls (cornices, wall points)
    ceiling_min_cover: float = 0.10    # >= 10% of the room's 10 cm cells must see the ceiling, else not observed
    level_tol_m: float = 0.02          # points within 2 cm of the level peak define the level
    level_subsample: int = 3           # every 3rd raw point for floor/ceiling levels (speed)
    ceiling_double_layer: bool = True  # v2 B-6 (alpha A-14): two big layers close together -> inferred, wide interval
    ceiling_ambiguous_m: float = 0.20  # alpha's value: drift doubles seen up to 11 cm; real soffits are usually deeper
    ceiling_rival_frac: float = 0.30   # a rival layer must cover >= 30% of what the chosen layer covers (alpha)

    # --- openings ---
    jamb_h_lo_m: float = 0.30          # jamb faces are measured between 0.3 m ...
    jamb_h_hi_m: float = 1.80          # ... and 1.8 m (below the door head, above skirting / thresholds)
    cut_band_m: float = 0.04           # half-thickness of the band around a door cut (inside a 10 cm wall)
    jamb_slice_m: float = 0.05         # one jamb edge estimate per 5 cm height slice
    jamb_mode: str = "alpha"           # v2 B-3: "alpha" = tall-face jamb rule first (v1 slices as fallback); "v1"
    jamb_search_alpha_m: float = 0.20  # search each jamb within +-20 cm of the neck/bump end (neck width is from
    #                                    the 2 cm DT, which the 4 cm wall dilation biases by up to ~8 cm)
    jamb_min_points: int = 15          # alpha: raw points needed on a jamb face
    jamb_min_cover: float = 0.40       # alpha: a jamb covers >= 40% of the 10 cm height bins in 0.3-1.8 m
    profile_bin_m: float = 0.05        # wall-plane occupancy grid for window / door search
    beyond_m: float = 0.10             # a LiDAR point >= 10 cm behind the wall plane means the ray went through
    window_min_w_m: float = 0.40
    window_min_h_m: float = 0.40
    window_sill_min_m: float = 0.30
    window_top_max_m: float = 2.30
    door_min_open_h_m: float = 1.80    # an opening that is open from the floor to >= 1.8 m is a door/passage
    mirror_match_m: float = 0.04       # reflected "through" points within 4 cm of real surfaces = mirror
    mirror_min_frac: float = 0.5
    mirror_min_w_m: float = 0.25       # a mirror narrower than this is noise, not a mirror
    ray_subsample: int = 8             # use every 8th raw ray for see-through statistics (speed)

    def to_dict(self) -> dict:
        return asdict(self)
