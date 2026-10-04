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

    # --- room segmentation (step 2) ---
    seed_h_m: float = 0.10             # a distance-transform peak must stand 10 cm above its saddle to seed a room
    door_max_m: float = 1.30           # constrictions narrower than this are doors (interior doors 0.6-1.2 m)
    door_min_m: float = 0.50           # ... and wider than this (narrower gaps are between furniture)
    jamb_search_m: float = 0.12        # a door must have wall evidence this close to both ends of its cut
    min_room_m2: float = 1.2           # smaller regions are merged into a neighbour (a WC is ~1.2 m2)
    unvisited_min_enclosure: float = 0.6   # a room never entered is kept only if >= 60% of its outline is measured wall

    # --- wall fitting (step 3) ---
    line_min_support_m: float = 0.30   # a candidate wall line needs >= 30 cm of wall evidence along it
    line_merge_m: float = 0.04         # candidate lines closer than 4 cm are the same wall
    room_wall_search_m: float = 0.25   # wall evidence within 25 cm of a room's free space belongs to that room
    jog_max_m: float = 0.06            # steps shorter than this between parallel walls are snapped away
    cell_cover_min: float = 0.5        # an arrangement cell is inside the room if >= 50% covered by the room mask
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

    # --- uncertainty (all 1-sigma, metres) ---
    lidar_sigma_m: float = 0.005       # per-surface systematic LiDAR bias (iPad/iPhone LiDAR ~1 cm at 95%)
    drift_sigma_m: float = 0.01        # residual pose drift per wall until the drift module supplies a value
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

    # --- openings ---
    jamb_h_lo_m: float = 0.30          # jamb faces are measured between 0.3 m ...
    jamb_h_hi_m: float = 1.80          # ... and 1.8 m (below the door head, above skirting / thresholds)
    cut_band_m: float = 0.04           # half-thickness of the band around a door cut (inside a 10 cm wall)
    jamb_slice_m: float = 0.05         # one jamb edge estimate per 5 cm height slice
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
