"""Every threshold of the boundary-first plan extractor, in one place, with the reason for its value.

Values are physical (metres, fractions), never capture-specific: the same numbers must work on an unseen
apartment at the walk-in test. docs/modules/plan_alpha.md explains each choice and the evidence behind it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AlphaParams:
    # --- rasters -------------------------------------------------------------------------------------------
    res_m: float = 0.02                 # plan raster cell = TSDF voxel size; finer adds no information
    margin_m: float = 0.3               # empty border around the scene so lines/cells never touch the edge

    # --- wall evidence (step 1) -----------------------------------------------------------------------------
    wall_max_ny: float = 0.3            # |n_y| < 0.3: surface within ~17 deg of vertical counts as wall-like
    axis_tol_deg: float = 15.0          # normal within 15 deg of +-x or +-z belongs to that Manhattan family
    band_lo_m: float = 0.3              # ignore skirting boards / floor clutter below 0.3 m
    band_hi_m: float = 2.0              # ... and ceiling clutter above 2.0 m (or ceiling - 0.3 if lower)
    high_from_m: float = 1.2            # "tall" evidence: above most furniture (sofas, counters ~0.9 m)

    # --- line detection (step 2) --------------------------------------------------------------------------
    peak_bin_m: float = 0.01            # 1 cm bins for the 1-D offset histogram
    peak_min_points: int = 60           # a peak needs >= 60 surface voxels (~0.25 m^2 of wall)
    line_tol_m: float = 0.03            # points within 3 cm of a line support it (TSDF noise + 1 voxel)
    occ_min_points: int = 8             # an along-line 2 cm bin is "occupied" with >= 8 voxels (~16 cm of height)
    occ_high_min_points: int = 3        # ... "tall-occupied" with >= 3 voxels above high_from_m
    min_line_support_m: float = 0.4     # a line must be occupied over >= 0.4 m in total
    merge_tol_m: float = 0.03           # same-sign peaks closer than 3 cm are one surface

    # --- free space / interior evidence (step 4) ----------------------------------------------------------
    n_rays: int = 400_000               # LiDAR rays sampled for 2-D free-space carving (deterministic seed)
    ray_stop_short_m: float = 0.05      # do not carve the last 5 cm before the hit (that is the surface)
    floor_band_m: float = 0.15          # up-facing points within +-15 cm of the floor level count as floor
    furniture_top_max_m: float = 1.2    # up-facing surfaces below 1.2 m (beds, tables) are interior evidence
    traj_radius_m: float = 0.15         # space around the phone is empty; small so it never leaks through a wall

    # --- labelling energy (step 4) ------------------------------------------------------------------------
    w_unknown: float = 0.6              # cost per m^2 of calling an unobserved cell "inside"
    w_smooth_m: float = 0.25            # cost per m of boundary that is NOT backed by wall evidence
    w_traj: float = 100.0               # cost of calling a cell the camera passed through "outside"

    # --- rooms (step 5) -----------------------------------------------------------------------------------
    max_door_m: float = 1.3             # gaps up to this width in a wall are doors/passages, not open plan
    min_door_m: float = 0.5             # narrower gaps are occlusion/noise, not walkable openings
    barrier_min_points: int = 2         # a pixel with >= 2 tall vertical surface voxels blocks room flooding
    peak_h_m: float = 0.10              # a distance peak must rise 10 cm above its surroundings to seed a room
    neck_ratio: float = 0.85            # neck >= 85% of the narrower basin's width = no constriction = same room
    snap_frac: float = 0.5              # a cell joins a room when that room's basin covers >= 50% of its pixels
    min_room_area_m2: float = 1.0       # smaller components are slivers (wall cores, shelves) unless thin-door
    min_room_width_m: float = 0.6       # a room narrower than this everywhere is not walkable
    min_jog_m: float = 0.12             # steps < 12 cm: drift-duplicated lines, door reveals; not room geometry
    min_sliver_m: float = 0.3           # polygon parts narrower than 30 cm are wall cores / shafts, not walkable floor

    # --- measurement on raw points (D-007) ----------------------------------------------------------------
    fit_band_m: float = 0.06            # raw points within +-6 cm of the wall line are fit candidates
    corner_trim_m: float = 0.15         # ignore 15 cm at each wall end (corner rounding, adjacent wall)
    opening_trim_m: float = 0.05        # ... and 5 cm around detected openings (jambs, frames)
    min_fit_points: int = 150           # fewer raw points -> offset is "inferred" from the TSDF line
    patch_m: float = 0.10               # 10 cm patches = independent samples for the standard error
    lidar_sigma_m: float = 0.005        # per-surface LiDAR ranging bias that averaging cannot remove
    point_noise_m: float = 0.006        # per-point LiDAR noise: rmse of the flattest walls (10th pct, 5.8-6.3 mm)
    max_tilt_deg: float = 5.0           # a wall may deviate this much from its Manhattan axis (= manhattan_tol)
    drift_sigma_m: float = 0.01         # residual drift between two ends of a wall (until drift module says)
    inferred_sigma_m: float = 0.02      # offset taken from the 2 cm TSDF line without raw support
    mc_samples: int = 2000              # Monte Carlo samples for area / perimeter / footprint intervals

    # --- ceiling height ------------------------------------------------------------------------------------
    ceil_shrink_m: float = 0.2          # stay 20 cm away from walls (crown moulding, wall points)
    ceil_min_points: int = 300          # fewer down-facing raw points inside the room -> not observed
    ceil_min_cover: float = 0.15        # ... or seen over < 15% of the room's area -> not observed
    ceil_ambiguous_m: float = 0.2       # a second big ceiling layer this close widens the interval to cover both

    # --- openings ------------------------------------------------------------------------------------------
    door_free_min: float = 0.5          # >= 50% of the gap footprint must be observed free / floor
    door_jamb_top_m: float = 1.8        # jambs are measured below 1.8 m (door heads start at ~2.0 m)
    max_leaf_m: float = 1.0             # wider than a single door leaf -> reported as a passage
    min_thickness_m: float = 0.04       # two opposite faces 4-45 cm apart are the two sides of one wall
    max_thickness_m: float = 0.45
    min_overlap_m: float = 0.3          # ... if they face each other over >= 30 cm
    jamb_min_points: int = 15           # raw points needed on a jamb face to measure it
    jamb_min_cover: float = 0.4         # a jamb face covers >= 40% of the 10 cm height bins in 0.3-1.8 m
    jamb_search_m: float = 0.3          # look for each jamb up to 30 cm beyond its end of the free run
    door_blocked_max: float = 0.3       # >= 30% of the opening solid at body height -> not walkable, rejected
    door_solid_pts: int = 50            # "solid" 5 cm bin: >= 50 raw points (~10 cm of surface height)
    window_lo_m: float = 0.3            # window search band on exterior walls (sill above 0.3 m) ...
    window_hi_m: float = 2.2            # ... to 2.2 m (head below the ceiling)
    window_bin_m: float = 0.05          # 5 cm (along x height) wall-face raster for holes / see-through
    window_min_w_m: float = 0.3         # smallest window we report
    window_min_h_m: float = 0.3
    see_through_min_m: float = 0.15     # a point >= 15 cm behind the wall face was seen THROUGH the wall
    see_through_min_hits: int = 3       # rays through a 5 cm bin needed to call it see-through
    glass_min_area_m2: float = 0.25     # a no-return hole must be >= 0.25 m^2 to be reported as glass
    mirror_match_m: float = 0.03        # reflected phantom point within 3 cm of a real point -> mirror
    mirror_frac: float = 0.5            # >= 50% of see-through points explained by reflection -> mirror

    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
