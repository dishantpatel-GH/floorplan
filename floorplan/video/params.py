"""Tunable parameters of the video tier, each with the reason it has its value (docs/modules/video_tier.md, "Decisions").

The video tier has its own parameter object instead of extending floorplan.config.Config because its knobs (motion
thresholds in image space, model resolutions, SfM settings) have no meaning for the LiDAR tier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class VideoParams:
    # --- keyframe selection from image motion alone (no poses in this tier) ---
    scan_width: int = 160                # motion/sharpness scan runs on tiny frames: fast and blur-insensitive
    kf_shift_frac: float = 0.15          # new keyframe after the image content moved 15% of the width (~9 deg) ...
    kf_max_dt_s: float = 1.0             # ... or 1 s passed (a slow walk still gets a keyframe every ~0.5 m)
    kf_sharp_window: int = 6             # pick the sharpest of the last 6 frames before the trigger (motion blur)

    # --- frame export ---
    sfm_long_side: int = 1024            # feature extraction resolution (hloc default for ALIKED)
    depth_long_side: int = 504           # DA3 processes at 504 px; <= 518 px keeps GPU use small (shared 8 GB GPU)

    # --- orientation and calibration (GeoCalib) ---
    orient_probe_frames: int = 12        # frames used to vote for the upright rotation
    gravity_every: int = 3               # run GeoCalib gravity on every 3rd keyframe (gravity is a global quantity)
    gravity_max_pitch_deg: float = 45.0  # GeoCalib is trained within +-45 deg pitch; steeper frames are not trusted
    flip_min_pitch_deg: float = 10.0     # flip to upside-down only if the median camera pitch is > +10 deg (Issue 15)

    # --- steep keyframes (D-084, steep.py): ceiling looks and straight-down looks ---
    steep_frames: bool = False           # leave them out of the scale votes and the fusion; a look at the end ends
                                         # the walk there, one in the middle cuts a scale segment after it
    steep_up_deg: float = 30.0           # take1's ceiling look peaks at +33..+40 deg, inside GeoCalib's 45; walks look
                                         # down (median pitch -15 to -33 deg on every capture)
    steep_down_deg: float = 50.0         # straight down, beyond GeoCalib's +-45 deg (gravity ignores them already)
    steep_min_kf: int = 3                # a look is at least 3 keyframes in a row; single ones are only left out
    steep_tail_kf: int = 20              # fewer keyframes after the last look: the walk ends where the look starts
    steep_up_half_kf: int = 15           # pitch against the local up (GeoCalib within +-15 keyframes): DPVO's
                                         # rotation drifts 15-20 deg on some take1 runs

    # --- visual odometry (DPVO, separate interpreter: envs/dpvo) ---
    dpvo_stride: int = 0                 # 0 = auto: DPVO sees ~dpvo_target_fps (v1 used 2 on the 60 fps sample)
    dpvo_target_fps: float = 30.0        # consecutive frames still overlap > 95% at 30 fps
    dpvo_height: int = 640               # upright frames resized to 640x480 (DPVO needs multiples of 16)
    dpvo_loop_closure: bool = False      # DPV-SLAM proximity loop closure (ablation flag, decision V-6)

    # --- structure from motion (hloc + pycolmap), used to self-calibrate the focal length ---
    calib_max_images: int = 120          # SfM on the first 120 keyframes is enough to calibrate one shared camera
    calib_min_registered: int = 20       # trust the SfM focal only if its model has >= 20 images
    max_keypoints: int = 2048
    seq_overlap: int = 10                # match each keyframe with its next 10 keyframes
    camera_model: str = "SIMPLE_RADIAL"  # one shared focal + 1 radial distortion term: enough for an iPhone main lens

    # --- camera path source: "dpvo" (frame-to-frame VO + scale step), "sfm" (global SfM, path_sfm.py) or
    #     "mapanything" (path_mapanything.py) ---
    path_source: str = "dpvo"
    # path_sfm.py: pairs that do not depend on the walk order, all models kept, metric scale from MoGe-2
    path_sfm_window: int = 8             # each keyframe with its next 8 keyframes
    path_sfm_retrieval_k: int = 10       # + its 10 most similar keyframes (VLAD of the ALIKED descriptors)
    path_sfm_long_every: int = 0         # + every keyframe against every n-th keyframe (0 = off)
    path_sfm_min_model: int = 10         # smaller models are dropped (their keyframes are filled or left out)
    path_sfm_threads: int = 1            # mapper threads; 1: the same matches give the same model
    path_sfm_abs_pose_min_inliers: int = 30      # a keyframe registers with >= 30 2D-3D inliers (COLMAP's value)
    path_sfm_abs_pose_min_inlier_ratio: float = 0.25  # ... and >= 25% of its matches (COLMAP's value)
    path_sfm_structure_less: bool = False        # registration from 2D-2D matches alone: off. Tested on single_room
                                                 # (5 models, 112 of 164 keyframes): 15 inliers, 15% and this on gave
                                                 # 1 model of 126 keyframes, but ARKit puts it +299% off in scale and
                                                 # 1.5 m off in shape, against +0.3, -0.7, -3.2, -6.0% and 1-7 cm for
                                                 # the strict models; on take1 it bent 27 keyframes (none before)
    path_sfm_max_local_scale: float = 1.25   # keyframes whose +-10-keyframe median MoGe/SfM depth ratio is off the
                                             # model's by > 1.25x are left out (a bent model; take1: 0.91-1.07 on the
                                             # default pairs, 2.85 with every keyframe linked to every 4th one)
    path_sfm_min_coverage: float = 0.8   # the SfM path replaces DPVO's only if its largest model holds >= 80% of
                                         # the keyframes; else the walk is in pieces and DPVO's path is kept. take1:
                                         # 93-99% on every cache; samples (ARKit): single_room 18-29% in 5-7 models
                                         # (pieces 0.2-6% off, joined plan -57%); floor_only 59%, a model folded by a
                                         # wrong link (2.4 m, 22 deg off ARKit)
    path_sfm_max_gap: int = 5            # unregistered keyframes filled from DPVO only in gaps of <= 5 (~2.5 s)
    path_sfm_fill_max_miss_m: float = 0.15   # ... and only if the fill meets the next registered keyframe within
                                             # 0.15 m + 30% of the gap's span
    path_sfm_link_window: int = 10       # keyframes each side of a link used to fit DPVO onto the models
    path_sfm_link_max_gap: int = 8       # models joined through a DPVO link only across <= 8 keyframes ...
    path_sfm_link_max_scale_ratio: float = 1.25  # ... if DPVO's metric scale agrees within 25% on both sides ...
    path_sfm_link_max_rot_deg: float = 5.0       # ... and its path fits both models within 5 deg ...
    path_sfm_link_max_pos_m: float = 0.15        # ... and 0.15 m + 10% of the window's path
    # path_mapanything.py: MapAnything windows chained into one metric path (docs/notes/video_mapanything_path.md)
    ma_window: int = 24                  # keyframes per window (24: 4.5 GB peak at 518 px, 6 s a window; 32: 5.0 GB)
    ma_overlap: int = 8                  # keyframes shared by consecutive windows; the chaining fit uses them
    ma_depth_input: bool = True          # MoGe-2 depth as metric depth input: MapAnything only registers the views
                                         # (take1 revisits: mean error 0.06 m with it, 0.21 m without)
    ma_chain: str = "points"             # "points": robust fit of the shared keyframes' depth points; "anchored":
                                         # the chain's poses as pose inputs (take1: drifts, revisit error 0.56 m)
    ma_sim3: bool = False                # "points": also fit a scale per link; off: each window keeps the metric scale
                                         # of its depth input (take1 link scales 0.95-1.07 would compound to 1.25)
    ma_average: bool = False             # a keyframe's pose = the average of its poses in every window of its segment

    # --- metric depth and scale ---
    depth_model: str = "moge2"           # "moge2" or "da3metric"; chosen by evidence (video_tier.md, decision V-5)
    scale_method: str = "auto"           # D-087: per segment, "pnp" where its votes are dense enough, else
                                         # "depth_agreement"; floor levelling on if one segment uses "pnp". The
                                         # default since 5 Oct, my call against the 09:40 rule (D-087): on the 9
                                         # take1 caches footprint error median 26% -> 3.2%, walls found 75 -> 84.
                                         # "depth_agreement" (D-076: Gaussian window, 1.5x clamp; the step before
                                         # D-075) and "pnp" (D-075 votes + vote-coverage self-check + floor
                                         # levelling) stay available (--video-scale).
    scale_auto_min_coverage: float = 0.5 # "auto": PnP's scale for a segment whose votes measure >= 50% of its
                                         # keyframes. Coverage: take1 0.67-1.00 on every segment of 20+ keyframes;
                                         # samples 0.31-0.44 where PnP is far off ARKit (+13.8% and +200.6% against
                                         # +3.3% and +18.1% for depth agreement), 0.51-0.67 elsewhere (PnP -12.7%
                                         # against -27.9% on with_ceiling, -5.9% against +29.6% on d066)
    scale_sigma_kf: float = 15.0         # depth_agreement: Gaussian time window (keyframes) for the local scale
    scale_max_gap: int = 4               # scale cost curves from keyframe pairs up to 4 keyframes apart
    scale_jump: float = 2.0              # a >2x step in the running-median scale = VO scale restart (segment cut)
    scale_min_contrast: float = 0.015    # D-066: a pair votes on scale only if +-20% changes its cost by this much
                                         # (was 0.005: near-rotation pairs at turns and doorway pauses voted with
                                         # arbitrary minima, local scale 0.002-45, fake jumps -> 15-36 segments)
    # fix loop (docs/FIX_LOOP.md): the local scale is the running median of PnP votes, pairs 2, 4, 6 keyframes apart.
    # Numbers in brackets are from my own video take1 (436 pairs solved).
    scale_vote_min_step_m: float = 0.08  # a pair votes only if PnP moved the camera >= 8 cm; shorter steps are mostly
                                         # turning (236 of 436 pairs are longer)
    scale_vote_max_dir_deg: float = 35.0 # ... DPVO's step points the same way within 35 deg (median 13.8, p75 31.1)
    scale_vote_max_rot_deg: float = 4.0  # ... and the two rotations agree within 4 deg (median 1.7, p90 3.9)
    scale_vote_half_kf: int = 8          # local scale = median of the >= 3 votes within +-8 keyframes (177 votes
                                         # measure 179 of 228 keyframes)
    bootstrap: int = 2000                # bootstrap resamples for the statistical part of the scale interval
    bootstrap_block: int = 10            # resample blocks of 10 consecutive pairs (neighbours are correlated)
    scale_model_sigma: float = 0.04      # systematic 1-sigma scale bias of the depth model (measured, decision V-7)

    # --- dense fusion of predicted depth ---
    depth_max_m: float = 4.0             # same range cap as LiDAR (D-006); monocular error grows with distance
    depth_edge_rel: float = 0.05         # drop pixels whose depth jumps > 5% to a neighbour (flying pixels)
    voxel_m: float = 0.02
    trunc_mult: float = 4.0
    weight_threshold: float = 5.0        # stricter than LiDAR (3): predicted depth is noisier, demand more views
    color_width: int = 512
    raw_stride: int = 4                  # raw points from every 4th pixel of the 378x504 depth map
    raw_voxel_m: float = 0.01            # 1 cm de-duplication grid

    # --- v2: focal-length uncertainty (video_tier.md v2, V2-2) ---
    focal_sigma_sfm: float = 0.04        # SfM self-calibration: 11 runs, errors -7.1..+4.1% (RMS ~3.9%); the 16:9 30 fps
                                         # phone copies are worse than the 4:3 originals (-0.7..-3.4%)
    focal_sigma_geocalib: float = 0.05   # GeoCalib median focal: -4.0, -0.8, +1.6, +2.5, +10.9% (RMS ~5.3%)
    focal_sigma_floor: float = 0.02      # both image estimates were biased the SAME way (short) on this data:
                                         # their fusion must not claim less than this
    focal_scale_sensitivity: float = 1.0 # scale error per focal error, measured ~1:1 (v1 Issue 13e)

    # --- v2: segments, pose graph, sheet ---
    max_walk_speed_mps: float = 1.5      # camera step across a VO restart larger than this x dt is not physical
    vo_rerun_segments: bool = True       # fresh DPVO run from every VO restart (V2-1)
    vo_rerun_min_keyframes: int = 60     # D-061: ... only for segments with at least this many keyframes
    vo_rerun_max: int = 8                # ... and at most this many (longest first): live-run time
    trust_max_scale_spread: float = 2.0  # segment self-check (V2-5): good 1.17-1.67, broken 2.18-2.25 (clamps hit);
                                         # scale_method "depth_agreement" only: the PnP scale follows DPVO's real
                                         # drift (spreads 2-107 on take1), so a spread says nothing about it (D-076)
    trust_max_residual: float = 0.058    # good 0.040-0.048, broken 0.063 (median truncated |log depth ratio|)
    trust_min_keyframes: int = 20        # shorter segments cannot be checked (and their scale is weak)
    trust_min_vote_coverage: float = 0.5 # D-076, "pnp": at least half of the segment's keyframes have >= 3 votes
                                         # within +-8 keyframes; elsewhere the scale is interpolated and DPVO's motion
                                         # is not measured. Sample segments that ARKit puts 113-213% off: 0.00-0.44;
                                         # take1: 0.67-1.00; single_room (ARKit +1.8 to +3.4%): 0.31-0.39 (one segment,
                                         # kept anyway). The vote residual (0.015-0.215) separates nothing: reported
    floor_level: bool | None = None      # D-076: move the camera path vertically so that every keyframe puts the
                                         # floor it sees at one height (a DPVO glitch times a large PnP scale dropped
                                         # take1's path 2.2 m). None: on with "pnp", off with "depth_agreement"
    floor_level_band_m: float = 0.30     # ... a keyframe's floor counts if its camera is within 0.30 m of the run's
                                         # median height above the floor (bed and table tops are 0.5-0.8 m closer)
    floor_level_half_kf: int = 8         # ... running median of the floor heights within +-8 keyframes (>= 3)
    drop_untrusted_segments: bool = True # leave untrusted segments' geometry out of the scene (unobserved)
    untrusted_sigma_floor: float = 0.25  # 1-sigma claimed when no segment passes the self-check
    use_pose_graph: bool = True          # join segments + loop closures on predicted-depth fragments (posegraph.py)
    use_sheet: bool = False              # D-067: opt-in; the paper-sheet scale cue (floorplan.scale) is off by default
    sheet_probe_per_segment: int = 40    # keyframes per segment searched for the sheet (looking down first)
    sheet_probe_total: int = 200         # D-061: at most ~this many in total (sim 1 BHK searched 1120 keyframes: 246 s)
    sheet_max_cues_per_segment: int = 6  # the same sheet seen in many frames is not independent evidence
    sheet_floor_rel_sigma: float = 0.05  # camera height above the RECONSTRUCTED floor vs truth: bias -4.7..+4.2%,
                                         # single-keyframe scatter 4.4% (sheet oracle, 3 segments, V2-6)

    seed: int = 0
    debug_dir: str = ""                  # where intermediate arrays are dumped (set to the work dir by the front-end)

    def to_dict(self) -> dict:
        return asdict(self)
