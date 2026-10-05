"""Video frames written as photos: the photo tier must read back the video's focal length and the frame times."""
import numpy as np

from floorplan.photo import images as I
from floorplan.video.rooms_as_photos import RoomSpec, crop_for_integer_f35, select_ceiling, select_turning, write_photo


def test_integer_f35_crop_keeps_the_focal():
    f35, cw, ch, f_after = crop_for_integer_f35(1280, 720, 945.5)
    assert (f35, cw, ch) == (28, 1274, 716)
    assert abs(f_after / 945.5 - 1) < 5e-4
    f35, cw, ch, f_after = crop_for_integer_f35(1920, 1080, 1500.0)
    assert cw <= 1920 and ch <= 1080 and abs(f_after / 1500.0 - 1) < 1e-3


def test_exif_round_trip_through_the_photo_loader(tmp_path):
    rgb = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8)
    f35, cw, ch, f_after = crop_for_integer_f35(1280, 720, 945.5)
    t0 = 1_791_126_735.0
    for k, t in enumerate((12.345, 15.5)):
        write_photo(rgb, tmp_path / "hall" / f"f{k}.jpg", f35, t0 + t, (cw, ch))
    photos = I.load_all(tmp_path)
    assert [p.rgb.shape for p in photos] == [(716, 1274, 3)] * 2
    assert photos[0].f35 == 28 and photos[0].f35_src == "exif_f35"
    assert abs((photos[1].time_s - photos[0].time_s) - 3.155) < 2e-3      # millisecond sub-seconds
    for long_side in (518, 1024):                                           # depth and SfM resolutions
        f, _ = I.focal_px(photos[0], long_side, 70.0, "diagonal")
        assert abs(f * 1274 / long_side / 945.5 - 1) < 5e-4


def test_turning_series_keeps_one_direction_and_skips_steep_frames():
    t = np.arange(0, 40, 1 / 30)
    yaw = np.where(t < 30, 12.0 * t, 360 - 20.0 * (t - 30))       # turn right 360 deg, then back left
    pitch = np.full_like(t, -22.0)
    pitch[(t > 9) & (t < 11)] = -60.0                                 # looking at the floor around yaw 120
    sharp = 100 + 50 * np.sin(7 * t) ** 2
    idx = select_turning(t, yaw, pitch, sharp, RoomSpec("hall", (0, 40), step_deg=55, max_frames=7))
    y = yaw[idx]
    assert np.all(np.diff(t[idx]) > 0) and np.all(np.diff(y) > 0)     # time order, one direction
    assert np.all(pitch[idx] > -40)
    assert len(idx) >= 5 and y[0] < 10 and y[-1] > 250
    assert all(t[i] < 30 for i in idx)                                 # the turn back adds nothing


def test_ceiling_frame_is_the_sharpest_upward_one():
    t = np.arange(0, 10, 0.1)
    pitch = np.where((t > 4) & (t < 6), 35.0, -20.0)
    sharp = np.ones_like(t)
    sharp[52] = 5.0
    assert select_ceiling(t, pitch, sharp, (0, 10)) == 52
    assert select_ceiling(t, pitch, sharp, (7, 10)) is None
