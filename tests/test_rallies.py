import numpy as np
from helpers import FPS, make_track

from vball.ball.track import BallTrack
from vball.rallies import Rally, ball_speed, detect_rallies

W, H = 1280, 720
N = 60 * FPS  # one minute of video


def detect(track):
    return detect_rallies(track, fps=FPS, width=W, height=H)


def test_speed_is_in_diagonals_per_second():
    track = make_track(N, flights=[(10, 12)])
    speed = ball_speed(track, FPS, W, H)
    diag = np.hypot(W, H)
    assert np.isnan(speed[10 * FPS])  # previous frame not visible
    assert abs(speed[10 * FPS + 5] - 600 / diag) < 1e-6


def test_single_flight_becomes_one_padded_rally():
    rallies = detect(make_track(N, flights=[(10, 18)]))
    assert len(rallies) == 1
    assert abs(rallies[0].start_frame - 9 * FPS) <= 15
    assert abs(rallies[0].end_frame - 19 * FPS) <= 15


def test_ball_held_still_is_not_a_rally():
    assert detect(make_track(N, statics=[(30, 45)])) == []


def test_short_gap_is_merged():
    assert len(detect(make_track(N, flights=[(10, 14), (15, 19)]))) == 1


def test_long_gap_splits_rallies():
    assert len(detect(make_track(N, flights=[(10, 14), (25, 29)]))) == 2


def test_blip_shorter_than_min_rally_is_dropped():
    assert detect(make_track(N, flights=[(30, 31)])) == []


def test_teleporting_false_positives_are_ignored():
    visible = np.zeros(N, dtype=bool)
    visible[300:600] = True
    x = np.full(N, np.nan)
    x[300:600] = np.where(np.arange(300) % 2 == 0, 100.0, 1200.0)
    y = np.where(visible, 300.0, np.nan)
    assert detect(BallTrack(visible=visible, x=x, y=y)) == []


def test_empty_track():
    assert detect(make_track(N)) == []


def test_rally_at_video_start_is_clipped_to_zero():
    rallies = detect(make_track(N, flights=[(0.2, 5)]))
    assert rallies[0].start_frame == 0


def test_rally_times():
    assert Rally(30, 300).start_s(30.0) == 1.0
    assert Rally(30, 300).end_s(30.0) == 10.0
