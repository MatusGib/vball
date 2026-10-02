import numpy as np

from vball.ball.pseudo import IGNORED, NEGATIVE, POSITIVE, clean_track, fill_gaps, pseudo_states
from vball.ball.track import BallTrack


def parabola_track(n=60):
    t = np.arange(n, dtype=float)
    x = 100 + 5 * t
    y = 400 - 12 * t + 0.25 * t**2
    return BallTrack(np.ones(n, dtype=bool), x, y)


def with_hidden(track, frames):
    visible = track.visible.copy()
    visible[list(frames)] = False
    return BallTrack(visible, np.where(visible, track.x, np.nan), np.where(visible, track.y, np.nan))


def test_short_gap_in_flight_is_filled_on_the_curve():
    truth = parabola_track()
    filled, count = fill_gaps(with_hidden(truth, range(20, 25)), max_gap=6, reach=12, max_rms_px=1.0)
    assert count == 5
    assert filled.visible[20:25].all()
    assert np.allclose(filled.x[20:25], truth.x[20:25]) and np.allclose(filled.y[20:25], truth.y[20:25])


def test_long_gap_is_not_filled():
    _, count = fill_gaps(with_hidden(parabola_track(), range(20, 30)), max_gap=6, reach=12, max_rms_px=1.0)
    assert count == 0


def test_gap_across_a_touch_is_not_filled():
    # a parabola fits a V shape surprisingly well (RMS < 2 px), so the residual alone can't catch touches
    track = parabola_track()
    x = track.x.copy()
    x[23:] = x[23] - 5 * np.arange(len(x) - 23)  # horizontal direction reverses at frame 23
    hidden = with_hidden(BallTrack(track.visible, x, track.y), range(21, 25))
    _, count = fill_gaps(hidden, max_gap=6, reach=12, max_rms_px=100.0)
    assert count == 0  # rejected by the horizontal-velocity check


def test_gap_where_ball_curves_upward_is_not_filled():
    t = np.arange(60, dtype=float)
    y = np.where(t < 23, 300 + 6 * t, 300 + 6 * 23 - 6 * (t - 23))  # falls, then is dug back up
    track = BallTrack(np.ones(60, dtype=bool), 100 + 5 * t, y)
    _, count = fill_gaps(with_hidden(track, range(21, 25)), max_gap=6, reach=12, max_rms_px=100.0)
    assert count == 0  # rejected: gravity can only bend the path downward (image y grows)


def test_gap_at_edge_or_with_too_few_points_is_not_filled():
    hidden = with_hidden(parabola_track(20), list(range(0, 3)) + list(range(5, 8)) + list(range(9, 20)))
    _, count = fill_gaps(hidden, max_gap=6, reach=12, max_rms_px=1.0)
    assert count == 0


def test_clean_removes_isolated_and_jumping_detections():
    track = parabola_track(30)
    visible = np.zeros(30, dtype=bool)
    visible[5:20] = True
    visible[27] = True  # isolated
    x = track.x.copy()
    x[12] += 500  # one-frame jump far off the path
    cleaned = clean_track(
        BallTrack(visible, np.where(visible, x, np.nan), np.where(visible, track.y, np.nan)), max_step_px=50
    )
    assert not cleaned.visible[27]
    assert not cleaned.visible[12]
    assert cleaned.visible[5:12].all() and cleaned.visible[13:20].all()


def test_states_positive_ignored_negative():
    visible = np.zeros(100, dtype=bool)
    visible[[20, 21, 70]] = True
    track = BallTrack(visible, np.where(visible, 1.0, np.nan), np.where(visible, 1.0, np.nan))
    state = pseudo_states(track, rallies=[(15, 30)], neg_margin=10)
    assert state[20] == state[21] == POSITIVE
    assert state[25] == IGNORED  # inside rally, not seen
    assert state[10] == IGNORED  # within the margin before the rally
    assert state[2] == NEGATIVE and state[50] == NEGATIVE
    assert state[70] == NEGATIVE  # detections in dead time are trained away
