import numpy as np
from helpers import FPS

from vball.ball.track import BallTrack
from vball.players import Players
from vball.serve import ServeParams, is_serve, serve_filter

N = 20 * FPS
START = 300  # active run start (frame)
BOX = (600.0, 400.0, 680.0, 600.0)  # server, 80 x 200 px; reach box x 540..740, y 300..600


def ball(points):
    """points: {frame: (x, y)}"""
    visible = np.zeros(N, dtype=bool)
    x, y = np.full(N, np.nan), np.full(N, np.nan)
    for f, (px, py) in points.items():
        visible[f], x[f], y[f] = True, px, py
    return BallTrack(visible, x, y)


def serve_ball():
    # held at the hand for 10 frames, then hit up and away (15 px/frame)
    pts = {f: (640.0, 380.0) for f in range(START - 10, START + 1)}
    pts.update({f: (640.0 + 5 * (f - START), 380.0 - 15 * (f - START)) for f in range(START + 1, START + 40)})
    return ball(pts)


def return_ball():
    # lobbed in from far away, caught and held by the player
    pts = {f: (640.0, 100.0 + 14 * (f - (START - 20))) for f in range(START - 20, START)}
    pts.update({f: (640.0, 400.0) for f in range(START, START + 40)})
    return ball(pts)


def players(court_y, box=BOX, frames=range(START - 40, START + 40)):
    rows = [(f, 7, *box, 0.9) for f in frames]
    court_xy = np.tile([4.5, court_y], (len(rows), 1))
    return Players.from_rows(rows), court_xy


def check(track, court_y, box=BOX):
    p, xy = players(court_y, box)
    return is_serve(START, track, p, xy, FPS, ServeParams())


def test_serve_from_behind_the_near_baseline():
    assert check(serve_ball(), -1.0)


def test_serve_from_behind_the_far_baseline():
    assert check(serve_ball(), 19.0)


def test_ball_returned_to_the_server_is_not_a_serve():
    assert not check(return_ball(), -1.0)


def test_throw_from_mid_court_is_not_a_serve():
    assert not check(serve_ball(), 5.0)


def test_ball_not_near_anyone_is_not_a_serve():
    assert not check(serve_ball(), -1.0, box=(100.0, 400.0, 180.0, 600.0))


def test_serve_long_before_the_run_is_ignored():
    p, xy = players(-1.0)
    assert not is_serve(START + 5 * FPS, serve_ball(), p, xy, FPS, ServeParams())


def test_filter_sorts_players_by_frame():
    p, xy = players(-1.0)
    order = np.random.default_rng(0).permutation(len(p))
    shuffled = Players(p.frame[order], p.track_id[order], p.box[order], p.score[order])
    keep = serve_filter(serve_ball(), shuffled, xy[order], FPS)
    assert keep(START, START + 100)
