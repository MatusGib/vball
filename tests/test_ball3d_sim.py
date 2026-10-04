import numpy as np
from helpers import make_camera

from vball.ball.track import BallTrack
from vball.ball3d_sim import KINDS, run_simulation, sample_flights

FPS = 30


def test_sampled_flights_follow_their_kind():
    rng = np.random.default_rng(0)
    for kind in KINDS:
        for f in sample_flights(kind, 6, FPS, rng):
            near = f.p0[1] < 9
            if kind == "serve":
                assert f.p0[1] < 0 or f.p0[1] > 18
            else:
                assert 6.0 <= f.p0[1] <= 12.0  # within 3 m of the net
            if kind == "set":
                assert np.linalg.norm(f.v0) < 9.5
            assert f.n >= 8
            if kind != "set":
                assert near == (f.v0[1] > 0)  # heads for the other side


def test_simulation_report_has_a_row_per_kind_and_noise():
    n = 3000
    track = BallTrack(np.ones(n, dtype=bool), np.zeros(n), np.zeros(n))
    rows = run_simulation(make_camera(), track, [(0, n)], FPS, 1920, 1080, count=3, noise_levels=[3.0], seed=1)
    assert [(r["kind"], r["noise_px"]) for r in rows] == [(k, 3.0) for k in KINDS]
    for r in rows:
        assert r["flights"] == 3 and 0.0 <= r["fitted"] <= 1.0
        assert set(r) >= {"speed_kmh_med", "speed_kmh_p90", "apex_m_med", "net_z_m_med", "land_m_med"}
