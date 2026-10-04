import numpy as np
from helpers import make_camera

from vball.ball.track import BallTrack
from vball.ball3d import G, fit_flight, flight_metrics, simulate, split_flights

FPS = 30


def test_simulate_without_drag_is_the_closed_form_arc():
    pos = simulate([1.0, 2.0, 3.0], [0.5, 10.0, 4.0], 31, FPS, k=0.0)  # frame 30 = 1 s
    assert np.allclose(pos[30], [1.5, 12.0, 3.0 + 4.0 - 0.5 * G], atol=1e-6)


def test_drag_slows_the_ball_and_batches_work():
    p0, v0 = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]), np.array([[0.0, 20.0, 0.0], [0.0, 20.0, 0.0]])
    pos, vel = simulate(p0, v0, 31, FPS, return_velocity=True)
    assert pos.shape == (2, 31, 3) and np.allclose(pos[0], pos[1])
    assert 13.0 < np.linalg.norm(vel[0, 30]) < 16.0  # ~14 m/s^2 drag at 20 m/s
    assert pos[0, 30, 1] < 20.0


def test_metrics_of_a_drag_free_arc():
    m = flight_metrics([4.5, 2.0, 3.0], [0.0, 10.0, 2.0], 30, FPS, k=0.0)
    assert abs(m["speed_kmh"] - np.sqrt(104) * 3.6) < 1e-6
    assert abs(m["apex_m"] - (3 + 2 * 2 / G - 0.5 * G * (2 / G) ** 2)) < 0.01
    assert abs(m["net_z_m"] - (3 + 2 * 0.7 - 0.5 * G * 0.49)) < 0.01
    t_land = (2 + np.sqrt(4 + 2 * G * 3)) / G
    assert abs(m["land_y"] - (2 + 10 * t_land)) < 0.02 and abs(m["land_x"] - 4.5) < 1e-6


def test_split_cuts_at_a_touch_and_a_long_gap():
    n = 200
    visible = np.zeros(n, dtype=bool)
    x, y = np.full(n, np.nan), np.full(n, np.nan)
    f = np.arange(0, 20)  # flight 1: moving right and falling
    x[f], y[f] = 100 + 20 * f, 200 + 0.5 * f**2
    g = np.arange(20, 40)  # touch at 20: comes back left and rises
    x[g], y[g] = x[19] - 20 * (g - 19), y[19] - 15 * (g - 19)
    h = np.arange(80, 100)  # after a 1.3 s gap
    x[h], y[h] = 300 + 5 * (h - 80), 300.0
    visible[np.r_[f, g, h]] = True
    pieces = split_flights(BallTrack(visible, x, y), 0, n, FPS)
    assert [(a, b) for a, b in pieces] == [(0, 20), (20, 40), (80, 100)]


def serve_observations(cam, n=30, p0=(4.5, -1.0, 2.8), v0=(0.0, 22.0, 2.0)):
    traj = simulate(np.array(p0), np.array(v0), n, FPS)
    return np.arange(n), cam.project_ref(traj)


def test_fit_recovers_a_noiseless_serve():
    cam = make_camera()
    frames, uv = serve_observations(cam)
    fit = fit_flight(cam, frames, uv, FPS)
    assert fit is not None and fit.rms_px < 0.5
    assert abs(np.linalg.norm(fit.v0) - np.linalg.norm([0.0, 22.0, 2.0])) * 3.6 < 1.0
    assert np.allclose(fit.p0, [4.5, -1.0, 2.8], atol=0.1)


def test_fit_ignores_a_few_wrong_detections():
    cam = make_camera()
    frames, uv = serve_observations(cam)
    uv[[7, 18]] += [[300.0, -200.0], [-250.0, 150.0]]
    fit = fit_flight(cam, frames, uv, FPS)
    assert fit is not None and fit.inlier_share >= 0.9
    assert abs(np.linalg.norm(fit.v0) - np.linalg.norm([0.0, 22.0, 2.0])) * 3.6 < 2.0


def test_fit_rejects_a_path_no_ball_can_fly():
    cam = make_camera()
    a = np.linspace(0, 2 * np.pi, 30)
    uv = np.stack([960 + 200 * np.cos(a), 400 + 200 * np.sin(a)], axis=1)  # a circle
    assert fit_flight(cam, np.arange(30), uv, FPS) is None
