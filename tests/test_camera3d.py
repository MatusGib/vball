import numpy as np
from helpers import make_camera

from vball.camera3d import fit_camera
from vball.court import LANDMARKS, NET_LANDMARKS, Calibration, Segment

W, H = 1920, 1080


def calibration(with_net):
    cam = make_camera()
    names = list(LANDMARKS)
    pts = cam.project_ref(np.array([(*LANDMARKS[n], 0.0) for n in names]))
    points = [{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(names, pts)]
    if with_net:
        net = cam.project_ref(np.array([(*NET_LANDMARKS[n], 2.43) for n in NET_LANDMARKS]))
        points += [{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(NET_LANDMARKS, net)]
    return Calibration.create(0, points, [])


def test_recovers_focal_length_and_position_from_the_floor_alone():
    cam, err = fit_camera(calibration(with_net=False), W, H)
    assert abs(cam.K[0, 0] - 1400) < 14
    assert np.allclose(cam.centre(), [4.5, -5.0, 1.5], atol=0.05)
    assert len(err) == 10 and err.max() < 0.05


def test_net_points_join_the_fit():
    cam, err = fit_camera(calibration(with_net=True), W, H)
    assert len(err) == 12 and err.max() < 0.05
    top = np.array([[0.0, 9.0, 2.43]])
    assert np.allclose(cam.project_ref(top), make_camera().project_ref(top), atol=0.1)


def test_projection_follows_camera_bumps_and_rays_point_back():
    shift = np.array([[1.0, 0, 40.0], [0, 1.0, -10.0], [0, 0, 1.0]])
    cam = make_camera([Segment(0, 100, np.eye(3)), Segment(100, 200, shift)])
    p = np.array([[3.0, 12.0, 2.0]])
    assert np.allclose(cam.project(p, 150), cam.project_ref(p) + [40.0, -10.0])
    assert np.allclose(cam.to_ref(cam.project(p, 150), 150), cam.project_ref(p))
    v = p[0] - cam.centre()
    assert np.allclose(cam.rays(cam.project_ref(p))[0], v / np.linalg.norm(v))
