import numpy as np
import pytest

from vball.court import LANDMARKS, Calibration, Segment, apply_h, fit_homography, load_calibration, reprojection_errors, save_calibration

# a plausible behind-the-baseline camera: court (m) -> image (px)
COURT_TO_IMAGE = np.array([[110.0, -20.0, 465.0], [0.0, -25.0, 1000.0], [0.0, 0.035, 1.0]])


def image_of(names):
    court = np.array([LANDMARKS[n] for n in names])
    return apply_h(COURT_TO_IMAGE, court), court


def test_fit_recovers_exact_homography():
    names = ["far_left_corner", "far_right_corner", "center_left", "center_right", "near_attack_left"]
    image, court = image_of(names)
    H = fit_homography(image, court)
    assert np.allclose(apply_h(H, image), court, atol=1e-6)
    assert reprojection_errors(H, image, court).max() < 1e-3  # float noise only


def test_needs_four_non_degenerate_points():
    image, court = image_of(["far_left_corner", "far_right_corner", "center_left"])
    with pytest.raises(ValueError):
        fit_homography(image, court)
    line = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    with pytest.raises(ValueError):
        fit_homography(line, line)


def test_calibration_follows_camera_segments(tmp_path):
    names = ["far_left_corner", "far_right_corner", "center_left", "center_right"]
    image, court = image_of(names)
    shift = np.array([[1.0, 0, 40.0], [0, 1.0, 0], [0, 0, 1.0]])  # camera bumped: picture moves 40 px right
    cal = Calibration.create(
        ref_frame=10,
        points=[{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(names, image)],
        segments=[Segment(0, 100, np.eye(3)), Segment(100, 200, shift)],
    )
    assert np.allclose(apply_h(cal.image_to_court_at(50), image), court, atol=1e-6)
    assert np.allclose(apply_h(cal.image_to_court_at(150), image + [40.0, 0.0]), court, atol=1e-6)
    save_calibration(tmp_path / "court.json", cal)
    again = load_calibration(tmp_path / "court.json")
    assert np.allclose(again.image_to_court_at(150), cal.image_to_court_at(150))
    assert again.points == cal.points and again.ref_frame == 10
