import json

import numpy as np
import pytest

from vball.court import (
    LANDMARKS,
    Calibration,
    Segment,
    apply_h,
    calibration_json,
    fit_homography,
    load_calibration,
    reprojection_errors,
    save_calibration,
)

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


NET = [{"landmark": "net_left_top", "x": 300.0, "y": 100.0}, {"landmark": "net_right_top", "x": 900.0, "y": 105.0}]


def floor_points(names):
    image, court = image_of(names)
    return [{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(names, image)], image, court


def test_net_points_are_kept_out_of_the_floor_fit(tmp_path):
    points, image, court = floor_points(["far_left_corner", "far_right_corner", "center_left", "center_right"])
    shift = np.array([[1.0, 0, 40.0], [0, 1.0, 0], [0, 0, 1.0]])
    cal = Calibration.create(0, points + NET, [Segment(0, 100, np.eye(3)), Segment(100, 200, shift)])
    assert cal.points == points and cal.net_points == NET and cal.net_height_m == 2.43
    assert np.allclose(apply_h(cal.image_to_court, image), court, atol=1e-6)
    save_calibration(tmp_path / "court.json", cal)
    again = load_calibration(tmp_path / "court.json")
    assert again.net_points == NET and again.net_height_m == 2.43
    body = calibration_json(again)
    assert len(body["points"]) == 4 and body["net_height_m"] == 2.43
    assert body["segments"][0]["net_image"] == [[300.0, 100.0], [900.0, 105.0]]
    assert body["segments"][1]["net_image"] == [[340.0, 100.0], [940.0, 105.0]]  # moved with the camera


def test_calibration_without_net_points(tmp_path):
    points, _, _ = floor_points(["far_left_corner", "far_right_corner", "center_left", "center_right"])
    save_calibration(tmp_path / "court.json", Calibration.create(0, points, []))
    data = json.loads((tmp_path / "court.json").read_text())
    del data["net_points"], data["net_height_m"]  # a file saved before net points existed
    (tmp_path / "court.json").write_text(json.dumps(data))
    cal = load_calibration(tmp_path / "court.json")
    assert cal.net_points == [] and cal.net_height_m == 2.43
    assert calibration_json(cal)["segments"][0]["net_image"] == [None, None]


def test_shifted_copy_moves_every_point_and_is_marked_as_a_draft(tmp_path):
    from vball.court import shifted_copy

    points, image, court = floor_points(["far_left_corner", "far_right_corner", "center_left", "center_right"])
    src = Calibration.create(5, points + NET, [])
    assert src.copied_from is None
    cal = shifted_copy(src, (40.0, -10.0), ref_frame=300, segments=[Segment(0, 1000, np.eye(3))], copied_from=1)
    assert cal.copied_from == 1 and cal.ref_frame == 300
    assert np.allclose(apply_h(cal.image_to_court, image + [40.0, -10.0]), court, atol=1e-6)
    assert cal.net_points[0]["x"] == NET[0]["x"] + 40.0 and cal.net_points[0]["y"] == NET[0]["y"] - 10.0
    save_calibration(tmp_path / "court.json", cal)
    again = load_calibration(tmp_path / "court.json")
    assert again.copied_from == 1 and calibration_json(again)["copied_from"] == 1
