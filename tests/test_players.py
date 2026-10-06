import json

import numpy as np

from vball.court import Calibration, LANDMARKS, Segment, apply_h
from vball.players import Players, feet_to_court, load_players, player_stats, ravel_to_csv, save_players, with_court

COURT_TO_IMAGE = np.array([[110.0, -20.0, 465.0], [0.0, -25.0, 1000.0], [0.0, 0.035, 1.0]])


def calibration():
    names = ["far_left_corner", "far_right_corner", "near_left_corner", "near_right_corner"]
    image = apply_h(COURT_TO_IMAGE, np.array([LANDMARKS[n] for n in names]))
    points = [{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(names, image)]
    return Calibration.create(0, points, [Segment(0, 10_000, np.eye(3))])


def box_at(court_xy, half_width=20, height=120):
    fx, fy = apply_h(COURT_TO_IMAGE, [court_xy])[0]
    return (fx - half_width, fy - height, fx + half_width, fy)


def test_feet_above_the_floor_horizon_have_no_court_position():
    # COURT_TO_IMAGE's horizon (court y -> infinity) is at image y = -25 / 0.035 = -714 px; a spectator on a balcony
    # whose feet are above it would otherwise map behind the camera, i.e. "behind the near baseline"
    rows = [(0, 1, *box_at((4.5, 3.0)), 0.9), (0, 2, 445.0, -920.0, 485.0, -800.0, 0.9)]
    xy = feet_to_court(np.array([0, 0]), Players.from_rows(rows).box, calibration())
    assert np.allclose(xy[0], [4.5, 3.0], atol=1e-6) and np.isnan(xy[1]).all()


def test_round_trip(tmp_path):
    rows = [(0, 3, 10.0, 20.0, 30.0, 140.0, 0.9), (1, 3, 11.0, 20.0, 31.0, 140.0, 0.8)]
    save_players(tmp_path / "p.csv", rows)
    p = load_players(tmp_path / "p.csv")
    assert p.frame.tolist() == [0, 1] and p.track_id.tolist() == [3, 3]
    assert np.allclose(p.box[1], [11, 20, 31, 140]) and np.allclose(p.score, [0.9, 0.8])


def test_ravel_json_conversion(tmp_path):
    payload = {"format": "ravel-vb-predictions-v1", "predictions": [
        {"frame_index": 4, "class_name": "player", "track_id": 7, "score": 0.7, "bbox_xyxy": [1, 2, 3, 4]},
        {"frame_index": 4, "class_name": "ball", "score": 0.9, "bbox_xyxy": [5, 6, 7, 8]},
    ]}
    (tmp_path / "r.json").write_text(json.dumps(payload))
    ravel_to_csv(tmp_path / "r.json", tmp_path / "p.csv")
    p = load_players(tmp_path / "p.csv")
    assert p.frame.tolist() == [4] and p.track_id.tolist() == [7]


def test_with_court_maps_feet_and_sides():
    rows = [(0, 1, *box_at((4.5, 3.0)), 0.9), (0, 2, *box_at((2.0, 15.0)), 0.9), (0, 3, *box_at((4.5, 30.0)), 0.9)]
    p = Players.from_rows(rows)
    court_xy, side, on_court = with_court(p, calibration())
    assert np.allclose(court_xy[0], [4.5, 3.0], atol=1e-6) and np.allclose(court_xy[1], [2.0, 15.0], atol=1e-6)
    assert side.tolist() == [0, 1, 1]  # 0 = near, 1 = far
    assert on_court.tolist() == [True, True, False]  # 30 m away: off court


def test_player_stats_counts_full_sides():
    rows = []
    for f in range(10):
        for i in range(6):
            rows.append((f, i, *box_at((1.0 + i, 3.0)), 0.9))  # 6 near
            rows.append((f, 10 + i, *box_at((1.0 + i, 15.0)), 0.9))  # 6 far
        if f >= 5:
            rows = [r for r in rows if not (r[0] == f and r[1] >= 13)]  # far side drops to 3 in frames 5-9
    stats = player_stats(Players.from_rows(rows), calibration(), rallies=[(0, 10)])
    assert stats.frames == 10
    assert stats.full_sides_share == 0.5
    assert stats.ids_per_side_per_rally == 6.0


def test_a_track_keeps_its_team_side():
    spots = [(4.5, 3.0)] * 8 + [(4.5, 10.0), (13.0, 3.0)]  # 8 frames near, one over the net line, one off court
    rows = [(f, 1, *box_at(xy), 0.9) for f, xy in enumerate(spots)]
    rows += [(f, 2, *box_at((14.0, 5.0)), 0.9) for f in range(10)]  # a spectator beside the court, never on it
    court_xy, side, on_court = with_court(Players.from_rows(rows), calibration())
    assert side[:10].tolist() == [0] * 10  # still near (our team) when over the line
    assert on_court[:10].tolist() == [True] * 10  # still a player when chasing the ball off court
    assert on_court[10:].tolist() == [False] * 10
    assert np.allclose(court_xy[8], [4.5, 10.0], atol=1e-6)  # positions are not changed
