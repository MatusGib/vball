from pathlib import Path

from helpers import make_test_video, make_track, requires_ffmpeg, write_tracknet_csv

from vball import store
from vball.config import Paths
from vball.pipeline import process_match, redetect
from vball.rallies import RallyParams
from vball.video import probe


def fake_ball_runner(video: Path, out_csv: Path) -> None:
    n_frames = probe(video).n_frames
    write_tracknet_csv(out_csv, make_track(n_frames, flights=[(5.0, 12.0)]))


@requires_ffmpeg
def test_process_match_stores_detected_rallies(tmp_path):
    src = make_test_video(tmp_path / "match.mp4", seconds=20, fps=30)
    paths = Paths(tmp_path / "data")

    match_id = process_match(src, paths, encoder="libx264", ball_runner=fake_ball_runner)

    assert paths.work_video(match_id).exists()
    assert paths.ball_csv(match_id).exists()
    conn = store.connect(paths.db_path)
    match = store.get_match(conn, match_id)
    assert match["name"] == "match" and match["fps"] == 30.0
    rallies = store.get_rallies(conn, match_id)
    assert len(rallies) == 1
    assert abs(rallies[0]["start_s"] - (5.0 - RallyParams().pre_pad_s)) < 0.6
    assert abs(rallies[0]["end_s"] - (12.0 + RallyParams().post_pad_s)) < 0.6


@requires_ffmpeg
def test_redetect_replaces_rallies_from_cached_track(tmp_path):
    src = make_test_video(tmp_path / "match.mp4", seconds=20, fps=30)
    paths = Paths(tmp_path / "data")
    match_id = process_match(src, paths, encoder="libx264", ball_runner=fake_ball_runner)
    n_frames = probe(paths.work_video(match_id)).n_frames
    write_tracknet_csv(paths.ball_csv(match_id), make_track(n_frames, flights=[(1.0, 4.0), (10.0, 14.0)]))

    conn = store.connect(paths.db_path)
    rallies = redetect(conn, paths, match_id)

    assert len(rallies) == 2
    assert len(store.get_rallies(conn, match_id)) == 2


@requires_ffmpeg
def test_redetect_applies_the_serve_check_when_players_and_court_exist(tmp_path):
    import numpy as np

    from vball.court import LANDMARKS, Calibration, apply_h, save_calibration
    from vball.pipeline import serve_status
    from vball.players import save_players

    src = make_test_video(tmp_path / "match.mp4", seconds=20, fps=30)
    paths = Paths(tmp_path / "data")
    match_id = process_match(src, paths, encoder="libx264", ball_runner=fake_ball_runner)
    conn = store.connect(paths.db_path)
    assert serve_status(paths, match_id) == "serve check skipped: no player tracks"
    assert len(redetect(conn, paths, match_id)) == 1

    court_to_image = np.array([[20.0, 0.0, 70.0], [0.0, -10.0, 220.0], [0.0, 0.0, 1.0]])
    names = ["far_left_corner", "far_right_corner", "near_left_corner", "near_right_corner"]
    image = apply_h(court_to_image, np.array([LANDMARKS[n] for n in names]))
    points = [{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(names, image)]
    save_calibration(paths.court_json(match_id), Calibration.create(0, points, []))
    save_players(paths.players_csv(match_id), [(f, 1, 300, 10, 310, 30, 0.9) for f in range(600)])  # never near the ball

    assert serve_status(paths, match_id) == "serve check on"
    assert redetect(conn, paths, match_id) == []
