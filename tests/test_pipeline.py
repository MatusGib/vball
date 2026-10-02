from pathlib import Path

from helpers import make_test_video, make_track, requires_ffmpeg, write_tracknet_csv

from vball import store
from vball.config import Paths
from vball.pipeline import process_match, redetect
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
    assert abs(rallies[0]["start_s"] - 4.0) < 0.6
    assert abs(rallies[0]["end_s"] - 13.0) < 0.6


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
