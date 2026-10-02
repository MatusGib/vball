from pathlib import Path

from vball.config import Paths, default_paths


def test_match_file_layout(tmp_path):
    paths = Paths(tmp_path)
    assert paths.db_path == tmp_path / "vball.db"
    assert paths.match_dir(3) == tmp_path / "matches" / "3"
    assert paths.work_video(3) == tmp_path / "matches" / "3" / "work.mp4"
    assert paths.ball_csv(3) == tmp_path / "matches" / "3" / "ball.csv"
    assert paths.gt_csv(3) == tmp_path / "matches" / "3" / "gt_rallies.csv"
    assert paths.track_video(3) == tmp_path / "matches" / "3" / "track.mp4"
    assert paths.ball_test_csv(3) == tmp_path / "matches" / "3" / "ball_test.csv"
    assert paths.ball_train_dir == tmp_path / "ball_train"


def test_default_paths_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VBALL_DATA", str(tmp_path / "elsewhere"))
    assert default_paths().data_dir == Path(tmp_path / "elsewhere")
