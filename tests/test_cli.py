from helpers import make_track, write_tracknet_csv

from vball import store
from vball.cli import main
from vball.config import default_paths
from vball.video import VideoInfo


def seed(tmp_path, monkeypatch):
    monkeypatch.setenv("VBALL_DATA", str(tmp_path / "data"))
    paths = default_paths()
    conn = store.connect(paths.db_path)
    match_id = store.add_match(conn, "m", "src.mp4")
    store.set_video_info(conn, match_id, VideoInfo(1280, 720, 30.0, 1800))
    conn.close()
    write_tracknet_csv(paths.ball_csv(match_id), make_track(1800, flights=[(10, 18), (30, 36)]))
    return match_id


def test_redetect_then_eval(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    assert main(["redetect", str(match_id)]) == 0
    assert "2 rallies" in capsys.readouterr().out

    gt = tmp_path / "gt.csv"
    gt.write_text("start_s,end_s\n10.0,18.0\n30.0,36.0\n")
    assert main(["eval", str(match_id), str(gt)]) == 0
    out = capsys.readouterr().out
    assert "precision 1.00 recall 1.00" in out
    assert "dead time removed" in out
    assert "ball visible in" in out


def test_eval_defaults_to_labels_saved_by_web_app(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    main(["redetect", str(match_id)])
    default_paths().gt_csv(match_id).write_text("start_s,end_s\n10.0,18.0\n30.0,36.0\n")
    assert main(["eval", str(match_id)]) == 0
    assert "precision 1.00 recall 1.00" in capsys.readouterr().out


def test_eval_without_labels_explains(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    assert main(["eval", str(match_id)]) == 1
    assert "no labels" in capsys.readouterr().err


def test_unknown_match_returns_error(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch)
    assert main(["redetect", "999"]) == 1
    assert "no match with id 999" in capsys.readouterr().err
