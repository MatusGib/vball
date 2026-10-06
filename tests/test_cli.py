from pathlib import Path

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


def test_eval_warns_about_unapproved_labels(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    main(["redetect", str(match_id)])
    default_paths().gt_csv(match_id).write_text("start_s,end_s,approved\n10.0,18.0,1\n30.0,36.0,0\n")
    assert main(["eval", str(match_id)]) == 0
    assert "1 of 2 labels not approved" in capsys.readouterr().out


def test_eval_without_labels_explains(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    assert main(["eval", str(match_id)]) == 1
    assert "no labels" in capsys.readouterr().err


def test_unknown_match_returns_error(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch)
    assert main(["redetect", "999"]) == 1
    assert "no match with id 999" in capsys.readouterr().err


def test_track_to_custom_out_keeps_rallies(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    main(["redetect", str(match_id)])
    calls = []

    def fake_run_tracknet(video, out_csv, weights, threshold, small_video):
        calls.append({"threshold": threshold, "small_video": small_video})
        write_tracknet_csv(out_csv, make_track(1800))

    monkeypatch.setattr("vball.cli.run_tracknet", fake_run_tracknet)
    out = tmp_path / "exp.csv"
    assert main(["track", str(match_id), "--threshold", "0.3", "--out", str(out)]) == 0
    assert out.exists()
    assert calls[0]["threshold"] == 0.3
    assert calls[0]["small_video"] == default_paths().track_video(match_id)
    conn = store.connect(default_paths().db_path)
    assert len(store.get_rallies(conn, match_id)) == 2  # untouched


def test_track_default_replaces_ball_csv_and_redetects(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "vball.cli.run_tracknet",
        lambda video, out_csv, weights, threshold, small_video: write_tracknet_csv(out_csv, make_track(1800)),
    )
    assert main(["track", str(match_id)]) == 0
    assert "0 rallies" in capsys.readouterr().out


def test_balleval_scores_against_clicked_frames(tmp_path, monkeypatch, capsys):
    from vball.ball.testset import BallTestItem, save_ball_test

    match_id = seed(tmp_path, monkeypatch)  # ball flies 10-18 s and 30-36 s at y=360
    save_ball_test(
        default_paths().ball_test_csv(match_id),
        [BallTestItem(10 * 30 + 1, "ball", 60.0, 360.0), BallTestItem(100, "none"), BallTestItem(101)],
    )
    assert main(["balleval", str(match_id)]) == 0
    out = capsys.readouterr().out
    assert "TP 1 FP 0 FN 0 TN 1" in out
    assert "2 of 3 test frames labelled" in out


def test_balleval_without_test_frames_explains(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    assert main(["balleval", str(match_id)]) == 1
    assert "Ball check" in capsys.readouterr().err


def test_tunerallies_prints_current_and_best(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    default_paths().gt_csv(match_id).write_text("start_s,end_s\n10.0,18.0\n30.0,36.0\n")
    assert main(["tunerallies", str(match_id), "--top", "2"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("current defaults: mean F1 1.000")
    assert len(lines) == 3 and all(line.startswith("mean F1") for line in lines[1:])


def test_process_passes_weights_and_threshold_to_ball_tracker(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VBALL_DATA", str(tmp_path / "data"))
    seen = {}

    def fake_process_match(video, paths, name, encoder, ball_runner):
        ball_runner(tmp_path / "work.mp4", tmp_path / "ball.csv")
        return 7

    def fake_run_tracknet(video, out_csv, weights, threshold):
        seen.update(weights=weights, threshold=threshold)

    monkeypatch.setattr("vball.cli.pipeline.process_match", fake_process_match)
    monkeypatch.setattr("vball.cli.run_tracknet", fake_run_tracknet)
    assert main(["process", "m.mp4", "--weights", "base.pt", "--threshold", "0.5"]) == 0
    assert seen == {"weights": Path("base.pt"), "threshold": 0.5}
    assert "match 7 processed" in capsys.readouterr().out


def test_finetune_adds_vballnet_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("VBALL_DATA", str(tmp_path / "data"))
    captured = {}
    monkeypatch.setattr("vball.ball.finetune.prepare_sources", lambda conn, paths, ids: ["OWN"])
    monkeypatch.setattr("vball.ball.external.clip_sources", lambda root, cache_dir: ["EXT1", "EXT2"])
    monkeypatch.setattr(
        "vball.ball.train_data.build_cache",
        lambda sources, out_dir, n_windows: captured.update(sources=sources, n=n_windows),
    )
    monkeypatch.setattr("vball.ball.finetune.finetune", lambda *args, **kwargs: [0.0])
    argv = ["finetune", "--matches", "1", "--out", str(tmp_path / "v2.pt"), "--vballnet", str(tmp_path / "vn"),
            "--rebuild-cache", "--windows", "3000"]
    assert main(argv) == 0
    assert captured == {"sources": ["OWN", "EXT1", "EXT2"], "n": 3000}


def test_players_command_dispatches_backend(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr("vball.players.run_yolo", lambda video, out_csv: calls.append(("yolo", out_csv)))
    assert main(["players", str(match_id), "--backend", "yolo"]) == 0
    assert calls == [("yolo", default_paths().players_csv(match_id))]


def test_playereval_needs_calibration(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    from vball.players import save_players

    save_players(default_paths().players_csv(match_id), [(0, 1, 0, 0, 10, 10, 0.9)])
    assert main(["playereval", str(match_id)]) == 1
    assert "calibrate the court" in capsys.readouterr().err


def test_camera_needs_a_calibration(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    assert main(["camera", str(match_id)]) == 1
    assert "calibrate" in capsys.readouterr().err


def test_courtcopy_needs_a_calibrated_source(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    assert main(["courtcopy", str(match_id), str(match_id)]) == 1
    assert "calibrate" in capsys.readouterr().err
