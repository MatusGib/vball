from vball.ball.tracknet import downscale_command, rescale_tracknet_csv, tracknet_command, tracknet_csv_path


def test_command_uses_streaming_nonoverlap_mode_and_posix_paths(tmp_path):
    video = tmp_path / "matches" / "1" / "work.mp4"
    cmd = tracknet_command(video, tmp_path / "out", tmp_path / "w.pt", batch_size=4, python="py")
    assert cmd[:2] == ["py", "predict.py"]
    assert cmd[cmd.index("--video_file") + 1] == video.as_posix()
    assert cmd[cmd.index("--tracknet_file") + 1] == (tmp_path / "w.pt").as_posix()
    assert cmd[cmd.index("--save_dir") + 1] == (tmp_path / "out").as_posix()
    assert cmd[cmd.index("--eval_mode") + 1] == "nonoverlap"
    assert cmd[cmd.index("--batch_size") + 1] == "4"
    assert "--large_video" in cmd
    assert "--inpaintnet_file" not in cmd


def test_csv_path_matches_predict_py_naming(tmp_path):
    assert tracknet_csv_path(tmp_path, tmp_path / "work.mp4") == tmp_path / "work_ball.csv"


def test_downscale_command_targets_tracknet_input_size(tmp_path):
    cmd = downscale_command(tmp_path / "work.mp4", tmp_path / "track.mp4")
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-vf") + 1] == "scale=512:288"
    assert "-an" in cmd
    assert cmd[cmd.index("-g") + 1] == "30"  # cheap seeks for the median background
    assert cmd[-1] == str(tmp_path / "track.mp4")


def test_rescale_csv_maps_visible_points_to_work_resolution(tmp_path):
    src = tmp_path / "small.csv"
    src.write_text("Frame,Visibility,X,Y\n0,1,256,144\n1,0,0,0\n")
    dst = tmp_path / "ball.csv"
    rescale_tracknet_csv(src, dst, sx=3.75, sy=3.75)
    assert dst.read_text().splitlines() == ["Frame,Visibility,X,Y", "0,1,960,540", "1,0,0,0"]


def test_command_passes_threshold(tmp_path):
    cmd = tracknet_command(tmp_path / "t.mp4", tmp_path / "out", tmp_path / "w.pt", threshold=0.3, python="py")
    assert cmd[cmd.index("--threshold") + 1] == "0.3"


def test_vendored_predict_has_threshold_patch():
    from vball.config import TRACKNET_DIR

    source = (TRACKNET_DIR / "predict.py").read_text()
    assert "--threshold" in source
    assert "y_pred > HEATMAP_THRESHOLD" in source
