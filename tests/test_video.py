from helpers import make_test_video, requires_ffmpeg

from vball.video import VideoInfo, ingest, ingest_command, probe, target_fps


def test_target_fps_rounds_and_caps():
    assert target_fps(29.97) == 30
    assert target_fps(59.94) == 60
    assert target_fps(25.0) == 25
    assert target_fps(120.0) == 60


def test_duration_from_frames():
    assert VideoInfo(width=1920, height=1080, fps=30.0, n_frames=90).duration_s == 3.0


def test_ingest_command_normalises_fps_height_and_gop(tmp_path):
    cmd = ingest_command(tmp_path / "in.mov", tmp_path / "out.mp4", fps=30, max_height=1080, encoder="h264_nvenc")
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-vf") + 1] == "fps=30,scale=-2:'min(ih,1080)'"
    assert cmd[cmd.index("-g") + 1] == "30"
    assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc"
    assert "-cq" in cmd
    assert cmd[-1] == str(tmp_path / "out.mp4")


@requires_ffmpeg
def test_probe_reads_synthetic_video(tmp_path):
    info = probe(make_test_video(tmp_path / "v.mp4", seconds=2, fps=25))
    assert (info.width, info.height, info.fps, info.n_frames) == (320, 240, 25.0, 50)


@requires_ffmpeg
def test_ingest_converts_fps_and_downscales(tmp_path):
    src = make_test_video(tmp_path / "v.mp4", seconds=1, fps=120)
    info = ingest(src, tmp_path / "work" / "work.mp4", max_height=120, encoder="libx264")
    assert (info.width, info.height, info.fps) == (160, 120, 60.0)
    assert abs(info.n_frames - 60) <= 1
