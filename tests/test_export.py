from pathlib import Path

from helpers import make_test_video, requires_ffmpeg

from vball.export import clip_command, concat_list, export_rallies
from vball.video import probe


def test_clip_command_seeks_before_input_and_stream_copies():
    cmd = clip_command(Path("in.mp4"), Path("out.mp4"), 12.5, 20.0)
    assert cmd[cmd.index("-ss") + 1] == "12.500"
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-t") + 1] == "7.500"
    assert cmd[cmd.index("-c") + 1] == "copy"


def test_concat_list_escapes_quotes(tmp_path):
    p = tmp_path / "it's.mp4"
    assert concat_list([p]) == f"file '{p.resolve().as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"


@requires_ffmpeg
def test_export_rallies_concatenates_clips(tmp_path):
    src = make_test_video(tmp_path / "src.mp4", seconds=10)
    out = tmp_path / "condensed.mp4"
    clips = export_rallies(src, [(1.0, 3.0), (5.0, 8.0)], out, tmp_path / "clips")
    assert len(clips) == 2 and all(c.exists() for c in clips)
    assert abs(probe(out).duration_s - 5.0) <= 1.5
