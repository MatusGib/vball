import numpy as np
from helpers import make_test_video, requires_ffmpeg

from vball.ball.external import clip_source, find_vballnet_clips
from vball.ball.pseudo import IGNORED, NEGATIVE, POSITIVE
from vball.video import probe


def make_clip(root, split, match, stem, rows, labelled=True):
    d = root / split / match
    (d / "video").mkdir(parents=True)
    (d / "csv").mkdir()
    make_test_video(d / "video" / f"{stem}.mp4", seconds=1, fps=30, size="640x360")
    name = f"{stem}_ball.csv" if labelled else f"{stem}_predict_ball.csv"
    lines = "".join(f"{f},{v},{x},{y}\n" for f, v, x, y in rows)
    (d / "csv" / name).write_text("Frame,Visibility,X,Y\n" + lines)


@requires_ffmpeg
def test_find_skips_prediction_only_clips(tmp_path):
    make_clip(tmp_path, "train", "match1", "a", [(0, 1, 10, 10)])
    make_clip(tmp_path, "test", "match2", "b", [(0, 1, 10, 10)], labelled=False)
    clips = find_vballnet_clips(tmp_path)
    assert [c.video.stem for c in clips] == ["a"]


@requires_ffmpeg
def test_clip_source_states_and_scaling(tmp_path):
    make_clip(tmp_path, "train", "match1", "a", [(0, 0, -1, -1), (1, 1, 320, 180), (5, 1, 640, 0)])
    src = clip_source(find_vballnet_clips(tmp_path)[0], tmp_path / "cache")
    assert probe(src.track_video).width == 512
    assert len(src.state) == 30
    assert src.state[0] == NEGATIVE  # hand-labelled "no ball"
    assert src.state[1] == POSITIVE and (src.x[1], src.y[1]) == (256.0, 144.0)
    assert src.state[2] == IGNORED  # no row for this frame
    assert np.isnan(src.x[0])
