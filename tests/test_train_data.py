import cv2
import numpy as np
from helpers import make_test_video, requires_ffmpeg

from vball.ball.pseudo import IGNORED, NEGATIVE, POSITIVE
from vball.ball.train_data import (
    H,
    SEQ_LEN,
    W,
    TrainSource,
    WindowDataset,
    build_cache,
    build_input,
    choose_windows,
    heatmaps,
    read_median,
)


def test_heatmaps_are_discs_only_where_present():
    xy = np.zeros((SEQ_LEN, 2))
    xy[0] = (100, 50)
    present = np.zeros(SEQ_LEN, dtype=bool)
    present[0] = True
    hm = heatmaps(xy, present)
    assert hm.shape == (SEQ_LEN, H, W)
    assert hm[0, 50, 100] == 1 and hm[0, 50, 102] == 1 and hm[0, 50, 103] == 0
    assert hm[1:].sum() == 0


def test_choose_windows_mixes_positive_and_negative():
    state = np.full(200, IGNORED, dtype=np.int8)
    state[:50] = NEGATIVE
    state[100:110] = POSITIVE
    starts = choose_windows(state, n_pos=5, n_neg=3, rng=np.random.default_rng(0))
    assert len(starts) == 8
    windows = [state[s : s + SEQ_LEN] for s in starts]
    assert sum((w == POSITIVE).any() for w in windows) == 5
    assert sum((w == NEGATIVE).all() for w in windows) == 3


@requires_ffmpeg
def test_input_matches_tracknet_inference(tmp_path):
    from vball.ball.finetune import tracknet_modules

    video = make_test_video(tmp_path / "t.mp4", seconds=1, fps=30, size=f"{W}x{H}")
    median = read_median(video)
    _, dataset = tracknet_modules()
    theirs = dataset.Video_IterableDataset(
        str(video), seq_len=SEQ_LEN, sliding_step=SEQ_LEN, bg_mode="concat", median=np.moveaxis(median, -1, 0)
    )
    _, x_theirs = next(iter(theirs))
    cap = cv2.VideoCapture(str(video))
    frames = np.stack([cap.read()[1][..., ::-1] for _ in range(SEQ_LEN)])
    assert np.allclose(build_input(frames, median), x_theirs, atol=1e-6)


@requires_ffmpeg
def test_cache_round_trip(tmp_path):
    video = make_test_video(tmp_path / "t.mp4", seconds=2, fps=30, size=f"{W}x{H}")
    n = 60
    state = np.full(n, NEGATIVE, dtype=np.int8)
    state[20:30] = POSITIVE
    x = np.where(state == POSITIVE, 200.0, np.nan)
    y = np.where(state == POSITIVE, 100.0, np.nan)
    build_cache([TrainSource(video, state, x, y)], tmp_path / "cache", n_windows=6, neg_fraction=0.5, seed=0)
    ds = WindowDataset(tmp_path / "cache")
    assert len(ds) == 6
    xb, yb, mask = ds[0]
    assert xb.shape == (3 * (SEQ_LEN + 1), H, W) and yb.shape == (SEQ_LEN, H, W) and mask.shape == (SEQ_LEN,)
    positives = [ds[i] for i in range(len(ds)) if ds[i][1].sum() > 0]
    assert positives, "at least one window has a ball heatmap"
    _, yp, _ = positives[0]
    frame_with_ball = int(np.flatnonzero(yp.numpy().sum(axis=(1, 2)))[0])
    assert yp[frame_with_ball, 100, 200] == 1
