"""8-frame training windows for TrackNet, built exactly like its inference input and cached on disk."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset

from vball.ball.pseudo import IGNORED, NEGATIVE, POSITIVE
from vball.ball.tracknet import TRACKNET_H as H
from vball.ball.tracknet import TRACKNET_W as W

SEQ_LEN = 8
RADIUS = 2.5  # TrackNet's label disc radius at 512x288
_YY, _XX = np.mgrid[0:H, 0:W]


@dataclass
class TrainSource:
    track_video: Path  # 512x288 copy
    state: np.ndarray  # per-frame POSITIVE / NEGATIVE / IGNORED
    x: np.ndarray  # ball x at 512x288 (NaN where unknown)
    y: np.ndarray


def read_median(video: Path, max_samples: int = 1800) -> np.ndarray:
    """Median background (H, W, 3) RGB uint8, sampled the way TrackNet's inference does."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, n // max_samples)
    frames = []
    i = 0
    while cap.grab():
        if i % step == 0:
            frames.append(cap.retrieve()[1])
        i += 1
    cap.release()
    return np.ascontiguousarray(np.median(np.stack(frames), axis=0)[..., ::-1]).astype(np.uint8)


def build_input(frames_rgb: np.ndarray, median_rgb: np.ndarray) -> np.ndarray:
    """(SEQ_LEN, H, W, 3) uint8 frames + (H, W, 3) median -> (3 * (SEQ_LEN + 1), H, W) float32 in [0, 1]."""
    stack = np.concatenate([median_rgb[None], frames_rgb], axis=0)
    return stack.transpose(0, 3, 1, 2).reshape(-1, H, W).astype(np.float32) / 255.0


def heatmaps(xy: np.ndarray, present: np.ndarray) -> np.ndarray:
    out = np.zeros((SEQ_LEN, H, W), dtype=np.float32)
    for i in np.flatnonzero(present):
        out[i] = ((_XX - xy[i, 0]) ** 2 + (_YY - xy[i, 1]) ** 2 <= RADIUS**2).astype(np.float32)
    return out


def choose_windows(state: np.ndarray, n_pos: int, n_neg: int, rng: np.random.Generator) -> np.ndarray:
    """Window start frames: n_pos windows with a known ball, n_neg windows entirely in dead time."""
    windows = sliding_window_view(state, SEQ_LEN)
    starts = np.arange(len(windows))
    pos = starts[(windows == POSITIVE).any(axis=1)]
    neg = starts[(windows == NEGATIVE).all(axis=1)]
    pick_pos = rng.choice(pos, size=min(n_pos, len(pos)), replace=False)
    pick_neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
    return np.sort(np.concatenate([pick_pos, pick_neg]))


def build_cache(
    sources: list[TrainSource], out_dir: Path, n_windows: int = 2000, neg_fraction: float = 0.2, seed: int = 0
) -> None:
    """Choose windows across sources (in proportion to their positive frames), decode only the frames
    they need into a uint8 memmap, and save labels next to it."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = np.array([(s.state == POSITIVE).sum() for s in sources], dtype=float)
    shares = weights / weights.sum()
    picks = []
    for src, share in zip(sources, shares):
        k = max(1, round(n_windows * share))
        picks.append(choose_windows(src.state, round(k * (1 - neg_fraction)), round(k * neg_fraction), rng))

    needed = [np.unique((starts[:, None] + np.arange(SEQ_LEN)).ravel()) for starts in picks]
    frames = np.lib.format.open_memmap(
        out_dir / "frames.npy", mode="w+", dtype=np.uint8, shape=(sum(len(f) for f in needed), H, W, 3)
    )
    windows, xy, state, source, medians = [], [], [], [], []
    offset = 0
    for s, (src, starts, frame_ids) in enumerate(zip(sources, picks, needed)):
        row_of = {int(f): offset + r for r, f in enumerate(frame_ids)}
        cap = cv2.VideoCapture(str(src.track_video))
        wanted = set(row_of)
        i = 0
        while wanted and cap.grab():
            if i in wanted:
                frames[row_of[i]] = cap.retrieve()[1][..., ::-1]
                wanted.discard(i)
            i += 1
        cap.release()
        for start in starts:
            span = np.arange(start, start + SEQ_LEN)
            windows.append([row_of[int(f)] for f in span])
            xy.append(np.stack([src.x[span], src.y[span]], axis=1))
            state.append(src.state[span])
            source.append(s)
        medians.append(read_median(src.track_video))
        offset += len(frame_ids)
    frames.flush()
    np.savez(
        out_dir / "meta.npz",
        windows=np.array(windows, dtype=np.int64),
        xy=np.nan_to_num(np.array(xy, dtype=np.float32)),
        state=np.array(state, dtype=np.int8),
        source=np.array(source, dtype=np.int64),
        medians=np.stack(medians),
        videos=np.array([str(s.track_video) for s in sources]),
    )


class WindowDataset(Dataset):
    def __init__(self, cache_dir: Path):
        self.frames = np.load(cache_dir / "frames.npy", mmap_mode="r")
        meta = np.load(cache_dir / "meta.npz")
        self.windows, self.xy, self.state = meta["windows"], meta["xy"], meta["state"]
        self.source, self.medians = meta["source"], meta["medians"]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        x = build_input(np.asarray(self.frames[self.windows[i]]), self.medians[self.source[i]])
        y = heatmaps(self.xy[i], self.state[i] == POSITIVE)
        mask = (self.state[i] != IGNORED).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)
