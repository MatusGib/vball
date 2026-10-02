import math
from dataclasses import dataclass

import numpy as np

from vball.ball.track import BallTrack


@dataclass(frozen=True)
class Rally:
    start_frame: int  # inclusive
    end_frame: int  # exclusive

    def start_s(self, fps: float) -> float:
        return self.start_frame / fps

    def end_s(self, fps: float) -> float:
        return self.end_frame / fps


@dataclass(frozen=True)
class RallyParams:
    min_speed: float = 0.15  # frame diagonals per second; slower = held / rolling ball
    max_speed: float = 4.0  # faster = detector jumping between false positives
    window_s: float = 1.0  # smoothing window for the in-flight signal
    min_active_frac: float = 0.3  # fraction of in-flight frames in the window to count as active
    max_gap_s: float = 2.0  # merge active runs separated by at most this
    min_rally_s: float = 2.0  # drop runs shorter than this
    pre_pad_s: float = 1.0  # include the serve toss
    post_pad_s: float = 1.0  # include the ball landing


def ball_speed(track: BallTrack, fps: float, width: int, height: int) -> np.ndarray:
    """Speed between consecutive frames in frame diagonals per second; NaN if either frame is invisible."""
    speed = np.full(len(track), np.nan)
    speed[1:] = np.hypot(np.diff(track.x), np.diff(track.y)) * fps / math.hypot(width, height)
    return speed


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _merge(intervals: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def detect_rallies(
    track: BallTrack, fps: float, width: int, height: int, params: RallyParams = RallyParams()
) -> list[Rally]:
    n = len(track)
    speed = ball_speed(track, fps, width, height)
    in_flight = ((speed >= params.min_speed) & (speed <= params.max_speed)).astype(float)
    window = max(1, round(params.window_s * fps))
    activity = np.convolve(in_flight, np.ones(window) / window, mode="same")
    runs = _merge(_runs(activity >= params.min_active_frac), round(params.max_gap_s * fps))
    runs = [(s, e) for s, e in runs if e - s >= round(params.min_rally_s * fps)]
    pre, post = round(params.pre_pad_s * fps), round(params.post_pad_s * fps)
    padded = [(max(0, s - pre), min(n, e + post)) for s, e in runs]
    return [Rally(s, e) for s, e in _merge(padded, 0)]
