"""Automatic ball labels (pseudo-labels) for fine-tuning TrackNet from its own tracks."""

import math

import numpy as np

from vball.ball.track import BallTrack
from vball.rallies import RallyParams, runs

POSITIVE, NEGATIVE, IGNORED = 1, 0, -1


def _with_visible(track: BallTrack, visible: np.ndarray, x=None, y=None) -> BallTrack:
    x = track.x if x is None else x
    y = track.y if y is None else y
    return BallTrack(visible, np.where(visible, x, np.nan), np.where(visible, y, np.nan))


def clean_track(track: BallTrack, max_step_px: float, neighbor_frames: int = 2) -> BallTrack:
    """Drop detections with no other detection within ±neighbor_frames, or that jump more than
    max_step_px per frame away from every such neighbour (false positives)."""
    visible = track.visible
    keep = visible.copy()
    n = len(track)
    for i in np.flatnonzero(visible):
        steps = [
            math.hypot(track.x[j] - track.x[i], track.y[j] - track.y[i]) / abs(j - i)
            for j in range(max(0, i - neighbor_frames), min(n, i + neighbor_frames + 1))
            if j != i and visible[j]
        ]
        if not steps or min(steps) > max_step_px:
            keep[i] = False
    return _with_visible(track, keep)


def fill_gaps(
    track: BallTrack,
    max_gap: int = 6,
    reach: int = 12,
    min_side: int = 3,
    max_rms_px: float = 3.0,
    max_dvx_px: float = 1.5,
    max_lift_px: float = 0.05,
) -> tuple[BallTrack, int]:
    """Fill short gaps from a quadratic fit through the detections around them, only where the ball is
    in free flight: small fit residual, the same horizontal speed on both sides (no horizontal force),
    and a path that does not bend upward (gravity bends it down, i.e. image y'' >= 0).

    Returns the filled track and the number of frames filled."""
    visible = track.visible.copy()
    x, y = track.x.copy(), track.y.copy()
    n = len(track)
    filled = 0
    for start, end in runs(~track.visible):
        if end - start > max_gap or start == 0 or end >= n:
            continue
        left = [j for j in range(max(0, start - reach), start) if track.visible[j]][-4:]
        right = [j for j in range(end, min(n, end + reach)) if track.visible[j]][:4]
        if len(left) < min_side or len(right) < min_side:
            continue
        t = np.array(left + right, dtype=float)
        fx = np.polyfit(t, track.x[left + right], 2)
        fy = np.polyfit(t, track.y[left + right], 2)
        residual = (np.polyval(fx, t) - track.x[left + right]) ** 2 + (np.polyval(fy, t) - track.y[left + right]) ** 2
        if math.sqrt(residual.mean()) > max_rms_px:
            continue
        vx_left = np.polyfit(left, track.x[left], 1)[0]
        vx_right = np.polyfit(right, track.x[right], 1)[0]
        if abs(vx_left - vx_right) > max_dvx_px or fy[0] < -max_lift_px:
            continue  # a touch happened inside the gap
        gap = np.arange(start, end)
        x[gap] = np.polyval(fx, gap)
        y[gap] = np.polyval(fy, gap)
        visible[gap] = True
        filled += end - start
    return _with_visible(track, visible, x, y), filled


def pseudo_states(track: BallTrack, rallies: list[tuple[int, int]], neg_margin: int) -> np.ndarray:
    """Per-frame training state: POSITIVE where the ball is known inside a rally, NEGATIVE in dead time
    further than neg_margin frames from any rally, IGNORED elsewhere (rally frames the model missed)."""
    n = len(track)
    state = np.full(n, NEGATIVE, dtype=np.int8)
    for start, end in rallies:
        state[max(0, start - neg_margin) : min(n, end + neg_margin)] = IGNORED
    for start, end in rallies:
        inside = slice(max(0, start), min(n, end))
        state[inside] = np.where(track.visible[inside], POSITIVE, IGNORED)
    return state


def make_pseudo_labels(
    track: BallTrack, rallies: list[tuple[int, int]], fps: float, width: int, height: int
) -> tuple[BallTrack, np.ndarray, dict]:
    """Clean, gap-fill and label a match's ball track. Thresholds scale with the video size."""
    diag = math.hypot(width, height)
    cleaned = clean_track(track, max_step_px=RallyParams().max_speed * diag / fps)
    scale = width / 512  # thresholds are defined at TrackNet's 512 px width
    filled, n_filled = fill_gaps(cleaned, max_rms_px=3.0 * scale, max_dvx_px=1.5 * scale, max_lift_px=0.05 * scale)
    state = pseudo_states(filled, rallies, neg_margin=round(2 * fps))
    stats = {
        "removed": int(track.visible.sum() - cleaned.visible.sum()),
        "filled": n_filled,
        "positive": int((state == POSITIVE).sum()),
        "ignored": int((state == IGNORED).sum()),
        "negative": int((state == NEGATIVE).sum()),
    }
    return filled, state, stats
