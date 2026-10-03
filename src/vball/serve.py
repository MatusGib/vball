"""Serve check: a rally starts with the ball leaving a player who stands behind a baseline.

Between rallies the ball is thrown, rolled or lobbed back to the server, often over the net; those runs end
at the server instead of starting there (docs/superpowers/specs/2026-10-04-serve-check-net-design.md)."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vball.ball.track import BallTrack
from vball.court import COURT_LENGTH, COURT_WIDTH
from vball.players import Players


@dataclass(frozen=True)
class ServeParams:
    search_before_s: float = 1.0  # look for the server from run start - this ...
    search_after_s: float = 0.5  # ... to run start + this
    baseline_margin_m: float = 0.5  # foot y < margin or > 18 - margin counts as behind a baseline
    side_margin_m: float = 1.5  # foot x within [-margin, 9 + margin]
    reach_w: float = 0.75  # player box widened by this * box width each side
    reach_h: float = 0.5  # and by this * box height above the top (hand at the hit)
    leave_s: float = 0.7  # where the ball is this long after the contact decides
    min_serve_rally_s: float = 1.5  # replaces RallyParams.min_rally_s when the serve check is on


def reach_box(box: np.ndarray, p: ServeParams) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    return x1 - p.reach_w * w, y1 - p.reach_h * h, x2 + p.reach_w * w, y2


def _inside(pt: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _leaves(
    c: int, track_id: int, box: np.ndarray, track: BallTrack, players: Players, p: ServeParams, fps: float
) -> bool:
    """After contact frame c, does the ball end up outside the server's reach and farther from them?"""
    later = np.flatnonzero(track.visible[c + 1 : c + round(p.leave_s * fps) + 1])
    if len(later) == 0:
        return False
    f = c + 1 + int(later[-1])
    a, b = np.searchsorted(players.frame, [f, f + 1])
    same = np.flatnonzero(players.track_id[a:b] == track_id)
    if len(same):
        box = players.box[a + same[0]]  # the server's box when the ball is looked at again
    centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    start, end = (track.x[c], track.y[c]), (track.x[f], track.y[f])
    return not _inside(end, reach_box(box, p)) and math.dist(end, centre) > math.dist(start, centre)


def is_serve(
    start: int, track: BallTrack, players: Players, court_xy: np.ndarray, fps: float, p: ServeParams
) -> bool:
    """Does the active run starting at frame `start` begin with a serve? `players` must be sorted by frame;
    court_xy is each row's foot position in court metres (players.with_court)."""
    lo = max(0, start - round(p.search_before_s * fps))
    hi = min(len(track), start + round(p.search_after_s * fps) + 1)
    a, b = np.searchsorted(players.frame, [lo, hi])
    x, y = court_xy[a:b, 0], court_xy[a:b, 1]
    behind = (
        (x >= -p.side_margin_m) & (x <= COURT_WIDTH + p.side_margin_m)
        & ((y < p.baseline_margin_m) | (y > COURT_LENGTH - p.baseline_margin_m))
    )
    for i in a + np.flatnonzero(behind):
        c = int(players.frame[i])
        if track.visible[c] and _inside((track.x[c], track.y[c]), reach_box(players.box[i], p)):
            if _leaves(c, int(players.track_id[i]), players.box[i], track, players, p, fps):
                return True
    return False


def serve_filter(
    track: BallTrack, players: Players, court_xy: np.ndarray, fps: float, params: ServeParams = ServeParams()
) -> Callable[[int, int], bool]:
    """keep(start, end) for detect_rallies."""
    order = np.argsort(players.frame, kind="stable")
    players = Players(players.frame[order], players.track_id[order], players.box[order], players.score[order])
    court_xy = court_xy[order]
    return lambda start, end: is_serve(start, track, players, court_xy, fps, params)
