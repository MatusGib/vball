"""Score a ball track against hand-clicked test frames (TrackNet's TP / FP / FN / TN convention)."""

import math
from dataclasses import dataclass

from vball.ball.testset import BallTestItem
from vball.ball.track import BallTrack


def tolerance_px(width: int) -> float:
    """TrackNet's 4 px tolerance at 512 px width, scaled to the work video."""
    return 4.0 * width / 512


@dataclass(frozen=True)
class BallMetrics:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n else 0.0

    def summary(self) -> str:
        return (
            f"ball: TP {self.tp} FP {self.fp} FN {self.fn} TN {self.tn} | "
            f"precision {self.precision:.2f} recall {self.recall:.2f} accuracy {self.accuracy:.2f}"
        )


def ball_metrics(track: BallTrack, items: list[BallTestItem], tol_px: float) -> BallMetrics:
    tp = fp = fn = tn = 0
    for item in items:
        if item.status not in ("ball", "none") or not 0 <= item.frame < len(track):
            continue
        detected = bool(track.visible[item.frame])
        if item.status == "ball":
            if not detected:
                fn += 1
            elif math.hypot(track.x[item.frame] - item.x, track.y[item.frame] - item.y) <= tol_px:
                tp += 1
            else:
                fp += 1  # detected, but somewhere else
        elif detected:
            fp += 1
        else:
            tn += 1
    return BallMetrics(tp, fp, fn, tn)
