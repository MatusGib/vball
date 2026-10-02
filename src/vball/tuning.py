"""Grid search of rally-detection parameters against hand labels."""

import itertools
from dataclasses import dataclass, replace

import numpy as np

from vball.ball.track import BallTrack
from vball.evaluate import rally_metrics
from vball.rallies import RallyParams, detect_rallies

GRID = {
    "min_speed": [0.05, 0.15],
    "window_s": [1.0, 2.0],
    "min_active_frac": [0.1, 0.15, 0.2, 0.3],
    "max_gap_s": [2.0, 3.0, 4.0, 5.0],
    "min_rally_s": [0.5, 1.0, 1.5, 2.0],
    "pre_pad_s": [0.5, 1.0],
    "post_pad_s": [0.5, 1.0, 1.5],
}


@dataclass
class TuneCase:
    track: BallTrack
    fps: float
    width: int
    height: int
    gt: list[tuple[float, float]]


@dataclass
class TuneResult:
    params: dict
    f1s: list[float]

    @property
    def mean_f1(self) -> float:
        return float(np.mean(self.f1s))


def grid_search(cases: list[TuneCase], grid: dict = GRID, base: RallyParams = RallyParams()) -> list[TuneResult]:
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        rally_params = replace(base, **params)
        f1s = []
        for c in cases:
            rallies = detect_rallies(c.track, c.fps, c.width, c.height, rally_params)
            pred = [(r.start_s(c.fps), r.end_s(c.fps)) for r in rallies]
            f1s.append(rally_metrics(pred, c.gt).f1)
        results.append(TuneResult(params, f1s))
    return sorted(results, key=lambda r: r.mean_f1, reverse=True)
