"""Grid search of rally-detection parameters against hand labels."""

import itertools
from dataclasses import dataclass, fields, replace

import numpy as np

from vball.ball.track import BallTrack
from vball.evaluate import rally_metrics
from vball.players import Players
from vball.rallies import RallyParams, detect_rallies
from vball.serve import ServeParams, serve_filter

GRID = {
    "min_speed": [0.05, 0.15],
    "window_s": [1.0, 2.0],
    "min_active_frac": [0.1, 0.15, 0.2, 0.3],
    "max_gap_s": [2.0, 3.0, 4.0, 5.0],
    "min_rally_s": [0.5, 1.0, 1.5, 2.0],
    "pre_pad_s": [0.5, 1.0],
    "post_pad_s": [0.5, 1.0, 1.5],
}
# with the serve check on, the other rally parameters stay at their defaults
SERVE_GRID = {
    "min_serve_rally_s": [0.5, 1.0, 1.5],
    "reach_w": [0.5, 0.75, 1.0],
    "reach_h": [0.25, 0.5, 1.0],
    "baseline_margin_m": [0.5, 1.5],
}
RALLY_KEYS = {f.name for f in fields(RallyParams)}


@dataclass
class TuneCase:
    track: BallTrack
    fps: float
    width: int
    height: int
    gt: list[tuple[float, float]]
    serve: tuple[Players, np.ndarray] | None = None  # (players, court_xy) for the serve check


@dataclass
class TuneResult:
    params: dict
    f1s: list[float]

    @property
    def mean_f1(self) -> float:
        return float(np.mean(self.f1s))


def grid_search(
    cases: list[TuneCase], grid: dict = GRID, base: RallyParams = RallyParams(), serve_base: ServeParams = ServeParams()
) -> list[TuneResult]:
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        rally_params = replace(base, **{k: v for k, v in params.items() if k in RALLY_KEYS})
        serve_params = replace(serve_base, **{k: v for k, v in params.items() if k not in RALLY_KEYS})
        f1s = []
        for c in cases:
            keep, rp = None, rally_params
            if c.serve is not None:
                keep = serve_filter(c.track, *c.serve, c.fps, serve_params)
                rp = replace(rally_params, min_rally_s=serve_params.min_serve_rally_s)
            rallies = detect_rallies(c.track, c.fps, c.width, c.height, rp, keep=keep)
            pred = [(r.start_s(c.fps), r.end_s(c.fps)) for r in rallies]
            f1s.append(rally_metrics(pred, c.gt).f1)
        results.append(TuneResult(params, f1s))
    return sorted(results, key=lambda r: r.mean_f1, reverse=True)
