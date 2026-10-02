import numpy as np

from vball.ball.metrics import ball_metrics, tolerance_px
from vball.ball.testset import BallTestItem
from vball.ball.track import BallTrack


def track_with(points: dict[int, tuple[float, float]], n: int = 10) -> BallTrack:
    visible = np.zeros(n, dtype=bool)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    for f, (px, py) in points.items():
        visible[f], x[f], y[f] = True, px, py
    return BallTrack(visible, x, y)


def test_tolerance_is_4px_at_tracknet_width():
    assert tolerance_px(512) == 4.0
    assert tolerance_px(1920) == 15.0


def test_counts_each_case():
    track = track_with({0: (100, 100), 1: (300, 300), 3: (50, 50)})
    items = [
        BallTestItem(0, "ball", 105, 100),  # TP (5 px away)
        BallTestItem(1, "ball", 100, 100),  # FP: detected somewhere else
        BallTestItem(2, "ball", 10, 10),  # FN: not detected
        BallTestItem(3, "none"),  # FP: detection where there is no ball
        BallTestItem(4, "none"),  # TN
        BallTestItem(5, "skip"),  # ignored
        BallTestItem(6, "todo"),  # ignored
    ]
    m = ball_metrics(track, items, tol_px=15.0)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 2, 1, 1)
    assert m.precision == 1 / 3 and m.recall == 0.5 and m.accuracy == 2 / 5
    assert "recall 0.50" in m.summary()


def test_empty_metrics_do_not_divide_by_zero():
    m = ball_metrics(track_with({}), [], tol_px=15.0)
    assert (m.precision, m.recall, m.accuracy) == (0.0, 0.0, 0.0)
