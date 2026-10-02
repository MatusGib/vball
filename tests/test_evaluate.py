import math

import numpy as np

from vball.evaluate import (
    interval_iou,
    load_intervals_csv,
    match_intervals,
    rally_metrics,
    restrict_to_span,
    visible_fraction,
)


def test_interval_iou():
    assert interval_iou((0, 10), (0, 10)) == 1.0
    assert interval_iou((0, 10), (5, 15)) == 5 / 15
    assert interval_iou((0, 1), (2, 3)) == 0.0


def test_matching_is_one_to_one_and_respects_threshold():
    pred = [(0, 10), (1, 9), (50, 51)]
    gt = [(0, 10), (40, 60)]
    assert match_intervals(pred, gt, min_iou=0.5) == [(0, 0)]


def test_perfect_metrics():
    m = rally_metrics([(0, 10), (20, 30)], [(0, 10), (20, 30)])
    assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0)
    assert m.mean_abs_start_err_s == 0.0


def test_false_positive_and_miss():
    m = rally_metrics(pred=[(0, 10), (100, 110)], gt=[(0, 10), (20, 30)])
    assert m.n_matched == 1
    assert m.precision == 0.5 and m.recall == 0.5


def test_start_end_errors():
    m = rally_metrics([(0.5, 10.5)], [(0, 10)])
    assert math.isclose(m.mean_abs_start_err_s, 0.5) and math.isclose(m.mean_abs_end_err_s, 0.5)


def test_no_predictions():
    m = rally_metrics([], [(0, 10)])
    assert m.precision == 0.0 and m.recall == 0.0 and math.isnan(m.mean_abs_start_err_s)


def test_summary_mentions_precision_and_recall():
    assert "precision 1.00 recall 1.00" in rally_metrics([(0, 10)], [(0, 10)]).summary()


def test_restrict_to_span_keeps_only_labelled_region():
    pred = [(0, 10), (100, 110), (1000, 1010)]
    gt = [(98, 108), (200, 210)]
    assert restrict_to_span(pred, gt, margin_s=5) == [(100, 110)]


def test_visible_fraction_inside_intervals():
    visible = np.array([True, False, True, True, False, False])
    assert visible_fraction(visible, [(0.0, 2.0)], fps=1.0) == 0.5


def test_load_intervals_csv(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("start_s,end_s\n1.5,9.0\n20,31.25\n")
    assert load_intervals_csv(p) == [(1.5, 9.0), (20.0, 31.25)]
