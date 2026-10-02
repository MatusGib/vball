from helpers import FPS, make_track

from vball.tuning import TuneCase, grid_search


def test_grid_search_ranks_by_mean_f1():
    # one rally in which the ball is lost for 3.5 s (shorter gaps merge anyway through the padding)
    track = make_track(60 * FPS, flights=[(10, 12), (15.5, 18)])
    case = TuneCase(track, FPS, 1280, 720, gt=[(9.5, 19.5)])
    results = grid_search([case], grid={"max_gap_s": [2.0, 4.0]})
    assert [r.params["max_gap_s"] for r in results] == [4.0, 2.0]  # bridging the gap wins
    assert results[0].mean_f1 == 1.0 and results[1].mean_f1 < 1.0
    assert results[0].f1s == [1.0]
