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


def test_grid_search_with_serve_check_drops_runs_without_a_server():
    import numpy as np

    from vball.players import Players

    track = make_track(60 * FPS, flights=[(10, 18), (30, 38)])  # ball starts each flight at (40, 360)
    frames = range(int(9.5 * FPS), 11 * FPS)
    server = Players.from_rows([(f, 1, 0.0, 300.0, 80.0, 500.0, 0.9) for f in frames])
    court_xy = np.tile([4.5, -1.0], (len(server), 1))
    gt = [(9.5, 19.5)]
    plain = grid_search([TuneCase(track, FPS, 1280, 720, gt)], grid={"max_gap_s": [2.0]})
    served = grid_search(
        [TuneCase(track, FPS, 1280, 720, gt, serve=(server, court_xy))], grid={"min_serve_rally_s": [1.5]}
    )
    assert plain[0].f1s[0] < 1.0  # the second flight is a false rally
    assert served[0].f1s == [1.0]
