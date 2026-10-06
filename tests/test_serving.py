import numpy as np
from helpers import FPS
from test_serve import BOX, START, ball, serve_ball

from vball import serving
from vball.players import Players


def standing(rows_y, frames=range(START - 40, START + 40)):
    """One player per entry of rows_y (court y of their feet), standing still; no ball contact."""
    rows, xy = [], []
    for tid, y in enumerate(rows_y):
        for f in frames:
            rows.append((f, tid, *BOX, 0.9))
            xy.append((4.5, y))
    p = Players.from_rows(rows)
    order = np.argsort(p.frame, kind="stable")
    return Players(p.frame[order], p.track_id[order], p.box[order], p.score[order]), np.array(xy)[order]


def test_ball_leaving_a_player_behind_the_far_baseline_decides():
    p, xy = standing([19.0])
    end, frame, how = serving.serving_end(START, serve_ball(), p, xy, FPS)
    assert (end, how) == ("far", "contact") and frame is not None


def test_without_a_contact_the_end_with_more_players_behind_its_baseline_serves():
    p, xy = standing([-1.0, 19.0, 19.5])
    assert serving.serving_end(START, ball({}), p, xy, FPS)[::2] == ("far", "count")


def test_nobody_behind_a_baseline_is_unknown():
    p, xy = standing([3.0, 12.0])
    assert serving.serving_end(START, ball({}), p, xy, FPS) == (None, None, "none")


def rally(start, length, end, serve_after=0.0, **kw):
    return {"start_s": start, "end_s": start + length, "end": end, "how": "count", "serve_s": start + serve_after,
            "defense_near": None, "defense_far": None, **kw}


def test_winner_is_the_next_server_and_short_rallies_are_aces_or_errors():
    rows = serving.outcomes([
        rally(0, 2.0, "near"),  # near serves again next: ace
        rally(10, 2.0, "near"),  # far serves next: serve error
        rally(20, 8.0, "far"),  # long: in
        rally(30, 3.5, "far"),  # middle band, no receive seen: check
        rally(40, 3.5, "near", defense_far=True),  # middle band, the far end received it: in
        rally(50, 9.0, "near"),
    ])
    assert [r["winner"] for r in rows[:5]] == ["near", "far", "far", "near", "near"]
    assert [r["outcome"] for r in rows[:5]] == ["ace", "error", "in", "check", "in"]
    assert (rows[4]["score_near"], rows[4]["score_far"]) == (3, 2)


def test_short_rally_ignores_a_receive_seen():
    rows = serving.outcomes([rally(0, 1.5, "near", defense_far=True), rally(10, 9.0, "far")])
    assert rows[0]["outcome"] == "error"


def test_last_rally_winner_from_a_legal_set_end():
    rows = [rally(i * 10, 9.0, "near") for i in range(25)]  # near wins 24 in a row, then the last rally
    assert serving.outcomes(rows)[-1]["winner"] == "near"
    rows = [rally(i * 10, 9.0, "near" if i % 2 else "far") for i in range(10)]  # 4-5 midway: no legal end
    assert serving.outcomes(rows)[-1]["winner"] is None


def test_fixes_match_by_start_time():
    rows = [rally(0, 9.0, "near"), rally(10.2, 2.0, "near"), rally(20, 9.0, "near")]
    fixes = {"10.0": {"end": "far"}, "0.3": {"outcome": "error"}}
    out = serving.outcomes(rows, fixes)
    assert [r["end"] for r in out] == ["near", "far", "near"]
    assert out[1]["end_fixed"] and out[1]["end_auto"] == "near"
    assert out[0]["outcome"] == "error" and out[0]["outcome_auto"] == "in"
    assert out[1]["outcome"] == "error"  # far served, near served next


def test_team_summary_counts_side_outs():
    rows = serving.outcomes([rally(0, 9, "near"), rally(10, 9, "far"), rally(20, 9, "far"), rally(30, 2, "near"),
                             rally(40, 9, "near")])
    s = serving.team_summary(rows)
    # winners: far, far, near, near, ? ; near received twice (rallies 1, 2) and won one back
    assert s["near"]["received"] == 2 and s["near"]["sideout_pct"] == 0.5
    assert s["near"]["aces"] == 1 and s["far"]["serves"] == 2


def test_rotation_names_our_servers():
    rows = serving.outcomes([rally(i * 10, 9, e) for i, e in enumerate(["near", "near", "far", "near", None, "near",
                                                                         "far", "near"])])
    serving.assign_servers(rows, "near", ["A", "B"])
    assert [r["server"] for r in rows] == ["A", "A", None, "B", None, "B", None, "A"]
    per = {x["name"]: x for x in serving.player_serves(rows, {0: 70.0, 1: 80.0, 3: 60.0})}
    assert per["A"]["serves"] == 3 and per["A"]["top_kmh"] == 80.0 and per["B"]["measured"] == 1


def test_serving_csv_round_trip(tmp_path):
    rows = [rally(1.25, 5.0, "far", defense_near=True, defense_far=False), rally(9, 4, None)]
    serving.save_serving(tmp_path / "s.csv", rows)
    back = serving.load_serving(tmp_path / "s.csv")
    assert back[0]["end"] == "far" and back[0]["defense_near"] is True and back[1]["end"] is None
    assert back[1]["defense_far"] is None
