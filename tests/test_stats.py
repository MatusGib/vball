import numpy as np

from vball.players import Players
from vball.stats import classify_flights, leaderboards, load_meta, player_stats, rally_stats, save_meta


def flight(rally, frame, p0, v0, speed, apex, net_z, land=(None, None), fitted=True):
    keys = ["p0_x", "p0_y", "p0_z", "v0_x", "v0_y", "v0_z"]
    return {"rally": rally, "start_frame": frame, "fitted": fitted, **dict(zip(keys, [*p0, *v0])),
            "speed_kmh": speed, "apex_m": apex, "net_z_m": net_z, "land_x": land[0], "land_y": land[1]}


def test_rally_stats():
    s = rally_stats([(0.0, 10.0), (20.0, 25.0), (30.0, 60.0)], 100.0)
    assert s["count"] == 3 and s["total_s"] == 45.0 and abs(s["dead_share"] - 0.55) < 1e-9
    assert s["longest"][0] == {"time_s": 30.0, "length_s": 30.0} and s["median_s"] == 10.0
    assert rally_stats([], 100.0) == {"count": 0}


def test_flights_are_classified_with_plausibility():
    rows = [
        flight(0, 30, (4, -1, 2.8), (0, 19, 2), 70.0, 3.2, 2.9, (4, 14)),  # serve, near
        flight(0, 60, (5, 7, 2.2), (0, 0.5, 6), 22.0, 4.5, None),  # set
        flight(0, 90, (4, 8, 3.0), (0, 20, -4), 75.0, 3.0, 2.8, (4, 12)),  # attack
        flight(0, 120, (4, 3, 1.0), (0, 2, 5), 20.0, 2.0, None),  # a pass far from the net: none of them
        flight(1, 300, (4, 20, 3), (0, -20, 1), 120.0, 3.4, 1.5),  # serve, far, implausible
        flight(1, 330, (4, 8, 3), (0, 20, -4), 0.0, 0.0, 2.9, fitted=False),  # not fitted: ignored
    ]
    c = classify_flights(rows, fps=30.0)
    assert [(s["rally"], s["side"], s["plausible"]) for s in c["serves"]] == [(0, "near", True), (1, "far", False)]
    assert c["serves"][0]["time_s"] == 1.0
    assert [s["apex_m"] for s in c["sets"]] == [4.5] and c["sets"][0]["plausible"]
    assert [a["speed_kmh"] for a in c["attacks"]] == [75.0] and c["attacks"][0]["plausible"]


def test_player_stats_per_side_inside_rallies_only():
    rows = [(f, 1, 0, 0, 10, 10, 0.9) for f in range(10)] + [(f, 2, 0, 0, 10, 10, 0.9) for f in range(10)]
    rows += [(f, 3, 0, 0, 10, 10, 0.9) for f in range(10)] + [(f, 4, 0, 0, 10, 10, 0.9) for f in range(20, 25)]
    players = Players.from_rows(sorted(rows))
    xy = np.array([{1: (2.5, 3.0), 2: (6.5, 6.0), 3: (4.5, 12.0), 4: (4.5, 2.0)}[r[1]] for r in sorted(rows)])
    s = player_stats(players, xy, np.ones(len(rows), dtype=bool), [(0, 10)])
    assert s["rally_frames"] == 10
    assert s["near"]["players_per_frame"] == 2.0 and s["far"]["players_per_frame"] == 1.0
    assert s["near"]["mean_net_distance_m"] == 4.5 and s["far"]["mean_net_distance_m"] == 3.0
    heat = np.array(s["near"]["heat"])
    assert heat.shape == (26, 13) and heat.max() == 1.0 and heat[4 + 3, 2 + 2] == 1.0  # row y+4, column x+2


def test_leaderboards_rank_plausible_items_across_matches():
    def stats(mid, speeds):
        serves = [{"time_s": 1.0, "side": "near", "speed_kmh": v, "plausible": v <= 100} for v in speeds]
        return {"id": mid, "name": f"m{mid}", "our_side": None, "flights": {"serves": serves, "sets": [], "attacks": []},
                "rallies": {"count": 1, "longest": [{"time_s": 0.0, "length_s": 10.0 * mid}]}}
    lb = leaderboards([stats(1, [60.0, 120.0]), stats(2, [80.0])], n=2)
    assert [(s["match_id"], s["speed_kmh"]) for s in lb["serves"]] == [(2, 80.0), (1, 60.0)]
    assert [r["match_id"] for r in lb["rallies"]] == [2, 1]


def test_meta_defaults_and_round_trip(tmp_path):
    assert load_meta(tmp_path / "meta.json") == {"our_side": None}
    save_meta(tmp_path / "meta.json", {"our_side": "far"})
    assert load_meta(tmp_path / "meta.json") == {"our_side": "far"}


def test_physically_impossible_flights_are_not_plausible():
    rows = [
        flight(0, 30, (4, -1, 2.8), (0, 25, 3), 90.0, 5.0, 4.7, (4, 14)),  # 90 km/h serve 2.3 m over the tape
        flight(0, 60, (5, 7, 5.8), (0, 0.5, 0.5), 6.0, 5.9, None),  # "set" starting 5.8 m up
        flight(0, 90, (4, 8, 4.6), (0, 20, -4), 75.0, 4.6, 3.0, (4, 12)),  # "attack" hit 4.6 m up
    ]
    c = classify_flights(rows, fps=30.0)
    assert [x["plausible"] for k in ("serves", "sets", "attacks") for x in c[k]] == [False, False, False]
