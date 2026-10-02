from vball import store
from vball.rallies import Rally
from vball.video import VideoInfo


def make_db(tmp_path):
    conn = store.connect(tmp_path / "db" / "vball.db")
    match_id = store.add_match(conn, "Final", "C:/videos/final.mp4")
    store.set_video_info(conn, match_id, VideoInfo(1920, 1080, 30.0, 5400))
    return conn, match_id


def test_add_and_get_match(tmp_path):
    conn, match_id = make_db(tmp_path)
    match = store.get_match(conn, match_id)
    assert match["name"] == "Final"
    assert (match["width"], match["height"], match["fps"], match["n_frames"]) == (1920, 1080, 30.0, 5400)
    assert store.get_match(conn, 999) is None


def test_replace_rallies_overwrites(tmp_path):
    conn, match_id = make_db(tmp_path)
    store.replace_rallies(conn, match_id, [Rally(0, 30), Rally(60, 90)])
    store.replace_rallies(conn, match_id, [Rally(30, 300)])
    rallies = store.get_rallies(conn, match_id)
    assert len(rallies) == 1
    assert rallies[0]["idx"] == 0
    assert (rallies[0]["start_s"], rallies[0]["end_s"]) == (1.0, 10.0)


def test_list_matches_counts_rallies(tmp_path):
    conn, match_id = make_db(tmp_path)
    store.replace_rallies(conn, match_id, [Rally(0, 30), Rally(60, 90)])
    store.add_match(conn, "Empty", "x.mp4")
    matches = store.list_matches(conn)
    assert [m["name"] for m in matches] == ["Final", "Empty"]
    assert matches[0]["n_rallies"] == 2
    assert matches[0]["rally_frames"] == 60
    assert matches[1]["n_rallies"] == 0
