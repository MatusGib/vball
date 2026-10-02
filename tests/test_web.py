from fastapi.testclient import TestClient

from vball import store
from vball.config import Paths
from vball.rallies import Rally
from vball.video import VideoInfo
from vball.web.app import create_app


def make_client(tmp_path):
    paths = Paths(tmp_path / "data")
    conn = store.connect(paths.db_path)
    match_id = store.add_match(conn, "Final", "src.mp4")
    store.set_video_info(conn, match_id, VideoInfo(1280, 720, 30.0, 900))
    store.replace_rallies(conn, match_id, [Rally(30, 300), Rally(450, 600)])
    conn.close()
    paths.match_dir(match_id).mkdir(parents=True)
    paths.work_video(match_id).write_bytes(bytes(range(256)) * 4)
    return TestClient(create_app(paths)), match_id


def test_list_matches(tmp_path):
    client, _ = make_client(tmp_path)
    matches = client.get("/api/matches").json()
    assert matches[0]["name"] == "Final" and matches[0]["n_rallies"] == 2


def test_rallies_in_seconds(tmp_path):
    client, match_id = make_client(tmp_path)
    rallies = client.get(f"/api/matches/{match_id}/rallies").json()
    assert (rallies[0]["start_s"], rallies[0]["end_s"]) == (1.0, 10.0)


def test_unknown_match_is_404(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/api/matches/999/rallies").status_code == 404


def test_video_supports_range_requests(tmp_path):
    client, match_id = make_client(tmp_path)
    res = client.get(f"/media/{match_id}/work.mp4", headers={"Range": "bytes=0-99"})
    assert res.status_code == 206
    assert len(res.content) == 100


def test_labels_start_empty(tmp_path):
    client, match_id = make_client(tmp_path)
    assert client.get(f"/api/matches/{match_id}/labels").json() == []


def test_labels_round_trip_sorted_with_approval(tmp_path):
    client, match_id = make_client(tmp_path)
    body = [
        {"start_s": 20.0, "end_s": 31.5, "approved": False},
        {"start_s": 1.5, "end_s": 9.0, "approved": True},
    ]
    res = client.put(f"/api/matches/{match_id}/labels", json=body)
    assert res.status_code == 200
    expected = [body[1], body[0]]
    assert res.json() == expected
    assert client.get(f"/api/matches/{match_id}/labels").json() == expected


def test_labels_reject_end_before_start(tmp_path):
    client, match_id = make_client(tmp_path)
    body = [{"start_s": 9.0, "end_s": 1.5, "approved": True}]
    assert client.put(f"/api/matches/{match_id}/labels", json=body).status_code == 422


def test_labels_unknown_match_is_404(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/api/matches/999/labels").status_code == 404
    assert client.put("/api/matches/999/labels", json=[]).status_code == 404


def test_index_page(tmp_path):
    client, _ = make_client(tmp_path)
    res = client.get("/")
    assert res.status_code == 200 and "<title>vball</title>" in res.text
    assert client.get("/app.js").status_code == 200
