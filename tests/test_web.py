import cv2
import numpy as np
from fastapi.testclient import TestClient
from helpers import make_test_video, make_track, requires_ffmpeg, write_tracknet_csv

from vball import store
from vball.config import Paths
from vball.court import LANDMARKS, apply_h
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


def test_ball_test_is_sampled_once_from_rallies(tmp_path):
    client, match_id = make_client(tmp_path)  # rallies at frames 30-300 and 450-600
    first = client.get(f"/api/matches/{match_id}/ball-test?n=10").json()
    assert len(first) == 10
    assert all(i["status"] == "todo" for i in first)
    assert all(30 <= i["frame"] < 300 or 450 <= i["frame"] < 600 for i in first)
    assert client.get(f"/api/matches/{match_id}/ball-test?n=50").json() == first  # persisted, not resampled


def test_ball_test_update(tmp_path):
    client, match_id = make_client(tmp_path)
    frame = client.get(f"/api/matches/{match_id}/ball-test?n=5").json()[0]["frame"]
    res = client.put(f"/api/matches/{match_id}/ball-test/{frame}", json={"status": "ball", "x": 12.5, "y": 30.0})
    assert res.json() == {"frame": frame, "status": "ball", "x": 12.5, "y": 30.0}
    assert client.get(f"/api/matches/{match_id}/ball-test").json()[0]["status"] == "ball"
    assert client.put(f"/api/matches/{match_id}/ball-test/99999", json={"status": "none"}).status_code == 404
    assert client.put(f"/api/matches/{match_id}/ball-test/{frame}", json={"status": "maybe"}).status_code == 422


@requires_ffmpeg
def test_frame_jpeg(tmp_path):
    client, match_id = make_client(tmp_path)  # work video is a placeholder; replace with a real one
    video = tmp_path / "data" / "matches" / str(match_id) / "work.mp4"
    make_test_video(video, seconds=2, fps=25)
    res = client.get(f"/api/matches/{match_id}/frames/10.jpg")
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    image = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (240, 320, 3)
    assert client.get(f"/api/matches/{match_id}/frames/100000.jpg").status_code == 404


def test_ball_check_page_is_served_and_linked(tmp_path):
    client, _ = make_client(tmp_path)
    assert "<title>vball · Ball check</title>" in client.get("/ball.html").text
    assert client.get("/ball.js").status_code == 200
    assert 'href="/ball.html"' in client.get("/").text


def test_ball_track_endpoint(tmp_path):
    client, match_id = make_client(tmp_path)  # 1280x720 at 30 fps, 900 frames
    write_tracknet_csv(tmp_path / "data" / "matches" / str(match_id) / "ball.csv", make_track(900, flights=[(1.0, 2.0)]))
    data = client.get(f"/api/matches/{match_id}/ball").json()
    assert data["fps"] == 30.0 and (data["width"], data["height"]) == (1280, 720)
    assert len(data["x"]) == len(data["y"]) == 900
    assert data["x"][0] is None
    assert (data["x"][31], data["y"][31]) == (60.0, 360.0)


def test_ball_track_missing_is_404(tmp_path):
    client, match_id = make_client(tmp_path)
    assert client.get(f"/api/matches/{match_id}/ball").status_code == 404


def test_viewer_has_overlay_and_ball_toggle(tmp_path):
    client, _ = make_client(tmp_path)
    page = client.get("/").text
    assert 'id="overlay"' in page and 'id="show-ball"' in page


COURT_TO_IMAGE = np.array([[110.0, -20.0, 465.0], [0.0, -25.0, 1000.0], [0.0, 0.035, 1.0]])


def court_points(names):
    image = apply_h(COURT_TO_IMAGE, np.array([LANDMARKS[n] for n in names]))
    return [{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(names, image)]


def test_court_calibration_round_trip(tmp_path):
    client, match_id = make_client(tmp_path)  # placeholder video -> one identity segment
    assert client.get(f"/api/matches/{match_id}/court").status_code == 404
    names = ["far_left_corner", "far_right_corner", "center_left", "center_right", "near_attack_right"]
    res = client.put(f"/api/matches/{match_id}/court", json={"ref_frame": 5, "points": court_points(names)})
    assert res.status_code == 200
    body = res.json()
    assert body["mean_error_px"] < 0.01 and len(body["segments"]) == 1
    H = np.array(body["segments"][0]["court_to_image"])
    assert np.allclose(apply_h(H, [[0.0, 18.0]]), apply_h(COURT_TO_IMAGE, [[0.0, 18.0]]), atol=1e-3)
    assert client.get(f"/api/matches/{match_id}/court").json()["ref_frame"] == 5


def test_court_calibration_rejects_bad_input(tmp_path):
    client, match_id = make_client(tmp_path)
    three = court_points(["far_left_corner", "far_right_corner", "center_left"])
    assert client.put(f"/api/matches/{match_id}/court", json={"ref_frame": 0, "points": three}).status_code == 422
    bad = court_points(["far_left_corner", "far_right_corner", "center_left", "center_right"])
    bad[0]["landmark"] = "goal_post"
    assert client.put(f"/api/matches/{match_id}/court", json={"ref_frame": 0, "points": bad}).status_code == 422


def test_viewer_has_calibration_controls(tmp_path):
    client, _ = make_client(tmp_path)
    page = client.get("/").text
    for element_id in ("show-court", "btn-calibrate", "calib-panel", "calib-save", "loupe"):
        assert f'id="{element_id}"' in page


def test_players_window_with_court_positions(tmp_path):
    from vball.players import save_players

    client, match_id = make_client(tmp_path)
    names = ["far_left_corner", "far_right_corner", "center_left", "center_right"]
    client.put(f"/api/matches/{match_id}/court", json={"ref_frame": 0, "points": court_points(names)})
    fx, fy = apply_h(COURT_TO_IMAGE, [[4.5, 15.0]])[0]
    save_players(
        tmp_path / "data" / "matches" / str(match_id) / "players.csv",
        [(10, 4, fx - 20, fy - 120, fx + 20, fy, 0.9), (500, 5, 0, 0, 10, 10, 0.9)],
    )
    rows = client.get(f"/api/matches/{match_id}/players?start=0&end=100").json()
    assert len(rows) == 1 and rows[0]["frame"] == 10 and rows[0]["id"] == 4 and rows[0]["side"] == 1
    # boxes are stored to 0.1 px, so allow 1 cm on court
    assert abs(rows[0]["court"][0] - 4.5) < 0.01 and abs(rows[0]["court"][1] - 15.0) < 0.01


def test_players_window_limits(tmp_path):
    from vball.players import save_players

    client, match_id = make_client(tmp_path)
    assert client.get(f"/api/matches/{match_id}/players?start=0&end=10").status_code == 404  # no players.csv
    save_players(tmp_path / "data" / "matches" / str(match_id) / "players.csv", [(1, 1, 0, 0, 10, 10, 0.9)])
    assert client.get(f"/api/matches/{match_id}/players?start=0&end=5000").status_code == 422
    page = client.get("/").text
    assert 'id="show-players"' in page and 'id="minimap"' in page


def test_court_calibration_accepts_net_points(tmp_path):
    client, match_id = make_client(tmp_path)
    points = court_points(["far_left_corner", "far_right_corner", "center_left", "center_right"])
    points.append({"landmark": "net_left_top", "x": 400.0, "y": 300.0})
    res = client.put(f"/api/matches/{match_id}/court", json={"ref_frame": 0, "points": points})
    assert res.status_code == 200
    body = res.json()
    assert len(body["points"]) == 4 and body["net_points"][0]["landmark"] == "net_left_top"
    assert body["segments"][0]["net_image"] == [[400.0, 300.0], None]
