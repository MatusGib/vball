import sqlite3
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vball import stats, store
from vball.ball.testset import BallTestItem, load_ball_test, sample_test_frames, save_ball_test
from vball.camera import camera_segments
from vball.court import LANDMARKS, NET_LANDMARKS, Calibration, calibration_json, load_calibration, save_calibration
from vball.ball.track import load_tracknet_csv
from vball.ball3d import flight_track, load_flights
from vball.camera3d import fit_camera
from vball.config import Paths
from vball.labels import Label, load_labels, save_labels
from vball.players import load_players, with_court

STATIC_DIR = Path(__file__).parent / "static"


class LabelBody(BaseModel):
    start_s: float
    end_s: float
    approved: bool = True


class BallTestBody(BaseModel):
    status: Literal["todo", "ball", "none", "skip"]
    x: float = 0.0
    y: float = 0.0


class CourtPoint(BaseModel):
    landmark: str
    x: float
    y: float


class ServeFix(BaseModel):
    end: Literal["near", "far"] | None = None
    outcome: Literal["ace", "error", "in"] | None = None


class MetaBody(BaseModel):
    """Only the fields sent are changed."""

    our_side: Literal["near", "far"] | None = None
    lineup: list[str] | None = None
    serve_fix: dict[str, ServeFix] | None = None


class CourtBody(BaseModel):
    ref_frame: int
    points: list[CourtPoint]


def create_app(paths: Paths) -> FastAPI:
    app = FastAPI(title="vball")

    def db() -> Iterator[sqlite3.Connection]:
        conn = store.connect(paths.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def require_match(conn: sqlite3.Connection, match_id: int) -> dict:
        match = store.get_match(conn, match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="match not found")
        return match

    def read_labels(match_id: int) -> list[dict]:
        path = paths.gt_csv(match_id)
        return [asdict(label) for label in load_labels(path)] if path.exists() else []

    @app.get("/api/matches")
    def list_matches(conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        return store.list_matches(conn)

    @app.get("/api/matches/{match_id}/rallies")
    def list_rallies(match_id: int, conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        require_match(conn, match_id)
        return store.get_rallies(conn, match_id)

    @app.get("/api/matches/{match_id}/labels")
    def get_labels(match_id: int, conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        require_match(conn, match_id)
        return read_labels(match_id)

    @app.put("/api/matches/{match_id}/labels")
    def put_labels(match_id: int, labels: list[LabelBody], conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        require_match(conn, match_id)
        if any(l.start_s < 0 or l.end_s <= l.start_s for l in labels):
            raise HTTPException(status_code=422, detail="each label needs 0 <= start < end")
        save_labels(paths.gt_csv(match_id), [Label(l.start_s, l.end_s, l.approved) for l in labels])
        return read_labels(match_id)

    @app.get("/api/matches/{match_id}/ball-test")
    def get_ball_test(match_id: int, n: int = 100, conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        match = require_match(conn, match_id)
        path = paths.ball_test_csv(match_id)
        if not path.exists():
            gt = paths.gt_csv(match_id)
            if gt.exists():
                intervals = [(label.start_s, label.end_s) for label in load_labels(gt)]
            else:
                intervals = [(r["start_s"], r["end_s"]) for r in store.get_rallies(conn, match_id)]
            frames = sample_test_frames(intervals, match["fps"], n, seed=match_id)
            save_ball_test(path, [BallTestItem(f) for f in frames])
        return [asdict(item) for item in load_ball_test(path)]

    @app.put("/api/matches/{match_id}/ball-test/{frame}")
    def put_ball_test(
        match_id: int, frame: int, body: BallTestBody, conn: sqlite3.Connection = Depends(db)
    ) -> dict:
        require_match(conn, match_id)
        path = paths.ball_test_csv(match_id)
        items = load_ball_test(path) if path.exists() else []
        index = next((i for i, item in enumerate(items) if item.frame == frame), None)
        if index is None:
            raise HTTPException(status_code=404, detail="frame is not in the test set")
        items[index] = BallTestItem(frame, body.status, body.x, body.y)
        save_ball_test(path, items)
        return asdict(items[index])

    @app.get("/api/matches/{match_id}/ball")
    def get_ball(match_id: int, conn: sqlite3.Connection = Depends(db)) -> dict:
        match = require_match(conn, match_id)
        path = paths.ball_csv(match_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="no ball track")
        track = load_tracknet_csv(path, match["n_frames"])

        def column(values):
            return [round(float(v), 1) if seen else None for v, seen in zip(values, track.visible)]

        return {
            "fps": match["fps"],
            "width": match["width"],
            "height": match["height"],
            "x": column(track.x),
            "y": column(track.y),
        }

    @app.get("/api/matches/{match_id}/court")
    def get_court(match_id: int, conn: sqlite3.Connection = Depends(db)) -> dict:
        require_match(conn, match_id)
        path = paths.court_json(match_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="court not calibrated")
        return calibration_json(load_calibration(path))

    @app.put("/api/matches/{match_id}/court")
    def put_court(match_id: int, body: CourtBody, conn: sqlite3.Connection = Depends(db)) -> dict:
        match = require_match(conn, match_id)
        unknown = [p.landmark for p in body.points if p.landmark not in LANDMARKS and p.landmark not in NET_LANDMARKS]
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown landmarks: {unknown}")
        # reads the whole 512x288 copy to find camera moves: ~30-60 s on a real set
        segments = camera_segments(
            paths.track_video(match_id), body.ref_frame, match["n_frames"], scale=match["width"] / 512
        )
        try:
            cal = Calibration.create(body.ref_frame, [p.model_dump() for p in body.points], segments)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        save_calibration(paths.court_json(match_id), cal)
        return calibration_json(cal)

    player_cache: dict[int, tuple] = {}  # match id -> (players mtime, court mtime, (players, court mapping))

    def players_for(match_id: int):
        csv_path, court_path = paths.players_csv(match_id), paths.court_json(match_id)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="no player tracks")
        key = (csv_path.stat().st_mtime, court_path.stat().st_mtime if court_path.exists() else 0.0)
        cached = player_cache.get(match_id)
        if cached is None or cached[:2] != key:
            players = load_players(csv_path)
            court = with_court(players, load_calibration(court_path)) if court_path.exists() else None
            player_cache[match_id] = (*key, (players, court))
        return player_cache[match_id][2]

    @app.get("/api/matches/{match_id}/players")
    def get_players(match_id: int, start: int, end: int, conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        require_match(conn, match_id)
        if not 0 <= start < end or end - start > 1800:
            raise HTTPException(status_code=422, detail="window must be 1-1800 frames")
        players, court = players_for(match_id)
        rows = []
        for i in np.flatnonzero((players.frame >= start) & (players.frame < end)):
            row = {
                "frame": int(players.frame[i]),
                "id": int(players.track_id[i]),
                "box": [round(float(v), 1) for v in players.box[i]],
                "court": None,
                "side": None,
            }
            if court is not None:
                court_xy, side, on_court = court
                if on_court[i]:
                    row["court"] = [round(float(v), 3) for v in court_xy[i]]
                    row["side"] = int(side[i])
            rows.append(row)
        return rows

    flight_cache: dict[int, tuple] = {}  # match id -> ((flights mtime, court mtime), response)

    @app.get("/api/matches/{match_id}/flights")
    def get_flights(match_id: int, conn: sqlite3.Connection = Depends(db)) -> dict:
        match = require_match(conn, match_id)
        csv_path, court_path = paths.flights_csv(match_id), paths.court_json(match_id)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="no 3D flights")
        if not court_path.exists():
            raise HTTPException(status_code=404, detail="court not calibrated")
        key = (csv_path.stat().st_mtime, court_path.stat().st_mtime)
        cached = flight_cache.get(match_id)
        if cached is None or cached[0] != key:
            rows = [r for r in load_flights(csv_path) if r["fitted"] and r.get("p0_x") is not None]
            if not rows:
                raise HTTPException(status_code=404, detail="flights.csv has no 3D states; re-run vball ball3d")
            cam, _ = fit_camera(load_calibration(court_path), match["width"], match["height"])
            body = {"fps": match["fps"], "flights": [flight_track(cam, r, match["fps"]) for r in rows]}
            flight_cache[match_id] = (key, body)
        return flight_cache[match_id][1]

    @app.get("/api/stats")
    def get_all_stats(conn: sqlite3.Connection = Depends(db)) -> dict:
        all_stats = [stats.match_stats(paths, conn, m["id"]) for m in store.list_matches(conn)]
        return {"matches": all_stats, "leaderboards": stats.leaderboards(all_stats)}

    player_stats_cache: dict[int, tuple] = {}  # match id -> (key, player stats)

    @app.get("/api/matches/{match_id}/stats")
    def get_match_stats(match_id: int, conn: sqlite3.Connection = Depends(db)) -> dict:
        match = require_match(conn, match_id)
        body = stats.match_stats(paths, conn, match_id)
        csv_path, court_path = paths.players_csv(match_id), paths.court_json(match_id)
        if not csv_path.exists():
            body["missing"]["players"] = f"no player tracks: run uv run vball players {match_id}"
        elif not court_path.exists():
            body["missing"]["players"] = "calibrate the court first (viewer: Calibrate court)"
        else:
            intervals, _ = stats.rally_intervals(paths, conn, match_id)
            key = (csv_path.stat().st_mtime, court_path.stat().st_mtime, tuple(intervals))
            cached = player_stats_cache.get(match_id)
            if cached is None or cached[0] != key:
                player_stats_cache[match_id] = (key, stats.match_player_stats(paths, match_id, intervals, match["fps"]))
            body["players"] = player_stats_cache[match_id][1]
        return body

    @app.put("/api/matches/{match_id}/meta")
    def put_meta(match_id: int, body: MetaBody, conn: sqlite3.Connection = Depends(db)) -> dict:
        require_match(conn, match_id)
        meta = stats.load_meta(paths.meta_json(match_id))
        changes = body.model_dump(exclude_unset=True)
        if "lineup" in changes:
            changes["lineup"] = [n.strip() for n in changes["lineup"] or [] if n.strip()]
        if "serve_fix" in changes:
            changes["serve_fix"] = {
                k: {f: v for f, v in fix.items() if v is not None}
                for k, fix in (changes["serve_fix"] or {}).items()
                if any(v is not None for v in fix.values())
            }
        stats.save_meta(paths.meta_json(match_id), {**meta, **changes})
        return stats.load_meta(paths.meta_json(match_id))

    @app.get("/api/matches/{match_id}/serving")
    def get_serving(match_id: int, conn: sqlite3.Connection = Depends(db)) -> dict:
        require_match(conn, match_id)
        section, missing = stats.match_serving(paths, conn, match_id)
        if section is None:
            raise HTTPException(status_code=404, detail=missing)
        return section

    @app.get("/api/matches/{match_id}/frames/{frame}.jpg")
    def get_frame(match_id: int, frame: int, conn: sqlite3.Connection = Depends(db)) -> Response:
        match = require_match(conn, match_id)
        if not 0 <= frame < match["n_frames"]:
            raise HTTPException(status_code=404, detail="frame out of range")
        cap = cv2.VideoCapture(str(paths.work_video(match_id)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(status_code=404, detail="frame could not be decoded")
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")

    paths.matches_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=paths.matches_dir), name="media")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
