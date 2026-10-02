import sqlite3
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import cv2
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vball import store
from vball.ball.testset import BallTestItem, load_ball_test, sample_test_frames, save_ball_test
from vball.config import Paths
from vball.labels import Label, load_labels, save_labels

STATIC_DIR = Path(__file__).parent / "static"


class LabelBody(BaseModel):
    start_s: float
    end_s: float
    approved: bool = True


class BallTestBody(BaseModel):
    status: Literal["todo", "ball", "none", "skip"]
    x: float = 0.0
    y: float = 0.0


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
