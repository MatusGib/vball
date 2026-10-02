import sqlite3
from collections.abc import Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from vball import store
from vball.config import Paths
from vball.evaluate import load_intervals_csv, save_intervals_csv

STATIC_DIR = Path(__file__).parent / "static"


def create_app(paths: Paths) -> FastAPI:
    app = FastAPI(title="vball")

    def db() -> Iterator[sqlite3.Connection]:
        conn = store.connect(paths.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def require_match(conn: sqlite3.Connection, match_id: int) -> None:
        if store.get_match(conn, match_id) is None:
            raise HTTPException(status_code=404, detail="match not found")

    @app.get("/api/matches")
    def list_matches(conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        return store.list_matches(conn)

    @app.get("/api/matches/{match_id}/rallies")
    def list_rallies(match_id: int, conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        require_match(conn, match_id)
        return store.get_rallies(conn, match_id)

    @app.get("/api/matches/{match_id}/labels")
    def get_labels(match_id: int, conn: sqlite3.Connection = Depends(db)) -> list[tuple[float, float]]:
        require_match(conn, match_id)
        path = paths.gt_csv(match_id)
        return load_intervals_csv(path) if path.exists() else []

    @app.put("/api/matches/{match_id}/labels")
    def put_labels(
        match_id: int, labels: list[tuple[float, float]], conn: sqlite3.Connection = Depends(db)
    ) -> list[tuple[float, float]]:
        require_match(conn, match_id)
        if any(start < 0 or end <= start for start, end in labels):
            raise HTTPException(status_code=422, detail="each label needs 0 <= start < end")
        save_intervals_csv(paths.gt_csv(match_id), labels)
        return load_intervals_csv(paths.gt_csv(match_id))

    paths.matches_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=paths.matches_dir), name="media")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
