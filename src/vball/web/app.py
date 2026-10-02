import sqlite3
from collections.abc import Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from vball import store
from vball.config import Paths

STATIC_DIR = Path(__file__).parent / "static"


def create_app(paths: Paths) -> FastAPI:
    app = FastAPI(title="vball")

    def db() -> Iterator[sqlite3.Connection]:
        conn = store.connect(paths.db_path)
        try:
            yield conn
        finally:
            conn.close()

    @app.get("/api/matches")
    def list_matches(conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        return store.list_matches(conn)

    @app.get("/api/matches/{match_id}/rallies")
    def list_rallies(match_id: int, conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        if store.get_match(conn, match_id) is None:
            raise HTTPException(status_code=404, detail="match not found")
        return store.get_rallies(conn, match_id)

    paths.matches_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=paths.matches_dir), name="media")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
