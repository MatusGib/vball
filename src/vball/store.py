import sqlite3
from pathlib import Path

from vball.rallies import Rally
from vball.video import VideoInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    fps REAL,
    n_frames INTEGER,
    width INTEGER,
    height INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS rallies (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    UNIQUE (match_id, idx)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def add_match(conn: sqlite3.Connection, name: str, source_path: str) -> int:
    with conn:
        cur = conn.execute("INSERT INTO matches (name, source_path) VALUES (?, ?)", (name, source_path))
    return cur.lastrowid


def set_video_info(conn: sqlite3.Connection, match_id: int, info: VideoInfo) -> None:
    with conn:
        conn.execute(
            "UPDATE matches SET fps = ?, n_frames = ?, width = ?, height = ? WHERE id = ?",
            (info.fps, info.n_frames, info.width, info.height, match_id),
        )


def get_match(conn: sqlite3.Connection, match_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)).fetchone()
    return dict(row) if row else None


def list_matches(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT m.id, m.name, m.fps, m.n_frames, m.width, m.height, m.created_at,
               COUNT(r.id) AS n_rallies,
               COALESCE(SUM(r.end_frame - r.start_frame), 0) AS rally_frames
        FROM matches m LEFT JOIN rallies r ON r.match_id = m.id
        GROUP BY m.id ORDER BY m.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def replace_rallies(conn: sqlite3.Connection, match_id: int, rallies: list[Rally]) -> None:
    with conn:
        conn.execute("DELETE FROM rallies WHERE match_id = ?", (match_id,))
        conn.executemany(
            "INSERT INTO rallies (match_id, idx, start_frame, end_frame) VALUES (?, ?, ?, ?)",
            [(match_id, i, r.start_frame, r.end_frame) for i, r in enumerate(rallies)],
        )


def get_rallies(conn: sqlite3.Connection, match_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT r.id, r.idx, r.start_frame, r.end_frame,
               r.start_frame / m.fps AS start_s, r.end_frame / m.fps AS end_s
        FROM rallies r JOIN matches m ON m.id = r.match_id
        WHERE r.match_id = ? ORDER BY r.idx
        """,
        (match_id,),
    ).fetchall()
    return [dict(r) for r in rows]
