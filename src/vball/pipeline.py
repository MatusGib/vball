import sqlite3
from collections.abc import Callable
from pathlib import Path

from vball import store, video
from vball.ball.track import load_tracknet_csv
from vball.ball.tracknet import run_tracknet
from vball.config import Paths
from vball.rallies import Rally, RallyParams, detect_rallies

BallRunner = Callable[[Path, Path], object]  # (work_video, out_csv)


def process_match(
    src: Path,
    paths: Paths,
    name: str | None = None,
    encoder: str = "libx264",
    ball_runner: BallRunner = run_tracknet,
) -> int:
    """Ingest src, track the ball, detect rallies. Returns the new match id."""
    conn = store.connect(paths.db_path)
    try:
        match_id = store.add_match(conn, name or src.stem, str(src.resolve()))
        info = video.ingest(src, paths.work_video(match_id), encoder=encoder)
        store.set_video_info(conn, match_id, info)
        ball_runner(paths.work_video(match_id), paths.ball_csv(match_id))
        redetect(conn, paths, match_id)
        return match_id
    finally:
        conn.close()


def redetect(
    conn: sqlite3.Connection, paths: Paths, match_id: int, params: RallyParams = RallyParams()
) -> list[Rally]:
    """Re-run rally detection from the cached ball track (seconds, no GPU)."""
    match = store.get_match(conn, match_id)
    track = load_tracknet_csv(paths.ball_csv(match_id), match["n_frames"])
    rallies = detect_rallies(track, match["fps"], match["width"], match["height"], params)
    store.replace_rallies(conn, match_id, rallies)
    return rallies
