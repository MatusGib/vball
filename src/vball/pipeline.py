import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from vball import store, video
from vball.ball.track import load_tracknet_csv
from vball.ball.tracknet import run_tracknet
from vball.config import Paths
from vball.court import load_calibration
from vball.players import load_players, with_court
from vball.rallies import Rally, RallyParams, detect_rallies
from vball.serve import ServeParams, serve_filter

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


def serve_status(paths: Paths, match_id: int) -> str:
    if not paths.players_csv(match_id).exists():
        return "serve check skipped: no player tracks"
    if not paths.court_json(match_id).exists():
        return "serve check skipped: no court calibration"
    return "serve check on"


def serve_keep(paths: Paths, match_id: int, track, fps: float, params: ServeParams = ServeParams()):
    """keep(start, end) for detect_rallies, or None when the match lacks player tracks or a court."""
    if serve_status(paths, match_id) != "serve check on":
        return None
    players = load_players(paths.players_csv(match_id))
    court_xy, _, _ = with_court(players, load_calibration(paths.court_json(match_id)))
    return serve_filter(track, players, court_xy, fps, params)


def redetect(
    conn: sqlite3.Connection,
    paths: Paths,
    match_id: int,
    params: RallyParams = RallyParams(),
    serve_params: ServeParams = ServeParams(),
) -> list[Rally]:
    """Re-run rally detection from the cached ball track (seconds, no GPU); serve check when possible."""
    match = store.get_match(conn, match_id)
    track = load_tracknet_csv(paths.ball_csv(match_id), match["n_frames"])
    keep = serve_keep(paths, match_id, track, match["fps"], serve_params)
    if keep is not None:
        params = replace(params, min_rally_s=serve_params.min_serve_rally_s)
    rallies = detect_rallies(track, match["fps"], match["width"], match["height"], params, keep=keep)
    store.replace_rallies(conn, match_id, rallies)
    return rallies
