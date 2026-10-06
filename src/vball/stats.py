"""Match and cross-match statistics from rallies, 3D ball flights and player tracks
(docs/superpowers/specs/2026-10-06-stats-page-design.md)."""

import json
import sqlite3
from pathlib import Path

import numpy as np

from vball import serving, store
from vball.ball3d import load_flights
from vball.config import Paths
from vball.court import COURT_LENGTH, COURT_WIDTH, NET_Y, load_calibration
from vball.labels import load_labels
from vball.players import Players, load_players, with_court

HEAT_X = (-2.0, 11.0)  # metres, 1 m cells: 13 columns
HEAT_Y = (-4.0, 22.0)  # 26 rows


META_DEFAULTS = {"our_side": None, "lineup": [], "serve_fix": {}, "real_score": None}


def load_meta(path: Path) -> dict:
    return {**META_DEFAULTS, **(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})}


def save_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta), encoding="utf-8")


def rally_intervals(paths: Paths, conn: sqlite3.Connection, match_id: int) -> tuple[list[tuple[float, float]], str]:
    """Hand labels when the match has them, else the detected rallies (seconds)."""
    gt = paths.gt_csv(match_id)
    if gt.exists():
        return [(l.start_s, l.end_s) for l in load_labels(gt)], "labels"
    return [(r["start_s"], r["end_s"]) for r in store.get_rallies(conn, match_id)], "detected"


def rally_stats(intervals, duration_s: float, top: int = 5) -> dict:
    lengths = np.array([e - s for s, e in intervals], dtype=float)
    if not len(lengths):
        return {"count": 0}
    order = np.argsort(-lengths, kind="stable")[:top]
    return {
        "count": len(lengths),
        "total_s": float(lengths.sum()),
        "dead_share": float(1 - lengths.sum() / duration_s) if duration_s else None,
        "mean_s": float(lengths.mean()),
        "median_s": float(np.median(lengths)),
        "longest": [{"time_s": float(intervals[i][0]), "length_s": float(lengths[i])} for i in order],
    }


def _lands_near_court(r: dict) -> bool:
    return r["land_x"] is None or (-2 <= r["land_x"] <= COURT_WIDTH + 2 and -2 <= r["land_y"] <= COURT_LENGTH + 2)


def classify_flights(rows: list[dict], fps: float) -> dict:
    """Serves (first net-crossing flight of a rally), sets and attacks among fitted flights, each marked plausible
    or not (rules in the spec)."""
    serves, sets, attacks, served = [], [], [], set()
    for r in rows:
        if not r["fitted"] or r.get("p0_x") is None:
            continue
        item = {"rally": r["rally"], "time_s": r["start_frame"] / fps, "side": "near" if r["p0_y"] < NET_Y else "far",
                "speed_kmh": r["speed_kmh"], "apex_m": r["apex_m"], "net_z_m": r["net_z_m"],
                "land_x": r["land_x"], "land_y": r["land_y"]}
        from_net = abs(r["p0_y"] - NET_Y)
        if r["net_z_m"] is not None and r["rally"] not in served:
            served.add(r["rally"])
            item["plausible"] = bool(
                40 <= r["speed_kmh"] <= 100 and 2.43 < r["net_z_m"] <= 4.5 and _lands_near_court(r)
            )
            serves.append(item)
        elif from_net <= 4 and r["v0_z"] > 0 and r["net_z_m"] is None:
            item["plausible"] = bool(r["p0_z"] <= 3.5 and r["apex_m"] <= 8 and r["speed_kmh"] <= 40)
            sets.append(item)
        elif from_net <= 3 and r["p0_z"] >= 2.3 and r["v0_z"] < 0 and r["net_z_m"] is not None:
            item["plausible"] = bool(r["p0_z"] <= 4.0 and 30 <= r["speed_kmh"] <= 130)
            attacks.append(item)
    return {"serves": serves, "sets": sets, "attacks": attacks}


def player_stats(players: Players, court_xy: np.ndarray, on_court: np.ndarray, intervals_frames) -> dict:
    """Per side, over on-court feet in rally frames: players per frame, mean distance from the net, and a heat map
    (rows y from -4 m, columns x from -2 m, 1 m cells, scaled to max 1). `players` must be sorted by frame."""
    in_rally = np.zeros(len(players), dtype=bool)
    n_frames = 0
    for s, e in intervals_frames:
        a, b = np.searchsorted(players.frame, [s, e])
        in_rally[a:b] = True
        n_frames += e - s
    keep = in_rally & on_court
    out = {"rally_frames": int(n_frames)}
    bins = [int(HEAT_X[1] - HEAT_X[0]), int(HEAT_Y[1] - HEAT_Y[0])]
    for side, mask in (("near", court_xy[:, 1] < NET_Y), ("far", court_xy[:, 1] >= NET_Y)):
        xy = court_xy[keep & mask]
        heat, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=[HEAT_X, HEAT_Y])
        heat = heat.T / heat.max() if heat.max() > 0 else heat.T
        out[side] = {
            "players_per_frame": float(len(xy) / n_frames) if n_frames else 0.0,
            "mean_net_distance_m": float(np.abs(xy[:, 1] - NET_Y).mean()) if len(xy) else None,
            "heat": np.round(heat, 3).tolist(),
        }
    return out


def match_player_stats(paths: Paths, match_id: int, intervals_s, fps: float) -> dict:
    """player_stats for a match's players.csv + court.json (takes seconds on a full set: cache it)."""
    players = load_players(paths.players_csv(match_id))
    order = np.argsort(players.frame, kind="stable")
    players = Players(players.frame[order], players.track_id[order], players.box[order], players.score[order])
    court_xy, _, on_court = with_court(players, load_calibration(paths.court_json(match_id)))
    return player_stats(players, court_xy, on_court, [(round(s * fps), round(e * fps)) for s, e in intervals_s])


def match_serving(paths: Paths, conn: sqlite3.Connection, match_id: int) -> tuple[dict | None, str | None]:
    """(serving section, None) or (None, what to run): every rally with its serving end, winner, serve outcome, score
    and server, the team summary and per-player serves. Fixes and the lineup come from meta.json."""
    path = paths.serving_csv(match_id)
    if not path.exists():
        if not paths.court_json(match_id).exists():
            return None, f"calibrate the court, then run: uv run vball serving {match_id}"
        return None, f"run: uv run vball serving {match_id}"
    match = store.get_match(conn, match_id)
    intervals, source = rally_intervals(paths, conn, match_id)
    rows = serving.load_serving(path)
    if len(rows) != len(intervals) or any(abs(r["start_s"] - s) > 0.05 for r, (s, _) in zip(rows, intervals)):
        return None, f"the rallies changed since serving.csv was made: run uv run vball serving {match_id}"
    meta = load_meta(paths.meta_json(match_id))
    rallies = serving.outcomes(rows, meta["serve_fix"])
    serving.assign_servers(rallies, meta["our_side"], meta["lineup"])
    speeds = {}
    if paths.flights_csv(match_id).exists():
        for x in classify_flights(load_flights(paths.flights_csv(match_id)), match["fps"])["serves"]:
            if x["plausible"]:
                speeds[x["rally"]] = x["speed_kmh"]
    for i, r in enumerate(rallies):
        r["speed_kmh"] = speeds.get(i)
    return {
        "source": source, "rallies": rallies, "summary": serving.team_summary(rallies),
        "players": serving.player_serves(rallies, speeds), "lineup": meta["lineup"], "our_side": meta["our_side"],
        "serve_fix": meta["serve_fix"], "real_score": meta["real_score"],
    }, None


def match_stats(paths: Paths, conn: sqlite3.Connection, match_id: int) -> dict:
    """Everything cheap for one match; `missing` says which sections are absent and what to run."""
    match = store.get_match(conn, match_id)
    intervals, source = rally_intervals(paths, conn, match_id)
    out = {
        "id": match_id, "name": match["name"], "our_side": load_meta(paths.meta_json(match_id)).get("our_side"),
        "rallies": {**rally_stats(intervals, match["n_frames"] / match["fps"]), "source": source},
        "missing": {},
    }
    if paths.flights_csv(match_id).exists():
        out["flights"] = classify_flights(load_flights(paths.flights_csv(match_id)), match["fps"])
    elif paths.court_json(match_id).exists() and load_calibration(paths.court_json(match_id)).copied_from:
        out["missing"]["flights"] = "the court is a draft: check and save it in the viewer (Calibrate court) first"
    elif not paths.court_json(match_id).exists():
        out["missing"]["flights"] = f"calibrate the court (with the net clicks), then run: uv run vball ball3d {match_id}"
    else:
        out["missing"]["flights"] = f"run: uv run vball ball3d {match_id}"
    section, missing = match_serving(paths, conn, match_id)
    if section:
        out["serving"] = {k: v for k, v in section.items() if k not in ("rallies", "serve_fix")}
        last = section["rallies"][-1] if section["rallies"] else None
        out["serving"]["score"] = {"near": last["score_near"], "far": last["score_far"]} if last else None
    else:
        out["missing"]["serving"] = missing
    return out


def leaderboards(all_stats: list[dict], n: int = 10) -> dict:
    """Top plausible serves (speed), sets (apex), attacks (speed) and longest rallies across matches."""
    def top(kind: str, key: str) -> list[dict]:
        items = [
            {**x, "match_id": s["id"], "match_name": s["name"], "our_side": s["our_side"]}
            for s in all_stats for x in s.get("flights", {}).get(kind, []) if x["plausible"]
        ]
        return sorted(items, key=lambda x: -x[key])[:n]

    rallies = [
        {**r, "match_id": s["id"], "match_name": s["name"]} for s in all_stats for r in s["rallies"].get("longest", [])
    ]
    return {
        "serves": top("serves", "speed_kmh"),
        "sets": top("sets", "apex_m"),
        "attacks": top("attacks", "speed_kmh"),
        "rallies": sorted(rallies, key=lambda r: -r["length_s"])[:n],
    }
