"""Which end served each rally, who won it, how the serve ended, and who served
(docs/superpowers/specs/2026-10-06-serving-score-design.md)."""

import csv
from pathlib import Path

import numpy as np

from vball.ball.track import BallTrack
from vball.court import COURT_LENGTH, COURT_WIDTH, NET_Y
from vball.players import Players
from vball.serve import ServeParams, _inside, _leaves, reach_box

BEFORE_S, AFTER_S = 1.0, 1.5  # look for the server from rally start - this to rally start + this
BEHIND_M = 0.3  # feet this far behind a baseline
SIDE_M = 1.5  # feet within the sidelines +- this
ON_SERVE_S, PLAYED_S = 2.5, 4.5  # rally end this soon after the serve: ended on it; this late: the serve was played
ACTION_FROM_S, ACTION_TO_S = 0.4, 3.5  # action detector window after the serve
RECEIVE_FROM_S = 0.5  # a receive this soon after the serve is the server's own swing
RECEIVE_CONF = 0.4
FIX_MATCH_S = 0.5
OUTCOMES = ("ace", "error", "in")
COLUMNS = ["start_s", "end_s", "end", "how", "serve_s", "defense_near", "defense_far"]
ACTION_COLUMNS = ["frame", "action", "conf", "x1", "y1", "x2", "y2", "court_x", "court_y"]


def serving_end(
    start: int, track: BallTrack, players: Players, court_xy: np.ndarray, fps: float, p: ServeParams = ServeParams()
) -> tuple[str | None, int | None, str]:
    """(end, serve frame, how) for a rally starting at frame `start`; players sorted by frame, court_xy their feet.
    The ball leaving a player behind a baseline decides; otherwise the end with more players behind its baseline."""
    lo, hi = max(0, start - round(BEFORE_S * fps)), start + round(AFTER_S * fps)
    a, b = np.searchsorted(players.frame, [lo, hi])
    x, y = court_xy[a:b, 0], court_xy[a:b, 1]
    across = (x >= -SIDE_M) & (x <= COURT_WIDTH + SIDE_M)
    near, far = across & (y < -BEHIND_M), across & (y > COURT_LENGTH + BEHIND_M)
    for i in a + np.flatnonzero(near | far):
        c = int(players.frame[i])
        if c < len(track) and track.visible[c] and _inside((track.x[c], track.y[c]), reach_box(players.box[i], p)):
            if _leaves(c, int(players.track_id[i]), players.box[i], track, players, p, fps):
                return ("near" if court_xy[i, 1] < NET_Y else "far"), c, "contact"
    n_near, n_far = int(near.sum()), int(far.sum())
    if n_near == n_far:
        return None, None, "none"
    return ("near" if n_near > n_far else "far"), None, "count"


def match_serving(intervals_s, track: BallTrack, players: Players, court_xy: np.ndarray, fps: float) -> list[dict]:
    """One row per rally (seconds); players sorted by frame."""
    rows = []
    for s, e in intervals_s:
        end, frame, how = serving_end(round(s * fps), track, players, court_xy, fps)
        rows.append({"start_s": s, "end_s": e, "end": end, "how": how,
                     "serve_s": frame / fps if frame is not None else s, "defense_near": None, "defense_far": None})
    return rows


def action_windows(rows: list[dict], fps: float) -> list[tuple[int, int]]:
    return [(round((r["serve_s"] + ACTION_FROM_S) * fps), round((r["serve_s"] + ACTION_TO_S) * fps)) for r in rows]


def detect_actions(video: Path, windows: list[tuple[int, int]], cal, model_path: Path, every: int) -> list[dict]:
    """VolleyVision action boxes (block, defense, serve, set, spike) on every `every`-th frame of each window, with
    the court position of each box's feet."""
    import cv2
    from ultralytics import YOLO  # heavy imports, only when used

    from vball.players import feet_to_court

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video))
    rows = []
    for a, b in windows:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        for f in range(a, b + 1):
            if (f - a) % every:
                if not cap.grab():
                    break
                continue
            ok, image = cap.read()
            if not ok:
                break
            res = model.predict(image, imgsz=1280, conf=0.15, half=True, verbose=False)[0]
            for k, conf, box in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
                rows.append({"frame": f, "action": model.names[int(k)], "conf": conf, "x1": box[0], "y1": box[1],
                             "x2": box[2], "y2": box[3]})
    cap.release()
    if rows:
        xy = feet_to_court(np.array([r["frame"] for r in rows]), np.array([[r["x1"], r["y1"], r["x2"], r["y2"]]
                                                                             for r in rows]), cal)
        for r, (x, y) in zip(rows, xy):
            r["court_x"], r["court_y"] = float(x), float(y)
    return rows


def mark_defense(rows: list[dict], actions: list[dict], fps: float) -> None:
    """Did the detector see a confident 'defense' on each half in [serve + 0.5 s, serve + 3.5 s]?"""
    frames = np.array([a["frame"] for a in actions], dtype=int)
    for r in rows:
        lo, hi = (r["serve_s"] + RECEIVE_FROM_S) * fps, (r["serve_s"] + ACTION_TO_S) * fps
        seen = [actions[i] for i in np.flatnonzero((frames >= lo) & (frames <= hi))
                if actions[i]["action"] == "defense" and actions[i]["conf"] >= RECEIVE_CONF]
        r["defense_near"] = any(a["court_y"] < NET_Y for a in seen)
        r["defense_far"] = any(a["court_y"] >= NET_Y for a in seen)


def _write(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, columns, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else round(r[k], 3) if isinstance(r[k], float) else r[k])
                             for k in columns})


def save_serving(path: Path, rows: list[dict]) -> None:
    _write(path, COLUMNS, rows)


def load_serving(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "start_s": float(r["start_s"]), "end_s": float(r["end_s"]), "end": r["end"] or None, "how": r["how"],
                "serve_s": float(r["serve_s"]),
                "defense_near": None if r["defense_near"] == "" else r["defense_near"] == "True",
                "defense_far": None if r["defense_far"] == "" else r["defense_far"] == "True",
            })
    return rows


def save_actions(path: Path, rows: list[dict]) -> None:
    _write(path, ACTION_COLUMNS, rows)


def load_actions(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [{k: (int(v) if k == "frame" else v if k == "action" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


def fix_for(start_s: float, fixes: dict) -> tuple[str, dict]:
    """(key, fix) for the fix whose key (rally start, seconds) is nearest start_s within FIX_MATCH_S; a new key and
    an empty fix when there is none."""
    best, gap = (fix_key(start_s), {}), FIX_MATCH_S
    for key, fix in fixes.items():
        d = abs(float(key) - start_s)
        if d <= gap:
            best, gap = (key, fix), d
    return best


def fix_key(start_s: float) -> str:
    return f"{start_s:.1f}"


def legal_end(a: int, b: int, target: int) -> bool:
    return max(a, b) >= target and abs(a - b) >= 2


def other(end: str) -> str:
    return "far" if end == "near" else "near"


def auto_outcome(r: dict) -> str:
    """The owner's rule: a serve the receivers played was in; a rally that ends right after the serve is an ace if
    the server's end won it and a serve error if not."""
    receive = r.get(f"defense_{other(r['end'])}") if r["end"] else None
    gap = r["end_s"] - r["serve_s"]
    if r["end"] is None:
        return "check"
    if gap <= ON_SERVE_S:
        if r["winner"] is None:
            return "check"
        # touched or not: the receivers could not keep it in play / the serve was out or in the net (a receiver
        # catching or bumping an out ball looks like a receive to the action detector, so it is not asked here)
        return "ace" if r["winner"] == r["end"] else "error"
    return "in" if gap >= PLAYED_S or receive else "check"


def last_winner(rallies: list[dict]) -> str | None:
    """Nobody serves after the last rally: a fixed ace / error decides, else the end for which the final score is a
    legal set end (25, else 15, and two ahead), when exactly one end qualifies."""
    last = rallies[-1]
    if last["end"] and last["outcome_fixed"] in ("ace", "error"):
        return last["end"] if last["outcome_fixed"] == "ace" else other(last["end"])
    n = sum(r["winner"] == "near" for r in rallies[:-1])
    f = sum(r["winner"] == "far" for r in rallies[:-1])
    for target in (25, 15):
        ends = [side for side, (a, b) in (("near", (n + 1, f)), ("far", (n, f + 1))) if legal_end(a, b, target)]
        if len(ends) == 1:
            return ends[0]
    return None


def outcomes(rows: list[dict], fixes: dict | None = None) -> list[dict]:
    """Rows with the fixes applied, the winner of each rally (the next rally's serving end), the serve outcome (ace,
    error, in or check) and the running score."""
    fixes = fixes or {}
    out = []
    for r in rows:
        key, f = fix_for(r["start_s"], fixes)
        out.append({**r, "fix_key": key, "end_auto": r["end"], "end": f.get("end") or r["end"], "end_fixed": "end" in f,
                    "outcome_fixed": f.get("outcome"), "gap_s": r["end_s"] - r["serve_s"]})
    for i, r in enumerate(out):
        r["winner"] = out[i + 1]["end"] if i + 1 < len(out) else None
    if out:
        out[-1]["winner"] = last_winner(out)
    near = far = 0
    for r in out:
        r["outcome_auto"] = auto_outcome(r)
        r["outcome"] = r["outcome_fixed"] or r["outcome_auto"]
        near += r["winner"] == "near"
        far += r["winner"] == "far"
        r["score_near"], r["score_far"] = near, far
    return out


def team_summary(rallies: list[dict]) -> dict:
    """Per end: points, serves, points won on serve, side-out %, aces, serve errors, serving % ((serves - errors) /
    serves over decided serves); plus how many rallies need checking."""
    out = {"check": sum(r["outcome"] == "check" for r in rallies),
           "unknown_end": sum(r["end"] is None for r in rallies),
           "unknown_winner": sum(r["winner"] is None for r in rallies)}
    for side in ("near", "far"):
        serves = [r for r in rallies if r["end"] == side]
        received = [r for r in rallies if r["end"] == other(side) and r["winner"]]
        decided = [r for r in serves if r["outcome"] in OUTCOMES]
        errors = sum(r["outcome"] == "error" for r in serves)
        sideouts = sum(r["winner"] == side for r in received)
        out[side] = {
            "points": sum(r["winner"] == side for r in rallies),
            "serves": len(serves),
            "won_on_serve": sum(r["winner"] == side for r in serves),
            "received": len(received),
            "sideout_pct": sideouts / len(received) if received else None,
            "aces": sum(r["outcome"] == "ace" for r in serves),
            "errors": errors,
            "serving_pct": (len(decided) - errors) / len(decided) if decided else None,
        }
    return out


def assign_servers(rallies: list[dict], our_end: str | None, lineup: list[str]) -> None:
    """Our server for each of our serves (r['server']): each time we win the serve back, the next player in the
    lineup serves. Rallies with an unknown serving end don't break a serving turn."""
    k, ours_before = -1, False
    for r in rallies:
        r["server"] = None
        if not our_end or not lineup or r["end"] is None:
            continue
        ours = r["end"] == our_end
        if ours:
            if not ours_before:
                k += 1
            r["server"] = lineup[k % len(lineup)]
        ours_before = ours


def player_serves(rallies: list[dict], speeds: dict[int, float]) -> list[dict]:
    """Per named server: serves, aces, errors, serving %, top and median plausible 3D speed (speeds: rally index ->
    km/h)."""
    by_name: dict[str, list[int]] = {}
    for i, r in enumerate(rallies):
        if r.get("server"):
            by_name.setdefault(r["server"], []).append(i)
    out = []
    for name, idx in by_name.items():
        rs = [rallies[i] for i in idx]
        decided = [r for r in rs if r["outcome"] in OUTCOMES]
        errors = sum(r["outcome"] == "error" for r in rs)
        sp = [speeds[i] for i in idx if i in speeds]
        out.append({
            "name": name, "serves": len(rs), "aces": sum(r["outcome"] == "ace" for r in rs), "errors": errors,
            "check": sum(r["outcome"] == "check" for r in rs),
            "serving_pct": (len(decided) - errors) / len(decided) if decided else None,
            "top_kmh": max(sp) if sp else None, "median_kmh": float(np.median(sp)) if sp else None,
            "measured": len(sp),
        })
    return out
