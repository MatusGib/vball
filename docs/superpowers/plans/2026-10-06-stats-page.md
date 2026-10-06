# Stats Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Stats page (match + cross-match leaderboards, side-level player numbers) whose rows open the viewer at the moment.

**Architecture:** `stats.py` holds pure statistics over existing files (labels/rallies, `flights.csv`, `players.csv` + `court.json`, `meta.json`). FastAPI exposes `/api/stats`, `/api/matches/{id}/stats`, `PUT /api/matches/{id}/meta`. A static `stats.html` + `stats.js` renders tiles, leaderboards and heat maps; the viewer accepts `?match&t&show` deep links.

**Tech Stack:** numpy, FastAPI, vanilla JS + canvas, pytest.

Spec: `docs/superpowers/specs/2026-10-06-stats-page-design.md`. Tests: `uv run --no-sync pytest -q`.

---

### Task 1: Statistics module

**Files:** create `src/vball/stats.py`, `tests/test_stats.py`; modify `src/vball/config.py` (`meta_json`).

- [ ] **Step 1: Failing tests** — `tests/test_stats.py`:

```python
import numpy as np

from vball.players import Players
from vball.stats import classify_flights, leaderboards, load_meta, player_stats, rally_stats, save_meta


def flight(rally, frame, p0, v0, speed, apex, net_z, land=(None, None), fitted=True):
    keys = ["p0_x", "p0_y", "p0_z", "v0_x", "v0_y", "v0_z"]
    return {"rally": rally, "start_frame": frame, "fitted": fitted, **dict(zip(keys, [*p0, *v0])),
            "speed_kmh": speed, "apex_m": apex, "net_z_m": net_z, "land_x": land[0], "land_y": land[1]}


def test_rally_stats():
    s = rally_stats([(0.0, 10.0), (20.0, 25.0), (30.0, 60.0)], 100.0)
    assert s["count"] == 3 and s["total_s"] == 45.0 and abs(s["dead_share"] - 0.55) < 1e-9
    assert s["longest"][0] == {"time_s": 30.0, "length_s": 30.0} and s["median_s"] == 10.0
    assert rally_stats([], 100.0) == {"count": 0}


def test_flights_are_classified_with_plausibility():
    rows = [
        flight(0, 30, (4, -1, 2.8), (0, 19, 2), 70.0, 3.2, 2.9, (4, 14)),  # serve, near
        flight(0, 60, (5, 7, 2.2), (0, 0.5, 6), 22.0, 4.5, None),  # set
        flight(0, 90, (4, 8, 3.0), (0, 20, -4), 75.0, 3.0, 2.8, (4, 12)),  # attack
        flight(0, 120, (4, 3, 1.0), (0, 2, 5), 20.0, 2.0, None),  # a pass far from the net: none of them
        flight(1, 300, (4, 20, 3), (0, -20, 1), 120.0, 3.4, 1.5),  # serve, far, implausible
        flight(1, 330, (4, 8, 3), (0, 20, -4), 0.0, 0.0, 2.9, fitted=False),  # not fitted: ignored
    ]
    c = classify_flights(rows, fps=30.0)
    assert [(s["rally"], s["side"], s["plausible"]) for s in c["serves"]] == [(0, "near", True), (1, "far", False)]
    assert c["serves"][0]["time_s"] == 1.0
    assert [s["apex_m"] for s in c["sets"]] == [4.5] and c["sets"][0]["plausible"]
    assert [a["speed_kmh"] for a in c["attacks"]] == [75.0] and c["attacks"][0]["plausible"]


def test_player_stats_per_side_inside_rallies_only():
    rows = [(f, 1, 0, 0, 10, 10, 0.9) for f in range(10)] + [(f, 2, 0, 0, 10, 10, 0.9) for f in range(10)]
    rows += [(f, 3, 0, 0, 10, 10, 0.9) for f in range(10)] + [(f, 4, 0, 0, 10, 10, 0.9) for f in range(20, 25)]
    players = Players.from_rows(sorted(rows))
    xy = np.array([{1: (2.5, 3.0), 2: (6.5, 6.0), 3: (4.5, 12.0), 4: (4.5, 2.0)}[r[1]] for r in sorted(rows)])
    s = player_stats(players, xy, np.ones(len(rows), dtype=bool), [(0, 10)])
    assert s["rally_frames"] == 10
    assert s["near"]["players_per_frame"] == 2.0 and s["far"]["players_per_frame"] == 1.0
    assert s["near"]["mean_net_distance_m"] == 4.5 and s["far"]["mean_net_distance_m"] == 3.0
    heat = np.array(s["near"]["heat"])
    assert heat.shape == (26, 13) and heat.max() == 1.0 and heat[4 + 3, 2 + 2] == 1.0  # row y+4, column x+2


def test_leaderboards_rank_plausible_items_across_matches():
    def stats(mid, speeds):
        serves = [{"time_s": 1.0, "side": "near", "speed_kmh": v, "plausible": v <= 100} for v in speeds]
        return {"id": mid, "name": f"m{mid}", "our_side": None, "flights": {"serves": serves, "sets": [], "attacks": []},
                "rallies": {"count": 1, "longest": [{"time_s": 0.0, "length_s": 10.0 * mid}]}}
    lb = leaderboards([stats(1, [60.0, 120.0]), stats(2, [80.0])], n=2)
    assert [(s["match_id"], s["speed_kmh"]) for s in lb["serves"]] == [(2, 80.0), (1, 60.0)]
    assert [r["match_id"] for r in lb["rallies"]] == [2, 1]


def test_meta_defaults_and_round_trip(tmp_path):
    assert load_meta(tmp_path / "meta.json") == {"our_side": None}
    save_meta(tmp_path / "meta.json", {"our_side": "far"})
    assert load_meta(tmp_path / "meta.json") == {"our_side": "far"}
```

- [ ] **Step 2:** `uv run --no-sync pytest tests/test_stats.py -q` → FAIL (no module).

- [ ] **Step 3: Implement.** `config.py` `Paths`:

```python
    def meta_json(self, match_id: int) -> Path:
        """Per-match settings, e.g. which end our team plays (stats page)."""
        return self.match_dir(match_id) / "meta.json"
```

`src/vball/stats.py`:

```python
"""Match and cross-match statistics from rallies, 3D ball flights and player tracks
(docs/superpowers/specs/2026-10-06-stats-page-design.md)."""

import json
import sqlite3
from pathlib import Path

import numpy as np

from vball import store
from vball.ball3d import load_flights
from vball.config import Paths
from vball.court import COURT_LENGTH, COURT_WIDTH, NET_Y, load_calibration
from vball.labels import load_labels
from vball.players import Players, load_players, with_court

HEAT_X = (-2.0, 11.0)  # metres, 1 m cells: 13 columns
HEAT_Y = (-4.0, 22.0)  # 26 rows


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"our_side": None}


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
            item["plausible"] = bool(40 <= r["speed_kmh"] <= 100 and r["net_z_m"] > 2.43 and _lands_near_court(r))
            serves.append(item)
        elif from_net <= 4 and r["v0_z"] > 0 and r["net_z_m"] is None:
            item["plausible"] = bool(r["apex_m"] <= 8 and r["speed_kmh"] <= 40)
            sets.append(item)
        elif from_net <= 3 and r["p0_z"] >= 2.3 and r["v0_z"] < 0 and r["net_z_m"] is not None:
            item["plausible"] = bool(30 <= r["speed_kmh"] <= 130)
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
    elif not paths.court_json(match_id).exists():
        out["missing"]["flights"] = f"calibrate the court (with the net clicks), then run: uv run vball ball3d {match_id}"
    else:
        out["missing"]["flights"] = f"run: uv run vball ball3d {match_id}"
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
```

- [ ] **Step 4:** run → 5 pass. **Commit** `feat: stats module (rallies, serves/sets/attacks, side-level players, leaderboards)`.

### Task 2: Stats API

**Files:** modify `src/vball/web/app.py`; test `tests/test_web.py`.

- [ ] **Step 1: Failing tests** (append to `tests/test_web.py`):

```python
def test_stats_without_flights_says_what_to_run(tmp_path):
    client, match_id = make_client(tmp_path)
    body = client.get("/api/stats").json()
    m = body["matches"][0]
    assert m["rallies"]["count"] == 2 and m["rallies"]["source"] == "detected"
    assert "vball ball3d" in m["missing"]["flights"] and body["leaderboards"]["serves"] == []


def test_meta_sets_our_side(tmp_path):
    client, match_id = make_client(tmp_path)
    assert client.put(f"/api/matches/{match_id}/meta", json={"our_side": "middle"}).status_code == 422
    assert client.put(f"/api/matches/{match_id}/meta", json={"our_side": "near"}).json() == {"our_side": "near"}
    assert client.get(f"/api/matches/{match_id}/stats").json()["our_side"] == "near"


def test_match_stats_include_side_level_players(tmp_path):
    from vball.players import save_players

    client, match_id = make_client(tmp_path)
    assert "vball players" in client.get(f"/api/matches/{match_id}/stats").json()["missing"]["players"]
    names = ["far_left_corner", "far_right_corner", "center_left", "center_right"]
    client.put(f"/api/matches/{match_id}/court", json={"ref_frame": 0, "points": court_points(names)})
    fx, fy = apply_h(COURT_TO_IMAGE, [[4.5, 15.0]])[0]
    save_players(tmp_path / "data" / "matches" / str(match_id) / "players.csv",
                 [(f, 4, fx - 20, fy - 120, fx + 20, fy, 0.9) for f in range(40, 100)])
    p = client.get(f"/api/matches/{match_id}/stats").json()["players"]
    assert p["far"]["players_per_frame"] > 0 and p["near"]["players_per_frame"] == 0
    assert abs(p["far"]["mean_net_distance_m"] - 6.0) < 0.05


def test_stats_page_is_served_and_linked(tmp_path):
    client, _ = make_client(tmp_path)
    assert 'id="stats-root"' in client.get("/stats.html").text
    assert 'href="/stats.html"' in client.get("/").text
```

- [ ] **Step 2:** run → FAIL.

- [ ] **Step 3: Implement** in `app.py`: `from vball import stats` (and `store` already imported); model

```python
class MetaBody(BaseModel):
    our_side: Literal["near", "far"] | None = None
```

routes (before the frames route):

```python
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
        stats.save_meta(paths.meta_json(match_id), {"our_side": body.our_side})
        return stats.load_meta(paths.meta_json(match_id))
```

`index.html` header: after the Ball check link add
`<a class="header-link" href="/stats.html" title="Hardest serves, highest sets, longest rallies, where each side plays">Stats →</a>`.

Create a placeholder `stats.html` containing `<main id="stats-root"></main>` (filled in Task 3).

- [ ] **Step 4:** run → pass. **Commit** `feat: stats API and match side setting`.

### Task 3: Stats page and viewer deep links

**Files:** create `src/vball/web/static/stats.html`, `src/vball/web/static/stats.js`; modify `style.css`, `app.js` (`loadMatches`).

- [ ] **Step 1: Read the dataviz skill** before writing the tiles and heat maps (colour, tile and heat-map rules).

- [ ] **Step 2: Viewer deep links** — in `loadMatches`, replace `if (matches.length) await selectMatch(matches[matches.length - 1].id);` with:

```js
  // deep links from the Stats page: /?match=2&t=91.3&show=speed
  const params = new URLSearchParams(location.search);
  for (const name of (params.get("show") || "").split(",").filter(Boolean)) {
    const box = $(`#show-${name}`);
    if (box) box.checked = true;
  }
  const wanted = Number(params.get("match"));
  const id = matchesById[wanted] ? wanted : matches.at(-1)?.id;
  if (id == null) return;
  const t = Number(params.get("t"));
  if (params.has("t") && t >= 0) video.addEventListener("loadedmetadata", () => (video.currentTime = t), { once: true });
  await selectMatch(id);
```

- [ ] **Step 3: Page.** `stats.html` (same header style as the viewer, `<main id="stats-root">`): match picker (All matches + each match), a tile row, a leaderboards grid (four tables), and for a single match a side switch (segmented Near / Far / not set) and two heat-map canvases with per-side numbers. `stats.js`:
  - `GET /api/stats` once; picker change filters to one match and fetches `/api/matches/{id}/stats` for the player section.
  - Tiles (all matches or one): rallies, mean rally length, plausible serves / measured, hardest serve.
  - Tables: rank, match, time (m:ss), value, side ("Us"/"Them" when the match has `our_side`, else "Near"/"Far"); attacks table carries an "experimental" badge and the error note; each row is a link to `/?match=<id>&t=<time_s - 1>&show=speed` (rallies: `show=` nothing).
  - "N not counted (implausible fit)" under the serve table, from the per-match flight lists.
  - Heat maps: canvas per side drawing the court lines (x 0–9, y 0–18, net at 9) over the 13×26 grid in a single-hue sequential scale, near side at the bottom.
  - Side switch: `PUT /api/matches/{id}/meta`, then re-render labels.
  - Empty states from `missing` (e.g. "No 3D flights: run uv run vball ball3d 5").

- [ ] **Step 4: Browser check** on a test server (port 8001, never the user's 8000): all-matches leaderboards render with values from Kent 1/2 and Brunel away 3; one match shows heat maps; the side switch persists across reload; clicking a serve row opens the viewer at that time with Show speed ticked; no console errors. `uv run --no-sync pytest -q`. **Commit** `feat: stats page with leaderboards, side heat maps and viewer deep links`.

### Task 4: Docs

- [ ] `docs/PROJECT_SUMMARY.md`: progress row and a "Stats page" line; commit and push `phase2b`.
