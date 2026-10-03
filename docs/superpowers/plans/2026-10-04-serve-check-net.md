# Serve Check and Net Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop detected "rallies" that do not start with a serve (ball returned to the server between rallies), and add two net-tape clicks to court calibration for later 3D work.

**Architecture:** A new `serve.py` decides, for each active ball run, whether the ball leaves a player standing behind a baseline (player tracks + court calibration). `detect_rallies` takes an optional `keep(start, end)` test; `pipeline.redetect` and `vball tunerallies` build it when a match has `players.csv` and `court.json`. Net points are stored in `court.json` beside the floor points but kept out of the floor homography; the viewer draws the net and pre-loads an existing calibration.

**Tech Stack:** Python 3.12 (uv), numpy, FastAPI, vanilla JS canvas, pytest.

Spec: `docs/superpowers/specs/2026-10-04-serve-check-net-design.md`.

Run tests with `uv run --no-sync pytest -q` (116 passing before this plan).

---

## File map

- `src/vball/court.py` — `NET_HEIGHT_M`, `NET_LANDMARKS`; `Calibration.net_points/net_height_m`; JSON save/load; `calibration_json` adds `net_image` per segment.
- `src/vball/web/app.py` — `PUT /court` accepts net landmarks.
- `src/vball/web/static/app.js`, `index.html` — net landmarks in the calibration list, pre-load existing calibration, draw the net.
- `src/vball/rallies.py` — `detect_rallies(..., keep=None)`.
- `src/vball/serve.py` (new) — `ServeParams`, `reach_box`, `is_serve`, `serve_filter`.
- `src/vball/pipeline.py` — `serve_status`, `serve_keep`; `redetect` uses them.
- `src/vball/tuning.py` — `TuneCase.serve`, `SERVE_GRID`, serve-aware `grid_search`.
- `src/vball/cli.py` — `redetect`/`track` print the serve-check status; `tunerallies` uses `SERVE_GRID` when possible.
- Tests: `tests/test_court.py`, `tests/test_web.py`, `tests/test_rallies.py`, `tests/test_serve.py` (new), `tests/test_pipeline.py`, `tests/test_tuning.py`.

---

### Task 1: Net points in the calibration model

**Files:**
- Modify: `src/vball/court.py`
- Test: `tests/test_court.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_court.py`; add `calibration_json` to the `vball.court` import line)

```python
NET = [{"landmark": "net_left_top", "x": 300.0, "y": 100.0}, {"landmark": "net_right_top", "x": 900.0, "y": 105.0}]


def floor_points(names):
    image, court = image_of(names)
    return [{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(names, image)], image, court


def test_net_points_are_kept_out_of_the_floor_fit(tmp_path):
    points, image, court = floor_points(["far_left_corner", "far_right_corner", "center_left", "center_right"])
    shift = np.array([[1.0, 0, 40.0], [0, 1.0, 0], [0, 0, 1.0]])
    cal = Calibration.create(0, points + NET, [Segment(0, 100, np.eye(3)), Segment(100, 200, shift)])
    assert cal.points == points and cal.net_points == NET and cal.net_height_m == 2.43
    assert np.allclose(apply_h(cal.image_to_court, image), court, atol=1e-6)
    save_calibration(tmp_path / "court.json", cal)
    again = load_calibration(tmp_path / "court.json")
    assert again.net_points == NET and again.net_height_m == 2.43
    body = calibration_json(again)
    assert len(body["points"]) == 4 and body["net_height_m"] == 2.43
    assert body["segments"][0]["net_image"] == [[300.0, 100.0], [900.0, 105.0]]
    assert body["segments"][1]["net_image"] == [[340.0, 100.0], [940.0, 105.0]]  # moved with the camera


def test_calibration_without_net_points(tmp_path):
    points, _, _ = floor_points(["far_left_corner", "far_right_corner", "center_left", "center_right"])
    save_calibration(tmp_path / "court.json", Calibration.create(0, points, []))
    data = json.loads((tmp_path / "court.json").read_text())
    del data["net_points"], data["net_height_m"]  # a file saved before net points existed
    (tmp_path / "court.json").write_text(json.dumps(data))
    cal = load_calibration(tmp_path / "court.json")
    assert cal.net_points == [] and cal.net_height_m == 2.43
    assert calibration_json(cal)["segments"][0]["net_image"] == [None, None]
```

Also add `import json` at the top of `tests/test_court.py`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync pytest tests/test_court.py -q`
Expected: FAIL (`Calibration` has no `net_points`; `calibration_json` import may fail first).

- [ ] **Step 3: Implement** in `src/vball/court.py`

After `COURT_WIDTH, COURT_LENGTH, NET_Y = ...`:

```python
NET_HEIGHT_M = 2.43  # men's indoor; 2.24 for women
```

After `LANDMARKS = {...}`:

```python
# top of the net tape at each sideline (court x, y); the height is Calibration.net_height_m.
# Not floor points: stored for 3D work, never used in the floor homography.
NET_LANDMARKS = {"net_left_top": (0.0, NET_Y), "net_right_top": (COURT_WIDTH, NET_Y)}
```

`Calibration`:

```python
@dataclass
class Calibration:
    ref_frame: int
    points: list[dict]  # {"landmark", "x", "y"} in work-video pixels at ref_frame (floor landmarks only)
    image_to_court: np.ndarray  # at ref_frame
    segments: list[Segment] = field(default_factory=list)
    net_points: list[dict] = field(default_factory=list)  # NET_LANDMARKS clicked at ref_frame
    net_height_m: float = NET_HEIGHT_M

    @classmethod
    def create(cls, ref_frame: int, points: list[dict], segments: list[Segment]) -> "Calibration":
        floor = [p for p in points if p["landmark"] in LANDMARKS]
        net = [p for p in points if p["landmark"] in NET_LANDMARKS]
        image = np.array([[p["x"], p["y"]] for p in floor])
        court = np.array([LANDMARKS[p["landmark"]] for p in floor])
        return cls(ref_frame, floor, fit_homography(image, court), segments, net)
```

`save_calibration` data dict gains `"net_points": cal.net_points, "net_height_m": cal.net_height_m,`.

`load_calibration`:

```python
    return Calibration(
        data["ref_frame"], data["points"], np.array(data["image_to_court"]), segments,
        data.get("net_points", []), data.get("net_height_m", NET_HEIGHT_M),
    )
```

`calibration_json`: add a helper inside and the new keys:

```python
    def net_image(seg: Segment) -> list:
        out = []
        for name in NET_LANDMARKS:
            p = next((q for q in cal.net_points if q["landmark"] == name), None)
            out.append(None if p is None else [round(float(v), 1) for v in apply_h(seg.ref_to_frame, [[p["x"], p["y"]]])[0]])
        return out
```

and in the returned dict: each segment entry gets `"net_image": net_image(s),`; top level gets `"net_points": cal.net_points, "net_height_m": cal.net_height_m, "net_landmarks": NET_LANDMARKS,`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --no-sync pytest tests/test_court.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vball/court.py tests/test_court.py
git commit -m "feat: net tape points in court calibration (kept out of the floor fit)"
```

### Task 2: Court API accepts net points

**Files:**
- Modify: `src/vball/web/app.py` (imports, `put_court`)
- Test: `tests/test_web.py`

- [ ] **Step 1: Failing test** (append)

```python
def test_court_calibration_accepts_net_points(tmp_path):
    client, match_id = make_client(tmp_path)
    points = court_points(["far_left_corner", "far_right_corner", "center_left", "center_right"])
    points.append({"landmark": "net_left_top", "x": 400.0, "y": 300.0})
    res = client.put(f"/api/matches/{match_id}/court", json={"ref_frame": 0, "points": points})
    assert res.status_code == 200
    body = res.json()
    assert len(body["points"]) == 4 and body["net_points"][0]["landmark"] == "net_left_top"
    assert body["segments"][0]["net_image"] == [[400.0, 300.0], None]
```

- [ ] **Step 2:** `uv run --no-sync pytest tests/test_web.py -q -k net` → FAIL (422 unknown landmark).

- [ ] **Step 3: Implement**: import `NET_LANDMARKS` from `vball.court`; in `put_court`:

```python
        unknown = [p.landmark for p in body.points if p.landmark not in LANDMARKS and p.landmark not in NET_LANDMARKS]
```

- [ ] **Step 4:** `uv run --no-sync pytest tests/test_web.py -q` → pass.

- [ ] **Step 5: Commit** `git commit -am "feat: court API accepts net tape points"`

### Task 3: Viewer — net clicks, pre-loaded calibration, net drawing

**Files:**
- Modify: `src/vball/web/static/app.js` (court calibration section), `src/vball/web/static/index.html` (calibration help)

- [ ] **Step 1: `LANDMARK_ORDER`** — append `"net_left_top", "net_right_top"` after `"near_right_corner"`.

- [ ] **Step 2: Segment lookup** — replace `courtMatrixAt` with:

```js
function courtSegmentAt(frame) {
  return court.segments.find((s) => frame >= s.start_frame && frame < s.end_frame) || court.segments.at(-1);
}
```

and in `drawCourt` use `const seg = courtSegmentAt(frame); const H = seg.court_to_image;`.

- [ ] **Step 3: Draw the net** at the end of `drawCourt`:

```js
  // net: a post from each centre-line end up to the clicked tape top, and the tape between them
  const tops = (seg.net_image || []).map((top, i) => top && [project(H, [i * 9, 9]), top]).filter(Boolean);
  ctx.strokeStyle = "rgba(250, 204, 21, 0.95)";
  for (const [[fx, fy], [tx, ty]] of tops) {
    ctx.beginPath();
    ctx.moveTo(r.x + fx * r.scale, r.y + fy * r.scale);
    ctx.lineTo(r.x + tx * r.scale, r.y + ty * r.scale);
    ctx.stroke();
  }
  if (tops.length === 2) {
    ctx.beginPath();
    ctx.moveTo(r.x + tops[0][1][0] * r.scale, r.y + tops[0][1][1] * r.scale);
    ctx.lineTo(r.x + tops[1][1][0] * r.scale, r.y + tops[1][1][1] * r.scale);
    ctx.stroke();
  }
```

- [ ] **Step 4: Pre-load and next-missing landmark.** Add:

```js
function nextMissing(from) {
  let i = from;
  while (i < LANDMARK_ORDER.length && calib.points.some((p) => p.landmark === LANDMARK_ORDER[i])) i += 1;
  return i;
}
```

`startCalibration` becomes:

```js
function startCalibration() {
  video.pause();
  if (court) {
    // re-open the saved calibration at its frame so only missing points (e.g. the net) need clicking
    video.currentTime = court.ref_frame / matchFps;
    const saved = [...court.points, ...(court.net_points || [])].map(({ landmark, x, y }) => ({ landmark, x, y }));
    calib = { frame: court.ref_frame, index: 0, points: saved };
  } else {
    calib = { frame: Math.round(video.currentTime * matchFps), index: 0, points: [] };
  }
  calib.index = nextMissing(0);
  $("#calib-result").textContent = "";
  renderCalib();
}
```

Click handler: replace `calib.index += 1;` with `calib.index = nextMissing(calib.index + 1);`. Skip handler: `if (calib && calib.index < LANDMARK_ORDER.length) calib.index = nextMissing(calib.index + 1);`.

`renderCalib`: Save needs 4 floor points:

```js
  $("#calib-save").disabled = calib.points.filter((p) => !p.landmark.startsWith("net_")).length < 4;
```

- [ ] **Step 5: Help text** in `index.html` `#calib-panel` paragraph, append: ` The last two are the top of the net tape where it meets each antenna; they are optional. Re-opening a saved calibration keeps its points, so you only click what is missing.`

- [ ] **Step 6: Browser check** on a separate server (`uv run --no-sync vball serve --port 8001`, background) — never on the user's port 8000 server. Back up `data/matches/1/court.json` first and restore it afterwards. Open match 1, *Calibrate court*: the video seeks to the saved frame, the list shows the 10 floor points ticked and *net left top* active. Click two net points, *Save*, tick *Show court*: yellow posts and tape appear at the clicks and follow a camera segment change. Check the console for errors. Restore the backup.

- [ ] **Step 7:** `uv run --no-sync pytest -q` → all pass. Commit `git commit -am "feat: net tape clicks, re-open saved calibration, draw the net"`

### Task 4: `detect_rallies` takes a `keep` test

**Files:**
- Modify: `src/vball/rallies.py`
- Test: `tests/test_rallies.py`

- [ ] **Step 1: Failing test** (append)

```python
def test_keep_rejects_runs_before_padding():
    track = make_track(N, flights=[(10, 18), (30, 38)])
    seen = []

    def keep(start, end):
        seen.append(start)
        return start < 20 * FPS

    rallies = detect_rallies(track, fps=FPS, width=W, height=H, keep=keep)
    assert len(rallies) == 1 and rallies[0].start_frame < 20 * FPS
    assert len(seen) == 2 and abs(seen[1] - 30 * FPS) <= 15  # unpadded run start
```

- [ ] **Step 2:** run → FAIL (unexpected keyword `keep`).

- [ ] **Step 3: Implement**: add `from collections.abc import Callable`; signature gains `keep: Callable[[int, int], bool] | None = None`; after the `min_rally_s` filter:

```python
    if keep is not None:
        active = [(s, e) for s, e in active if keep(s, e)]
```

- [ ] **Step 4:** `uv run --no-sync pytest tests/test_rallies.py -q` → pass.
- [ ] **Step 5: Commit** `git commit -am "feat: detect_rallies accepts a keep(start, end) test"`

### Task 5: Serve check module

**Files:**
- Create: `src/vball/serve.py`
- Test: `tests/test_serve.py`

- [ ] **Step 1: Failing tests** — `tests/test_serve.py`:

```python
import numpy as np
from helpers import FPS

from vball.ball.track import BallTrack
from vball.players import Players
from vball.serve import ServeParams, is_serve, serve_filter

N = 20 * FPS
START = 300  # active run start (frame)
BOX = (600.0, 400.0, 680.0, 600.0)  # server, 80 x 200 px; reach box x 540..740, y 300..600


def ball(points):
    """points: {frame: (x, y)}"""
    visible = np.zeros(N, dtype=bool)
    x, y = np.full(N, np.nan), np.full(N, np.nan)
    for f, (px, py) in points.items():
        visible[f], x[f], y[f] = True, px, py
    return BallTrack(visible, x, y)


def serve_ball():
    # held at the hand for 10 frames, then hit up and away (15 px/frame)
    pts = {f: (640.0, 380.0) for f in range(START - 10, START + 1)}
    pts.update({f: (640.0 + 5 * (f - START), 380.0 - 15 * (f - START)) for f in range(START + 1, START + 40)})
    return ball(pts)


def return_ball():
    # lobbed in from far away, caught and held by the player
    pts = {f: (640.0, 100.0 + 14 * (f - (START - 20))) for f in range(START - 20, START)}
    pts.update({f: (640.0, 400.0) for f in range(START, START + 40)})
    return ball(pts)


def players(court_y, box=BOX, frames=range(START - 40, START + 40)):
    rows = [(f, 7, *box, 0.9) for f in frames]
    court_xy = np.tile([4.5, court_y], (len(rows), 1))
    return Players.from_rows(rows), court_xy


def check(track, court_y, box=BOX):
    p, xy = players(court_y, box)
    return is_serve(START, track, p, xy, FPS, ServeParams())


def test_serve_from_behind_the_near_baseline():
    assert check(serve_ball(), -1.0)


def test_serve_from_behind_the_far_baseline():
    assert check(serve_ball(), 19.0)


def test_ball_returned_to_the_server_is_not_a_serve():
    assert not check(return_ball(), -1.0)


def test_throw_from_mid_court_is_not_a_serve():
    assert not check(serve_ball(), 5.0)


def test_ball_not_near_anyone_is_not_a_serve():
    assert not check(serve_ball(), -1.0, box=(100.0, 400.0, 180.0, 600.0))


def test_serve_long_before_the_run_is_ignored():
    p, xy = players(-1.0)
    assert not is_serve(START + 5 * FPS, serve_ball(), p, xy, FPS, ServeParams())


def test_filter_sorts_players_by_frame():
    p, xy = players(-1.0)
    order = np.random.default_rng(0).permutation(len(p))
    shuffled = Players(p.frame[order], p.track_id[order], p.box[order], p.score[order])
    keep = serve_filter(serve_ball(), shuffled, xy[order], FPS)
    assert keep(START, START + 100)
```

- [ ] **Step 2:** `uv run --no-sync pytest tests/test_serve.py -q` → FAIL (no module `vball.serve`).

- [ ] **Step 3: Implement** `src/vball/serve.py`:

```python
"""Serve check: a rally starts with the ball leaving a player who stands behind a baseline.

Between rallies the ball is thrown, rolled or lobbed back to the server, often over the net; those runs end
at the server instead of starting there (docs/superpowers/specs/2026-10-04-serve-check-net-design.md)."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vball.ball.track import BallTrack
from vball.court import COURT_LENGTH, COURT_WIDTH
from vball.players import Players


@dataclass(frozen=True)
class ServeParams:
    search_before_s: float = 1.0  # look for the server from run start - this ...
    search_after_s: float = 0.5  # ... to run start + this
    baseline_margin_m: float = 0.5  # foot y < margin or > 18 - margin counts as behind a baseline
    side_margin_m: float = 1.5  # foot x within [-margin, 9 + margin]
    reach_w: float = 0.75  # player box widened by this * box width each side
    reach_h: float = 0.5  # and by this * box height above the top (hand at the hit)
    leave_s: float = 0.7  # where the ball is this long after the contact decides
    min_serve_rally_s: float = 1.5  # replaces RallyParams.min_rally_s when the serve check is on


def reach_box(box: np.ndarray, p: ServeParams) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    return x1 - p.reach_w * w, y1 - p.reach_h * h, x2 + p.reach_w * w, y2


def _inside(pt: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _leaves(c: int, track_id: int, box: np.ndarray, track: BallTrack, players: Players, p: ServeParams, fps: float) -> bool:
    """After contact frame c, does the ball end up outside the server's reach and farther from them?"""
    later = np.flatnonzero(track.visible[c + 1 : c + round(p.leave_s * fps) + 1])
    if len(later) == 0:
        return False
    f = c + 1 + int(later[-1])
    a, b = np.searchsorted(players.frame, [f, f + 1])
    same = np.flatnonzero(players.track_id[a:b] == track_id)
    if len(same):
        box = players.box[a + same[0]]  # the server's box when the ball is looked at again
    centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    start, end = (track.x[c], track.y[c]), (track.x[f], track.y[f])
    return not _inside(end, reach_box(box, p)) and math.dist(end, centre) > math.dist(start, centre)


def is_serve(
    start: int, track: BallTrack, players: Players, court_xy: np.ndarray, fps: float, p: ServeParams
) -> bool:
    """Does the active run starting at frame `start` begin with a serve? `players` must be sorted by frame;
    court_xy is each row's foot position in court metres (players.with_court)."""
    lo = max(0, start - round(p.search_before_s * fps))
    hi = min(len(track), start + round(p.search_after_s * fps) + 1)
    a, b = np.searchsorted(players.frame, [lo, hi])
    x, y = court_xy[a:b, 0], court_xy[a:b, 1]
    behind = (
        (x >= -p.side_margin_m) & (x <= COURT_WIDTH + p.side_margin_m)
        & ((y < p.baseline_margin_m) | (y > COURT_LENGTH - p.baseline_margin_m))
    )
    for i in a + np.flatnonzero(behind):
        c = int(players.frame[i])
        if track.visible[c] and _inside((track.x[c], track.y[c]), reach_box(players.box[i], p)):
            if _leaves(c, int(players.track_id[i]), players.box[i], track, players, p, fps):
                return True
    return False


def serve_filter(
    track: BallTrack, players: Players, court_xy: np.ndarray, fps: float, params: ServeParams = ServeParams()
) -> Callable[[int, int], bool]:
    """keep(start, end) for detect_rallies."""
    order = np.argsort(players.frame, kind="stable")
    players = Players(players.frame[order], players.track_id[order], players.box[order], players.score[order])
    court_xy = court_xy[order]
    return lambda start, end: is_serve(start, track, players, court_xy, fps, params)
```

- [ ] **Step 4:** `uv run --no-sync pytest tests/test_serve.py -q` → 7 pass.
- [ ] **Step 5: Commit** `git add src/vball/serve.py tests/test_serve.py && git commit -m "feat: serve check (ball leaves a player behind a baseline)"`

### Task 6: Use the serve check in redetect

**Files:**
- Modify: `src/vball/pipeline.py`, `src/vball/cli.py` (`redetect`, `track` branches)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Failing test** (append to `tests/test_pipeline.py`)

```python
@requires_ffmpeg
def test_redetect_applies_the_serve_check_when_players_and_court_exist(tmp_path):
    import numpy as np

    from vball.court import LANDMARKS, Calibration, apply_h, save_calibration
    from vball.pipeline import serve_status
    from vball.players import save_players

    src = make_test_video(tmp_path / "match.mp4", seconds=20, fps=30)
    paths = Paths(tmp_path / "data")
    match_id = process_match(src, paths, encoder="libx264", ball_runner=fake_ball_runner)
    conn = store.connect(paths.db_path)
    assert serve_status(paths, match_id) == "serve check skipped: no player tracks"
    assert len(redetect(conn, paths, match_id)) == 1

    court_to_image = np.array([[20.0, 0.0, 70.0], [0.0, -10.0, 220.0], [0.0, 0.0, 1.0]])
    names = ["far_left_corner", "far_right_corner", "near_left_corner", "near_right_corner"]
    image = apply_h(court_to_image, np.array([LANDMARKS[n] for n in names]))
    points = [{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(names, image)]
    save_calibration(paths.court_json(match_id), Calibration.create(0, points, []))
    save_players(paths.players_csv(match_id), [(f, 1, 300, 10, 310, 30, 0.9) for f in range(600)])  # never near the ball

    assert serve_status(paths, match_id) == "serve check on"
    assert redetect(conn, paths, match_id) == []
```

- [ ] **Step 2:** run → FAIL (`serve_status` missing).

- [ ] **Step 3: Implement** in `src/vball/pipeline.py`:

```python
from dataclasses import replace

from vball.court import load_calibration
from vball.players import load_players, with_court
from vball.serve import ServeParams, serve_filter


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
```

`redetect` becomes:

```python
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
```

In `cli.py`, after each `pipeline.redetect(...)` print in the `redetect` and `track` branches, add `print(pipeline.serve_status(paths, args.match_id))`.

- [ ] **Step 4:** `uv run --no-sync pytest -q` → all pass.
- [ ] **Step 5: Commit** `git commit -am "feat: redetect applies the serve check when players and court exist"`

### Task 7: Serve-aware rally tuning

**Files:**
- Modify: `src/vball/tuning.py`, `src/vball/cli.py` (`tunerallies`)
- Test: `tests/test_tuning.py`

- [ ] **Step 1: Failing test** (append)

```python
def test_grid_search_with_serve_check_drops_runs_without_a_server():
    import numpy as np

    from vball.players import Players

    track = make_track(60 * FPS, flights=[(10, 18), (30, 38)])  # ball starts each flight at (40, 360)
    frames = range(int(9.5 * FPS), 11 * FPS)
    server = Players.from_rows([(f, 1, 0.0, 300.0, 80.0, 500.0, 0.9) for f in frames])
    court_xy = np.tile([4.5, -1.0], (len(server), 1))
    gt = [(9.5, 19.5)]
    plain = grid_search([TuneCase(track, FPS, 1280, 720, gt)], grid={"max_gap_s": [2.0]})
    served = grid_search(
        [TuneCase(track, FPS, 1280, 720, gt, serve=(server, court_xy))], grid={"min_serve_rally_s": [1.5]}
    )
    assert plain[0].f1s[0] < 1.0  # the second flight is a false rally
    assert served[0].f1s == [1.0]
```

- [ ] **Step 2:** run → FAIL (`TuneCase` has no `serve`).

- [ ] **Step 3: Implement** in `src/vball/tuning.py`:

```python
from dataclasses import dataclass, fields, replace

from vball.players import Players
from vball.serve import ServeParams, serve_filter

SERVE_GRID = {
    "min_serve_rally_s": [0.5, 1.0, 1.5],
    "reach_w": [0.5, 0.75, 1.0],
    "reach_h": [0.25, 0.5, 1.0],
    "baseline_margin_m": [0.5, 1.5],
}
RALLY_KEYS = {f.name for f in fields(RallyParams)}
```

`TuneCase` gains `serve: tuple[Players, np.ndarray] | None = None  # (players, court_xy) for the serve check`.

`grid_search(cases, grid=GRID, base=RallyParams(), serve_base=ServeParams())`; per combination:

```python
        params = dict(zip(grid, values))
        rally_params = replace(base, **{k: v for k, v in params.items() if k in RALLY_KEYS})
        serve_params = replace(serve_base, **{k: v for k, v in params.items() if k not in RALLY_KEYS})
        f1s = []
        for c in cases:
            keep, rp = None, rally_params
            if c.serve is not None:
                keep = serve_filter(c.track, *c.serve, c.fps, serve_params)
                rp = replace(rally_params, min_rally_s=serve_params.min_serve_rally_s)
            rallies = detect_rallies(c.track, c.fps, c.width, c.height, rp, keep=keep)
```

(rest unchanged).

`cli.py` `tunerallies`: build each case with `serve=` when `pipeline.serve_status(paths, match_id) == "serve check on"` (load players, `with_court`); if every case has it use `SERVE_GRID`, else `GRID` and print which. Replace the `current` line with:

```python
        grid = SERVE_GRID if all(c.serve is not None for c in cases) else GRID
        print("serve check on" if grid is SERVE_GRID else "serve check off (a match lacks player tracks or a court)")
        defaults = {**vars(RallyParams()), **vars(ServeParams())}
        current = grid_search(cases, grid={k: [defaults[k]] for k in grid})[0]
```

and `grid_search(cases, grid)` for the ranking.

- [ ] **Step 4:** `uv run --no-sync pytest -q` → all pass.
- [ ] **Step 5: Commit** `git commit -am "feat: tunerallies tunes the serve check when players and court exist"`

### Task 8: Player backend decision (part 2, Task 8 of the court/players plan)

- [ ] Wait for the background YOLO/RAVEL jobs (`data/exp/players/yolo.log`, `ravel.log`) and the user's calibrations of #1, #2, #11.
- [ ] `uv run --no-sync vball playereval <id> --players data/exp/players/players_<backend>_m<id>.csv` for both backends on each calibrated set; record results.
- [ ] Spot-check in the viewer (Show players) on the user's server after a restart.
- [ ] Choose a backend; copy its CSVs to `data/matches/<id>/players.csv` for 1, 2, 11; start the chosen backend on the other sets in ≤ 2 h background batches.

### Task 9: Evaluate the serve check

- [ ] Baseline numbers before redetect: `uv run --no-sync vball eval 1` and `eval 2` (current F1 0.92 / 0.90).
- [ ] `uv run --no-sync vball tunerallies 1 2` → note current-defaults line and top results.
- [ ] Set `ServeParams` defaults to the best combination if it beats the defaults; `vball redetect 1`, `redetect 2`, `eval 1`, `eval 2`. Success: ≥ 5 of the 8 false rallies gone, no previously matched rally lost, F1 above 0.92 / 0.90. Re-run the error listing (false rallies / misses) and contact-sheet any newly lost rally to see which step rejected it.
- [ ] Held-out: once the user has labelled #11, `redetect 11` and `eval 11` with and without `players.csv` (rename it temporarily) — F1 must not drop.
- [ ] Commit tuned defaults: `git commit -am "feat: tuned serve check defaults"`.

### Task 10: Results and summary

- [ ] Write `docs/results/phase2b-court-players.md` (calibration errors, camera segments, backend comparison and decision, serve check before/after per set, held-out result, failure modes).
- [ ] Update `docs/PROJECT_SUMMARY.md`: progress table, decisions (serve check rather than net crossing; net clicks stored for 3D), results, next steps (3D ball arcs from net + court for speed and over-the-tape).
- [ ] `uv run --no-sync pytest -q`; commit; push `phase2b`; ask the user how to merge (superpowers:finishing-a-development-branch).
