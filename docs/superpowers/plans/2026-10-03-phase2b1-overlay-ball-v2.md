# vball Phase 2b Part 1 (Ball Overlay, External Benchmark, Ball v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user see the tracked ball on the video, measure the external VballNet ball models on our clicked frames, and fine-tune a v2 tracker that adds the VballNet hand labels to our pseudo-labels.

**Architecture:** A new `/api/matches/{id}/ball` endpoint feeds a `<canvas>` overlay synced per frame with `requestVideoFrameCallback`. `vball/ball/external.py` turns VballNet clips into labelled `TrainSource`s (real negatives), which `vball finetune --vballnet` mixes into the existing cache. External ONNX models run in their own uv environment and are scored with `vball balleval --pred`. Spec: `docs/superpowers/specs/2026-10-03-phase2b-design.md`.

**Tech Stack:** Python 3.12 (uv), FastAPI, vanilla JS canvas, PyTorch, ffmpeg; external repo with its own Python 3.14 env.

**Conventions:** Branch `phase2b` (exists, holds the reference-doc commit). Git Bash at `C:/Users/mateusz/Projects/Vball`; Python via `uv run --no-sync`. Files with backslashes via the editor tool. Commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Held-out sets: match 2 (Kent set 2) and 11 (Brunel away set 3). Current default: `models/tracknet_vball_v1.pt` at threshold 0.3.

---

### Task 1: Ball track API

**Files:** Modify `src/vball/web/app.py`; Test `tests/test_web.py`

- [ ] **Step 1: Failing tests** — add `make_track, write_tracknet_csv` to the `from helpers import ...` line of `tests/test_web.py`, then append:

```python
def test_ball_track_endpoint(tmp_path):
    client, match_id = make_client(tmp_path)  # 1280x720 at 30 fps, 900 frames
    write_tracknet_csv(tmp_path / "data" / "matches" / str(match_id) / "ball.csv", make_track(900, flights=[(1.0, 2.0)]))
    data = client.get(f"/api/matches/{match_id}/ball").json()
    assert data["fps"] == 30.0 and (data["width"], data["height"]) == (1280, 720)
    assert len(data["x"]) == len(data["y"]) == 900
    assert data["x"][0] is None
    assert (data["x"][31], data["y"][31]) == (60.0, 360.0)


def test_ball_track_missing_is_404(tmp_path):
    client, match_id = make_client(tmp_path)
    assert client.get(f"/api/matches/{match_id}/ball").status_code == 404
```

- [ ] **Step 2:** `uv run --no-sync pytest tests/test_web.py -q` → 2 failures.

- [ ] **Step 3: Implement** — in `src/vball/web/app.py` import `from vball.ball.track import load_tracknet_csv` and add before the mounts:

```python
    @app.get("/api/matches/{match_id}/ball")
    def get_ball(match_id: int, conn: sqlite3.Connection = Depends(db)) -> dict:
        match = require_match(conn, match_id)
        path = paths.ball_csv(match_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="no ball track")
        track = load_tracknet_csv(path, match["n_frames"])

        def column(values):
            return [round(float(v), 1) if seen else None for v, seen in zip(values, track.visible)]

        return {
            "fps": match["fps"],
            "width": match["width"],
            "height": match["height"],
            "x": column(track.x),
            "y": column(track.y),
        }
```

- [ ] **Step 4:** `uv run --no-sync pytest -q` → all pass.
- [ ] **Step 5: Commit** `feat: ball track API for the viewer overlay`.

---

### Task 2: Overlay canvas and "Show ball"

**Files:** Modify `src/vball/web/static/index.html`, `app.js`, `style.css`; Test `tests/test_web.py`

- [ ] **Step 1: Failing test**

```python
def test_viewer_has_overlay_and_ball_toggle(tmp_path):
    client, _ = make_client(tmp_path)
    page = client.get("/").text
    assert 'id="overlay"' in page and 'id="show-ball"' in page
```

- [ ] **Step 2:** run → 1 failure.

- [ ] **Step 3: HTML** — in `index.html` replace `<video id="video" controls preload="metadata"></video>` with:

```html
      <div class="video-wrap">
        <video id="video" controls preload="metadata"></video>
        <canvas id="overlay" class="overlay" aria-hidden="true"></canvas>
      </div>
```

and in `<header>` after the speed `<label>` add:

```html
    <label title="Draw the tracked ball and its last 8 positions on the video"><input type="checkbox" id="show-ball"> Show ball</label>
```

- [ ] **Step 4: CSS** (append to `style.css`):

```css
.video-wrap { position: relative; }
.overlay { position: absolute; left: 0; top: 0; pointer-events: none; }
```

- [ ] **Step 5: JS** — in `app.js`, in `selectMatch` after `video.src = ...` add `loadBall(id);`, and add this section before `// ---------- input ----------`:

```javascript
// ---------- overlay ----------

const overlay = $("#overlay");
let ballData = null; // {fps, width, height, x: [...], y: [...]} in work-video pixels, null = not detected

async function loadBall(id) {
  ballData = null;
  try {
    ballData = await getJson(`/api/matches/${id}/ball`);
  } catch {
    ballData = null; // no ball track for this match yet
  }
  drawOverlay();
}

// Where the picture sits inside the <video> box (object-fit: contain adds letterbox bars).
function contentRect() {
  const w = video.clientWidth;
  const h = video.clientHeight;
  const vw = video.videoWidth || 16;
  const vh = video.videoHeight || 9;
  const scale = Math.min(w / vw, h / vh);
  return { x: (w - vw * scale) / 2, y: (h - vh * scale) / 2, scale };
}

function drawOverlay(mediaTime = video.currentTime) {
  const dpr = window.devicePixelRatio || 1;
  const w = video.clientWidth;
  const h = video.clientHeight;
  overlay.style.width = `${w}px`;
  overlay.style.height = `${h}px`;
  overlay.width = Math.round(w * dpr);
  overlay.height = Math.round(h * dpr);
  const ctx = overlay.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!$("#show-ball").checked || !ballData) return;
  const r = contentRect();
  const s = (r.scale * (video.videoWidth || ballData.width)) / ballData.width;
  const frame = Math.round(mediaTime * ballData.fps);
  for (let k = 8; k >= 0; k--) {
    const f = frame - k;
    const x = ballData.x[f];
    const y = ballData.y[f];
    if (f < 0 || x == null) continue;
    ctx.beginPath();
    ctx.arc(r.x + x * s, r.y + y * s, k === 0 ? 9 : 4, 0, 2 * Math.PI);
    if (k === 0) {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#facc15";
      ctx.stroke();
    } else {
      ctx.fillStyle = `rgba(250, 204, 21, ${0.6 * (1 - k / 9)})`;
      ctx.fill();
    }
  }
}

function onVideoFrame(_now, meta) {
  drawOverlay(meta.mediaTime);
  video.requestVideoFrameCallback(onVideoFrame);
}
if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
  video.requestVideoFrameCallback(onVideoFrame);
} else {
  video.addEventListener("timeupdate", () => drawOverlay());
}
video.addEventListener("seeked", () => drawOverlay());
video.addEventListener("loadedmetadata", () => drawOverlay());
window.addEventListener("resize", () => drawOverlay());
$("#show-ball").onchange = () => drawOverlay();
```

- [ ] **Step 6:** `uv run --no-sync pytest -q` → all pass.

- [ ] **Step 7: Browser check (copy of the data, port 8766)** — copy `data/vball.db` and match 2 (`ball.csv`, `ball_test.csv`, hard-link `work.mp4`) to `data/uitest`, serve with `VBALL_DATA=data/uitest`. In the page: select match 2, tick "Show ball", seek to a clicked test frame that is a TP (from `ball_test.csv`; `video.currentTime = (frame + 0.5) / fps`), screenshot: the yellow ring must sit on the ball. Play a rally: the ring and trail must follow the ball. Untick: overlay clears. Delete `data/uitest`.

- [ ] **Step 8: Commit** `feat: Show ball overlay on the viewer (frame-synced canvas)`.

---

### Task 3: VballNet clips as labelled training sources

**Files:** Create `src/vball/ball/external.py`; Test `tests/test_external.py`

- [ ] **Step 1: Failing tests** — `tests/test_external.py`:

```python
import numpy as np
from helpers import make_test_video, requires_ffmpeg

from vball.ball.external import clip_source, find_vballnet_clips
from vball.ball.pseudo import IGNORED, NEGATIVE, POSITIVE
from vball.video import probe


def make_clip(root, split, match, stem, rows, labelled=True):
    d = root / split / match
    (d / "video").mkdir(parents=True)
    (d / "csv").mkdir()
    make_test_video(d / "video" / f"{stem}.mp4", seconds=1, fps=30, size="640x360")
    name = f"{stem}_ball.csv" if labelled else f"{stem}_predict_ball.csv"
    lines = "".join(f"{f},{v},{x},{y}\n" for f, v, x, y in rows)
    (d / "csv" / name).write_text("Frame,Visibility,X,Y\n" + lines)


@requires_ffmpeg
def test_find_skips_prediction_only_clips(tmp_path):
    make_clip(tmp_path, "train", "match1", "a", [(0, 1, 10, 10)])
    make_clip(tmp_path, "test", "match2", "b", [(0, 1, 10, 10)], labelled=False)
    clips = find_vballnet_clips(tmp_path)
    assert [c.video.stem for c in clips] == ["a"]


@requires_ffmpeg
def test_clip_source_states_and_scaling(tmp_path):
    make_clip(tmp_path, "train", "match1", "a", [(0, 0, -1, -1), (1, 1, 320, 180), (5, 1, 640, 0)])
    src = clip_source(find_vballnet_clips(tmp_path)[0], tmp_path / "cache")
    assert probe(src.track_video).width == 512
    assert len(src.state) == 30
    assert src.state[0] == NEGATIVE  # hand-labelled "no ball"
    assert src.state[1] == POSITIVE and (src.x[1], src.y[1]) == (256.0, 144.0)
    assert src.state[2] == IGNORED  # no row for this frame
    assert np.isnan(src.x[0])
```

- [ ] **Step 2:** run → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/vball/ball/external.py`:

```python
"""External hand-labelled ball data (VballNet dataset, TrackNetV3 layout) as training sources."""

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vball.ball.pseudo import IGNORED, NEGATIVE, POSITIVE
from vball.ball.tracknet import TRACKNET_H, TRACKNET_W, downscale_command
from vball.ball.train_data import TrainSource
from vball.video import probe


@dataclass(frozen=True)
class LabelledClip:
    video: Path
    labels: Path


def find_vballnet_clips(root: Path) -> list[LabelledClip]:
    """Clips under root/{split}/{match}/video/*.mp4 with hand labels in ../csv/<stem>_ball.csv.
    Clips that only have <stem>_predict_ball.csv (model output) are skipped."""
    clips = []
    for video in sorted(root.glob("*/*/video/*.mp4")):
        labels = video.parent.parent / "csv" / f"{video.stem}_ball.csv"
        if labels.exists():
            clips.append(LabelledClip(video, labels))
    return clips


def clip_source(clip: LabelledClip, cache_dir: Path) -> TrainSource:
    """Labelled frames: visible -> POSITIVE, not visible -> NEGATIVE (real labels); unlabelled -> IGNORED."""
    info = probe(clip.video)
    small = cache_dir / f"{clip.video.parent.parent.parent.name}_{clip.video.parent.parent.name}_{clip.video.stem}.mp4"
    if not small.exists():
        small.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(downscale_command(clip.video, small), check=True)
    n = probe(small).n_frames
    state = np.full(n, IGNORED, dtype=np.int8)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    with open(clip.labels, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["Frame"])
            if not 0 <= frame < n:
                continue
            if int(float(row["Visibility"])) == 1:
                state[frame] = POSITIVE
                x[frame] = float(row["X"]) * TRACKNET_W / info.width
                y[frame] = float(row["Y"]) * TRACKNET_H / info.height
            else:
                state[frame] = NEGATIVE
    return TrainSource(small, state, x, y)


def clip_sources(root: Path, cache_dir: Path) -> list[TrainSource]:
    return [clip_source(clip, cache_dir) for clip in find_vballnet_clips(root)]
```

- [ ] **Step 4:** `uv run --no-sync pytest -q` → all pass.
- [ ] **Step 5: Commit** `feat: VballNet hand-labelled clips as ball training sources`.

---

### Task 4: `vball finetune --vballnet`

**Files:** Modify `src/vball/cli.py`; Test `tests/test_cli.py`

- [ ] **Step 1: Failing test**

```python
def test_finetune_adds_vballnet_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("VBALL_DATA", str(tmp_path / "data"))
    captured = {}
    monkeypatch.setattr("vball.ball.finetune.prepare_sources", lambda conn, paths, ids: ["OWN"])
    monkeypatch.setattr("vball.ball.external.clip_sources", lambda root, cache_dir: ["EXT1", "EXT2"])
    monkeypatch.setattr(
        "vball.ball.train_data.build_cache",
        lambda sources, out_dir, n_windows: captured.update(sources=sources, n=n_windows),
    )
    monkeypatch.setattr("vball.ball.finetune.finetune", lambda *args, **kwargs: [0.0])
    argv = ["finetune", "--matches", "1", "--out", str(tmp_path / "v2.pt"), "--vballnet", str(tmp_path / "vn"),
            "--rebuild-cache", "--windows", "3000"]
    assert main(argv) == 0
    assert captured == {"sources": ["OWN", "EXT1", "EXT2"], "n": 3000}
```

- [ ] **Step 2:** run → fails (`unrecognized arguments: --vballnet`).

- [ ] **Step 3: Implement** — parser: `f.add_argument("--vballnet", type=Path, help="VballNet dataset root (…/vballnet-dataset/data) to add as labelled sources")`. In the `finetune` branch import `from vball.ball import external` and after `sources = ft.prepare_sources(...)` (outside the `try/finally`):

```python
        if args.vballnet:
            sources += external.clip_sources(args.vballnet, paths.data_dir / "external_cache" / "vballnet")
            print(f"{len(sources)} training sources including VballNet clips")
```

- [ ] **Step 4:** `uv run --no-sync pytest -q` → all pass.
- [ ] **Step 5: Commit** `feat: vball finetune --vballnet adds external hand labels`.

---

### Task 5: Benchmark the external ball models (no code)

- [ ] **Step 1: Run each ONNX model on the held-out work videos** (first run creates the repo's own Python 3.14 env; run in the background):

```bash
R=data/external/asigatchov-fast-volleyball-tracking-inference
for model in VballNetV4c_seq9_grayscale_20260908_213829 VballNetGridV3_seq9_grayscale_20260908_225156 VballNetGridV2b_seq9_grayscale_20260909_001145 VballNetFastV1_seq9_grayscale_233_h288_w512; do
  for m in 2 11; do
    out="$(pwd)/data/exp/ext/$model/m$m"
    uv run --directory "$R" src/inference_onnx_seq_gray_v2.py --video_path "$(pwd)/data/matches/$m/work.mp4" \
      --model_path "models/$model.onnx" --output_dir "$out" --only_csv
  done
done
```
Expected: `data/exp/ext/<model>/m<id>/work/ball.csv` (columns `Frame,Visibility,X,Y,Radius`, pixels of the work video). Note wall time per run.

- [ ] **Step 2: Score**

```bash
for f in data/exp/ext/*/m*/work/ball.csv; do m=$(echo $f | sed 's#.*/m\([0-9]*\)/work.*#\1#'); echo "$f"; uv run --no-sync vball balleval $m --pred "$f" | head -1; done
```
Also score at 30 px tolerance with the inline snippet used in phase 2a (`ball_metrics(..., 30.0)`), since near-camera balls are large.

- [ ] **Step 3: Record** a table in `docs/results/phase2b-ball.md` (create it): model, precision/recall at 15 and 30 px for both held-out sets, run time, next to v1 @ 0.3. Commit `docs: external ball model benchmark`.

---

### Task 6: Fine-tune v2, evaluate, decide

- [ ] **Step 1: Train** (background, `timeout: 7200000`):

```bash
uv run --no-sync vball finetune --matches 1 3 4 5 6 7 8 9 10 --vballnet data/external/vballnet-dataset/data \
  --out models/tracknet_vball_v2.pt --windows 3000 --rebuild-cache > data/exp/finetune_v2.log 2>&1
```
Check the log: 89 VballNet sources, loss per epoch falling.

- [ ] **Step 2: Track held-out sets and Kent 1**

```bash
for m in 1 2 11; do for t in 0.3 0.5; do
  uv run --no-sync vball track $m --weights models/tracknet_vball_v2.pt --threshold $t --out data/exp/m${m}_v2_t${t/./}.csv
done; done
```

- [ ] **Step 3: Score and re-tune** — `vball balleval 2|11 --pred data/exp/m{2,11}_v2_t{03,05}.csv`; rally re-tune with `tuning.grid_search` on Kent 1+2 v2 tracks (same snippet as phase 2a).

- [ ] **Step 4: Decide** — adopt v2 (set `TRACKNET_WEIGHTS`, threshold, rally params; re-track all 11 sets with `vball track <id>`, backing up `ball.csv` first) only if held-out recall is higher at no worse precision than v1 @ 0.3 and rally F1 within −0.02; otherwise keep v1. Record the table and decision in `docs/results/phase2b-ball.md`; also update the comment above `TRACKNET_WEIGHTS` with the rebuild command if adopted. Run `uv run --no-sync pytest -q`.

- [ ] **Step 5: Commit and push** `feat: ball model v2 with VballNet labels (results in phase2b-ball.md)` (or `docs:` if not adopted); `git push -u origin phase2b`.
