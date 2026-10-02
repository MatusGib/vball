# vball Phase 2a (Ball Tracker Improvement) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise TrackNet ball recall in rally frames on held-out indoor footage by +15 points (precision within −3, rally F1 within −0.02), via a threshold sweep and then self-training on automatic pseudo-labels.

**Architecture:** New commands re-track a match with any weights/threshold (`vball track`) and score it against ~100 hand-clicked frames per held-out set (`vball balleval`, clicked on a new `/ball.html` page). Pseudo-labels are made from the existing ball tracks (clean outliers, fill short flight gaps with quadratic fits, ignore unseen rally frames, negatives in dead time); 8-frame windows are cached as a uint8 memmap and the vendored TrackNet is fine-tuned with a masked WBCE loss in fp16. `vball tunerallies` re-tunes the rally rules on the new tracks. Spec: `docs/superpowers/specs/2026-10-03-phase2a-ball-tracker-design.md`.

**Tech Stack:** Python 3.12 (uv), PyTorch 2.14 cu130, OpenCV, NumPy, FastAPI, ffmpeg, vanilla JS.

**Conventions for every task**
- Shell: Git Bash at `C:/Users/mateusz/Projects/Vball`, branch `phase2a` (create it in Task 1). Python only via `uv run`; if the web server is running, use `uv run --no-sync` (a running `vball.exe` blocks uv's re-sync).
- Write files containing backslashes with an editor tool, not shell heredocs (the shell eats one backslash).
- Every commit ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (second `-m`).
- Held-out sets, never used for training: **Kent set 2** (match 2) and **Brunel away set 3**.

---

## File structure

```
src/vball/config.py              + TRACKNET_THRESHOLD, Paths.track_video / ball_test_csv / ball_train_dir
src/vball/rallies.py             _runs -> runs (public, reused by pseudo-labels)
src/vball/ball/tracknet.py       threshold param, keeps 512x288 track.mp4
src/vball/ball/testset.py        BallTestItem, sample_test_frames, load/save_ball_test
src/vball/ball/metrics.py        BallMetrics, ball_metrics, tolerance_px
src/vball/ball/pseudo.py         clean_track, fill_gaps, pseudo_states, make_pseudo_labels
src/vball/ball/train_data.py     read_median, build_input, heatmaps, choose_windows, build_cache, WindowDataset
src/vball/ball/finetune.py       tracknet_modules, load/save_tracknet, masked_wbce, finetune, prepare_sources
src/vball/tuning.py              GRID, grid_search
src/vball/cli.py                 + track, balleval, finetune, tunerallies
src/vball/web/app.py             + ball-test endpoints, frame JPEG endpoint
src/vball/web/static/ball.html, ball.js   Ball check page; index.html gets a link
third_party/TrackNetV3/predict.py         patch 3: --threshold
tests/test_testset.py, test_ball_metrics.py, test_pseudo.py, test_train_data.py, test_finetune.py, test_tuning.py (+ additions to test_config, test_tracknet, test_cli, test_web)
docs/results/phase2a-ball.md
```

---

### Task 1: Branch and process the remaining sets (runs in the background)

- [ ] **Step 1: Create the branch**

```bash
git checkout -b phase2a
```

- [ ] **Step 2: Start batch A in the background (~85 min)**

```bash
D="C:/Users/mateusz/Downloads/M2_games"
for spec in \
  "Kent set 3|Imperial M2 vs Kent set 3/Imperial M2 vs Kent set 3 (1080p_30fps_H264-128kbit_AAC).mp4" \
  "Kent set 4|Imperial M2 vs Kent set 4/Imperial M2 vs Kent set 4 (1080p_30fps_H264-128kbit_AAC).mp4" \
  "Brunel home set 1|Imperial M2 vs Brunel M2 set 1/Imperial M2 vs Brunel M2 set 1 (1080p_30fps_H264-128kbit_AAC)_2.mp4" \
  "Brunel home set 2|Imperial M2 vs Brunel M2 set 2/Imperial M2 vs Brunel M2 set 2 (1080p_30fps_H264-128kbit_AAC).mp4" \
  "Brunel home set 3|Imperial M2 vs Brunel M2 set 3/Imperial M2 vs Brunel M2 set 3 (1080p_30fps_H264-128kbit_AAC).mp4" \
  "Brunel home set 4|Imperial M2 vs Brunel M2 set 4/Imperial M2 vs Brunel M2 set 4 (1080p_30fps_H264-128kbit_AAC).mp4"; do
  name="${spec%%|*}"; file="${spec#*|}"
  uv run --no-sync vball process "$D/$file" --name "$name" >> data/process_batch_a.log 2>&1
done
```
Run with the Bash tool's `run_in_background: true`, `timeout: 7200000`.

- [ ] **Step 3: When batch A finishes, start batch B (~70 min)**

Same loop with:
```
"Brunel away set 1|Imperial M2 vs Brunel M2 set 1/Imperial M2 vs Brunel M2 set 1 (1080p_30fps_H264-128kbit_AAC).mp4"
"Brunel away set 2|Imperial M2 vs Brunel M2 set 2/Imperial M2 vs Brunel M2 set 2 (1080p_60fps_H264-128kbit_AAC).mp4"
"Brunel away set 3|Imperial M2 vs Brunel M2 set 3/Imperial M2 vs Brunel M2 set 3 (1080p_60fps_H264-128kbit_AAC).mp4"
```
logging to `data/process_batch_b.log`.

- [ ] **Step 4: Verify**

Run: `curl -s http://127.0.0.1:8000/api/matches` (or `uv run --no-sync python -c "from vball import store; from vball.config import default_paths; [print(m['id'], m['name'], m['n_rallies'], m['fps']) for m in store.list_matches(store.connect(default_paths().db_path))]"`)
Expected: 11 matches; note the ids of "Brunel away set 3" (held out) and all training ids for Tasks 12–13.

Tasks 2–11 can proceed while the batches run (they don't need the GPU), except tests that run TrackNet.

---

### Task 2: Threshold parameter, kept 512×288 video, new paths

**Files:**
- Modify: `src/vball/config.py`, `src/vball/ball/tracknet.py`, `third_party/TrackNetV3/predict.py`, `third_party/TrackNetV3/VENDORED.md`
- Test: `tests/test_config.py`, `tests/test_tracknet.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py::test_match_file_layout` (inside the function):
```python
    assert paths.track_video(3) == tmp_path / "matches" / "3" / "track.mp4"
    assert paths.ball_test_csv(3) == tmp_path / "matches" / "3" / "ball_test.csv"
    assert paths.ball_train_dir == tmp_path / "ball_train"
```

Append to `tests/test_tracknet.py`:
```python
def test_command_passes_threshold(tmp_path):
    cmd = tracknet_command(tmp_path / "t.mp4", tmp_path / "out", tmp_path / "w.pt", threshold=0.3, python="py")
    assert cmd[cmd.index("--threshold") + 1] == "0.3"


def test_vendored_predict_has_threshold_patch():
    from vball.config import TRACKNET_DIR

    source = (TRACKNET_DIR / "predict.py").read_text()
    assert "--threshold" in source
    assert "y_pred > HEATMAP_THRESHOLD" in source
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py tests/test_tracknet.py -q`
Expected: 3 failures (`AttributeError` for the paths, `TypeError` for `threshold`, assertion for the patch).

- [ ] **Step 3: Implement config additions**

In `src/vball/config.py`, below `TRACKNET_WEIGHTS = ...` add:
```python
TRACKNET_THRESHOLD = 0.5  # heatmap value above which TrackNet reports a ball
```
and in `class Paths` add:
```python
    def track_video(self, match_id: int) -> Path:
        """512x288 copy of the work video that TrackNet reads (kept for re-tracking and training)."""
        return self.match_dir(match_id) / "track.mp4"

    def ball_test_csv(self, match_id: int) -> Path:
        """Hand-clicked ball positions used to score the ball tracker."""
        return self.match_dir(match_id) / "ball_test.csv"

    @property
    def ball_train_dir(self) -> Path:
        return self.data_dir / "ball_train"
```

- [ ] **Step 4: Patch vendored predict.py (patch 3)**

In `third_party/TrackNetV3/predict.py`:
1. After the line `print(f"Using PyTorch device: {device}")` add:
```python

# vball patch 3: configurable heatmap threshold (upstream hard-codes 0.5)
HEATMAP_THRESHOLD = 0.5
```
2. In `def predict(...)`, change `y_pred = y_pred > 0.5` to `y_pred = y_pred > HEATMAP_THRESHOLD`.
3. After `parser.add_argument('--traj_len', ...)` add:
```python
    parser.add_argument('--threshold', type=float, default=0.5, help='heatmap threshold for a ball detection (vball patch 3)')
```
4. After `args = parser.parse_args()` add:
```python
    HEATMAP_THRESHOLD = args.threshold
```

Append to `third_party/TrackNetV3/VENDORED.md` under "Local patches":
```
3. predict.py: `--threshold` argument (module-level `HEATMAP_THRESHOLD`, default 0.5) replaces the hard-coded heatmap threshold in `predict()`.
```

- [ ] **Step 5: Update the runner**

Replace `tracknet_command` and `run_tracknet` in `src/vball/ball/tracknet.py` with:
```python
def tracknet_command(
    video: Path,
    out_dir: Path,
    weights: Path,
    batch_size: int = 4,
    threshold: float = TRACKNET_THRESHOLD,
    python: str = sys.executable,
) -> list[str]:
    return [
        python, "predict.py",
        "--video_file", video.as_posix(),
        "--tracknet_file", weights.as_posix(),
        "--save_dir", out_dir.as_posix(),
        "--eval_mode", "nonoverlap",
        "--large_video",
        "--batch_size", str(batch_size),
        "--threshold", str(threshold),
    ]
```
```python
def run_tracknet(
    video: Path,
    out_csv: Path,
    weights: Path = TRACKNET_WEIGHTS,
    batch_size: int = 4,
    threshold: float = TRACKNET_THRESHOLD,
    small_video: Path | None = None,
) -> Path:
    """Run vendored TrackNetV3 on video; write ball positions in video pixels to out_csv.

    The 512x288 copy TrackNet reads is kept at small_video (default: track.mp4 next to out_csv) and reused."""
    if not weights.exists():
        raise FileNotFoundError(f"TrackNet weights not found at {weights}; run scripts/download_models.py")
    small = small_video or out_csv.parent / "track.mp4"
    if not small.exists():
        small.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(downscale_command(video, small), check=True)
    tmp_dir = (out_csv.parent / "tracknet_tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        tracknet_command(small.resolve(), tmp_dir, weights.resolve(), batch_size, threshold),
        cwd=TRACKNET_DIR,
        check=True,
    )
    info = probe(video)
    rescale_tracknet_csv(
        tracknet_csv_path(tmp_dir, small), out_csv, info.width / TRACKNET_W, info.height / TRACKNET_H
    )
    shutil.rmtree(tmp_dir)
    return out_csv
```
and change the import to `from vball.config import TRACKNET_DIR, TRACKNET_THRESHOLD, TRACKNET_WEIGHTS`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/vball/config.py src/vball/ball/tracknet.py third_party/TrackNetV3/predict.py third_party/TrackNetV3/VENDORED.md tests/test_config.py tests/test_tracknet.py
git commit -m "feat: TrackNet threshold parameter and reusable 512x288 track video" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `vball track` command

**Files:**
- Modify: `src/vball/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:
```python
def test_track_to_custom_out_keeps_rallies(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    main(["redetect", str(match_id)])
    calls = []

    def fake_run_tracknet(video, out_csv, weights, threshold, small_video):
        calls.append({"threshold": threshold, "small_video": small_video})
        write_tracknet_csv(out_csv, make_track(1800))

    monkeypatch.setattr("vball.cli.run_tracknet", fake_run_tracknet)
    out = tmp_path / "exp.csv"
    assert main(["track", str(match_id), "--threshold", "0.3", "--out", str(out)]) == 0
    assert out.exists()
    assert calls[0]["threshold"] == 0.3
    assert calls[0]["small_video"] == default_paths().track_video(match_id)
    conn = store.connect(default_paths().db_path)
    assert len(store.get_rallies(conn, match_id)) == 2  # untouched


def test_track_default_replaces_ball_csv_and_redetects(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "vball.cli.run_tracknet",
        lambda video, out_csv, weights, threshold, small_video: write_tracknet_csv(out_csv, make_track(1800)),
    )
    assert main(["track", str(match_id)]) == 0
    assert "0 rallies" in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -q`
Expected: 2 failures (`invalid choice: 'track'`).

- [ ] **Step 3: Implement**

In `src/vball/cli.py` add imports:
```python
from vball.ball.tracknet import run_tracknet
from vball.config import TRACKNET_THRESHOLD, TRACKNET_WEIGHTS, default_paths
```
(replace the existing `from vball.config import default_paths`). In `build_parser()` before the `serve` parser add:
```python
    t = sub.add_parser("track", help="re-run ball tracking on a processed match")
    t.add_argument("match_id", type=int)
    t.add_argument("--weights", type=Path, default=TRACKNET_WEIGHTS)
    t.add_argument("--threshold", type=float, default=TRACKNET_THRESHOLD)
    t.add_argument("--out", type=Path, help="write here instead of the match's ball.csv (rallies are left alone)")
```
In `main()`, inside the `try:` after the `redetect` branch add:
```python
        elif args.command == "track":
            out = args.out or paths.ball_csv(args.match_id)
            run_tracknet(
                paths.work_video(args.match_id),
                out,
                weights=args.weights,
                threshold=args.threshold,
                small_video=paths.track_video(args.match_id),
            )
            print(f"wrote {out}")
            if args.out is None:
                print(f"{len(pipeline.redetect(conn, paths, args.match_id))} rallies")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vball/cli.py tests/test_cli.py
git commit -m "feat: vball track re-runs ball tracking with chosen weights/threshold" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Ball test-set storage and sampling

**Files:**
- Create: `src/vball/ball/testset.py`
- Test: `tests/test_testset.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_testset.py`:
```python
from vball.ball.testset import BallTestItem, load_ball_test, sample_test_frames, save_ball_test


def test_samples_only_rally_frames_sorted_unique():
    frames = sample_test_frames([(1.0, 2.0), (10.0, 11.0)], fps=30.0, n=20, seed=1)
    assert len(frames) == 20 == len(set(frames))
    assert frames == sorted(frames)
    assert all(30 <= f < 60 or 300 <= f < 330 for f in frames)


def test_sampling_is_reproducible_and_capped():
    assert sample_test_frames([(0, 10)], 30.0, 5, seed=3) == sample_test_frames([(0, 10)], 30.0, 5, seed=3)
    assert sample_test_frames([(0.0, 0.2)], 30.0, 100, seed=3) == [0, 1, 2, 3, 4, 5]


def test_round_trip(tmp_path):
    items = [BallTestItem(30, "ball", 100.5, 200.0), BallTestItem(10), BallTestItem(20, "none")]
    path = tmp_path / "x" / "ball_test.csv"
    save_ball_test(path, items)
    assert load_ball_test(path) == sorted(items, key=lambda i: i.frame)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_testset.py -q`
Expected: `ModuleNotFoundError: No module named 'vball.ball.testset'`

- [ ] **Step 3: Implement**

`src/vball/ball/testset.py`:
```python
"""Hand-clicked ball positions on sampled rally frames, used to score the ball tracker."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATUSES = ("todo", "ball", "none", "skip")  # skip = can't tell


@dataclass(frozen=True)
class BallTestItem:
    frame: int
    status: str = "todo"
    x: float = 0.0  # work-video pixels, only meaningful when status == "ball"
    y: float = 0.0


def sample_test_frames(intervals_s: list[tuple[float, float]], fps: float, n: int, seed: int) -> list[int]:
    """Up to n distinct frames drawn uniformly from the given time intervals."""
    if not intervals_s:
        return []
    frames = np.unique(np.concatenate([np.arange(int(s * fps), int(e * fps)) for s, e in intervals_s]))
    if len(frames) > n:
        frames = np.random.default_rng(seed).choice(frames, size=n, replace=False)
    return sorted(int(f) for f in frames)


def load_ball_test(path: Path) -> list[BallTestItem]:
    with open(path, newline="") as f:
        return [
            BallTestItem(int(r["frame"]), r["status"], float(r["x"]), float(r["y"])) for r in csv.DictReader(f)
        ]


def save_ball_test(path: Path, items: list[BallTestItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["frame", "status", "x", "y"])
        for item in sorted(items, key=lambda i: i.frame):
            writer.writerow([item.frame, item.status, round(item.x, 1), round(item.y, 1)])
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_testset.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/vball/ball/testset.py tests/test_testset.py
git commit -m "feat: ball test-set sampling and storage" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Ball metrics and `vball balleval`

**Files:**
- Create: `src/vball/ball/metrics.py`
- Modify: `src/vball/cli.py`
- Test: `tests/test_ball_metrics.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_ball_metrics.py`:
```python
import numpy as np

from vball.ball.metrics import ball_metrics, tolerance_px
from vball.ball.testset import BallTestItem
from vball.ball.track import BallTrack


def track_with(points: dict[int, tuple[float, float]], n: int = 10) -> BallTrack:
    visible = np.zeros(n, dtype=bool)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    for f, (px, py) in points.items():
        visible[f], x[f], y[f] = True, px, py
    return BallTrack(visible, x, y)


def test_tolerance_is_4px_at_tracknet_width():
    assert tolerance_px(512) == 4.0
    assert tolerance_px(1920) == 15.0


def test_counts_each_case():
    track = track_with({0: (100, 100), 1: (300, 300), 3: (50, 50)})
    items = [
        BallTestItem(0, "ball", 105, 100),  # TP (5 px away)
        BallTestItem(1, "ball", 100, 100),  # FP: detected somewhere else
        BallTestItem(2, "ball", 10, 10),  # FN: not detected
        BallTestItem(3, "none"),  # FP: detection where there is no ball
        BallTestItem(4, "none"),  # TN
        BallTestItem(5, "skip"),  # ignored
        BallTestItem(6, "todo"),  # ignored
    ]
    m = ball_metrics(track, items, tol_px=15.0)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 2, 1, 1)
    assert m.precision == 1 / 3 and m.recall == 0.5 and m.accuracy == 2 / 5
    assert "recall 0.50" in m.summary()


def test_empty_metrics_do_not_divide_by_zero():
    m = ball_metrics(track_with({}), [], tol_px=15.0)
    assert (m.precision, m.recall, m.accuracy) == (0.0, 0.0, 0.0)
```

Append to `tests/test_cli.py`:
```python
def test_balleval_scores_against_clicked_frames(tmp_path, monkeypatch, capsys):
    from vball.ball.testset import BallTestItem, save_ball_test

    match_id = seed(tmp_path, monkeypatch)  # ball flies 10-18 s and 30-36 s at y=360
    save_ball_test(
        default_paths().ball_test_csv(match_id),
        [BallTestItem(10 * 30 + 1, "ball", 60.0, 360.0), BallTestItem(100, "none"), BallTestItem(101)],
    )
    assert main(["balleval", str(match_id)]) == 0
    out = capsys.readouterr().out
    assert "TP 1 FP 0 FN 0 TN 1" in out
    assert "2 of 3 test frames labelled" in out


def test_balleval_without_test_frames_explains(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    assert main(["balleval", str(match_id)]) == 1
    assert "Ball check" in capsys.readouterr().err
```
(Frame 301 of `make_track`'s flight starting at 10 s: x = 40 + 600·(1/30) = 60, y = 360.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ball_metrics.py tests/test_cli.py -q`
Expected: `ModuleNotFoundError: No module named 'vball.ball.metrics'` and `invalid choice: 'balleval'`.

- [ ] **Step 3: Implement metrics**

`src/vball/ball/metrics.py`:
```python
"""Score a ball track against hand-clicked test frames (TrackNet's TP / FP / FN / TN convention)."""

import math
from dataclasses import dataclass

from vball.ball.testset import BallTestItem
from vball.ball.track import BallTrack


def tolerance_px(width: int) -> float:
    """TrackNet's 4 px tolerance at 512 px width, scaled to the work video."""
    return 4.0 * width / 512


@dataclass(frozen=True)
class BallMetrics:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n else 0.0

    def summary(self) -> str:
        return (
            f"ball: TP {self.tp} FP {self.fp} FN {self.fn} TN {self.tn} | "
            f"precision {self.precision:.2f} recall {self.recall:.2f} accuracy {self.accuracy:.2f}"
        )


def ball_metrics(track: BallTrack, items: list[BallTestItem], tol_px: float) -> BallMetrics:
    tp = fp = fn = tn = 0
    for item in items:
        if item.status not in ("ball", "none") or not 0 <= item.frame < len(track):
            continue
        detected = bool(track.visible[item.frame])
        if item.status == "ball":
            if not detected:
                fn += 1
            elif math.hypot(track.x[item.frame] - item.x, track.y[item.frame] - item.y) <= tol_px:
                tp += 1
            else:
                fp += 1  # detected, but somewhere else
        elif detected:
            fp += 1
        else:
            tn += 1
    return BallMetrics(tp, fp, fn, tn)
```

- [ ] **Step 4: Add `balleval` to the CLI**

Imports in `src/vball/cli.py`:
```python
from vball.ball.metrics import ball_metrics, tolerance_px
from vball.ball.testset import load_ball_test
```
Parser (before `serve`):
```python
    b = sub.add_parser("balleval", help="score a ball track against the frames clicked on the Ball check page")
    b.add_argument("match_id", type=int)
    b.add_argument("--pred", type=Path, help="ball CSV to score (default: the match's ball.csv)")
```
Branch in `main()` (inside the `try:`):
```python
        elif args.command == "balleval":
            test_path = paths.ball_test_csv(args.match_id)
            if not test_path.exists():
                print("no ball test frames yet; click them on the Ball check page (/ball.html)", file=sys.stderr)
                return 1
            items = load_ball_test(test_path)
            track = load_tracknet_csv(args.pred or paths.ball_csv(args.match_id), match["n_frames"])
            print(ball_metrics(track, items, tolerance_px(match["width"])).summary())
            done = sum(item.status in ("ball", "none") for item in items)
            print(f"{done} of {len(items)} test frames labelled")
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/vball/ball/metrics.py src/vball/cli.py tests/test_ball_metrics.py tests/test_cli.py
git commit -m "feat: ball precision/recall against clicked frames (vball balleval)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Web API for the Ball check page

**Files:**
- Modify: `src/vball/web/app.py`
- Test: `tests/test_web.py`

- [ ] **Step 1: Write the failing tests**

Add to the imports at the top of `tests/test_web.py`:
```python
import cv2
import numpy as np
from helpers import make_test_video, requires_ffmpeg
```
and append:
```python
def test_ball_test_is_sampled_once_from_rallies(tmp_path):
    client, match_id = make_client(tmp_path)  # rallies at frames 30-300 and 450-600
    first = client.get(f"/api/matches/{match_id}/ball-test?n=10").json()
    assert len(first) == 10
    assert all(i["status"] == "todo" for i in first)
    assert all(30 <= i["frame"] < 300 or 450 <= i["frame"] < 600 for i in first)
    assert client.get(f"/api/matches/{match_id}/ball-test?n=50").json() == first  # persisted, not resampled


def test_ball_test_update(tmp_path):
    client, match_id = make_client(tmp_path)
    frame = client.get(f"/api/matches/{match_id}/ball-test?n=5").json()[0]["frame"]
    res = client.put(f"/api/matches/{match_id}/ball-test/{frame}", json={"status": "ball", "x": 12.5, "y": 30.0})
    assert res.json() == {"frame": frame, "status": "ball", "x": 12.5, "y": 30.0}
    assert client.get(f"/api/matches/{match_id}/ball-test").json()[0]["status"] == "ball"
    assert client.put(f"/api/matches/{match_id}/ball-test/99999", json={"status": "none"}).status_code == 404
    assert client.put(f"/api/matches/{match_id}/ball-test/{frame}", json={"status": "maybe"}).status_code == 422


@requires_ffmpeg
def test_frame_jpeg(tmp_path):
    client, match_id = make_client(tmp_path)  # work video is a placeholder; replace with a real one
    video = tmp_path / "data" / "matches" / str(match_id) / "work.mp4"
    make_test_video(video, seconds=2, fps=25)
    res = client.get(f"/api/matches/{match_id}/frames/10.jpg")
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    image = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (240, 320, 3)
    assert client.get(f"/api/matches/{match_id}/frames/100000.jpg").status_code == 404
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_web.py -q`
Expected: 3 failures (404 for the new routes).

- [ ] **Step 3: Implement**

In `src/vball/web/app.py` add imports:
```python
from typing import Literal

import cv2
from fastapi import Response

from vball.ball.testset import BallTestItem, load_ball_test, sample_test_frames, save_ball_test
```
Add a body model next to `LabelBody`:
```python
class BallTestBody(BaseModel):
    status: Literal["todo", "ball", "none", "skip"]
    x: float = 0.0
    y: float = 0.0
```
Change `require_match` to return the match:
```python
    def require_match(conn: sqlite3.Connection, match_id: int) -> dict:
        match = store.get_match(conn, match_id)
        if match is None:
            raise HTTPException(status_code=404, detail="match not found")
        return match
```
Add routes before the `app.mount(...)` lines:
```python
    @app.get("/api/matches/{match_id}/ball-test")
    def get_ball_test(match_id: int, n: int = 100, conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        match = require_match(conn, match_id)
        path = paths.ball_test_csv(match_id)
        if not path.exists():
            gt = paths.gt_csv(match_id)
            if gt.exists():
                intervals = [(label.start_s, label.end_s) for label in load_labels(gt)]
            else:
                intervals = [(r["start_s"], r["end_s"]) for r in store.get_rallies(conn, match_id)]
            frames = sample_test_frames(intervals, match["fps"], n, seed=match_id)
            save_ball_test(path, [BallTestItem(f) for f in frames])
        return [asdict(item) for item in load_ball_test(path)]

    @app.put("/api/matches/{match_id}/ball-test/{frame}")
    def put_ball_test(
        match_id: int, frame: int, body: BallTestBody, conn: sqlite3.Connection = Depends(db)
    ) -> dict:
        require_match(conn, match_id)
        path = paths.ball_test_csv(match_id)
        items = load_ball_test(path) if path.exists() else []
        index = next((i for i, item in enumerate(items) if item.frame == frame), None)
        if index is None:
            raise HTTPException(status_code=404, detail="frame is not in the test set")
        items[index] = BallTestItem(frame, body.status, body.x, body.y)
        save_ball_test(path, items)
        return asdict(items[index])

    @app.get("/api/matches/{match_id}/frames/{frame}.jpg")
    def get_frame(match_id: int, frame: int, conn: sqlite3.Connection = Depends(db)) -> Response:
        match = require_match(conn, match_id)
        if not 0 <= frame < match["n_frames"]:
            raise HTTPException(status_code=404, detail="frame out of range")
        cap = cv2.VideoCapture(str(paths.work_video(match_id)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(status_code=404, detail="frame could not be decoded")
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vball/web/app.py tests/test_web.py
git commit -m "feat: ball test-set API and frame JPEG endpoint" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Ball check page

**Files:**
- Create: `src/vball/web/static/ball.html`, `src/vball/web/static/ball.js`
- Modify: `src/vball/web/static/index.html` (link), `src/vball/web/static/style.css`
- Test: `tests/test_web.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web.py`:
```python
def test_ball_check_page_is_served_and_linked(tmp_path):
    client, _ = make_client(tmp_path)
    assert "<title>vball · Ball check</title>" in client.get("/ball.html").text
    assert client.get("/ball.js").status_code == 200
    assert 'href="/ball.html"' in client.get("/").text
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_web.py -q`
Expected: 1 failure (404).

- [ ] **Step 3: Write `ball.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>vball · Ball check</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <header>
    <h1><a href="/">vball</a> · Ball check</h1>
    <select id="match" aria-label="Match"></select>
    <span id="progress" class="badge"></span>
  </header>
  <main class="ballcheck">
    <section>
      <div class="frame-wrap">
        <img id="frame" alt="Video frame to label">
        <div id="marker" class="marker" hidden></div>
      </div>
      <div id="status" class="status idle" role="status" aria-live="polite">
        <span id="state" class="state"></span>
        <span id="message" class="message"></span>
      </div>
      <div class="controls">
        <button id="btn-prev">◀ Previous <kbd>←</kbd></button>
        <button id="btn-next">Next ▶ <kbd>→</kbd></button>
        <span class="sep"></span>
        <button id="btn-none" class="primary">No ball visible <kbd>X</kbd></button>
        <button id="btn-skip">Can't tell <kbd>K</kbd></button>
        <span class="sep"></span>
        <button id="btn-peek-back">Peek −2 frames <kbd>Q</kbd></button>
        <button id="btn-peek-fwd">Peek +2 frames <kbd>W</kbd></button>
      </div>
      <p class="help">Click the centre of the ball. Peeking shows nearby frames to help find a blurred ball; clicks always label the target frame, so return to it (press Q/W back to 0) first. The tracker's own guess is never shown, so the test stays honest.</p>
    </section>
  </main>
  <script src="/ball.js"></script>
</body>
</html>
```

- [ ] **Step 4: Write `ball.js`**

```javascript
const $ = (sel) => document.querySelector(sel);
const img = $("#frame");
const marker = $("#marker");
let matchId = null;
let items = []; // [{frame, status, x, y}]
let idx = 0;
let peek = 0; // frames away from the target frame being shown
let message = { text: "", kind: "", until: 0 };

async function getJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

function say(text, kind) {
  message = { text, kind, until: Date.now() + 4000 };
  render();
}

async function loadMatches() {
  const matches = await getJson("/api/matches");
  const sel = $("#match");
  sel.innerHTML = "";
  for (const m of matches) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `#${m.id} ${m.name}`;
    sel.appendChild(opt);
  }
  sel.onchange = () => selectMatch(Number(sel.value));
  if (matches.length) await selectMatch(matches[0].id);
}

async function selectMatch(id) {
  matchId = id;
  $("#match").value = String(id);
  items = await getJson(`/api/matches/${id}/ball-test`);
  const todo = items.findIndex((i) => i.status === "todo");
  show(todo >= 0 ? todo : 0);
}

function show(i) {
  if (!items.length) return;
  idx = Math.max(0, Math.min(items.length - 1, i));
  peek = 0;
  loadImage();
}

function loadImage() {
  img.src = `/api/matches/${matchId}/frames/${items[idx].frame + peek}.jpg`;
  render();
}

function render() {
  const done = items.filter((i) => i.status === "ball" || i.status === "none").length;
  const skipped = items.filter((i) => i.status === "skip").length;
  $("#progress").textContent = `${done} labelled · ${skipped} skipped · ${items.length - done - skipped} to do`;
  const item = items[idx];
  if (!item) return;
  const what = { todo: "not labelled yet", ball: `ball at (${Math.round(item.x)}, ${Math.round(item.y)})`, none: "no ball visible", skip: "skipped (can't tell)" }[item.status];
  const peeking = peek ? ` · PEEKING ${peek > 0 ? "+" : ""}${peek} frames (Q/W to go back)` : "";
  $("#state").textContent = `Frame ${idx + 1} of ${items.length} (video frame ${item.frame}): ${what}${peeking}`;
  $("#status").className = `status ${peek ? "rec" : "idle"}`;
  const showMsg = message.text && Date.now() < message.until;
  $("#message").textContent = showMsg ? message.text : "";
  $("#message").className = `message ${showMsg ? message.kind : ""}`;
  placeMarker();
}

function placeMarker() {
  const item = items[idx];
  if (!item || item.status !== "ball" || peek || !img.naturalWidth) {
    marker.hidden = true;
    return;
  }
  marker.hidden = false;
  marker.style.left = `${(item.x / img.naturalWidth) * img.clientWidth}px`;
  marker.style.top = `${(item.y / img.naturalHeight) * img.clientHeight}px`;
}

async function save(status, x = 0, y = 0) {
  const item = items[idx];
  try {
    items[idx] = await getJson(`/api/matches/${matchId}/ball-test/${item.frame}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, x, y }),
    });
  } catch (err) {
    say(`Not saved (${err.message})`, "warn");
    return;
  }
  const next = items.findIndex((i, j) => j > idx && i.status === "todo");
  const any = next >= 0 ? next : items.findIndex((i) => i.status === "todo");
  if (any < 0) {
    say("All test frames are labelled. Thank you!", "ok");
    render();
    return;
  }
  say(`Saved frame ${idx + 1}`, "ok");
  show(any);
}

img.addEventListener("click", (e) => {
  if (peek) {
    say("You're peeking at another frame. Press Q/W to return to the target frame, then click.", "warn");
    return;
  }
  const rect = img.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * img.naturalWidth;
  const y = ((e.clientY - rect.top) / rect.height) * img.naturalHeight;
  save("ball", x, y);
});
img.addEventListener("load", render);
window.addEventListener("resize", placeMarker);

function peekBy(delta) {
  peek += delta;
  loadImage();
}

function bind(id, fn) {
  $(id).onclick = (e) => {
    e.currentTarget.blur();
    fn();
  };
}
bind("#btn-prev", () => show(idx - 1));
bind("#btn-next", () => show(idx + 1));
bind("#btn-none", () => save("none"));
bind("#btn-skip", () => save("skip"));
bind("#btn-peek-back", () => peekBy(-2));
bind("#btn-peek-fwd", () => peekBy(2));

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLSelectElement || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  const actions = {
    arrowleft: () => show(idx - 1),
    arrowright: () => show(idx + 1),
    x: () => save("none"),
    k: () => save("skip"),
    q: () => peekBy(-2),
    w: () => peekBy(2),
  };
  const action = actions[e.key.toLowerCase()];
  if (!action) return;
  e.preventDefault();
  action();
});

setInterval(render, 500); // lets messages expire
loadMatches().catch((err) => say(String(err), "warn"));
```

- [ ] **Step 5: Styles and link**

Append to `src/vball/web/static/style.css`:
```css
h1 a { color: inherit; text-decoration: none; }
.ballcheck { grid-template-columns: 1fr; }
.frame-wrap { position: relative; display: inline-block; max-width: 100%; }
.frame-wrap img { display: block; max-width: 100%; max-height: calc(100vh - 230px); cursor: crosshair; }
.marker { position: absolute; width: 22px; height: 22px; margin: -11px 0 0 -11px; border: 2px solid var(--label); border-radius: 50%; pointer-events: none; box-shadow: 0 0 0 2px #0008; }
```
In `src/vball/web/static/index.html`, in `<header>` after the speed `<label>` add:
```html
    <a class="header-link" href="/ball.html" title="Click the ball in sampled frames to score the ball tracker">Ball check →</a>
```
and append to `style.css`:
```css
.header-link { margin-left: auto; color: var(--accent); }
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 7: Check in the browser (on a copy of the data)**

Copy `data/vball.db` and match 2 (hard-link `work.mp4`) into `data/uitest`, run `VBALL_DATA=data/uitest uv run vball serve --port 8766`, open `http://127.0.0.1:8766/ball.html`, select match 2 and verify: a frame shows; clicking marks a green circle and advances; X / K advance; Q/W change the frame and show "PEEKING"; clicking while peeking warns; progress counts update; reload keeps labels. Delete `data/uitest` afterwards.

- [ ] **Step 8: Commit**

```bash
git add src/vball/web/static tests/test_web.py
git commit -m "feat: Ball check page for clicking ball test frames" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Pseudo-labels

**Files:**
- Modify: `src/vball/rallies.py` (rename `_runs` → `runs`)
- Create: `src/vball/ball/pseudo.py`
- Test: `tests/test_pseudo.py`

- [ ] **Step 1: Make `runs` public**

In `src/vball/rallies.py` rename `def _runs(` to `def runs(` and its one call `_runs(activity ...)` to `runs(activity ...)`. Run `uv run pytest tests/test_rallies.py -q` → all pass.

- [ ] **Step 2: Write the failing tests**

`tests/test_pseudo.py`:
```python
import numpy as np

from vball.ball.pseudo import IGNORED, NEGATIVE, POSITIVE, clean_track, fill_gaps, pseudo_states
from vball.ball.track import BallTrack


def parabola_track(n=60):
    t = np.arange(n, dtype=float)
    x = 100 + 5 * t
    y = 400 - 12 * t + 0.25 * t**2
    return BallTrack(np.ones(n, dtype=bool), x, y)


def with_hidden(track, frames):
    visible = track.visible.copy()
    visible[list(frames)] = False
    return BallTrack(visible, np.where(visible, track.x, np.nan), np.where(visible, track.y, np.nan))


def test_short_gap_in_flight_is_filled_on_the_curve():
    truth = parabola_track()
    filled, count = fill_gaps(with_hidden(truth, range(20, 25)), max_gap=6, reach=12, max_rms_px=1.0)
    assert count == 5
    assert filled.visible[20:25].all()
    assert np.allclose(filled.x[20:25], truth.x[20:25]) and np.allclose(filled.y[20:25], truth.y[20:25])


def test_long_gap_is_not_filled():
    _, count = fill_gaps(with_hidden(parabola_track(), range(20, 30)), max_gap=6, reach=12, max_rms_px=1.0)
    assert count == 0


def test_gap_across_a_touch_is_not_filled():
    # a parabola fits a V shape surprisingly well (RMS < 2 px), so the residual alone can't catch touches
    track = parabola_track()
    x = track.x.copy()
    x[23:] = x[23] - 5 * np.arange(len(x) - 23)  # horizontal direction reverses at frame 23
    hidden = with_hidden(BallTrack(track.visible, x, track.y), range(21, 25))
    _, count = fill_gaps(hidden, max_gap=6, reach=12, max_rms_px=100.0)
    assert count == 0  # rejected by the horizontal-velocity check


def test_gap_where_ball_curves_upward_is_not_filled():
    t = np.arange(60, dtype=float)
    y = np.where(t < 23, 300 + 6 * t, 300 + 6 * 23 - 6 * (t - 23))  # falls, then is dug back up
    track = BallTrack(np.ones(60, dtype=bool), 100 + 5 * t, y)
    _, count = fill_gaps(with_hidden(track, range(21, 25)), max_gap=6, reach=12, max_rms_px=100.0)
    assert count == 0  # rejected: gravity can only bend the path downward (image y grows)


def test_gap_at_edge_or_with_too_few_points_is_not_filled():
    hidden = with_hidden(parabola_track(20), list(range(0, 3)) + list(range(5, 8)) + list(range(9, 20)))
    _, count = fill_gaps(hidden, max_gap=6, reach=12, max_rms_px=1.0)
    assert count == 0


def test_clean_removes_isolated_and_jumping_detections():
    track = parabola_track(30)
    visible = np.zeros(30, dtype=bool)
    visible[5:20] = True
    visible[27] = True  # isolated
    x = track.x.copy()
    x[12] += 500  # one-frame jump far off the path
    cleaned = clean_track(BallTrack(visible, np.where(visible, x, np.nan), np.where(visible, track.y, np.nan)), max_step_px=50)
    assert not cleaned.visible[27]
    assert not cleaned.visible[12]
    assert cleaned.visible[5:12].all() and cleaned.visible[13:20].all()


def test_states_positive_ignored_negative():
    visible = np.zeros(100, dtype=bool)
    visible[[20, 21, 70]] = True
    track = BallTrack(visible, np.where(visible, 1.0, np.nan), np.where(visible, 1.0, np.nan))
    state = pseudo_states(track, rallies=[(15, 30)], neg_margin=10)
    assert state[20] == state[21] == POSITIVE
    assert state[25] == IGNORED  # inside rally, not seen
    assert state[10] == IGNORED  # within the margin before the rally
    assert state[2] == NEGATIVE and state[50] == NEGATIVE
    assert state[70] == NEGATIVE  # detections in dead time are trained away
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_pseudo.py -q`
Expected: `ModuleNotFoundError: No module named 'vball.ball.pseudo'`

- [ ] **Step 4: Implement**

`src/vball/ball/pseudo.py`:
```python
"""Automatic ball labels (pseudo-labels) for fine-tuning TrackNet from its own tracks."""

import math

import numpy as np

from vball.ball.track import BallTrack
from vball.rallies import RallyParams, runs

POSITIVE, NEGATIVE, IGNORED = 1, 0, -1


def _with_visible(track: BallTrack, visible: np.ndarray, x=None, y=None) -> BallTrack:
    x = track.x if x is None else x
    y = track.y if y is None else y
    return BallTrack(visible, np.where(visible, x, np.nan), np.where(visible, y, np.nan))


def clean_track(track: BallTrack, max_step_px: float, neighbor_frames: int = 2) -> BallTrack:
    """Drop detections with no other detection within ±neighbor_frames, or that jump more than
    max_step_px per frame away from every such neighbour (false positives)."""
    visible = track.visible
    keep = visible.copy()
    n = len(track)
    for i in np.flatnonzero(visible):
        steps = [
            math.hypot(track.x[j] - track.x[i], track.y[j] - track.y[i]) / abs(j - i)
            for j in range(max(0, i - neighbor_frames), min(n, i + neighbor_frames + 1))
            if j != i and visible[j]
        ]
        if not steps or min(steps) > max_step_px:
            keep[i] = False
    return _with_visible(track, keep)


def fill_gaps(
    track: BallTrack,
    max_gap: int = 6,
    reach: int = 12,
    min_side: int = 3,
    max_rms_px: float = 3.0,
    max_dvx_px: float = 1.5,
    max_lift_px: float = 0.05,
) -> tuple[BallTrack, int]:
    """Fill short gaps from a quadratic fit through the detections around them, only where the ball is
    in free flight: small fit residual, the same horizontal speed on both sides (no horizontal force),
    and a path that does not bend upward (gravity bends it down, i.e. image y'' >= 0).

    Returns the filled track and the number of frames filled."""
    visible = track.visible.copy()
    x, y = track.x.copy(), track.y.copy()
    n = len(track)
    filled = 0
    for start, end in runs(~track.visible):
        if end - start > max_gap or start == 0 or end >= n:
            continue
        left = [j for j in range(max(0, start - reach), start) if track.visible[j]][-4:]
        right = [j for j in range(end, min(n, end + reach)) if track.visible[j]][:4]
        if len(left) < min_side or len(right) < min_side:
            continue
        t = np.array(left + right, dtype=float)
        fx = np.polyfit(t, track.x[left + right], 2)
        fy = np.polyfit(t, track.y[left + right], 2)
        residual = (np.polyval(fx, t) - track.x[left + right]) ** 2 + (np.polyval(fy, t) - track.y[left + right]) ** 2
        if math.sqrt(residual.mean()) > max_rms_px:
            continue
        vx_left = np.polyfit(left, track.x[left], 1)[0]
        vx_right = np.polyfit(right, track.x[right], 1)[0]
        if abs(vx_left - vx_right) > max_dvx_px or fy[0] < -max_lift_px:
            continue  # a touch happened inside the gap
        gap = np.arange(start, end)
        x[gap] = np.polyval(fx, gap)
        y[gap] = np.polyval(fy, gap)
        visible[gap] = True
        filled += end - start
    return _with_visible(track, visible, x, y), filled


def pseudo_states(track: BallTrack, rallies: list[tuple[int, int]], neg_margin: int) -> np.ndarray:
    """Per-frame training state: POSITIVE where the ball is known inside a rally, NEGATIVE in dead time
    further than neg_margin frames from any rally, IGNORED elsewhere (rally frames the model missed)."""
    n = len(track)
    state = np.full(n, NEGATIVE, dtype=np.int8)
    for start, end in rallies:
        state[max(0, start - neg_margin) : min(n, end + neg_margin)] = IGNORED
    for start, end in rallies:
        inside = slice(max(0, start), min(n, end))
        state[inside] = np.where(track.visible[inside], POSITIVE, IGNORED)
    return state


def make_pseudo_labels(
    track: BallTrack, rallies: list[tuple[int, int]], fps: float, width: int, height: int
) -> tuple[BallTrack, np.ndarray, dict]:
    """Clean, gap-fill and label a match's ball track. Thresholds scale with the video size."""
    diag = math.hypot(width, height)
    cleaned = clean_track(track, max_step_px=RallyParams().max_speed * diag / fps)
    scale = width / 512  # thresholds are defined at TrackNet's 512 px width
    filled, n_filled = fill_gaps(cleaned, max_rms_px=3.0 * scale, max_dvx_px=1.5 * scale, max_lift_px=0.05 * scale)
    state = pseudo_states(filled, rallies, neg_margin=round(2 * fps))
    stats = {
        "removed": int(track.visible.sum() - cleaned.visible.sum()),
        "filled": n_filled,
        "positive": int((state == POSITIVE).sum()),
        "ignored": int((state == IGNORED).sum()),
        "negative": int((state == NEGATIVE).sum()),
    }
    return filled, state, stats
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/vball/rallies.py src/vball/ball/pseudo.py tests/test_pseudo.py
git commit -m "feat: pseudo-labels from ball tracks (clean, fill flight gaps, ignore misses)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Training data (inputs identical to TrackNet inference, cached windows)

**Files:**
- Create: `src/vball/ball/train_data.py`, `src/vball/ball/finetune.py` (only `tracknet_modules` in this task)
- Test: `tests/test_train_data.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_train_data.py`:
```python
import cv2
import numpy as np
import pytest
from helpers import make_test_video, requires_ffmpeg

from vball.ball.pseudo import IGNORED, NEGATIVE, POSITIVE
from vball.ball.train_data import (
    H,
    SEQ_LEN,
    W,
    TrainSource,
    WindowDataset,
    build_cache,
    build_input,
    choose_windows,
    heatmaps,
    read_median,
)


def test_heatmaps_are_discs_only_where_present():
    xy = np.zeros((SEQ_LEN, 2))
    xy[0] = (100, 50)
    present = np.zeros(SEQ_LEN, dtype=bool)
    present[0] = True
    hm = heatmaps(xy, present)
    assert hm.shape == (SEQ_LEN, H, W)
    assert hm[0, 50, 100] == 1 and hm[0, 50, 102] == 1 and hm[0, 50, 103] == 0
    assert hm[1:].sum() == 0


def test_choose_windows_mixes_positive_and_negative():
    state = np.full(200, IGNORED, dtype=np.int8)
    state[:50] = NEGATIVE
    state[100:110] = POSITIVE
    starts = choose_windows(state, n_pos=5, n_neg=3, rng=np.random.default_rng(0))
    assert len(starts) == 8
    windows = [state[s : s + SEQ_LEN] for s in starts]
    assert sum((w == POSITIVE).any() for w in windows) == 5
    assert sum((w == NEGATIVE).all() for w in windows) == 3


@requires_ffmpeg
def test_input_matches_tracknet_inference(tmp_path):
    from vball.ball.finetune import tracknet_modules

    video = make_test_video(tmp_path / "t.mp4", seconds=1, fps=30, size=f"{W}x{H}")
    median = read_median(video)
    _, dataset = tracknet_modules()
    theirs = dataset.Video_IterableDataset(
        str(video), seq_len=SEQ_LEN, sliding_step=SEQ_LEN, bg_mode="concat", median=np.moveaxis(median, -1, 0)
    )
    _, x_theirs = next(iter(theirs))
    cap = cv2.VideoCapture(str(video))
    frames = np.stack([cap.read()[1][..., ::-1] for _ in range(SEQ_LEN)])
    assert np.allclose(build_input(frames, median), x_theirs, atol=1e-6)


@requires_ffmpeg
def test_cache_round_trip(tmp_path):
    video = make_test_video(tmp_path / "t.mp4", seconds=2, fps=30, size=f"{W}x{H}")
    n = 60
    state = np.full(n, NEGATIVE, dtype=np.int8)
    state[20:30] = POSITIVE
    x = np.where(state == POSITIVE, 200.0, np.nan)
    y = np.where(state == POSITIVE, 100.0, np.nan)
    build_cache([TrainSource(video, state, x, y)], tmp_path / "cache", n_windows=6, neg_fraction=0.5, seed=0)
    ds = WindowDataset(tmp_path / "cache")
    assert len(ds) == 6
    xb, yb, mask = ds[0]
    assert xb.shape == (3 * (SEQ_LEN + 1), H, W) and yb.shape == (SEQ_LEN, H, W) and mask.shape == (SEQ_LEN,)
    positives = [ds[i] for i in range(len(ds)) if ds[i][1].sum() > 0]
    assert positives, "at least one window has a ball heatmap"
    _, yp, _ = positives[0]
    frame_with_ball = int(np.flatnonzero(yp.numpy().sum(axis=(1, 2)))[0])
    assert yp[frame_with_ball, 100, 200] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_train_data.py -q`
Expected: `ModuleNotFoundError: No module named 'vball.ball.train_data'`

- [ ] **Step 3: Implement `tracknet_modules`**

`src/vball/ball/finetune.py` (start of the file; the rest comes in Task 10):
```python
"""Fine-tune the vendored TrackNetV3 on pseudo-labelled windows."""

import importlib
import sys
from types import ModuleType

from vball.config import TRACKNET_DIR


def tracknet_modules() -> tuple[ModuleType, ModuleType]:
    """Import the vendored TrackNetV3 `utils.general` and `dataset` modules."""
    path = str(TRACKNET_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("utils.general"), importlib.import_module("dataset")
```

- [ ] **Step 4: Implement `train_data.py`**

`src/vball/ball/train_data.py`:
```python
"""8-frame training windows for TrackNet, built exactly like its inference input and cached on disk."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset

from vball.ball.pseudo import IGNORED, NEGATIVE, POSITIVE
from vball.ball.tracknet import TRACKNET_H as H
from vball.ball.tracknet import TRACKNET_W as W

SEQ_LEN = 8
RADIUS = 2.5  # TrackNet's label disc radius at 512x288
_YY, _XX = np.mgrid[0:H, 0:W]


@dataclass
class TrainSource:
    track_video: Path  # 512x288 copy
    state: np.ndarray  # per-frame POSITIVE / NEGATIVE / IGNORED
    x: np.ndarray  # ball x at 512x288 (NaN where unknown)
    y: np.ndarray


def read_median(video: Path, max_samples: int = 1800) -> np.ndarray:
    """Median background (H, W, 3) RGB uint8, sampled the way TrackNet's inference does."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, n // max_samples)
    frames = []
    i = 0
    while cap.grab():
        if i % step == 0:
            frames.append(cap.retrieve()[1])
        i += 1
    cap.release()
    return np.ascontiguousarray(np.median(np.stack(frames), axis=0)[..., ::-1]).astype(np.uint8)


def build_input(frames_rgb: np.ndarray, median_rgb: np.ndarray) -> np.ndarray:
    """(SEQ_LEN, H, W, 3) uint8 frames + (H, W, 3) median -> (3 * (SEQ_LEN + 1), H, W) float32 in [0, 1]."""
    stack = np.concatenate([median_rgb[None], frames_rgb], axis=0)
    return stack.transpose(0, 3, 1, 2).reshape(-1, H, W).astype(np.float32) / 255.0


def heatmaps(xy: np.ndarray, present: np.ndarray) -> np.ndarray:
    out = np.zeros((SEQ_LEN, H, W), dtype=np.float32)
    for i in np.flatnonzero(present):
        out[i] = ((_XX - xy[i, 0]) ** 2 + (_YY - xy[i, 1]) ** 2 <= RADIUS**2).astype(np.float32)
    return out


def choose_windows(state: np.ndarray, n_pos: int, n_neg: int, rng: np.random.Generator) -> np.ndarray:
    """Window start frames: n_pos windows with a known ball, n_neg windows entirely in dead time."""
    windows = sliding_window_view(state, SEQ_LEN)
    starts = np.arange(len(windows))
    pos = starts[(windows == POSITIVE).any(axis=1)]
    neg = starts[(windows == NEGATIVE).all(axis=1)]
    pick_pos = rng.choice(pos, size=min(n_pos, len(pos)), replace=False)
    pick_neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
    return np.sort(np.concatenate([pick_pos, pick_neg]))


def build_cache(
    sources: list[TrainSource], out_dir: Path, n_windows: int = 2000, neg_fraction: float = 0.2, seed: int = 0
) -> None:
    """Choose windows across sources (in proportion to their positive frames), decode only the frames
    they need into a uint8 memmap, and save labels next to it."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = np.array([(s.state == POSITIVE).sum() for s in sources], dtype=float)
    shares = weights / weights.sum()
    picks = []
    for src, share in zip(sources, shares):
        k = max(1, round(n_windows * share))
        picks.append(choose_windows(src.state, round(k * (1 - neg_fraction)), round(k * neg_fraction), rng))

    needed = [np.unique((starts[:, None] + np.arange(SEQ_LEN)).ravel()) for starts in picks]
    frames = np.lib.format.open_memmap(
        out_dir / "frames.npy", mode="w+", dtype=np.uint8, shape=(sum(len(f) for f in needed), H, W, 3)
    )
    windows, xy, state, source, medians = [], [], [], [], []
    offset = 0
    for s, (src, starts, frame_ids) in enumerate(zip(sources, picks, needed)):
        row_of = {int(f): offset + r for r, f in enumerate(frame_ids)}
        cap = cv2.VideoCapture(str(src.track_video))
        wanted = set(row_of)
        i = 0
        while wanted and cap.grab():
            if i in wanted:
                frames[row_of[i]] = cap.retrieve()[1][..., ::-1]
                wanted.discard(i)
            i += 1
        cap.release()
        for start in starts:
            span = np.arange(start, start + SEQ_LEN)
            windows.append([row_of[int(f)] for f in span])
            xy.append(np.stack([src.x[span], src.y[span]], axis=1))
            state.append(src.state[span])
            source.append(s)
        medians.append(read_median(src.track_video))
        offset += len(frame_ids)
    frames.flush()
    np.savez(
        out_dir / "meta.npz",
        windows=np.array(windows, dtype=np.int64),
        xy=np.nan_to_num(np.array(xy, dtype=np.float32)),
        state=np.array(state, dtype=np.int8),
        source=np.array(source, dtype=np.int64),
        medians=np.stack(medians),
        videos=np.array([str(s.track_video) for s in sources]),
    )


class WindowDataset(Dataset):
    def __init__(self, cache_dir: Path):
        self.frames = np.load(cache_dir / "frames.npy", mmap_mode="r")
        meta = np.load(cache_dir / "meta.npz")
        self.windows, self.xy, self.state = meta["windows"], meta["xy"], meta["state"]
        self.source, self.medians = meta["source"], meta["medians"]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        x = build_input(np.asarray(self.frames[self.windows[i]]), self.medians[self.source[i]])
        y = heatmaps(self.xy[i], self.state[i] == POSITIVE)
        mask = (self.state[i] != IGNORED).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_train_data.py -q`
Expected: `4 passed`. If `test_input_matches_tracknet_inference` fails, the training input differs from inference — fix `build_input` (channel order / scaling) before continuing; never relax the tolerance.

- [ ] **Step 6: Commit**

```bash
git add src/vball/ball/train_data.py src/vball/ball/finetune.py tests/test_train_data.py
git commit -m "feat: TrackNet training windows with inference-identical inputs, memmap cache" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Fine-tuning loop and `vball finetune`

**Files:**
- Modify: `src/vball/ball/finetune.py`, `src/vball/cli.py`
- Test: `tests/test_finetune.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_finetune.py`:
```python
import numpy as np
import pytest
import torch

from vball.ball.finetune import finetune, load_tracknet, masked_wbce
from vball.ball.pseudo import IGNORED, POSITIVE
from vball.ball.train_data import H, SEQ_LEN, W, WindowDataset
from vball.config import TRACKNET_WEIGHTS

requires_weights = pytest.mark.skipif(not TRACKNET_WEIGHTS.exists(), reason="TrackNet weights not downloaded")


def test_masked_wbce_ignores_masked_frames():
    y = torch.zeros(1, SEQ_LEN, 4, 4)
    good = torch.full((1, SEQ_LEN, 4, 4), 0.01)
    bad = good.clone()
    bad[0, 3] = 0.99  # confident wrong answer on frame 3
    mask_all = torch.ones(1, SEQ_LEN)
    mask_skip3 = mask_all.clone()
    mask_skip3[0, 3] = 0
    assert masked_wbce(bad, y, mask_all) > masked_wbce(good, y, mask_all)
    assert torch.isclose(masked_wbce(bad, y, mask_skip3), masked_wbce(good, y, mask_skip3))


def write_tiny_cache(cache_dir):
    rng = np.random.default_rng(0)
    cache_dir.mkdir(parents=True)
    frames = np.lib.format.open_memmap(cache_dir / "frames.npy", mode="w+", dtype=np.uint8, shape=(SEQ_LEN, H, W, 3))
    frames[:] = rng.integers(0, 255, size=(SEQ_LEN, H, W, 3), dtype=np.uint8)
    frames.flush()
    state = np.full((1, SEQ_LEN), POSITIVE, dtype=np.int8)
    state[0, 5] = IGNORED
    np.savez(
        cache_dir / "meta.npz",
        windows=np.arange(SEQ_LEN)[None],
        xy=np.full((1, SEQ_LEN, 2), 100.0, dtype=np.float32),
        state=state,
        source=np.zeros(1, dtype=np.int64),
        medians=np.zeros((1, H, W, 3), dtype=np.uint8),
        videos=np.array(["synthetic"]),
    )


@requires_weights
def test_finetune_runs_and_saves_upstream_format(tmp_path):
    write_tiny_cache(tmp_path / "cache")
    out = tmp_path / "tuned.pt"
    losses = finetune(tmp_path / "cache", TRACKNET_WEIGHTS, out, epochs=2, batch_size=1, accum=1, device="cpu", log=lambda s: None)
    assert len(losses) == 2 and all(np.isfinite(losses))
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert {"model", "param_dict"} <= set(ckpt)
    assert ckpt["param_dict"]["seq_len"] == SEQ_LEN and ckpt["param_dict"]["bg_mode"] == "concat"
    model, _ = load_tracknet(out, "cpu")  # loads like predict.py does
    assert sum(p.numel() for p in model.parameters()) > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_finetune.py -q`
Expected: `ImportError: cannot import name 'finetune'`

- [ ] **Step 3: Implement the loop**

Append to `src/vball/ball/finetune.py` (add imports at the top: `from collections.abc import Callable`, `from pathlib import Path`, `import torch`, `from torch.utils.data import DataLoader`, `from vball.ball.train_data import WindowDataset`):
```python
def load_tracknet(weights: Path, device: str) -> tuple[torch.nn.Module, dict]:
    general, _ = tracknet_modules()
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    params = ckpt["param_dict"]
    model = general.get_model("TrackNet", params["seq_len"], params["bg_mode"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, params


def save_tracknet(model: torch.nn.Module, params: dict, path: Path, epoch: int) -> None:
    """Same checkpoint layout as upstream, so predict.py loads it unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "param_dict": params, "epoch": epoch}, path)


def masked_wbce(y_pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """TrackNet's weighted BCE, averaged per frame, skipping frames where mask == 0. Computed in fp32."""
    p = y_pred.float().clamp(1e-7, 1 - 1e-7)
    loss = -(torch.square(1 - p) * y * torch.log(p) + torch.square(p) * (1 - y) * torch.log(1 - p))
    per_frame = loss.mean(dim=(2, 3))
    return (per_frame * mask).sum() / mask.sum().clamp(min=1.0)


def finetune(
    cache_dir: Path,
    init_weights: Path,
    out_path: Path,
    epochs: int = 4,
    batch_size: int = 2,
    accum: int = 4,
    lr: float = 1e-4,
    device: str | None = None,
    log: Callable[[str], None] = print,
) -> list[float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device == "cuda"
    model, params = load_tracknet(init_weights, device)
    loader = DataLoader(WindowDataset(cache_dir), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        total = 0.0
        for step, (x, y, mask) in enumerate(loader, start=1):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                y_pred = model(x)
            loss = masked_wbce(y_pred, y, mask)
            scaler.scale(loss / accum).backward()
            if step % accum == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total += loss.item()
        losses.append(total / max(1, len(loader)))
        log(f"epoch {epoch}/{epochs} loss {losses[-1]:.5f}")
        save_tracknet(model, {**params, "finetuned_from": str(init_weights)}, out_path, epoch)
    return losses
```

- [ ] **Step 4: Source preparation + CLI**

Append to `src/vball/ball/finetune.py` (imports: `import subprocess`, `import sqlite3`, `from vball import store`, `from vball.ball.pseudo import make_pseudo_labels`, `from vball.ball.track import load_tracknet_csv`, `from vball.ball.tracknet import TRACKNET_H, TRACKNET_W, downscale_command`, `from vball.ball.train_data import TrainSource`, `from vball.config import Paths`, `from vball.labels import load_labels`):
```python
def prepare_sources(
    conn: sqlite3.Connection, paths: Paths, match_ids: list[int], log: Callable[[str], None] = print
) -> list[TrainSource]:
    """Pseudo-label each match's ball track; rally intervals come from hand labels when present."""
    sources = []
    for match_id in match_ids:
        match = store.get_match(conn, match_id)
        fps, width, height = match["fps"], match["width"], match["height"]
        gt = paths.gt_csv(match_id)
        if gt.exists():
            rallies = [(round(l.start_s * fps), round(l.end_s * fps)) for l in load_labels(gt)]
        else:
            rallies = [(r["start_frame"], r["end_frame"]) for r in store.get_rallies(conn, match_id)]
        track = load_tracknet_csv(paths.ball_csv(match_id), match["n_frames"])
        filled, state, stats = make_pseudo_labels(track, rallies, fps, width, height)
        log(f"match {match_id} {match['name']}: {stats}")
        small = paths.track_video(match_id)
        if not small.exists():
            subprocess.run(downscale_command(paths.work_video(match_id), small), check=True)
        sources.append(
            TrainSource(small, state, filled.x * TRACKNET_W / width, filled.y * TRACKNET_H / height)
        )
    return sources
```

In `src/vball/cli.py` add `from vball.ball import finetune as ft` and `from vball.ball.train_data import build_cache`, a parser:
```python
    f = sub.add_parser("finetune", help="fine-tune TrackNet on pseudo-labels from processed matches")
    f.add_argument("--matches", type=int, nargs="+", required=True, help="training match ids (never the held-out ones)")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--init", type=Path, default=TRACKNET_WEIGHTS)
    f.add_argument("--windows", type=int, default=2000)
    f.add_argument("--epochs", type=int, default=4)
    f.add_argument("--batch-size", type=int, default=2)
    f.add_argument("--rebuild-cache", action="store_true")
```
and, before `conn = store.connect(...)` (it needs no single match), the branch:
```python
    if args.command == "finetune":
        conn = store.connect(paths.db_path)
        try:
            sources = ft.prepare_sources(conn, paths, args.matches)
        finally:
            conn.close()
        cache = paths.ball_train_dir
        if args.rebuild_cache or not (cache / "meta.npz").exists():
            build_cache(sources, cache, n_windows=args.windows)
        ft.finetune(cache, args.init, args.out, epochs=args.epochs, batch_size=args.batch_size)
        print(f"wrote {args.out}")
        return 0
```
(Delete `data/ball_train` or pass `--rebuild-cache` whenever the training match list changes.)

- [ ] **Step 5: Run tests**

Run: `uv run pytest -q`
Expected: all pass (the CPU fine-tune test takes ~30–60 s).

- [ ] **Step 6: Commit**

```bash
git add src/vball/ball/finetune.py src/vball/cli.py tests/test_finetune.py
git commit -m "feat: TrackNet fine-tuning with masked WBCE, fp16 and gradient accumulation (vball finetune)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: `vball tunerallies`

**Files:**
- Create: `src/vball/tuning.py`
- Modify: `src/vball/cli.py`
- Test: `tests/test_tuning.py`

- [ ] **Step 1: Write the failing test**

`tests/test_tuning.py`:
```python
from helpers import FPS, make_track

from vball.tuning import TuneCase, grid_search


def test_grid_search_ranks_by_mean_f1():
    # one rally in which the ball is lost for 3.5 s (shorter gaps merge anyway through the padding)
    track = make_track(60 * FPS, flights=[(10, 12), (15.5, 18)])
    case = TuneCase(track, FPS, 1280, 720, gt=[(9.5, 19.5)])
    results = grid_search([case], grid={"max_gap_s": [2.0, 4.0]})
    assert [r.params["max_gap_s"] for r in results] == [4.0, 2.0]  # bridging the gap wins
    assert results[0].mean_f1 == 1.0 and results[1].mean_f1 < 1.0
    assert results[0].f1s == [1.0]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tuning.py -q`
Expected: `ModuleNotFoundError: No module named 'vball.tuning'`

- [ ] **Step 3: Implement**

`src/vball/tuning.py`:
```python
"""Grid search of rally-detection parameters against hand labels."""

import itertools
from dataclasses import dataclass, replace

import numpy as np

from vball.ball.track import BallTrack
from vball.evaluate import rally_metrics
from vball.rallies import RallyParams, detect_rallies

GRID = {
    "min_speed": [0.05, 0.15],
    "window_s": [1.0, 2.0],
    "min_active_frac": [0.1, 0.15, 0.2, 0.3],
    "max_gap_s": [2.0, 3.0, 4.0, 5.0],
    "min_rally_s": [0.5, 1.0, 1.5, 2.0],
    "pre_pad_s": [0.5, 1.0],
    "post_pad_s": [0.5, 1.0, 1.5],
}


@dataclass
class TuneCase:
    track: BallTrack
    fps: float
    width: int
    height: int
    gt: list[tuple[float, float]]


@dataclass
class TuneResult:
    params: dict
    f1s: list[float]

    @property
    def mean_f1(self) -> float:
        return float(np.mean(self.f1s))


def grid_search(cases: list[TuneCase], grid: dict = GRID, base: RallyParams = RallyParams()) -> list[TuneResult]:
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        rally_params = replace(base, **params)
        f1s = []
        for c in cases:
            rallies = detect_rallies(c.track, c.fps, c.width, c.height, rally_params)
            pred = [(r.start_s(c.fps), r.end_s(c.fps)) for r in rallies]
            f1s.append(rally_metrics(pred, c.gt).f1)
        results.append(TuneResult(params, f1s))
    return sorted(results, key=lambda r: r.mean_f1, reverse=True)
```

CLI (`src/vball/cli.py`): import `from vball.tuning import TuneCase, grid_search`; parser:
```python
    g = sub.add_parser("tunerallies", help="grid-search rally parameters against hand-labelled matches")
    g.add_argument("match_ids", type=int, nargs="+")
    g.add_argument("--top", type=int, default=8)
```
branch (before the single-match section, like `finetune`):
```python
    if args.command == "tunerallies":
        conn = store.connect(paths.db_path)
        try:
            cases = []
            for match_id in args.match_ids:
                m = store.get_match(conn, match_id)
                track = load_tracknet_csv(paths.ball_csv(match_id), m["n_frames"])
                gt = [(l.start_s, l.end_s) for l in load_labels(paths.gt_csv(match_id))]
                cases.append(TuneCase(track, m["fps"], m["width"], m["height"], gt))
        finally:
            conn.close()
        current = grid_search(cases, grid={k: [v] for k, v in vars(RallyParams()).items() if k in GRID})[0]
        print(f"current defaults: mean F1 {current.mean_f1:.3f} per match {[round(f, 3) for f in current.f1s]}")
        for r in grid_search(cases)[: args.top]:
            print(f"mean F1 {r.mean_f1:.3f} per match {[round(f, 3) for f in r.f1s]} {r.params}")
        return 0
```
(add `from vball.rallies import RallyParams` and `from vball.tuning import GRID`.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vball/tuning.py src/vball/cli.py tests/test_tuning.py
git commit -m "feat: vball tunerallies grid search over hand-labelled matches" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Test frames, baseline and threshold sweep (needs the user, ~15 min)

- [ ] **Step 1: User clicks test frames**

Restart the server (`uv run vball serve --port 8000`, background, `timeout: 7200000`). Ask the user to open `http://127.0.0.1:8000/ball.html` and label all 100 frames of **Kent set 2** and of **Brunel away set 3** (click ball / X no ball / K can't tell).

- [ ] **Step 2: Baseline**

```bash
uv run --no-sync vball balleval 2
uv run --no-sync vball balleval <brunel-away-3-id>
```
Record both lines in `docs/results/phase2a-ball.md` (create it with the table from Task 13 Step 6).

- [ ] **Step 3: Threshold sweep**

For t in 0.4 0.3 0.2, for both held-out matches:
```bash
mkdir -p data/exp
uv run --no-sync vball track 2 --threshold 0.3 --out data/exp/m2_t03.csv
uv run --no-sync vball balleval 2 --pred data/exp/m2_t03.csv
```
(~3 min per run on the kept track.mp4.) Record precision/recall per threshold. If one meets the spec target (recall +15, precision ≥ baseline −3), note it as the candidate default.

- [ ] **Step 4: Commit results so far**

```bash
git add docs/results/phase2a-ball.md
git commit -m "docs: phase 2a ball baseline and threshold sweep" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Fine-tune, evaluate, decide

- [ ] **Step 1: Memory check (fp16, batch 2)**

```bash
uv run --no-sync python -c "
import torch
from vball.ball.finetune import load_tracknet
from vball.config import TRACKNET_WEIGHTS
m, _ = load_tracknet(TRACKNET_WEIGHTS, 'cuda'); m.train()
x = torch.rand(2, 27, 288, 512, device='cuda')
with torch.autocast('cuda', dtype=torch.float16):
    y = m(x)
y.float().mean().backward()
print('peak GB', round(torch.cuda.max_memory_allocated() / 1e9, 2))
"
```
Expected: below ~3.5 GB. If out of memory, use `--batch-size 1` and pass `accum=8` (edit the default in `finetune`).

- [ ] **Step 2: Fine-tune on all non-held-out matches**

```bash
uv run --no-sync vball finetune --matches 1 <kent3> <kent4> <brunel-home-1..4> <brunel-away-1> <brunel-away-2> --out models/tracknet_vball_v1.pt --rebuild-cache
```
Run in the background (`timeout: 7200000`). The log prints pseudo-label stats per match (check `removed`/`filled` look sane: filled should be a few % of positives) and the loss per epoch (should fall).

- [ ] **Step 3: Evaluate on held-out frames**

```bash
uv run --no-sync vball track 2 --weights models/tracknet_vball_v1.pt --out data/exp/m2_v1.csv
uv run --no-sync vball balleval 2 --pred data/exp/m2_v1.csv
```
and the same for Brunel away set 3; also try the best threshold from Task 12 with the new weights.

- [ ] **Step 4: Rally check**

For Kent sets 1 and 2 re-track with the best configuration into the real `ball.csv` (`vball track 1 --weights ... --threshold ...`, same for 2), then:
```bash
uv run --no-sync vball eval 1 && uv run --no-sync vball eval 2
uv run --no-sync vball tunerallies 1 2
```
If the tuned mean F1 beats the current defaults by > 0.01, update `RallyParams` defaults (and the tests that read them), then `vball redetect` all matches.

- [ ] **Step 5: Adopt or reject**

If the spec targets are met: set `TRACKNET_WEIGHTS = MODELS_DIR / "tracknet_vball_v1.pt"` and/or `TRACKNET_THRESHOLD` in `src/vball/config.py`, re-track all matches (`vball track <id>` for each), run `uv run pytest -q`. If not met: keep the defaults, record why, and list the next lever (assisted correction of the frames the auto-labels miss).

- [ ] **Step 6: Write `docs/results/phase2a-ball.md`**

```markdown
# Phase 2a results: ball tracker

Held-out test frames: Kent set 2 (n=…, ball visible in …), Brunel away set 3 (n=…, 60 fps).
Tolerance 4 px at 512 width (15 px at 1080p).

| Config | Kent 2 P | Kent 2 R | Brunel away 3 P | Brunel away 3 R |
|---|---|---|---|---|
| beach weights, threshold 0.5 (baseline) | | | | |
| beach weights, threshold 0.4 / 0.3 / 0.2 | | | | |
| vball_v1, threshold … | | | | |

Pseudo-labels: … positives, … filled by trajectory fits, … removed as outliers (from `vball finetune` log).
Training: … windows, … epochs, … min on RTX 3050 4 GB, peak … GB.
Rally F1 after re-tracking: Kent 1 …, Kent 2 … (phase 1: 0.94 / 0.89).
Decision: …
```

- [ ] **Step 7: Commit and push**

```bash
git add docs/results/phase2a-ball.md src/vball/config.py src/vball/rallies.py tests
git commit -m "feat: adopt improved ball tracker configuration (phase 2a results)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin phase2a
```
