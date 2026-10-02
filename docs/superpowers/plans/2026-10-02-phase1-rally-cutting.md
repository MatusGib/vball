# vball Phase 1 (Foundation + Rally Cutting) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process an indoor phone recording of a volleyball match into detected rallies, view them back-to-back in a local web app, export a rallies-only video, and measure rally-detection accuracy against hand labels.

**Architecture:** ffmpeg normalises the source into `work.mp4`; vendored TrackNetV3 (volleyball fine-tuned weights) writes a per-frame ball CSV; a pure-Python heuristic turns the ball track into rallies stored in SQLite; FastAPI serves a JSON API, the video (HTTP Range) and a static HTML/JS viewer that also records ground-truth labels. Spec: `docs/superpowers/specs/2026-10-02-vball-design.md`.

**Tech Stack:** Python 3.12 (uv), PyTorch CUDA 13.0 wheels, NumPy, OpenCV, ffmpeg/ffprobe, SQLite (stdlib), FastAPI + Uvicorn, pytest, plain HTML/JS.

**Conventions for every task**
- Shell: Git Bash from the repo root `C:/Users/mateusz/Projects/Vball`. Run Python only through `uv run ...`.
- Every commit message ends with the trailer line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (pass it as a second `-m`).
- Machine facts: RTX 3050 Laptop 4 GB VRAM, NVIDIA driver 596 (supports CUDA 13), Windows 11.

---

## File structure

```
pyproject.toml                    project + deps (torch from the PyTorch cu130 index)
.python-version                   3.12
.gitignore
README.md
scripts/download_models.py        fetch TrackNet volleyball weights into models/
third_party/TrackNetV3/           vendored upstream (MIT) + VENDORED.md describing 2 patches
src/vball/
  __init__.py                     version
  config.py                       Paths (data dir layout), model/vendor locations
  video.py                        VideoInfo, probe(), target_fps(), ingest_command(), ingest()
  ball/__init__.py
  ball/track.py                   BallTrack, load_tracknet_csv()
  ball/tracknet.py                tracknet_command(), tracknet_csv_path(), run_tracknet()
  rallies.py                      Rally, RallyParams, ball_speed(), detect_rallies()
  evaluate.py                     interval IoU, matching, RallyMetrics, restrict_to_span(), visible_fraction(), load_intervals_csv()
  store.py                        SQLite schema + queries
  export.py                       clip_command(), concat_list(), export_rallies()
  pipeline.py                     process_match(), redetect()
  cli.py                          `vball` command: process / redetect / export / eval / serve
  web/__init__.py
  web/app.py                      create_app()
  web/static/index.html, app.js, style.css
tests/
  helpers.py                      make_test_video(), make_track(), write_tracknet_csv(), requires_ffmpeg
  test_smoke.py, test_config.py, test_video.py, test_track.py, test_rallies.py,
  test_evaluate.py, test_store.py, test_tracknet.py, test_export.py,
  test_pipeline.py, test_web.py, test_cli.py
docs/results/phase1-rallies.md    measured results on a real match (Task 13)
```

---

### Task 1: Environment and project scaffold

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `src/vball/__init__.py`, `tests/test_smoke.py`

- [ ] **Step 1: Install ffmpeg**

Run (PowerShell or Git Bash): `winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements`
Then open a NEW terminal and run: `ffmpeg -hide_banner -version | head -1 && ffprobe -hide_banner -version | head -1 && ffmpeg -hide_banner -encoders | grep nvenc`
Expected: two version lines and a line containing `h264_nvenc`. If `h264_nvenc` is missing, use `--encoder libx264` wherever `process` is run later.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "vball"
version = "0.1.0"
description = "Personal volleyball video analysis: rally cutting, action tagging, player stats"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch>=2.9",
    "numpy>=2.0",
    "opencv-python>=4.10",
    "pandas>=2.2",
    "pillow>=10.4",
    "parse>=1.20",
    "tqdm>=4.66",
    "pycocotools>=2.0.8",
    "huggingface_hub>=0.25",
    "fastapi>=0.115",
    "uvicorn>=0.30",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "httpx>=0.27"]

[project.scripts]
vball = "vball.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vball"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `.python-version`, `.gitignore`, package init**

`.python-version`:
```
3.12
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
data/
models/
```

`src/vball/__init__.py`:
```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Write the smoke test**

`tests/test_smoke.py`:
```python
import torch

import vball


def test_package_version():
    assert vball.__version__ == "0.1.0"


def test_cuda_is_available():
    assert torch.cuda.is_available(), "PyTorch cannot see the GPU - check the cu130 wheel and NVIDIA driver"
```

- [ ] **Step 5: Create the environment and run the tests**

Run: `uv sync --extra dev && uv run pytest -v`
Expected: `2 passed`. (uv downloads Python 3.12 automatically; torch comes from the cu130 index.)
If `test_cuda_is_available` fails: run `uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"`; the version must end in `+cu130`. If it does not, delete `.venv` and `uv.lock` and re-run Step 5.

- [ ] **Step 6: Commit and push**

```bash
git add pyproject.toml uv.lock .python-version .gitignore src/vball/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold vball project with uv, torch cu130, pytest" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```
(The repo, `origin` remote and `.gitignore` already exist from the planning commit; Step 3 rewrites `.gitignore` with identical content.)

```bash
git status --short   # expected: empty
```

---

### Task 2: Paths configuration

**Files:**
- Create: `src/vball/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path

from vball.config import Paths, default_paths


def test_match_file_layout(tmp_path):
    paths = Paths(tmp_path)
    assert paths.db_path == tmp_path / "vball.db"
    assert paths.match_dir(3) == tmp_path / "matches" / "3"
    assert paths.work_video(3) == tmp_path / "matches" / "3" / "work.mp4"
    assert paths.ball_csv(3) == tmp_path / "matches" / "3" / "ball.csv"


def test_default_paths_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VBALL_DATA", str(tmp_path / "elsewhere"))
    assert default_paths().data_dir == Path(tmp_path / "elsewhere")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vball.config'`

- [ ] **Step 3: Write the implementation**

`src/vball/config.py`:
```python
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
TRACKNET_DIR = REPO_ROOT / "third_party" / "TrackNetV3"
TRACKNET_WEIGHTS = MODELS_DIR / "tracknet_volleyball.pt"


@dataclass(frozen=True)
class Paths:
    data_dir: Path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "vball.db"

    @property
    def matches_dir(self) -> Path:
        return self.data_dir / "matches"

    def match_dir(self, match_id: int) -> Path:
        return self.matches_dir / str(match_id)

    def work_video(self, match_id: int) -> Path:
        return self.match_dir(match_id) / "work.mp4"

    def ball_csv(self, match_id: int) -> Path:
        return self.match_dir(match_id) / "ball.csv"


def default_paths() -> Paths:
    return Paths(Path(os.environ.get("VBALL_DATA", REPO_ROOT / "data")))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/vball/config.py tests/test_config.py
git commit -m "feat: data directory layout and model paths" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Video probe and ingest

**Files:**
- Create: `src/vball/video.py`, `tests/helpers.py`
- Test: `tests/test_video.py`

- [ ] **Step 1: Write the test helper**

`tests/helpers.py`:
```python
import shutil
import subprocess
from pathlib import Path

import pytest

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def make_test_video(path: Path, seconds: float, fps: int = 25, size: str = "320x240") -> Path:
    """Synthetic H.264 + AAC test video with a keyframe every second."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", str(fps),
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )
    return path
```

- [ ] **Step 2: Write the failing tests**

`tests/test_video.py`:
```python
from helpers import make_test_video, requires_ffmpeg

from vball.video import VideoInfo, ingest, ingest_command, probe, target_fps


def test_target_fps_rounds_and_caps():
    assert target_fps(29.97) == 30
    assert target_fps(59.94) == 60
    assert target_fps(25.0) == 25
    assert target_fps(120.0) == 60


def test_duration_from_frames():
    assert VideoInfo(width=1920, height=1080, fps=30.0, n_frames=90).duration_s == 3.0


def test_ingest_command_normalises_fps_height_and_gop(tmp_path):
    cmd = ingest_command(tmp_path / "in.mov", tmp_path / "out.mp4", fps=30, max_height=1080, encoder="h264_nvenc")
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-vf") + 1] == "fps=30,scale=-2:'min(ih,1080)'"
    assert cmd[cmd.index("-g") + 1] == "30"
    assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc"
    assert "-cq" in cmd
    assert cmd[-1] == str(tmp_path / "out.mp4")


@requires_ffmpeg
def test_probe_reads_synthetic_video(tmp_path):
    info = probe(make_test_video(tmp_path / "v.mp4", seconds=2, fps=25))
    assert (info.width, info.height, info.fps, info.n_frames) == (320, 240, 25.0, 50)


@requires_ffmpeg
def test_ingest_converts_fps_and_downscales(tmp_path):
    src = make_test_video(tmp_path / "v.mp4", seconds=1, fps=120)
    info = ingest(src, tmp_path / "work" / "work.mp4", max_height=120, encoder="libx264")
    assert (info.width, info.height, info.fps) == (160, 120, 60.0)
    assert abs(info.n_frames - 60) <= 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_video.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vball.video'`

- [ ] **Step 4: Write the implementation**

`src/vball/video.py`:
```python
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

MAX_FPS = 60
ENCODER_ARGS = {
    "libx264": ["-preset", "veryfast", "-crf", "20"],
    "h264_nvenc": ["-preset", "p4", "-cq", "23"],
}


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    rate = stream["avg_frame_rate"] if stream["avg_frame_rate"] != "0/0" else stream["r_frame_rate"]
    fps = float(Fraction(rate))
    nb_frames = stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        n_frames = int(nb_frames)
    else:
        n_frames = round(float(data["format"]["duration"]) * fps)
    return VideoInfo(int(stream["width"]), int(stream["height"]), fps, n_frames)


def target_fps(src_fps: float) -> int:
    return min(round(src_fps), MAX_FPS)


def ingest_command(src: Path, dst: Path, fps: int, max_height: int, encoder: str) -> list[str]:
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(src),
        "-vf", f"fps={fps},scale=-2:'min(ih,{max_height})'",
        "-c:v", encoder, *ENCODER_ARGS.get(encoder, []),
        "-g", str(fps), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ]


def ingest(src: Path, dst: Path, max_height: int = 1080, encoder: str = "h264_nvenc") -> VideoInfo:
    """Transcode to constant fps, <= max_height, keyframe every second. Returns info of dst."""
    fps = target_fps(probe(src).fps)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(ingest_command(src, dst, fps, max_height, encoder), check=True)
    return probe(dst)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_video.py -v`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add src/vball/video.py tests/helpers.py tests/test_video.py
git commit -m "feat: ffprobe video info and ffmpeg ingest to constant-fps work video" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Ball track loading

**Files:**
- Create: `src/vball/ball/__init__.py` (empty), `src/vball/ball/track.py`
- Modify: `tests/helpers.py` (append track helpers)
- Test: `tests/test_track.py`

- [ ] **Step 1: Write the failing test**

`tests/test_track.py`:
```python
import numpy as np

from vball.ball.track import load_tracknet_csv


def test_load_fills_missing_and_invisible_frames(tmp_path):
    csv_path = tmp_path / "ball.csv"
    csv_path.write_text(
        "Frame,Visibility,X,Y\n"
        "0,1,100,200\n"
        "1,0,0,0\n"
        "3,1,110,190\n"
        "9,1,999,999\n"  # beyond n_frames: ignored
    )
    track = load_tracknet_csv(csv_path, n_frames=5)
    assert len(track) == 5
    assert track.visible.tolist() == [True, False, False, True, False]
    assert track.x[0] == 100 and track.y[3] == 190
    assert np.isnan(track.x[1]) and np.isnan(track.x[2]) and np.isnan(track.y[4])


def test_load_header_only_file(tmp_path):
    csv_path = tmp_path / "ball.csv"
    csv_path.write_text("Frame,Visibility,X,Y\n")
    track = load_tracknet_csv(csv_path, n_frames=3)
    assert not track.visible.any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_track.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vball.ball'`

- [ ] **Step 3: Write the implementation**

Create empty `src/vball/ball/__init__.py`.

`src/vball/ball/track.py`:
```python
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BallTrack:
    """Per-frame ball position in work-video pixels. x/y are NaN where the ball is not visible."""

    visible: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.visible)


def load_tracknet_csv(path: Path, n_frames: int) -> BallTrack:
    visible = np.zeros(n_frames, dtype=bool)
    x = np.full(n_frames, np.nan)
    y = np.full(n_frames, np.nan)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["Frame"])
            if int(row["Visibility"]) == 1 and 0 <= frame < n_frames:
                visible[frame] = True
                x[frame] = float(row["X"])
                y[frame] = float(row["Y"])
    return BallTrack(visible=visible, x=x, y=y)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_track.py -v`
Expected: `2 passed`

- [ ] **Step 5: Append synthetic-track helpers to `tests/helpers.py`**

Add these imports at the top of `tests/helpers.py` (below the existing imports):
```python
import numpy as np

from vball.ball.track import BallTrack
```

Append at the end of `tests/helpers.py`:
```python
FPS = 30


def make_track(n_frames: int, flights=(), statics=(), fps: int = FPS) -> BallTrack:
    """flights: (start_s, end_s) with the ball moving at a constant 600 px/s back and forth.
    statics: (start_s, end_s) with the ball visible but held still (1 px jitter)."""
    visible = np.zeros(n_frames, dtype=bool)
    x = np.full(n_frames, np.nan)
    y = np.full(n_frames, np.nan)
    for start_s, end_s in flights:
        a, b = int(start_s * fps), min(int(end_s * fps), n_frames)
        phase = (600.0 * np.arange(b - a) / fps) % 1200.0
        x[a:b] = 40.0 + np.where(phase < 600.0, phase, 1200.0 - phase)
        y[a:b] = 360.0
        visible[a:b] = True
    for start_s, end_s in statics:
        a, b = int(start_s * fps), min(int(end_s * fps), n_frames)
        x[a:b] = 640.0 + (np.arange(b - a) % 2)
        y[a:b] = 600.0
        visible[a:b] = True
    return BallTrack(visible=visible, x=x, y=y)


def write_tracknet_csv(path: Path, track: BallTrack) -> Path:
    """Write a track in TrackNetV3's output format (invisible frames as 0,0,0)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Frame,Visibility,X,Y"]
    for i in range(len(track)):
        if track.visible[i]:
            lines.append(f"{i},1,{int(track.x[i])},{int(track.y[i])}")
        else:
            lines.append(f"{i},0,0,0")
    path.write_text("\n".join(lines) + "\n")
    return path
```

Add a round-trip test to `tests/test_track.py`:
```python
from helpers import make_track, write_tracknet_csv


def test_round_trip_with_helper(tmp_path):
    track = make_track(90, flights=[(1.0, 2.0)])
    loaded = load_tracknet_csv(write_tracknet_csv(tmp_path / "b.csv", track), n_frames=90)
    assert loaded.visible.tolist() == track.visible.tolist()
```

- [ ] **Step 6: Run all tests**

Run: `uv run pytest -v`
Expected: all pass (`12 passed`).

- [ ] **Step 7: Commit**

```bash
git add src/vball/ball tests/helpers.py tests/test_track.py
git commit -m "feat: load TrackNet ball CSV into dense per-frame track" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Rally detection

**Files:**
- Create: `src/vball/rallies.py`
- Test: `tests/test_rallies.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_rallies.py`:
```python
import numpy as np
from helpers import FPS, make_track

from vball.ball.track import BallTrack
from vball.rallies import Rally, ball_speed, detect_rallies

W, H = 1280, 720
N = 60 * FPS  # one minute of video


def detect(track):
    return detect_rallies(track, fps=FPS, width=W, height=H)


def test_speed_is_in_diagonals_per_second():
    track = make_track(N, flights=[(10, 12)])
    speed = ball_speed(track, FPS, W, H)
    diag = np.hypot(W, H)
    assert np.isnan(speed[10 * FPS])  # previous frame not visible
    assert abs(speed[10 * FPS + 5] - 600 / diag) < 1e-6


def test_single_flight_becomes_one_padded_rally():
    rallies = detect(make_track(N, flights=[(10, 18)]))
    assert len(rallies) == 1
    assert abs(rallies[0].start_frame - 9 * FPS) <= 15
    assert abs(rallies[0].end_frame - 19 * FPS) <= 15


def test_ball_held_still_is_not_a_rally():
    assert detect(make_track(N, statics=[(30, 45)])) == []


def test_short_gap_is_merged():
    assert len(detect(make_track(N, flights=[(10, 14), (15, 19)]))) == 1


def test_long_gap_splits_rallies():
    assert len(detect(make_track(N, flights=[(10, 14), (25, 29)]))) == 2


def test_blip_shorter_than_min_rally_is_dropped():
    assert detect(make_track(N, flights=[(30, 31)])) == []


def test_teleporting_false_positives_are_ignored():
    visible = np.zeros(N, dtype=bool)
    visible[300:600] = True
    x = np.full(N, np.nan)
    x[300:600] = np.where(np.arange(300) % 2 == 0, 100.0, 1200.0)
    y = np.where(visible, 300.0, np.nan)
    assert detect(BallTrack(visible=visible, x=x, y=y)) == []


def test_empty_track():
    assert detect(make_track(N)) == []


def test_rally_at_video_start_is_clipped_to_zero():
    rallies = detect(make_track(N, flights=[(0.2, 5)]))
    assert rallies[0].start_frame == 0


def test_rally_times():
    assert Rally(30, 300).start_s(30.0) == 1.0
    assert Rally(30, 300).end_s(30.0) == 10.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rallies.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vball.rallies'`

- [ ] **Step 3: Write the implementation**

`src/vball/rallies.py`:
```python
import math
from dataclasses import dataclass

import numpy as np

from vball.ball.track import BallTrack


@dataclass(frozen=True)
class Rally:
    start_frame: int  # inclusive
    end_frame: int  # exclusive

    def start_s(self, fps: float) -> float:
        return self.start_frame / fps

    def end_s(self, fps: float) -> float:
        return self.end_frame / fps


@dataclass(frozen=True)
class RallyParams:
    min_speed: float = 0.15  # frame diagonals per second; slower = held / rolling ball
    max_speed: float = 4.0  # faster = detector jumping between false positives
    window_s: float = 1.0  # smoothing window for the in-flight signal
    min_active_frac: float = 0.3  # fraction of in-flight frames in the window to count as active
    max_gap_s: float = 2.0  # merge active runs separated by at most this
    min_rally_s: float = 2.0  # drop runs shorter than this
    pre_pad_s: float = 1.0  # include the serve toss
    post_pad_s: float = 1.0  # include the ball landing


def ball_speed(track: BallTrack, fps: float, width: int, height: int) -> np.ndarray:
    """Speed between consecutive frames in frame diagonals per second; NaN if either frame is invisible."""
    speed = np.full(len(track), np.nan)
    speed[1:] = np.hypot(np.diff(track.x), np.diff(track.y)) * fps / math.hypot(width, height)
    return speed


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _merge(intervals: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def detect_rallies(
    track: BallTrack, fps: float, width: int, height: int, params: RallyParams = RallyParams()
) -> list[Rally]:
    n = len(track)
    speed = ball_speed(track, fps, width, height)
    in_flight = ((speed >= params.min_speed) & (speed <= params.max_speed)).astype(float)
    window = max(1, round(params.window_s * fps))
    activity = np.convolve(in_flight, np.ones(window) / window, mode="same")
    runs = _merge(_runs(activity >= params.min_active_frac), round(params.max_gap_s * fps))
    runs = [(s, e) for s, e in runs if e - s >= round(params.min_rally_s * fps)]
    pre, post = round(params.pre_pad_s * fps), round(params.post_pad_s * fps)
    padded = [(max(0, s - pre), min(n, e + post)) for s, e in runs]
    return [Rally(s, e) for s, e in _merge(padded, 0)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rallies.py -v`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add src/vball/rallies.py tests/test_rallies.py
git commit -m "feat: heuristic rally detection from ball track" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Evaluation metrics

**Files:**
- Create: `src/vball/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluate.py`:
```python
import math

import numpy as np

from vball.evaluate import (
    interval_iou,
    load_intervals_csv,
    match_intervals,
    rally_metrics,
    restrict_to_span,
    visible_fraction,
)


def test_interval_iou():
    assert interval_iou((0, 10), (0, 10)) == 1.0
    assert interval_iou((0, 10), (5, 15)) == 5 / 15
    assert interval_iou((0, 1), (2, 3)) == 0.0


def test_matching_is_one_to_one_and_respects_threshold():
    pred = [(0, 10), (1, 9), (50, 51)]
    gt = [(0, 10), (40, 60)]
    assert match_intervals(pred, gt, min_iou=0.5) == [(0, 0)]


def test_perfect_metrics():
    m = rally_metrics([(0, 10), (20, 30)], [(0, 10), (20, 30)])
    assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0)
    assert m.mean_abs_start_err_s == 0.0


def test_false_positive_and_miss():
    m = rally_metrics(pred=[(0, 10), (100, 110)], gt=[(0, 10), (20, 30)])
    assert m.n_matched == 1
    assert m.precision == 0.5 and m.recall == 0.5


def test_start_end_errors():
    m = rally_metrics([(0.5, 10.5)], [(0, 10)])
    assert math.isclose(m.mean_abs_start_err_s, 0.5) and math.isclose(m.mean_abs_end_err_s, 0.5)


def test_no_predictions():
    m = rally_metrics([], [(0, 10)])
    assert m.precision == 0.0 and m.recall == 0.0 and math.isnan(m.mean_abs_start_err_s)


def test_summary_mentions_precision_and_recall():
    assert "precision 1.00 recall 1.00" in rally_metrics([(0, 10)], [(0, 10)]).summary()


def test_restrict_to_span_keeps_only_labelled_region():
    pred = [(0, 10), (100, 110), (1000, 1010)]
    gt = [(98, 108), (200, 210)]
    assert restrict_to_span(pred, gt, margin_s=5) == [(100, 110)]


def test_visible_fraction_inside_intervals():
    visible = np.array([True, False, True, True, False, False])
    assert visible_fraction(visible, [(0.0, 2.0)], fps=1.0) == 0.5


def test_load_intervals_csv(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("start_s,end_s\n1.5,9.0\n20,31.25\n")
    assert load_intervals_csv(p) == [(1.5, 9.0), (20.0, 31.25)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vball.evaluate'`

- [ ] **Step 3: Write the implementation**

`src/vball/evaluate.py`:
```python
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Interval = tuple[float, float]


def interval_iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_intervals(pred: list[Interval], gt: list[Interval], min_iou: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by descending IoU. Returns sorted (pred_index, gt_index) pairs."""
    pairs = sorted(
        ((interval_iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches = []
    for iou, i, j in pairs:
        if iou < min_iou:
            break
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matches.append((i, j))
    return sorted(matches)


@dataclass(frozen=True)
class RallyMetrics:
    n_pred: int
    n_gt: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    mean_abs_start_err_s: float
    mean_abs_end_err_s: float

    def summary(self) -> str:
        return (
            f"rallies: pred {self.n_pred} gt {self.n_gt} matched {self.n_matched} | "
            f"precision {self.precision:.2f} recall {self.recall:.2f} f1 {self.f1:.2f} | "
            f"start err {self.mean_abs_start_err_s:.2f}s end err {self.mean_abs_end_err_s:.2f}s"
        )


def rally_metrics(pred: list[Interval], gt: list[Interval], min_iou: float = 0.5) -> RallyMetrics:
    matches = match_intervals(pred, gt, min_iou)
    k = len(matches)
    precision = k / len(pred) if pred else 0.0
    recall = k / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    start_err = float(np.mean([abs(pred[i][0] - gt[j][0]) for i, j in matches])) if matches else math.nan
    end_err = float(np.mean([abs(pred[i][1] - gt[j][1]) for i, j in matches])) if matches else math.nan
    return RallyMetrics(len(pred), len(gt), k, precision, recall, f1, start_err, end_err)


def restrict_to_span(pred: list[Interval], gt: list[Interval], margin_s: float = 5.0) -> list[Interval]:
    """Keep predictions overlapping the labelled time span, so partial labelling (one set) is fair."""
    if not gt:
        return pred
    lo = min(g[0] for g in gt) - margin_s
    hi = max(g[1] for g in gt) + margin_s
    return [p for p in pred if p[1] > lo and p[0] < hi]


def visible_fraction(visible: np.ndarray, intervals: list[Interval], fps: float) -> float:
    """Fraction of frames inside the intervals where the ball was detected."""
    total = seen = 0
    for start_s, end_s in intervals:
        a, b = max(0, int(start_s * fps)), min(len(visible), int(end_s * fps))
        total += max(0, b - a)
        seen += int(visible[a:b].sum())
    return seen / total if total else 0.0


def load_intervals_csv(path: Path) -> list[Interval]:
    with open(path, newline="") as f:
        return [(float(row["start_s"]), float(row["end_s"])) for row in csv.DictReader(f)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add src/vball/evaluate.py tests/test_evaluate.py
git commit -m "feat: rally evaluation metrics (IoU matching, P/R/F1, boundary error)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: SQLite store

**Files:**
- Create: `src/vball/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_store.py`:
```python
from vball import store
from vball.rallies import Rally
from vball.video import VideoInfo


def make_db(tmp_path):
    conn = store.connect(tmp_path / "db" / "vball.db")
    match_id = store.add_match(conn, "Final", "C:/videos/final.mp4")
    store.set_video_info(conn, match_id, VideoInfo(1920, 1080, 30.0, 5400))
    return conn, match_id


def test_add_and_get_match(tmp_path):
    conn, match_id = make_db(tmp_path)
    match = store.get_match(conn, match_id)
    assert match["name"] == "Final"
    assert (match["width"], match["height"], match["fps"], match["n_frames"]) == (1920, 1080, 30.0, 5400)
    assert store.get_match(conn, 999) is None


def test_replace_rallies_overwrites(tmp_path):
    conn, match_id = make_db(tmp_path)
    store.replace_rallies(conn, match_id, [Rally(0, 30), Rally(60, 90)])
    store.replace_rallies(conn, match_id, [Rally(30, 300)])
    rallies = store.get_rallies(conn, match_id)
    assert len(rallies) == 1
    assert rallies[0]["idx"] == 0
    assert (rallies[0]["start_s"], rallies[0]["end_s"]) == (1.0, 10.0)


def test_list_matches_counts_rallies(tmp_path):
    conn, match_id = make_db(tmp_path)
    store.replace_rallies(conn, match_id, [Rally(0, 30), Rally(60, 90)])
    store.add_match(conn, "Empty", "x.mp4")
    matches = store.list_matches(conn)
    assert [m["name"] for m in matches] == ["Final", "Empty"]
    assert matches[0]["n_rallies"] == 2
    assert matches[0]["rally_frames"] == 60
    assert matches[1]["n_rallies"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'store'`

- [ ] **Step 3: Write the implementation**

`src/vball/store.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/vball/store.py tests/test_store.py
git commit -m "feat: SQLite store for matches and rallies" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Vendor TrackNetV3, download weights, runner

**Files:**
- Create: `third_party/TrackNetV3/` (vendored), `third_party/TrackNetV3/VENDORED.md`, `scripts/download_models.py`, `src/vball/ball/tracknet.py`
- Test: `tests/test_tracknet.py`

- [ ] **Step 1: Vendor the upstream code at a pinned commit**

```bash
git clone https://github.com/qaz812345/TrackNetV3.git ../TrackNetV3-src
git -C ../TrackNetV3-src checkout 6eda442ada1740573f200f836d93edc9a541ee86
mkdir -p third_party/TrackNetV3
cp -r ../TrackNetV3-src/LICENSE ../TrackNetV3-src/README.md ../TrackNetV3-src/dataset.py \
      ../TrackNetV3-src/model.py ../TrackNetV3-src/predict.py ../TrackNetV3-src/test.py \
      ../TrackNetV3-src/utils third_party/TrackNetV3/
rm -rf ../TrackNetV3-src
```

- [ ] **Step 2: Patch `torch.load` for torch ≥ 2.6**

In `third_party/TrackNetV3/predict.py` there are two calls. Change:
```python
    tracknet_ckpt = torch.load(
        args.tracknet_file,
        map_location=device
    )
```
to:
```python
    tracknet_ckpt = torch.load(
        args.tracknet_file,
        map_location=device,
        weights_only=False,
    )
```
and change:
```python
        inpaintnet_ckpt = torch.load(
            args.inpaintnet_file,
            map_location=device
        )
```
to:
```python
        inpaintnet_ckpt = torch.load(
            args.inpaintnet_file,
            map_location=device,
            weights_only=False,
        )
```

- [ ] **Step 3: Patch the empty-final-window crash**

In `third_party/TrackNetV3/dataset.py`, class `Video_IterableDataset`, method `__iter__`: when the frame count is a multiple of the window (nonoverlap mode), the last loop iteration has an empty `frame_list` and `frame_list[-1]` raises `IndexError`. Change:
```python
                frame_list.append(frame)
                end_f_id += 1

            # Form a sequence
```
to:
```python
                frame_list.append(frame)
                end_f_id += 1

            if not frame_list:
                break

            # Form a sequence
```

- [ ] **Step 4: Write `third_party/TrackNetV3/VENDORED.md`**

```markdown
# Vendored TrackNetV3

Source: https://github.com/qaz812345/TrackNetV3 at commit 6eda442ada1740573f200f836d93edc9a541ee86 (MIT, see LICENSE).
Copied: LICENSE, README.md, dataset.py, model.py, predict.py, test.py, utils/.

Local patches:
1. predict.py: `torch.load(..., weights_only=False)` (torch >= 2.6 defaults to weights_only=True; checkpoints contain a param dict).
2. dataset.py `Video_IterableDataset.__iter__`: `if not frame_list: break` to avoid IndexError when the frame count is a multiple of seq_len.

Run from this directory (it imports `test`, `dataset`, `utils` relative to the cwd). Pass video paths with forward slashes: predict.py derives the output name with `split('/')`.
```

- [ ] **Step 5: Write the model download script**

`scripts/download_models.py`:
```python
"""Download the volleyball-fine-tuned TrackNetV3 weights (MIT) into models/."""

import shutil

import torch
from huggingface_hub import hf_hub_download

from vball.config import MODELS_DIR, TRACKNET_WEIGHTS

REPO_ID = "deadfast/beach-volley-vision-models"
REVISION = "b03107eddc5bc32d2f89f70b225e3b67da9c34e7"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(REPO_ID, "tracknet_best.pt", revision=REVISION)
    shutil.copyfile(cached, TRACKNET_WEIGHTS)
    ckpt = torch.load(TRACKNET_WEIGHTS, map_location="cpu", weights_only=False)
    print("keys:", sorted(ckpt.keys()))
    print("param_dict:", ckpt.get("param_dict"))


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/download_models.py`
Expected: `keys:` includes `'model'` and `'param_dict'`, and `param_dict` contains `seq_len` and `bg_mode`.
**If either key is missing, STOP and report the printed keys** — the checkpoint is not in upstream TrackNetV3 format and the runner must be adapted before continuing.

- [ ] **Step 6: Write the failing test for the runner command**

`tests/test_tracknet.py`:
```python
from vball.ball.tracknet import tracknet_command, tracknet_csv_path


def test_command_uses_streaming_nonoverlap_mode_and_posix_paths(tmp_path):
    video = tmp_path / "matches" / "1" / "work.mp4"
    cmd = tracknet_command(video, tmp_path / "out", tmp_path / "w.pt", batch_size=4, python="py")
    assert cmd[:2] == ["py", "predict.py"]
    assert cmd[cmd.index("--video_file") + 1] == video.as_posix()
    assert cmd[cmd.index("--tracknet_file") + 1] == (tmp_path / "w.pt").as_posix()
    assert cmd[cmd.index("--save_dir") + 1] == (tmp_path / "out").as_posix()
    assert cmd[cmd.index("--eval_mode") + 1] == "nonoverlap"
    assert cmd[cmd.index("--batch_size") + 1] == "4"
    assert "--large_video" in cmd
    assert "--inpaintnet_file" not in cmd


def test_csv_path_matches_predict_py_naming(tmp_path):
    assert tracknet_csv_path(tmp_path, tmp_path / "work.mp4") == tmp_path / "work_ball.csv"
```

- [ ] **Step 7: Run test to verify it fails**

Run: `uv run pytest tests/test_tracknet.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vball.ball.tracknet'`

- [ ] **Step 8: Write the runner**

`src/vball/ball/tracknet.py`:
```python
import shutil
import subprocess
import sys
from pathlib import Path

from vball.config import TRACKNET_DIR, TRACKNET_WEIGHTS


def tracknet_command(
    video: Path, out_dir: Path, weights: Path, batch_size: int = 8, python: str = sys.executable
) -> list[str]:
    return [
        python, "predict.py",
        "--video_file", video.as_posix(),
        "--tracknet_file", weights.as_posix(),
        "--save_dir", out_dir.as_posix(),
        "--eval_mode", "nonoverlap",
        "--large_video",
        "--batch_size", str(batch_size),
    ]


def tracknet_csv_path(out_dir: Path, video: Path) -> Path:
    return out_dir / f"{video.stem}_ball.csv"


def run_tracknet(video: Path, out_csv: Path, weights: Path = TRACKNET_WEIGHTS, batch_size: int = 8) -> Path:
    """Run vendored TrackNetV3 on video and move its CSV to out_csv."""
    if not weights.exists():
        raise FileNotFoundError(f"TrackNet weights not found at {weights}; run scripts/download_models.py")
    tmp_dir = (out_csv.parent / "tracknet_tmp").resolve()
    subprocess.run(
        tracknet_command(video.resolve(), tmp_dir, weights.resolve(), batch_size),
        cwd=TRACKNET_DIR,
        check=True,
    )
    tracknet_csv_path(tmp_dir, video).replace(out_csv)
    shutil.rmtree(tmp_dir)
    return out_csv
```

- [ ] **Step 9: Run tests**

Run: `uv run pytest tests/test_tracknet.py -v`
Expected: `2 passed`

- [ ] **Step 10: Smoke-test on real footage (manual, needs one of your match videos)**

Cut a 60-second excerpt from the middle of a match (replace the source path):
```bash
mkdir -p data/smoke
ffmpeg -y -ss 600 -i "D:/Volleyball/match1.mp4" -t 60 -c copy data/smoke/excerpt_src.mp4
uv run python -c "from pathlib import Path; from vball.video import ingest; print(ingest(Path('data/smoke/excerpt_src.mp4'), Path('data/smoke/excerpt.mp4')))"
cd third_party/TrackNetV3 && uv run python predict.py --video_file ../../data/smoke/excerpt.mp4 --tracknet_file ../../models/tracknet_volleyball.pt --save_dir ../../data/smoke/pred --eval_mode nonoverlap --large_video --batch_size 8 --output_video; cd ../..
```
Expected: `data/smoke/pred/excerpt_ball.csv` with one row per frame and `data/smoke/pred/excerpt.mp4` with the ball trajectory drawn. While it runs, check `nvidia-smi` memory stays under 4 GB (lower `--batch_size` to 4 if CUDA out of memory). Open the overlay video and note roughly how often the ball is marked correctly during play — write the observation down for Task 13.

- [ ] **Step 11: Commit**

```bash
git add third_party scripts/download_models.py src/vball/ball/tracknet.py tests/test_tracknet.py
git commit -m "feat: vendored TrackNetV3 (patched) with volleyball weights and runner" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Rally clip and rallies-only export

**Files:**
- Create: `src/vball/export.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_export.py`:
```python
from pathlib import Path

from helpers import make_test_video, requires_ffmpeg

from vball.export import clip_command, concat_list, export_rallies
from vball.video import probe


def test_clip_command_seeks_before_input_and_stream_copies():
    cmd = clip_command(Path("in.mp4"), Path("out.mp4"), 12.5, 20.0)
    assert cmd[cmd.index("-ss") + 1] == "12.500"
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-t") + 1] == "7.500"
    assert cmd[cmd.index("-c") + 1] == "copy"


def test_concat_list_escapes_quotes(tmp_path):
    p = tmp_path / "it's.mp4"
    assert concat_list([p]) == f"file '{p.resolve().as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"


@requires_ffmpeg
def test_export_rallies_concatenates_clips(tmp_path):
    src = make_test_video(tmp_path / "src.mp4", seconds=10)
    out = tmp_path / "condensed.mp4"
    clips = export_rallies(src, [(1.0, 3.0), (5.0, 8.0)], out, tmp_path / "clips")
    assert len(clips) == 2 and all(c.exists() for c in clips)
    assert abs(probe(out).duration_s - 5.0) <= 1.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vball.export'`

- [ ] **Step 3: Write the implementation**

`src/vball/export.py`:
```python
import subprocess
from pathlib import Path

FFMPEG = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]


def clip_command(src: Path, dst: Path, start_s: float, end_s: float) -> list[str]:
    """Stream-copy cut; starts at the keyframe at or before start_s (work videos have one per second)."""
    return [
        *FFMPEG,
        "-ss", f"{start_s:.3f}", "-i", str(src),
        "-t", f"{end_s - start_s:.3f}",
        "-c", "copy", "-avoid_negative_ts", "make_zero",
        str(dst),
    ]


def concat_list(paths: list[Path]) -> str:
    lines = []
    for p in paths:
        escaped = p.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'\n")
    return "".join(lines)


def export_rallies(src: Path, intervals: list[tuple[float, float]], out_path: Path, clips_dir: Path) -> list[Path]:
    """Write one clip per rally into clips_dir and a concatenated rallies-only video at out_path."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, (start_s, end_s) in enumerate(intervals, start=1):
        dst = clips_dir / f"rally_{i:03d}.mp4"
        subprocess.run(clip_command(src, dst, start_s, end_s), check=True)
        clips.append(dst)
    list_file = clips_dir / "concat.txt"
    list_file.write_text(concat_list(clips), encoding="utf-8")
    subprocess.run(
        [*FFMPEG, "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_path)],
        check=True,
    )
    return clips
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_export.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/vball/export.py tests/test_export.py
git commit -m "feat: per-rally clips and rallies-only video export" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Processing pipeline

**Files:**
- Create: `src/vball/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
from pathlib import Path

from helpers import make_test_video, make_track, requires_ffmpeg, write_tracknet_csv

from vball import store
from vball.config import Paths
from vball.pipeline import process_match, redetect
from vball.video import probe


def fake_ball_runner(video: Path, out_csv: Path) -> None:
    n_frames = probe(video).n_frames
    write_tracknet_csv(out_csv, make_track(n_frames, flights=[(5.0, 12.0)]))


@requires_ffmpeg
def test_process_match_stores_detected_rallies(tmp_path):
    src = make_test_video(tmp_path / "match.mp4", seconds=20, fps=30)
    paths = Paths(tmp_path / "data")

    match_id = process_match(src, paths, encoder="libx264", ball_runner=fake_ball_runner)

    assert paths.work_video(match_id).exists()
    assert paths.ball_csv(match_id).exists()
    conn = store.connect(paths.db_path)
    match = store.get_match(conn, match_id)
    assert match["name"] == "match" and match["fps"] == 30.0
    rallies = store.get_rallies(conn, match_id)
    assert len(rallies) == 1
    assert abs(rallies[0]["start_s"] - 4.0) < 0.6
    assert abs(rallies[0]["end_s"] - 13.0) < 0.6


@requires_ffmpeg
def test_redetect_replaces_rallies_from_cached_track(tmp_path):
    src = make_test_video(tmp_path / "match.mp4", seconds=20, fps=30)
    paths = Paths(tmp_path / "data")
    match_id = process_match(src, paths, encoder="libx264", ball_runner=fake_ball_runner)
    n_frames = probe(paths.work_video(match_id)).n_frames
    write_tracknet_csv(paths.ball_csv(match_id), make_track(n_frames, flights=[(1.0, 4.0), (10.0, 14.0)]))

    conn = store.connect(paths.db_path)
    rallies = redetect(conn, paths, match_id)

    assert len(rallies) == 2
    assert len(store.get_rallies(conn, match_id)) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vball.pipeline'`

- [ ] **Step 3: Write the implementation**

`src/vball/pipeline.py`:
```python
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
    encoder: str = "h264_nvenc",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/vball/pipeline.py tests/test_pipeline.py
git commit -m "feat: process_match pipeline and cached redetect" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Web app (API, video serving, viewer with GT labelling)

**Files:**
- Create: `src/vball/web/__init__.py` (empty), `src/vball/web/app.py`, `src/vball/web/static/index.html`, `src/vball/web/static/app.js`, `src/vball/web/static/style.css`
- Test: `tests/test_web.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_web.py`:
```python
from fastapi.testclient import TestClient

from vball import store
from vball.config import Paths
from vball.rallies import Rally
from vball.video import VideoInfo
from vball.web.app import create_app


def make_client(tmp_path):
    paths = Paths(tmp_path / "data")
    conn = store.connect(paths.db_path)
    match_id = store.add_match(conn, "Final", "src.mp4")
    store.set_video_info(conn, match_id, VideoInfo(1280, 720, 30.0, 900))
    store.replace_rallies(conn, match_id, [Rally(30, 300), Rally(450, 600)])
    conn.close()
    paths.match_dir(match_id).mkdir(parents=True)
    paths.work_video(match_id).write_bytes(bytes(range(256)) * 4)
    return TestClient(create_app(paths)), match_id


def test_list_matches(tmp_path):
    client, _ = make_client(tmp_path)
    matches = client.get("/api/matches").json()
    assert matches[0]["name"] == "Final" and matches[0]["n_rallies"] == 2


def test_rallies_in_seconds(tmp_path):
    client, match_id = make_client(tmp_path)
    rallies = client.get(f"/api/matches/{match_id}/rallies").json()
    assert (rallies[0]["start_s"], rallies[0]["end_s"]) == (1.0, 10.0)


def test_unknown_match_is_404(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/api/matches/999/rallies").status_code == 404


def test_video_supports_range_requests(tmp_path):
    client, match_id = make_client(tmp_path)
    res = client.get(f"/media/{match_id}/work.mp4", headers={"Range": "bytes=0-99"})
    assert res.status_code == 206
    assert len(res.content) == 100


def test_index_page(tmp_path):
    client, _ = make_client(tmp_path)
    res = client.get("/")
    assert res.status_code == 200 and "<title>vball</title>" in res.text
    assert client.get("/app.js").status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vball.web'`

- [ ] **Step 3: Write the backend**

Create empty `src/vball/web/__init__.py`.

`src/vball/web/app.py`:
```python
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
```

- [ ] **Step 4: Write the frontend**

`src/vball/web/static/index.html`:
```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>vball</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <header>
    <h1>vball</h1>
    <select id="match" aria-label="Match"></select>
    <label><input type="checkbox" id="continuous" checked> Rallies only</label>
    <label>Speed
      <select id="speed">
        <option value="1">1×</option>
        <option value="1.5">1.5×</option>
        <option value="2">2×</option>
      </select>
    </label>
  </header>
  <main>
    <section>
      <video id="video" controls preload="metadata"></video>
      <p class="help">N / P: next / previous rally · S: mark rally start (serve toss) · E: mark rally end (ball dead) · U: undo last label</p>
    </section>
    <aside>
      <h2>Rallies <span id="rally-count"></span></h2>
      <ol id="rallies"></ol>
      <h2>Ground-truth labels <span id="label-count"></span></h2>
      <p id="pending" class="help"></p>
      <button id="download">Download GT CSV</button>
      <button id="clear">Clear labels</button>
    </aside>
  </main>
  <script src="/app.js"></script>
</body>
</html>
```

`src/vball/web/static/app.js`:
```javascript
const $ = (sel) => document.querySelector(sel);
const video = $("#video");
let matchId = null;
let rallies = [];
let current = -1;
let labels = [];
let pendingStart = null;

function fmt(t) {
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(1).padStart(4, "0")}`;
}

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

async function loadMatches() {
  const matches = await getJson("/api/matches");
  const sel = $("#match");
  sel.innerHTML = "";
  for (const m of matches) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `#${m.id} ${m.name} (${m.n_rallies} rallies)`;
    sel.appendChild(opt);
  }
  sel.onchange = () => selectMatch(Number(sel.value));
  if (matches.length) await selectMatch(matches[matches.length - 1].id);
}

async function selectMatch(id) {
  matchId = id;
  $("#match").value = String(id);
  rallies = await getJson(`/api/matches/${id}/rallies`);
  video.src = `/media/${id}/work.mp4`;
  current = -1;
  loadLabels();
  renderRallies();
  renderLabels();
}

function renderRallies() {
  const list = $("#rallies");
  list.innerHTML = "";
  rallies.forEach((r, i) => {
    const li = document.createElement("li");
    li.textContent = `${fmt(r.start_s)} (${(r.end_s - r.start_s).toFixed(1)} s)`;
    if (i === current) li.className = "active";
    li.onclick = () => playRally(i);
    list.appendChild(li);
  });
  $("#rally-count").textContent = `(${rallies.length})`;
}

function playRally(i) {
  if (i < 0 || i >= rallies.length) return;
  current = i;
  video.currentTime = rallies[i].start_s;
  video.play();
  renderRallies();
}

video.addEventListener("timeupdate", () => {
  if (current < 0 || video.currentTime < rallies[current].end_s) return;
  if ($("#continuous").checked && current + 1 < rallies.length) {
    playRally(current + 1);
  } else {
    video.pause();
    current = -1;
    renderRallies();
  }
});

function labelKey() {
  return `vball-gt-${matchId}`;
}

function loadLabels() {
  try {
    labels = JSON.parse(localStorage.getItem(labelKey())) || [];
  } catch {
    labels = [];
  }
  pendingStart = null;
}

function saveLabels() {
  try {
    localStorage.setItem(labelKey(), JSON.stringify(labels));
  } catch {
    // storage unavailable (private window): labels live until the page is closed
  }
}

function renderLabels() {
  $("#label-count").textContent = `(${labels.length})`;
  $("#pending").textContent = pendingStart === null ? "" : `start marked at ${fmt(pendingStart)}; press E at rally end`;
}

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLSelectElement || e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key === "n") {
    playRally(current + 1);
  } else if (key === "p") {
    playRally(Math.max(0, current - 1));
  } else if (key === "s") {
    pendingStart = video.currentTime;
    renderLabels();
  } else if (key === "e" && pendingStart !== null && video.currentTime > pendingStart) {
    labels.push([pendingStart, video.currentTime]);
    pendingStart = null;
    saveLabels();
    renderLabels();
  } else if (key === "u") {
    labels.pop();
    saveLabels();
    renderLabels();
  } else {
    return;
  }
  e.preventDefault();
});

$("#download").onclick = () => {
  const rows = [...labels].sort((a, b) => a[0] - b[0]).map(([s, t]) => `${s.toFixed(2)},${t.toFixed(2)}`);
  const blob = new Blob([["start_s,end_s", ...rows].join("\n") + "\n"], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `gt_match_${matchId}.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};

$("#clear").onclick = () => {
  if (!confirm("Delete all ground-truth labels for this match?")) return;
  labels = [];
  saveLabels();
  renderLabels();
};

$("#speed").onchange = (e) => {
  video.defaultPlaybackRate = Number(e.target.value);
  video.playbackRate = Number(e.target.value);
};

loadMatches().catch((err) => document.body.prepend(String(err)));
```

`src/vball/web/static/style.css`:
```css
:root {
  color-scheme: light dark;
  --bg: #fafafa;
  --fg: #1a1a1a;
  --muted: #666;
  --accent: #2563eb;
  --row: #e8eefc;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #111; --fg: #eee; --muted: #999; --accent: #60a5fa; --row: #1e293b; }
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
header { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; padding: 8px 16px; }
h1 { font-size: 18px; margin: 0; }
main { display: grid; grid-template-columns: 1fr 320px; gap: 16px; padding: 0 16px 16px; }
@media (max-width: 800px) { main { grid-template-columns: 1fr; } }
video { width: 100%; background: #000; }
.help { color: var(--muted); }
aside { overflow-y: auto; max-height: calc(100vh - 60px); }
h2 { font-size: 15px; margin: 12px 0 6px; }
ol { margin: 0; padding-left: 28px; }
li { cursor: pointer; padding: 2px 4px; border-radius: 4px; font-variant-numeric: tabular-nums; }
li:hover, li.active { background: var(--row); }
li.active { color: var(--accent); font-weight: 600; }
button { margin: 4px 8px 4px 0; }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_web.py -v`
Expected: `5 passed`. If `test_video_supports_range_requests` returns 200 instead of 206, run `uv pip show starlette`; Range support needs Starlette ≥ 0.39 — upgrade FastAPI (`uv add "fastapi>=0.115"`, `uv sync --extra dev`) and re-run.

- [ ] **Step 6: Commit**

```bash
git add src/vball/web tests/test_web.py
git commit -m "feat: web viewer with rally playback and ground-truth labelling" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: CLI

**Files:**
- Create: `src/vball/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
from helpers import make_track, write_tracknet_csv

from vball import store
from vball.cli import main
from vball.config import default_paths
from vball.video import VideoInfo


def seed(tmp_path, monkeypatch):
    monkeypatch.setenv("VBALL_DATA", str(tmp_path / "data"))
    paths = default_paths()
    conn = store.connect(paths.db_path)
    match_id = store.add_match(conn, "m", "src.mp4")
    store.set_video_info(conn, match_id, VideoInfo(1280, 720, 30.0, 1800))
    conn.close()
    write_tracknet_csv(paths.ball_csv(match_id), make_track(1800, flights=[(10, 18), (30, 36)]))
    return match_id


def test_redetect_then_eval(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    assert main(["redetect", str(match_id)]) == 0
    assert "2 rallies" in capsys.readouterr().out

    gt = tmp_path / "gt.csv"
    gt.write_text("start_s,end_s\n10.0,18.0\n30.0,36.0\n")
    assert main(["eval", str(match_id), str(gt)]) == 0
    out = capsys.readouterr().out
    assert "precision 1.00 recall 1.00" in out
    assert "dead time removed" in out
    assert "ball visible in" in out


def test_unknown_match_returns_error(tmp_path, monkeypatch, capsys):
    seed(tmp_path, monkeypatch)
    assert main(["redetect", "999"]) == 1
    assert "no match with id 999" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vball.cli'`

- [ ] **Step 3: Write the implementation**

`src/vball/cli.py`:
```python
import argparse
import sys
from pathlib import Path

from vball import pipeline, store
from vball.ball.track import load_tracknet_csv
from vball.config import default_paths
from vball.evaluate import load_intervals_csv, rally_metrics, restrict_to_span, visible_fraction
from vball.export import export_rallies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vball", description="Volleyball video analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="ingest a match video, track the ball, detect rallies")
    p.add_argument("video", type=Path)
    p.add_argument("--name")
    p.add_argument("--encoder", default="h264_nvenc", help="ffmpeg encoder; use libx264 without NVENC")

    r = sub.add_parser("redetect", help="re-run rally detection from the cached ball track")
    r.add_argument("match_id", type=int)

    e = sub.add_parser("export", help="write per-rally clips and a rallies-only video")
    e.add_argument("match_id", type=int)
    e.add_argument("--out", type=Path)

    v = sub.add_parser("eval", help="compare detected rallies with a ground-truth CSV (start_s,end_s)")
    v.add_argument("match_id", type=int)
    v.add_argument("gt_csv", type=Path)

    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = default_paths()

    if args.command == "process":
        match_id = pipeline.process_match(args.video, paths, name=args.name, encoder=args.encoder)
        print(f"match {match_id} processed")
        return 0

    if args.command == "serve":
        import uvicorn

        from vball.web.app import create_app

        print(f"open http://127.0.0.1:{args.port}")
        uvicorn.run(create_app(paths), host="127.0.0.1", port=args.port)
        return 0

    conn = store.connect(paths.db_path)
    try:
        match = store.get_match(conn, args.match_id)
        if match is None:
            print(f"no match with id {args.match_id}", file=sys.stderr)
            return 1

        if args.command == "redetect":
            rallies = pipeline.redetect(conn, paths, args.match_id)
            print(f"{len(rallies)} rallies")

        elif args.command == "export":
            intervals = [(r["start_s"], r["end_s"]) for r in store.get_rallies(conn, args.match_id)]
            out = args.out or paths.match_dir(args.match_id) / "rallies_only.mp4"
            export_rallies(paths.work_video(args.match_id), intervals, out, paths.match_dir(args.match_id) / "clips")
            print(f"wrote {out}")

        elif args.command == "eval":
            pred = [(r["start_s"], r["end_s"]) for r in store.get_rallies(conn, args.match_id)]
            gt = load_intervals_csv(args.gt_csv)
            print(rally_metrics(restrict_to_span(pred, gt), gt).summary())
            duration_s = match["n_frames"] / match["fps"]
            kept_s = sum(end - start for start, end in pred)
            print(f"dead time removed: {1 - kept_s / duration_s:.0%} of {duration_s / 60:.1f} min")
            track = load_tracknet_csv(paths.ball_csv(args.match_id), match["n_frames"])
            print(f"ball visible in {visible_fraction(track.visible, gt, match['fps']):.0%} of labelled rally frames")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: `2 passed`

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: `49 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/vball/cli.py tests/test_cli.py
git commit -m "feat: vball CLI (process, redetect, export, eval, serve)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Real match run, ground truth, tuning, results

This task needs the user: a real match file and ~15 minutes of labelling one set.

**Files:**
- Create: `docs/results/phase1-rallies.md`
- Possibly modify: `src/vball/rallies.py` (`RallyParams` defaults only)

- [ ] **Step 1: Process a full match and time it**

```bash
time uv run vball process "D:/Volleyball/match1.mp4" --name "match1"
```
Expected: ends with `match 1 processed`. Record wall time for ingest and TrackNet (the two progress outputs). If CUDA runs out of memory, change `batch_size` default in `run_tracknet` to 4 and re-run.

- [ ] **Step 2: Watch the result**

```bash
uv run vball serve
```
Open http://127.0.0.1:8000, enable "Rallies only", press N to step through rallies. Note obvious failures (missed serves, between-rally ball handling shown as rallies, rallies cut short).

- [ ] **Step 3: Label ground truth for one set**

In the web app, play the first set at 1.5–2× speed. Press S at each serve toss and E when the ball is dead. Click "Download GT CSV" and save it as `data/gt/match1_set1.csv`.

- [ ] **Step 4: Evaluate**

```bash
uv run vball eval 1 data/gt/match1_set1.csv
```
Expected: three lines (rally metrics, dead time removed, ball visibility). Phase 1 target: precision ≥ 0.85, recall ≥ 0.90.

- [ ] **Step 5: Tune (repeat until target met or no further gain)**

Change one `RallyParams` default in `src/vball/rallies.py`, then:
```bash
uv run vball redetect 1 && uv run vball eval 1 data/gt/match1_set1.csv
```
Guidance: false rallies between points → raise `min_speed` or `min_rally_s`; rallies split in two → raise `max_gap_s`; serves cut off → raise `pre_pad_s`; missed rallies with low "ball visible" (< 60 %) → the ball model is the bottleneck, not the thresholds — record this, it decides the phase-2 fine-tune. After each change run `uv run pytest tests/test_rallies.py` — if a synthetic test breaks, the new default contradicts a documented behaviour; adjust the test only if the new behaviour is the intended one.

- [ ] **Step 6: Record results**

`docs/results/phase1-rallies.md` (fill in measured numbers):
```markdown
# Phase 1 results: rally detection

Match: match1 (<duration> min, <fps> fps, <resolution>), GT: set 1 (<n> rallies)

| Run | Params changed | Precision | Recall | F1 | Start err | End err | Dead time removed |
|---|---|---|---|---|---|---|---|
| baseline | defaults | | | | | | |

Ball visible in <x>% of labelled rally frames.
Processing time on RTX 3050 4 GB: ingest <m> min, TrackNet <m> min.
Observed failure modes: ...
Decision for phase 2 (fine-tune ball model? court mask? classifier?): ...
```

- [ ] **Step 7: Export a rallies-only video**

```bash
uv run vball export 1
```
Expected: `wrote ...data/matches/1/rallies_only.mp4`; watch it end to end.

- [ ] **Step 8: Commit**

```bash
git add docs/results/phase1-rallies.md src/vball/rallies.py
git commit -m "docs: phase 1 rally detection results on a real match" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: README and push

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# vball

Personal Balltime-style volleyball video analysis. Phase 1: rally cutting.

## Setup (Windows)

1. `winget install --id Gyan.FFmpeg -e`
2. Install [uv](https://docs.astral.sh/uv/), then in this folder: `uv sync --extra dev`
3. `uv run python scripts/download_models.py`

## Use

```bash
uv run vball process "D:/Volleyball/match1.mp4" --name match1   # ingest + ball tracking + rallies
uv run vball serve                                              # http://127.0.0.1:8000
uv run vball export 1                                           # clips + rallies_only.mp4
uv run vball eval 1 data/gt/match1_set1.csv                     # accuracy vs hand labels
uv run vball redetect 1                                         # re-run rally rules after tuning
```

Data lives in `data/` (override with `VBALL_DATA`). Design: `docs/superpowers/specs/2026-10-02-vball-design.md`.

Third-party: `third_party/TrackNetV3` (MIT), weights from `deadfast/beach-volley-vision-models` (MIT).
````

- [ ] **Step 2: Run the full suite one last time**

Run: `uv run pytest -v`
Expected: all pass.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "docs: README with setup and usage" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

---

## Later phases (separate plans, written after Task 13's results)

- **Phase 2:** court calibration UI + homography, player detection/tracking, pose, touch detection, action grammar, optional ball-model fine-tune.
- **Phase 3:** team colour clustering, jersey OCR + roster confirmation, per-player clips and count stats.
- **Phase 4:** ball metrics (serve speed, attack/jump height), heat maps, rotations, quality grades, optional LLM second opinion.
