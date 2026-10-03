# vball Phase 2b Part 2 (Court Calibration, Players + Tracking) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map video pixels to court metres per match (following camera bumps), detect and track every player, and show court lines, player boxes and a top-down mini map in the viewer.

**Architecture:** `vball/court.py` holds the court model, homography fitting and the `Calibration` (reference homography + per-segment camera corrections from `vball/camera.py`, ORB + RANSAC on the 512×288 copy). The web app gets a calibration mode (click landmarks on the paused video) and court/player APIs. Players come from one of two backends into a common `players.csv`; `vball/players.py` adds court coordinates at read time and computes label-free quality proxies. Spec: `docs/superpowers/specs/2026-10-03-phase2b-design.md`. Execute after Part 1 (reuses its overlay canvas).

**Tech Stack:** OpenCV (ORB, findHomography), NumPy, FastAPI/Pydantic, vanilla JS canvas, Ultralytics YOLO11 + ByteTrack, external RAVEL-VB (own env).

**Conventions:** as Part 1 (branch `phase2b`, `uv run --no-sync`, trailer, editor tool for backslashes).

## File structure

```
src/vball/court.py        LANDMARKS, COURT_LINES, fit_homography, apply_h, reprojection_errors, Segment, Calibration, save/load, calibration_json
src/vball/camera.py       estimate_shift, displacement, camera_segments
src/vball/players.py      Players, save/load_players, ravel_to_csv, run_ravel, run_yolo, with_court, player_stats
src/vball/web/app.py      + /court (GET, PUT), /players?start&end
src/vball/web/static/*    calibration mode, Show court / Show players toggles, mini map
src/vball/cli.py          + players, playereval
tests/test_court.py, test_camera.py, test_players.py (+ test_web.py, test_cli.py additions)
```

---

### Task 1: Court model and homography

**Files:** Create `src/vball/court.py`; Test `tests/test_court.py`

- [ ] **Step 1: Failing tests** — `tests/test_court.py`:

```python
import numpy as np
import pytest

from vball.court import LANDMARKS, Calibration, Segment, apply_h, fit_homography, load_calibration, reprojection_errors, save_calibration

# a plausible behind-the-baseline camera: court (m) -> image (px)
COURT_TO_IMAGE = np.array([[110.0, -20.0, 465.0], [0.0, -25.0, 1000.0], [0.0, 0.035, 1.0]])


def image_of(names):
    court = np.array([LANDMARKS[n] for n in names])
    return apply_h(COURT_TO_IMAGE, court), court


def test_fit_recovers_exact_homography():
    names = ["far_left_corner", "far_right_corner", "center_left", "center_right", "near_attack_left"]
    image, court = image_of(names)
    H = fit_homography(image, court)
    assert np.allclose(apply_h(H, image), court, atol=1e-6)
    assert reprojection_errors(H, image, court).max() < 1e-3  # float noise only


def test_needs_four_non_degenerate_points():
    image, court = image_of(["far_left_corner", "far_right_corner", "center_left"])
    with pytest.raises(ValueError):
        fit_homography(image, court)
    line = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    with pytest.raises(ValueError):
        fit_homography(line, line)


def test_calibration_follows_camera_segments(tmp_path):
    names = ["far_left_corner", "far_right_corner", "center_left", "center_right"]
    image, court = image_of(names)
    shift = np.array([[1.0, 0, 40.0], [0, 1.0, 0], [0, 0, 1.0]])  # camera bumped: picture moves 40 px right
    cal = Calibration.create(
        ref_frame=10,
        points=[{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(names, image)],
        segments=[Segment(0, 100, np.eye(3)), Segment(100, 200, shift)],
    )
    assert np.allclose(apply_h(cal.image_to_court_at(50), image), court, atol=1e-6)
    assert np.allclose(apply_h(cal.image_to_court_at(150), image + [40.0, 0.0]), court, atol=1e-6)
    save_calibration(tmp_path / "court.json", cal)
    again = load_calibration(tmp_path / "court.json")
    assert np.allclose(again.image_to_court_at(150), cal.image_to_court_at(150))
    assert again.points == cal.points and again.ref_frame == 10
```

- [ ] **Step 2:** run → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/vball/court.py`:

```python
"""Volleyball court model and pixel <-> court-metre homographies.

Court coordinates: x across the court 0..9 m (left -> right as seen from the camera), y along it 0..18 m
from the near (camera-side) baseline; the net is at y = 9."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

COURT_WIDTH, COURT_LENGTH, NET_Y = 9.0, 18.0, 9.0
LANDMARKS = {
    "near_left_corner": (0.0, 0.0),
    "near_right_corner": (9.0, 0.0),
    "near_attack_left": (0.0, 6.0),
    "near_attack_right": (9.0, 6.0),
    "center_left": (0.0, 9.0),
    "center_right": (9.0, 9.0),
    "far_attack_left": (0.0, 12.0),
    "far_attack_right": (9.0, 12.0),
    "far_left_corner": (0.0, 18.0),
    "far_right_corner": (9.0, 18.0),
}
COURT_LINES = [
    ((0.0, 0.0), (0.0, 18.0)),
    ((9.0, 0.0), (9.0, 18.0)),
    ((0.0, 0.0), (9.0, 0.0)),
    ((0.0, 18.0), (9.0, 18.0)),
    ((0.0, 9.0), (9.0, 9.0)),
    ((0.0, 6.0), (9.0, 6.0)),
    ((0.0, 12.0), (9.0, 12.0)),
]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return homog[:, :2] / homog[:, 2:3]


def fit_homography(image_pts: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    """Least-squares image -> court homography from >= 4 correspondences."""
    image_pts = np.asarray(image_pts, dtype=np.float64)
    court_pts = np.asarray(court_pts, dtype=np.float64)
    if len(image_pts) < 4:
        raise ValueError("need at least 4 landmarks")
    H, _ = cv2.findHomography(image_pts, court_pts, 0)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("landmarks are degenerate (e.g. all on one line)")
    return H


def reprojection_errors(H: np.ndarray, image_pts: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    """Pixel distance between each clicked point and its landmark projected back into the image."""
    back = apply_h(np.linalg.inv(H), court_pts)
    return np.linalg.norm(back - np.asarray(image_pts, dtype=np.float64), axis=1)


@dataclass
class Segment:
    start_frame: int
    end_frame: int  # exclusive
    ref_to_frame: np.ndarray  # image homography: reference-frame pixels -> pixels of frames in this segment


@dataclass
class Calibration:
    ref_frame: int
    points: list[dict]  # {"landmark", "x", "y"} in work-video pixels at ref_frame
    image_to_court: np.ndarray  # at ref_frame
    segments: list[Segment] = field(default_factory=list)

    @classmethod
    def create(cls, ref_frame: int, points: list[dict], segments: list[Segment]) -> "Calibration":
        image = np.array([[p["x"], p["y"]] for p in points])
        court = np.array([LANDMARKS[p["landmark"]] for p in points])
        return cls(ref_frame, points, fit_homography(image, court), segments)

    def errors(self) -> np.ndarray:
        image = np.array([[p["x"], p["y"]] for p in self.points])
        court = np.array([LANDMARKS[p["landmark"]] for p in self.points])
        return reprojection_errors(self.image_to_court, image, court)

    def segment_at(self, frame: int) -> Segment:
        for seg in self.segments:
            if seg.start_frame <= frame < seg.end_frame:
                return seg
        if not self.segments:
            return Segment(0, 1 << 62, np.eye(3))
        return self.segments[-1] if frame >= self.segments[-1].start_frame else self.segments[0]

    def image_to_court_at(self, frame: int) -> np.ndarray:
        return self.image_to_court @ np.linalg.inv(self.segment_at(frame).ref_to_frame)


def save_calibration(path: Path, cal: Calibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ref_frame": cal.ref_frame,
        "points": cal.points,
        "image_to_court": cal.image_to_court.tolist(),
        "segments": [
            {"start_frame": s.start_frame, "end_frame": s.end_frame, "ref_to_frame": s.ref_to_frame.tolist()}
            for s in cal.segments
        ],
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def load_calibration(path: Path) -> Calibration:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [Segment(s["start_frame"], s["end_frame"], np.array(s["ref_to_frame"])) for s in data["segments"]]
    return Calibration(data["ref_frame"], data["points"], np.array(data["image_to_court"]), segments)


def calibration_json(cal: Calibration) -> dict:
    """What the viewer needs: clicked points with their errors and a court -> image matrix per segment."""
    errors = cal.errors()
    return {
        "ref_frame": cal.ref_frame,
        "points": [{**p, "error_px": round(float(e), 1)} for p, e in zip(cal.points, errors)],
        "mean_error_px": round(float(errors.mean()), 2),
        "segments": [
            {
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "court_to_image": np.linalg.inv(cal.image_to_court_at(s.start_frame)).tolist(),
            }
            for s in (cal.segments or [Segment(0, 1 << 62, np.eye(3))])
        ],
        "lines": COURT_LINES,
        "landmarks": LANDMARKS,
    }
```

- [ ] **Step 4:** `uv run --no-sync pytest -q` → all pass.
- [ ] **Step 5: Commit** `feat: court model, homography fitting and segment-aware calibration`.

---

### Task 2: Camera-bump detection

**Files:** Create `src/vball/camera.py`; Test `tests/test_camera.py`

- [ ] **Step 1: Failing tests** — `tests/test_camera.py`:

```python
import cv2
import numpy as np

from vball.camera import camera_segments, displacement


def textured(w=512, h=288, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 90, np.uint8)
    for _ in range(120):
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 40)), y + int(rng.integers(8, 40))), color, -1)
    return img


def write_video(path, frames, fps=30):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def test_displacement_of_identity_and_shift():
    assert displacement(np.eye(3), 512, 288) == 0.0
    assert abs(displacement(np.array([[1, 0, 12.0], [0, 1, 0], [0, 0, 1]]), 512, 288) - 12.0) < 1e-9


def test_detects_a_camera_bump(tmp_path):
    base = textured()
    shifted = cv2.warpAffine(base, np.float32([[1, 0, 12], [0, 1, 0]]), (512, 288), borderMode=cv2.BORDER_REFLECT)
    write_video(tmp_path / "v.mp4", [base] * 30 + [shifted] * 30)
    segments = camera_segments(tmp_path / "v.mp4", ref_frame=0, n_frames=60, scale=1.0, step=5, threshold_px=3.0)
    assert len(segments) == 2
    assert segments[0].start_frame == 0 and 25 < segments[1].start_frame <= 30 and segments[1].end_frame == 60
    assert displacement(segments[0].ref_to_frame, 512, 288) < 1.0
    assert abs(segments[1].ref_to_frame[0, 2] - 12) < 1.5


def test_unreadable_video_gives_one_identity_segment(tmp_path):
    (tmp_path / "bad.mp4").write_bytes(b"not a video")
    segments = camera_segments(tmp_path / "bad.mp4", ref_frame=0, n_frames=100, scale=3.75)
    assert len(segments) == 1 and segments[0].end_frame == 100 and np.allclose(segments[0].ref_to_frame, np.eye(3))
```

- [ ] **Step 2:** run → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `src/vball/camera.py`:

```python
"""Detect camera moves (bumps, re-aiming) by matching the static background against a reference frame."""

from pathlib import Path

import cv2
import numpy as np

from vball.court import Segment


def displacement(H: np.ndarray, width: int, height: int) -> float:
    """Mean movement in pixels of a 3x3 grid of image points under H."""
    xs, ys = np.meshgrid(np.linspace(0, width, 3), np.linspace(0, height, 3))
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(9)], axis=1)
    moved = pts @ H.T
    moved = moved[:, :2] / moved[:, 2:3]
    return float(np.linalg.norm(moved - pts[:, :2], axis=1).mean())


def estimate_shift(ref_gray: np.ndarray, gray: np.ndarray, orb, matcher) -> np.ndarray | None:
    """Image homography ref -> frame from ORB matches, or None if too few reliable matches."""
    kp1, d1 = orb.detectAndCompute(ref_gray, None)
    kp2, d2 = orb.detectAndCompute(gray, None)
    if d1 is None or d2 is None or len(kp1) < 12 or len(kp2) < 12:
        return None
    good = [m for m, n in (p for p in matcher.knnMatch(d1, d2, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inliers is None or inliers.sum() < 12:
        return None
    return H


def camera_segments(
    video: Path, ref_frame: int, n_frames: int, scale: float, step: int = 60, threshold_px: float = 3.0
) -> list[Segment]:
    """Sample every `step` frames of the (512x288) video, estimate ref -> frame homographies, and split
    into segments wherever the picture moves by more than threshold_px (measured in work-video pixels,
    i.e. after scaling by `scale`). Returns homographies in work-video pixels."""
    identity = [Segment(0, n_frames, np.eye(3))]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return identity
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
    ok, ref = cap.read()
    if not ok:
        cap.release()
        return identity
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    h, w = ref_gray.shape
    orb = cv2.ORB_create(1500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    S = np.diag([scale, scale, 1.0])
    S_inv = np.linalg.inv(S)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    samples: list[tuple[int, np.ndarray]] = []
    last = np.eye(3)
    i = 0
    while i < n_frames and cap.grab():
        if i % step == 0:
            frame = cap.retrieve()[1]
            H = estimate_shift(ref_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), orb, matcher)
            last = S @ H @ S_inv if H is not None else last
            samples.append((i, last))
        i += 1
    cap.release()
    if not samples:
        return identity

    segments = [Segment(0, n_frames, samples[0][1])]
    for frame, H in samples[1:]:
        current = segments[-1].ref_to_frame
        if displacement(np.linalg.inv(current) @ H, w * scale, h * scale) > threshold_px:
            segments[-1].end_frame = frame
            segments.append(Segment(frame, n_frames, H))
    return segments
```

- [ ] **Step 4:** `uv run --no-sync pytest -q` → all pass. (Boundary lands on the first sample after the bump: frame 30 with step 5.)
- [ ] **Step 5: Real-footage check** — run on Kent 1 (bump at ~35 % of the set):

```bash
uv run --no-sync python -c "
from vball.camera import camera_segments, displacement
from vball.config import default_paths
from vball import store
p = default_paths(); m = store.get_match(store.connect(p.db_path), 1)
for s in camera_segments(p.track_video(1), 0, m['n_frames'], m['width'] / 512):
    print(s.start_frame, s.end_frame, round(displacement(s.ref_to_frame, m['width'], m['height']), 1))"
```
Expected: a segment boundary near 35 % of the set with displacement ≈ 119 px (phase-correlation estimate from phase 2b planning). Paste the output into the commit message.
- [ ] **Step 6: Commit** `feat: camera-bump detection with ORB + RANSAC`.

---

### Task 3: Court API

**Files:** Modify `src/vball/web/app.py`, `src/vball/config.py` (`Paths.court_json`); Test `tests/test_web.py`, `tests/test_config.py`

- [ ] **Step 1: Failing tests** — `tests/test_config.py::test_match_file_layout` add `assert paths.court_json(3) == tmp_path / "matches" / "3" / "court.json"`; `tests/test_web.py`:

```python
import numpy as np  # (already imported for the frame test)
from vball.court import LANDMARKS, apply_h

COURT_TO_IMAGE = np.array([[110.0, -20.0, 465.0], [0.0, -25.0, 1000.0], [0.0, 0.035, 1.0]])


def court_points(names):
    image = apply_h(COURT_TO_IMAGE, np.array([LANDMARKS[n] for n in names]))
    return [{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(names, image)]


def test_court_calibration_round_trip(tmp_path):
    client, match_id = make_client(tmp_path)  # placeholder video -> one identity segment
    assert client.get(f"/api/matches/{match_id}/court").status_code == 404
    names = ["far_left_corner", "far_right_corner", "center_left", "center_right", "near_attack_right"]
    res = client.put(f"/api/matches/{match_id}/court", json={"ref_frame": 5, "points": court_points(names)})
    assert res.status_code == 200
    body = res.json()
    assert body["mean_error_px"] < 0.01 and len(body["segments"]) == 1
    H = np.array(body["segments"][0]["court_to_image"])
    assert np.allclose(apply_h(H, [[0.0, 18.0]]), apply_h(COURT_TO_IMAGE, [[0.0, 18.0]]), atol=1e-3)
    assert client.get(f"/api/matches/{match_id}/court").json()["ref_frame"] == 5


def test_court_calibration_rejects_bad_input(tmp_path):
    client, match_id = make_client(tmp_path)
    three = court_points(["far_left_corner", "far_right_corner", "center_left"])
    assert client.put(f"/api/matches/{match_id}/court", json={"ref_frame": 0, "points": three}).status_code == 422
    bad = court_points(["far_left_corner", "far_right_corner", "center_left", "center_right"])
    bad[0]["landmark"] = "goal_post"
    assert client.put(f"/api/matches/{match_id}/court", json={"ref_frame": 0, "points": bad}).status_code == 422
```

- [ ] **Step 2:** run → failures.

- [ ] **Step 3: Implement** — `config.py` Paths: `def court_json(self, match_id): return self.match_dir(match_id) / "court.json"`. In `app.py` import `from vball.camera import camera_segments` and `from vball.court import LANDMARKS, Calibration, calibration_json, load_calibration, save_calibration`, add models:

```python
class CourtPoint(BaseModel):
    landmark: str
    x: float
    y: float


class CourtBody(BaseModel):
    ref_frame: int
    points: list[CourtPoint]
```

and routes:

```python
    @app.get("/api/matches/{match_id}/court")
    def get_court(match_id: int, conn: sqlite3.Connection = Depends(db)) -> dict:
        require_match(conn, match_id)
        path = paths.court_json(match_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="court not calibrated")
        return calibration_json(load_calibration(path))

    @app.put("/api/matches/{match_id}/court")
    def put_court(match_id: int, body: CourtBody, conn: sqlite3.Connection = Depends(db)) -> dict:
        match = require_match(conn, match_id)
        unknown = [p.landmark for p in body.points if p.landmark not in LANDMARKS]
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown landmarks: {unknown}")
        segments = camera_segments(
            paths.track_video(match_id), body.ref_frame, match["n_frames"], scale=match["width"] / 512
        )
        try:
            cal = Calibration.create(body.ref_frame, [p.model_dump() for p in body.points], segments)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        save_calibration(paths.court_json(match_id), cal)
        return calibration_json(cal)
```
(PUT takes ~30–60 s on a real set: camera segments read the whole 512×288 copy.)

- [ ] **Step 4:** `uv run --no-sync pytest -q` → all pass.
- [ ] **Step 5: Commit** `feat: court calibration API with camera segments`.

---

### Task 4: Calibration mode and "Show court" in the viewer

**Files:** Modify `index.html`, `app.js`, `style.css`; Test `tests/test_web.py`

- [ ] **Step 1: Failing test**

```python
def test_viewer_has_calibration_controls(tmp_path):
    client, _ = make_client(tmp_path)
    page = client.get("/").text
    for element_id in ("show-court", "btn-calibrate", "calib-panel", "calib-save"):
        assert f'id="{element_id}"' in page
```

- [ ] **Step 2:** run → 1 failure.

- [ ] **Step 3: HTML** — header: `<label><input type="checkbox" id="show-court"> Show court</label>`. Controls row: `<button id="btn-calibrate" title="Click known court points on the paused video">Calibrate court</button>`. In `<aside>`, first section:

```html
      <section id="calib-panel" hidden>
        <h2>Court calibration</h2>
        <p class="help">Pause on a frame where the court lines are clear. Click each landmark on the video in turn; <em>Skip</em> any you can't see. At least 4 are needed. Left/right as seen from the camera; "near" is the camera end.</p>
        <ol id="calib-list" class="list"></ol>
        <button id="calib-skip">Skip</button>
        <button id="calib-undo">Undo</button>
        <button id="calib-save" class="primary" disabled>Save calibration</button>
        <button id="calib-cancel">Cancel</button>
        <p id="calib-result" class="help"></p>
      </section>
```

- [ ] **Step 4: JS** — add to `app.js` (overlay section from Part 1 is extended; `drawOverlay` calls `drawCourt(ctx, frame, r)` before the ball):

```javascript
// ---------- court calibration ----------

const LANDMARK_ORDER = [
  "far_left_corner", "far_right_corner", "far_attack_left", "far_attack_right",
  "center_left", "center_right", "near_attack_left", "near_attack_right",
  "near_left_corner", "near_right_corner",
];
let court = null; // GET /court response
let calib = null; // {frame, index, points: [{landmark, x, y}]} while calibrating

async function loadCourt(id) {
  try {
    court = await getJson(`/api/matches/${id}/court`);
  } catch {
    court = null;
  }
  drawOverlay();
}

function project(H, [x, y]) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
}

function courtMatrixAt(frame) {
  const seg = court.segments.find((s) => frame >= s.start_frame && frame < s.end_frame) || court.segments.at(-1);
  return seg.court_to_image;
}

function drawCourt(ctx, frame, r) {
  if (calib) {
    ctx.fillStyle = "#22d3ee";
    for (const p of calib.points) {
      ctx.beginPath();
      ctx.arc(r.x + p.x * r.scale, r.y + p.y * r.scale, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  if (!$("#show-court").checked || !court) return;
  const H = courtMatrixAt(frame);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(34, 211, 238, 0.9)";
  for (const [a, b] of court.lines) {
    const [ax, ay] = project(H, a);
    const [bx, by] = project(H, b);
    ctx.beginPath();
    ctx.moveTo(r.x + ax * r.scale, r.y + ay * r.scale);
    ctx.lineTo(r.x + bx * r.scale, r.y + by * r.scale);
    ctx.stroke();
  }
}

function renderCalib() {
  $("#calib-panel").hidden = !calib;
  overlay.style.pointerEvents = calib ? "auto" : "none";
  overlay.style.cursor = calib ? "crosshair" : "";
  if (!calib) return;
  const list = $("#calib-list");
  list.innerHTML = "";
  LANDMARK_ORDER.forEach((name, i) => {
    const li = document.createElement("li");
    const done = calib.points.find((p) => p.landmark === name);
    li.textContent = `${name.replaceAll("_", " ")}${done ? " ✓" : ""}`;
    if (i === calib.index) li.className = "active";
    list.appendChild(li);
  });
  $("#calib-save").disabled = calib.points.length < 4;
  drawOverlay();
}

function startCalibration() {
  video.pause();
  calib = { frame: Math.round(video.currentTime * (ballData?.fps || 30)), index: 0, points: [] };
  $("#calib-result").textContent = "";
  renderCalib();
}

overlay.addEventListener("click", (e) => {
  if (!calib || calib.index >= LANDMARK_ORDER.length) return;
  const rect = overlay.getBoundingClientRect();
  const r = contentRect();
  const x = (e.clientX - rect.left - r.x) / r.scale;
  const y = (e.clientY - rect.top - r.y) / r.scale;
  calib.points.push({ landmark: LANDMARK_ORDER[calib.index], x, y });
  calib.index += 1;
  renderCalib();
});

bind("#btn-calibrate", startCalibration);
bind("#calib-skip", () => {
  if (calib && calib.index < LANDMARK_ORDER.length) calib.index += 1;
  renderCalib();
});
bind("#calib-undo", () => {
  if (!calib) return;
  const last = calib.points.pop();
  calib.index = last ? LANDMARK_ORDER.indexOf(last.landmark) : 0;
  renderCalib();
});
bind("#calib-cancel", () => {
  calib = null;
  renderCalib();
});
bind("#calib-save", async () => {
  $("#calib-result").textContent = "Saving… (checking the whole video for camera moves, ~1 min)";
  try {
    const res = await fetch(`/api/matches/${matchId}/court`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ref_frame: calib.frame, points: calib.points }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
    court = await res.json();
    calib = null;
    $("#show-court").checked = true;
    renderCalib();
    $("#calib-result").textContent = "";
    say(`Court saved: mean error ${court.mean_error_px} px, ${court.segments.length} camera segment(s)`, "ok");
  } catch (err) {
    $("#calib-result").textContent = `Not saved: ${err.message}`;
  }
});
$("#show-court").onchange = () => drawOverlay();
```

In `drawOverlay`, after `ctx.clearRect(...)`, compute `const r = contentRect(); const frame = Math.round(mediaTime * (ballData?.fps || 30)); drawCourt(ctx, frame, r);` and keep the ball drawing after it (move the early `return` for the ball toggle below `drawCourt`). In `selectMatch` add `loadCourt(id);` and set `calib = null; renderCalib();`. Note `bind` is defined in the input section: place this section after it, or move `bind` above.

- [ ] **Step 5:** `uv run --no-sync pytest -q` → all pass.

- [ ] **Step 6: Browser check (copy of match 1, port 8766)** — calibrate on a frame before the bump: click the visible landmarks; save; mean error ≤ 10 px; lines overlay the court. Seek past the bump (~35 % into the set): lines must still sit on the court lines (camera segments). Cancel/Undo/Skip behave. Delete `data/uitest`.

- [ ] **Step 7: Commit** `feat: court calibration mode and Show court overlay`.

---

### Task 5: Players data, backends, court enrichment, quality proxies

**Files:** Create `src/vball/players.py`; Modify `pyproject.toml` (ultralytics); Test `tests/test_players.py`

- [ ] **Step 1: Dependencies** — add to `[tool.uv.sources]`: `torchvision = [{ index = "pytorch-cu130" }]`, then `uv add ultralytics lap` (stop the web server first so uv can re-sync). Verify: `uv run python -c "import torch, torchvision, ultralytics; print(torch.__version__, torchvision.__version__, ultralytics.__version__, torch.cuda.is_available())"` → versions with `+cu130`, `True`. Ultralytics is AGPL-3.0: fine for this personal project; note it in README if the repo is ever published. The first `run_yolo` downloads `yolo11s.pt` (~19 MB, Ultralytics GitHub releases) into `models/`: ask the user before that first run.

- [ ] **Step 2: Failing tests** — `tests/test_players.py`:

```python
import json

import numpy as np

from vball.court import Calibration, LANDMARKS, Segment, apply_h
from vball.players import Players, load_players, player_stats, ravel_to_csv, save_players, with_court

COURT_TO_IMAGE = np.array([[110.0, -20.0, 465.0], [0.0, -25.0, 1000.0], [0.0, 0.035, 1.0]])


def calibration():
    names = ["far_left_corner", "far_right_corner", "near_left_corner", "near_right_corner"]
    image = apply_h(COURT_TO_IMAGE, np.array([LANDMARKS[n] for n in names]))
    points = [{"landmark": n, "x": float(x), "y": float(y)} for n, (x, y) in zip(names, image)]
    return Calibration.create(0, points, [Segment(0, 10_000, np.eye(3))])


def box_at(court_xy, half_width=20, height=120):
    fx, fy = apply_h(COURT_TO_IMAGE, [court_xy])[0]
    return (fx - half_width, fy - height, fx + half_width, fy)


def test_round_trip(tmp_path):
    rows = [(0, 3, 10.0, 20.0, 30.0, 140.0, 0.9), (1, 3, 11.0, 20.0, 31.0, 140.0, 0.8)]
    save_players(tmp_path / "p.csv", rows)
    p = load_players(tmp_path / "p.csv")
    assert p.frame.tolist() == [0, 1] and p.track_id.tolist() == [3, 3]
    assert np.allclose(p.box[1], [11, 20, 31, 140]) and np.allclose(p.score, [0.9, 0.8])


def test_ravel_json_conversion(tmp_path):
    payload = {"format": "ravel-vb-predictions-v1", "predictions": [
        {"frame_index": 4, "class_name": "player", "track_id": 7, "score": 0.7, "bbox_xyxy": [1, 2, 3, 4]},
        {"frame_index": 4, "class_name": "ball", "score": 0.9, "bbox_xyxy": [5, 6, 7, 8]},
    ]}
    (tmp_path / "r.json").write_text(json.dumps(payload))
    ravel_to_csv(tmp_path / "r.json", tmp_path / "p.csv")
    p = load_players(tmp_path / "p.csv")
    assert p.frame.tolist() == [4] and p.track_id.tolist() == [7]


def test_with_court_maps_feet_and_sides():
    rows = [(0, 1, *box_at((4.5, 3.0)), 0.9), (0, 2, *box_at((2.0, 15.0)), 0.9), (0, 3, *box_at((4.5, 30.0)), 0.9)]
    p = Players.from_rows(rows)
    court_xy, side, on_court = with_court(p, calibration())
    assert np.allclose(court_xy[0], [4.5, 3.0], atol=1e-6) and np.allclose(court_xy[1], [2.0, 15.0], atol=1e-6)
    assert side.tolist() == [0, 1, 1]  # 0 = near, 1 = far
    assert on_court.tolist() == [True, True, False]  # 30 m away: off court


def test_player_stats_counts_full_sides():
    rows = []
    for f in range(10):
        for i in range(6):
            rows.append((f, i, *box_at((1.0 + i, 3.0)), 0.9))  # 6 near
            rows.append((f, 10 + i, *box_at((1.0 + i, 15.0)), 0.9))  # 6 far
        if f >= 5:
            rows = [r for r in rows if not (r[0] == f and r[1] >= 13)]  # far side drops to 3 in frames 5-9
    stats = player_stats(Players.from_rows(rows), calibration(), rallies=[(0, 10)])
    assert stats.frames == 10
    assert stats.full_sides_share == 0.5
    assert stats.ids_per_side_per_rally == 6.0
```

- [ ] **Step 3:** run → `ModuleNotFoundError`.

- [ ] **Step 4: Implement** `src/vball/players.py`:

```python
"""Player detections/tracks: storage, backends (YOLO11 + ByteTrack, RAVEL-VB), court mapping and quality proxies."""

import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vball.config import MODELS_DIR
from vball.court import NET_Y, Calibration, Segment, apply_h

COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "score"]
ON_COURT_X = (-1.5, 10.5)  # metres, sidelines plus a margin
ON_COURT_Y = (-4.0, 22.0)  # baselines plus the serve zone


@dataclass
class Players:
    frame: np.ndarray  # (N,) int
    track_id: np.ndarray  # (N,) int
    box: np.ndarray  # (N, 4) x1 y1 x2 y2, work-video pixels
    score: np.ndarray  # (N,)

    @classmethod
    def from_rows(cls, rows) -> "Players":
        a = np.array(rows, dtype=np.float64).reshape(-1, 7)
        return cls(a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2:6], a[:, 6])

    def __len__(self) -> int:
        return len(self.frame)


def save_players(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow([int(r[0]), int(r[1]), *(round(float(v), 1) for v in r[2:6]), round(float(r[6]), 3)])


def load_players(path: Path) -> Players:
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return Players.from_rows(data) if data.size else Players.from_rows([])


def ravel_to_csv(json_path: Path, out_csv: Path) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = [
        (p["frame_index"], p["track_id"], *p["bbox_xyxy"], p["score"])
        for p in payload["predictions"]
        if p.get("class_name") == "player" and p.get("track_id") is not None
    ]
    save_players(out_csv, rows)


def run_ravel(video: Path, out_csv: Path, repo: Path) -> None:
    """RAVEL-VB runs in its own uv environment (Python >= 3.14, OpenVINO)."""
    json_path = out_csv.with_suffix(".ravel.json")
    subprocess.run(
        ["uv", "run", "--directory", str(repo), "src/inference_player_ball_openvino.py",
         str(video.resolve()), "--output", str(json_path.resolve())],
        check=True,
    )
    ravel_to_csv(json_path, out_csv)


def run_yolo(video: Path, out_csv: Path, model: str = "yolo11s.pt", imgsz: int = 960) -> None:
    from ultralytics import YOLO  # heavy import, only when used

    detector = YOLO(str(MODELS_DIR / model))
    rows = []
    results = detector.track(
        source=str(video), stream=True, classes=[0], tracker="bytetrack.yaml",
        imgsz=imgsz, half=True, verbose=False, persist=True,
    )
    for frame, result in enumerate(results):
        if result.boxes is None or result.boxes.id is None:
            continue
        ids = result.boxes.id.int().tolist()
        for box, track_id, conf in zip(result.boxes.xyxy.tolist(), ids, result.boxes.conf.tolist()):
            rows.append((frame, track_id, *box, conf))
    save_players(out_csv, rows)


def with_court(players: Players, cal: Calibration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Court (x, y) of each player's feet (bottom-centre of the box), side (0 near, 1 far), on-court flag."""
    feet = np.stack([(players.box[:, 0] + players.box[:, 2]) / 2, players.box[:, 3]], axis=1)
    court_xy = np.zeros_like(feet)
    segments = cal.segments or [Segment(0, 1 << 62, np.eye(3))]
    for i, seg in enumerate(segments):
        # same rule as Calibration.segment_at: frames before the first / after the last segment use those
        start = -np.inf if i == 0 else seg.start_frame
        end = np.inf if i == len(segments) - 1 else seg.end_frame
        rows = (players.frame >= start) & (players.frame < end)
        if rows.any():
            court_xy[rows] = apply_h(cal.image_to_court @ np.linalg.inv(seg.ref_to_frame), feet[rows])
    side = (court_xy[:, 1] >= NET_Y).astype(int)
    on_court = (
        (court_xy[:, 0] >= ON_COURT_X[0]) & (court_xy[:, 0] <= ON_COURT_X[1])
        & (court_xy[:, 1] >= ON_COURT_Y[0]) & (court_xy[:, 1] <= ON_COURT_Y[1])
    )
    return court_xy, side, on_court


@dataclass(frozen=True)
class PlayerStats:
    frames: int  # rally frames looked at
    full_sides_share: float  # share of rally frames with 5-7 on-court players on each side
    ids_per_side_per_rally: float  # mean distinct track ids per side per rally (ideal 6)

    def summary(self) -> str:
        return (
            f"players: {self.frames} rally frames | 5-7 per side in {self.full_sides_share:.0%} | "
            f"{self.ids_per_side_per_rally:.1f} track ids per side per rally"
        )


def player_stats(players: Players, cal: Calibration, rallies: list[tuple[int, int]]) -> PlayerStats:
    _, side, on_court = with_court(players, cal)
    frames = full = 0
    ids = []
    for start, end in rallies:
        in_rally = (players.frame >= start) & (players.frame < end) & on_court
        for s in (0, 1):
            ids.append(len(np.unique(players.track_id[in_rally & (side == s)])))
        counts = np.zeros((end - start, 2), dtype=int)
        np.add.at(counts, (players.frame[in_rally] - start, side[in_rally]), 1)
        frames += end - start
        full += int(((counts >= 5) & (counts <= 7)).all(axis=1).sum())
    return PlayerStats(frames, full / frames if frames else 0.0, float(np.mean(ids)) if ids else 0.0)
```

- [ ] **Step 5:** `uv run --no-sync pytest -q` → all pass.
- [ ] **Step 6: Commit** `feat: player tracks (YOLO11/RAVEL-VB backends), court mapping, quality proxies`.

---

### Task 6: `vball players` and `vball playereval`

**Files:** Modify `src/vball/cli.py`, `src/vball/config.py` (`Paths.players_csv`); Test `tests/test_cli.py`

- [ ] **Step 1: Failing tests**

```python
def test_players_command_dispatches_backend(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr("vball.players.run_yolo", lambda video, out_csv: calls.append(("yolo", out_csv)))
    assert main(["players", str(match_id), "--backend", "yolo"]) == 0
    assert calls == [("yolo", default_paths().players_csv(match_id))]


def test_playereval_needs_calibration(tmp_path, monkeypatch, capsys):
    match_id = seed(tmp_path, monkeypatch)
    from vball.players import save_players
    save_players(default_paths().players_csv(match_id), [(0, 1, 0, 0, 10, 10, 0.9)])
    assert main(["playereval", str(match_id)]) == 1
    assert "calibrate the court" in capsys.readouterr().err
```

- [ ] **Step 2:** run → failures.

- [ ] **Step 3: Implement** — `Paths.players_csv(self, match_id): return self.match_dir(match_id) / "players.csv"`. CLI parsers:

```python
    pl = sub.add_parser("players", help="detect and track players")
    pl.add_argument("match_id", type=int)
    pl.add_argument("--backend", choices=["yolo", "ravel"], default="yolo")
    pl.add_argument("--out", type=Path)
    pl.add_argument("--ravel-repo", type=Path, default=Path("data/external/asigatchov-fast-volleyball-tracking-inference"))

    pe = sub.add_parser("playereval", help="label-free player tracking quality on rally frames")
    pe.add_argument("match_id", type=int)
    pe.add_argument("--players", type=Path)
```
branches (inside the single-match `try:`; import `from vball import players as pl_mod` and `from vball.court import load_calibration` at the top of `cli.py` — `players.py` imports Ultralytics lazily, so this stays fast):

```python
        elif args.command == "players":
            out = args.out or paths.players_csv(args.match_id)
            if args.backend == "yolo":
                pl_mod.run_yolo(paths.work_video(args.match_id), out)
            else:
                pl_mod.run_ravel(paths.work_video(args.match_id), out, args.ravel_repo)
            print(f"wrote {out}")

        elif args.command == "playereval":
            if not paths.court_json(args.match_id).exists():
                print("calibrate the court first (viewer: Calibrate court)", file=sys.stderr)
                return 1
            cal = load_calibration(paths.court_json(args.match_id))
            players = pl_mod.load_players(args.players or paths.players_csv(args.match_id))
            fps = match["fps"]
            if paths.gt_csv(args.match_id).exists():
                rallies = [(round(l.start_s * fps), round(l.end_s * fps)) for l in load_labels(paths.gt_csv(args.match_id))]
            else:
                rallies = [(r["start_frame"], r["end_frame"]) for r in store.get_rallies(conn, args.match_id)]
            print(pl_mod.player_stats(players, cal, rallies).summary())
```

- [ ] **Step 4:** `uv run --no-sync pytest -q` → all pass.
- [ ] **Step 5: Commit** `feat: vball players and playereval commands`.

---

### Task 7: Players API, boxes overlay, mini map

**Files:** Modify `src/vball/web/app.py`, `index.html`, `app.js`, `style.css`; Test `tests/test_web.py`

- [ ] **Step 1: Failing tests**

```python
def test_players_window_with_court_positions(tmp_path):
    client, match_id = make_client(tmp_path)
    from vball.players import save_players
    names = ["far_left_corner", "far_right_corner", "center_left", "center_right"]
    client.put(f"/api/matches/{match_id}/court", json={"ref_frame": 0, "points": court_points(names)})
    fx, fy = apply_h(COURT_TO_IMAGE, [[4.5, 15.0]])[0]
    save_players(tmp_path / "data" / "matches" / str(match_id) / "players.csv",
                 [(10, 4, fx - 20, fy - 120, fx + 20, fy, 0.9), (500, 5, 0, 0, 10, 10, 0.9)])
    rows = client.get(f"/api/matches/{match_id}/players?start=0&end=100").json()
    assert len(rows) == 1 and rows[0]["frame"] == 10 and rows[0]["id"] == 4 and rows[0]["side"] == 1
    assert abs(rows[0]["court"][0] - 4.5) < 1e-3 and abs(rows[0]["court"][1] - 15.0) < 1e-3


def test_players_window_limits(tmp_path):
    client, match_id = make_client(tmp_path)
    assert client.get(f"/api/matches/{match_id}/players?start=0&end=10").status_code == 404  # no players.csv
    from vball.players import save_players
    save_players(tmp_path / "data" / "matches" / str(match_id) / "players.csv", [(1, 1, 0, 0, 10, 10, 0.9)])
    assert client.get(f"/api/matches/{match_id}/players?start=0&end=5000").status_code == 422
    assert 'id="show-players"' in client.get("/").text and 'id="minimap"' in client.get("/").text
```

- [ ] **Step 2:** run → failures.

- [ ] **Step 3: API** — in `app.py` (`from vball.players import load_players, with_court`), a small per-match cache keyed by file mtimes, and:

```python
    player_cache: dict[int, tuple[float, float, tuple]] = {}

    def players_for(match_id: int):
        csv_path, court_path = paths.players_csv(match_id), paths.court_json(match_id)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="no player tracks")
        key = (csv_path.stat().st_mtime, court_path.stat().st_mtime if court_path.exists() else 0.0)
        cached = player_cache.get(match_id)
        if cached is None or cached[:2] != key:
            players = load_players(csv_path)
            court = with_court(players, load_calibration(court_path)) if court_path.exists() else None
            player_cache[match_id] = (*key, (players, court))
        return player_cache[match_id][2]

    @app.get("/api/matches/{match_id}/players")
    def get_players(match_id: int, start: int, end: int, conn: sqlite3.Connection = Depends(db)) -> list[dict]:
        require_match(conn, match_id)
        if not 0 <= start < end or end - start > 1800:
            raise HTTPException(status_code=422, detail="window must be 1-1800 frames")
        players, court = players_for(match_id)
        rows = []
        for i in np.flatnonzero((players.frame >= start) & (players.frame < end)):
            row = {"frame": int(players.frame[i]), "id": int(players.track_id[i]),
                   "box": [round(float(v), 1) for v in players.box[i]], "court": None, "side": None}
            if court is not None:
                court_xy, side, on_court = court
                if on_court[i]:
                    row["court"] = [round(float(v), 3) for v in court_xy[i]]
                    row["side"] = int(side[i])
            rows.append(row)
        return rows
```
(`import numpy as np` at the top of `app.py`.)

- [ ] **Step 4: Viewer** — header `<label><input type="checkbox" id="show-players"> Show players</label>`; inside `.video-wrap` after the overlay canvas `<canvas id="minimap" class="minimap" width="120" height="220" hidden></canvas>`; CSS `.minimap { position: absolute; right: 8px; top: 8px; background: #0b3d2ecc; border-radius: 4px; pointer-events: none; }`. JS:

```javascript
// ---------- players ----------

let playerChunks = new Map(); // chunk start frame -> rows
const CHUNK = 300;

async function playersAround(frame) {
  const start = Math.max(0, Math.floor(frame / CHUNK) * CHUNK);
  for (const s of [start, start + CHUNK]) {
    if (!playerChunks.has(s)) {
      playerChunks.set(s, null);
      getJson(`/api/matches/${matchId}/players?start=${s}&end=${s + CHUNK}`)
        .then((rows) => playerChunks.set(s, rows))
        .catch(() => playerChunks.set(s, []));
    }
  }
  return (playerChunks.get(start) || []).filter((p) => p.frame === frame);
}

const SIDE_COLORS = ["#60a5fa", "#f87171"]; // near, far

function drawPlayers(ctx, frame, r) {
  const mini = $("#minimap");
  mini.hidden = !($("#show-players").checked && court);
  if (!$("#show-players").checked) return;
  playersAround(frame).then(() => {}); // prefetch
  const rows = (playerChunks.get(Math.floor(frame / CHUNK) * CHUNK) || []).filter((p) => p.frame === frame);
  ctx.lineWidth = 2;
  ctx.font = "12px system-ui";
  for (const p of rows) {
    const [x1, y1, x2, y2] = p.box;
    ctx.strokeStyle = p.side === null ? "rgba(200,200,200,0.5)" : SIDE_COLORS[p.side];
    ctx.strokeRect(r.x + x1 * r.scale, r.y + y1 * r.scale, (x2 - x1) * r.scale, (y2 - y1) * r.scale);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillText(String(p.id), r.x + x1 * r.scale, r.y + y1 * r.scale - 3);
  }
  if (!court) return;
  const m = mini.getContext("2d");
  const sx = mini.width / 11, sy = mini.height / 22; // court 9 x 18 m plus 1 m margin each way
  m.clearRect(0, 0, mini.width, mini.height);
  m.strokeStyle = "#e5e7eb";
  m.strokeRect(sx, sy, 9 * sx, 18 * sy);
  m.beginPath();
  m.moveTo(sx, 10 * sy); m.lineTo(10 * sx, 10 * sy); // net (far side drawn at the top)
  m.stroke();
  for (const p of rows) {
    if (!p.court) continue;
    m.fillStyle = SIDE_COLORS[p.side];
    m.beginPath();
    m.arc((p.court[0] + 1) * sx, (19 - p.court[1]) * sy, 4, 0, 2 * Math.PI);
    m.fill();
  }
}
```
Call `drawPlayers(ctx, frame, r)` in `drawOverlay` after `drawCourt`; in `selectMatch` reset `playerChunks = new Map();`; `$("#show-players").onchange = () => drawOverlay();`.

- [ ] **Step 5:** `uv run --no-sync pytest -q` → all pass.
- [ ] **Step 6: Commit** `feat: players API, player boxes and mini map overlay`.

---

### Task 8: Calibrate, choose the player backend, run everything (needs the user)

- [ ] **Step 1: User calibrates** all 11 sets in the viewer (restart the server first; ~1 min each, longer save while camera segments are computed). Record mean errors; any > 10 px → recalibrate on a clearer frame.

- [ ] **Step 2: Time both backends on a 2-minute excerpt** of Kent 2 (cut with `ffmpeg -ss 600 -t 120 -c copy`, process it as a throwaway match via `VBALL_DATA=data/exp/excerpt`), `--backend yolo` and `--backend ravel`. If RAVEL-VB on CPU would take > 60 min for a full set, compare on excerpts only.

- [ ] **Step 3: Compare on held-out sets** — run both backends (output to `data/exp/players_<backend>_m<id>.csv`) on Kent 2 and Brunel away 3 (+ Kent 1), then `vball playereval <id> --players <csv>` for each; watch rallies with "Show players" on. Pick the backend with the better full-sides share and fewer ids per side, unless the visual check shows a clear problem.

- [ ] **Step 4: Run the chosen backend on all 11 sets** (`vball players <id> --backend <b>`, background, split into ≤ 2 h batches).

- [ ] **Step 5: Results** — `docs/results/phase2b-court-players.md`: calibration errors per set, camera segments found (vs the phase-correlation shifts measured during planning: Kent 1 +119 px, Kent 2 +34/+20, Brunel away 1 −11/−13, Brunel away 3 −7/−45), player backend comparison table (full-sides share, ids per side, runtime), decision, known failure modes (from watching).

- [ ] **Step 6: Commit and push**; then use superpowers:finishing-a-development-branch.
