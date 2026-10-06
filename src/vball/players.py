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


def team_sides(track_id: np.ndarray, side: np.ndarray, on_court: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One side per track (its side in most of its on-court frames), and a track on court in at least half of its
    frames is a player in all of them: someone stepping over the net line or chasing the ball off court for a moment
    keeps their team."""
    ids, inverse = np.unique(track_id, return_inverse=True)
    frames = np.bincount(inverse, minlength=len(ids))
    on = np.bincount(inverse, weights=on_court.astype(float), minlength=len(ids))
    far = np.bincount(inverse, weights=(on_court & (side == 1)).astype(float), minlength=len(ids))
    far_all = np.bincount(inverse, weights=(side == 1).astype(float), minlength=len(ids))
    team = np.where(on > 0, far > on / 2, far_all > frames / 2).astype(int)  # never on court: where it mostly is
    player = (on > 0) & (on >= frames / 2)
    return team[inverse], player[inverse]


def feet_to_court(frame: np.ndarray, box: np.ndarray, cal: Calibration) -> np.ndarray:
    """Court (x, y) in metres of the bottom-centre of each box (feet), using the camera segment of its frame."""
    feet = np.stack([(box[:, 0] + box[:, 2]) / 2, box[:, 3]], axis=1)
    court_xy = np.zeros_like(feet)
    segments = cal.segments or [Segment(0, 1 << 62, np.eye(3))]
    for i, seg in enumerate(segments):
        # same rule as Calibration.segment_at: frames before the first / after the last segment use those
        start = -np.inf if i == 0 else seg.start_frame
        end = np.inf if i == len(segments) - 1 else seg.end_frame
        rows = (frame >= start) & (frame < end)
        if rows.any():
            court_xy[rows] = apply_h(cal.image_to_court @ np.linalg.inv(seg.ref_to_frame), feet[rows])
    return court_xy


def with_court(players: Players, cal: Calibration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Court (x, y) of each player's feet (bottom-centre of the box), side (0 near, 1 far) and on-court flag, both
    per track rather than per frame (team_sides)."""
    court_xy = feet_to_court(players.frame, players.box, cal)
    side = (court_xy[:, 1] >= NET_Y).astype(int)
    on_court = (
        (court_xy[:, 0] >= ON_COURT_X[0]) & (court_xy[:, 0] <= ON_COURT_X[1])
        & (court_xy[:, 1] >= ON_COURT_Y[0]) & (court_xy[:, 1] <= ON_COURT_Y[1])
    )
    side, on_court = team_sides(players.track_id, side, on_court)
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
