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
