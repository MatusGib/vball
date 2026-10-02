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
