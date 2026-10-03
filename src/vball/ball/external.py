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
    match_dir = clip.video.parent.parent
    small = cache_dir / f"{match_dir.parent.name}_{match_dir.name}_{clip.video.stem}.mp4"
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
