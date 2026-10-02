import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vball.ball.track import BallTrack

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
