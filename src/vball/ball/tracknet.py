import csv
import shutil
import subprocess
import sys
from pathlib import Path

from vball.config import TRACKNET_DIR, TRACKNET_THRESHOLD, TRACKNET_WEIGHTS
from vball.video import probe

# TrackNetV3 input size. Feeding a pre-scaled video avoids slow per-frame 1080p decode + resize in Python.
TRACKNET_W, TRACKNET_H = 512, 288


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


def tracknet_csv_path(out_dir: Path, video: Path) -> Path:
    return out_dir / f"{video.stem}_ball.csv"


def downscale_command(src: Path, dst: Path) -> list[str]:
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(src),
        "-vf", f"scale={TRACKNET_W}:{TRACKNET_H}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-g", "30",
        "-an", str(dst),
    ]


def rescale_tracknet_csv(src: Path, dst: Path, sx: float, sy: float) -> Path:
    """Copy a TrackNet CSV, scaling visible X/Y by (sx, sy)."""
    with open(src, newline="") as f_in, open(dst, "w", newline="") as f_out:
        writer = csv.writer(f_out, lineterminator="\n")
        writer.writerow(["Frame", "Visibility", "X", "Y"])
        for row in csv.DictReader(f_in):
            visible = int(row["Visibility"]) == 1
            x = round(float(row["X"]) * sx) if visible else 0
            y = round(float(row["Y"]) * sy) if visible else 0
            writer.writerow([row["Frame"], row["Visibility"], x, y])
    return dst


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
