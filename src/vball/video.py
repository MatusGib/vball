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
