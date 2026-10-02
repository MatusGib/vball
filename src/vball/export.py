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
