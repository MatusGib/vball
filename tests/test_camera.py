import cv2
import numpy as np

from vball.camera import camera_segments, displacement, split_segments


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


def shift(dx, dy=0.0):
    return np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1.0]])


def test_noise_and_single_outliers_do_not_split():
    rng = np.random.default_rng(0)
    samples = [(f, shift(7 + rng.uniform(-2, 2), rng.uniform(-2, 2))) for f in range(0, 3000, 60)]
    samples[20] = (samples[20][0], shift(47, 12))  # one bad estimate
    segments = split_segments(samples, n_frames=3000, width=1920, height=1080, threshold_px=6.0)
    assert len(segments) == 1
    assert abs(segments[0].ref_to_frame[0, 2] - 7) < 1.0  # median, not the outlier


def test_persistent_move_opens_a_segment_with_median_transform():
    samples = [(f, shift(0)) for f in range(0, 1200, 60)] + [(f, shift(119, 1)) for f in range(1200, 3000, 60)]
    segments = split_segments(samples, n_frames=3000, width=1920, height=1080, threshold_px=6.0)
    assert [(s.start_frame, s.end_frame) for s in segments] == [(0, 1200), (1200, 3000)]
    assert abs(segments[1].ref_to_frame[0, 2] - 119) < 1e-9
