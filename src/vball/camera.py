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


MIN_RESPONSE = 0.05  # phase-correlation peak below this = unreliable sample


def estimate_shift(ref_gray: np.ndarray, gray: np.ndarray, window: np.ndarray) -> np.ndarray | None:
    """Ref -> frame image translation (3x3) from phase correlation of the whole picture, or None if weak.

    Camera bumps are modelled as pure shifts: on our gyms, ORB feature matching locks onto repeated wall
    panels and lights and flips between two answers, while whole-picture phase correlation stays steady."""
    (dx, dy), response = cv2.phaseCorrelate(np.float32(ref_gray), np.float32(gray), window)
    if response < MIN_RESPONSE:
        return None
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def split_segments(
    samples: list[tuple[int, np.ndarray]], n_frames: int, width: int, height: int, threshold_px: float = 10.0
) -> list[Segment]:
    """Group sampled ref -> frame transforms into camera segments (default threshold: 10 px at 1080p,
    i.e. a few centimetres on court; smaller wobbles are absorbed into the segment median).

    A new segment starts only when two consecutive samples both differ from the current segment by more
    than threshold_px and agree with each other (so single bad estimates are ignored). Each segment's
    transform is the element-wise median of its samples."""

    def differs(a: np.ndarray, b: np.ndarray) -> bool:
        return displacement(np.linalg.inv(a) @ b, width, height) > threshold_px

    if not samples:
        return [Segment(0, n_frames, np.eye(3))]
    groups: list[list[tuple[int, np.ndarray]]] = [[samples[0]]]
    anchor = samples[0][1]
    i = 1
    while i < len(samples):
        frame, H = samples[i]
        moved = differs(anchor, H)
        confirmed = moved and i + 1 < len(samples) and differs(anchor, samples[i + 1][1]) and not differs(
            H, samples[i + 1][1]
        )
        if confirmed:
            groups.append([samples[i]])
            anchor = np.median(np.stack([H, samples[i + 1][1]]), axis=0)
        elif not moved:
            groups[-1].append(samples[i])
            anchor = np.median(np.stack([s[1] for s in groups[-1]]), axis=0)
        i += 1  # an unconfirmed outlier is simply dropped

    segments = []
    for k, group in enumerate(groups):
        start = 0 if k == 0 else group[0][0]
        end = groups[k + 1][0][0] if k + 1 < len(groups) else n_frames
        segments.append(Segment(start, end, np.median(np.stack([s[1] for s in group]), axis=0)))
    return segments


def camera_segments(
    video: Path, ref_frame: int, n_frames: int, scale: float, step: int = 60, threshold_px: float = 10.0
) -> list[Segment]:
    """Sample every `step` frames of the (512x288) video, estimate ref -> frame shifts and split them
    into camera segments (see split_segments). Transforms and threshold are in work-video pixels, i.e.
    after scaling the 512x288 estimates by `scale`."""
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
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    S = np.diag([scale, scale, 1.0])
    S_inv = np.linalg.inv(S)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    samples: list[tuple[int, np.ndarray]] = []
    i = 0
    while i < n_frames and cap.grab():
        if i % step == 0:
            H = estimate_shift(ref_gray, cv2.cvtColor(cap.retrieve()[1], cv2.COLOR_BGR2GRAY), window)
            if H is not None:
                samples.append((i, S @ H @ S_inv))
        i += 1
    cap.release()
    return split_segments(samples, n_frames, round(w * scale), round(h * scale), threshold_px)
