"""Volleyball court model and pixel <-> court-metre homographies.

Court coordinates: x across the court 0..9 m (left -> right as seen from the camera), y along it 0..18 m
from the near (camera-side) baseline; the net is at y = 9."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

COURT_WIDTH, COURT_LENGTH, NET_Y = 9.0, 18.0, 9.0
NET_HEIGHT_M = 2.43  # men's indoor; 2.24 for women
LANDMARKS = {
    "near_left_corner": (0.0, 0.0),
    "near_right_corner": (9.0, 0.0),
    "near_attack_left": (0.0, 6.0),
    "near_attack_right": (9.0, 6.0),
    "center_left": (0.0, 9.0),
    "center_right": (9.0, 9.0),
    "far_attack_left": (0.0, 12.0),
    "far_attack_right": (9.0, 12.0),
    "far_left_corner": (0.0, 18.0),
    "far_right_corner": (9.0, 18.0),
}
# top of the net tape at each sideline (court x, y); the height is Calibration.net_height_m.
# Not floor points: stored for 3D work, never used in the floor homography.
NET_LANDMARKS = {"net_left_top": (0.0, NET_Y), "net_right_top": (COURT_WIDTH, NET_Y)}
COURT_LINES = [
    ((0.0, 0.0), (0.0, 18.0)),
    ((9.0, 0.0), (9.0, 18.0)),
    ((0.0, 0.0), (9.0, 0.0)),
    ((0.0, 18.0), (9.0, 18.0)),
    ((0.0, 9.0), (9.0, 9.0)),
    ((0.0, 6.0), (9.0, 6.0)),
    ((0.0, 12.0), (9.0, 12.0)),
]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return homog[:, :2] / homog[:, 2:3]


def fit_homography(image_pts: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    """Least-squares image -> court homography from >= 4 correspondences."""
    image_pts = np.asarray(image_pts, dtype=np.float64)
    court_pts = np.asarray(court_pts, dtype=np.float64)
    if len(image_pts) < 4:
        raise ValueError("need at least 4 landmarks")
    H, _ = cv2.findHomography(image_pts, court_pts, 0)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("landmarks are degenerate (e.g. all on one line)")
    return H


def reprojection_errors(H: np.ndarray, image_pts: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    """Pixel distance between each clicked point and its landmark projected back into the image."""
    back = apply_h(np.linalg.inv(H), court_pts)
    return np.linalg.norm(back - np.asarray(image_pts, dtype=np.float64), axis=1)


@dataclass
class Segment:
    start_frame: int
    end_frame: int  # exclusive
    ref_to_frame: np.ndarray  # image homography: reference-frame pixels -> pixels of frames in this segment


@dataclass
class Calibration:
    ref_frame: int
    points: list[dict]  # {"landmark", "x", "y"} in work-video pixels at ref_frame (floor landmarks only)
    image_to_court: np.ndarray  # at ref_frame
    segments: list[Segment] = field(default_factory=list)
    net_points: list[dict] = field(default_factory=list)  # NET_LANDMARKS clicked at ref_frame
    net_height_m: float = NET_HEIGHT_M
    copied_from: int | None = None  # a draft carried over from another set's calibration (vball courtcopy)

    @classmethod
    def create(cls, ref_frame: int, points: list[dict], segments: list[Segment]) -> "Calibration":
        floor = [p for p in points if p["landmark"] in LANDMARKS]
        net = [p for p in points if p["landmark"] in NET_LANDMARKS]
        image = np.array([[p["x"], p["y"]] for p in floor])
        court = np.array([LANDMARKS[p["landmark"]] for p in floor])
        return cls(ref_frame, floor, fit_homography(image, court), segments, net)

    def errors(self) -> np.ndarray:
        image = np.array([[p["x"], p["y"]] for p in self.points])
        court = np.array([LANDMARKS[p["landmark"]] for p in self.points])
        return reprojection_errors(self.image_to_court, image, court)

    def segment_at(self, frame: int) -> Segment:
        for seg in self.segments:
            if seg.start_frame <= frame < seg.end_frame:
                return seg
        if not self.segments:
            return Segment(0, 1 << 62, np.eye(3))
        return self.segments[-1] if frame >= self.segments[-1].start_frame else self.segments[0]

    def image_to_court_at(self, frame: int) -> np.ndarray:
        return self.image_to_court @ np.linalg.inv(self.segment_at(frame).ref_to_frame)


def shifted_copy(
    cal: Calibration, shift: tuple[float, float], ref_frame: int, segments: list[Segment], copied_from: int
) -> Calibration:
    """Another set's calibration moved by a picture shift (same tripod spot, camera turned slightly): a draft."""
    dx, dy = shift
    moved = [{**p, "x": p["x"] + dx, "y": p["y"] + dy} for p in cal.points + cal.net_points]
    new = Calibration.create(ref_frame, moved, segments)
    new.net_height_m, new.copied_from = cal.net_height_m, copied_from
    return new


def save_calibration(path: Path, cal: Calibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ref_frame": cal.ref_frame,
        "points": cal.points,
        "image_to_court": cal.image_to_court.tolist(),
        "segments": [
            {"start_frame": s.start_frame, "end_frame": s.end_frame, "ref_to_frame": s.ref_to_frame.tolist()}
            for s in cal.segments
        ],
        "net_points": cal.net_points,
        "net_height_m": cal.net_height_m,
        "copied_from": cal.copied_from,
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def load_calibration(path: Path) -> Calibration:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [Segment(s["start_frame"], s["end_frame"], np.array(s["ref_to_frame"])) for s in data["segments"]]
    return Calibration(
        data["ref_frame"], data["points"], np.array(data["image_to_court"]), segments,
        data.get("net_points", []), data.get("net_height_m", NET_HEIGHT_M), data.get("copied_from"),
    )


def calibration_json(cal: Calibration) -> dict:
    """What the viewer needs: clicked points with their errors and a court -> image matrix per segment."""
    errors = cal.errors()

    def net_image(seg: Segment) -> list:
        out = []
        for name in NET_LANDMARKS:
            p = next((q for q in cal.net_points if q["landmark"] == name), None)
            if p is None:
                out.append(None)
            else:
                out.append([round(float(v), 1) for v in apply_h(seg.ref_to_frame, [[p["x"], p["y"]]])[0]])
        return out

    return {
        "ref_frame": cal.ref_frame,
        "points": [{**p, "error_px": round(float(e), 1)} for p, e in zip(cal.points, errors)],
        "mean_error_px": round(float(errors.mean()), 2),
        "segments": [
            {
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "court_to_image": np.linalg.inv(cal.image_to_court_at(s.start_frame)).tolist(),
                "net_image": net_image(s),
            }
            for s in (cal.segments or [Segment(0, 1 << 62, np.eye(3))])
        ],
        "lines": COURT_LINES,
        "landmarks": LANDMARKS,
        "net_points": cal.net_points,
        "net_height_m": cal.net_height_m,
        "net_landmarks": NET_LANDMARKS,
        "copied_from": cal.copied_from,
    }
