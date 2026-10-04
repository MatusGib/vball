"""Pinhole camera fitted to the court calibration: floor landmarks plus the optional net-tape tops.

Court coordinates in metres: x across (0..9), y along (near baseline 0, far 18), z up. Square pixels, principal
point at the image centre, no lens distortion."""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

from vball.court import LANDMARKS, NET_LANDMARKS, Calibration, apply_h


@dataclass
class Camera3D:
    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # 3x3 rotation, court -> camera
    t: np.ndarray  # (3,) translation, court -> camera
    cal: Calibration  # clicked points and camera-bump segments

    def project_ref(self, pts: np.ndarray) -> np.ndarray:
        """Court metres (N, 3) -> pixels (N, 2) in the reference frame (before camera bumps)."""
        cam = np.asarray(pts, dtype=np.float64).reshape(-1, 3) @ self.R.T + self.t
        uv = cam @ self.K.T
        return uv[:, :2] / uv[:, 2:3]

    def project(self, pts: np.ndarray, frame: int) -> np.ndarray:
        return apply_h(self.cal.segment_at(frame).ref_to_frame, self.project_ref(pts))

    def to_ref(self, uv: np.ndarray, frame: int) -> np.ndarray:
        """Pixels seen in `frame` -> where they would be in the reference frame."""
        return apply_h(np.linalg.inv(self.cal.segment_at(frame).ref_to_frame), uv)

    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    def rays(self, uv_ref: np.ndarray) -> np.ndarray:
        """Unit directions (N, 3) in court metres through reference-frame pixels."""
        uv_ref = np.asarray(uv_ref, dtype=np.float64).reshape(-1, 2)
        d = np.hstack([uv_ref, np.ones((len(uv_ref), 1))]) @ np.linalg.inv(self.K).T @ self.R
        return d / np.linalg.norm(d, axis=1, keepdims=True)


def clicked_points(cal: Calibration) -> tuple[np.ndarray, np.ndarray, int]:
    """Court metres (N, 3) and pixels (N, 2) of every clicked point; the first n_floor are on the floor."""
    obj = [(*LANDMARKS[p["landmark"]], 0.0) for p in cal.points]
    obj += [(*NET_LANDMARKS[p["landmark"]], cal.net_height_m) for p in cal.net_points]
    img = [(p["x"], p["y"]) for p in cal.points + cal.net_points]
    return np.array(obj, dtype=np.float64), np.array(img, dtype=np.float64), len(cal.points)


def _pose(obj, img, n_floor, f, width, height):
    K = np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])
    _, rvec, tvec = cv2.solvePnP(obj[:n_floor], img[:n_floor], K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if len(obj) > n_floor:  # refine with the net points, starting from the floor-only pose
        _, rvec, tvec = cv2.solvePnP(obj, img, K, None, rvec, tvec, useExtrinsicGuess=True)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    return K, rvec, tvec, np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)


def fit_camera(cal: Calibration, width: int, height: int) -> tuple[Camera3D, np.ndarray]:
    """Focal length by a bounded 1-D search, pose by solvePnP. Points clicked outside the picture (guessed
    off-screen corners) are left out of the fit. Returns the camera and every clicked point's reprojection error
    in pixels (floor points first, then net points)."""
    obj, img, n_floor = clicked_points(cal)
    inside = (img[:, 0] >= 0) & (img[:, 0] < width) & (img[:, 1] >= 0) & (img[:, 1] < height)
    used_obj, used_img, used_floor = obj[inside], img[inside], int(inside[:n_floor].sum())

    def rms(log_f: float) -> float:
        err = _pose(used_obj, used_img, used_floor, math.exp(log_f), width, height)[3]
        return float(np.sqrt(np.mean(err**2)))

    best = minimize_scalar(rms, bounds=(math.log(0.3 * width), math.log(5.0 * width)), method="bounded")
    K, rvec, tvec, _ = _pose(used_obj, used_img, used_floor, math.exp(best.x), width, height)
    cam = Camera3D(K, cv2.Rodrigues(rvec)[0], tvec.reshape(3), cal)
    return cam, np.linalg.norm(cam.project_ref(obj) - img, axis=1)
