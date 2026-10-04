"""3D ball flights from one camera: drag physics, flight segmentation, robust multi-seed fit, metrics
(docs/superpowers/specs/2026-10-04-ball3d-design.md). Court metres, z up."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from vball.ball.track import BallTrack
from vball.camera3d import Camera3D
from vball.court import NET_Y

G = 9.81
DRAG_K = 0.036  # 1/m: rho * Cd * A / (2 m) for a volleyball (Cd 0.47, 0.27 kg, 21 cm)
SUBSTEPS = 2  # RK4 steps per frame
NOISE_PX = 6.0  # tracker jitter per image axis at 1080p
SEED_HEIGHTS = (1.0, 2.5, 4.0)
SEED_DISTANCES = (6.0, 12.0, 20.0)  # metres along the ray, when a height plane is not reachable
LOWER = np.array([-6.0, -6.0, 0.0, -40.0, -40.0, -40.0])  # p0 (court +- 6 m, 0-6 m up), v0 (m/s)
UPPER = np.array([15.0, 24.0, 6.0, 40.0, 40.0, 40.0])


def simulate(p0, v0, n: int, fps: float, k: float = DRAG_K, return_velocity: bool = False):
    """Positions at frames 0..n-1 from start state (p0, v0): (n, 3), or (B, n, 3) for (B, 3) inputs.
    With return_velocity, also the velocities in the same shape."""
    batched = np.ndim(p0) == 2
    p = np.atleast_2d(np.asarray(p0, dtype=np.float64)).copy()
    v = np.atleast_2d(np.asarray(v0, dtype=np.float64)).copy()
    gravity = np.array([0.0, 0.0, -G])

    def acc(vel):
        return gravity - k * np.linalg.norm(vel, axis=1, keepdims=True) * vel

    dt = 1.0 / (fps * SUBSTEPS)
    pos, vels = np.empty((len(p), n, 3)), np.empty((len(p), n, 3))
    for i in range(n):
        pos[:, i], vels[:, i] = p, v
        for _ in range(SUBSTEPS):
            a1 = acc(v)
            a2 = acc(v + 0.5 * dt * a1)
            a3 = acc(v + 0.5 * dt * a2)
            a4 = acc(v + dt * a3)
            p = p + dt * (6 * v + dt * (a1 + a2 + a3)) / 6
            v = v + dt * (a1 + 2 * a2 + 2 * a3 + a4) / 6
    if not batched:
        pos, vels = pos[0], vels[0]
    return (pos, vels) if return_velocity else pos


def flight_metrics(p0, v0, n: int, fps: float, extend_s: float = 0.5, k: float = DRAG_K) -> dict:
    """Start speed, apex over the n observed frames, and the net crossing / landing within n frames + extend_s."""
    traj = simulate(p0, v0, n + round(extend_s * fps), fps, k)
    m = {"speed_kmh": float(np.linalg.norm(v0) * 3.6), "apex_m": float(traj[:n, 2].max()),
         "net_z_m": None, "land_x": None, "land_y": None}
    dy = traj[:, 1] - NET_Y
    cross = np.flatnonzero(dy[:-1] * dy[1:] <= 0)
    if len(cross):
        i = int(cross[0])
        a = dy[i] / (dy[i] - dy[i + 1]) if dy[i] != dy[i + 1] else 0.0
        m["net_z_m"] = float(traj[i, 2] + a * (traj[i + 1, 2] - traj[i, 2]))
    below = np.flatnonzero(traj[:, 2] <= 0.0)
    if len(below) and below[0] > 0:
        i = int(below[0]) - 1
        a = traj[i, 2] / (traj[i, 2] - traj[i + 1, 2])
        land = traj[i] + a * (traj[i + 1] - traj[i])
        m["land_x"], m["land_y"] = float(land[0]), float(land[1])
    return m


def split_flights(
    track: BallTrack, start: int, end: int, fps: float,
    max_gap_s: float = 0.5, jump_px: float = 25.0, min_obs: int = 8, window: int = 6,
) -> list[tuple[int, int]]:
    """Cut a rally's visible ball positions into flights: at gaps over max_gap_s, and where a quadratic through
    the last `window` points misses the next point by more than jump_px (a touch). Returns (first, last + 1)."""
    pieces, cur = [], []
    for f in (start + np.flatnonzero(track.visible[start:end])).tolist():
        if cur and f - cur[-1] > max_gap_s * fps:
            pieces.append(cur)
            cur = []
        elif len(cur) >= window:
            last = np.array(cur[-window:])
            px = np.polyval(np.polyfit(last - f, track.x[last], 2), 0.0)
            py = np.polyval(np.polyfit(last - f, track.y[last], 2), 0.0)
            if math.hypot(px - track.x[f], py - track.y[f]) > jump_px:
                pieces.append(cur)
                cur = []
        cur.append(f)
    pieces.append(cur)
    return [(p[0], p[-1] + 1) for p in pieces if len(p) >= min_obs]
