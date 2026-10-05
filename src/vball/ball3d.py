"""3D ball flights from one camera: drag physics, flight segmentation, robust multi-seed fit, metrics
(docs/superpowers/specs/2026-10-04-ball3d-design.md). Court metres, z up."""

import csv
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


def _sse(t: np.ndarray, v: np.ndarray, deg: int) -> float:
    return float(((np.polyval(np.polyfit(t, v, deg), t) - v) ** 2).sum())


def _refine(track: BallTrack, frames: np.ndarray, gain: float, noise_px: float, min_side: int = 6) -> list:
    """Cut where two quadratics (per image axis) fit much better than one cubic, recursively: touches too soft for
    the one-step test, like the hit at the top of a serve toss (docs/results/phase2d-ball3d.md)."""
    if len(frames) < 2 * min_side:
        return [frames]
    t, x, y = frames - frames[0], track.x[frames], track.y[frames]
    single = _sse(t, x, 3) + _sse(t, y, 3)
    best, k_best = np.inf, 0
    for k in range(min_side, len(frames) - min_side + 1):
        two = _sse(t[:k], x[:k], 2) + _sse(t[:k], y[:k], 2) + _sse(t[k:], x[k:], 2) + _sse(t[k:], y[k:], 2)
        if two < best:
            best, k_best = two, k
    if single - best < gain * noise_px**2:
        return [frames]
    return _refine(track, frames[:k_best], gain, noise_px) + _refine(track, frames[k_best:], gain, noise_px)


def split_flights(
    track: BallTrack, start: int, end: int, fps: float,
    max_gap_s: float = 0.5, jump_px: float = 25.0, min_obs: int = 8, window: int = 6,
    split_gain: float = 30.0, noise_px: float = NOISE_PX,
) -> list[tuple[int, int]]:
    """Cut a rally's visible ball positions into flights: at gaps over max_gap_s, where a quadratic through the
    last `window` points misses the next point by more than jump_px (a touch), then by _refine. Returns
    (first, last + 1)."""
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
    refined = [r for p in pieces if p for r in _refine(track, np.array(p), split_gain, noise_px)]
    return [(int(p[0]), int(p[-1]) + 1) for p in refined if len(p) >= min_obs]


@dataclass
class Flight:
    start_frame: int  # first observed frame; p0, v0 are the state there
    end_frame: int  # last observed frame + 1
    n_obs: int
    p0: np.ndarray
    v0: np.ndarray
    rms_px: float  # over inliers
    inlier_share: float


def _pixels(cam: Camera3D, pts: np.ndarray) -> np.ndarray:
    """Like cam.project_ref, but points behind or at the camera are pushed to 0.5 m in front (finite residuals)."""
    c = pts @ cam.R.T + cam.t
    z = np.maximum(c[:, 2:3], 0.5)
    return c[:, :2] / z * [cam.K[0, 0], cam.K[1, 1]] + cam.K[:2, 2]


def _free_residuals(x, cam, t, uv_ref):
    pts = x[:3] + np.outer(t, x[3:]) + np.outer(0.5 * t**2, [0.0, 0.0, -G])
    return (_pixels(cam, pts) - uv_ref).ravel()


def _drag_residuals(x, cam, idx, n, fps, uv_ref):
    return (_pixels(cam, simulate(x[:3], x[3:], n, fps)[idx]) - uv_ref).ravel()


def _drag_jacobian(x, cam, idx, n, fps, uv_ref):
    """Forward differences, all 7 trajectories integrated in one batch."""
    steps = 1e-4 * np.maximum(1.0, np.abs(x))
    X = np.tile(x, (7, 1))
    X[1:] += np.diag(steps)
    traj = simulate(X[:, :3], X[:, 3:], n, fps)
    r = [(_pixels(cam, traj[b, idx]) - uv_ref).ravel() for b in range(7)]
    return np.stack([(r[j + 1] - r[0]) / steps[j] for j in range(6)], axis=1)


def _seed_points(cam: Camera3D, ray: np.ndarray) -> list[np.ndarray]:
    c = cam.centre()
    pts = [c + (h - c[2]) / ray[2] * ray for h in SEED_HEIGHTS if abs(ray[2]) > 1e-6 and (h - c[2]) / ray[2] > 0]
    return pts if len(pts) >= 2 else pts + [c + d * ray for d in SEED_DISTANCES]


def _seeds(cam: Camera3D, uv_ref: np.ndarray, duration: float) -> list[np.ndarray]:
    first, last = cam.rays(uv_ref[[0, -1]])
    seeds = []
    for a in _seed_points(cam, first):
        for b in _seed_points(cam, last):
            v = (b - a) / duration + [0.0, 0.0, 0.5 * G * duration]
            seeds.append(np.clip(np.concatenate([a, v]), LOWER + 1e-6, UPPER - 1e-6))
    return seeds


def fit_flight(cam: Camera3D, frames, uv, fps: float, noise_px: float = NOISE_PX) -> Flight | None:
    """Fit a drag-affected 3D arc to tracked pixels (uv in each frame's own pixels). None if the arc does not
    explain at least 80% of the observations within 3 * noise_px."""
    frames = np.asarray(frames, dtype=int)
    uv = np.asarray(uv, dtype=np.float64)
    uv_ref = np.vstack([cam.to_ref(uv[i : i + 1], int(f)) for i, f in enumerate(frames)])
    idx = frames - frames[0]
    t = idx / fps
    seeds = _seeds(cam, uv_ref, max(t[-1], 1.0 / fps))
    fits = [
        least_squares(_free_residuals, s, bounds=(LOWER, UPPER), loss="soft_l1", f_scale=noise_px,
                      args=(cam, t, uv_ref))
        for s in seeds
    ]
    best = min(fits, key=lambda r: r.cost)
    res = least_squares(
        _drag_residuals, best.x, jac=_drag_jacobian, bounds=(LOWER, UPPER), loss="soft_l1", f_scale=noise_px,
        args=(cam, idx, int(idx[-1]) + 1, fps, uv_ref),
    )
    if np.any(np.isclose(res.x, LOWER, atol=0.05) | np.isclose(res.x, UPPER, atol=0.05)):
        return None  # pinned to a limit: the data did not decide the arc
    err = np.linalg.norm(res.fun.reshape(-1, 2), axis=1)
    inliers = err <= 3 * noise_px
    if inliers.mean() < 0.8:
        return None
    return Flight(int(frames[0]), int(frames[-1]) + 1, len(frames), res.x[:3], res.x[3:],
                  float(np.sqrt(np.mean(err[inliers] ** 2))), float(inliers.mean()))


STATE_COLUMNS = ["p0_x", "p0_y", "p0_z", "v0_x", "v0_y", "v0_z"]  # fitted state at start_frame's first sighting
FLIGHT_COLUMNS = ["rally", "start_frame", "end_frame", "n_obs", "first", "fitted", "rms_px",
                  "speed_kmh", "apex_m", "net_z_m", "land_x", "land_y", *STATE_COLUMNS]


def match_flights(cam: Camera3D, track: BallTrack, rallies, fps: float, noise_px: float = NOISE_PX) -> list[dict]:
    """Every flight in every rally (frame intervals), fitted where possible."""
    rows = []
    for r, (s, e) in enumerate(rallies):
        for i, (a, b) in enumerate(split_flights(track, s, e, fps)):
            frames = a + np.flatnonzero(track.visible[a:b])
            uv = np.stack([track.x[frames], track.y[frames]], axis=1)
            fit = fit_flight(cam, frames, uv, fps, noise_px)
            row = {"rally": r, "start_frame": a, "end_frame": b, "n_obs": len(frames), "first": i == 0,
                   "fitted": fit is not None}
            if fit is not None:
                row["rms_px"] = fit.rms_px
                row.update(flight_metrics(fit.p0, fit.v0, fit.end_frame - fit.start_frame, fps))
                row.update(zip(STATE_COLUMNS, [*fit.p0.tolist(), *fit.v0.tolist()]))
            rows.append(row)
    return rows


def save_flights(path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, FLIGHT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})


def load_flights(path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k in ("first", "fitted"):
                    row[k] = v == "True"
                elif k in ("rally", "start_frame", "end_frame", "n_obs"):
                    row[k] = int(v)
                else:
                    row[k] = float(v) if v else None
            rows.append(row)
    return rows


def flight_track(cam: Camera3D, row: dict, fps: float) -> dict:
    """A fitted flight frame by frame for the viewer: image position, speed (km/h) and height (m)."""
    n = row["end_frame"] - row["start_frame"]
    state = [row[k] for k in STATE_COLUMNS]
    pos, vel = simulate(state[:3], state[3:], n, fps, return_velocity=True)
    uv = np.vstack([cam.project(pos[i : i + 1], row["start_frame"] + i) for i in range(n)])
    return {
        "start": row["start_frame"], "end": row["end_frame"], "rally": row["rally"], "first": row["first"],
        "uv": np.round(uv, 1).tolist(),
        "speed_kmh": np.round(np.linalg.norm(vel, axis=1) * 3.6, 1).tolist(),
        "z_m": np.round(pos[:, 2], 2).tolist(),
        "net_z_m": row["net_z_m"],
    }


def summary(rows: list[dict]) -> str:
    fitted = [r for r in rows if r["fitted"]]
    serves = [r for r in fitted if r["first"]]
    lines = [f"flights {len(rows)}, fitted {len(fitted)} ({len(fitted) / max(1, len(rows)):.0%}); "
             f"first flights fitted {len(serves)} of {sum(r['first'] for r in rows)}"]
    if serves:
        speed = np.array([r["speed_kmh"] for r in serves])
        net = [r["net_z_m"] for r in serves if r["net_z_m"] is not None]
        land = [(r["land_x"], r["land_y"]) for r in serves if r["land_x"] is not None]
        near_court = sum(-2 <= x <= 11 and -2 <= y <= 20 for x, y in land)
        lines.append(f"serve speed km/h: p10 {np.percentile(speed, 10):.0f} median {np.median(speed):.0f} "
                     f"p90 {np.percentile(speed, 90):.0f} | in 40-100: {np.mean((speed >= 40) & (speed <= 100)):.0%}")
        lines.append(f"serve net crossing: {len(net)} crossing, above 2.43 m: "
                     f"{np.mean(np.array(net) > 2.43) if net else 0:.0%} | landings {len(land)}, "
                     f"within 2 m of the court: {near_court / max(1, len(land)):.0%}")
    return "\n".join(lines)
