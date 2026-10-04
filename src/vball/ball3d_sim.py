"""Simulated flights through a real camera, with the tracker's noise and dropouts, to measure how well each 3D
metric can be recovered (docs/superpowers/specs/2026-10-04-ball3d-design.md)."""

import math
from dataclasses import dataclass

import numpy as np

from vball.ball.track import BallTrack
from vball.ball3d import fit_flight, flight_metrics, simulate
from vball.camera3d import Camera3D
from vball.court import COURT_LENGTH, NET_Y

KINDS = ("serve", "attack", "set")
MAX_S = 2.5  # longest flight simulated


@dataclass
class SimFlight:
    kind: str
    p0: np.ndarray
    v0: np.ndarray
    n: int  # frames from the touch to the end of the flight


def _candidate(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.uniform
    if kind == "serve":
        p0, aim = np.array([u(0.5, 8.5), u(-3.0, -0.5), u(2.2, 3.2)]), (u(1.0, 8.0), u(12.0, 17.0))
        speed, elev = u(12, 28), u(-5, 20)
    elif kind == "attack":
        p0, aim = np.array([u(0.5, 8.5), u(6.0, 8.5), u(2.6, 3.4)]), (u(0.0, 9.0), u(10.0, 18.0))
        speed, elev = u(12, 25), u(-35, -5)
    else:  # set, along the net
        p0 = np.array([u(2.0, 7.0), u(6.0, 8.5), u(1.8, 2.4)])
        aim, speed, elev = (0.0 if rng.random() < 0.5 else 9.0, p0[1]), u(5, 9), u(50, 75)
    az, el = math.atan2(aim[0] - p0[0], aim[1] - p0[1]), math.radians(elev)
    return p0, speed * np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])


def _length(kind: str, z: np.ndarray) -> int:
    """Frames until the flight ends: landing (serve, attack) or falling back to hitting height (set)."""
    if kind == "set":
        apex = int(np.argmax(z))
        down = np.flatnonzero(z[apex:] < 2.5)
        return apex + int(down[0]) if len(down) else len(z)
    below = np.flatnonzero(z <= 0.0)
    return int(below[0]) if len(below) else len(z)


def _keep(kind: str, traj: np.ndarray, m: dict) -> bool:
    if kind == "set":
        return bool((traj[:, 1] < NET_Y).all() and (traj[:, 0] > -1.0).all() and (traj[:, 0] < 10.0).all())
    return (
        m["net_z_m"] is not None and m["net_z_m"] > (2.5 if kind == "serve" else 2.43)
        and m["land_y"] is not None and NET_Y < m["land_y"] <= COURT_LENGTH + 0.5 and -0.5 <= m["land_x"] <= 9.5
    )


def sample_flights(kind: str, count: int, fps: float, rng: np.random.Generator, far_share: float = 0.5):
    """Realistic flights of one kind; about far_share of them are mirrored to start at the far end."""
    out = []
    while len(out) < count:
        p0, v0 = _candidate(kind, rng)
        traj = simulate(p0, v0, round(MAX_S * fps), fps)
        n = _length(kind, traj[:, 2])
        if n < 8 or not _keep(kind, traj[:n], flight_metrics(p0, v0, n, fps)):
            continue
        if rng.random() < far_share:
            p0, v0 = p0 * [1, -1, 1] + [0.0, COURT_LENGTH, 0.0], v0 * [1, -1, 1]
        out.append(SimFlight(kind, p0, v0, n))
    return out


def _visibility(track: BallTrack, intervals, n: int, rng: np.random.Generator) -> np.ndarray:
    """The tracker's real hit/miss pattern: n consecutive frames from a random point inside a rally."""
    usable = [(s, e) for s, e in intervals if e - s > n]
    s, e = usable[rng.integers(len(usable))]
    a = int(rng.integers(s, e - n))
    return track.visible[a : a + n]


def _errors(cam, flight, fps, noise_px, visible, rng, width, height) -> dict | str:
    """Fit one simulated flight as the tracker would see it. 'unseen' / 'rejected', or the metric errors."""
    pos, vel = simulate(flight.p0, flight.v0, flight.n, fps, return_velocity=True)
    uv = cam.project(pos, cam.cal.ref_frame) + rng.normal(0.0, noise_px, (flight.n, 2))
    seen = visible[: flight.n] & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    idx = np.flatnonzero(seen)
    if len(idx) < 8:
        return "unseen"
    fit = fit_flight(cam, idx + cam.cal.ref_frame, uv[idx], fps, noise_px)
    if fit is None:
        return "rejected"
    span = int(idx[-1] - idx[0]) + 1
    truth = flight_metrics(pos[idx[0]], vel[idx[0]], span, fps)
    got = flight_metrics(fit.p0, fit.v0, span, fps)
    err = {"speed_kmh": abs(got["speed_kmh"] - truth["speed_kmh"]), "apex_m": abs(got["apex_m"] - truth["apex_m"])}
    if truth["net_z_m"] is not None and got["net_z_m"] is not None:
        err["net_z_m"] = abs(got["net_z_m"] - truth["net_z_m"])
    if truth["land_x"] is not None and got["land_x"] is not None:
        err["land_m"] = math.hypot(got["land_x"] - truth["land_x"], got["land_y"] - truth["land_y"])
    return err


def run_simulation(cam: Camera3D, track: BallTrack, intervals, fps: float, width: int, height: int,
                   count: int = 100, noise_levels=(3.0, 6.0, 9.0), seed: int = 0) -> list[dict]:
    """One row per (kind, noise): share fitted / rejected / unseen and median + 90th percentile errors."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind in KINDS:
        flights = sample_flights(kind, count, fps, rng)
        for noise in noise_levels:
            results = [_errors(cam, f, fps, noise, _visibility(track, intervals, f.n, rng), rng, width, height)
                       for f in flights]
            errs = [r for r in results if isinstance(r, dict)]
            row = {"kind": kind, "noise_px": noise, "flights": len(flights), "fitted": len(errs) / len(flights),
                   "rejected": results.count("rejected") / len(flights),
                   "unseen": results.count("unseen") / len(flights)}
            for key in ("speed_kmh", "apex_m", "net_z_m", "land_m"):
                vals = [e[key] for e in errs if key in e]
                row[f"{key}_n"] = len(vals)
                row[f"{key}_med"] = float(np.median(vals)) if vals else None
                row[f"{key}_p90"] = float(np.percentile(vals, 90)) if vals else None
            rows.append(row)
    return rows
