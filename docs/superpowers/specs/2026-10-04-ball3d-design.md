# 3D ball flights — design (experiment)

Date: 2026-10-04. Branch: `phase2b`. Decision wanted from it: which ball metrics are trustworthy from one
phone camera behind the baseline. Nothing here is user-facing until a metric earns it.

## Facts this design rests on

- Footage: 1920×1080, 30 fps (Brunel away 2–3: 60 fps). Camera low, behind the near baseline, looking down the
  court, so serves and most attacks travel nearly along the line of sight (the worst case for monocular depth).
- Ball track: TrackNet at 512×288 rescaled to 1080p. Against clicked frames (hits within 30 px): a constant
  offset of +1–2 px in x and +4–6 px in y, then 4–9 px standard deviation per axis (Kent 2 at 30 fps: ~9 px,
  Brunel away 3 at 60 fps: ~5 px).
- Calibration: 10 floor landmarks + (new) 2 net-tape tops at `net_height_m` (2.43 m), camera-bump segments as
  image shifts.
- Volleyball drag is not negligible: k = ρ·Cd·A / (2m) ≈ 1.2·0.47·0.0347 / (2·0.27) ≈ 0.036 m⁻¹, i.e. a 20 m/s
  ball decelerates ~14 m/s² — more than gravity.

## 1. Camera model — `src/vball/camera3d.py`

Pinhole camera with square pixels, principal point at the image centre, no lens distortion. Unknowns: focal
length f, rotation R, translation t (7). Fitted to every clicked point at `ref_frame` — floor points
`(x, y, 0)` and net points `(x, 9, net_height_m)` — by minimising reprojection error: for a given f,
`cv2.solvePnP` gives R, t; f is found by a bounded 1-D search (`scipy.optimize.minimize_scalar` on log f).
Without net points the same fit works on the floor points alone (planar case).

```python
@dataclass
class Camera3D:
    K: np.ndarray            # 3x3
    R: np.ndarray            # 3x3, court -> camera
    t: np.ndarray            # (3,)
    cal: Calibration         # for camera-bump segments
    def project(self, pts: np.ndarray, frame: int) -> np.ndarray      # (N,3) metres -> (N,2) px in that frame
    def centre(self) -> np.ndarray                                     # camera position in court metres
def fit_camera(cal: Calibration, width: int, height: int) -> tuple[Camera3D, np.ndarray]  # + per-point error px
```

`project` maps through the reference-frame camera, then the segment's `ref_to_frame` shift.

`vball camera <id>` prints f, the camera position (sanity: behind the near baseline, 1–3 m up), and the floor
and net reprojection errors.

## 2. Flights — `src/vball/ball3d.py`

**Physics.** State (position, velocity) in court metres, z up; `dv/dt = -g·ẑ - k·|v|·v`, integrated with RK4 at
4 sub-steps per frame (`simulate(p0, v0, n_frames, fps, k)`).

**Segmentation** (`split_flights(track, start, end, fps)`): inside a rally, visible frames are cut where the
ball is lost for more than 0.5 s, and where a short window fit breaks — a degree-2 polynomial per image axis
fitted to the last 6 points predicts the next point; an error over `max(25 px, 4σ)` starts a new flight (a
touch). Flights with fewer than 8 visible frames are dropped.

**Fit** (`fit_flight(cam, frames, uv, fps)`): unknowns p0, v0 at the flight's first frame (6). Residuals: the
projected simulated positions against the tracked pixels (the 2–6 px constant offset is left in: it is below
the jitter), `least_squares` with `soft_l1` loss (scale σ) and bounds (p0 within the court ± 6 m,
0 ≤ z0 ≤ 6 m, each velocity component within ±40 m/s). Monocular depth makes the fit multi-modal, so it starts
from 9 seeds: the first and last observation rays intersected with planes at heights {1, 2.5, 4} m. Seeds are
fitted with the closed-form drag-free arc (fast); the best is refined with drag. Accepted if at least 80% of
observations lie within `3σ` of the fitted arc (TrackNet has occasional wrong detections); `rms_px` is over
those inliers. σ defaults to 6 px.

**Metrics** per accepted flight: start speed (km/h), apex height, net-crossing height (z where y = 9, if the
flight crosses), landing point (x, y where z reaches 0, if it does within the flight + 0.5 s), flight time.

`vball ball3d <id>` writes `data/matches/<id>/flights.csv`
(`rally,start_frame,end_frame,n_obs,rms_px,speed_kmh,apex_m,net_z_m,land_x,land_y,first`) and prints
acceptance rates and metric distributions.

## 3. Judging it

**Simulation** (`src/vball/ball3d_sim.py`, `vball ball3dsim <id>`): with the match's fitted camera, generate
flights of three kinds, from both ends:

| Kind | Start | Speed | Elevation | Kept if |
|---|---|---|---|---|
| serve | behind a baseline, z 2.2–3.2 m | 12–28 m/s | −5…+20° | crosses the net above 2.5 m, lands in the far court |
| attack | 0.5–3 m from the net, z 2.6–3.4 m | 12–25 m/s | −35…−5° | crosses the net, lands in the far court |
| set | 0.5–3 m from the net, z 1.8–2.4 m | 5–9 m/s | 50–75°, along the net | stays on its side |

Project, add the measured offset-free noise (σ = 3, 6, 9 px per axis), drop frames using visibility masks cut
from the match's real rally frames, fit, and report per kind and σ: share of flights accepted and the median /
90th-percentile errors of start speed, apex, net-crossing height and landing point.

**Real data** (Kent 1, Kent 2, Brunel away 3 with net clicks): acceptance rate, and plausibility of the first
flight of each rally (serve): speed 40–100 km/h, net crossing above 2.43 m, landing within 2 m of the court.

**Verdict per metric** in `docs/results/phase2d-ball3d.md`: *usable*, *rough* or *not with one camera*, with
the simulated error that justifies it. A landing ground-truth page (user clicks floor contacts) is a follow-up
only if landing points look promising.

## Out of scope

Touch types (2c), viewer overlays of fitted arcs, landing maps, spin/float modelling.
