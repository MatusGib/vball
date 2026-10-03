# Serve check and net calibration — design

Date: 2026-10-04. Branch: `phase2b` (part 3, after the player-backend decision of part 2, Task 8).

## Problem

The rally detector (`src/vball/rallies.py`) treats any sustained ball movement as a rally. On the two
labelled sets (Kent 1, Kent 2) it produces 8 false rallies and misses 7 real ones:

| Match | Pred | Labels | Matched | False rallies | Missed |
|---|---|---|---|---|---|
| Kent 1 (#1) | 41 | 42 | 38 | 3 | 4 |
| Kent 2 (#2) | 42 | 40 | 37 | 5 | 3 |

Five of the false rallies were inspected frame by frame (contact sheets): all five are the ball being
returned to the server between rallies (thrown, rolled, kicked or lobbed), three of them **over the net**.
All seven misses are short (1.3–3.4 s: aces, serve errors) and are cut by `min_rally_s = 1.5` or lost by the
tracker.

So "the serve crosses the net" (the user's first idea) would not reject the false rallies. What separates
them is direction: a serve **starts** at a player standing behind a baseline and leaves them; a return to the
server **ends** at that player.

## Goals

1. Reject ball movement that does not start with a serve, using the player tracks and court calibration we
   already have (no 3D).
2. Let the minimum rally length drop so short aces/serve errors are kept, now that false rallies are filtered.
3. Add two optional net-tape clicks to court calibration now, so 3D ball paths (later) need no recalibration.

Non-goals: 3D ball trajectories, ball speed in km/h, a 2D "ball crossed the net line" test (ambiguous with a
low camera: a high near-court ball also appears above the tape). These are noted as next steps.

## 1. Serve check — `src/vball/serve.py` (new)

Input: the ball track, player tracks with court positions (`players.with_court`), fps, and an active run
`(start, end)` in frames (before padding). Output: `True` if the run begins with a serve.

```python
@dataclass(frozen=True)
class ServeParams:
    search_before_s: float = 1.0   # look for the server from start - this ...
    search_after_s: float = 0.5    # ... to start + this
    baseline_margin_m: float = 0.5 # foot y < margin or > 18 - margin counts as behind a baseline
    side_margin_m: float = 1.5     # foot x within [-margin, 9 + margin]
    reach_w: float = 0.75          # box widened by this * box width each side
    reach_h: float = 1.5           # and by this * box height above the top (toss)
    leave_s: float = 0.7           # the ball must be away from the server this long after the contact
    leave_dist_h: float = 2.0      # "away" = farther than this * box height from the box centre
```

Rule, for the run's search window `[start - before, start + after]`:

1. **Server candidates**: player rows in the window whose foot point is on court sideways
   (`-side_margin <= x <= 9 + side_margin`) and behind a baseline (`y < baseline_margin` or
   `y > 18 - baseline_margin`).
2. **Contact**: the first visible ball position in the window that lies inside a candidate's reach box
   (the candidate's box in the same frame, widened by `reach_w` box widths each side and `reach_h` box heights
   above the top). Gives the contact frame `c` and the server's track id.
3. **Leaves**: the last visible ball position in `(c, c + leave_s]` is farther than `leave_dist_h` box
   heights from the server's box centre (the server's box at that frame if tracked, otherwise their box at
   `c`).

The run is a rally iff a contact exists whose ball then leaves. A return to the server fails step 3 (the ball
arrives and stays) or step 2 (the thrower is not behind a baseline).

The far half-court is ~60 px tall in the Kent image, so 1 px of foot error is ~0.15 m there; margins are
generous and step 3 carries most of the weight.

## 2. Wiring

- `detect_rallies(..., keep=None)`: optional `keep(start, end) -> bool` applied to each active run after the
  length filter and before padding. `None` = today's behaviour.
- `serve.serve_filter(track, players, court_xy, fps, params) -> Callable` builds `keep`.
- `pipeline.redetect` uses the filter when the match has both `players.csv` and `court.json`, otherwise prints
  `serve check skipped: no player tracks / court calibration` and behaves as before.
- `vball tunerallies` searches `min_rally_s` together with a small `ServeParams` grid when every given match has
  players and a court; it always prints the current defaults first.
- Prerequisite: the player backend chosen in part 2 Task 8 is run on the labelled sets and written to
  `players.csv`.

## 3. Net tape clicks

- `court.NET_LANDMARKS = {"net_left_top": (0.0, 9.0), "net_right_top": (9.0, 9.0)}` (court x, y; height is
  `net_height_m`, default `NET_HEIGHT_M = 2.43`, men's indoor).
- `Calibration` gains `net_points: list[dict]` (same shape as `points`) and `net_height_m: float`.
  `Calibration.create` splits clicked points by name; only floor landmarks feed the homography and the errors.
- `court.json` gains `net_points` and `net_height_m`; both optional on load, so existing files still load.
- `calibration_json` adds `net_height_m` and, per segment, `net_image`: the clicked net points moved into that
  segment with its `ref_to_frame` (or `null` per point not clicked).
- `PUT /court` accepts net landmarks; unknown names are still 422; fewer than 4 floor points is still 422.
- Viewer: the calibration list gets `net left top`, `net right top` after the floor points (Skip-able).
  *Show court* draws, per segment, a post from each centre-line end up to its net point and the tape line
  between the two. *Calibrate court* on a calibrated match seeks to the saved `ref_frame`, preloads the saved
  points and starts at the first missing landmark, so Kent 1/2 only need the two net clicks and *Save*.

## 4. Testing and success criteria

Unit tests (synthetic data): a serve from behind the near baseline is kept; one from behind the far baseline is
kept; a return to the server (ball arrives at a player behind the baseline) is rejected; a throw from mid-court
is rejected; no ball near any player is rejected; `detect_rallies` with `keep=None` is unchanged. Court: net
points round-trip through `court.json`, don't change the homography, old files load. Web: PUT with net points,
GET returns `net_image`.

Evaluation (`vball eval` after `vball redetect`):

- Kent 1 and Kent 2: at least 5 of the 8 false rallies removed, no currently matched rally lost, F1 above
  0.92 / 0.90.
- Held-out check: the user labels Brunel away 3 (#11) rallies; parameters are tuned on Kent 1–2 only and
  F1 on #11 must not drop versus no serve check.

Results go to `docs/results/phase2b-court-players.md` and `docs/PROJECT_SUMMARY.md`.

## Next step noted (not built)

Net clicks + floor homography give the full camera projection; fitting gravity-constrained 3D ball arcs to the
track then gives serve/attack speed in km/h and a true "went over the tape" test (cf. MonoTrack for badminton).
