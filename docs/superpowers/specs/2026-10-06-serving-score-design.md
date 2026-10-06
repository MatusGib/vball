# Serving end, score and serve outcomes (design, 2026-10-06)

Evidence: `docs/results/phase2e-player-stats-feasibility.md`. Order agreed with the owner: (1) automatic serving end
with a one-key fix in the viewer, (2) score, side-out %, aces and errors on the Stats page, (3) per-player serve stats
from the rotation. The owner's rule for serve in/out: a serve the receivers played was in; when the rally ends right
after the serve, the next server says who won it (server's team serves again = ace, otherwise a serve error).

## Serving end (per rally, `vball serving <id>` → `serving.csv`)

Window: rally start − 1.0 s … + 1.5 s. Players' feet (bottom-centre of the box, court homography), x within the
sidelines ± 1.5 m.

1. **Contact:** the first player row behind a baseline (feet y < −0.3 m or > 18.3 m) whose reach box holds the ball,
   and the ball then leaves them (the serve check's rule, `vball.serve`). Their end serves; that frame is the serve time.
2. **Count** (no contact found): the end with more player rows behind its baseline in the window. Tie: unknown.

Serve time without a contact: the rally start. Measured on Kent 3: 34 / 36 rallies right; fails on the 0.5 m-high
Brunel away camera, hence the fix key.

**Receive seen:** the VolleyVision action detector (`models/volleyvision/actions_yv8m.pt`, classes block, defence, serve,
set, spike) runs on ~15 frames/s from serve + 0.4 s to serve + 3.5 s; a "defense" box with confidence ≥ 0.4 whose feet are
on the receiving half means the receivers played the serve. Detections are cached in `actions.csv`. It sees the far team
(facing the camera) much better than the near team.

## Outcomes (computed when read, so fixes apply at once)

- Winner of rally *i* = serving end of rally *i + 1*. Last rally: the side for which the final score is a legal set end
  (≥ 25 and 2 ahead, or ≥ 15 and 2 ahead), when exactly one side qualifies; otherwise unknown.
- Gap = rally end − serve time.
  - gap ≤ 2.5 s: the rally ended on the serve → **ace** if the server's end won, **serve error** if not.
  - gap ≥ 4.5 s, or receive seen: **in** (played).
  - otherwise **check**.
- Fixes (viewer keys) override the serving end and the outcome. Stored in `meta.json` as
  `serve_fix: {"<rally start, 0.1 s>": {"end": "near"|"far", "outcome": "ace"|"error"|"in"}}`, matched to rallies
  by start time (±0.5 s) so they survive small label edits.

## Team numbers (Stats page, picked match)

Running score (near–far), final score, points won on own serve, side-out % (rallies won when receiving / rallies
received), serves, aces, serve errors, serves in %. "Us/them" from `our_side`. Warnings: rallies not approved (detected
rallies add false points), rallies still marked *check* or with an unknown end, `serving.csv` older than the rallies.

## Per-player serves (rotation)

`meta.json` `lineup`: our players in serving order, starting with our first server of the set (names or shirt
numbers, typed on the Stats page). Each time our team wins the serve back, the next player serves. Per player: serves,
aces, errors, in %, and the plausible 3D serve speeds of their serves (top and median). Substitutions are not modelled.

## Viewer

A serve line under the status bar for the rally at the playhead: "Serve: near end (us) · ace · 12–9 · Mateusz".
Keys: **V** flips the serving end of the rally at the playhead, **O** cycles its outcome (ace → error → in → auto).
Both save to `meta.json` at once.

## API

- `GET /api/matches/{id}/serving` → per rally (start, end, serve time, serving end auto / fixed, how, gap, receive,
  outcome auto / fixed, winner, score after, server name) and the team summary; 404 with the command to run when
  `serving.csv` is missing.
- `PUT /api/matches/{id}/meta` merges the fields it is given (`our_side`, `lineup`, `serve_fix`).
- `/api/matches/{id}/stats` gains `serving` (summary + per player) when available.

## Tests

Pure functions (`vball.serving`): serving end from contact / count, winner chain and last-rally inference, outcome
bands, fixes by start time, rotation, side-out maths. Web: meta merge, serving endpoint with a fixture `serving.csv`.
