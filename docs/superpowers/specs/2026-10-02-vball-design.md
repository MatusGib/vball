# vball: personal Balltime-style volleyball analysis — design

Date: 2026-10-02
Status: approved approach (A: modular CV pipeline + volleyball rules)

## Goal

A local tool that takes a phone recording of an indoor volleyball match and produces what
Balltime's "Volleyball AI" produces: rally-cut video, every touch tagged with its action
(serve, pass/reception, set, attack, block, dig), the touch attributed to a player of my
team, per-player/team stats, and ball metrics (trajectory, serve speed, attack/jump height).
Viewed in a local web app.

## Constraints (from the user)

| Topic | Decision |
|---|---|
| Footage | Indoor, single phone on a tripod, behind the baseline, raised (stands / 2–4 m) |
| Teams | Identify players of **my team only**; opponents are "opponent" |
| Labelling | **Minimal**. Ground truth only for evaluation (e.g. one set of rally boundaries) |
| Stats | Start with counts + outcomes; data model must allow quality grades later |
| Interface | Local web app (FastAPI backend + browser frontend) |
| Hardware | Windows 11, NVIDIA RTX 3050 Laptop **4 GB VRAM**, Python via uv |
| Use | Personal, non-commercial |

Hardware note: the user expected 8 GB+, the machine reports 4 GB. Inference of all planned
models fits in 4 GB with small batches / fp16. Fine-tuning TrackNet locally is possible at
batch 2–4; heavier training goes to Colab/RunPod if needed.

## Non-goals (for now)

- Uploading video through the browser (processing is started from the CLI).
- Moving/broadcast cameras, beach volleyball.
- Identifying opponent players by number.
- Multi-user accounts, cloud hosting.

## Architecture

```
match.mp4
 ├─ 1. Ingest        ffmpeg → work.mp4 (constant fps ≤60, ≤1080p, keyframe every 1 s, faststart)
 ├─ 2. Court setup   (phase 2) web UI: click 4 court corners + net-top → homography
 ├─ 3. Perception    GPU, run once per video, cached as files in data/matches/<id>/
 │     ball     TrackNetV3 (volleyball fine-tune) → ball.csv (Frame,Visibility,X,Y)
 │     players  (phase 2) person detector + multi-object tracker → tracks
 │     pose     (phase 2) pose estimator on players near the ball
 ├─ 4. Game logic    CPU, pure Python functions over cached perception
 │     rallies → touches → actions → team & player → rally outcome
 ├─ 5. Metrics       (phase 4) homography + pose → speeds, heights, landing zones
 ├─ 6. Store         SQLite (data/vball.db) + per-match folder
 └─ 7. Web app       FastAPI JSON API + static HTML/JS frontend, video served with HTTP Range
```

Principles:

1. **Expensive GPU work runs once and is cached on disk.** Game-logic stages read caches and
   re-run in seconds, so tuning a threshold never re-runs a model.
2. **Each stage is a module with a typed input and output** (e.g. `BallTrack → list[Rally]`)
   and is unit-tested on synthetic data plus evaluated on a small hand-labelled real sample.
3. **Rules first, learning later.** Volleyball's structure (serve → pass → set → attack, max 3
   touches per side, net separates teams) replaces most labelled data. Where rules are not
   enough, the rules' own outputs (corrected by the user while watching) become training data.

## Stage design

### 1. Ingest
`ffmpeg` transcodes the source into `work.mp4`: constant frame rate `min(round(src_fps), 60)`,
height `min(src_height, 1080)`, H.264 (`h264_nvenc` by default, `libx264` fallback), GOP = fps
(keyframe every second, needed for fast browser seeking and stream-copy clip export), AAC
audio kept (later: whistle / contact sounds). All frame indices in the system refer to
`work.mp4`. Phone VFR video is therefore normalised once.

### 3a. Ball tracking
Vendored upstream TrackNetV3 (MIT, commit `6eda442`) run as a subprocess with
`--large_video --eval_mode nonoverlap` (streaming, 8× faster than the sliding ensemble).
Weights: `deadfast/beach-volley-vision-models` `tracknet_best.pt` (MIT, beach volleyball
fine-tune) as the starting point. Two small patches to the vendored code: `torch.load(...,
weights_only=False)` (torch ≥2.6 default changed) and a guard for an empty final frame window.
InpaintNet is not used (its weights are badminton-only).
If indoor accuracy is insufficient (phase 2 decision), fine-tune on the user's gym with
model-assisted labelling (predict → user corrects a few hundred frames → retrain).

### 4a. Rally detection (phase 1)
Input: ball track, fps, frame size. Heuristic, all thresholds in `RallyParams`:
1. Ball speed between consecutive visible frames, in frame-diagonals per second
   (resolution-independent).
2. A frame is "in flight" if `min_speed ≤ speed ≤ max_speed` (slow = held/rolling ball,
   too fast = detector jumping between false positives).
3. Centered moving average of "in flight" over `window_s`; active where ≥ `min_active_frac`.
4. Active runs merged across gaps ≤ `max_gap_s` (ball briefly out of frame / occluded),
   runs shorter than `min_rally_s` dropped, padded by `pre_pad_s` / `post_pad_s`.
Known weakness: balls thrown back to the server between rallies and warm-ups on adjacent
courts. If evaluation shows these dominate errors, phase 2 adds a court-region mask
(from court calibration) and/or a small game-state classifier trained on the rally cuts the
user has already accepted.

### 4b. Touches and actions (phase 2)
Touch = local extremum of ball direction change / speed change, gated by proximity of a
tracked player's wrist/hand (pose) or body box to the ball. Side of net from homography
(ball/player court y vs net line). Action from rally grammar per side:
first touch of rally = serve; first touch on receiving side = reception (after serve) or dig
(after attack); a touch at/above net height directly after the opponent's attack = block;
the touch that sends the ball over the net = attack (or free ball if it was not jumped/hard);
a middle touch = set. Outcome: last touch + ball landing side (in/out from homography) or
next rally's server side.

### 4c. Team and player identity (phase 3)
Team: k-means (k=2, + referee/other) on torso colour histograms of player tracks, seeded by
side of net in set 1; teams swap ends every set so colour, not side, is the identity.
Player: jersey-number recognition (scene-text model on torso crops of my team's tracks),
majority vote over the whole track, constrained to the roster the user confirms once per
match in the UI (~2 min). Tracks that break are re-linked by number + colour.

### 5. Metrics (phase 4)
Court homography (ground plane) + known net height give ball position on court at contact and
landing. Serve speed = court distance between serve contact and first landing/next touch over
time. Attack/jump height from pose ankles/hips vs. standing baseline. Heat maps of landing
zones per player.

### Optional: AI second opinion (phase 4+)
For touches the rules label with low confidence, send 3 cropped frames to a vision LLM.
Estimated cost ≈ $0.10–0.50 per match (vs ≈ $1–55 to label whole rallies by LLM). Off by
default; footage of minors must stay local unless the user opts in.

## Data model

Phase 1 tables:

```sql
matches(id, name, source_path, fps, n_frames, width, height, created_at)
rallies(id, match_id → matches, idx, start_frame, end_frame)   -- [start, end)
```

Later phases add (designed now so stats can grow without migration pain):

```sql
players(id, team_name, number, name)
touches(id, rally_id, frame, side, action, player_id NULL, confidence,
        ball_x, ball_y, court_x NULL, court_y NULL, next_touch_id NULL, quality NULL)
rally_outcome(rally_id, winner_side, reason)
```

Per-match folder `data/matches/<id>/`: `work.mp4`, `ball.csv`, later `players.parquet`,
`pose.parquet`, `court.json`, `clips/`.

## Web app

Phase 1: match picker, rally list, video player that plays rallies back-to-back
("rallies only" mode), keyboard N/P next/previous rally, and a label mode (S = rally start,
E = rally end, U = undo, CSV download, autosaved in localStorage) used to create evaluation
ground truth. Later: court-corner click tool, roster confirmation, filters by player/action,
stats tables, highlight export.

## Evaluation and testing

- Unit tests (pytest) for every pure function on synthetic inputs; ffmpeg-backed tests use
  tiny generated `testsrc` videos.
- Per-stage evaluation scripts against small hand labels:
  - Rallies: precision/recall/F1 at temporal IoU ≥ 0.5, mean start/end error, % dead time
    removed. Ground truth: one set (~45 rallies) labelled in the web app.
  - Touches (phase 2): recall/precision within ±0.2 s on ~20 rallies.
  - Identity (phase 3): % of my-team touches with correct player.

## Roadmap and acceptance criteria

| Phase | Delivers | Done when |
|---|---|---|
| 1 | Ingest, ball tracking, rally detection, condensed export, viewer + GT labelling | On one real match: rally P ≥ 0.85 and R ≥ 0.90 (IoU 0.5) on one labelled set; 90-min match processes in < 2 h on the RTX 3050 |
| 2 | Court calibration, player detection/tracking, pose, touch detection, action grammar | Touch R ≥ 0.80 within ±0.2 s and action accuracy ≥ 0.80 on 20 labelled rallies |
| 3 | Team colour clustering, jersey OCR + roster, per-player clips and count stats | ≥ 80 % of my-team touches attributed to the right player |
| 4 | Ball metrics, heat maps, rotation tracking, quality grades, optional LLM second opinion | Metrics plausible on spot checks (serve speeds 30–90 km/h, etc.) |

Each phase gets its own implementation plan written after the previous phase is evaluated.

## Risks

| Risk | Mitigation |
|---|---|
| Beach-trained ball model misses the ball indoors (gym lights, busy background) | Measure in phase 1; model-assisted fine-tune on own footage |
| Far-court ball is ~10–15 px | Keep ≥1080p work video; TrackNet uses motion over 8 frames |
| Dead-ball handling between rallies looks like play | Speed thresholds now; court mask / classifier later |
| 4 GB VRAM | Small batches, fp16, nonoverlap inference; cloud for training |
| Jersey numbers unreadable from behind | Track-level voting, roster constraint, user correction in UI |

## Licences of reused work

TrackNetV3 (MIT), beach-volley-vision weights (MIT). Later candidates: Ultralytics YOLO
(AGPL-3.0 — fine for personal use, never distribute), RF-DETR (Apache-2.0), RTMPose/rtmlib
(Apache-2.0). VolleyVision data is CC BY-NC-ND (personal training use only).
