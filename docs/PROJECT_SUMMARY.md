# vball: project summary

A personal clone of Balltime's "Volleyball AI": turn a phone recording of an indoor volleyball match into rally clips, tagged touches and player stats. This page explains what exists, why it was built this way, and what the measurements show. Details live in the linked documents.

Last updated: 2026-10-06.

## Where things stand

| Phase | What | Status |
|---|---|---|
| 1 | Rally cutting: ingest, ball tracking, rally detection, viewer, export | Done, merged |
| 2a | Better ball tracker (fine-tuned on our own footage) | Done, merged |
| 2b part 1 | Ball overlay in the viewer, external ball models, ball v2 | Done (branch `phase2b`) |
| 2b part 2 | Court + net calibration (with magnifier), player detection + tracking (YOLO11 chosen), opt-in serve check | Done (branch `phase2b`); players on all 11 sets |
| 2d | 3D ball flights from one camera: speed, height, net crossing, landing; *Show speed* overlay | Experiment: serves and sets usable, attacks rough ([phase2d-ball3d.md](results/phase2d-ball3d.md)) |
| Stats | Stats page: hardest serves, highest sets, fastest attacks (experimental), longest rallies, side heat maps; rows open the viewer at the moment | Done (branch `phase2b`) |
| UI | Hall Floor redesign (light + dark), Sources page | Done (branch `phase2b`) |
| Score | Serving end per rally → rally winner, score, side-out %, aces / serve errors (owner's rule), per-player serves from the serving order; viewer keys V / O to fix | Done (branch `phase2b`); feasibility in [phase2e](results/phase2e-player-stats-feasibility.md) |
| 2c | Touch detection and action tagging (serve, pass, set, attack, block, dig) | Planned |
| 3 | Team and player identity (jersey numbers), per-player stats | Planned |
| 4 | Per-player metrics, quality grades | Planned (ball metrics and side heat maps started in 2d / Stats) |

Data processed: 11 sets from 3 matches (Kent, Brunel home, Brunel away), 4.4 hours of video, 491 rallies detected. 175 automated tests.

## The setup and the constraints that shaped it

| Constraint | Consequence |
|---|---|
| Indoor matches, one phone on a tripod behind the baseline, roughly standing height | Most public datasets (broadcast side view) are a poor fit; the camera is static, so a court can be calibrated once per video |
| Minimal hand labelling | Rely on volleyball rules and automatic labels; hand labels only for honest evaluation (rally boundaries for 2 sets, 300 clicked ball frames) |
| Identify **my team** only | Jersey reading only needed for one team (phase 3) |
| Laptop: RTX 3050 with **4 GB** VRAM, Windows | Small batches, fp16, pre-scaled inputs; ~13–15 min to process a 23-min set |
| Local web app | FastAPI + plain HTML/JS viewer at `http://127.0.0.1:8000` |
| Personal use | Non-commercial licences (VballNet data, AGPL Ultralytics) are acceptable |

## Architecture

```
match.mp4 → ingest (ffmpeg, constant fps, ≤1080p)       → work.mp4
          → 512×288 copy                                  → track.mp4
          → ball tracker (TrackNetV3, fine-tuned)         → ball.csv
          → rally rules (ball-in-flight heuristic)        → SQLite rallies
          → web viewer: rallies, labels, Ball check, Show ball overlay
```

Principles: expensive GPU work runs once and is cached as files; every later stage is a fast pure-Python step that can be re-tuned in seconds; each stage is measured against a small honest test set before moving on.

Main commands: `vball process`, `serve`, `eval`, `export`, `redetect`, `track`, `balleval`, `finetune`, `tunerallies` (see README).

## Decisions and why

| Decision | Why | Alternatives considered |
|---|---|---|
| Approach A: chain of models + volleyball rules | Works with little labelling, each part testable, runs locally | End-to-end touch spotter (needs thousands of labels); sending video to an LLM (imprecise timing, cost, privacy) |
| TrackNetV3 for the ball, started from beach-volleyball weights | Proven small-fast-ball tracker; MIT; volleyball weights available | YOLO-style detectors (no motion cue) |
| Feed TrackNet a pre-scaled 512×288 video | 1080p decode + resize in Python ran at 2.5 frames/s; pre-scaling gave ~92 frames/s | Batch tuning alone (VRAM-limited) |
| Rally detection as a tunable heuristic, tuned by grid search on labels | Seconds to re-tune on cached tracks; parameters land on a broad plateau, not a knife edge | Learned classifier (needs more labels) |
| Self-training the ball tracker on automatic labels (2a) | No manual labelling; gap filling uses physics (same horizontal speed, gravity bends down) so touches aren't bridged | Manual correction tool (1–2 h of clicking) |
| Fixed adoption rules **before** training | Prevents talking ourselves into a model after seeing results | — |
| Keep ball tracker v1 as default after 2b | v2 missed the pre-set precision rule; v1 still best for cutting rallies | Adopt v2 or an external model now |
| Labels saved on the server, with an approve step | Labels must not be lost; copied detections must be visibly reviewed before they count | Browser-only storage (first version) |

## Results

### Rally cutting (phase 1, re-tuned in 2a)

| Set | Precision | Recall | F1 |
|---|---|---|---|
| Kent 1 (tuned on) | 0.93 | 0.90 | 0.92 |
| Kent 2 (held out) | 0.88 | 0.93 | 0.90 |

A 23-minute set becomes ~6 minutes of play (about 75 % dead time removed). Remaining errors: very short rallies (aces, 1–3 s) and the ball being thrown back to the server between points. Details: [phase1-rallies.md](results/phase1-rallies.md).

### Ball tracking (2a, 2b) on 200 clicked frames from two held-out sets

| Tracker | Kent 2 precision / recall | Brunel away 3 (unseen gym, 60 fps) | Mean F1 |
|---|---|---|---|
| Beach weights (start) | 0.64 / 0.49 | 0.84 / 0.32 | — |
| **ours v1 @ 0.3 (current default)** | 0.67 / 0.73 | 0.87 / 0.66 | 0.72 |
| ours v2 @ 0.3 (+ VballNet labels) | 0.65 / 0.90 | 0.83 / 0.84 | 0.79 |
| VballNetV4c (external, MIT) | 0.84 / 0.94 | 0.83 / 0.69 | 0.82 |
| VballNetGridV3 (external, MIT) | 0.89 / 0.80 | 0.97 / 0.68 | 0.82 |

Self-training roughly doubled recall in the unseen gym (0.32 → 0.66). Adding the VballNet hand labels (v2) helped further, but lost a little precision, so it did not pass its pre-set rule. External VballNet models are as good or better overall; none wins everywhere. The ball source for touch detection will be chosen in phase 2c by measuring touch accuracy with each candidate. Details: [phase2a-ball.md](results/phase2a-ball.md), [phase2b-ball.md](results/phase2b-ball.md).

## External sources used

From [references/external_data.md](references/external_data.md):

| Source | Licence | Used for |
|---|---|---|
| TrackNetV3 (vendored, 3 small patches) | MIT | Ball tracker architecture |
| deadfast beach-volley-vision weights | MIT | Starting weights |
| VballNet dataset (29k hand-labelled ball frames, indoor amateur) | not stated, personal use | Training ball v2 |
| fast-volleyball-tracking-inference (VballNet ONNX models, RAVEL-VB) | MIT | Ball benchmark; RAVEL-VB tested for players, not chosen |
| Ultralytics YOLO11 + ByteTrack | AGPL-3.0 | Player detection and tracking |
| VolleyVision YOLOv8m action detector | AGPL-3.0 (README: CC BY-NC-ND) | Did the receivers play the serve (`vball serving`) |
| VREN rally notation | not stated | Reference rates: 16% of points are serve errors, 5% aces |

Not usable: Court-Keypoint-Detection (no weights, data or licence); single-image Roboflow ball sets (the tracker needs consecutive frames). Broadcast-view action datasets are kept for phase 2c.

## Known limitations

- Very short rallies and between-point ball returns still confuse rally detection (court calibration in 2b should help).
- The camera was bumped in 5 of 11 sets (up to 119 px); calibration must follow these moves (handled in 2b part 2).
- Royal Holloway videos are portrait and not processed yet (need letterboxing).
- Fine-tuned model files are not in git (130 MB limit); rebuild commands are in `src/vball/config.py` and the README.
- The web server must be restarted after backend changes; the viewer now says so when it can't load data.

## Next

Per-player stats beyond serves need to know who touched the ball: jersey numbers on the near side look readable
(phase 3), touch attribution at flight splits (passform's rule), and fewer player id switches (SportsMOT, needs the
owner's CodaLab sign-up). Serve-outcome checks and approved rallies for the detected sets make the scores exact.

## Document map

- Overall design: [specs/2026-10-02-vball-design.md](superpowers/specs/2026-10-02-vball-design.md)
- Phase designs: [2a](superpowers/specs/2026-10-03-phase2a-ball-tracker-design.md), [2b](superpowers/specs/2026-10-03-phase2b-design.md)
- Plans: [docs/superpowers/plans/](superpowers/plans/)
- Results: [docs/results/](results/)
- Initial research: [research/volleyball-ai-sources.md](research/volleyball-ai-sources.md)
