# vball

Personal Balltime-style volleyball video analysis. Phase 1: rally cutting.

## Setup (Windows)

1. `winget install --id Gyan.FFmpeg -e`
2. Install [uv](https://docs.astral.sh/uv/), then in this folder: `uv sync --extra dev`
3. `uv run python scripts/download_models.py` (beach-volleyball TrackNet weights, the starting point)
4. The default ball tracker is fine-tuned on our own footage and is not in git. After processing some matches with `--weights models/tracknet_volleyball.pt`, rebuild it with
   `uv run vball finetune --matches <ids> --out models/tracknet_vball_v1.pt --rebuild-cache` (see `docs/results/phase2a-ball.md`).

## Use

```bash
uv run vball process "D:/Volleyball/match1.mp4" --name match1   # ingest + ball tracking + rallies
uv run vball serve                                              # http://127.0.0.1:8000
uv run vball export 1                                           # clips + rallies_only.mp4
uv run vball eval 1 data/gt/match1_set1.csv                     # accuracy vs hand labels
uv run vball redetect 1                                         # re-run rally rules after tuning
uv run vball track 1 --threshold 0.3                            # re-run ball tracking (any --weights)
uv run vball balleval 1                                         # ball precision/recall vs Ball check clicks
uv run vball tunerallies 1 2                                    # grid-search rally params on labelled matches
```

Ball check (`/ball.html`): click the ball in 100 sampled rally frames per match (X = no ball, K = can't tell) to score the ball tracker.

`process` encodes with `libx264` by default. With an NVIDIA driver >= 610, `--encoder h264_nvenc` is faster.

In the web app: N / P = next / previous rally, S / E = mark ground-truth rally start / end, U = undo; "Download GT CSV" saves the labels for `vball eval`.

Data lives in `data/` (override with `VBALL_DATA`). Design: `docs/superpowers/specs/2026-10-02-vball-design.md`.

Third-party: `third_party/TrackNetV3` (MIT), weights from `deadfast/beach-volley-vision-models` (MIT).
