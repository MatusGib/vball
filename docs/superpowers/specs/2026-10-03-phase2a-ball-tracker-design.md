# Phase 2a: ball tracker improvement — design

Date: 2026-10-03
Parent spec: `2026-10-02-vball-design.md` (phase 2, first sub-project)
Decisions (user): ball tracker first; labels mostly automatic; ~15 min of manual clicking for an honest test set.

## Problem

Phase 1 measured the volleyball-fine-tuned TrackNetV3 (beach weights) on indoor footage: the ball is detected in only ~50 % of labelled rally frames (Kent sets 1 and 2), but almost never in dead time (1.2 %). Touch detection, action tagging and ball metrics in later phases all need the ball in far more rally frames.

## Goal and success criteria

Raise ball recall in rally frames without new false detections, measured on hand-clicked test frames from **held-out** footage:

| Metric (TrackNet convention, tolerance 4 px at 512×288 ≈ 15 px at 1080p) | Baseline | Target |
|---|---|---|
| Recall on test frames where the ball is visible | measured in Task 9 | +15 points absolute |
| Precision | measured | must not drop by more than 3 points |
| Rally F1, Kent sets 1 and 2 (phase 1 metric) | 0.94 / 0.89 | no worse than −0.02 after re-tuning rally params |

Held-out data: **Kent set 2** (home gym) and **Brunel away set 3** (other gym, 60 fps). Neither is used for training.

## Approach

Ordered cheapest first; each step is measured before the next.

1. **Measure the baseline** on the clicked test frames.
2. **Threshold sweep (no training):** TrackNet reports a ball where its heatmap > 0.5. Expose the threshold (vendored patch 3) and sweep 0.5 → 0.2. If a lower threshold already meets the target, it becomes the default.
3. **Self-training fine-tune** on automatically produced labels (pseudo-labels) from the other ~10 landscape sets.

## Pseudo-labels

For each training set, from the current ball track and its rally intervals (hand labels where they exist, else detected rallies):

- **Clean:** drop isolated detections (no other detection within ±2 frames) and detections that jump more than `max_step` from both neighbours (false positives).
- **Fill gaps:** a gap of ≤ 6 frames with ≥ 3 detections on each side (within a 12-frame reach) is filled from a quadratic fit of x(t) and y(t) through those detections, only if the ball is in free flight across the gap: fit RMS ≤ 3 px at 512 width, the same horizontal speed on both sides (±1.5 px/frame; no horizontal force acts in flight), and a path that does not bend upward (gravity can only bend it down). The residual alone is not enough: a parabola fits a touch's V shape within ~2 px. Perspective makes the quadratic approximate, hence short gaps only. These filled frames are the new positives the model currently misses.
- **Per-frame state:** inside rallies, visible → **positive** (x, y); inside rallies, not visible → **ignored** (never teach "no ball" where the model may simply have failed); outside rallies by more than 2 s → **negative** (keeps the tracker silent in dead time, which rally detection relies on).

## Training

- Own small loop (upstream `train.py` expects PNG-per-frame folders and cannot initialise from a checkpoint). Reuses the vendored `TrackNet` model; input built exactly like upstream inference (median background + 8 RGB frames at 512×288, /255), checked by a parity test against `Video_IterableDataset.__process__`.
- Loss: upstream WBCE, computed per frame and masked by the ignored frames.
- Windows: 8 consecutive frames; positive windows (≥ 1 positive frame) and negative windows (all negative) sampled 4:1, default 2,000 windows, cached once as a uint8 memmap (~7 GB) so epochs don't re-decode video.
- Adam, lr 1e-4, fp16 autocast, batch 2 with gradient accumulation 4 (fits 4 GB; measured in Task 10), 4 epochs. Initial weights: the beach checkpoint. Output saved in upstream checkpoint format so `predict.py` loads it unchanged.

## Ball test set (manual, ~15 min)

New web page `/ball.html` ("Ball check"). For a match it samples 100 frames uniformly from rally time (seeded, reproducible), shows each full-resolution frame, and the user clicks the ball centre, presses **X** for "no ball visible", or **K** to skip "can't tell". Q/W peek two frames back/forward (motion helps find a blurred ball); the click always applies to the target frame. The model's prediction is never shown (no anchoring). Saved per match to `data/matches/<id>/ball_test.csv` (`frame,status,x,y`).

## New commands

| Command | Purpose |
|---|---|
| `vball track <id> [--weights] [--threshold] [--out]` | re-run ball tracking on an existing match (default output `ball.csv` + rally re-detection) |
| `vball balleval <id> [--pred CSV]` | precision / recall / accuracy against the clicked test frames |
| `vball finetune --matches ... --out` | build pseudo-label cache and fine-tune |
| `vball tunerallies <ids...>` | grid-search rally params against hand labels (phase 1's ad-hoc search made permanent; ball changes shift detection density) |

`run_tracknet` keeps the 512×288 copy as `data/matches/<id>/track.mp4` (reused by re-tracking and training).

## Data

Process the remaining landscape sets first (~3 h in the background): Kent 3–4, Brunel home 1–4 (the `_2` / 30 fps files of 12 Dec), Brunel away 1–3 (the 4 Dec files; sets 2–3 are 60 fps). Royal Holloway (portrait) stays out until letterboxing exists.

## Risks

| Risk | Mitigation |
|---|---|
| Self-training only reinforces what the model already sees | Gap filling adds new positives; measured honestly on clicked held-out frames; threshold sweep as a no-training fallback |
| Wrong pseudo-labels (false positives inside rallies) | Cleaning step; positives only inside rallies; low learning rate, few epochs |
| Fine-tune breaks dead-time silence → rally detection worse | Dead-time negatives in training; rally F1 re-checked and params re-tuned with `vball tunerallies` |
| 60 fps sets behave differently | Brunel away set 3 (60 fps) is one of the two held-out test sets |
| 4 GB VRAM | batch 2 + fp16 + accumulation; measured before the full run |
