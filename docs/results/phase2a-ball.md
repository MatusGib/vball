# Phase 2a results: ball tracker

Status: done. Fine-tuned model adopted as the default (decision below).

## Clicked test frames (the honest test)

100 frames per set sampled uniformly from labelled/detected rally time, clicked without seeing any prediction: Kent set 2 (89 ball, 11 no ball), Brunel away set 3 (87 / 13), Kent set 1 (86 / 14; a training set, so not held out).

Tolerance 15 px at 1080p (TrackNet's 4 px at 512 width):

| Set | Configuration | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Kent 2 (held out) | beach 0.5 (baseline) | 34 | 19 | 36 | 11 | 0.64 | 0.49 |
| | beach 0.3 | 36 | 35 | 22 | 7 | 0.51 | 0.62 |
| | **vball_v1 0.5** | 45 | 19 | 25 | 11 | **0.70** | **0.64** |
| | vball_v1 0.3 | 48 | 24 | 18 | 10 | 0.67 | 0.73 |
| Brunel away 3 (held out, other gym, 60 fps) | beach 0.5 (baseline) | 26 | 5 | 56 | 13 | 0.84 | 0.32 |
| | beach 0.3 | 41 | 9 | 40 | 10 | 0.82 | 0.51 |
| | **vball_v1 0.5** | 47 | 5 | 35 | 13 | **0.90** | **0.57** |
| | vball_v1 0.3 | 53 | 8 | 27 | 12 | 0.87 | 0.66 |
| Kent 1 (training set) | beach 0.5 | 31 | 9 | 47 | 13 | 0.78 | 0.40 |
| | vball_v1 0.5 | 46 | 12 | 29 | 13 | 0.79 | 0.61 |

Spec target (recall +15 points, precision no worse than −3) is **met by vball_v1 at both thresholds on both held-out sets**. Lowering the threshold on the beach weights instead raises recall but costs precision.

Most "FP" on ball frames are near misses, not other objects: for vball_v1 0.5 on Kent 2, of 64 detections on ball frames 45 are within 15 px, 9 within 15–30 px, 8 within 30–60 px, 2 further. Near the camera the ball is 30–60 px wide, so 15 px is strict. At 30 px tolerance:

| Set | beach 0.5 | beach 0.3 | vball_v1 0.5 | vball_v1 0.3 |
|---|---|---|---|---|
| Kent 2 P / R | 0.87 / 0.56 | 0.80 / 0.72 | 0.84 / 0.68 | 0.83 / 0.77 |
| Brunel away 3 P / R | 0.94 / 0.34 | 0.90 / 0.53 | 0.94 / 0.58 | 0.93 / 0.68 |

## Data

11 landscape sets processed (Kent 1–4, Brunel home 1–4, Brunel away 1–3; away 2–3 at 60 fps). Held out from training: **Kent set 2** (#2) and **Brunel away set 3** (#11, other gym, 60 fps). Processing time ≈ 14 min per 30 fps set, ≈ 28 min per 60 fps set (RTX 3050 4 GB, libx264).

## Pseudo-labels (training sets #1, #3–#10)

Per set: 52–109 detections removed as outliers (~1 %), 241–777 frames added by free-flight gap filling (3–9 % of positives). Rally intervals from hand labels for #1, from detected rallies elsewhere.

## Fine-tune `models/tracknet_vball_v1.pt`

2,000 cached windows (6.7 GB memmap), 4 epochs, batch 2 × accumulation 4, fp16, lr 1e-4: 21 min including cache build. Peak GPU memory at batch 2: 1.3 GB. Masked WBCE 6e-5 → 5e-5 (mean over all pixels, so absolute values are tiny).

## Proxy metrics (no clicked frames needed)

Share of frames with a detection, threshold 0.5. "Rally frames" are hand-labelled rallies for Kent 2; for Brunel away 3 the rallies detected with the base model (biased toward the base model). "Dead time" is > 2 s from any rally.

| Held-out set | Model | Ball found in rally frames | Ball found in dead time | Rally F1 (current params) |
|---|---|---|---|---|
| Kent set 2 | beach weights | 51.6 % | 1.80 % | 0.892 |
| Kent set 2 | vball_v1 | **63.4 %** | 3.07 % | 0.871 |
| Brunel away set 3 | beach weights | 38.0 % | 1.65 % | — |
| Brunel away set 3 | vball_v1 | **59.5 %** | 2.89 % | — |

More detections in rallies on both held-out sets (+12 and +21 points), including the unseen gym at 60 fps. Dead-time detections also rose by ~1.3 points: either false positives or real balls between rallies; the clicked frames will tell. Rally F1 dipped with the phase 1 parameters; re-tune pending.

## Threshold 0.3 (proxy metrics)

| Held-out set | Model, threshold | Rally frames | Dead time | Rally F1 (current params) |
|---|---|---|---|---|
| Kent set 2 | beach, 0.3 | 69.0 % | 4.46 % | 0.817 |
| Kent set 2 | vball_v1, 0.3 | 74.0 % | 5.57 % | 0.784 |
| Brunel away set 3 | beach, 0.3 | 55.9 % | 5.51 % | — |
| Brunel away set 3 | vball_v1, 0.3 | 69.0 % | 5.27 % | — |

A lower threshold also raises in-rally detections (beach 0.3 beats v1 0.5 on Kent 2) but costs more dead-time detections and generalises worse to the other gym (55.9 % vs 59.5 %).

## Rally detection re-tuned on v1 tracks (Kent 1 + 2, `tuning.grid_search`)

| Tracks | Current params mean F1 | Re-tuned mean F1 (Kent 1, Kent 2) | Change |
|---|---|---|---|
| beach, 0.5 | 0.915 | 0.918 (0.911, 0.925) | — |
| vball_v1, 0.5 | 0.884 | **0.920 (0.938, 0.902)** | `min_active_frac` 0.2 → 0.3 |

With one parameter change, rally detection on v1 tracks matches the baseline (held-out Kent 2: 0.902 vs 0.892). Kent 1 was a training set for v1, so its 0.938 is optimistic.

Rally detection re-tuned for vball_v1 at threshold 0.3 (Kent 1 + 2): current params 0.800 → re-tuned **0.909** (Kent 1 0.916, held-out Kent 2 0.902) with `min_active_frac` 0.3, `max_gap_s` 2.0, `min_rally_s` 1.5 (other params unchanged). Neighbouring settings score 0.906, so this is a plateau, not a knife edge.

## Decision

Adopted **vball_v1 at threshold 0.3** with the re-tuned rally params:

| Criterion (spec) | Target | Result (held-out Kent 2 / Brunel away 3) |
|---|---|---|
| Ball recall | +15 points | 0.49 → 0.73 (+24) / 0.32 → 0.66 (+34) |
| Ball precision | ≥ baseline − 3 | 0.64 → 0.67 / 0.84 → 0.87 |
| Rally F1 (Kent 1 + 2 mean) | ≥ 0.915 − 0.02 | 0.909 (held-out Kent 2: 0.902 vs 0.892 before) |

vball_v1 at 0.5 had slightly better rally F1 (0.920) but lower ball recall; recall matters more for the next phase (touch detection), and rally F1 at 0.3 is within target.

Defaults changed: `TRACKNET_WEIGHTS = models/tracknet_vball_v1.pt`, `TRACKNET_THRESHOLD = 0.3`; the downloaded beach model is now `BASE_TRACKNET_WEIGHTS` (download script and `vball finetune --init` default). All 11 matches re-tracked; previous tracks kept as `data/exp/m<id>_base_t05_backup.csv`.

Checked with the real commands after re-tracking: `vball eval` rally F1 Kent 1 0.92, Kent 2 0.90; `vball balleval` matches the table above.

## Next levers (not done)

- Ball recall is still 0.66–0.73: a second self-training round on vball_v1's own tracks, or assisted correction of the frames the tracker misses.
- Between-rally ball returns and false rallies: court-region mask (phase 2b court calibration).
- Kent 2 rally labels were mostly copied from detections without edge edits, so its boundary errors are not meaningful.
