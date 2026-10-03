# Phase 2a results: ball tracker

Status: fine-tune done; **precision/recall on clicked test frames pending** (Ball check labels for Kent set 2 and Brunel away set 3 not yet made).

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

## Pending

- Clicked-frame precision/recall: baseline, threshold 0.3 (base and v1), v1 at 0.5.
- `vball tunerallies` on v1 tracks; adopt or reject v1 as default.
