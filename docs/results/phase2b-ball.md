# Phase 2b results: ball overlay, external ball models, ball v2

Status: external benchmark done (GridV3 pending), v2 trained, v2 evaluation pending.

## External ball models (fast-volleyball-tracking-inference, MIT)

Run with the repo's own script (`src/inference_onnx_seq_gray_v2.py`, own uv env, Python 3.14, onnxruntime on CPU) on the full work videos of the held-out sets, scored with `vball balleval --pred` against the 100 clicked frames per set.

| Model | Kent 2 P / R @15 px | @30 px | Brunel away 3 P / R @15 px | @30 px | Run time (Kent 2 / Brunel 3, CPU) |
|---|---|---|---|---|---|
| ours v1 @ 0.3 (current default) | 0.67 / 0.73 | 0.83 / 0.77 | 0.87 / 0.66 | 0.93 / 0.68 | ~10 / ~18 min (GPU) |
| **VballNetV4c** | **0.84 / 0.94** | 0.84 / 0.94 | 0.83 / **0.69** | 0.85 / 0.70 | 12 / 19 min |
| VballNetGridV2b | 0.86 / 0.77 | 0.90 / 0.78 | **0.98** / 0.65 | 0.98 / 0.65 | 14 / 25 min |
| VballNetFastV1 | 0.74 / 0.86 | 0.82 / 0.87 | 0.89 / 0.60 | 0.89 / 0.60 | 7 / 11 min |
| VballNetGridV3 | pending | | | | |

Rally detection on Kent 2 (only held-out set with rally labels) using each model's ball track:

| Track | Rally F1, current params | Re-tuned on Kent 2 only (optimistic) | Ball found in all frames |
|---|---|---|---|
| ours v1 @ 0.3 | 0.902 | 0.911 | 23 % |
| VballNetV4c | 0.857 | 0.900 | 38 % |
| VballNetGridV2b | 0.860 | 0.902 | 33 % |

Reading: VballNetV4c is the most accurate ball locator during play (Kent 2 recall 0.94 at precision 0.84), but it also tracks balls held or tossed between points, which makes dead time look like play; our tracker is at least as good for cutting rallies. Candidate combination for phase 2c: our tracker for rallies, V4c (or the best of v2/V4c) for ball positions inside rallies.

Notes on running the external script:
- It infers model settings from the file name. Output planes are read from the model at runtime (so V4c with 2 planes and GridV2b with 4 planes decode correctly), but input and grid sizes were hard-coded (768×432, 48×27): **GridV3 (1024×576, 64×36 grid) crashed** on both sets. A local patch to the downloaded copy (not in our repo) reads them from the model's input/output shapes; GridV3 re-run pending.
- First run creates the repo's own environment (Python 3.14 downloaded by uv).

## Ball v2 (`models/tracknet_vball_v2.pt`)

`vball finetune --matches 1 3 4 5 6 7 8 9 10 --vballnet data/external/vballnet-dataset/data --windows 3000 --rebuild-cache`: 98 sources (9 own pseudo-labelled sets + 89 VballNet hand-labelled clips), 3,000 windows (9.8 GB cache), 4 epochs, loss 8e-5 → 5e-5, ~40 min including cache build.

| Held-out set | v1 @ 0.3 (current) | v2 @ 0.3 | v2 @ 0.5 | VballNetV4c |
|---|---|---|---|---|
| Kent 2 P / R @15 px (@30 px) | 0.67 / 0.73 (0.83 / 0.77) | 0.65 / **0.90** (0.83 / 0.92) | 0.71 / 0.72 (0.90 / 0.77) | **0.84 / 0.94** (0.84 / 0.94) |
| Brunel away 3 P / R @15 px (@30 px) | 0.87 / 0.66 (0.93 / 0.68) | 0.83 / **0.84** (0.90 / 0.85) | 0.87 / 0.68 (0.92 / 0.69) | 0.83 / 0.69 (0.85 / 0.70) |

Rally F1 (Kent 1 + 2): v1 @ 0.3 0.909 (0.916, 0.902); v2 @ 0.3 0.834 with current params, 0.893 (0.881, 0.905) re-tuned (`window_s` 2, `min_rally_s` 2, `post_pad_s` 0.5).

Against the adoption rule fixed before training (higher recall at **no worse** precision than v1 @ 0.3, rally F1 within −0.02): v2 @ 0.3 gains +17 / +18 recall points but loses 2 / 4 precision points (level / −3 at 30 px), rally F1 −0.016. **Strictly, v2 does not qualify.** Its recall in the unseen gym (0.84) is the best of any model, so the VballNet labels clearly help generalisation.
