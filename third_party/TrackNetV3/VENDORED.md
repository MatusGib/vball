# Vendored TrackNetV3

Source: https://github.com/qaz812345/TrackNetV3 at commit 6eda442ada1740573f200f836d93edc9a541ee86 (MIT, see LICENSE).
Copied: LICENSE, README.md, dataset.py, model.py, predict.py, test.py, utils/.

Local patches:
1. predict.py: `torch.load(..., weights_only=False)` (torch >= 2.6 defaults to weights_only=True; checkpoints contain a param dict).
2. dataset.py `Video_IterableDataset.__iter__`: `if not frame_list: break` to avoid IndexError when the frame count is a multiple of seq_len.
3. predict.py: `--threshold` argument (module-level `HEATMAP_THRESHOLD`, default 0.5) replaces the hard-coded heatmap threshold in `predict()`.

Run from this directory (it imports `test`, `dataset`, `utils` relative to the cwd). Pass video paths with forward slashes: predict.py derives the output name with `split('/')`.
