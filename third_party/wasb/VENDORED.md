# Vendored WASB model

Source: https://github.com/nttcom/WASB-SBDT at commit 923462cacdeb3353b84ddebdedb3f4b7a8553b0f (MIT, see LICENSE.md).
Copied: `src/models/hrnet.py`, `src/configs/model/wasb.yaml`, `LICENSE.md`. No local patches.

Weights (not in git): the volleyball checkpoint from the repo's MODEL_ZOO.md,
`uvx gdown 1M9y4wPJqLc0K-z-Bo5DP8Ft5XwJuLqIS -O models/wasb/` → `models/wasb/wasb_volleyball_best.pth.tar` (6 MB).
Used by `vball wasb <id>` (src/vball/ball/wasb.py).
