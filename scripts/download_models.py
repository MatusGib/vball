"""Download the volleyball-fine-tuned TrackNetV3 weights (MIT) into models/."""

import shutil

import torch
from huggingface_hub import hf_hub_download

from vball.config import BASE_TRACKNET_WEIGHTS, MODELS_DIR

REPO_ID = "deadfast/beach-volley-vision-models"
REVISION = "b03107eddc5bc32d2f89f70b225e3b67da9c34e7"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(REPO_ID, "tracknet_best.pt", revision=REVISION)
    shutil.copyfile(cached, BASE_TRACKNET_WEIGHTS)
    ckpt = torch.load(BASE_TRACKNET_WEIGHTS, map_location="cpu", weights_only=False)
    print("keys:", sorted(ckpt.keys()))
    print("param_dict:", ckpt.get("param_dict"))


if __name__ == "__main__":
    main()
