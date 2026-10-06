"""WASB ball detector (third_party/wasb, MIT; volleyball weights) over a whole set. More precise ball positions inside
rallies than our TrackNet (Kent 2 clicked frames: precision 1.00 / recall 0.85 vs 0.67 / 0.73) and more fitted 3D
flights, but it also follows the ball between points, so rally cutting keeps our tracker (docs/results/phase2e)."""

import sys
from pathlib import Path

import numpy as np

from vball.config import REPO_ROOT

WASB_DIR = REPO_ROOT / "third_party" / "wasb"
IN_W, IN_H = 512, 288  # the model's input; track.mp4 is already this size
THRESHOLD = 0.5  # heatmap peak (after sigmoid) that counts as a ball, as in WASB's own detector config
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _Cfg(dict):
    """WASB reads its model config both as cfg["x"] and cfg.x."""

    def __getattr__(self, key):
        return self[key]


def _cfg(d):
    return _Cfg({k: _cfg(v) for k, v in d.items()}) if isinstance(d, dict) else d


def peaks_to_rows(heatmaps: np.ndarray, first_frame: int, sx: float, sy: float) -> list[str]:
    """TrackNet-format CSV rows (Frame,Visibility,X,Y) for consecutive heatmaps (n, IN_H, IN_W) of sigmoid scores:
    the peak, scaled to work-video pixels, where it beats THRESHOLD."""
    rows = []
    for k, hm in enumerate(heatmaps):
        if hm.max() > THRESHOLD:
            y, x = np.unravel_index(hm.argmax(), hm.shape)
            rows.append(f"{first_frame + k},1,{int(x * sx)},{int(y * sy)}")
        else:
            rows.append(f"{first_frame + k},0,0,0")
    return rows


def run_wasb(track_video: Path, out_csv: Path, width: int, height: int, weights: Path) -> int:
    """WASB's step 3 over the 512x288 track video: frames t..t+2 in, their three heatmaps out. Returns frames read."""
    import cv2
    import torch
    import yaml

    sys.path.insert(0, str(WASB_DIR))
    from hrnet import HRNet  # vendored

    model = HRNet(_cfg(yaml.safe_load((WASB_DIR / "wasb.yaml").read_text())))
    state = torch.load(weights, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model = model.cuda().eval().half()
    cap = cv2.VideoCapture(str(track_video))
    rows, frame = ["Frame,Visibility,X,Y"], 0
    while True:
        imgs = []
        for _ in range(3):
            ok, img = cap.read()
            if not ok:
                break
            imgs.append(cv2.resize(img, (IN_W, IN_H)) if img.shape[:2] != (IN_H, IN_W) else img)
        if not imgs:
            break
        n = len(imgs)
        imgs += [imgs[-1]] * (3 - n)
        x = (np.stack(imgs)[..., ::-1].astype(np.float32) / 255 - MEAN) / STD  # (3, H, W, 3) RGB
        x = torch.from_numpy(x.transpose(0, 3, 1, 2).reshape(1, 9, IN_H, IN_W)).cuda().half()
        with torch.no_grad():
            hm = model(x)[0].float().sigmoid()[0, :n].cpu().numpy()
        rows += peaks_to_rows(hm, frame, width / IN_W, height / IN_H)
        frame += n
    cap.release()
    out_csv.write_text("\n".join(rows) + "\n")
    return frame
