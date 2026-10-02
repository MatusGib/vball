"""Fine-tune the vendored TrackNetV3 on pseudo-labelled windows."""

import importlib
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.data import DataLoader

from vball import store
from vball.ball.pseudo import make_pseudo_labels
from vball.ball.track import load_tracknet_csv
from vball.ball.tracknet import TRACKNET_H, TRACKNET_W, downscale_command
from vball.ball.train_data import TrainSource, WindowDataset
from vball.config import TRACKNET_DIR, Paths
from vball.labels import load_labels


def tracknet_modules() -> tuple[ModuleType, ModuleType]:
    """Import the vendored TrackNetV3 `utils.general` and `dataset` modules."""
    path = str(TRACKNET_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("utils.general"), importlib.import_module("dataset")


def load_tracknet(weights: Path, device: str) -> tuple[torch.nn.Module, dict]:
    general, _ = tracknet_modules()
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    params = ckpt["param_dict"]
    model = general.get_model("TrackNet", params["seq_len"], params["bg_mode"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, params


def save_tracknet(model: torch.nn.Module, params: dict, path: Path, epoch: int) -> None:
    """Same checkpoint layout as upstream, so predict.py loads it unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "param_dict": params, "epoch": epoch}, path)


def masked_wbce(y_pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """TrackNet's weighted BCE, averaged per frame, skipping frames where mask == 0. Computed in fp32."""
    p = y_pred.float().clamp(1e-7, 1 - 1e-7)
    loss = -(torch.square(1 - p) * y * torch.log(p) + torch.square(p) * (1 - y) * torch.log(1 - p))
    per_frame = loss.mean(dim=(2, 3))
    return (per_frame * mask).sum() / mask.sum().clamp(min=1.0)


def finetune(
    cache_dir: Path,
    init_weights: Path,
    out_path: Path,
    epochs: int = 4,
    batch_size: int = 2,
    accum: int = 4,
    lr: float = 1e-4,
    device: str | None = None,
    log: Callable[[str], None] = print,
) -> list[float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device == "cuda"
    model, params = load_tracknet(init_weights, device)
    loader = DataLoader(WindowDataset(cache_dir), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        total = 0.0
        for step, (x, y, mask) in enumerate(loader, start=1):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                y_pred = model(x)
            loss = masked_wbce(y_pred, y, mask)
            scaler.scale(loss / accum).backward()
            if step % accum == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total += loss.item()
        losses.append(total / max(1, len(loader)))
        log(f"epoch {epoch}/{epochs} loss {losses[-1]:.5f}")
        save_tracknet(model, {**params, "finetuned_from": str(init_weights)}, out_path, epoch)
    return losses


def prepare_sources(
    conn: sqlite3.Connection, paths: Paths, match_ids: list[int], log: Callable[[str], None] = print
) -> list[TrainSource]:
    """Pseudo-label each match's ball track; rally intervals come from hand labels when present."""
    sources = []
    for match_id in match_ids:
        match = store.get_match(conn, match_id)
        fps, width, height = match["fps"], match["width"], match["height"]
        gt = paths.gt_csv(match_id)
        if gt.exists():
            rallies = [(round(l.start_s * fps), round(l.end_s * fps)) for l in load_labels(gt)]
        else:
            rallies = [(r["start_frame"], r["end_frame"]) for r in store.get_rallies(conn, match_id)]
        track = load_tracknet_csv(paths.ball_csv(match_id), match["n_frames"])
        filled, state, stats = make_pseudo_labels(track, rallies, fps, width, height)
        log(f"match {match_id} {match['name']}: {stats}")
        small = paths.track_video(match_id)
        if not small.exists():
            subprocess.run(downscale_command(paths.work_video(match_id), small), check=True)
        sources.append(TrainSource(small, state, filled.x * TRACKNET_W / width, filled.y * TRACKNET_H / height))
    return sources
