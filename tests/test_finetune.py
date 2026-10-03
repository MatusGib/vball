import numpy as np
import pytest
import torch

from vball.ball.finetune import finetune, load_tracknet, masked_wbce
from vball.ball.pseudo import IGNORED, POSITIVE
from vball.ball.train_data import H, SEQ_LEN, W
from vball.config import BASE_TRACKNET_WEIGHTS

requires_weights = pytest.mark.skipif(not BASE_TRACKNET_WEIGHTS.exists(), reason="TrackNet weights not downloaded")


def test_masked_wbce_ignores_masked_frames():
    y = torch.zeros(1, SEQ_LEN, 4, 4)
    good = torch.full((1, SEQ_LEN, 4, 4), 0.01)
    bad = good.clone()
    bad[0, 3] = 0.99  # confident wrong answer on frame 3
    mask_all = torch.ones(1, SEQ_LEN)
    mask_skip3 = mask_all.clone()
    mask_skip3[0, 3] = 0
    assert masked_wbce(bad, y, mask_all) > masked_wbce(good, y, mask_all)
    assert torch.isclose(masked_wbce(bad, y, mask_skip3), masked_wbce(good, y, mask_skip3))


def write_tiny_cache(cache_dir):
    rng = np.random.default_rng(0)
    cache_dir.mkdir(parents=True)
    frames = np.lib.format.open_memmap(cache_dir / "frames.npy", mode="w+", dtype=np.uint8, shape=(SEQ_LEN, H, W, 3))
    frames[:] = rng.integers(0, 255, size=(SEQ_LEN, H, W, 3), dtype=np.uint8)
    frames.flush()
    state = np.full((1, SEQ_LEN), POSITIVE, dtype=np.int8)
    state[0, 5] = IGNORED
    np.savez(
        cache_dir / "meta.npz",
        windows=np.arange(SEQ_LEN)[None],
        xy=np.full((1, SEQ_LEN, 2), 100.0, dtype=np.float32),
        state=state,
        source=np.zeros(1, dtype=np.int64),
        medians=np.zeros((1, H, W, 3), dtype=np.uint8),
        videos=np.array(["synthetic"]),
    )


@requires_weights
def test_finetune_runs_and_saves_upstream_format(tmp_path):
    write_tiny_cache(tmp_path / "cache")
    out = tmp_path / "tuned.pt"
    losses = finetune(
        tmp_path / "cache", BASE_TRACKNET_WEIGHTS, out, epochs=2, batch_size=1, accum=1, device="cpu", log=lambda s: None
    )
    assert len(losses) == 2 and all(np.isfinite(losses))
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert {"model", "param_dict"} <= set(ckpt)
    assert ckpt["param_dict"]["seq_len"] == SEQ_LEN and ckpt["param_dict"]["bg_mode"] == "concat"
    model, _ = load_tracknet(out, "cpu")  # loads like predict.py does
    assert sum(p.numel() for p in model.parameters()) > 0
