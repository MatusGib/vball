import torch

import vball


def test_package_version():
    assert vball.__version__ == "0.1.0"


def test_cuda_is_available():
    assert torch.cuda.is_available(), "PyTorch cannot see the GPU - check the cu130 wheel and NVIDIA driver"
