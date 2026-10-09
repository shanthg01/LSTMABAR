"""Global seeding for reproducible runs."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    With ``deterministic=True`` PyTorch is asked to use deterministic kernels, which can be
    slower and raises on ops that have no deterministic implementation. Call it before any
    CUDA work so the cuBLAS workspace setting takes effect.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
