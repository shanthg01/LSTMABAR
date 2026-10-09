"""Audio-domain losses for training and parameter recovery."""

from torch import Tensor


def multi_resolution_stft_loss(
    pred: Tensor,
    target: Tensor,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    hop_ratio: float = 0.25,
    eps: float = 1e-7,
) -> Tensor:
    """Mean over resolutions of spectral convergence + log-magnitude L1 (Yamamoto et al.).

    ``pred`` and ``target`` are ``(B, T)``; returns a scalar. Differentiable w.r.t. ``pred``.
    """
    raise NotImplementedError
