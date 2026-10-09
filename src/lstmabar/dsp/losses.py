"""Audio-domain losses for training and parameter recovery."""

import torch
from torch import Tensor


def _stft_mag(x: Tensor, n_fft: int, hop: int, eps: float) -> Tensor:
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x, n_fft, hop_length=hop, win_length=n_fft, window=window, center=True,
        pad_mode="reflect", return_complex=True,
    )
    # Clamp the power before sqrt so silent bins have finite gradients.
    return (spec.real.square() + spec.imag.square()).clamp(min=eps).sqrt()


def multi_resolution_stft_loss(
    pred: Tensor,
    target: Tensor,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    hop_ratio: float = 0.25,
    eps: float = 1e-7,
) -> Tensor:
    """Mean over resolutions of spectral convergence + log-magnitude L1 (Yamamoto et al.).

    Hann window, hop = ``int(n_fft * hop_ratio)``. Spectral convergence is computed per
    example and then averaged over the batch.

    ``pred`` and ``target`` are ``(B, T)``; returns a scalar. Differentiable w.r.t. ``pred``.

    Details: centred STFT with reflect padding (so ``T`` must exceed ``n_fft // 2``);
    magnitudes are ``sqrt(max(|X|^2, eps))``; spectral convergence is
    ``||M_t - M_p||_F / (||M_t||_F + eps)`` per example; the log term is the mean of
    ``|log M_t - log M_p|`` over all bins, frames and examples.
    """
    if pred.shape != target.shape or pred.dim() != 2:
        raise ValueError(
            f"expected matching (B, T) inputs, got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    target = target.to(device=pred.device, dtype=pred.dtype)
    total = pred.new_zeros(())
    for n_fft in fft_sizes:
        hop = max(1, int(n_fft * hop_ratio))
        mp = _stft_mag(pred, n_fft, hop, eps)
        mt = _stft_mag(target, n_fft, hop, eps)
        sc = torch.linalg.vector_norm(mt - mp, dim=(-2, -1)) / (
            torch.linalg.vector_norm(mt, dim=(-2, -1)) + eps
        )
        log_mag = (torch.log(mt) - torch.log(mp)).abs().mean()
        total = total + sc.mean() + log_mag
    return total / len(fft_sizes)
