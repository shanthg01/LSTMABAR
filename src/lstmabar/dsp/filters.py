"""Differentiable biquad filters (RBJ Audio EQ Cookbook), applied by frequency sampling.

All parameters are physical units with shape ``(B,)``; audio is ``(B, T)``.
Coefficients are normalized so ``a0 == 1``.
"""

from typing import Literal

from torch import Tensor

FilterKind = Literal["lowpass", "highpass", "peak", "lowshelf", "highshelf"]


def biquad_coeffs(
    kind: FilterKind,
    freq_hz: Tensor,
    q: Tensor,
    gain_db: Tensor,
    sample_rate: int,
) -> tuple[Tensor, Tensor]:
    """RBJ cookbook biquad. Returns ``(b, a)``, each ``(B, 3)`` with ``a[:, 0] == 1``.

    ``gain_db`` is ignored for lowpass/highpass. Shelves use ``q`` as the cookbook Q
    (shelf slope via Q). ``freq_hz`` is clamped to (0, Nyquist).
    """
    raise NotImplementedError


def freqz(b: Tensor, a: Tensor, n_fft: int) -> Tensor:
    """Complex frequency response on the rfft grid: ``(B, n_fft // 2 + 1)``."""
    raise NotImplementedError


def apply_filters(x: Tensor, coeffs: list[tuple[Tensor, Tensor]]) -> Tensor:
    """Apply a cascade of biquads to ``x`` ``(B, T)`` by multiplying their responses.

    Uses an FFT size of at least ``2 * T`` (zero-padded) so circular wrap of the truncated
    impulse response is negligible; output is trimmed back to ``(B, T)``.
    """
    raise NotImplementedError


def biquad(
    x: Tensor,
    kind: FilterKind,
    freq_hz: Tensor,
    sample_rate: int,
    q: Tensor | None = None,
    gain_db: Tensor | None = None,
) -> Tensor:
    """Convenience: single biquad. Defaults: ``q = 1/sqrt(2)``, ``gain_db = 0``."""
    raise NotImplementedError


def tilt(x: Tensor, tilt_db: Tensor, sample_rate: int, pivot_hz: float = 1000.0) -> Tensor:
    """Tone tilt: low shelf of ``-tilt_db/2`` and high shelf of ``+tilt_db/2`` at ``pivot_hz``.

    Positive ``tilt_db`` brightens, negative darkens; 0 dB is (near) identity.
    """
    raise NotImplementedError
