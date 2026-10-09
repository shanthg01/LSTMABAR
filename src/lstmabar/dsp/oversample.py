"""Band-limited integer-factor resampling for running nonlinearities without aliasing."""

from collections.abc import Callable

from torch import Tensor


def upsample(x: Tensor, factor: int) -> Tensor:
    """``(B, T)`` → ``(B, T * factor)`` with an anti-imaging low-pass.

    ``factor`` in {1, 2, 4, 8}. Unity passband gain (zero-stuffing is compensated by
    ``x factor``) and zero phase on its own: sample ``n`` of ``x`` maps to sample
    ``n * factor`` of the output.
    """
    raise NotImplementedError


def downsample(x: Tensor, factor: int) -> Tensor:
    """``(B, T * factor)`` → ``(B, T)`` with an anti-aliasing low-pass before decimation.

    Requires ``x.shape[-1] % factor == 0``. Unity passband gain and zero phase on its own.
    """
    raise NotImplementedError


def oversampled(fn: Callable[[Tensor], Tensor], x: Tensor, factor: int) -> Tensor:
    """Run ``fn`` at ``factor``× the sample rate: ``downsample(fn(upsample(x)))``.

    Output has the same shape as ``x`` and is time-aligned with it (zero group delay or
    delay compensated).
    """
    raise NotImplementedError
