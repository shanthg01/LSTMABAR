"""Task 4.4 (wave 1d): real-pedal test set (EGFxSet units, a few pOD-set pedals).

Contract: ``run(cfg, args)`` downloads only the needed archives into ``data.test_real_dir``
(one at a time, R9), pairs each real-pedal recording with its clean input, and writes
``TestRealItem`` records to ``DataLayout.test_real_manifest``. ToneTwist / pOD-set audio is
``internal_only`` and never leaves the machine.
"""

from __future__ import annotations


def run(cfg, args) -> int:
    raise NotImplementedError("P4 task 4.4 (wave 1d)")
