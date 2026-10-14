"""``lstmabar eval run`` / ``eval report`` (task 5.4). Wave 0: signatures; wave 2 implements.

``run``: for each method in ``eval.methods`` (``--methods``): build, ``fit(train)``,
``select(val)``, score val (selection table), the synthetic test splits and gold, write
per-unit scores to a run dir (``create_run_dir``), then the report
(``reports/<eval.report_name>.{md,json}``). ``report``: rebuild the report from a run dir.
"""

from __future__ import annotations


def run(cfg, args) -> int:
    raise NotImplementedError


def report(cfg, args) -> int:
    raise NotImplementedError


__all__ = ["report", "run"]
