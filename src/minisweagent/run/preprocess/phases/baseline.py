"""Baseline phase — run harness at baseline + profile + build metrics.

Inputs: ``ctx.harness_path``, ``ctx.kernel_path``, ``ctx.repo_root``,
``ctx.test_command``, ``ctx.gpu_id``.

Output: ``ctx.profiling``, ``ctx.baseline_metrics_path``,
``ctx.baseline_metrics``, ``ctx.benchmark_baseline``,
``ctx.full_benchmark_baseline``.

Skeleton: delegates to the legacy monolith for this commit.  Subsequent
commits move the logic here.
"""

from __future__ import annotations

import logging

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

logger = logging.getLogger(__name__)


class BaselinePhase(Phase):
    """Captures baseline correctness, profile, and per-op metrics.

    This is steps 4 + 5 + 6 of the legacy monolith.  The orchestrator's
    legacy fallback path handles the body for now.
    """

    name = "baseline"

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()
        if ctx.baseline_metrics_path:
            logger.info("  baseline_metrics already populated — skipping.")
            ctx.phases_run.append(self.name)
            return
        logger.debug("BaselinePhase: deferring to legacy fallback.")
        ctx.phases_run.append(self.name)


__all__ = ["BaselinePhase"]
