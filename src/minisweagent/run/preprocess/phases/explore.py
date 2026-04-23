"""Explore phase — render commandment + (future) kernel analysis rubric.

Inputs: ``ctx.kernel_path``, ``ctx.harness_path``, ``ctx.discovery``,
``ctx.baseline_metrics``, ``ctx.profiling``, ``ctx.codebase_context_path``.

Output: ``ctx.commandment``, ``ctx.commandment_path``, and (future)
``ctx.kernel_analysis_md``.

Skeleton: delegates to the legacy monolith for this commit.  Subsequent
commits swap the 530-LoC ``commandment.py`` branching for a Jinja
template and add the ``KernelAnalysisAgent`` subagent.
"""

from __future__ import annotations

import logging

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

logger = logging.getLogger(__name__)


class ExplorePhase(Phase):
    """Produces ``ctx.commandment`` (markdown) + future analysis rubric.

    This is step 7 of the legacy monolith.  The legacy fallback path
    handles the body for now.
    """

    name = "explore"

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()
        if ctx.commandment_path:
            logger.info("  commandment already rendered — skipping.")
            ctx.phases_run.append(self.name)
            return
        logger.debug("ExplorePhase: deferring to legacy fallback.")
        ctx.phases_run.append(self.name)


__all__ = ["ExplorePhase"]
