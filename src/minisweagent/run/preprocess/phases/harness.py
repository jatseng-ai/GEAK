"""Harness phase — produce a validated harness + test_command.

Inputs: ``ctx.kernel_path``, ``ctx.repo_root``, ``ctx.discovery``,
``ctx.harness`` (optional explicit path), plus CLI eval hooks
(``eval_command``, ``correctness_command``, ``performance_command``).

Output: ``ctx.harness_path``, ``ctx.test_command``,
``ctx.harness_results``, ``ctx.testcase_selection``.

This is the most complex phase: it picks among four candidate sources
(explicit --harness, cache hit, LLM-agent-generated, ad-hoc test
command) and validates statically + at runtime in all modes
(``--correctness``, ``--profile``, ``--benchmark``, ``--full-benchmark``).

Skeleton: delegates to the legacy monolith for this commit.  Subsequent
commits will move the per-candidate logic into
``subagents/preprocess/harness_builder.py`` (HarnessBuilder LLM
subagent) and inline the validation + cache helpers here.
"""

from __future__ import annotations

import logging

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

logger = logging.getLogger(__name__)


class HarnessPhase(Phase):
    """Produces ``ctx.harness_path`` + ``ctx.test_command``.

    Implementation note (this commit): the body delegates to the
    portion of ``run_preprocessor`` that handles steps 3b + 4 of the
    legacy monolith.  Because that portion is entangled with the rest
    of the monolith's single function, this skeleton currently
    *returns early* and leaves the orchestrator to fall back to the
    legacy ``run_preprocessor`` path via ``LegacyFallbackPhase``.
    The next commit extracts the harness logic in-place.
    """

    name = "harness"

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()
        if ctx.harness_path:
            logger.info("  harness_path already set by previous phase (%s) — skipping.", ctx.harness_path)
            ctx.phases_run.append(self.name)
            return
        # Body is delegated to the legacy monolith for now.  The
        # orchestrator's fallback path handles this case.
        logger.debug("HarnessPhase: no harness_path yet; deferring to legacy fallback.")
        ctx.phases_run.append(self.name)


__all__ = ["HarnessPhase"]
