"""Harness phase — produce a validated harness + test_command.

Inputs: ``ctx.kernel_path``, ``ctx.repo_root``, ``ctx.discovery``,
``ctx.harness`` (optional explicit path), plus CLI eval hooks
(``eval_command``, ``correctness_command``, ``performance_command``).

Output: ``ctx.harness_path``, ``ctx.test_command``,
``ctx.harness_results``, ``ctx.testcase_selection``.

This is the most complex phase: it picks among four candidate sources
(explicit ``--harness``, cache hit, LLM-agent-generated, ad-hoc test
command) and validates statically + at runtime in all modes
(``--correctness``, ``--profile``, ``--benchmark``, ``--full-benchmark``).

**Current status (transitional).** The full 6-layer fallback chain
still lives in the legacy ``preprocessor.py`` monolith and is invoked
by the orchestrator's legacy-fallback path when this skeleton does
not populate a harness.  Subsequent commits will migrate the chain
into ``HarnessBuilder`` (subagent) + inline helpers here.

What this skeleton DOES do today:
  - If ``ctx.harness_path`` was populated by an upstream phase or
    caller, run the contract validator on it (§13.2-A row 8
    coverage).
  - §13.2-A row 6: when DiscoveryPhase detected a merged kernel and
    wrote ``ctx.split_harness_hint``, opportunistically promote the
    split harness to ``ctx.harness`` (if the caller did NOT supply
    an explicit ``--harness``) — but only when it passes static
    ``validate_harness``.  Matches legacy
    ``preprocessor.py:518-523``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

logger = logging.getLogger(__name__)


class HarnessPhase(Phase):
    """Produces ``ctx.harness_path`` + ``ctx.test_command``.

    Skeleton today; defers to the legacy monolith via the
    orchestrator's fallback path when it cannot resolve a harness on
    its own.  The split-harness-hint pickup is the only full-fidelity
    behaviour this skeleton owns today.
    """

    name = "harness"

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()

        # §13.2-A row 6: split-harness-hint pickup.  DiscoveryPhase
        # detected a merged kernel file and wrote the split harness
        # path to ``ctx.split_harness_hint``.  If the caller did not
        # supply an explicit ``--harness`` AND the split harness
        # passes static validation, promote it.  Legacy equivalent:
        # ``preprocessor.py:518-523``.
        if ctx.split_harness_hint and not ctx.harness:
            candidate = Path(ctx.split_harness_hint)
            if candidate.exists():
                try:
                    from minisweagent.run.preprocess.harness_utils import (
                        validate_harness as static_validate_harness,
                    )

                    ok, errors = static_validate_harness(candidate)
                    if ok:
                        ctx.harness = str(candidate)
                        logger.info(
                            "  Promoted split harness to --harness: %s",
                            candidate,
                        )
                    else:
                        logger.debug(
                            "  Split harness failed static validation "
                            "(%d error(s)); falling back to discovery chain.",
                            len(errors),
                        )
                except Exception as exc:
                    logger.debug(
                        "  Split harness static validation raised %s: %s",
                        type(exc).__name__,
                        exc,
                    )

        if ctx.harness_path:
            logger.info(
                "  harness_path already set by previous phase (%s) — running contract validator.",
                ctx.harness_path,
            )
            self._validate_if_present(ctx.harness_path)
            ctx.phases_run.append(self.name)
            return

        # Body is delegated to the legacy monolith for now.  The
        # orchestrator's fallback path handles the 6-layer chain.
        logger.debug("HarnessPhase: no harness_path yet; deferring to legacy fallback.")
        ctx.phases_run.append(self.name)

    @staticmethod
    def _validate_if_present(path_str: str | None) -> None:
        """Run the universal harness contract validator when a path is available."""
        if not path_str:
            return
        try:
            from minisweagent.kernel_languages.contract import validate_harness

            validate_harness(Path(path_str))
        except Exception as exc:
            logger.warning("[yellow]validate_harness: %s[/yellow]", exc)


__all__ = ["HarnessPhase"]
