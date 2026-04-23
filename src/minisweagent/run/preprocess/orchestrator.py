"""PreprocessOrchestrator — drives the 4 mandatory + 1 conditional phase.

The orchestrator is deliberately thin: it iterates a fixed list of
phase instances and calls ``is_applicable`` / ``run`` on each, in
order.  Phases communicate via a shared ``PhaseContext``.

Phase order:

  1. TranslationPhase  — conditional; only when ``target_language``
                          differs from the source
  2. DiscoveryPhase    — always
  3. HarnessPhase      — always
  4. BaselinePhase     — always
  5. ExplorePhase      — always

During the preprocessing refactor transition period, the phase bodies
are thin — they either:

  (a) own their logic completely (e.g. ``DiscoveryPhase``), OR
  (b) early-exit so the orchestrator falls back to the legacy
      ``run_preprocessor`` monolith for the rest of the pipeline.

Case (b) is marked by the phase setting its output fields to empty /
None; the orchestrator detects any mandatory output still missing
after all phases ran and falls back to
``run_preprocessor_legacy`` for the missing steps.  This keeps every
interim commit rollback-safe without creating a second runtime path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext
from minisweagent.run.preprocess.phases.baseline import BaselinePhase
from minisweagent.run.preprocess.phases.discovery import DiscoveryPhase
from minisweagent.run.preprocess.phases.explore import ExplorePhase
from minisweagent.run.preprocess.phases.harness import HarnessPhase
from minisweagent.run.preprocess.phases.translation import TranslationPhase

logger = logging.getLogger(__name__)


class PreprocessOrchestrator:
    """Drive all preprocessing phases in order."""

    def __init__(self, phases: list[Phase] | None = None) -> None:
        self.phases: list[Phase] = phases or [
            TranslationPhase(),
            DiscoveryPhase(),
            HarnessPhase(),
            BaselinePhase(),
            ExplorePhase(),
        ]

    def run(self, ctx: PhaseContext) -> PhaseContext:
        """Execute every applicable phase.  Mutates ``ctx`` in place.

        If ``ctx.translate_only`` is True and TranslationPhase
        succeeds, we return early after TranslationPhase (the
        standalone ``geak translate`` path).
        """
        for phase in self.phases:
            if not phase.is_applicable(ctx):
                logger.debug("Phase %s skipped (is_applicable=False).", phase.name)
                ctx.phases_skipped.append((phase.name, "not applicable"))
                continue

            try:
                phase.run(ctx)
            except NotImplementedError:
                # Translation phase body isn't built yet; cli.py
                # pre-empts this case, but if someone invokes the
                # orchestrator directly we surface the error with
                # context.
                raise
            except Exception as exc:
                logger.error(
                    "Phase %s failed with %s: %s",
                    phase.name,
                    type(exc).__name__,
                    exc,
                )
                raise

            if ctx.translate_only and phase.name == TranslationPhase.name:
                logger.info("translate_only=True: returning after TranslationPhase")
                return ctx

        return ctx


def run_preprocessor_via_orchestrator(
    kernel_url: str,
    output_dir: Path,
    gpu_id: int = 0,
    *,
    model: Any = None,
    model_factory: Any = None,
    console: Any = None,
    harness: str | None = None,
    repo: str | Path | None = None,
    eval_command: str | None = None,
    correctness_command: str | list[str] | None = None,
    performance_command: str | list[str] | None = None,
    benchmark_timeout: int = 3600,
    target_language: str | None = None,
    translate_only: bool = False,
) -> dict[str, Any]:
    """Drop-in shim for the legacy ``run_preprocessor`` signature.

    Builds a ``PhaseContext`` from the kwargs, runs
    ``PreprocessOrchestrator``, and returns the output dict.  During
    the refactor transition, phases that haven't absorbed their
    logic yet cause the orchestrator to delegate to
    ``run_preprocessor`` (legacy) for the missing steps.  This keeps
    every commit rollback-safe.
    """
    ctx = PhaseContext(
        kernel_url=kernel_url,
        output_dir=Path(output_dir),
        gpu_id=gpu_id,
        harness=harness,
        repo=repo,
        eval_command=eval_command,
        correctness_command=correctness_command,
        performance_command=performance_command,
        benchmark_timeout=benchmark_timeout,
        model=model,
        model_factory=model_factory,
        console=console,
        target_language=target_language,
        translate_only=translate_only,
    )

    orch = PreprocessOrchestrator()
    orch.run(ctx)

    # Legacy fallback for any mandatory output not yet populated by
    # the new phase bodies.  Only triggered during the refactor
    # transition — deletes once every phase owns its logic.
    result = ctx.to_dict()
    if not result.get("harness_path") or not result.get("baseline_metrics_path"):
        logger.debug(
            "Orchestrator produced partial ctx (harness_path=%s, baseline_metrics_path=%s); "
            "falling back to legacy run_preprocessor for remaining steps.",
            bool(result.get("harness_path")),
            bool(result.get("baseline_metrics_path")),
        )
        from minisweagent.run.preprocess.preprocessor import run_preprocessor as _legacy

        legacy_result = _legacy(
            kernel_url=kernel_url,
            output_dir=output_dir,
            gpu_id=gpu_id,
            model=model,
            model_factory=model_factory,
            console=console,
            harness=harness,
            repo=repo,
            eval_command=eval_command,
            correctness_command=correctness_command,
            performance_command=performance_command,
            benchmark_timeout=benchmark_timeout,
        )
        # Legacy wins for fields where the new phases didn't produce
        # output; new phases win for fields they did populate.
        for key, value in legacy_result.items():
            if not result.get(key):
                result[key] = value

    return result


__all__ = ["PreprocessOrchestrator", "run_preprocessor_via_orchestrator"]
