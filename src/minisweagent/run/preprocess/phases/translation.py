"""Translation phase (CONDITIONAL) — runs before Discovery when ``target_language ≠ source``.

Per the execution plan §0.5(b), translation runs as a preprocess phase
(NOT as a ``run_pipeline`` mode).  When triggered:

  a. Run the source kernel's test harness once in correctness mode to
     capture golden tensors.
  b. Hand the source code + golden tensors to
     ``TranslationAgent`` (``subagents/translation/translator.py``),
     which runs a verify-retry loop with ``tensor_allclose`` as the
     verifier.  TranslationAgent is a standalone ``SubagentBase``
     subclass — it does NOT inherit from or compose
     ``OptimizationAgent`` — each attempt is a direct ``model.query``
     call.
  c. Run ``validate_translation_performance`` (0.5× fail / 0.8× warn
     per language pair, adapted from PR #153).
  d. Swap ``ctx.kernel_path`` to the translated file, ``ctx.language``
     to ``target_language``, and continue.
  e. If ``translate_only=True``, the orchestrator returns ``ctx``
     early; otherwise Discovery / Harness / Baseline / Explore follow
     normally, operating on the translated kernel.

Implementation is scheduled for completion in a subsequent commit of
this refactor PR.  Until then this phase raises
``NotImplementedError`` with an actionable pointer; ``cli.py``
pre-empts that by rejecting the flag combination up-front, so the
only way to reach this is by instantiating the phase directly.
"""

from __future__ import annotations

import logging

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

logger = logging.getLogger(__name__)


class TranslationPhase(Phase):
    """Gate: only runs when ``target_language`` is set and differs from source."""

    name = "translation"

    def is_applicable(self, ctx: PhaseContext) -> bool:
        if not ctx.target_language:
            return False
        # Source language detection happens in DiscoveryPhase, but the
        # orchestrator runs TranslationPhase BEFORE Discovery — so at
        # this point we only know the target.  We consider the phase
        # applicable whenever target_language is set; the phase body
        # short-circuits if the source (inferred from kernel path
        # extension) matches.
        return True

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()

        # Cheap short-circuit: if the source kernel's extension maps
        # to the same canonical language as the target, there's
        # nothing to translate.
        src_lang = self._infer_source_language(ctx.kernel_url)
        if src_lang and src_lang == ctx.target_language:
            logger.info(
                "  target_language=%s matches inferred source language; translation skipped (no-op).",
                ctx.target_language,
            )
            ctx.phases_skipped.append((self.name, f"source={src_lang} already matches target"))
            return

        # Full translation implementation is scheduled for a
        # subsequent commit in this PR.  See TranslationAgent at
        # subagents/translation/translator.py for the narrow-loop
        # contract.
        raise NotImplementedError(
            f"TranslationPhase body is not implemented yet (source={src_lang!r} -> "
            f"target={ctx.target_language!r}).  The TranslationAgent skeleton lives at "
            f"subagents/translation/translator.py; the full implementation lands in a "
            f"subsequent commit of the preprocessing refactor PR."
        )

    @staticmethod
    def _infer_source_language(kernel_url: str) -> str | None:
        """Map a source path's extension to a canonical language string."""
        from pathlib import Path as _P

        if not kernel_url:
            return None
        suffix = _P(kernel_url).suffix.lower()
        return {
            ".py": "triton",  # heuristic — refined by DiscoveryPhase later
            ".hip": "hip",
            ".cu": "cuda",
        }.get(suffix)


__all__ = ["TranslationPhase"]
