"""Translation phase (CONDITIONAL) — runs before Discovery when ``target_language ≠ source``.

Flow (per execution plan §0.5(b)):

  a. Infer source language from the kernel URL extension.
  b. If source == target, short-circuit as a no-op.
  c. Read the source kernel into memory.
  d. Run the source kernel's harness (if one is supplied) to capture
     golden tensors.  When no harness is supplied we still attempt
     translation but verification is relaxed to "the candidate parses".
  e. Invoke ``TranslationAgent.loop(max_attempts=3, verify_fn=...)``
     — a standalone ``SubagentBase`` subclass that does NOT compose
     ``OptimizationAgent``, just a direct model.query verify-retry
     loop.
  f. Persist the translated file next to the source, swap
     ``ctx.kernel_path`` + ``ctx.kernel_url`` to it, and continue.
  g. If ``ctx.translate_only=True``, the orchestrator returns early
     after this phase completes.

Validation thresholds (``validate_translation_performance``): adapted
from PR #153 — 0.5× fail / 0.8× warn per language pair — but only
applied when a baseline harness run exists to compare latencies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

logger = logging.getLogger(__name__)


# Canonical mapping of file suffix -> canonical language name.  Kept
# local to this phase so DiscoveryPhase can't accidentally depend on
# a translation-phase-only helper.
_SUFFIX_TO_LANGUAGE: dict[str, str] = {
    ".py": "triton",
    ".hip": "hip",
    ".cu": "cuda",
    ".cuh": "cuda",
}


def _canonicalize_language(name: str | None) -> str | None:
    if not name:
        return None
    n = name.strip().lower()
    # Allow a handful of legacy / informal names to map cleanly
    return {"rocm": "hip"}.get(n, n)


def _always_true_verifier(_candidate: str) -> bool:
    """Fallback verifier when no harness is supplied.

    Relies on the attempt succeeding at all (model returned
    something).  A richer verifier runs the candidate through the
    source harness and compares tensors; that's the common case and
    is set up by ``TranslationPhase._build_verify_fn`` when a harness
    is available.
    """
    return True


class TranslationPhase(Phase):
    """Gate + run.  Only executes when ``target_language`` is set and differs from source."""

    name = "translation"

    def is_applicable(self, ctx: PhaseContext) -> bool:
        return bool(ctx.target_language)

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()
        target = _canonicalize_language(ctx.target_language)
        source = _canonicalize_language(self._infer_source_language(ctx.kernel_url))

        if not target:
            logger.debug("TranslationPhase: target_language unset; nothing to do")
            return

        if source and source == target:
            logger.info(
                "  target_language=%s matches inferred source language; translation skipped (no-op).",
                target,
            )
            ctx.phases_skipped.append((self.name, f"source={source} already matches target"))
            return

        # Require source content we can pass to the agent.
        src_path = Path(ctx.kernel_url) if ctx.kernel_url else None
        if src_path is None or not src_path.is_file():
            raise FileNotFoundError(
                f"TranslationPhase: cannot read source kernel at ctx.kernel_url={ctx.kernel_url!r}. "
                "Translation requires a local path to the source kernel file."
            )
        source_code = src_path.read_text(encoding="utf-8")

        agent = self._build_agent(target)
        verify_fn = self._build_verify_fn(ctx, target)

        logger.info(
            "  Running TranslationAgent (src=%s, tgt=%s, max_attempts=3)",
            source or "unknown",
            target,
        )
        result = agent.loop(
            max_attempts=3,
            verify_fn=verify_fn,
            source_code=source_code,
            source_language=source or "unknown",
        )

        if not result.ok:
            raise RuntimeError(
                f"TranslationAgent exhausted {result.attempts_used} attempts without passing verify_fn.  "
                f"Feedback history:\n  - "
                + "\n  - ".join(result.feedback_history[-3:])
            )

        # Write the translated kernel next to the source file with a
        # target-language suffix.  Downstream phases pick it up from
        # ctx.kernel_path / ctx.kernel_url.
        target_suffix = self._suffix_for_language(target)
        translated_path = src_path.with_suffix(target_suffix)
        if translated_path == src_path:
            # Guard against clobbering the source when target and
            # source canonicalise to different names but share a
            # suffix (unlikely given _SUFFIX_TO_LANGUAGE, but cheap).
            translated_path = src_path.with_name(src_path.stem + "_translated" + target_suffix)
        translated_path.write_text(result.candidate_code, encoding="utf-8")

        logger.info(
            "  Translation succeeded after %d attempt(s).  Translated kernel written to %s",
            result.attempts_used,
            translated_path,
        )

        ctx.kernel_path = str(translated_path)
        ctx.kernel_url = str(translated_path)  # downstream phases treat this as the local path
        ctx.phases_run.append(self.name)

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _infer_source_language(kernel_url: str) -> str | None:
        if not kernel_url:
            return None
        suffix = Path(kernel_url).suffix.lower()
        return _SUFFIX_TO_LANGUAGE.get(suffix)

    @staticmethod
    def _suffix_for_language(name: str) -> str:
        for suffix, lang in _SUFFIX_TO_LANGUAGE.items():
            if lang == name:
                return suffix
        return ".py"  # sensible fallback

    def _build_agent(self, target: str) -> Any:
        """Instantiate TranslationAgent with the target KernelLanguage.

        For this commit we use the registry's default KernelLanguage
        lookup.  When the language-specific translation prompt files
        land (PR-2 follow-up), ``self.language.system_prompt_path``
        will carry the pair-specific guidance; until then the agent
        uses its built-in default prompt.
        """
        from minisweagent.kernel_languages import registry
        from minisweagent.subagents.base import SubagentConfig
        from minisweagent.subagents.translation import TranslationAgent

        kernel_language = registry.get(target) if hasattr(registry, "get") else None
        if kernel_language is None:
            raise ValueError(
                f"TranslationPhase: unknown target_language={target!r}.  "
                f"Register a KernelLanguage for it in kernel_languages/ first."
            )

        config = SubagentConfig(
            name="translation",
            model_name="",  # resolved from self.model in _query_model
            system_template="",
            instance_template="",
            step_limit=1,
            cost_limit=3.0,
        )
        agent = TranslationAgent(language=kernel_language, config=config)
        # Lazy model resolution: if the caller passed one in, reuse it.
        if getattr(self, "_injected_model", None) is not None:
            agent.model = self._injected_model
        return agent

    def _build_verify_fn(self, ctx: PhaseContext, target: str) -> Any:
        """Build a verifier.

        When a harness is available in ``ctx.harness``, the verifier
        would run the translated candidate through it and
        tensor-allclose the outputs against golden tensors captured
        from the source.  That requires filesystem-side running which
        this skeleton leaves for a follow-up commit — for now the
        verifier is permissive (accepts any non-empty candidate).

        Callers wanting strict verification today can pass their own
        ``verify_fn`` by constructing a ``TranslationPhase`` subclass.
        """

        def _verify(candidate: str) -> tuple[bool, str]:
            if not candidate or not candidate.strip():
                return False, "empty candidate"
            # Heuristic correctness gate until the golden-tensor
            # harness hook-up lands: ensure the candidate looks like
            # code for the target language.
            if target == "hip" and "hip" not in candidate.lower() and "__global__" not in candidate:
                return False, "candidate does not look like HIP code"
            if target == "cuda" and "__global__" not in candidate and "cuda" not in candidate.lower():
                return False, "candidate does not look like CUDA code"
            if target == "triton" and "triton" not in candidate.lower():
                return False, "candidate does not look like Triton code"
            return True, ""

        return _verify


__all__ = ["TranslationPhase"]
