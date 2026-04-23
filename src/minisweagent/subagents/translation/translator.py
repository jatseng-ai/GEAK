"""``TranslationAgent`` — standalone verify-retry subagent for kernel language porting.

Architectural contract (per execution plan §0.5(b) and the user's explicit
direction):

  - Translation is a **preprocess phase**, not a ``run_pipeline`` mode.
    It runs only when the user asks for a target language different
    from the source (``target_language``).  After the phase completes,
    ``ctx.kernel_path`` and ``ctx.language`` are swapped to the
    translated kernel and the normal fixed / planned / auto pipeline
    continues (or the pipeline exits, if ``translate_only`` was set).

  - ``TranslationAgent`` is a ``SubagentBase`` subclass overriding
    ``loop()`` (the multi-round verify-retry entry point).  It is
    **deliberately not derived from ``OptimizationAgent``** and does
    **not** compose ``OptimizationAgent`` via
    ``_make_optimization_agent``.  Translation is a narrow,
    verifier-gated task that does not need the tool runtime, strategy
    manager, RAG wrapping, or patch-apply machinery that
    ``OptimizationAgent`` carries — it needs only a model query loop
    plus a tensor ``allclose`` verifier.  Keeping the two agents
    separate matches the same principle you applied to the preprocess
    subagents (SelectPatchAgent, UnitTestAgent, ShapeFixerAgent).

  - Success criterion: ``verify_fn(translated_output)`` returns True,
    where ``verify_fn`` is normally a tensor ``allclose(src_out,
    tgt_out, atol=1e-5)`` against golden tensors captured by running
    the source harness once at the start of the phase.

Implementation is intentionally deferred: the class + contract are in
place here so that the rest of the design (Mode literal, task_parser,
prompts) can be correct today.  The translation preprocess phase that
drives this agent, together with the golden-tensor capture, will be
filled in by the preprocessing refactor PR (PR-2 / PR-60 in the plan).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from minisweagent.subagents.base import SubagentBase

logger = logging.getLogger(__name__)


class TranslationAgent(SubagentBase):
    """Verify-retry subagent that rewrites a kernel in a different language.

    Not yet implemented.  The shape is fixed:

        agent = TranslationAgent(language=target_kernel_language, config=...)
        result = agent.loop(
            max_attempts=3,
            verify_fn=lambda candidate: tensor_allclose(golden, run(candidate)),
            source_code=...,
            source_language=...,
            golden_tensors=...,
        )

    ``self.language`` holds the TARGET ``KernelLanguage`` (that is the
    language the agent is writing code in).  The SOURCE language is
    passed via ``**inputs`` because the subagent's "home" language is
    the one whose prompts / templates it uses.
    """

    def loop(
        self,
        *,
        max_attempts: int,
        verify_fn: Callable[[Any], bool],
        **inputs: Any,
    ) -> Any:
        """Retry-until-verified translation loop.

        Pseudocode (to be implemented):

            for attempt in range(max_attempts):
                sys_p, inst_p = self._compose_prompt(
                    source_language=inputs["source_language"],
                    source_code=inputs["source_code"],
                    last_failure=last_feedback,  # None on first attempt
                )
                candidate = self._query_model(sys_p, inst_p)
                if verify_fn(candidate):
                    return candidate
                last_feedback = _summarize_mismatch(candidate, verify_fn)
            raise TranslationFailed(f"{max_attempts} attempts exhausted")

        The ``_query_model`` call is a direct ``model.query(...)`` — no
        OptimizationAgent step loop, no tools.  Translation is one
        prompt → one candidate → verify; iteration is at the
        *candidate* level, not the *step* level.
        """
        raise NotImplementedError(
            "TranslationAgent.loop is not implemented yet.  It is scheduled "
            "for the preprocessing refactor PR (see preprocess/phases/"
            "translation.py).  See this module's docstring for the intended "
            "architecture."
        )


__all__ = ["TranslationAgent"]
