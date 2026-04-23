"""``KernelAnalysisAgent`` — one-shot subagent producing the [A]-[D] analysis rubric.

Per execution plan §0.5(b) Explore phase, the rubric markdown summarises:

    [A] Primitives        — GEMM / reduce / elementwise / scatter / etc.
    [B] Shape Regimes     — tiles, head counts, sequence length bands
    [C] Profile Hotspots  — top-N kernels + roofline position
    [D] Attack Surfaces   — the ordered list of optimisations the
                             optimizer should try, derived from A+B+C

The output is consumed by ``compose_task_body`` in both fixed and
planned modes (prepended to the task body as structured context), and
by the ``CrossSessionMemoryAnalysisAgent`` when deciding whether KB
entries are transferable.

This implementation is a skeleton: the class + config contract are
in place, but the actual LLM call is scheduled for a follow-up
commit.  Until then, ``run()`` raises ``NotImplementedError`` with a
pointer to the legacy commandment generator (which carries a subset
of rubric data today).
"""

from __future__ import annotations

import logging
from typing import Any

from minisweagent.subagents.base import SubagentBase

logger = logging.getLogger(__name__)


class KernelAnalysisAgent(SubagentBase):
    """One-shot producer of the [A]-[D] kernel analysis markdown.

    Subclass override: ``run()``.  One-shot task — no ``loop()``.
    Not composed from ``OptimizationAgent``; uses direct model query.
    """

    def run(self, **inputs: Any) -> str | dict:
        """Produce the analysis rubric markdown.

        Expected inputs (when implemented):
          - ``kernel_code: str``             — full source of the target kernel
          - ``profile: dict | None``         — baseline_metrics + profile.json payload
          - ``codebase_context: str | None`` — CODEBASE_CONTEXT.md contents
          - ``out_path: Path``               — where to write the markdown

        Returns: ``str`` (path to kernel_analysis.md).
        """
        raise NotImplementedError(
            "KernelAnalysisAgent.run is not implemented yet.  Until this "
            "lands, the legacy ``generate_commandment`` in "
            "run/preprocess/commandment.py carries a subset of the rubric "
            "data (primitives + shape regimes only; no attack-surface "
            "ordering).  Full implementation is scheduled for a "
            "subsequent commit of the preprocessing refactor PR."
        )


__all__ = ["KernelAnalysisAgent"]
