"""Triton KernelLanguage instance.

Registered at import time. Prompt/template paths point to files in this folder
(some land later as content is migrated from the legacy locations).
"""

from __future__ import annotations

from pathlib import Path

from minisweagent.kernel_languages import registry
from minisweagent.kernel_languages.base import KernelLanguage

_DIR = Path(__file__).parent

TRITON = KernelLanguage(
    name="triton",
    file_extensions=frozenset({".py"}),
    detect_hints=(
        r"@triton\.jit\b",
        r"^import\s+triton",
        r"\bfrom\s+triton\b",
        r"\btl\.load\s*\(",
        r"\btl\.store\s*\(",
    ),
    kb_namespace="triton",
    # Paths that land later. None = "not yet populated"; the base class helpers
    # gracefully return "" from the corresponding property accessors.
    system_prompt_path=_DIR / "system_prompt.md" if (_DIR / "system_prompt.md").exists() else None,
    optimization_prompt_path=_DIR / "optimization_prompt.md" if (_DIR / "optimization_prompt.md").exists() else None,
    planner_strategy_hints_path=_DIR / "planner_strategy_hints.md" if (_DIR / "planner_strategy_hints.md").exists() else None,
    harness_template_path=_DIR / "harness.j2" if (_DIR / "harness.j2").exists() else None,
    commandment_template_path=_DIR / "commandment.j2" if (_DIR / "commandment.j2").exists() else None,
    tool_set=frozenset(),   # populated in PR-3; empty = use tools_runtime defaults
)

registry.register(TRITON)


__all__ = ["TRITON"]
