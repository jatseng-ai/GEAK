"""HIP KernelLanguage instance.

Registered at import time.
"""

from __future__ import annotations

from pathlib import Path

from minisweagent.kernel_languages import registry
from minisweagent.kernel_languages.base import KernelLanguage

_DIR = Path(__file__).parent

HIP = KernelLanguage(
    name="hip",
    # HIP wrappers are typically .py (pybind11 bindings, torch.utils.cpp_extension
    # wrappers), or raw .cu / .hip / .cpp. Triton also claims .py — the
    # detect_hints disambiguate.
    file_extensions=frozenset({".py", ".cu", ".hip", ".cpp", ".cxx"}),
    detect_hints=(
        r"__global__\s+void\b",
        r"hipLaunchKernelGGL\b",
        r"\bhip[A-Z]\w*\(",             # hipMalloc, hipMemcpy, etc.
        r'#include\s*[<"]hip/hip_runtime\.h',
        r"torch\.utils\.cpp_extension",
        r"scripts/task_runner\.py.*(compile|correctness|performance)",
    ),
    kb_namespace="hip",
    system_prompt_path=_DIR / "system_prompt.md" if (_DIR / "system_prompt.md").exists() else None,
    optimization_prompt_path=_DIR / "optimization_prompt.md" if (_DIR / "optimization_prompt.md").exists() else None,
    planner_strategy_hints_path=_DIR / "planner_strategy_hints.md" if (_DIR / "planner_strategy_hints.md").exists() else None,
    harness_template_path=_DIR / "harness.j2" if (_DIR / "harness.j2").exists() else None,
    commandment_template_path=_DIR / "commandment.j2" if (_DIR / "commandment.j2").exists() else None,
    tool_set=frozenset(),
)

registry.register(HIP)


__all__ = ["HIP"]
