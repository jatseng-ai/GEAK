"""`KernelLanguage` — central data object describing everything language-specific.

Every language (Triton, HIP, future additions) ships as a folder under
`kernel_languages/<name>/` with a `kernel_language.py` that instantiates this
dataclass. The dataclass holds PATHS to prompts and Jinja templates (lazy-loaded)
plus small metadata fields.

Per docs/refactor/EXECUTION_PLAN.md §16.1. Notable design choices:

- **Frozen dataclass**: prevents accidental mutation after detection.
- **Paths not strings**: prompt / template content is lazy-loaded via helpers
  so that multi-KB markdown doesn't cost startup time.
- **No `test_runner_command` / `profiler_command` fields**: these were
  redundant with `commandment.j2` and introduced drift risk (two sources of
  truth). The harness CLI surface is UNIVERSAL (`python3 {harness} --correctness|
  --benchmark|--full-benchmark|--profile` for every language, enforced by
  HarnessBuilder + validate_harness contract). Language-specific setup (HIP's
  `make`) and profiler invocation live ONLY in `commandment.j2`.
- **No hardcoded package names for fingerprinting**: the `baseline_fingerprint`
  mechanism was removed (see §15 Issue 4 resolution) — staleness detection is
  now content-based inside CrossSessionMemoryAnalysisAgent, not hash-based.

Minimal field set for PR-1: name, file_extensions, detect_hints, kb_namespace,
prompt/template paths. Extended fields (idioms, builder_hints, memory_hints,
translation_hints) land in later commits as they become required by consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import FrozenSet


@dataclass(frozen=True)
class KernelLanguage:
    """Describes one kernel language (Triton, HIP, FlyDSL, …).

    Single source of truth consumed by:
      - detection: `registry.detect_best(path)` -> KernelLanguage
      - preprocess: phases read `ctx.language.*` for templates/prompts
      - run_pipeline: `_resolve_tools(ctx.language.tool_set, mode)`
      - compose_task_body: assembles prompt from language's markdown files
    """

    # ─── identity ───
    name: str
    """Unique language identifier: 'triton', 'hip', 'flydsl', ..."""

    file_extensions: FrozenSet[str]
    """File extensions this language claims, e.g. frozenset({'.py'}) for Triton.
    Used by `registry.detect_best(path)` as a prefilter."""

    detect_hints: tuple[str, ...] = ()
    """Regex patterns that boost detection confidence when scanning kernel
    source. Triton: (r'@triton\\.jit', r'^import triton'); HIP:
    (r'__global__\\s+void', r'hipLaunchKernelGGL'). Used to disambiguate when
    multiple languages could plausibly match the same file extension."""

    # ─── memory / KB ───
    kb_namespace: str = ""
    """Cross-session memory namespace. KB queries filter entries by this name
    so that 'triton fast_rms_fp8' memories don't get returned for HIP kernels.
    Defaults to the language `name` if empty at construction — but since the
    dataclass is frozen we require callers to set it explicitly."""

    # ─── prompt / template paths (lazy-loaded) ───
    system_prompt_path: Path | None = None
    """Path to the system prompt markdown (fed to OptimizationAgent's system
    template). Replaces today's hardcoded SYSTEM_PROMPT in heterogeneous/prompts.py."""

    optimization_prompt_path: Path | None = None
    """Path to the per-round task instruction template."""

    planner_strategy_hints_path: Path | None = None
    """Path to planner strategy hints (replaces today's TASKGEN_SYSTEM_PROMPT
    language-biased content in heterogeneous/prompts.py)."""

    harness_template_path: Path | None = None
    """Path to Jinja template for harness.py (consumed by HarnessBuilder
    subagent; lands in PR-2)."""

    commandment_template_path: Path | None = None
    """Path to Jinja template for COMMANDMENT.md. SINGLE SOURCE OF TRUTH for
    Setup/Correctness/Benchmark/FullBenchmark/Profile commands across the
    pipeline. Per-language quirks (HIP's `make`, `rocprof`, Triton's `python3`,
    Metrix profiler calls) live ONLY here — there are deliberately no
    `test_runner_command` or `profiler_command` fields on this dataclass."""

    # ─── tools (populated in PR-3) ───
    tool_set: FrozenSet[str] = frozenset()
    """Tool names the main agent receives for this language. Examples:
    Triton: frozenset({'bash', 'str_replace_editor', 'save_and_test', 'submit',
                       'strategy_manager', 'profile_kernel', 'query', 'optimize',
                       'baseline_metrics', 'resolve_kernel_url', 'sub_agent'}).
    Empty default means 'use whatever tools_runtime selects by default' — safe
    for migration. PR-3 populates per-language sets and flips CI gate."""

    # ─── lazy-loaded content helpers ───
    # Not @cached_property on a frozen dataclass — frozen dataclasses disallow
    # setattr. We load-and-return each time; cost is a file read (<1 KB typical).

    def _load(self, p: Path | None) -> str:
        if p is None:
            return ""
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")

    @property
    def system_prompt(self) -> str:
        return self._load(self.system_prompt_path)

    @property
    def optimization_prompt(self) -> str:
        return self._load(self.optimization_prompt_path)

    @property
    def planner_strategy_hints(self) -> str:
        return self._load(self.planner_strategy_hints_path)

    @property
    def harness_template(self) -> str:
        return self._load(self.harness_template_path)

    @property
    def commandment_template(self) -> str:
        return self._load(self.commandment_template_path)

    def __repr__(self) -> str:
        return f"KernelLanguage(name={self.name!r}, kb_namespace={self.kb_namespace!r})"
