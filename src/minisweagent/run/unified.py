"""Unified pipeline entry for homogeneous + heterogeneous + translate modes.

Historically homogeneous (``run_homogeneous_agent``) and heterogeneous
(``run_orchestrator``) took very different paths through the codebase even
though they ultimately drove the same optimisation loop.  This module
collapses the surface area exposed to ``run/mini.py``:

    final_report = run_pipeline(ctx, mode)

``run_pipeline`` is responsible for resolving the tool set, composing the
task body (via ``run/compose.py``), and dispatching to the shared pool
runner.  For the moment the body still delegates to the existing
``run_orchestrator`` / ``run_homogeneous_agent`` helpers so behavior is byte
compatible; those helpers will be collapsed into ``run/pool_runner.py`` in
the next commit, after which this file becomes the only call site for
ParallelAgent entry points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from minisweagent.run.compose import ComposeInputs, Mode, compose_task_body

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Context passed through ``run_pipeline``.

    This is a light wrapper over the existing dict-shaped ``preprocess_ctx``
    to keep the migration tax low.  Fields added here are the *new* pieces
    the unified path needs that were previously scattered across call sites
    (tool profile, RAG toggle, GPU ids, model factory).
    """

    preprocess_ctx: dict[str, Any]
    user_prompt: str
    kernel_language: str | None = None
    output_dir: Path | None = None
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    model: Any = None
    model_factory: Callable[[], Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    max_rounds: int | None = None
    env: Any = None
    env_class: Any = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    repo: Path | None = None
    test_command: str | None = None
    metric: str | None = None
    rag_enabled: bool = False
    extra_addenda: list[str] = field(default_factory=list)
    num_parallel: int | None = None
    model_name: str | None = None
    console: Any = None


# ── Tool resolution ───────────────────────────────────────────────────
#
# Single site for deciding which tools each mode exposes to the agent.
# This replaces the scattered per-call-site ``ToolRuntime(tool_profile=...,
# use_strategy_manager=...)`` constructions.


def _resolve_tools(ctx: PipelineContext, mode: Mode):
    """Return a ``ToolRuntime`` instance configured for ``mode``.

    Kept as a free function so callers can inspect the resolved tool set
    (for tests, debugging, the `one resolution site' CI gate).
    """
    from minisweagent.tools.tools_runtime import ToolRuntime

    runtime = ToolRuntime(
        tool_profile="full",
        use_strategy_manager=True,
    )

    if not ctx.rag_enabled:
        runtime.disable_tools(["query", "optimize"])
    else:
        try:
            runtime.wrap_rag_tools_with_postprocessor()
        except Exception as exc:
            logger.warning("Failed to wrap RAG tools with postprocessor: %s", exc)

    logger.debug("_resolve_tools: mode=%s rag=%s", mode, ctx.rag_enabled)
    return runtime


# ── Pipeline dispatch ─────────────────────────────────────────────────


def run_pipeline(ctx: PipelineContext, mode: Mode):
    """Drive one full optimization pipeline and return the final report.

    This is the canonical entry point.  For the moment it still delegates
    to the legacy helpers (``run_orchestrator`` for heterogeneous,
    ``run_homogeneous_agent`` for homogeneous) but first:
      - composes the task body once (homogeneous path),
      - resolves tools once,
      - logs a unified banner for observability.
    """
    logger.info(
        "run_pipeline: mode=%s kernel_language=%s output_dir=%s max_rounds=%s rag_enabled=%s",
        mode,
        ctx.kernel_language,
        ctx.output_dir,
        ctx.max_rounds,
        ctx.rag_enabled,
    )

    # Tool resolution happens once regardless of mode.  Heterogeneous mode
    # currently builds its own ToolRuntime inside
    # `agents/heterogeneous/orchestrator.py` — we do not override it here
    # yet to avoid behavior drift; the single-site guarantee becomes
    # enforced once `run/pool_runner.py` lands.
    _ = _resolve_tools(ctx, mode)

    if mode == "heterogeneous":
        return _run_heterogeneous(ctx)
    if mode == "homogeneous":
        return _run_homogeneous(ctx)
    if mode == "translate":
        # Hook for the upcoming TranslationLoop.  Until PR-60 lands we
        # surface a clear error rather than silently falling back.
        raise NotImplementedError(
            "mode='translate' is not wired through run_pipeline yet; use `geak translate` directly."
        )
    raise ValueError(f"Unknown pipeline mode: {mode!r}")


def _run_heterogeneous(ctx: PipelineContext):
    """Dispatch into the existing heterogeneous path.

    The heterogeneous planner (``task_generator``) composes its own
    per-task bodies, so we only massage the top-level preprocess context
    (constraint / directive addenda, rag flag) before delegating.
    """
    from minisweagent.run.orchestrator import run_orchestrator

    pctx = dict(ctx.preprocess_ctx)
    pctx["user_instructions"] = ctx.user_prompt
    pctx["rag_enabled"] = ctx.rag_enabled
    pctx["output_dir"] = str(ctx.output_dir) if ctx.output_dir else pctx.get("output_dir")

    # Extra addenda (user-specified constraints / directives extracted by
    # the caller) get appended to the commandment so every sub-task sees
    # them.
    if ctx.extra_addenda:
        addendum = "\n\n".join(a.strip() for a in ctx.extra_addenda if a and a.strip())
        if addendum:
            base = pctx.get("commandment") or ""
            pctx["commandment"] = (base + "\n\n" + addendum).strip()

    return run_orchestrator(
        preprocess_ctx=pctx,
        gpu_ids=ctx.gpu_ids,
        model=ctx.model,
        model_factory=ctx.model_factory,
        output_dir=ctx.output_dir,
        max_rounds=ctx.max_rounds,
        heterogeneous=True,
    )


def _run_homogeneous(ctx: PipelineContext):
    """Dispatch into the existing homogeneous path after composing the body."""
    from minisweagent.agents.homogeneous.homogeneous_agent import run_homogeneous_agent

    body = compose_task_body(
        ComposeInputs(
            user_prompt=ctx.user_prompt,
            mode="homogeneous",
            preprocess_ctx=ctx.preprocess_ctx,
            kernel_language=ctx.kernel_language,
            extra_addenda=ctx.extra_addenda,
        )
    )

    agent_config = dict(ctx.config.get("agent", {}))
    agent_config["save_patch"] = True
    if ctx.test_command is not None:
        agent_config["test_command"] = ctx.test_command
    if ctx.metric is not None:
        agent_config["metric"] = ctx.metric
    if ctx.output_dir is not None:
        agent_config["patch_output_dir"] = str(ctx.output_dir)

    kwargs: dict[str, Any] = dict(
        config=ctx.config,
        task_content=body,
        model=ctx.model,
        env=ctx.env,
        env_class=ctx.env_class,
        env_kwargs=ctx.env_kwargs,
        agent_config=agent_config,
        repo=ctx.repo,
    )
    if ctx.num_parallel is not None:
        kwargs["num_parallel"] = ctx.num_parallel
    if ctx.gpu_ids:
        # run_homogeneous_agent takes a string and re-parses internally;
        # re-serialize the canonical list[int] form.
        kwargs["gpu_ids"] = ",".join(str(g) for g in ctx.gpu_ids)
    if ctx.output_dir is not None:
        kwargs["output_dir"] = ctx.output_dir
    if ctx.model_name is not None:
        kwargs["model_name"] = ctx.model_name
    if ctx.console is not None:
        kwargs["console"] = ctx.console

    return run_homogeneous_agent(**kwargs)


__all__ = ["PipelineContext", "run_pipeline"]
