"""Unified pipeline entry for fixed + planned + auto + translate modes.

Historically fixed (``run_homogeneous_agent``) and planned
(``run_orchestrator``) took very different paths through the codebase even
though they ultimately drove the same optimisation loop.  This module
collapses the surface area exposed to the CLI:

    final_report = run_pipeline(ctx, mode)

Mode vocabulary (matches the execution plan end state):

  - ``fixed``     — one task body, replicated across ``num_parallel`` copies
                    (was "homogeneous" in the legacy vocabulary).
  - ``planned``   — N planner-generated task bodies (one per strategy)
                    dispatched in parallel (was "heterogeneous").
  - ``auto``      — default.  Picks ``fixed`` or ``planned`` per kernel
                    based on language heuristics (Triton→planned,
                    HIP→fixed); future iterations will let the controller
                    select per-round.
  - ``translate`` — source→target language translation loop (verify-retry);
                    currently raises NotImplementedError until PR-60 lands.

``run_pipeline`` is responsible for resolving the tool set, composing the
task body (via ``run/compose.py``), and dispatching to the shared pool
runner.  For the moment the body still delegates to the existing
``run_orchestrator`` / ``run_homogeneous_agent`` helpers so behavior is byte
compatible; those helpers will be collapsed into ``run/pool_runner.py`` in
a later commit, after which this file becomes the only call site for
ParallelAgent entry points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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


def _resolve_auto_mode(ctx: PipelineContext) -> Mode:
    """Map ``auto`` to a concrete ``fixed``/``planned`` decision.

    Current heuristic: Triton kernels (which benefit from diverse strategy
    planning because they have a rich optimization surface) go through the
    planner; everything else (HIP, CUDA, unknown) defaults to identical
    parallel copies.  Future revisions will let a controller pick per
    round based on KB state and available GPUs.
    """
    discovery = ctx.preprocess_ctx.get("discovery") or {}
    kernel_info = discovery.get("kernel") or {}
    inferred = kernel_info.get("type") or ctx.kernel_language or "unknown"
    if str(inferred).strip().lower() == "triton":
        resolved: Mode = "planned"
    else:
        resolved = "fixed"
    logger.info("auto-mode resolved to %s (kernel_type=%s)", resolved, inferred)
    return resolved


def run_pipeline(ctx: PipelineContext, mode: Mode):
    """Drive one full optimization pipeline and return the final report.

    This is the canonical entry point.  It resolves ``auto`` to a concrete
    mode, resolves tools once, composes the task body once (for ``fixed``),
    and dispatches.  For the moment it still delegates to the legacy
    helpers (``run_orchestrator`` for ``planned``, ``run_homogeneous_agent``
    for ``fixed``) but through the unified shape.
    """
    logger.info(
        "run_pipeline: mode=%s kernel_language=%s output_dir=%s max_rounds=%s rag_enabled=%s",
        mode,
        ctx.kernel_language,
        ctx.output_dir,
        ctx.max_rounds,
        ctx.rag_enabled,
    )

    if mode == "auto":
        mode = _resolve_auto_mode(ctx)

    # Tool resolution happens once regardless of mode.  Planned mode still
    # builds its own ToolRuntime inside
    # ``agents/heterogeneous/orchestrator.py`` — we do not override it
    # here yet to avoid behavior drift; the single-site guarantee becomes
    # enforced once ``run/pool_runner.py`` lands.
    _ = _resolve_tools(ctx, mode)

    if mode == "planned":
        return _run_planned(ctx)
    if mode == "fixed":
        return _run_fixed(ctx)
    if mode == "translate":
        # Hook for the upcoming TranslationLoop.  Until PR-60 lands we
        # surface a clear error rather than silently falling back.
        raise NotImplementedError(
            "mode='translate' is not wired through run_pipeline yet; use `geak translate` directly."
        )
    raise ValueError(f"Unknown pipeline mode: {mode!r}")


def _run_planned(ctx: PipelineContext):
    """Dispatch into the planner-driven parallel path.

    The planner (``task_generator``) composes its own per-task bodies, so
    here we only massage the top-level preprocess context (commandment
    presence check, constraint / directive addenda, rag flag) before
    delegating.
    """
    from minisweagent.run.orchestrator import run_orchestrator

    pctx = dict(ctx.preprocess_ctx)
    commandment = pctx.get("commandment")
    if not commandment:
        # Planned mode requires the commandment because the planner LLM
        # references it per sub-task.  Fixed mode skips this check.
        raise RuntimeError(
            "planned mode requires ``commandment`` in preprocess_ctx; "
            "check preprocessor logs for failures."
        )

    pctx.setdefault("user_instructions", ctx.user_prompt)
    pctx["rag_enabled"] = ctx.rag_enabled
    pctx["output_dir"] = str(ctx.output_dir) if ctx.output_dir else pctx.get("output_dir")

    # Extra addenda (user-specified constraints / directives extracted by
    # the caller) get appended to the commandment so every sub-task sees
    # them.
    if ctx.extra_addenda:
        addendum = "\n\n".join(a.strip() for a in ctx.extra_addenda if a and a.strip())
        if addendum:
            pctx["commandment"] = (commandment + "\n\n" + addendum).strip()
            if ctx.output_dir is not None:
                try:
                    _cm_path = Path(ctx.output_dir) / "COMMANDMENT.md"
                    _cm_path.write_text(pctx["commandment"], encoding="utf-8")
                    logger.info("Enriched commandment written to %s", _cm_path)
                except Exception as exc:
                    logger.warning("Failed to persist enriched commandment: %s", exc)

    # run_orchestrator still takes the legacy ``heterogeneous=True`` kwarg
    # until its internals are renamed in the orchestration PR.  The public
    # API surface here uses the new mode vocabulary.
    return run_orchestrator(
        preprocess_ctx=pctx,
        gpu_ids=ctx.gpu_ids,
        model=ctx.model,
        model_factory=ctx.model_factory,
        output_dir=ctx.output_dir,
        max_rounds=ctx.max_rounds,
        heterogeneous=True,
    )


def _run_fixed(ctx: PipelineContext):
    """Dispatch into the identical-copies parallel path after composing the body."""
    from minisweagent.agents.homogeneous.homogeneous_agent import run_homogeneous_agent

    body = compose_task_body(
        ComposeInputs(
            user_prompt=ctx.user_prompt,
            mode="fixed",
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
