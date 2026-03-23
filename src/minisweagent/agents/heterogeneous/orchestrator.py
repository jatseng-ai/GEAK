"""Heterogeneous orchestrator: LLM-driven multi-round optimization.

In heterogeneous mode, an LLM agent drives the optimization loop by
calling tools in sequence each round:

1. ``generate_tasks`` -- create diverse optimization task files
2. ``dispatch_tasks`` -- run them in parallel across GPUs
3. ``collect_results`` -- review what each task achieved
4. ``finalize`` -- signal completion (final round only)

The LLM decides what strategies to try each round based on profiling
data, prior results, and COMMANDMENT constraints.  The round loop is
implicit -- the LLM's behavior drives iteration, not a Python for-loop.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from minisweagent.debug_runtime import emit_debug_log, model_tools_snapshot

logger = logging.getLogger(__name__)


# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the GEAK orchestrator – an expert at planning and coordinating
GPU kernel optimisation.

You have been given the results of a preprocessing pipeline:
* Profiling data with per-kernel bottleneck analysis
* Baseline metrics (duration, throughput, bottleneck classification)
* A COMMANDMENT.md that specifies the rules every sub-agent must follow

You also have access to **bash** (execute shell commands),
**str_replace_editor** (view / edit files), **profile_kernel** (GPU
profiling), and **strategy_manager**.  Use these only when you need to
inspect artefacts, debug a failure, or gather information the
orchestration tools above cannot provide.

## IMPORTANT: Phased Execution

The orchestration runs in TWO phases:

### Phase 1: Exploration (current phase)
During exploration, you should ONLY:
- Read and understand the kernel source code
- Review profiling data and baseline metrics
- Analyze the COMMANDMENT.md
- Plan your optimization strategy

Do NOT call generate_tasks, dispatch_tasks, collect_results, or finalize
during exploration. Simply respond with "Ready to begin optimization rounds"
when you have finished exploring.

### Phase 2: Round Loop
The system will explicitly tell you "Begin round N" to start each round.
WAIT for this instruction before calling any orchestration tools.

Within each round you MUST call these tools in order:
1. **generate_tasks** – produce optimisation task files for this round.
2. **dispatch_tasks** – run those tasks in parallel across available GPUs.
3. **collect_results** – review what each task achieved.

After collect_results, respond with your evaluation and WAIT for the next
round instruction. The system will automatically run validation (FULL_BENCHMARK
and PROFILE) on the best kernel from each round.

Only call **finalize** when the system tells you it is the FINAL round.
The finalize call should include:
- summary: A comprehensive summary of optimizations achieved
- best_patch: Path to the best patch file
- total_speedup: The verified speedup (e.g., "1.06x" or "6%")

Rules:
- Do NOT modify preprocessor artefacts (test harness, test command,
  discovery, profiling, COMMANDMENT.md).
- Do NOT run tasks yourself; always dispatch via **dispatch_tasks**.
- Do NOT call finalize until explicitly told it is the FINAL round.
- After **collect_results**, review each sub-agent's output against
  its original task intent:
  1. Did it actually optimise the *kernel*, or did it modify something
     else (e.g. test harness, benchmark framework)?  Reject the latter.
  2. Did it report a before/after performance comparison using baseline
     metrics?  If not, note that the result is unverified.
  3. Did it violate the COMMANDMENT?  Reject if so.
  4. Did the correctness tests pass?  Reject if tests failed.
  Mark rejected results as "rejected" and explain why.
- For cross-round decisions, treat the system-provided FULL_BENCHMARK
  evaluation as canonical. Raw task-local speedups are provisional and
  may be noisy or invalidated by later verification.
"""

INSTANCE_TEMPLATE = """\
## Preprocessor Context

Kernel: {kernel_path}
Repo root: {repo_root}
Test command: {test_command}
Available GPUs: {gpu_ids}
Output directory: {output_dir}

### Codebase Context (repo structure and key files)
{codebase_context}

### Baseline Metrics
{baseline_metrics_summary}

### Profiling Summary
{profiling_summary}

### COMMANDMENT (rules for sub-agents)
{commandment_excerpt}

{memory_context}

---

Begin by reading the kernel source and profiling data to understand the
optimisation landscape.  Then follow the round instructions.
"""


# ── Tool implementations ─────────────────────────────────────────────


def tool_generate_tasks(
    ctx: dict[str, Any],
    round_num: int = 1,
    previous_results_dir: str | None = None,
    **_extra,
) -> str:
    """Generate optimisation tasks for a given round.

    Returns a JSON string with a ``tasks`` key listing the created task file paths.
    """
    from minisweagent.agents.heterogeneous.task_generator import generate_tasks as _gen

    output_dir = Path(ctx["output_dir"]) / "tasks" / f"round_{round_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    prev_dir = None
    if previous_results_dir:
        prev_dir = Path(previous_results_dir)
    elif round_num > 1:
        prev_dir = Path(ctx["output_dir"]) / "results" / f"round_{round_num - 1}"

    emit_debug_log(
        "heterogeneous_orchestrator:tool_generate_tasks:before_gen",
        "Invoking task generator with orchestrator model",
        {
            "round_num": round_num,
            "prev_dir": str(prev_dir),
        },
        hypothesis_id="H0",
    )

    try:
        task_files = _gen(
            discovery_result=ctx.get("discovery_result"),
            output_dir=output_dir,
            model=ctx["model"],
            agent_class=ctx.get("agent_class"),
            kernel_path=ctx.get("kernel_path"),
            repo_root=ctx.get("repo_root"),
            test_command=ctx.get("test_command"),
            commandment_path=ctx.get("commandment_path"),
            baseline_metrics_path=ctx.get("baseline_metrics_path"),
            profiling_path=ctx.get("profiling_path"),
            codebase_context_path=ctx.get("codebase_context_path"),
            previous_round_dir=prev_dir,
            starting_patch=ctx.get("starting_patch"),
            gpu_ids=ctx.get("gpu_ids"),
        )
    except Exception as gen_exc:
        if "LimitsExceeded" in type(gen_exc).__name__:
            logger.warning(
                "Task generator hit limits (round %d), treating as convergence: %s",
                round_num,
                gen_exc,
            )
            return json.dumps({"tasks": [], "convergence": True, "reason": str(gen_exc)})
        raise

    emit_debug_log(
        "heterogeneous_orchestrator:tool_generate_tasks:after_gen",
        "Task generator completed",
        {
            "round_num": round_num,
            "task_count": len(task_files),
        },
        hypothesis_id="H0",
    )

    result: dict[str, Any] = {
        "tasks": [str(f) for f in task_files],
        "round": round_num,
        "output_dir": str(output_dir),
    }

    if task_files:
        from minisweagent.run.task_file import read_task_file
        summaries = []
        for tf in task_files[:10]:
            meta, body = read_task_file(tf)
            summaries.append({
                "file": str(tf),
                "label": meta.get("label", tf.stem),
                "agent_type": meta.get("agent_type", "strategy_agent"),
                "priority": meta.get("priority", 10),
                "num_gpus": meta.get("num_gpus", 1),
                "round": round_num,
            })
        result["task_summaries"] = summaries
    return json.dumps(result, default=str)


def _dispatch_stage_name(priority: int) -> str:
    if priority <= 5:
        return "high"
    if priority <= 10:
        return "medium"
    return "low"


def _group_task_files_by_dispatch_stage(task_files: list[Path]) -> list[tuple[str, list[Path]]]:
    """Group tasks by priority tier for staged dispatch."""
    from minisweagent.run.task_file import read_task_file

    buckets: dict[str, list[Path]] = {}
    for tf in task_files:
        meta, _ = read_task_file(tf)
        pri = int(meta.get("priority", 10))
        stage = _dispatch_stage_name(pri)
        buckets.setdefault(stage, []).append(tf)
    order = ["high", "medium", "low"]
    return [(s, buckets[s]) for s in order if s in buckets]


def _stage_found_improvement(results_dir: Path, task_files: list[Path]) -> bool:
    """Return True if any task in the stage produced speedup > 1.0."""
    for task_file in task_files:
        meta, _ = __import__("minisweagent.run.task_file", fromlist=["read_task_file"]).read_task_file(task_file)
        label = str(meta.get("label") or task_file.stem)
        best_results_path = results_dir / label / "best_results.json"
        if not best_results_path.is_file():
            continue
        try:
            payload = json.loads(best_results_path.read_text())
            if float(payload.get("best_patch_speedup", 0) or 0) > 1.0:
                return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return False


def tool_dispatch_tasks(
    ctx: dict[str, Any],
    task_files: list[str] | None = None,
    **_extra,
) -> str:
    """Dispatch task files to GPUs for parallel execution.

    Returns a JSON summary of completed results.
    """
    from minisweagent.run.dispatch import run_task_batch

    output_dir = Path(ctx["output_dir"])
    gpu_ids = ctx.get("gpu_ids", [0])

    if not task_files:
        tasks_base = output_dir / "tasks"
        if tasks_base.is_dir():
            round_dirs = sorted(
                (d for d in tasks_base.iterdir() if d.is_dir() and d.name.startswith("round_")),
                key=lambda d: d.name,
            )
            if round_dirs:
                task_files = sorted(str(f) for f in round_dirs[-1].glob("*.md"))

    if not task_files:
        return json.dumps({"error": "No task files found"})

    task_paths = [Path(f) for f in task_files]
    stages = _group_task_files_by_dispatch_stage(task_paths)
    round_match = None
    for tf in task_paths[:1]:
        for part in tf.parts:
            if part.startswith("round_"):
                round_match = part
                break
    results_base = output_dir / "results" / (round_match or "round_1")

    all_results: list[dict] = []
    for stage_name, stage_tasks in stages:
        stage_result = run_task_batch(
            task_files=stage_tasks,
            gpu_ids=gpu_ids,
            output_dir=results_base,
            model_factory=ctx.get("model_factory"),
            starting_patch=ctx.get("starting_patch"),
        )
        all_results.append({
            "stage": stage_name,
            "tasks": len(stage_tasks),
            "result": stage_result if isinstance(stage_result, dict) else str(stage_result),
        })
        if _stage_found_improvement(results_base, stage_tasks):
            for remaining_stage, remaining_tasks in stages:
                if remaining_stage == stage_name:
                    continue
                if _dispatch_stage_name(0) == remaining_stage:
                    continue
            break

    return json.dumps({
        "status": "completed",
        "results_dir": str(results_base),
        "stages": all_results,
    }, default=str)


def tool_collect_results(
    ctx: dict[str, Any],
    results_dir: str | None = None,
    **_extra,
) -> str:
    """Read and summarize results from completed tasks."""
    output_dir = Path(ctx["output_dir"])
    if results_dir:
        base = Path(results_dir)
    else:
        base = output_dir / "results"
        if base.is_dir():
            round_dirs = sorted(
                (d for d in base.iterdir() if d.is_dir() and d.name.startswith("round_")),
                key=lambda d: d.name,
            )
            base = round_dirs[-1] if round_dirs else base

    from minisweagent.agents.heterogeneous.task_generator import _scan_previous_results

    summaries = _scan_previous_results(base)
    return "\n\n".join(summaries) if summaries else "No results found."


def tool_finalize(
    ctx: dict[str, Any],
    summary: str,
    best_patch: str | None = None,
    total_speedup: str | None = None,
    **_extra,
) -> str:
    """Signal optimisation is complete.  Write final report.

    If best_patch or total_speedup are not provided, attempts to auto-detect
    them from the results directory.
    """
    output_dir = Path(ctx["output_dir"])

    if best_patch is None or total_speedup is None:
        best_speedup_val = 0.0
        best_patch_file = None

        results_dir = output_dir / "results"
        if results_dir.is_dir():
            for round_dir in sorted(results_dir.iterdir()):
                if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
                    continue
                for task_dir in sorted(round_dir.iterdir()):
                    if not task_dir.is_dir() or task_dir.name == "worktrees":
                        continue
                    br_file = task_dir / "best_results.json"
                    if br_file.exists():
                        try:
                            br = json.loads(br_file.read_text())
                            speedup = float(br.get("best_patch_speedup", 0))
                            if speedup > best_speedup_val:
                                best_speedup_val = speedup
                                best_patch_file = br.get("best_patch_file")
                        except (json.JSONDecodeError, ValueError, TypeError):
                            continue

        if best_patch is None and best_patch_file:
            best_patch = best_patch_file
        if total_speedup is None and best_speedup_val > 0:
            total_speedup = f"{best_speedup_val:.4f}x"

    report = {
        "status": "complete",
        "summary": summary,
        "best_patch": best_patch,
        "total_speedup": total_speedup,
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2, default=str))
    return json.dumps(report, default=str)


# ── Tool schemas for the LLM ─────────────────────────────────────────

_ORCHESTRATOR_SWE_TOOLS = {"bash", "str_replace_editor", "profile_kernel", "strategy_manager"}

_ORCHESTRATOR_ONLY_TOOLS: list[dict] = [
    {
        "name": "generate_tasks",
        "description": (
            "Generate optimisation task files for a round.  Returns a JSON "
            "object with a 'tasks' list of file paths.  An empty list means "
            "convergence – no more optimisations to try."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "round_num": {
                    "type": "integer",
                    "description": "Round number (1-based).",
                },
                "previous_results_dir": {
                    "type": "string",
                    "description": "Path to previous round's results directory (optional for round 1).",
                },
            },
            "required": ["round_num"],
        },
    },
    {
        "name": "dispatch_tasks",
        "description": (
            "Dispatch a list of task files to available GPUs for parallel "
            "execution.  Returns a JSON summary of results.  If task_files "
            "is omitted, auto-discovers from the latest round's task directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of task file paths to dispatch (auto-discovered if omitted).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "collect_results",
        "description": (
            "Read results from a completed round's output directory.  "
            "Returns a Markdown summary of patches, test outputs, and logs.  "
            "If results_dir is omitted, auto-discovers the latest round."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "results_dir": {
                    "type": "string",
                    "description": "Path to the results directory to scan (auto-discovered if omitted).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "finalize",
        "description": (
            "Signal that optimisation is complete.  Provide a summary of "
            "what was achieved, the best patch, and total speedup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Human-readable summary of the optimisation.",
                },
                "best_patch": {
                    "type": "string",
                    "description": "Path or identifier of the best patch.",
                },
                "total_speedup": {
                    "type": "string",
                    "description": "Total speedup achieved (e.g. '15%').",
                },
            },
            "required": ["summary"],
        },
    },
]


def build_tools_schema(toolruntime) -> list[dict]:
    """Merge ToolRuntime schemas (allowlisted) with orchestrator-specific tools."""
    swe_tools = [
        t for t in toolruntime.get_tools_schema()
        if t["name"] in _ORCHESTRATOR_SWE_TOOLS
    ]
    return swe_tools + _ORCHESTRATOR_ONLY_TOOLS


def dispatch_tool_call(
    ctx: dict[str, Any],
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    phase: str = "",
) -> str:
    """Route a tool call to the appropriate implementation.

    Exceptions are caught and returned as JSON error payloads so the
    orchestrator LLM can decide how to proceed.
    """
    ORCHESTRATION_TOOLS = {"generate_tasks", "dispatch_tasks", "collect_results", "finalize"}
    if phase == "explore" and tool_name in ORCHESTRATION_TOOLS:
        return json.dumps({
            "error": f"Cannot call {tool_name} during exploration phase. "
            "Please read and understand the kernel first, then respond with "
            "'Ready to begin optimization rounds' to proceed to the round loop."
        })

    try:
        if tool_name == "generate_tasks":
            return tool_generate_tasks(ctx, **tool_args)
        if tool_name == "dispatch_tasks":
            return tool_dispatch_tasks(ctx, **tool_args)
        if tool_name == "collect_results":
            return tool_collect_results(ctx, **tool_args)
        if tool_name == "finalize":
            return tool_finalize(ctx, **tool_args)
        result = ctx["toolruntime"].dispatch({"name": tool_name, "arguments": tool_args})
        return json.dumps(result, default=str) if isinstance(result, dict) else str(result)
    except Exception as exc:
        logger.error("Tool %s failed: %s", tool_name, exc, exc_info=True)
        return json.dumps({"error": f"{tool_name} failed: {exc}"})


# ── LLM step loop ────────────────────────────────────────────────────


def run_llm_steps(
    model,
    messages: list[dict],
    ctx: dict[str, Any],
    _print,
    console,
    *,
    phase: str,
) -> dict[str, Any] | None:
    """Run LLM tool-call steps until the LLM responds with text or calls ``finalize``.

    Returns a finalize report dict if the LLM called ``finalize``,
    otherwise ``None`` (the LLM responded with text, signalling it is
    ready for the next phase).
    """
    max_steps = int(os.getenv("GEAK_ORCHESTRATOR_STEP_LIMIT", "200"))
    step = 0
    _wm = ctx.get("working_memory")

    while step < max_steps:
        step += 1
        _print(
            f"[dim]{phase} step {step}[/dim]"
            if console
            else f"{phase} step {step}"
        )

        if _wm and phase != "explore":
            _wm.update_step(step, 0.0)
            _wm_text = _wm.format_for_injection()
            if _wm_text and not any("[Working Memory" in m.get("content", "") for m in messages[-3:]):
                messages.append({"role": "user", "content": f"[Working Memory Update]\n{_wm_text}"})

        response = model.query(messages)

        content_text = response.get("content", "") if isinstance(response, dict) else ""
        tool_call = response.get("tools") if isinstance(response, dict) else None

        if not tool_call:
            if phase.startswith("round_") and any(
                name in content_text for name in ("dispatch_tasks", "collect_results", "finalize")
            ):
                emit_debug_log(
                    "heterogeneous_orchestrator:run_llm_steps:no_tool_call",
                    "Orchestrator produced text mentioning missing orchestration tools",
                    {
                        "phase": phase,
                        "step": step,
                        "content_preview": content_text[:300],
                        "model_tools": model_tools_snapshot(model),
                    },
                    hypothesis_id="H3",
                )
            if content_text:
                _print(f"  Orchestrator: {content_text[:300]}")
            messages.append({"role": "assistant", "content": content_text})
            return None

        tool_name = tool_call.get("function", {}).get("name", "")
        tool_args = tool_call.get("function", {}).get("arguments", {})
        tool_id = tool_call.get("id", f"call_{phase}_{step}")

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {}

        _print(f"  Tool: {tool_name}({json.dumps(tool_args)[:200]})")

        messages.append(
            {
                "role": "assistant",
                "content": content_text,
                "tool_calls": tool_call,
            }
        )

        result_str = dispatch_tool_call(ctx, tool_name, tool_args, phase=phase)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result_str,
            }
        )

        _print(f"  Result: {result_str[:300]}")

        if _wm:
            try:
                from minisweagent.memory.working_memory import extract_insight_from_tool_result
                insight = extract_insight_from_tool_result(tool_name, result_str, 0)
                if insight:
                    insight.step = step
                    _wm.insights.append(insight)
                    if len(_wm.insights) > 5:
                        _wm.insights = _wm.insights[-5:]
                    if "speedup" in (insight.message or "").lower():
                        import re as _re
                        _sp = _re.search(r'(\d+\.\d+)x', insight.message)
                        if _sp:
                            _wm.update_speedup(float(_sp.group(1)))
            except Exception:
                pass

        if tool_name == "finalize":
            try:
                report = json.loads(result_str)
            except json.JSONDecodeError:
                report = {"summary": result_str}
            _print(
                "[bold green]Orchestrator: Optimisation finalised.[/bold green]"
                if console
                else "Orchestrator: Optimisation finalised."
            )
            return report

    logger.warning(
        "Orchestrator hit step limit (%d) for phase %s -- proceeding to next phase",
        max_steps,
        phase,
    )
    _print(f"  Step limit ({max_steps}) reached for {phase}, moving on...")
    return None


# ── Main entry point ─────────────────────────────────────────────────


def run_heterogeneous_orchestrator(
    preprocess_ctx: dict[str, Any],
    gpu_ids: list[int],
    model,
    model_factory,
    output_dir: Path,
    max_rounds: int,
    start_round: int,
    _print,
    console,
) -> dict[str, Any]:
    """Run the heterogeneous orchestrator with LLM-driven tool calling.

    This is the main heterogeneous entry point, called by
    ``run/orchestrator.py:run_orchestrator`` when ``heterogeneous=True``.
    """
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent
    from minisweagent.run.preprocess.discovery_types import DiscoveryResult
    from minisweagent.run.postprocess.evaluation import evaluate_round_best as _evaluate_round_best
    from minisweagent.run.postprocess.results import (
        auto_finalize,
        merge_round_evaluation_into_final_report,
        record_final_outcome,
    )
    from minisweagent.tools.tools_runtime import ToolRuntime

    disc_dict = preprocess_ctx.get("discovery") or {}
    kernel_path = preprocess_ctx.get("kernel_path", "")
    discovery_result = DiscoveryResult.from_dict(disc_dict, kernel_path)

    preprocess_dir = output_dir
    for candidate in ("resolved.json", "discovery.json", "profile.json"):
        if (output_dir / candidate).exists():
            preprocess_dir = output_dir
            break

    toolruntime = ToolRuntime(tool_profile="full", use_strategy_manager=True)

    ctx: dict[str, Any] = {
        **preprocess_ctx,
        "discovery_result": discovery_result,
        "output_dir": str(output_dir),
        "preprocess_dir": str(preprocess_dir),
        "gpu_ids": gpu_ids,
        "model": model,
        "model_factory": model_factory,
        "agent_class": StrategyInteractiveAgent,
        "toolruntime": toolruntime,
    }

    tools_schema = build_tools_schema(toolruntime)
    model_impl = getattr(model, "_impl", model)
    _orig = getattr(model_impl, "tools", None)
    original_tools = list(_orig) if isinstance(_orig, list) else _orig
    model_impl.tools = tools_schema

    bm = preprocess_ctx.get("baseline_metrics") or {}
    bm_summary = json.dumps(bm, indent=2, default=str) if bm else "Not available"

    prof = preprocess_ctx.get("profiling") or {}
    prof_summary = json.dumps(prof, indent=2, default=str)[:2000] if prof else "Not available"

    cmd = preprocess_ctx.get("commandment") or ""
    cmd_excerpt = cmd[:1500] + ("..." if len(cmd) > 1500 else "") if cmd else "Not available"

    codebase_ctx = ""
    _codebase_ctx_path = preprocess_dir / "CODEBASE_CONTEXT.md"
    if _codebase_ctx_path.exists():
        codebase_ctx = _codebase_ctx_path.read_text().strip()

    _memory_context = ""
    try:
        from minisweagent.memory.integration import assemble_memory_context
        _bm = preprocess_ctx.get("baseline_metrics") or {}
        _memory_context = assemble_memory_context(
            kernel_path=str(preprocess_ctx.get("kernel_path", "")),
            bottleneck_type=_bm.get("bottleneck"),
            profiling_metrics=_bm,
        )
        if _memory_context:
            _memory_context = "### Optimization Memory (from past runs)\n" + _memory_context
    except Exception as _mem_exc:
        logger.debug("Memory context assembly failed: %s", _mem_exc)

    _working_mem = None
    try:
        from minisweagent.memory.integration import is_working_memory_enabled
        if is_working_memory_enabled():
            from minisweagent.memory.cross_session_memory import classify_kernel_category
            from minisweagent.memory.working_memory import WorkingMemory
            _kpath = str(preprocess_ctx.get("kernel_path", ""))
            _wm_notebook_dir = str(output_dir / "_working_memory")
            _working_mem = WorkingMemory(
                kernel_category=classify_kernel_category(_kpath) if _kpath else "unknown",
                max_steps=int(os.getenv("GEAK_AGENT_STEP_LIMIT", "100")),
                notebook_dir=_wm_notebook_dir,
                notebook_writer_id="orchestrator",
            )
            _bm_dict = preprocess_ctx.get("baseline_metrics") or {}
            if _bm_dict.get("bottleneck"):
                _working_mem.bottleneck_type = str(_bm_dict["bottleneck"])
            if _bm_dict.get("benchmark_duration_us"):
                _working_mem.baseline_latency_ms = float(_bm_dict["benchmark_duration_us"]) / 1000.0
            elif _bm_dict.get("duration_us"):
                _working_mem.baseline_latency_ms = float(_bm_dict["duration_us"]) / 1000.0
            _working_mem.sync_notebook_baseline()
            ctx["working_memory"] = _working_mem
    except Exception as _wm_exc:
        logger.debug("WorkingMemory init failed: %s", _wm_exc)

    instance_msg = INSTANCE_TEMPLATE.format(
        kernel_path=str(preprocess_ctx.get("kernel_path", "N/A")),
        repo_root=str(preprocess_ctx.get("repo_root", "N/A")),
        test_command=str(preprocess_ctx.get("test_command", "N/A")),
        gpu_ids=str(gpu_ids),
        output_dir=str(output_dir),
        codebase_context=codebase_ctx or "Not available",
        baseline_metrics_summary=bm_summary,
        profiling_summary=prof_summary,
        commandment_excerpt=cmd_excerpt,
        memory_context=_memory_context,
    )

    start_label = (
        f"rounds {start_round}-{max_rounds}" if start_round > 1
        else f"{max_rounds} rounds"
    )
    _print(
        f"[bold cyan]--- Orchestrator starting ({start_label}, {len(gpu_ids)} GPUs) ---[/bold cyan]"
        if console
        else f"--- Orchestrator starting ({start_label}, {len(gpu_ids)} GPUs) ---"
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instance_msg},
    ]

    if start_round > 1:
        for prev_round in range(1, start_round):
            eval_path = output_dir / f"round_{prev_round}_evaluation.json"
            if eval_path.exists():
                try:
                    round_eval = json.loads(eval_path.read_text())
                    ctx[f"round_{prev_round}_eval"] = round_eval
                    eval_summary = json.dumps(round_eval, indent=2, default=str)[:2000]
                    messages.append({
                        "role": "user",
                        "content": (
                            f"## Round {prev_round} Evaluation (prior run)\n\n"
                            f"The best kernel from round {prev_round} was evaluated "
                            f"with FULL_BENCHMARK and PROFILE:\n```\n{eval_summary}\n```\n"
                            "Use this data to inform your strategy."
                        ),
                    })
                    _print(f"  Loaded prior evaluation: {eval_path.name}")
                except (json.JSONDecodeError, OSError) as exc:
                    _print(f"  Warning: could not load {eval_path.name}: {exc}")

    try:
        if start_round <= 1:
            _print(
                "[bold cyan]--- Exploration phase ---[/bold cyan]"
                if console
                else "--- Exploration phase ---"
            )
            finalize_result = run_llm_steps(
                model, messages, ctx, _print, console, phase="explore",
            )
            if finalize_result is not None:
                return finalize_result

        for round_num in range(start_round, max_rounds + 1):
            is_last = round_num == max_rounds
            round_header = (
                f"--- Round {round_num}/{max_rounds}"
                f"{' (final round)' if is_last else ''} ---"
            )
            _print(
                f"[bold cyan]{round_header}[/bold cyan]"
                if console
                else round_header
            )

            if is_last:
                round_instruction = (
                    f"Begin round {round_num} (FINAL round). "
                    "Call generate_tasks, dispatch_tasks, collect_results, "
                    "then call **finalize** with a full summary of the best "
                    "results across all rounds."
                )
            else:
                round_instruction = (
                    f"Begin round {round_num}/{max_rounds}. "
                    "Call generate_tasks, dispatch_tasks, collect_results. "
                    "Then evaluate the results and respond with your analysis. "
                    "Focus on strategies not yet tried or that build on "
                    "previous successes. For later-round decisions, prefer the "
                    "system-provided FULL_BENCHMARK verified outcomes over raw "
                    "task-local speedup claims."
                )
            messages.append({"role": "user", "content": round_instruction})

            finalize_result = run_llm_steps(
                model, messages, ctx, _print, console,
                phase=f"round_{round_num}",
            )

            round_results_dir = output_dir / "results" / f"round_{round_num}"
            round_eval = _evaluate_round_best(
                ctx, round_num, round_results_dir, _print,
            )
            if round_eval:
                ctx[f"round_{round_num}_eval"] = round_eval
                if _working_mem:
                    _working_mem.record_round_evaluation(round_eval)
                eval_summary = json.dumps(round_eval, indent=2, default=str)[:2000]
                messages.append({
                    "role": "user",
                    "content": (
                        f"## Round {round_num} Evaluation\n\n"
                        f"The best kernel from round {round_num} was evaluated "
                        f"with FULL_BENCHMARK and PROFILE:\n```\n{eval_summary}\n```\n"
                        "Use this data to inform your next-round strategy. "
                        "Treat the FULL_BENCHMARK result as canonical and use "
                        "task-local speedups only as supporting evidence."
                    ),
                })
            if round_eval and round_eval.get("best_patch"):
                current_speedup_val = (
                    round_eval.get("full_benchmark", {}).get("verified_speedup")
                    or round_eval.get("benchmark_speedup", 0)
                )
                best_global_speedup = ctx.get("_best_global_speedup", 0)
                if current_speedup_val >= best_global_speedup:
                    ctx["starting_patch"] = round_eval["best_patch"]
                    ctx["_best_global_speedup"] = current_speedup_val

            if finalize_result is not None:
                if round_eval:
                    finalize_result = merge_round_evaluation_into_final_report(
                        ctx, output_dir, finalize_result, round_eval,
                    )
                else:
                    record_final_outcome(ctx, finalize_result)
                return finalize_result
    finally:
        if original_tools is not None:
            model_impl.tools = original_tools
        elif hasattr(model_impl, "tools"):
            model_impl.tools = []

    _print(
        "[yellow]Orchestrator completed all rounds without calling finalize – auto-selecting best result...[/yellow]"
        if console
        else "Orchestrator completed all rounds without calling finalize – auto-selecting best result..."
    )

    return auto_finalize(ctx, _print)
