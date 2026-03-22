"""Homogeneous orchestrator: identical task replicated across GPUs.

Each round writes a single task file and dispatches N copies (one per
GPU) via ``run_task_batch``.  After all agents finish, per-round
evaluation (FULL_BENCHMARK + PROFILE) verifies the best result.  Early
stopping kicks in when a round fails to improve over prior best.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_homogeneous_orchestrator(
    preprocess_ctx: dict[str, Any],
    gpu_ids: list[int],
    output_dir: Path,
    max_rounds: int,
    start_round: int,
    _print,
    console,
    model_factory=None,
) -> dict[str, Any]:
    """Run the orchestrator in homogeneous mode.

    Each round writes a single task file and dispatches it in-process
    via ``run_task_batch``, so all agents work on the same optimization
    task.  Per-round evaluation and early stopping are reused from the
    shared postprocess modules.
    """
    from minisweagent.run.postprocess.evaluation import evaluate_round_best
    from minisweagent.run.postprocess.results import auto_finalize
    from minisweagent.run.task_file import write_task_file

    pp_dir = output_dir
    starting_patch = preprocess_ctx.get("starting_patch")

    start_label = (
        f"rounds {start_round}-{max_rounds}" if start_round > 1
        else f"{max_rounds} rounds"
    )
    _print(
        f"[bold cyan]--- Orchestrator (homogeneous) starting ({start_label}, {len(gpu_ids)} GPUs) ---[/bold cyan]"
        if console
        else f"--- Orchestrator (homogeneous) starting ({start_label}, {len(gpu_ids)} GPUs) ---"
    )

    ctx: dict[str, Any] = {**preprocess_ctx, "output_dir": str(output_dir)}

    for round_num in range(start_round, max_rounds + 1):
        is_last = round_num == max_rounds
        round_header = (
            f"--- Homogeneous round {round_num}/{max_rounds}"
            f"{' (final)' if is_last else ''} ---"
        )
        _print(
            f"[bold cyan]{round_header}[/bold cyan]"
            if console else round_header
        )

        task_dir = output_dir / "tasks" / f"round_{round_num}"
        task_dir.mkdir(parents=True, exist_ok=True)

        task_file = task_dir / "00_optimize.md"
        metadata: dict[str, Any] = {
            "label": "kernel_optimization",
            "priority": 10,
            "agent_type": "strategy_agent",
            "kernel_path": preprocess_ctx.get("kernel_path"),
            "repo_root": preprocess_ctx.get("repo_root"),
            "test_command": preprocess_ctx.get("test_command"),
            "harness_path": preprocess_ctx.get("harness_path"),
            "commandment": str(pp_dir / "COMMANDMENT.md"),
            "baseline_metrics": str(pp_dir / "baseline_metrics.json"),
            "profiling": str(pp_dir / "profile.json"),
            "codebase_context": str(pp_dir / "CODEBASE_CONTEXT.md"),
            "benchmark_baseline": str(pp_dir / "benchmark_baseline.txt"),
            "round": round_num,
            "starting_patch": starting_patch,
        }

        task_body = (
            "Optimize this GPU kernel for maximum performance.\n\n"
            "Follow the workflow described in the pipeline instructions.\n"
            "Use the discovered tests and benchmarks for correctness and performance validation.\n"
            "Report final speedup when done."
        )

        write_task_file(task_file, metadata, task_body, relative_to=task_file.parent)
        _print(f"  Task file: {task_file}")

        results_dir = output_dir / "results" / f"round_{round_num}"
        results_dir.mkdir(parents=True, exist_ok=True)

        n_agents = len(gpu_ids)
        _print(f"  Dispatching: run_task_batch({task_file.name}, {n_agents} agents on {n_agents} GPUs)")
        try:
            from minisweagent.run.dispatch import run_task_batch

            run_task_batch(
                task_files=[task_file] * n_agents,
                gpu_ids=gpu_ids,
                output_dir=results_dir,
                model_factory=model_factory,
                console=console,
            )
        except Exception as exc:
            _print(f"  [yellow]Round {round_num} dispatch failed: {exc}[/yellow]")

        round_eval = evaluate_round_best(
            ctx, round_num, results_dir, _print,
        )
        if round_eval:
            ctx[f"round_{round_num}_eval"] = round_eval
            if round_eval.get("best_patch"):
                current_speedup_val = (
                    round_eval.get("full_benchmark", {}).get("verified_speedup")
                    or round_eval.get("benchmark_speedup", 0)
                )
                best_global_speedup = ctx.get("_best_global_speedup", 0)
                if current_speedup_val >= best_global_speedup:
                    starting_patch = round_eval["best_patch"]
                    ctx["starting_patch"] = starting_patch
                    ctx["_best_global_speedup"] = current_speedup_val

        early_stop_threshold = float(os.getenv("GEAK_EARLY_STOP_THRESHOLD", "0.005"))
        if round_eval and round_num >= 2:
            current_speedup = (
                round_eval.get("full_benchmark", {}).get("verified_speedup")
                or round_eval.get("benchmark_speedup", 1.0)
            )
            prior_speedups = []
            for r in range(1, round_num):
                rev = ctx.get(f"round_{r}_eval", {})
                s = (
                    rev.get("full_benchmark", {}).get("verified_speedup")
                    or rev.get("benchmark_speedup", 1.0)
                )
                prior_speedups.append(s)
            best_prior = max(prior_speedups) if prior_speedups else 1.0
            if current_speedup <= best_prior * (1 + early_stop_threshold):
                _print(
                    f"  Early stopping: round {round_num} ({current_speedup:.4f}x) "
                    f"did not improve over prior best ({best_prior:.4f}x)"
                )
                break

    return auto_finalize(ctx, _print)
