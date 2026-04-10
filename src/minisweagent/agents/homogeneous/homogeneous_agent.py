#!/usr/bin/env python3
"""
Homogeneous Agent Runner - Run multiple identical agents in parallel.

This module provides a simplified interface to run ParallelAgent with
homogeneous configuration (all agents run the same task with identical settings).
"""

import copy
import logging
import time
from pathlib import Path

from rich.console import Console

from minisweagent.agents.parallel_agent import BestPatchResult, ParallelAgent
from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent
from minisweagent.models import get_model

logger = logging.getLogger(__name__)


def parse_gpu_ids(gpu_ids_str: str | None) -> list[int]:
    """Parse comma-separated GPU IDs string to list of integers."""
    if not gpu_ids_str:
        return [0]
    return [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip()]


def run_homogeneous_agent(
    config: dict,
    task_content: str,
    model,
    env,
    env_class,
    env_kwargs: dict,
    agent_config: dict,
    repo: Path | None = None,
    num_parallel: int | None = None,
    gpu_ids: str | None = None,
    output_dir: Path | None = None,
    model_name: str | None = None,
    console: Console | None = None,
    max_rounds: int = 1,
    preprocess_ctx: dict | None = None,
) -> BestPatchResult | None:
    """
    Run homogeneous parallel agents.

    This function is called from mini.py when agent_mode is 'homogeneous'.
    Configuration is already loaded and merged by mini.py.

    Args:
        config: Merged configuration dict
        task_content: Task description
        model: Model instance
        env: Environment instance
        env_class: Environment class for factory
        env_kwargs: Environment kwargs for factory
        tools_settings: Tools settings from config
        agent_config: Base agent configuration
        repo: Repository path for git worktree management
        num_parallel: Number of parallel agents
        gpu_ids: Comma-separated GPU IDs
        output_dir: Output directory
        model_name: Model name for factory
        console: Rich console for output

    Returns:
        The ParallelAgent instance after execution
    """
    if console is None:
        console = Console(highlight=False)

    # Parse configuration values
    parallel_config = config.get("parallel", {})

    # Number of parallel agents
    final_num_parallel = (
        num_parallel or parallel_config.get("num_parallel") or config.get("agent", {}).get("num_parallel") or 1
    )
    _np_source = (
        "arg"
        if num_parallel
        else "parallel config"
        if parallel_config.get("num_parallel")
        else "agent config"
        if config.get("agent", {}).get("num_parallel")
        else "default"
    )
    logger.debug("num_parallel=%d (source=%s)", final_num_parallel, _np_source)

    # GPU IDs
    final_gpu_ids = parse_gpu_ids(gpu_ids or parallel_config.get("gpu_ids") or config.get("agent", {}).get("gpu_ids"))
    logger.debug("gpu_ids=%s", final_gpu_ids)

    # Repository path
    final_repo = repo
    if not final_repo:
        final_repo = parallel_config.get("repo") or config.get("agent", {}).get("repo")

    final_repo = Path(final_repo).resolve()
    if not final_repo.exists():
        raise ValueError(f"Repository path does not exist: {final_repo}")

    # GEAK homogeneous flow always uses strategy interactive agent.
    base_agent_class = StrategyInteractiveAgent

    # Configure agent for homogeneous mode
    agent_config["mode"] = "yolo"
    agent_config["confirm_exit"] = False
    agent_config.setdefault("use_strategy_manager", True)
    agent_config["num_parallel"] = final_num_parallel
    agent_config["gpu_ids"] = final_gpu_ids
    agent_config["repo"] = str(final_repo)
    agent_config["agent_class"] = base_agent_class

    # Create output directory (pop from agent_config as ParallelAgentConfig doesn't accept it)
    final_output_dir = Path(agent_config.pop("output_dir", None) or output_dir or "optimization_logs")
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Set patch_output_dir to output_dir so patches are saved alongside logs
    agent_config["patch_output_dir"] = str(final_output_dir)

    # Get model config for factory
    model_config = config.get("model", {})

    logger.info(
        "\n[bold cyan]%s[/bold cyan]\n  [bold]Homogeneous Agent[/bold] (%d agents, GPUs %s)\n[bold cyan]%s[/bold cyan]",
        "=" * 60,
        final_num_parallel,
        final_gpu_ids,
        "=" * 60,
    )
    logger.info("  repo=%s, output_dir=%s", final_repo, final_output_dir)
    logger.info("[dim]Sub-agents are working — expect no output for several minutes.[/dim]")

    task_content = task_content + "\n\n" + "The current worktree is: " + str(final_repo)
    best_result = None

    try:
        for round_num in range(1, max_rounds + 1):
            is_last = round_num == max_rounds
            logger.info(
                "\n[bold cyan]%s[/bold cyan]\n  [bold]Homogeneous Round %d/%d[/bold]%s\n[bold cyan]%s[/bold cyan]",
                "=" * 60,
                round_num,
                max_rounds,
                " [bold red](FINAL)[/bold red]" if is_last else "",
                "=" * 60,
            )

            # Set starting_patch from previous round's best result
            if round_num > 1 and best_result and best_result.patch_dir:
                best_patch_path = best_result.patch_dir / f"{best_result.patch_id}.patch"
                if best_patch_path.exists():
                    agent_config["starting_patch"] = str(best_patch_path)
                    logger.info("Starting from best patch of round %d: %s", round_num - 1, best_patch_path)

            # Enrich prompt for round 2+ with previous results
            round_task = task_content
            if round_num > 1:
                from minisweagent.agents.heterogeneous.result_scanning import scan_previous_results

                prev_summary = scan_previous_results(final_output_dir / "results")
                if prev_summary:
                    round_task = (
                        f"{task_content}\n\n---\n\n"
                        f"## Previous Round Results\n"
                        f"Below are results from previous optimization rounds. "
                        f"Use these to inform your strategy — avoid repeating failed approaches "
                        f"and build on successful ones.\n\n"
                        f"{prev_summary}"
                    )

            # Create a fresh ParallelAgent for each round
            agent = ParallelAgent(model, env, **agent_config)

            _t0 = time.monotonic()
            best_result = agent.run(
                round_task,
                console=console,
                model_factory=lambda: get_model(model_name, model_config.copy()),
                env_factory=lambda: env_class(**copy.deepcopy(env_kwargs)),
                round_num=round_num,
            )
            _elapsed = time.monotonic() - _t0

            if best_result:
                logger.info(
                    "Round %d completed in %.0fs. Best patch: %s (agent %d)",
                    round_num,
                    _elapsed,
                    best_result.patch_id,
                    best_result.agent_id,
                )
                console.print(
                    f"\n[bold green]Round {round_num} best patch:[/bold green] "
                    f"{best_result.patch_id} (agent {best_result.agent_id})"
                )
            else:
                logger.info("Round %d completed in %.0fs. No best patch selected.", round_num, _elapsed)
                console.print(f"\n[bold yellow]Round {round_num}: No best patch selected[/bold yellow]")

        # Write final_report.json using auto_finalize (selects best across all rounds)
        from minisweagent.run.postprocess.results import auto_finalize

        ctx = {"output_dir": str(final_output_dir)}
        if preprocess_ctx:
            ctx["kernel_path"] = preprocess_ctx.get("kernel_path", "")
            ctx["repo_root"] = preprocess_ctx.get("repo_root", "")
            ctx["baseline_metrics"] = preprocess_ctx.get("baseline_metrics")
        auto_finalize(ctx)
        logger.info("Wrote final_report.json to %s", final_output_dir / "final_report.json")

    except Exception as e:
        logger.error("Homogeneous agent failed: %s", e, exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise

    return best_result
