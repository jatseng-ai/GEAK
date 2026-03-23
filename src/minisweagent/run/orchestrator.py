"""Orchestrator: dispatch to homogeneous or heterogeneous optimization mode.

This module is the thin entry point for ``geak-orchestrate``.  It reads
preprocessor artefacts, resolves configuration, and delegates to:

- ``agents.homogeneous.orchestrator`` -- identical task on all GPUs
- ``agents.heterogeneous.orchestrator`` -- LLM-generated diverse tasks

All heavy lifting lives in those modules and in the shared postprocess
package (``run.postprocess.evaluation``, ``run.postprocess.results``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from minisweagent.run.pipeline_helpers import DEFAULT_HETEROGENEOUS, DEFAULT_PIPELINE_OUTPUT_DIR

logger = logging.getLogger(__name__)


def run_orchestrator(
    preprocess_ctx: dict[str, Any],
    gpu_ids: list[int],
    model,
    model_factory,
    *,
    output_dir: Path | None = None,
    max_rounds: int | None = None,
    start_round: int = 1,
    heterogeneous: bool = DEFAULT_HETEROGENEOUS,
    console=None,
) -> dict[str, Any]:
    """Run the orchestrator agent loop.

    Parameters
    ----------
    preprocess_ctx:
        Context dict returned by ``run_preprocessor()``.
    gpu_ids:
        List of GPU device IDs available for task execution.
    model:
        LLM model instance for the orchestrator.
    model_factory:
        Callable returning a new model instance (for sub-agents).
    output_dir:
        Override output directory (defaults to preprocess_ctx source).
    max_rounds:
        Maximum optimisation rounds (default: from GEAK_MAX_ROUNDS env or 5).
    start_round:
        Round number to start from (1-based, default 1).
    heterogeneous:
        If True, use LLM-generated diverse tasks per round.
        If False (default), use homogeneous mode where all agents get the same task.
    console:
        Optional Rich console for progress messages.
    """
    _out = output_dir or Path(preprocess_ctx.get("output_dir", DEFAULT_PIPELINE_OUTPUT_DIR))
    _out = Path(_out)
    _out.mkdir(parents=True, exist_ok=True)

    max_rounds = max_rounds or int(os.getenv("GEAK_MAX_ROUNDS", "5"))

    def _print(msg: str) -> None:
        if console:
            console.print(msg)
        else:
            print(msg, file=sys.stderr)

    if not heterogeneous:
        from minisweagent.agents.homogeneous.orchestrator import run_homogeneous_orchestrator

        return run_homogeneous_orchestrator(
            preprocess_ctx, gpu_ids, _out, max_rounds, start_round, _print, console,
            model_factory=model_factory,
        )

    from minisweagent.agents.heterogeneous.orchestrator import run_heterogeneous_orchestrator

    return run_heterogeneous_orchestrator(
        preprocess_ctx, gpu_ids, model, model_factory,
        _out, max_rounds, start_round, _print, console,
    )


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    """CLI: ``geak-orchestrate --preprocess-dir <dir> [--gpu-ids 0,1] [--max-rounds 3]``."""
    import argparse

    from minisweagent.run.pipeline_helpers import DEFAULT_HETEROGENEOUS

    parser = argparse.ArgumentParser(
        description="GEAK orchestrator: LLM-driven task generation, dispatch, and iteration loop",
    )
    parser.add_argument(
        "--preprocess-dir",
        required=True,
        help="Directory containing preprocessor artefacts (resolved.json, discovery.json, profile.json, ...)",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU device IDs (default: all detected GPUs, or 0 as fallback)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum optimisation rounds (default: GEAK_MAX_ROUNDS env or 5)",
    )
    parser.add_argument(
        "--start-round",
        type=int,
        default=1,
        help="Round to resume from (1-based, default: 1). "
             "Skips exploration and loads prior round evaluations from disk.",
    )
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        default=DEFAULT_HETEROGENEOUS,
        help="Use LLM-generated diverse tasks per round. Default: homogeneous (all agents get the same task).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: from GEAK_MODEL env or geak.yaml)",
    )
    from minisweagent.run.pipeline_helpers import add_agent_filter_args, apply_agent_filter_env

    add_agent_filter_args(parser)
    args = parser.parse_args()
    apply_agent_filter_env(args)

    pp_dir = Path(args.preprocess_dir).resolve()
    if not pp_dir.is_dir():
        print(f"ERROR: preprocess directory not found: {args.preprocess_dir}", file=sys.stderr)
        sys.exit(1)

    # Reconstruct preprocessor context from artefact files
    ctx: dict[str, Any] = {}

    resolved_path = pp_dir / "resolved.json"
    if resolved_path.exists():
        resolved = json.loads(resolved_path.read_text())
        ctx["kernel_path"] = resolved.get("local_file_path")
        kernel_file = resolved.get("local_file_path", "")
        repo_path = resolved.get("local_repo_path")
        if kernel_file:
            from minisweagent.run.preprocess.resolve_kernel_url import _find_git_root
            git_root = _find_git_root(Path(kernel_file))
            if git_root:
                repo_path = str(git_root)
            elif not repo_path:
                repo_path = str(Path(kernel_file).parent)
        ctx["repo_root"] = repo_path or str(pp_dir)
    else:
        ctx["repo_root"] = str(pp_dir)

    discovery_path = pp_dir / "discovery.json"
    if discovery_path.exists():
        ctx["discovery"] = json.loads(discovery_path.read_text())

    profile_path = pp_dir / "profile.json"
    if profile_path.exists():
        ctx["profiling"] = json.loads(profile_path.read_text())

    baseline_path = pp_dir / "baseline_metrics.json"
    if baseline_path.exists():
        ctx["baseline_metrics"] = json.loads(baseline_path.read_text())

    testcase_sel_path = pp_dir / "testcase_selection.json"
    if testcase_sel_path.exists():
        ts = json.loads(testcase_sel_path.read_text())
        if isinstance(ts, dict):
            ctx.setdefault("test_command", ts.get("test_command"))
            ctx.setdefault("harness_path", ts.get("harness_path"))

    commandment_path = pp_dir / "COMMANDMENT.md"
    if commandment_path.exists():
        ctx["commandment"] = commandment_path.read_text()
        ctx["commandment_path"] = str(commandment_path)

    ctx["output_dir"] = str(pp_dir)
    ctx["preprocess_dir"] = str(pp_dir)

    # Paths for task generator
    if baseline_path.exists():
        ctx["baseline_metrics_path"] = str(baseline_path)
    if profile_path.exists():
        ctx["profiling_path"] = str(profile_path)
    codebase_ctx_path = pp_dir / "CODEBASE_CONTEXT.md"
    if codebase_ctx_path.exists():
        ctx["codebase_context_path"] = str(codebase_ctx_path)

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",") if g.strip()]
    else:
        try:
            from minisweagent.agents.agent_spec import detect_available_gpus
            gpu_ids = detect_available_gpus()
        except Exception:
            gpu_ids = [0]

    from minisweagent.run.pipeline_helpers import geak_model_factory, load_geak_model

    model_name = args.model or os.getenv("GEAK_MODEL")
    model = load_geak_model(model_name)
    factory = geak_model_factory(model_name)

    try:
        from rich.console import Console
        console = Console(highlight=False)
    except ImportError:
        console = None

    report = run_orchestrator(
        preprocess_ctx=ctx,
        gpu_ids=gpu_ids,
        model=model,
        model_factory=factory,
        output_dir=pp_dir,
        max_rounds=args.max_rounds,
        start_round=args.start_round,
        heterogeneous=args.heterogeneous,
        console=console,
    )

    if report:
        print(json.dumps(report, indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
