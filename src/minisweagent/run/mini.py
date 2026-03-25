#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import copy
import json
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any


class TeeOutput:
    """Capture stdout/stderr to buffer while keeping terminal output."""

    def __init__(self, original):
        self.terminal = original
        self.buffer = StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()

    def getvalue(self):
        return self.buffer.getvalue()


import typer
import yaml
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.agents.interactive_textual import TextualAgent
from minisweagent.agents.parallel_agent import ParallelAgent
from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent
from minisweagent.agents.unit_test_agent import run_unit_test_agent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.run.utils.config_editor import load_and_merge_configs
from minisweagent.run.utils.save import save_traj
from minisweagent.run.utils.task_parser import _resolve_path_case
from minisweagent.utils.log import logger

DEFAULT_CONFIG = Path(os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "mini.yaml"))
DEFAULT_OUTPUT = global_config_dir / "last_mini_run.traj.json"

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")
prompt_session = PromptSession(history=FileHistory(global_config_dir / "mini_task_history.txt"))


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


_HELP_TEXT = """Run mini-SWE-agent in your local environment.

[not dim]
There are two different user interfaces:

[bold green]mini[/bold green] Simple REPL-style interface
[bold green]mini -v[/bold green] Pager-style interface (Textual)

RAG knowledge retrieval:

[bold green]mini --rag[/bold green] Enable RAG retrieval from AMD/NVIDIA knowledge base
[bold green]mini --rag -d[/bold green] Enable RAG retrieval with debug output

More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/mini/[/bold green]
[/not dim]
"""


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    visual: bool = typer.Option(False, "-v", "--visual", help="Toggle (pager-style) UI (Textual) depending on the MSWEA_VISUAL_MODE_DEFAULT environment setting",),
    model_name: str | None = typer.Option( None, "-m", "--model", help="Model to use",),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path | None = typer.Option(None, "-c", "--config", help="Path to config file (overrides template selection)"),
    output: Path | None = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file"),
    exit_immediately: bool = typer.Option( False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Advanced"),
    # Strategy mode configuration
    enable_strategies: bool = typer.Option(True, "--enable-strategies/--no-enable-strategies", help="Enable optimization strategy management (optool command). Auto-selects appropriate template.", rich_help_panel="Advanced"),
    strategy_file: str = typer.Option(".optimization_strategies.md", "--strategy-file", help="Path to strategy file (relative to workspace)", rich_help_panel="Advanced"),
    # Patch mode configuration (always enabled)
    test_command: str | None = typer.Option(None, "--test_command", "--test-command", help="Test command to run for patch validation"),
    create_test: bool = typer.Option(
        False,
        "--create-test",
        "--create_test",
        help="Auto-create/search unit tests and infer test_command when missing (or to override it).",
        rich_help_panel="Advanced",
    ),
    patch_output: Path | None = typer.Option(None, "--patch-output", help="Output directory for patch files and test results"),
    metric: str | None = typer.Option(None, "--metric", help="Metric extraction task description for LLM"),
    num_parallel: int | None = typer.Option(None, "--num-parallel", help="Number of parallel patch agents to run (only effective with --save-patch). If not specified, reads from config file."),
    repo: Path | None = typer.Option(None, "--repo", help="Repository path for parallel execution. Required when num_parallel > 1. Each agent will get an isolated workdir using git worktree."),
    gpu_ids: str | None = typer.Option(None, "--gpu-ids", help="Comma-separated GPU IDs for agents (e.g., '0,1,2,3'). For single agent, uses first GPU. Defaults to '0'."),
    # RAG knowledge retrieval
    rag: bool = typer.Option(False, "--rag", help="Enable RAG retrieval from AMD/NVIDIA knowledge base"),
    debug: bool = typer.Option(False, "-d", "--debug", help="Enable debug output (only with --rag)"),
    # Pipeline orchestration flags
    heterogeneous: bool | None = typer.Option(None, "--heterogeneous/--no-heterogeneous", help="Override auto-detected mode. Default: auto-detect from kernel type (HIP=homogeneous, Triton=heterogeneous).", rich_help_panel="Pipeline"),
    max_rounds: int = typer.Option(3, "--max-rounds", help="Maximum optimization rounds. Set to 1 for single-round.", rich_help_panel="Pipeline"),
    start_round: int = typer.Option(1, "--start-round", help="Round to resume from (1-based).", rich_help_panel="Pipeline"),
    preprocess_dir: Path | None = typer.Option(None, "--preprocess-dir", help="Path to existing preprocessing artifacts. Skips auto-preprocessing.", rich_help_panel="Pipeline"),
    kernel_url: str | None = typer.Option(None, "--kernel-url", help="Kernel URL or local path. Triggers full pipeline mode (preprocess -> orchestrate).", rich_help_panel="Pipeline"),
) -> Any:
    # fmt: on
    # Capture all print output to trajectory
    tee_out, tee_err = TeeOutput(sys.stdout), TeeOutput(sys.stderr)
    sys.stdout, sys.stderr = tee_out, tee_err

    configure_if_first_time()
    
    # 1. Load base config (mini.yaml - always loaded as foundation)
    base_config_path = builtin_config_dir / "mini.yaml"
    console.print(f"Loading base config: [bold green]'{base_config_path.name}'[/bold green]")
    config = yaml.safe_load(base_config_path.read_text())
    
    # 2. Select and merge template based on enable_strategies flag
    if enable_strategies:
        template_name = "mini_kernel_strategy_list.yaml"
    else:
        template_name = "mini_system_prompt.yaml"
    
    template_path = builtin_config_dir / template_name
    console.print(f"Applying template: [bold green]'{template_name}'[/bold green] (save_patch always enabled)")
    template_config = yaml.safe_load(template_path.read_text())
    config = _deep_merge(config, template_config)
    
    # 3. Load user config if explicitly specified (final override)
    if config_spec:
        config_path = get_config_path(config_spec)
        console.print(f"[dim]Applying user config from '{config_path}' (final override)[/dim]")
        user_config = yaml.safe_load(config_path.read_text())
        config = _deep_merge(config, user_config)

    tools_cfg = config.get("tools") or {}
    if tools_cfg:
        if "bash" in tools_cfg:
            config.setdefault("model", {}).setdefault("bash_tool", tools_cfg["bash"])
        if "profiling" in tools_cfg:
            config.setdefault("model", {}).setdefault("profiling", tools_cfg["profiling"])
        if "profiling_type" in tools_cfg:
            config.setdefault("agent", {}).setdefault("profiling_type", tools_cfg["profiling_type"])
        if tools_cfg.get("profiling") and "profiling_type" not in tools_cfg:
            config.setdefault("agent", {}).setdefault("profiling_type", "profiling")
        if "strategy_manager" in tools_cfg:
            config.setdefault("agent", {}).setdefault("use_strategy_manager", tools_cfg["strategy_manager"])
            config.setdefault("model", {}).setdefault("use_strategy_manager", tools_cfg["strategy_manager"])

    # Backward compatibility: legacy top-level tool flags
    if "profiling" in config:
        config.setdefault("model", {}).setdefault("profiling", config["profiling"])
    if "profiling_type" in config:
        config.setdefault("agent", {}).setdefault("profiling_type", config["profiling_type"])
    if config.get("model", {}).get("profiling") and not config.get("agent", {}).get("profiling_type"):
        config.setdefault("agent", {})["profiling_type"] = "profiling"

    # Read task content - if task is a file path, read its content; otherwise use task as-is
    task_content = task
    if task:
        task_path = Path(task)
        if task_path.exists() and task_path.is_file():
            # Read file content regardless of extension (txt, md, etc.)
            task_content = task_path.read_text(encoding="utf-8")
            console.print(f"[bold green]Read task from file: {task_path}[/bold green]")
        elif not task.strip():
            # Empty task, prompt user
            task_content = None
    
    if not task_content and not (kernel_url or preprocess_dir):
        console.print("[bold yellow]What do you want to do?")
        task_content = prompt_session.prompt(
            "",
            multiline=True,
            bottom_toolbar=HTML(
                "Submit task: <b fg='yellow' bg='black'>Esc+Enter</b> | "
                "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
                "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
            ),
        )
        console.print("[bold green]Got that, thanks![/bold green]")

    if yolo:
        config.setdefault("agent", {})["mode"] = "yolo"
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
    if exit_immediately:
        config.setdefault("agent", {})["confirm_exit"] = False
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    # Set use_strategy_manager in model config based on enable_strategies flag
    config.setdefault("model", {})["use_strategy_manager"] = enable_strategies
    model = get_model(model_name, config.get("model", {}))

    # Print model info
    _model_name = getattr(model.config, "model_name", "unknown")
    _api_key = getattr(model.config, "api_key", None)
    if not _api_key:
        _api_key = os.getenv("AMD_LLM_API_KEY") or os.getenv("LLM_GATEWAY_KEY") or os.getenv("ANTHROPIC_API_KEY")
    _api_key_display = f"{_api_key[:8]}..." if _api_key and len(_api_key) > 8 else _api_key or "Not set"
    console.print(f"\\[mini-swe-agent] Using model: [bold cyan]{_model_name}[/bold cyan], API key: [bold cyan]{_api_key_display}[/bold cyan]")

    # ============ Environment setup: RAG or Local ============
    _env_kwargs = config.get("env", {})
    if rag:
        try:
            from minisweagent.mcp_integration.mcp_environment import MCPEnabledEnvironment
            from minisweagent.mcp_integration.prompts import INSTANCE_TEMPLATE, SYSTEM_TEMPLATE
            from minisweagent.mcp_integration.run_agent import DebugMCPEnvironment
        except ImportError as e:
            console.print("[red]Error: RAG retrieval requires langchain dependencies. Run: pip install -e '.[langchain]'[/red]")
            console.print(f"[red]Import error: {e}[/red]")
            raise typer.Exit(1)

        if debug:
            env = DebugMCPEnvironment(**_env_kwargs)
            console.print("[bold yellow]🐛 Debug mode enabled[/bold yellow]")
        else:
            env = MCPEnabledEnvironment(**_env_kwargs)

        config.setdefault("agent", {})["system_template"] = SYSTEM_TEMPLATE
        config.setdefault("agent", {})["instance_template"] = INSTANCE_TEMPLATE
        console.print("[bold green]🔌 RAG knowledge retrieval enabled[/bold green]")
    else:
        env = LocalEnvironment(**_env_kwargs)

    # Load and merge configurations: Command-line > extra_config from yaml > auto-detect
    result = load_and_merge_configs(
        config, repo, test_command, metric, num_parallel, gpu_ids, patch_output,
        task_content, yolo, model, console
    )
    if result == (None, None, None, None, None, None, None):
        console.print("[bold yellow]Continuing without automatic patch saving. You can still interact with the agent.[/bold yellow]")
        repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output, kernel_name = None, None, None, None, [0], None, None
    else:
        repo, test_command, metric, num_parallel, parsed_gpu_ids, patch_output, kernel_name = result

    # ============ LLM-driven pipeline param extraction ============
    # Only run if CLI flags haven't already set pipeline triggers
    if task_content and kernel_url is None and preprocess_dir is None:
        from minisweagent.run.utils.config_editor import prompt_missing_pipeline_params
        from minisweagent.run.utils.task_parser import parse_pipeline_params

        console.print("[bold cyan]Checking task for pipeline parameters...[/bold cyan]")
        pipeline_params = parse_pipeline_params(task_content, model)

        # Apply non-None extracted values (CLI flags still take priority)
        if pipeline_params.get("heterogeneous") is not None and heterogeneous is None:
            heterogeneous = pipeline_params["heterogeneous"]
        if pipeline_params.get("max_rounds") is not None:
            max_rounds = pipeline_params["max_rounds"]
        if pipeline_params.get("start_round") is not None:
            start_round = pipeline_params["start_round"]

        # Prompt for missing required params (kernel_url)
        pipeline_params, should_use_pipeline = prompt_missing_pipeline_params(
            pipeline_params, console, yolo
        )

        if should_use_pipeline:
            if pipeline_params.get("kernel_url"):
                kernel_url = pipeline_params["kernel_url"]
            if pipeline_params.get("preprocess_dir"):
                preprocess_dir = Path(pipeline_params["preprocess_dir"])

    # ============ Pipeline Mode: preprocess -> orchestrate ============
    # Triggered by: --preprocess-dir, --kernel-url, or LLM-extracted params
    _pipeline_kernel = kernel_url or (str(repo) if repo else None)
    _use_pipeline = preprocess_dir is not None or kernel_url is not None

    if _use_pipeline:
        from minisweagent.run.orchestrator import _probe_preprocess_dir, run_orchestrator

        model_cfg = config.get("model", {})
        _pipeline_output = patch_output or Path("geak_output")
        _pipeline_output = Path(_pipeline_output)
        _pipeline_output.mkdir(parents=True, exist_ok=True)

        console.print("[bold cyan]--- GEAK Full Pipeline Mode ---[/bold cyan]")

        # Step 1: Build preprocess_ctx
        if preprocess_dir:
            # Load existing artifacts
            pp_dir = Path(preprocess_dir).resolve()
            if not pp_dir.is_dir():
                console.print(f"[red]ERROR: preprocess directory not found: {preprocess_dir}[/red]")
                raise typer.Exit(1)

            manifest_path = pp_dir / "preprocess_context.json"
            if manifest_path.exists():
                from minisweagent.run.pipeline_types import PreprocessContext
                pc = PreprocessContext.from_dict(json.loads(manifest_path.read_text()))
                preprocess_ctx: dict[str, Any] = {
                    "kernel_path": pc.kernel_path,
                    "repo_root": pc.repo_root,
                    "harness_path": pc.harness_path,
                    "output_dir": pc.preprocess_dir,
                    "preprocess_dir": pc.preprocess_dir,
                    "commandment_path": pc.commandment_path,
                    "codebase_context_path": pc.codebase_context_path,
                    "baseline_metrics_path": pc.baseline_metrics_path,
                    "profiling_path": pc.profiling_result_path,
                    "discovery": pc.discovery,
                }
            else:
                logger.warning("preprocess_context.json not found, falling back to file probing")
                pc = _probe_preprocess_dir(pp_dir)
                preprocess_ctx = {
                    "kernel_path": pc.kernel_path,
                    "repo_root": pc.repo_root,
                    "harness_path": pc.harness_path,
                    "output_dir": pc.preprocess_dir,
                    "preprocess_dir": pc.preprocess_dir,
                    "commandment_path": pc.commandment_path,
                    "codebase_context_path": pc.codebase_context_path,
                    "baseline_metrics_path": pc.baseline_metrics_path,
                    "profiling_path": pc.profiling_result_path,
                    "discovery": pc.discovery,
                }

            # Load inline data from artifact files
            if pc.commandment_path and Path(pc.commandment_path).exists():
                preprocess_ctx["commandment"] = Path(pc.commandment_path).read_text()
            if pc.baseline_metrics_path and Path(pc.baseline_metrics_path).exists():
                preprocess_ctx["baseline_metrics"] = json.loads(Path(pc.baseline_metrics_path).read_text())
            if pc.profiling_result_path and Path(pc.profiling_result_path).exists():
                preprocess_ctx["profiling"] = json.loads(Path(pc.profiling_result_path).read_text())

            console.print(f"[bold green]Loaded preprocessing artifacts from: {pp_dir}[/bold green]")
        else:
            # Run preprocessing automatically
            from minisweagent.run.preprocess.preprocessor import run_preprocessor

            _kernel_source = kernel_url or str(repo)
            console.print(f"[dim]Kernel source: {_kernel_source}[/dim]")
            console.print(f"[dim]Output dir: {_pipeline_output}[/dim]")

            preprocess_ctx = run_preprocessor(
                _kernel_source,
                output_dir=_pipeline_output,
                gpu_id=parsed_gpu_ids[0] if parsed_gpu_ids else 0,
                model=model,
                model_factory=lambda: get_model(model_name, config.get("model", {})),
                console=console,
                harness=None,
                repo=repo,
                eval_command=test_command,
            )
            preprocess_ctx.setdefault("preprocess_dir", str(_pipeline_output))
            preprocess_ctx.setdefault("output_dir", str(_pipeline_output))

            console.print("[bold green]Preprocessing complete.[/bold green]")

        # Step 2: Auto-detect kernel type → set heterogeneous flag
        if heterogeneous is None:
            # Auto-detect from kernel type
            _kernel_path = preprocess_ctx.get("kernel_path", "")
            _discovery = preprocess_ctx.get("discovery") or {}
            _kernel_info = _discovery.get("kernel") or {}
            _kernel_type = _kernel_info.get("type")

            if not _kernel_type and _kernel_path:
                from minisweagent.agents.heterogeneous.task_generator import _infer_kernel_type
                _kernel_type = _infer_kernel_type(Path(_kernel_path))

            if _kernel_type == "triton":
                heterogeneous = True
                console.print(f"[bold cyan]Auto-detected kernel type: {_kernel_type} -> heterogeneous mode[/bold cyan]")
            else:
                heterogeneous = False
                _ktype_display = _kernel_type or "unknown"
                console.print(f"[bold cyan]Auto-detected kernel type: {_ktype_display} -> homogeneous mode[/bold cyan]")
        else:
            _mode_label = "heterogeneous" if heterogeneous else "homogeneous"
            console.print(f"[bold cyan]Mode override: {_mode_label}[/bold cyan]")

        # Step 3: Call run_orchestrator
        console.print(f"[dim]Rounds: {start_round}-{max_rounds}, GPUs: {parsed_gpu_ids}[/dim]")

        report = run_orchestrator(
            preprocess_ctx=preprocess_ctx,
            gpu_ids=parsed_gpu_ids or [0],
            model=model,
            model_factory=lambda: get_model(model_name, model_cfg),
            output_dir=_pipeline_output,
            max_rounds=max_rounds,
            start_round=start_round,
            heterogeneous=heterogeneous,
            console=console,
        )

        console.print("\n[bold green]Pipeline complete.[/bold green]")
        if report:
            report_dict = report.to_dict() if hasattr(report, "to_dict") else report
            console.print(f"[dim]{json.dumps(report_dict, indent=2, default=str)[:500]}[/dim]")
        return None

    # ============ Legacy Agent Mode (interactive / no pipeline) ============
    if create_test or not test_command:
        if not repo:
            raise ValueError("repo is required for --create-test or when test_command is missing. Please pass --repo.")
        console.print(
            "[bold yellow]No test_command provided (or --create-test enabled). "
            "Will auto-create/search unit tests and infer a test command via UnitTestAgent...[/bold yellow]"
        )
        test_command = run_unit_test_agent(
            model=get_model(model_name, config.get("model", {})),
            repo=repo,
            kernel_name=kernel_name or "unknown",
            log_dir=patch_output,
        )
        console.print(f"[bold green]Using UnitTestAgent test_command:[/bold green] {test_command}")

    # Choose base agent class
    if enable_strategies:
        base_agent_class = StrategyInteractiveAgent
        console.print(f"[bold cyan]Using Strategy Agent with strategy file: {strategy_file}[/bold cyan]")
    else:
        if visual == (os.getenv("MSWEA_VISUAL_MODE_DEFAULT", "false") == "false"):
            base_agent_class = TextualAgent
        else:
            base_agent_class = InteractiveAgent
        console.print(f"[bold cyan]Using Interactive Agent (visual={'on' if base_agent_class == TextualAgent else 'off'})[/bold cyan]")

    # Configure agent settings
    agent_config = config.get("agent", {})
    agent_config["use_strategy_manager"] = enable_strategies
    if enable_strategies:
        agent_config["strategy_file_path"] = strategy_file

    agent_config["save_patch"] = True
    agent_config["test_command"] = test_command or config.get("patch", {}).get("test_command")
    patch_dir = patch_output or config.get("patch", {}).get("patch_output_dir") or (global_config_dir / "patches")
    agent_config["patch_output_dir"] = str(patch_dir)
    agent_config["metric"] = metric or config.get("patch", {}).get("metric")

    log_dir = Path(patch_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    agent_log_file = log_dir / "mini_agent.log"

    agent_class = ParallelAgent
    agent_config["agent_class"] = base_agent_class
    agent_config["num_parallel"] = num_parallel or 1
    agent_config["gpu_ids"] = parsed_gpu_ids

    repo_path = repo or config.get("patch", {}).get("repo")
    if repo_path:
        p = Path(repo_path)
        if not p.exists():
            resolved = _resolve_path_case(p)
            if resolved is not None:
                p = resolved
        agent_config["repo"] = str(p.resolve())

    if num_parallel and num_parallel > 1:
        console.print(f"[bold cyan]Using Parallel Mode: {num_parallel} agents[/bold cyan]")
        console.print(f"[dim]GPU IDs: {parsed_gpu_ids}[/dim]")
        if agent_config.get("repo"):
            console.print(f"[dim]Repository: {agent_config['repo']}[/dim]")
        else:
            console.print("[bold yellow]Warning: No repo path specified for parallel execution[/bold yellow]")
    else:
        console.print("[bold cyan]Using Single Agent Mode[/bold cyan]")
        console.print(f"[dim]Using GPU: {parsed_gpu_ids[0]}[/dim]")
        if agent_config.get("repo"):
            console.print(f"[dim]Repository: {agent_config['repo']}[/dim]")

    agent = agent_class(model, env, **agent_config)
    agent.log_file = agent_log_file
    console.print(f"[dim]Agent log: {agent_log_file}[/dim]")

    try:
        run_result = agent.run(
            task_content,
            output=output,
            save_traj_fn=save_traj,
            console=console,
            model_factory=lambda: get_model(model_name, config.get("model", {})),
            env_factory=lambda: (MCPEnabledEnvironment if rag else LocalEnvironment)(**copy.deepcopy(_env_kwargs)),
        )
        if isinstance(run_result, tuple):
            _exit_status, result = run_result
        else:
            _exit_status, result = "Completed", run_result
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)

    return agent


if __name__ == "__main__":
    app()
