"""Explore phase — render commandment + (future) kernel analysis rubric.

Inputs  (read from ctx):
  - kernel_path, harness_path, repo_root
  - test_command, eval_command, correctness_command, performance_command
  - discovery  (to infer kernel language)
  - baseline_metrics, profiling, codebase_context_path  (future: feed
    ``KernelAnalysisAgent``)

Outputs (written to ctx):
  - commandment           (str markdown)
  - commandment_path      (str path to COMMANDMENT.md)
  - kernel_analysis_md    (future — populated by KernelAnalysisAgent)

Absorbs step 7 of the legacy monolith.  Future commit will replace the
``commandment.py`` 530-LoC Python branching with a language-driven
Jinja template + ``validate_commandment`` contract.  Until then this
phase delegates to the existing generator.
"""

from __future__ import annotations

import logging
from pathlib import Path

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

logger = logging.getLogger(__name__)


def _join_cmd(cmd: str | list[str] | None) -> str | None:
    if cmd is None:
        return None
    if isinstance(cmd, list):
        return " && ".join(c.strip() for c in cmd if c.strip()) or None
    return cmd.strip() or None


class ExplorePhase(Phase):
    """Render the commandment markdown.

    Two paths (mirrors legacy monolith step 7):

      1. ``eval_command`` path: call
         ``generate_commandment_from_commands(kernel_path,
         correctness_command, performance_command, repo_root)``.
      2. ``test_command`` path (Triton-style with a harness): call
         ``generate_commandment(kernel_path, harness_path, repo_root,
         kernel_language)``.

    Both write ``{output_dir}/COMMANDMENT.md`` and populate
    ``ctx.commandment`` + ``ctx.commandment_path``.
    """

    name = "explore"

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()
        if ctx.commandment_path:
            logger.info("  commandment already rendered; skipping.")
            return

        # Require prerequisites — defer to legacy fallback when missing.
        if not ctx.kernel_path:
            logger.debug("ExplorePhase: no kernel_path yet; deferring to legacy fallback.")
            return
        if not (ctx.test_command or ctx.eval_command):
            logger.info("  Skipping commandment (no test_command or eval_command)")
            ctx.phases_run.append(self.name)
            return

        output_dir = Path(ctx.output_dir)
        correctness_cmd = _join_cmd(ctx.correctness_command)
        perf_cmd = _join_cmd(ctx.performance_command)

        commandment: str | None = None

        if ctx.eval_command:
            try:
                from minisweagent.run.preprocess.commandment import (
                    generate_commandment_from_commands,
                )

                commandment = generate_commandment_from_commands(
                    kernel_path=ctx.kernel_path,
                    compile_command=None,
                    correctness_command=correctness_cmd,
                    performance_command=perf_cmd or ctx.eval_command,
                    repo_root=ctx.repo_root,
                )
                logger.info("  COMMANDMENT.md generated (from eval command)")
            except Exception as exc:
                logger.warning("[yellow]Commandment from command failed: %s[/yellow]", exc, exc_info=True)
        elif ctx.test_command:
            try:
                from minisweagent.run.preprocess.commandment import generate_commandment
                from minisweagent.run.preprocess.discovery_types import (
                    _infer_kernel_language,
                )
                from minisweagent.run.preprocess.harness_utils import (
                    extract_harness_path,
                )

                harness_path = ctx.harness_path or extract_harness_path(ctx.test_command)
                kernel_type = (ctx.discovery or {}).get("kernel", {}).get("type", "")
                kernel_language = _infer_kernel_language(Path(ctx.kernel_path), kernel_type)
                commandment = generate_commandment(
                    kernel_path=ctx.kernel_path,
                    harness_path=harness_path,
                    repo_root=ctx.repo_root,
                    kernel_language=kernel_language,
                )
                logger.info("  COMMANDMENT.md generated (from harness)")
            except Exception as exc:
                logger.warning("[yellow]Commandment failed: %s[/yellow]", exc, exc_info=True)

        ctx.commandment = commandment
        if commandment:
            cm_path = output_dir / "COMMANDMENT.md"
            cm_path.write_text(commandment)
            ctx.commandment_path = str(cm_path)

        ctx.phases_run.append(self.name)


__all__ = ["ExplorePhase"]
