"""Harness phase — produce a validated harness + test_command.

Inputs: ``ctx.kernel_path``, ``ctx.repo_root``, ``ctx.discovery``,
``ctx.harness`` (optional explicit path), plus CLI eval hooks
(``eval_command``, ``correctness_command``, ``performance_command``).

Output: ``ctx.harness_path``, ``ctx.test_command``,
``ctx.harness_results``, ``ctx.testcase_selection``.

Resolution order (priority, first-match wins):

  1. ``ctx.harness_path`` already set by an upstream phase or caller
     -> run the universal contract validator and return.
  2. ``ctx.harness`` (explicit ``--harness`` from CLI) + static
     ``validate_harness`` passes -> use it.
  3. ``ctx.split_harness_hint`` from DiscoveryPhase + static
     ``validate_harness`` passes -> promote it (§13.2-A row 6).
  4. ``HarnessBuilder`` (LLM subagent) — the new preferred path
     when a model is available.  Consumes
     ``ctx.language.harness_template`` +
     ``ctx.language.builder_hints`` +
     the kernel source + optional user test files from discovery.
     Produces a universal-contract harness.py under
     ``{output_dir}/harness.py`` and runs the contract validator.
  5. Legacy 6-layer monolith fallback (``preprocessor.py:621-944``)
     — invoked via the orchestrator's legacy shim when this phase
     does not populate ``ctx.harness_path``.

Layers 1-3 were already live as of Workstream I1.  **This commit
(D1) adds layer 4**: HarnessBuilder as the preferred path before
falling back to the monolith.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from minisweagent.run.preprocess.phases.base import Phase, PhaseContext

logger = logging.getLogger(__name__)


class HarnessPhase(Phase):
    """Produces ``ctx.harness_path`` + ``ctx.test_command``.

    With D1 landed: HarnessBuilder subagent is the preferred path
    (layer 4); layers 1-3 short-circuit it when applicable; layer 5
    (legacy monolith) runs via the orchestrator's legacy shim when
    this phase does not populate a harness.
    """

    name = "harness"

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()

        # ── Layer 1: harness_path already populated upstream ────────
        if ctx.harness_path:
            logger.info(
                "  harness_path already set by previous phase (%s) — running contract validator.",
                ctx.harness_path,
            )
            self._validate_if_present(ctx.harness_path)
            ctx.phases_run.append(self.name)
            return

        # ── Layer 2: explicit --harness from caller ─────────────────
        if ctx.harness:
            candidate = Path(ctx.harness)
            if candidate.exists() and self._passes_static_validation(candidate):
                ctx.harness_path = str(candidate.resolve())
                self._apply_test_command(ctx, ctx.harness_path)
                logger.info(
                    "  Using caller-supplied harness: %s",
                    ctx.harness_path,
                )
                self._validate_if_present(ctx.harness_path)
                ctx.phases_run.append(self.name)
                return

        # ── Layer 3: split-harness-hint pickup (§13.2-A row 6) ──────
        if ctx.split_harness_hint and not ctx.harness:
            candidate = Path(ctx.split_harness_hint)
            if candidate.exists() and self._passes_static_validation(candidate):
                ctx.harness = str(candidate)
                ctx.harness_path = str(candidate.resolve())
                self._apply_test_command(ctx, ctx.harness_path)
                logger.info("  Promoted split harness to --harness: %s", candidate)
                self._validate_if_present(ctx.harness_path)
                ctx.phases_run.append(self.name)
                return

        # ── Layer 4: HarnessBuilder subagent (the D1 addition) ──────
        if self._try_harness_builder(ctx):
            self._validate_if_present(ctx.harness_path)
            ctx.phases_run.append(self.name)
            return

        # ── Layer 5: defer to legacy monolith fallback ──────────────
        logger.debug("HarnessPhase: no harness_path yet; deferring to legacy fallback.")
        ctx.phases_run.append(self.name)

    # ------------------------------------------------------------------
    # Layer 2/3 helper: static (tuple-returning) validator from harness_utils
    # ------------------------------------------------------------------

    @staticmethod
    def _passes_static_validation(path: Path) -> bool:
        """Run the tuple-returning ``harness_utils.validate_harness``.

        This is the LEGACY static validator (different signature from
        the contract validator).  It's the one the legacy monolith
        uses at layers 2/3 (explicit --harness + split-hint), so we
        match that behaviour here for byte-parity.
        """
        try:
            from minisweagent.run.preprocess.harness_utils import (
                validate_harness as static_validate_harness,
            )

            ok, _errors = static_validate_harness(path)
            return bool(ok)
        except Exception as exc:
            logger.debug(
                "  harness_utils.validate_harness raised %s: %s",
                type(exc).__name__,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Layer 4: HarnessBuilder subagent invocation
    # ------------------------------------------------------------------

    def _try_harness_builder(self, ctx: PhaseContext) -> bool:
        """Attempt to produce a harness via the HarnessBuilder subagent.

        Returns ``True`` when the subagent succeeded and
        ``ctx.harness_path`` now points at a contract-satisfying
        harness.  Returns ``False`` when:

          - No ``KernelLanguage`` resolved (DiscoveryPhase didn't set
            ``ctx.language``).  Without a language, we have no
            harness template / builder hints to drive the LLM.
          - No model available (``ctx.model`` unset AND no model
            factory) — can't make the LLM call.
          - The language bundle has no ``harness_template`` content
            (partial bundles during migration).
          - The subagent raises ``HarnessBuildFailed`` after internal
            retries.

        All failure modes are swallowed (logged at WARN); the
        orchestrator's legacy fallback path then takes over.  That
        preserves the "new preferred path, legacy guaranteed
        fallback" invariant we use for every D-series migration.
        """
        if ctx.language is None:
            logger.debug(
                "  HarnessBuilder: ctx.language is None (DiscoveryPhase did not "
                "resolve a KernelLanguage); falling back to legacy."
            )
            return False

        try:
            template_blob = ctx.language.harness_template
        except Exception:
            template_blob = ""
        if not template_blob.strip():
            logger.debug(
                "  HarnessBuilder: language=%s has no harness_template; "
                "falling back to legacy.",
                getattr(ctx.language, "name", "?"),
            )
            return False

        model = self._resolve_model(ctx)
        if model is None:
            logger.debug(
                "  HarnessBuilder: no model available on ctx; "
                "falling back to legacy."
            )
            return False

        if not ctx.kernel_path or not Path(ctx.kernel_path).is_file():
            logger.debug(
                "  HarnessBuilder: no kernel_path yet; falling back to legacy."
            )
            return False

        try:
            from minisweagent.subagents.base import SubagentConfig
            from minisweagent.subagents.preprocess.harness_builder import (
                HarnessBuildFailed,
                HarnessBuilder,
            )
        except Exception as exc:
            logger.warning(
                "[yellow]HarnessBuilder import failed (%s); falling back to legacy.[/yellow]",
                exc,
            )
            return False

        out_path = Path(ctx.output_dir) / "harness.py"
        config = self._build_subagent_config(model)
        builder = HarnessBuilder(language=ctx.language, config=config)
        # Attach the caller's model so _query_model reuses it.
        builder.model = model  # type: ignore[attr-defined]

        user_tests = self._extract_user_test_files(ctx)

        try:
            result = builder.run(
                kernel_path=Path(ctx.kernel_path),
                out_path=out_path,
                repo_root=Path(ctx.repo_root) if ctx.repo_root else None,
                user_test_files=user_tests,
                discovery_context=self._read_codebase_context(ctx),
                max_retries=int(config.extra.get("max_retries", 1)),
            )
        except HarnessBuildFailed as exc:
            logger.warning(
                "[yellow]HarnessBuilder failed (%s); falling back to legacy.[/yellow]",
                exc,
            )
            return False
        except Exception as exc:  # noqa: BLE001 — shield the phase from LLM / model surprises
            logger.warning(
                "[yellow]HarnessBuilder raised %s: %s; falling back to legacy.[/yellow]",
                type(exc).__name__,
                exc,
            )
            return False

        harness_path = result.get("harness_path") if isinstance(result, dict) else None
        if not harness_path:
            return False

        ctx.harness_path = str(harness_path)
        ctx.harness = str(harness_path)
        self._apply_test_command(ctx, ctx.harness_path)
        logger.info(
            "  HarnessBuilder produced harness: %s (attempts=%s)",
            ctx.harness_path,
            result.get("attempts_used", "?"),
        )
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model(ctx: PhaseContext) -> Any:
        """Return a model instance, constructing one from the factory if needed."""
        if ctx.model is not None:
            return ctx.model
        factory = getattr(ctx, "model_factory", None)
        if callable(factory):
            try:
                return factory()
            except Exception as exc:
                logger.debug("model_factory raised %s: %s", type(exc).__name__, exc)
                return None
        return None

    @staticmethod
    def _build_subagent_config(model: Any) -> Any:
        """Build a ``SubagentConfig`` loaded from the on-disk YAML.

        The YAML provides the fallback templates + extra.max_retries.
        The live ``model_name`` is replaced with whatever the caller's
        model reports so telemetry stays consistent.
        """
        from minisweagent.subagents.base import SubagentConfig

        # Try the YAML; if it's missing, fall back to an in-memory
        # SubagentConfig so the subagent still runs (tests exercise
        # this path too).
        cfg_path = (
            Path(__file__).resolve().parents[2]  # src/minisweagent/run/
            .parent                               # src/minisweagent/
            / "subagents"
            / "preprocess"
            / "configs"
            / "harness_builder.yaml"
        )
        if cfg_path.is_file():
            try:
                cfg = SubagentConfig(
                    name="harness_builder",
                    model_name=getattr(model, "name", "harness_builder_model"),
                    system_template="",
                    instance_template="",
                    step_limit=1,
                    cost_limit=3.0,
                    temperature=0.2,
                    extra={"max_retries": 1},
                )
                # Reload real fields from YAML so extra.max_retries reflects
                # the on-disk value.
                real = SubagentBase_load_config(cfg_path)
                cfg.step_limit = real.step_limit
                cfg.cost_limit = real.cost_limit
                cfg.temperature = real.temperature
                cfg.system_template = real.system_template
                cfg.instance_template = real.instance_template
                cfg.extra = dict(real.extra)
                return cfg
            except Exception as exc:
                logger.debug("failed to load harness_builder.yaml: %s", exc)

        return SubagentConfig(
            name="harness_builder",
            model_name=getattr(model, "name", "harness_builder_model"),
            system_template="",
            instance_template="",
            step_limit=1,
            cost_limit=3.0,
            temperature=0.2,
            extra={"max_retries": 1},
        )

    @staticmethod
    def _extract_user_test_files(ctx: PhaseContext) -> list[Path]:
        """Pull test-file paths out of DiscoveryPhase's output.

        Tries a handful of well-known shapes in ``ctx.discovery``:
          - ``discovery.tests``            -> list of {command, file?} dicts
          - ``discovery.focused_test.file`` or ``.path``
          - ``discovery.user_tests``       -> list of absolute paths
        Missing / mal-shaped entries are skipped silently.
        """
        if not ctx.discovery:
            return []

        candidates: list[Path] = []
        tests = ctx.discovery.get("tests") or []
        for entry in tests:
            for key in ("file", "path", "harness_path"):
                raw = entry.get(key) if isinstance(entry, dict) else None
                if raw and isinstance(raw, str):
                    p = Path(raw)
                    if p.is_file():
                        candidates.append(p)

        focused = ctx.discovery.get("focused_test") or {}
        if isinstance(focused, dict):
            for key in ("file", "path", "harness_path"):
                raw = focused.get(key)
                if raw and isinstance(raw, str):
                    p = Path(raw)
                    if p.is_file():
                        candidates.append(p)

        for raw in ctx.discovery.get("user_tests") or []:
            if isinstance(raw, str):
                p = Path(raw)
                if p.is_file():
                    candidates.append(p)

        # Dedup preserving order
        seen: set[str] = set()
        unique: list[Path] = []
        for p in candidates:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    @staticmethod
    def _read_codebase_context(ctx: PhaseContext) -> str:
        """Return the codebase-context markdown (or empty string)."""
        if not ctx.codebase_context_path:
            return ""
        try:
            return Path(ctx.codebase_context_path).read_text(encoding="utf-8")
        except Exception:
            return ""

    @staticmethod
    def _apply_test_command(ctx: PhaseContext, harness_path: str) -> None:
        """Set ``ctx.test_command`` from a validated harness path.

        Matches the legacy monolith's ``_build_deterministic_test_command``
        shape (``python3 <harness> --correctness``) so downstream
        consumers that still read ``ctx.test_command`` (save_and_test,
        sub_agent_tool) see the same string shape.
        """
        import shlex
        import sys

        if not ctx.test_command:
            ctx.test_command = (
                f"{shlex.quote(sys.executable)} {shlex.quote(str(harness_path))} --correctness"
            )

    @staticmethod
    def _validate_if_present(path_str: str | None) -> None:
        """Run the universal harness contract validator when a path is available."""
        if not path_str:
            return
        try:
            from minisweagent.kernel_languages.contract import validate_harness

            validate_harness(Path(path_str))
        except Exception as exc:
            logger.warning("[yellow]validate_harness: %s[/yellow]", exc)


# --------------------------------------------------------------------------
# Small import-time shim — SubagentBase._load_config is a staticmethod but we
# want to use it without instantiating a base class, so expose it as a module
# helper here.  Keeps the HarnessPhase body free of "instantiate an abstract
# class" gymnastics.
# --------------------------------------------------------------------------


def SubagentBase_load_config(path: Path):
    from minisweagent.subagents.base import SubagentBase

    return SubagentBase._load_config(path)  # noqa: SLF001 — intentional helper


__all__ = ["HarnessPhase"]
