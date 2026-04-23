"""Regression tests for Workstream I1 — absorbed preprocessor capabilities.

Each test pins one row of plan §13.2-A (the 9 preprocessor-monolith
capabilities the phase modules had to absorb before the monolith
could be deleted).  A failure here means a capability that the
legacy monolith provided is no longer working through the phase
orchestrator — i.e. the monolith cannot yet be safely deleted.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from minisweagent.run.preprocess.orchestrator import PreprocessOrchestrator
from minisweagent.run.preprocess.phases.base import Phase, PhaseContext
from minisweagent.run.preprocess.phases.baseline import BaselinePhase
from minisweagent.run.preprocess.phases.discovery import DiscoveryPhase
from minisweagent.run.preprocess.phases.explore import ExplorePhase
from minisweagent.run.preprocess.phases.harness import HarnessPhase
from minisweagent.run.preprocess.phases.translation import TranslationPhase


class TestRow1CorrectnessDict:
    """§13.2-A row 1: ``ctx.correctness`` must be populated when
    BaselinePhase runs the eval_command correctness gate."""

    def test_phase_context_has_correctness_field(self) -> None:
        ctx = PhaseContext()
        assert hasattr(ctx, "correctness")
        assert ctx.correctness is None

    def test_correctness_included_in_to_dict_output(self) -> None:
        ctx = PhaseContext()
        ctx.correctness = {"returncode": 0}
        d = ctx.to_dict()
        assert "correctness" in d
        assert d["correctness"]["returncode"] == 0

    def test_baseline_phase_populates_correctness_on_eval_path(self, tmp_path: Path) -> None:
        ctx = PhaseContext(
            output_dir=tmp_path,
            eval_command="true",
            correctness_command="true",
        )
        ctx.kernel_path = "/tmp/k.py"
        ctx.repo_root = str(tmp_path)

        # _run_shell writes stdout/stderr files; we stub it at the module level
        # so we don't actually invoke /bin/bash.
        with patch(
            "minisweagent.run.preprocess.phases.baseline._run_shell",
        ) as mock_run:
            import subprocess as _sub

            mock_run.return_value = _sub.CompletedProcess(
                args=["true"], returncode=0, stdout="ok", stderr=""
            )
            BaselinePhase().run(ctx)

        assert ctx.correctness is not None
        assert ctx.correctness["command"] == "true"
        assert ctx.correctness["returncode"] == 0
        assert ctx.correctness["stdout_path"].endswith("correctness_stdout.txt")

    def test_baseline_phase_raises_on_non_zero_correctness(self, tmp_path: Path) -> None:
        ctx = PhaseContext(
            output_dir=tmp_path,
            eval_command="false",
            correctness_command="false",
        )
        ctx.kernel_path = "/tmp/k.py"
        ctx.repo_root = str(tmp_path)

        with patch(
            "minisweagent.run.preprocess.phases.baseline._run_shell",
        ) as mock_run:
            import subprocess as _sub

            mock_run.return_value = _sub.CompletedProcess(
                args=["false"], returncode=1, stdout="", stderr="boom"
            )
            with pytest.raises(RuntimeError, match="correctness_command failed"):
                BaselinePhase().run(ctx)


class TestRow3HarnessOnlyEnvVar:
    """§13.2-A row 3: ``GEAK_HARNESS_ONLY=1`` must make the orchestrator
    exit early after HarnessPhase (skipping Baseline + Explore)."""

    def _make_recording_phases(self) -> list[Phase]:
        class _RecordingPhase(Phase):
            def __init__(self, name: str) -> None:
                self.name = name

            def run(self, ctx: PhaseContext) -> None:
                ctx.phases_run.append(self.name)

        return [
            _RecordingPhase("translation"),
            _RecordingPhase("discovery"),
            _RecordingPhase("harness"),
            _RecordingPhase("baseline"),
            _RecordingPhase("explore"),
        ]

    def test_env_var_stops_after_harness_phase(self) -> None:
        # Patch HarnessPhase.name alias so orchestrator recognises the
        # stub phase as "harness" for the env-var comparison.
        phases = self._make_recording_phases()
        ctx = PhaseContext()

        with patch.dict(os.environ, {"GEAK_HARNESS_ONLY": "1"}, clear=False):
            PreprocessOrchestrator(phases=phases).run(ctx)

        assert ctx.phases_run == ["translation", "discovery", "harness"]
        # Baseline + Explore did NOT run
        assert "baseline" not in ctx.phases_run
        assert "explore" not in ctx.phases_run

    def test_env_var_unset_runs_all_phases(self) -> None:
        phases = self._make_recording_phases()
        ctx = PhaseContext()

        # Explicitly clear the env var for this test
        env = {k: v for k, v in os.environ.items() if k != "GEAK_HARNESS_ONLY"}
        with patch.dict(os.environ, env, clear=True):
            PreprocessOrchestrator(phases=phases).run(ctx)

        assert ctx.phases_run == ["translation", "discovery", "harness", "baseline", "explore"]


class TestRow5ExploreSetsTestCommand:
    """§13.2-A row 5: ExplorePhase must set ``ctx.test_command =
    eval_command`` on the eval path (legacy preprocessor.py:1204)."""

    def test_explore_overrides_test_command_on_eval_path(self, tmp_path: Path) -> None:
        ctx = PhaseContext(
            output_dir=tmp_path,
            eval_command="make && ./bench",
        )
        ctx.kernel_path = "/tmp/k.py"
        ctx.repo_root = str(tmp_path)
        # test_command is initially empty

        with patch(
            "minisweagent.run.preprocess.commandment.generate_commandment_from_commands",
            return_value="# commandment body",
        ):
            ExplorePhase().run(ctx)

        assert ctx.test_command == "make && ./bench"

    def test_explore_does_not_override_existing_test_command(self, tmp_path: Path) -> None:
        ctx = PhaseContext(
            output_dir=tmp_path,
            eval_command="make && ./bench",
        )
        ctx.kernel_path = "/tmp/k.py"
        ctx.repo_root = str(tmp_path)
        ctx.test_command = "pre-existing"

        with patch(
            "minisweagent.run.preprocess.commandment.generate_commandment_from_commands",
            return_value="# commandment body",
        ):
            ExplorePhase().run(ctx)

        assert ctx.test_command == "pre-existing"


class TestRow6SplitHarnessHintPickup:
    """§13.2-A row 6: HarnessPhase must promote DiscoveryPhase's
    split_harness_hint to ctx.harness when valid and no explicit
    --harness was supplied."""

    def test_phase_context_has_split_harness_hint(self) -> None:
        ctx = PhaseContext()
        assert hasattr(ctx, "split_harness_hint")
        assert ctx.split_harness_hint is None

    def test_harness_phase_promotes_valid_split_harness(self, tmp_path: Path) -> None:
        split_harness = tmp_path / "test_merged_harness.py"
        split_harness.write_text("# dummy harness content")
        ctx = PhaseContext()
        ctx.split_harness_hint = str(split_harness)

        with patch(
            "minisweagent.run.preprocess.harness_utils.validate_harness",
            return_value=(True, []),
        ):
            HarnessPhase().run(ctx)

        assert ctx.harness == str(split_harness)

    def test_harness_phase_ignores_invalid_split_harness(self, tmp_path: Path) -> None:
        split_harness = tmp_path / "bad_harness.py"
        split_harness.write_text("# dummy")
        ctx = PhaseContext()
        ctx.split_harness_hint = str(split_harness)

        with patch(
            "minisweagent.run.preprocess.harness_utils.validate_harness",
            return_value=(False, ["missing --correctness"]),
        ):
            HarnessPhase().run(ctx)

        assert ctx.harness is None

    def test_harness_phase_does_not_override_explicit_harness(self, tmp_path: Path) -> None:
        split_harness = tmp_path / "split.py"
        split_harness.write_text("# split harness")
        ctx = PhaseContext()
        ctx.split_harness_hint = str(split_harness)
        ctx.harness = "/user/supplied/harness.py"  # caller-supplied takes priority

        with patch(
            "minisweagent.run.preprocess.harness_utils.validate_harness",
            return_value=(True, []),
        ):
            HarnessPhase().run(ctx)

        assert ctx.harness == "/user/supplied/harness.py"  # unchanged

    def test_harness_phase_handles_nonexistent_hint_path(self) -> None:
        ctx = PhaseContext()
        ctx.split_harness_hint = "/nonexistent/path/harness.py"

        # Should not raise, just silently skip the pickup.
        HarnessPhase().run(ctx)
        assert ctx.harness is None


class TestTranslationPhaseStillSkippedWhenNoTarget:
    """Regression check: adding new fields + logic didn't break the
    default ``target_language=None`` fast path."""

    def test_translation_phase_is_not_applicable_by_default(self) -> None:
        ctx = PhaseContext()
        assert TranslationPhase().is_applicable(ctx) is False
