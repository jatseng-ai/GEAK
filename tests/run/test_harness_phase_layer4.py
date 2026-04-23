"""Tests for Workstream D1 — Layer 4 of HarnessPhase (HarnessBuilder invocation).

Pins:
  - HarnessBuilder is invoked when ctx.language is set + model is
    available + harness_template is populated
  - Falls back (returns False) when language is None
  - Falls back when model is unavailable
  - Falls back when harness_template is empty
  - Falls back when HarnessBuildFailed is raised
  - Success path populates ctx.harness_path + ctx.harness + ctx.test_command
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.preprocess.phases.base import PhaseContext
from minisweagent.run.preprocess.phases.harness import HarnessPhase


def _make_language(tmp_path: Path, *, template_body: str = "# jinja") -> MagicMock:
    lang = MagicMock()
    lang.name = "triton"
    lang.harness_template = template_body
    lang.builder_hints = "hints"
    lang.system_prompt = "you are the worker"
    (tmp_path / "system_prompt.md").write_text(lang.system_prompt)
    lang.system_prompt_path = tmp_path / "system_prompt.md"
    return lang


def _valid_harness_code() -> str:
    return """\
import argparse
p = argparse.ArgumentParser()
mutex = p.add_mutually_exclusive_group(required=True)
mutex.add_argument("--correctness", action="store_true")
mutex.add_argument("--benchmark", action="store_true")
mutex.add_argument("--full-benchmark", action="store_true")
mutex.add_argument("--profile", action="store_true")
a = p.parse_args()
print("GEAK_RESULT_LATENCY_MS=1.0")
print("GEAK_RESULT_SPEEDUP=1.0")
"""


class TestHarnessBuilderInvocation:
    def test_skipped_when_language_is_none(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = None  # no language detected
        ctx.model = MagicMock()

        HarnessPhase().run(ctx)
        # No harness_path set -> deferred to legacy
        assert ctx.harness_path is None or ctx.harness_path == ""

    def test_skipped_when_model_is_unavailable(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _make_language(tmp_path)
        ctx.model = None
        ctx.model_factory = None

        HarnessPhase().run(ctx)
        assert ctx.harness_path is None or ctx.harness_path == ""

    def test_skipped_when_harness_template_is_empty(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _make_language(tmp_path, template_body="")
        ctx.model = MagicMock()

        HarnessPhase().run(ctx)
        assert ctx.harness_path is None or ctx.harness_path == ""

    def test_success_path_populates_all_harness_fields(self, tmp_path: Path) -> None:
        """When HarnessBuilder succeeds, ctx.harness_path + ctx.harness +
        ctx.test_command should all be set."""
        kernel = tmp_path / "k.py"
        kernel.write_text("@triton.jit\ndef foo(): pass\n")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.repo_root = str(tmp_path)
        ctx.language = _make_language(tmp_path)
        # Model returns a valid harness on first call
        model = MagicMock()
        model.query = MagicMock(return_value=_valid_harness_code())
        ctx.model = model

        HarnessPhase().run(ctx)

        expected_harness = tmp_path / "harness.py"
        assert ctx.harness_path == str(expected_harness)
        assert ctx.harness == str(expected_harness)
        assert ctx.test_command is not None
        assert "--correctness" in ctx.test_command
        assert str(expected_harness) in ctx.test_command
        assert expected_harness.exists()

    def test_falls_back_when_harness_builder_fails(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _make_language(tmp_path)
        # Model always returns garbage -> HarnessBuildFailed after retries
        model = MagicMock()
        model.query = MagicMock(return_value="def main(): pass\n")
        ctx.model = model

        HarnessPhase().run(ctx)
        # Phase should NOT propagate the failure; ctx.harness_path stays
        # unset so the orchestrator falls back to the legacy path.
        assert ctx.harness_path is None or ctx.harness_path == ""

    def test_model_factory_used_when_model_is_none(self, tmp_path: Path) -> None:
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.kernel_path = str(kernel)
        ctx.language = _make_language(tmp_path)
        ctx.model = None
        factory_model = MagicMock()
        factory_model.query = MagicMock(return_value=_valid_harness_code())
        ctx.model_factory = lambda: factory_model

        HarnessPhase().run(ctx)
        assert ctx.harness_path == str(tmp_path / "harness.py")


class TestLayerOrdering:
    """Layer 1 > 2 > 3 > 4: earlier layers short-circuit the builder."""

    def test_layer1_existing_harness_path_wins(self, tmp_path: Path) -> None:
        harness = tmp_path / "explicit.py"
        harness.write_text(_valid_harness_code())
        ctx = PhaseContext(output_dir=tmp_path)
        ctx.harness_path = str(harness)
        ctx.language = _make_language(tmp_path)
        ctx.model = MagicMock()

        with patch(
            "minisweagent.subagents.preprocess.harness_builder.HarnessBuilder"
        ) as mock_builder:
            HarnessPhase().run(ctx)
            mock_builder.assert_not_called()

        # harness_path stays as-is
        assert ctx.harness_path == str(harness)

    def test_layer2_explicit_harness_wins_over_layer4(self, tmp_path: Path) -> None:
        harness = tmp_path / "user_supplied.py"
        harness.write_text(_valid_harness_code())
        kernel = tmp_path / "k.py"
        kernel.write_text("pass")
        ctx = PhaseContext(output_dir=tmp_path, harness=str(harness))
        ctx.kernel_path = str(kernel)
        ctx.language = _make_language(tmp_path)
        ctx.model = MagicMock()

        with patch(
            "minisweagent.subagents.preprocess.harness_builder.HarnessBuilder"
        ) as mock_builder:
            HarnessPhase().run(ctx)
            mock_builder.assert_not_called()

        assert ctx.harness_path == str(harness.resolve())


class TestApplyTestCommand:
    def test_test_command_shape_matches_legacy(self, tmp_path: Path) -> None:
        """The test_command string format matches
        ``_build_deterministic_test_command`` from the legacy monolith:

            python3 <harness_path> --correctness
        """
        import shlex
        import sys

        harness_path = str(tmp_path / "harness.py")
        ctx = PhaseContext()
        HarnessPhase._apply_test_command(ctx, harness_path)
        expected = (
            f"{shlex.quote(sys.executable)} {shlex.quote(harness_path)} --correctness"
        )
        assert ctx.test_command == expected

    def test_apply_test_command_respects_existing_value(self, tmp_path: Path) -> None:
        ctx = PhaseContext()
        ctx.test_command = "pre-existing"
        HarnessPhase._apply_test_command(ctx, str(tmp_path / "harness.py"))
        assert ctx.test_command == "pre-existing"
