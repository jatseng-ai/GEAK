"""``HarnessBuilder`` — one-shot subagent that adapts user tests into a contract harness.

Per execution plan §0.5(b) Harness phase:

  - Reads the user's existing test file(s) (discovery output).
  - Reads ``self.language.harness_template_path`` (Jinja) + optional
    ``self.language.builder_hints_path`` (language-specific idioms).
  - Produces a single harness.py file that conforms to the universal
    contract:
        * argparse with --correctness / --benchmark / --full-benchmark
          / --profile mutually-exclusive flags
        * emits ``GEAK_RESULT_LATENCY_MS=<float>`` /
          ``GEAK_RESULT_SPEEDUP=<float>`` on stdout
        * exits with the correctness boolean as return code

This implementation is a skeleton: the class + config contract +
prompt composition are in place, but the actual LLM call and
contract-validation loop are scheduled for a follow-up commit (the
existing ``create_validated_harness`` in harness_utils.py still
handles runtime harness creation via the legacy monolith's
UnitTestAgent fallback).  Skeletonising the class here lets us:

  1. Lock in the SubagentBase-subclass contract (no ``OptimizationAgent``
     composition — HarnessBuilder is a narrow one-shot task).
  2. Register the file location so the ``check_subagent_location.py``
     CI gate stays green.
  3. Give callers a stable import path when they eventually flip to
     the new agent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from minisweagent.subagents.base import SubagentBase

logger = logging.getLogger(__name__)


class HarnessBuilder(SubagentBase):
    """One-shot harness builder.  Overrides ``run``.

    Contract (target signature, filled in when implementation lands):

        harness_path = HarnessBuilder(language, config).run(
            user_test_files=[...],
            kernel_path=...,
            repo_root=...,
            out_path=...,
        )

    Subclass override: ``run()`` — SubagentBase contract requires
    exactly one of (run, loop); HarnessBuilder is a one-shot task so
    it uses run.  No ``loop()`` here.
    """

    def run(self, **inputs: Any) -> str | dict:
        """Build a validated harness from user test files.

        Not yet implemented.  The legacy monolith's
        ``create_validated_harness`` (in harness_utils.py) still
        handles runtime harness creation; this skeleton will replace
        that in a subsequent commit once the Jinja harness templates
        + contract validator are in place.

        Expected inputs (when implemented):
          - ``user_test_files: list[Path]``
          - ``kernel_path: Path``
          - ``repo_root: Path``
          - ``out_path: Path``
          - ``discovery_context: str``  (optional — codebase context)

        Returns: ``str`` (path to the built harness.py).
        """
        raise NotImplementedError(
            "HarnessBuilder.run is not implemented yet.  Until this lands, "
            "the preprocessing pipeline falls back to the legacy "
            "``create_validated_harness`` helper in "
            "run/preprocess/harness_utils.py.  See the class docstring for "
            "the target signature."
        )


__all__ = ["HarnessBuilder"]
