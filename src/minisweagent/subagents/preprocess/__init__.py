"""Preprocess subagents — one-shot LLM tasks run during preprocessing.

Populated by PR-2:
- HarnessBuilder      — adapts user test file into universal-contract harness
- KernelAnalysisAgent — produces analysis rubric markdown
- UnitTestAgent       — generates test skeleton when no user tests exist
- ShapeFixerAgent     — fixes kernel shape mismatches
"""
