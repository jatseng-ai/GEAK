"""Configuration for DRA (Deep Research Artifact) generation.

DRA is opt-in. The default config is conservative and bounded so the
multi-stage loop cannot balloon in cost.

Env vars:
  GEAK_DRA_DISABLE=1                   -> turn DRA off entirely
  GEAK_DRA_MODEL=claude-opus-4.6       -> model used for all DRA LLM calls
  GEAK_DRA_API_KEY=...                 -> optional override
  GEAK_DRA_MAX_QUESTIONS=8             -> max questions through full evidence (Stage 4)
  GEAK_DRA_MAX_BLINDSPOTS=4            -> max follow-up doubts in second pass (Stage 6)
  GEAK_DRA_RETRIEVAL_TOP_K=8           -> chunks per question retrieval call
  GEAK_DRA_RUN_EXPERIMENTAL=1          -> also produce experimental_directions.{md,json}
  GEAK_DRA_INDEX_PATH=...              -> override path to RAG index (otherwise default)
  GEAK_DRA_USE_PRIOR_RUNS=1            -> include cross_session prior runs in evidence
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class DRAConfig:
    enabled: bool = True
    model_name: str = "claude-opus-4.6"
    api_key: str | None = None

    # Cost / latency budget (the staged design does not bound itself).
    max_questions: int = 8
    max_blindspots: int = 4
    retrieval_top_k: int = 8

    # Optional second artifact.
    run_experimental: bool = True

    # Optional evidence sources.
    use_prior_runs: bool = False
    index_path: str | None = None

    @classmethod
    def from_env(cls) -> DRAConfig:
        if _env_bool("GEAK_DRA_DISABLE", False):
            return cls(enabled=False)

        return cls(
            enabled=True,
            model_name=os.environ.get("GEAK_DRA_MODEL", "claude-opus-4.6").strip(),
            api_key=os.environ.get("GEAK_DRA_API_KEY", "").strip() or None,
            max_questions=_env_int("GEAK_DRA_MAX_QUESTIONS", 8),
            max_blindspots=_env_int("GEAK_DRA_MAX_BLINDSPOTS", 4),
            retrieval_top_k=_env_int("GEAK_DRA_RETRIEVAL_TOP_K", 8),
            run_experimental=_env_bool("GEAK_DRA_RUN_EXPERIMENTAL", True),
            use_prior_runs=_env_bool("GEAK_DRA_USE_PRIOR_RUNS", False),
            index_path=os.environ.get("GEAK_DRA_INDEX_PATH", "").strip() or None,
        )
