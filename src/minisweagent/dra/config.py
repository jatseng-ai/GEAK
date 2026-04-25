"""Configuration for DRA (Deep Research Artifact) generation.

DRA is opt-in. The defaults are tuned for grounded multi-source research:

  - 50 ranked questions through the first synthesis pass
  - 3 blindspot rounds with up to 10 follow-up questions each (early-stop on dedup)
  - Per-question RAG-first / web-fallback / query-refinement loop
  - Web search via the open-websearch sidecar daemon (multi-engine, no API
    keys) plus four supplemental adapters: arxiv API, GitHub Code Search,
    ROCm docs sitemap, HN Algolia. Page content fetched via the AMD MCP
    `fetch` tool.
  - Each answer demands at least 4 distinct cited sources (or status=open)
  - A global ceiling of 200 LLM+web calls keeps a single DRA run bounded

To enable the open-websearch primary channel, start the sidecar once per
host. It speaks MCP-over-HTTP (NOT REST) at ``/mcp`` on port 3000:

    npm install -g open-websearch
    node $(npm root -g)/open-websearch/build/index.js   # MODE=http default

(Running via ``npx`` directly is unreliable in some environments because the
npx wrapper sometimes gets reaped immediately after spawning. Invoking the
``build/index.js`` entrypoint with ``node`` is the most stable path.)

If the daemon is not reachable the open-websearch adapter degrades to ``[]``
and the four supplemental adapters carry the load (no DRA failure).

Env vars:
  GEAK_DRA_DISABLE=1                   -> turn DRA off entirely
  GEAK_DRA_MODEL=claude-opus-4.6       -> model used for all DRA LLM calls
  GEAK_DRA_API_KEY=...                 -> optional override
  GEAK_DRA_MAX_QUESTIONS=50            -> max questions through full evidence (Stage 4)
  GEAK_DRA_MAX_BLINDSPOTS=10           -> max follow-up doubts PER blindspot round
  GEAK_DRA_MAX_BLINDSPOT_ROUNDS=3      -> how many rounds of Stage 5+6 to run
  GEAK_DRA_RETRIEVAL_TOP_K=12          -> RAG chunks per question retrieval call
  GEAK_DRA_MIN_SOURCES=4               -> required distinct sources per answer
  GEAK_DRA_MAX_TOTAL_CALLS=200         -> hard ceiling on LLM+web calls combined
  GEAK_DRA_WEB_DISABLE=1               -> skip web fallback (RAG only)
  GEAK_DRA_WEB_MAX_REFINEMENTS=2       -> per-question query refinement retries
  GEAK_DRA_WEB_CONCURRENCY=8           -> async parallelism for question batch
  GEAK_DRA_OPEN_WEBSEARCH_DISABLE=1    -> disable the open-websearch adapter
  GEAK_DRA_OPEN_WEBSEARCH_URL=...      -> override MCP endpoint (default http://localhost:3000/mcp)
  GEAK_DRA_RUN_EXPERIMENTAL=1          -> also produce experimental_directions.{md,json}
  GEAK_DRA_INDEX_PATH=...              -> override path to RAG index (otherwise default)
  GEAK_DRA_USE_PRIOR_RUNS=1            -> include cross_session prior runs in evidence
  GEAK_DRA_GITHUB_TOKEN=...            -> auth GitHub Code Search (10/min unauth -> 30/min auth)
  GEAK_DRA_MCP_FETCH_URL=...           -> override AMD MCP fetch endpoint
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


_DEFAULT_MCP_FETCH_URL = "https://mcp-platform.amd.com/mcp/commons_remote/"
# open-websearch speaks MCP-over-HTTP at /mcp (Streamable HTTP transport),
# not REST. The default points at the MCP endpoint of the local sidecar.
_DEFAULT_OPEN_WEBSEARCH_URL = "http://localhost:3000/mcp"


@dataclass
class DRAConfig:
    enabled: bool = True
    model_name: str = "claude-opus-4.6"
    api_key: str | None = None

    # Question / blindspot budgets
    max_questions: int = 50
    max_blindspots: int = 10  # per round
    max_blindspot_rounds: int = 3
    retrieval_top_k: int = 12

    # Quality / grounding
    min_sources_per_answer: int = 4

    # Hard safety ceiling on combined LLM + web calls per DRA invocation.
    # Conservatively above expected usage (~85 LLM + ~240 web under defaults
    # parallelised) so it does not bite during normal runs but still aborts
    # the runner gracefully if something loops.
    max_total_calls: int = 200

    # Web fallback
    web_search_enabled: bool = True
    web_max_refinements: int = 2
    web_concurrency: int = 8
    github_token: str | None = None
    mcp_fetch_url: str = _DEFAULT_MCP_FETCH_URL
    # open-websearch sidecar daemon: primary general-purpose SERP channel.
    # Adapter degrades gracefully if the daemon is unreachable.
    open_websearch_enabled: bool = True
    open_websearch_url: str = _DEFAULT_OPEN_WEBSEARCH_URL

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
            max_questions=_env_int("GEAK_DRA_MAX_QUESTIONS", 50),
            max_blindspots=_env_int("GEAK_DRA_MAX_BLINDSPOTS", 10),
            max_blindspot_rounds=_env_int("GEAK_DRA_MAX_BLINDSPOT_ROUNDS", 3),
            retrieval_top_k=_env_int("GEAK_DRA_RETRIEVAL_TOP_K", 12),
            min_sources_per_answer=_env_int("GEAK_DRA_MIN_SOURCES", 4),
            max_total_calls=_env_int("GEAK_DRA_MAX_TOTAL_CALLS", 200),
            web_search_enabled=not _env_bool("GEAK_DRA_WEB_DISABLE", False),
            web_max_refinements=_env_int("GEAK_DRA_WEB_MAX_REFINEMENTS", 2),
            web_concurrency=_env_int("GEAK_DRA_WEB_CONCURRENCY", 8),
            github_token=os.environ.get("GEAK_DRA_GITHUB_TOKEN", "").strip() or None,
            mcp_fetch_url=os.environ.get("GEAK_DRA_MCP_FETCH_URL", _DEFAULT_MCP_FETCH_URL).strip(),
            open_websearch_enabled=not _env_bool("GEAK_DRA_OPEN_WEBSEARCH_DISABLE", False),
            open_websearch_url=os.environ.get(
                "GEAK_DRA_OPEN_WEBSEARCH_URL", _DEFAULT_OPEN_WEBSEARCH_URL
            ).rstrip("/"),
            run_experimental=_env_bool("GEAK_DRA_RUN_EXPERIMENTAL", True),
            use_prior_runs=_env_bool("GEAK_DRA_USE_PRIOR_RUNS", False),
            index_path=os.environ.get("GEAK_DRA_INDEX_PATH", "").strip() or None,
        )
