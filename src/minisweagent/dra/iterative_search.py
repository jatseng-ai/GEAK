"""Per-question iterative search loop used by DRA Stage 3-4 / Stage 6.

For one research question we want a small set of high-signal "chunks" to feed
to the synthesizer. The shape of the loop is:

  1. Try the local RAG (HybridRetriever over our 339-doc KB).
  2. If RAG returns >= ``min_quality`` hits, return them and stop.
  3. Otherwise, fan out to public-API web adapters (arxiv, GitHub, HN,
     ROCm docs) via ``WebSearchSource`` and ``mcp_fetch.fetch_markdown`` to
     pull each top URL's content as markdown.
  4. Merge RAG hits + web hits, dedup by URL/title, keep top K.
  5. If the merged set is still weak (< ``min_quality`` items) and we have
     refinement budget left, ask the model to rewrite the query and recurse
     once. Track ``refinement_history`` so we never repeat a query.

Returns a uniform list of ``UnifiedHit`` (KB chunks and web pages share a
schema) plus the list of queries that were tried.

Bounded by:
  - ``per_question_call_budget`` web fetches per question (default 8)
  - ``max_refinements`` LLM rewrites per question (default 2)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from minisweagent.dra.evidence import EvidenceSource, SearchHit
from minisweagent.dra.mcp_fetch import McpFetchClient
from minisweagent.dra.web_search import WebHit, WebSearchSource

logger = logging.getLogger(__name__)


@dataclass
class UnifiedHit:
    """A chunk of evidence feeding the synthesizer, KB or web alike.

    ``origin`` is the canonical taxonomy used by ``Answer.evidence``:
        kb | web_search | web_arxiv | web_github | web_rocm_docs | web_hn | facts | prior_run
    """

    title: str
    content: str
    score: float
    origin: str
    url: str = ""
    chunk_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kb(cls, hit: SearchHit) -> UnifiedHit:
        return cls(
            title=hit.title,
            content=hit.content,
            score=hit.score,
            origin="kb",
            chunk_id=f"{hit.layer}/{hit.title}",
            extra={"layer": hit.layer, "category": hit.category, "retriever_source": hit.source},
        )

    @classmethod
    def from_web(cls, hit: WebHit, content: str) -> UnifiedHit:
        return cls(
            title=hit.title,
            content=content,
            score=hit.score,
            origin=hit.origin,
            url=hit.url,
            extra={"snippet": hit.snippet, **hit.extra},
        )


@dataclass
class SearchResult:
    """Output of ``iterative_search`` for one question."""

    hits: list[UnifiedHit]
    tried_queries: list[str]
    web_fetch_count: int = 0
    refinement_count: int = 0


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


async def iterative_search(
    question: str,
    evidence: EvidenceSource,
    web: WebSearchSource | None,
    fetch_client: McpFetchClient | None,
    *,
    top_k: int = 12,
    min_quality: int = 4,
    max_refinements: int = 2,
    per_question_fetch_budget: int = 8,
    refine_query_fn: Callable[[str, list[str], list[UnifiedHit]], Awaitable[str | None]] | None = None,
    on_call: Callable[[str], None] | None = None,
) -> SearchResult:
    """Run the RAG -> web -> refine loop for one question.

    Args:
      question:                The research question to find evidence for.
      evidence:                EvidenceSource for the local RAG path.
      web:                     WebSearchSource (None disables web fallback entirely).
      fetch_client:            McpFetchClient used to convert URLs -> markdown.
      top_k:                   Final number of merged hits to return.
      min_quality:             If RAG alone returns >= this many hits, skip web.
                               If the merged total is < this and refinements remain,
                               trigger a query rewrite.
      max_refinements:         Cap on LLM-driven query rewrites (NOT counting the
                               original query).
      per_question_fetch_budget: Cap on URLs fetched per question, regardless of
                               how many adapters returned hits.
      refine_query_fn:         Async callable that, given the question and the
                               history of tried queries plus the current weak
                               hits, returns a refined query (or None to stop).
                               Defaults to no-op if not provided.
      on_call:                 Optional sink called once per network call with
                               a tag like "rag" / "web_search" / "fetch" /
                               "refine". The runner uses this to tally calls
                               against the global ``max_total_calls`` ceiling.

    Returns SearchResult with merged hits and the queries actually issued.
    """
    tried: list[str] = []
    web_fetches = 0
    refinements = 0

    current_query = question.strip()
    accumulated: dict[str, UnifiedHit] = {}  # dedupe key: chunk_id or url

    for attempt in range(max_refinements + 1):
        tried.append(current_query)

        # ---- 1. RAG ----
        try:
            if on_call:
                on_call("rag")
            kb_hits = await asyncio.to_thread(evidence.search, current_query, top_k)
        except Exception as exc:
            logger.debug("[DRA] RAG search failed for %r: %s", current_query, exc)
            kb_hits = []

        for h in kb_hits:
            uh = UnifiedHit.from_kb(h)
            key = uh.chunk_id or uh.title
            if key not in accumulated:
                accumulated[key] = uh

        # If RAG already gave us enough, short-circuit.
        if len(accumulated) >= min_quality and len(kb_hits) >= min_quality:
            break

        # ---- 2. Web fallback ----
        if web is not None and fetch_client is not None and web_fetches < per_question_fetch_budget:
            try:
                if on_call:
                    on_call("web_search")
                web_hits = await web.search(current_query)
            except Exception as exc:
                logger.debug("[DRA] web search failed for %r: %s", current_query, exc)
                web_hits = []

            # Fetch each top URL we have not seen, up to the per-question budget.
            remaining = per_question_fetch_budget - web_fetches
            to_fetch = [h for h in web_hits if h.url not in accumulated][:remaining]

            async def _fetch_one(h: WebHit) -> UnifiedHit | None:
                if on_call:
                    on_call("fetch")
                content = await fetch_client.fetch_markdown(h.url, max_length=4000)
                if not content:
                    return None
                return UnifiedHit.from_web(h, content)

            results = await asyncio.gather(*(_fetch_one(h) for h in to_fetch), return_exceptions=True)
            web_fetches += len(to_fetch)
            for r in results:
                if isinstance(r, Exception) or r is None:
                    continue
                key = r.url or r.title
                if key not in accumulated:
                    accumulated[key] = r

        # ---- 3. Quality gate ----
        if len(accumulated) >= min_quality:
            break

        # ---- 4. Refinement ----
        if attempt >= max_refinements:
            break
        if refine_query_fn is None:
            break
        if on_call:
            on_call("refine")
        try:
            refined = await refine_query_fn(question, list(tried), list(accumulated.values()))
        except Exception as exc:
            logger.debug("[DRA] refine_query_fn failed: %s", exc)
            refined = None
        if not refined or refined.strip() in {q.strip() for q in tried}:
            break
        current_query = refined.strip()
        refinements += 1

    # Order: kb first by score desc, then web hits by score desc -- mixing
    # signals so the synthesizer sees both.
    merged = list(accumulated.values())
    merged.sort(key=lambda h: (h.origin != "kb", -h.score))
    return SearchResult(
        hits=merged[:top_k],
        tried_queries=tried,
        web_fetch_count=web_fetches,
        refinement_count=refinements,
    )
