"""Tests for iterative_search: RAG-first, web-fallback, query-refinement."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from minisweagent.dra.evidence import SearchHit
from minisweagent.dra.iterative_search import iterative_search
from minisweagent.dra.web_search import WebHit


@dataclass
class _StubEvidence:
    rag_hits: list[SearchHit]
    calls: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.calls = []

    def search(self, q: str, k: int) -> list[SearchHit]:
        self.calls.append(q)
        return self.rag_hits


@dataclass
class _StubWeb:
    web_hits_by_query: dict[str, list[WebHit]]
    calls: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.calls = []

    async def search(self, q: str) -> list[WebHit]:
        self.calls.append(q)
        return self.web_hits_by_query.get(q, [])


@dataclass
class _StubFetch:
    body_by_url: dict[str, str]
    calls: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.calls = []

    async def fetch_markdown(self, url: str, max_length: int = 5000) -> str | None:
        self.calls.append(url)
        return self.body_by_url.get(url)


def _kb_hit(title: str) -> SearchHit:
    return SearchHit(title=title, content=f"body of {title}", score=0.5, source="bm25")


def test_rag_alone_satisfies_min_quality_skips_web():
    ev = _StubEvidence(rag_hits=[_kb_hit(f"hit{i}") for i in range(5)])
    web = _StubWeb({})  # would explode if called, but won't be
    fetch = _StubFetch({})
    sr = asyncio.run(iterative_search("q", ev, web, fetch, top_k=10, min_quality=4, max_refinements=2))  # type: ignore[arg-type]
    assert len(sr.hits) == 5
    assert all(h.origin == "kb" for h in sr.hits)
    assert web.calls == []  # web not invoked
    assert fetch.calls == []
    assert sr.tried_queries == ["q"]


def test_web_fallback_fires_when_rag_insufficient():
    ev = _StubEvidence(rag_hits=[_kb_hit("only-one")])
    web_hits = [
        WebHit(url=f"http://x/{i}", title=f"webhit{i}", origin="web_arxiv") for i in range(4)
    ]
    web = _StubWeb({"q": web_hits})
    fetch = _StubFetch({h.url: f"<content for {h.url}>" for h in web_hits})
    sr = asyncio.run(iterative_search("q", ev, web, fetch, top_k=10, min_quality=4, max_refinements=2))  # type: ignore[arg-type]
    assert web.calls == ["q"]
    assert len(fetch.calls) == 4
    # KB hit + 4 web hits (ordered KB first)
    origins = [h.origin for h in sr.hits]
    assert origins.count("kb") == 1
    assert origins.count("web_arxiv") == 4


def test_refinement_runs_when_first_attempt_weak():
    ev = _StubEvidence(rag_hits=[])
    # first query returns 1 hit (still < min_quality=4); refined query returns enough
    web = _StubWeb({
        "q": [WebHit(url="u1", title="weak", origin="web_arxiv")],
        "refined-q": [
            WebHit(url=f"u{i+10}", title=f"good{i}", origin="web_github") for i in range(4)
        ],
    })
    fetch = _StubFetch({u: f"body-{u}" for u in ["u1", "u10", "u11", "u12", "u13"]})

    refine_calls: list[tuple[str, list[str]]] = []

    async def refine(q, tried, weak_hits):
        refine_calls.append((q, list(tried)))
        return "refined-q"

    sr = asyncio.run(
        iterative_search(  # type: ignore[arg-type]
            "q",
            ev,
            web,
            fetch,
            top_k=10,
            min_quality=4,
            max_refinements=2,
            refine_query_fn=refine,
        )
    )
    assert refine_calls and refine_calls[0][1] == ["q"]
    assert sr.refinement_count == 1
    assert sr.tried_queries == ["q", "refined-q"]
    assert len(sr.hits) >= 4


def test_refinement_terminates_at_max_even_if_still_weak():
    ev = _StubEvidence(rag_hits=[])
    web = _StubWeb({})
    fetch = _StubFetch({})

    async def always_refine(q, tried, weak):
        return f"refine-{len(tried)}"

    sr = asyncio.run(
        iterative_search(  # type: ignore[arg-type]
            "q", ev, web, fetch, top_k=10, min_quality=4, max_refinements=2, refine_query_fn=always_refine
        )
    )
    # Initial + 2 refinements = 3 tried queries
    assert sr.tried_queries == ["q", "refine-1", "refine-2"]
    assert sr.refinement_count == 2


def test_on_call_callback_fires_for_each_network_step():
    ev = _StubEvidence(rag_hits=[])
    web = _StubWeb({"q": [WebHit(url="u1", title="t", origin="web_hn")]})
    fetch = _StubFetch({"u1": "body"})
    counts: dict[str, int] = {}

    def sink(kind: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1

    asyncio.run(
        iterative_search(  # type: ignore[arg-type]
            "q", ev, web, fetch, top_k=10, min_quality=4, max_refinements=0, on_call=sink
        )
    )
    assert counts.get("rag") == 1
    assert counts.get("web_search") == 1
    assert counts.get("fetch") == 1


def test_no_web_returns_only_rag():
    ev = _StubEvidence(rag_hits=[_kb_hit("only")])
    sr = asyncio.run(
        iterative_search(  # type: ignore[arg-type]
            "q", ev, web=None, fetch_client=None, top_k=10, min_quality=4, max_refinements=2
        )
    )
    assert [h.title for h in sr.hits] == ["only"]
    assert sr.web_fetch_count == 0
