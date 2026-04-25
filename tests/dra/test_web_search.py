"""Unit tests for web search adapters.

We mock HTTP at the level of the helper functions
(``_requests_get_json`` / ``_requests_get_text``) and the
``McpFetchClient.get_or_create`` factory (for the open-websearch MCP
adapter) so the tests are fully hermetic and don't touch the network.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from minisweagent.dra import web_search

# ---- arxiv -----------------------------------------------------------------


_ARXIV_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2101.00001v1</id>
    <title>Fast K-NN on GPUs</title>
    <summary>A study of brute-force KNN on AMD CDNA.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2102.00002v2</id>
    <title>Wavefront-aware sorting</title>
    <summary>Bitonic top-K on wavefront-64 architectures.</summary>
  </entry>
</feed>
"""


def test_arxiv_adapter_parses_atom(monkeypatch):
    monkeypatch.setattr(web_search, "_requests_get_text", lambda url, headers=None: _ARXIV_SAMPLE_XML)
    hits = asyncio.run(web_search.search_arxiv("knn rocm", max_results=5))
    assert len(hits) == 2
    assert hits[0].url.startswith("http://arxiv.org/abs/")
    assert hits[0].origin == "web_arxiv"
    assert "Fast K-NN" in hits[0].title


def test_arxiv_adapter_handles_empty_text(monkeypatch):
    monkeypatch.setattr(web_search, "_requests_get_text", lambda url, headers=None: None)
    assert asyncio.run(web_search.search_arxiv("anything")) == []


# ---- GitHub ----------------------------------------------------------------


_GH_SAMPLE = {
    "items": [
        {
            "html_url": "https://github.com/foo/bar/blob/main/k.cu",
            "repository": {"full_name": "foo/bar"},
            "path": "k.cu",
            "score": 5.5,
        },
        {
            "html_url": "https://github.com/baz/qux/blob/main/sort.hip",
            "repository": {"full_name": "baz/qux"},
            "path": "sort.hip",
            "score": 3.1,
        },
    ]
}


def test_github_adapter_parses_items(monkeypatch):
    captured_headers = {}

    def fake_get_json(url, headers=None):
        captured_headers.update(headers or {})
        return _GH_SAMPLE

    monkeypatch.setattr(web_search, "_requests_get_json", fake_get_json)
    monkeypatch.delenv("GEAK_DRA_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    hits = asyncio.run(web_search.search_github("knn"))
    assert [h.origin for h in hits] == ["web_github", "web_github"]
    assert "foo/bar" in hits[0].title
    assert "Authorization" not in captured_headers  # unauth path


def test_github_adapter_uses_token_when_present(monkeypatch):
    captured_headers = {}

    def fake_get_json(url, headers=None):
        captured_headers.update(headers or {})
        return _GH_SAMPLE

    monkeypatch.setattr(web_search, "_requests_get_json", fake_get_json)
    monkeypatch.setenv("GEAK_DRA_GITHUB_TOKEN", "tok-xyz")
    asyncio.run(web_search.search_github("knn"))
    assert captured_headers.get("Authorization") == "Bearer tok-xyz"


# ---- HN --------------------------------------------------------------------


def test_hn_adapter_parses_hits(monkeypatch):
    monkeypatch.setattr(
        web_search,
        "_requests_get_json",
        lambda url, headers=None: {
            "hits": [
                {
                    "title": "Why GPU sorts are hard",
                    "url": "https://example.com/x",
                    "author": "alice",
                    "points": 200,
                    "num_comments": 50,
                }
            ]
        },
    )
    hits = asyncio.run(web_search.search_hn("gpu sort"))
    assert len(hits) == 1
    assert hits[0].origin == "web_hn"
    assert hits[0].url == "https://example.com/x"
    assert "200 points" in hits[0].snippet


# ---- ROCm docs sitemap -----------------------------------------------------


_ROCM_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html</loc></url>
  <url><loc>https://rocm.docs.amd.com/projects/rocPRIM/en/latest/reference/sort_helpers.html</loc></url>
  <url><loc>https://rocm.docs.amd.com/en/latest/about/release-notes.html</loc></url>
</urlset>
"""


def test_rocm_docs_scores_by_overlap(monkeypatch):
    web_search._ROCM_SITEMAP_CACHE = None  # reset cache between tests
    monkeypatch.setattr(web_search, "_requests_get_text", lambda url, headers=None: _ROCM_SITEMAP)
    hits = asyncio.run(web_search.search_rocm_docs("kernel language reference"))
    assert hits, "expected at least one ROCm docs hit"
    assert all(h.origin == "web_rocm_docs" for h in hits)
    assert "kernel_language" in hits[0].url


def test_rocm_docs_returns_empty_when_no_overlap(monkeypatch):
    web_search._ROCM_SITEMAP_CACHE = None
    monkeypatch.setattr(web_search, "_requests_get_text", lambda url, headers=None: _ROCM_SITEMAP)
    hits = asyncio.run(web_search.search_rocm_docs("totally unrelated zzzqqq xyzabc"))
    assert hits == []


# ---- WebSearchSource fans out + dedups ------------------------------------


def test_websearchsource_dedups_and_aggregates(monkeypatch):
    web_search._ROCM_SITEMAP_CACHE = []  # disable rocm docs

    async def fake_open(q, n, *, daemon_url=None, engines=None):
        return [web_search.WebHit(url="u_open", title="open1", origin="web_search", score=7.0)]

    async def fake_arxiv(q, n):
        return [web_search.WebHit(url="u1", title="a", origin="web_arxiv")]

    async def fake_github(q, n, token=None):
        return [
            web_search.WebHit(url="u1", title="duplicate", origin="web_github"),  # dup vs arxiv
            web_search.WebHit(url="u2", title="b", origin="web_github"),
        ]

    async def fake_hn(q, n):
        return [web_search.WebHit(url="u3", title="c", origin="web_hn")]

    monkeypatch.setattr(web_search, "search_open_websearch", fake_open)
    monkeypatch.setattr(web_search, "search_arxiv", fake_arxiv)
    monkeypatch.setattr(web_search, "search_github", fake_github)
    monkeypatch.setattr(web_search, "search_hn", fake_hn)

    src = web_search.WebSearchSource(enable_rocm_docs=False)
    hits = asyncio.run(src.search("anything"))
    # open-websearch comes first and wins URL collisions; then arxiv, github, hn.
    assert [h.url for h in hits] == ["u_open", "u1", "u2", "u3"]
    assert hits[0].origin == "web_search"


# ---- open-websearch adapter (MCP-over-HTTP) -------------------------------


def _make_text_content(text: str):
    """Mimic an MCP TextContent block: object with ``.text`` attribute."""

    class _TC:
        pass

    tc = _TC()
    tc.text = text
    return tc


def _make_call_result(text: str):
    """Mimic a fastmcp ``CallToolResult`` with the text payload in ``content``."""

    class _CR:
        pass

    cr = _CR()
    cr.content = [_make_text_content(text)]
    cr.structured_content = None
    cr.data = None
    return cr


def test_open_websearch_adapter_normalises_results(monkeypatch):
    """Stubs the MCP `search` tool and verifies WebHit shaping + scoring."""
    captured: dict = {}

    class _Stub:
        async def call_tool(self, tool_name, arguments):
            captured["tool"] = tool_name
            captured["args"] = arguments
            payload = json.dumps(
                {
                    "query": arguments["query"],
                    "engines": arguments["engines"],
                    "totalResults": 3,
                    "results": [
                        {
                            "title": "First",
                            "url": "https://a.example/1",
                            "description": "snippet 1",
                            "engine": "duckduckgo",
                        },
                        {
                            "title": "Second",
                            "url": "https://a.example/2",
                            "description": "snippet 2",
                            "engine": "brave",
                        },
                        {"title": "No URL", "description": "skipped"},
                    ],
                    "partialFailures": [],
                }
            )
            return _make_call_result(payload)

    monkeypatch.setattr(
        "minisweagent.dra.mcp_fetch.McpFetchClient.get_or_create",
        classmethod(lambda cls, server_url=None: _Stub()),
    )
    hits = asyncio.run(
        web_search.search_open_websearch(
            "ROCm wavefront",
            max_results=7,
            daemon_url="http://localhost:3000/mcp",
            engines=("duckduckgo", "brave"),
        )
    )
    assert [h.url for h in hits] == ["https://a.example/1", "https://a.example/2"]
    assert all(h.origin == "web_search" for h in hits)
    # Position-based score: rank 0 -> 7.0, rank 1 -> 6.0
    assert hits[0].score == 7.0
    assert hits[1].score == 6.0
    # Engine carried through extras for debugging
    assert hits[0].extra["engine"] == "duckduckgo"
    assert hits[1].extra["engine"] == "brave"
    # Tool called with the right shape
    assert captured["tool"] == "search"
    assert captured["args"]["query"] == "ROCm wavefront"
    assert captured["args"]["limit"] == 7
    assert captured["args"]["engines"] == ["duckduckgo", "brave"]


def test_open_websearch_adapter_handles_call_exception(monkeypatch):
    """When the MCP call_tool raises (daemon down, network error), return []."""

    class _BrokenStub:
        async def call_tool(self, tool_name, arguments):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(
        "minisweagent.dra.mcp_fetch.McpFetchClient.get_or_create",
        classmethod(lambda cls, server_url=None: _BrokenStub()),
    )
    assert asyncio.run(web_search.search_open_websearch("anything")) == []


def test_open_websearch_adapter_handles_empty_results(monkeypatch):
    """When the daemon returns the envelope with results: [], adapter returns []."""

    class _EmptyStub:
        async def call_tool(self, tool_name, arguments):
            payload = json.dumps(
                {
                    "query": arguments["query"],
                    "engines": arguments["engines"],
                    "totalResults": 0,
                    "results": [],
                    "partialFailures": [{"engine": "bing", "code": "engine_error"}],
                }
            )
            return _make_call_result(payload)

    monkeypatch.setattr(
        "minisweagent.dra.mcp_fetch.McpFetchClient.get_or_create",
        classmethod(lambda cls, server_url=None: _EmptyStub()),
    )
    assert asyncio.run(web_search.search_open_websearch("nothing matches")) == []


def test_open_websearch_adapter_handles_malformed_json(monkeypatch):
    """When the daemon returns garbage in the text block, return []."""

    class _GarbageStub:
        async def call_tool(self, tool_name, arguments):
            return _make_call_result("not valid json {{{")

    monkeypatch.setattr(
        "minisweagent.dra.mcp_fetch.McpFetchClient.get_or_create",
        classmethod(lambda cls, server_url=None: _GarbageStub()),
    )
    assert asyncio.run(web_search.search_open_websearch("anything")) == []


def test_open_websearch_adapter_skips_blank_query():
    assert asyncio.run(web_search.search_open_websearch("")) == []
    assert asyncio.run(web_search.search_open_websearch("   ")) == []


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_empty_queries_short_circuit(query):
    assert asyncio.run(web_search.search_arxiv(query)) == []
    assert asyncio.run(web_search.search_github(query)) == []
    assert asyncio.run(web_search.search_hn(query)) == []
    assert asyncio.run(web_search.search_rocm_docs(query)) == []
    assert asyncio.run(web_search.search_open_websearch(query)) == []
