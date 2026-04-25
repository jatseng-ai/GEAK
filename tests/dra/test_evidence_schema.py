"""Schema round-trip tests for the new DRA structured citation shape."""
from __future__ import annotations

import json

from minisweagent.dra.schemas import (
    Answer,
    BlindSpot,
    DeepSearchArtifact,
    EvidenceCite,
    Facts,
    Question,
    TaskgenGuidance,
)


def test_evidence_cite_round_trip():
    e = EvidenceCite(
        source_type="web_arxiv",
        title="Sample paper",
        url="https://arxiv.org/abs/1234.5678",
        snippet="abs",
        score=0.42,
    )
    d = e.to_dict()
    assert d["source_type"] == "web_arxiv"
    assert d["url"].startswith("https://arxiv.org/")
    # Must be JSON-serializable
    assert json.loads(json.dumps(d)) == d


def test_answer_with_structured_evidence():
    a = Answer(
        question="why is knn slow?",
        answer="memory pressure on heap",
        evidence=[
            EvidenceCite(source_type="kb", title="HIP heap", chunk_id="amd/heap"),
            EvidenceCite(source_type="web_github", title="rocPRIM/knn.cu", url="https://github.com/AMDResearch/rocPRIM/..."),
            EvidenceCite(source_type="web_arxiv", title="GPU sort", url="https://arxiv.org/abs/2102.00001"),
            EvidenceCite(source_type="web_rocm_docs", title="kernel_language.html", url="https://rocm.docs.amd.com/.../kernel_language.html"),
        ],
        affected=["src/knn_kernel.hip"],
        taskgen_implications="Investigate LDS-backed heap.",
        status="prefer",
        source_stage="first_pass",
        refinement_history=["original q", "refined q"],
    )
    d = a.to_dict()
    # Round-trips cleanly
    text = json.dumps(d)
    parsed = json.loads(text)
    assert parsed["status"] == "prefer"
    assert len(parsed["evidence"]) == 4
    assert {e["source_type"] for e in parsed["evidence"]} == {
        "kb",
        "web_github",
        "web_arxiv",
        "web_rocm_docs",
    }
    assert parsed["refinement_history"][-1] == "refined q"


def test_blindspot_carries_round():
    b = BlindSpot(description="x", why_it_matters="y", follow_up_question="z?", round=2)
    assert b.to_dict()["round"] == 2


def test_deep_search_artifact_serializes_with_evidence_cites():
    artifact = DeepSearchArtifact(
        inputs={"kernel_path": "/k.py"},
        facts=Facts(bottleneck_type="compute"),
        questions=[Question(question="q1", rank_score=10.0)],
        answers=[
            Answer(
                question="q1",
                answer="a",
                evidence=[
                    EvidenceCite(source_type="kb", title="t", chunk_id="x"),
                ],
                status="open",
                source_stage="first_pass",
            )
        ],
        blindspots=[BlindSpot(description="bs", round=1)],
        ranked_hypotheses=["H1"],
        taskgen_guidance=TaskgenGuidance(prefer_first=["do x"]),
    )
    s = artifact.to_json()
    parsed = json.loads(s)
    assert parsed["answers"][0]["evidence"][0]["source_type"] == "kb"
    assert parsed["blindspots"][0]["round"] == 1
