"""DRA runner: the Stage 0-7 loop.

This is the only file in `dra/` that the rest of the codebase calls. The
public entrypoint is `run_dra(...)`, which:

  1. Stage 0: extracts structured facts from inputs
  2. Stages 1+2: generates and ranks research questions (one LLM call)
  3. Stage 3: gathers evidence per top-ranked question
  4. Stage 4: synthesizes a structured answer per question
  5. Stage 5: runs a skeptical critique to surface blindspots
  6. Stage 6: targeted second evidence pass on the blindspots' follow-ups
  7. Stage 7: composes deep_search.{md,json}
  (optional) Experimental: produces experimental_directions.{md,json}

All inter-stage state is structured (see `schemas.py`). All filesystem and
retrieval access goes through `evidence.py`. All prompts live in
`prompts.py`. This file is the orchestration only.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from minisweagent.dra import prompts
from minisweagent.dra.config import DRAConfig
from minisweagent.dra.evidence import DRAInputs, EvidenceSource, SearchHit
from minisweagent.dra.schemas import (
    Answer,
    BlindSpot,
    DeepSearchArtifact,
    ExperimentalDirection,
    ExperimentalDirectionsArtifact,
    Facts,
    Question,
    TaskgenGuidance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_dra(
    inputs: DRAInputs,
    config: DRAConfig | None = None,
    model: Any | None = None,
) -> dict[str, Path]:
    """Run the full DRA pipeline and write artifacts to disk.

    Args:
        inputs:  Filesystem inputs (kernel, profile, baseline, ...).
        config:  DRAConfig; defaults to `DRAConfig.from_env()`.
        model:   An object with `.query(messages) -> {"content": str}`. If
                 omitted, an `AmdLlmModel` is constructed from the config.

    Returns:
        Dict with paths to the written artifacts. Always includes
        `deep_search_md` and `deep_search_json`. Includes
        `experimental_md` and `experimental_json` if experimental was run.

    Raises:
        RuntimeError if DRA is disabled in config.
    """
    cfg = config or DRAConfig.from_env()
    if not cfg.enabled:
        raise RuntimeError("DRA is disabled (GEAK_DRA_DISABLE=1)")

    inputs.output_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        from minisweagent.models.amd_llm import AmdLlmModel

        model = AmdLlmModel(model_name=cfg.model_name, api_key=cfg.api_key)
        if hasattr(model, "_impl") and hasattr(model._impl, "tools"):
            model._impl.tools = []

    evidence = EvidenceSource(
        inputs=inputs,
        retrieval_top_k=cfg.retrieval_top_k,
        index_path=cfg.index_path,
        use_prior_runs=cfg.use_prior_runs,
    )

    logger.info("[DRA] Stage 0: extracting facts")
    facts = _stage0_extract_facts(model, evidence)

    logger.info("[DRA] Stages 1+2: generating and ranking questions (max=%d)", cfg.max_questions)
    questions = _stage12_generate_and_rank_questions(model, facts, cfg.max_questions)
    logger.info("[DRA] Selected %d ranked questions", len(questions))

    logger.info("[DRA] Stages 3+4: per-question evidence + synthesis")
    first_pass_answers = _stage34_first_evidence_pass(
        model, evidence, facts, questions
    )

    logger.info("[DRA] Stage 5: blindspot analysis (max=%d)", cfg.max_blindspots)
    blindspots = _stage5_blindspot(
        model, facts, first_pass_answers, cfg.max_blindspots
    )
    logger.info("[DRA] Found %d blindspots", len(blindspots))

    logger.info("[DRA] Stage 6: targeted second-pass evidence")
    second_pass_answers = _stage6_second_pass(model, evidence, facts, blindspots)

    all_answers = first_pass_answers + second_pass_answers

    logger.info("[DRA] Stage 7: final synthesis")
    artifact = _stage7_final_synthesis(
        model=model,
        inputs=inputs,
        facts=facts,
        questions=questions,
        answers=all_answers,
        blindspots=blindspots,
    )

    written = _write_deep_search(artifact, inputs.output_dir)
    logger.info("[DRA] Wrote %s and %s", written["deep_search_md"], written["deep_search_json"])

    if cfg.run_experimental:
        logger.info("[DRA] Running experimental_directions pass")
        ed_artifact = _experimental_pass(model, inputs, facts, artifact)
        ed_written = _write_experimental(ed_artifact, inputs.output_dir)
        written.update(ed_written)
        logger.info("[DRA] Wrote %s and %s", ed_written["experimental_md"], ed_written["experimental_json"])

    return written


# ---------------------------------------------------------------------------
# Stage 0: Fact extraction
# ---------------------------------------------------------------------------


def _stage0_extract_facts(model: Any, evidence: EvidenceSource) -> Facts:
    profile = evidence.read_profile() or {}
    baseline = evidence.read_baseline_metrics() or {}
    discovery = evidence.read_discovery() or {}

    prompt = prompts.FACTS_PROMPT.format(
        kernel_text=evidence.read_kernel(),
        profile_json=_json(profile),
        baseline_json=_json(baseline),
        discovery_json=_json(discovery),
        codebase_context=evidence.read_codebase_context() or "(empty)",
        commandment=evidence.read_commandment() or "(empty)",
        previous_results=evidence.summarize_previous_results() or "(empty)",
        round_evaluations=evidence.round_evaluations_summary() or "(empty)",
    )
    payload = _llm_json(model, prompt, fallback={})
    return _facts_from_payload(payload)


# ---------------------------------------------------------------------------
# Stages 1+2: Question generation and ranking
# ---------------------------------------------------------------------------


def _stage12_generate_and_rank_questions(
    model: Any, facts: Facts, max_questions: int
) -> list[Question]:
    prompt = prompts.QUESTIONS_PROMPT.format(
        facts_json=_json(facts.to_dict()),
        max_questions=max_questions,
    )
    payload = _llm_json(model, prompt, fallback={"questions": []})

    raw = payload.get("questions") or []
    questions: list[Question] = []
    for q in raw:
        if not isinstance(q, dict) or "question" not in q:
            continue
        questions.append(
            Question(
                question=str(q.get("question", "")).strip(),
                rationale=str(q.get("rationale", "")).strip(),
                decision_impact=int(q.get("decision_impact", 0) or 0),
                actionability=int(q.get("actionability", 0) or 0),
                kernel_relevance=int(q.get("kernel_relevance", 0) or 0),
                rank_score=float(q.get("rank_score", 0.0) or 0.0),
            )
        )
    questions = [q for q in questions if q.question]
    questions.sort(key=lambda q: q.rank_score, reverse=True)
    return questions[:max_questions]


# ---------------------------------------------------------------------------
# Stages 3+4: First-pass evidence + per-question synthesis
# ---------------------------------------------------------------------------


def _stage34_first_evidence_pass(
    model: Any,
    evidence: EvidenceSource,
    facts: Facts,
    questions: list[Question],
) -> list[Answer]:
    answers: list[Answer] = []
    prior_run_ctx = evidence.prior_run_context()
    for q in questions:
        hits = evidence.search(q.question)
        ans = _synthesize_one(
            model=model,
            facts=facts,
            question=q.question,
            hits=hits,
            prior_run_ctx=prior_run_ctx,
            source_stage="first_pass",
        )
        answers.append(ans)
    return answers


# ---------------------------------------------------------------------------
# Stage 5: Blindspot critique
# ---------------------------------------------------------------------------


def _stage5_blindspot(
    model: Any, facts: Facts, answers: list[Answer], max_blindspots: int
) -> list[BlindSpot]:
    prompt = prompts.BLINDSPOT_PROMPT.format(
        facts_json=_json(facts.to_dict()),
        answers_json=_json([a.to_dict() for a in answers]),
        max_blindspots=max_blindspots,
    )
    payload = _llm_json(model, prompt, fallback={"blindspots": []})
    raw = payload.get("blindspots") or []
    out: list[BlindSpot] = []
    for b in raw[:max_blindspots]:
        if not isinstance(b, dict):
            continue
        desc = str(b.get("description", "")).strip()
        if not desc:
            continue
        out.append(
            BlindSpot(
                description=desc,
                why_it_matters=str(b.get("why_it_matters", "")).strip(),
                follow_up_question=str(b.get("follow_up_question", "")).strip(),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Stage 6: Targeted second pass
# ---------------------------------------------------------------------------


def _stage6_second_pass(
    model: Any,
    evidence: EvidenceSource,
    facts: Facts,
    blindspots: list[BlindSpot],
) -> list[Answer]:
    answers: list[Answer] = []
    prior_run_ctx = evidence.prior_run_context()
    for b in blindspots:
        if not b.follow_up_question:
            continue
        hits = evidence.search(b.follow_up_question)
        ans = _synthesize_one(
            model=model,
            facts=facts,
            question=b.follow_up_question,
            hits=hits,
            prior_run_ctx=prior_run_ctx,
            source_stage="second_pass",
        )
        answers.append(ans)
    return answers


# ---------------------------------------------------------------------------
# Stage 7: Final synthesis + Markdown rendering
# ---------------------------------------------------------------------------


def _stage7_final_synthesis(
    *,
    model: Any,
    inputs: DRAInputs,
    facts: Facts,
    questions: list[Question],
    answers: list[Answer],
    blindspots: list[BlindSpot],
) -> DeepSearchArtifact:
    prompt = prompts.FINAL_SYNTH_PROMPT.format(
        facts_json=_json(facts.to_dict()),
        answers_json=_json([a.to_dict() for a in answers]),
        blindspots_json=_json([b.to_dict() for b in blindspots]),
    )
    payload = _llm_json(model, prompt, fallback={})

    guidance_raw = payload.get("taskgen_guidance") or {}
    guidance = TaskgenGuidance(
        prefer_first=_str_list(guidance_raw.get("prefer_first")),
        deprioritize=_str_list(guidance_raw.get("deprioritize")),
        reject=_str_list(guidance_raw.get("reject")),
        open_questions=_str_list(guidance_raw.get("open_questions")),
    )

    artifact = DeepSearchArtifact(
        inputs=inputs.to_dict_for_artifact(),
        facts=facts,
        questions=questions,
        answers=answers,
        blindspots=blindspots,
        ranked_hypotheses=_str_list(payload.get("ranked_hypotheses")),
        taskgen_guidance=guidance,
    )

    md = payload.get("executive_summary_md")
    if isinstance(md, str) and md.strip():
        artifact_executive_md = md.strip()
    else:
        artifact_executive_md = _fallback_executive_summary(artifact)
    artifact._executive_md = artifact_executive_md  # type: ignore[attr-defined]
    return artifact


def _write_deep_search(artifact: DeepSearchArtifact, out_dir: Path) -> dict[str, Path]:
    json_path = out_dir / "deep_search.json"
    md_path = out_dir / "deep_search.md"
    json_path.write_text(artifact.to_json(), encoding="utf-8")
    md_path.write_text(_render_deep_search_md(artifact), encoding="utf-8")
    return {"deep_search_md": md_path, "deep_search_json": json_path}


def _render_deep_search_md(artifact: DeepSearchArtifact) -> str:
    lines: list[str] = []
    exec_md = getattr(artifact, "_executive_md", "") or _fallback_executive_summary(artifact)
    lines.append("# Deep Search\n")
    lines.append(f"_Generated: {artifact.timestamp}_\n")
    lines.append("## Executive Summary\n")
    lines.append(exec_md.rstrip() + "\n")

    lines.append("## Ranked Hypotheses\n")
    for i, h in enumerate(artifact.ranked_hypotheses, 1):
        lines.append(f"{i}. {h}")
    lines.append("")

    g = artifact.taskgen_guidance
    lines.append("## Task-Generator Guidance\n")
    lines.append("### Prefer first")
    for x in g.prefer_first:
        lines.append(f"- {x}")
    lines.append("\n### Deprioritize")
    for x in g.deprioritize:
        lines.append(f"- {x}")
    lines.append("\n### Reject")
    for x in g.reject:
        lines.append(f"- {x}")
    lines.append("\n### Open questions")
    for x in g.open_questions:
        lines.append(f"- {x}")
    lines.append("")

    if artifact.blindspots:
        lines.append("## Blindspots Still Open\n")
        for b in artifact.blindspots:
            lines.append(f"- **{b.description}** — {b.why_it_matters}")
        lines.append("")

    lines.append("## Per-Question Answers\n")
    for a in artifact.answers:
        lines.append(f"### Q ({a.source_stage}, status={a.status}): {a.question}")
        lines.append("")
        lines.append(a.answer)
        if a.affected:
            lines.append(f"\n**Affected:** {', '.join(a.affected)}")
        if a.evidence:
            lines.append(f"\n**Evidence:** {', '.join(a.evidence[:8])}")
        if a.taskgen_implications:
            lines.append(f"\n**Task-gen implication:** {a.taskgen_implications}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _fallback_executive_summary(artifact: DeepSearchArtifact) -> str:
    """Used when the model fails to provide `executive_summary_md`."""
    g = artifact.taskgen_guidance
    bits: list[str] = []
    if artifact.facts.bottleneck_type:
        bits.append(f"Bottleneck: **{artifact.facts.bottleneck_type}**.")
    if g.prefer_first:
        bits.append(f"Top recommendation: {g.prefer_first[0]}.")
    if not bits:
        bits.append("No executive summary produced.")
    return " ".join(bits)


# ---------------------------------------------------------------------------
# Experimental directions pass
# ---------------------------------------------------------------------------


def _experimental_pass(
    model: Any,
    inputs: DRAInputs,
    facts: Facts,
    deep_search: DeepSearchArtifact,
) -> ExperimentalDirectionsArtifact:
    summary = {
        "ranked_hypotheses": deep_search.ranked_hypotheses,
        "taskgen_guidance": deep_search.taskgen_guidance.to_dict(),
    }
    prompt = prompts.EXPERIMENTAL_PROMPT.format(
        facts_json=_json(facts.to_dict()),
        deep_search_summary_json=_json(summary),
    )
    payload = _llm_json(model, prompt, fallback={"directions": []})

    directions: list[ExperimentalDirection] = []
    for d in payload.get("directions") or []:
        if not isinstance(d, dict):
            continue
        thesis = str(d.get("thesis", "")).strip()
        if not thesis:
            continue
        directions.append(
            ExperimentalDirection(
                direction_id=str(d.get("direction_id") or _short_id()),
                thesis=thesis,
                why_orthogonal=str(d.get("why_orthogonal", "")).strip(),
                assumption_challenged=str(d.get("assumption_challenged", "")).strip(),
                strategy_family=str(d.get("strategy_family", "")).strip(),
                target_files_or_functions=_str_list(d.get("target_files_or_functions")),
                expected_upside=str(d.get("expected_upside", "")).strip(),
                implementation_cost=str(d.get("implementation_cost", "")).strip(),
                kill_criteria=str(d.get("kill_criteria", "")).strip(),
                notes_for_taskgen=str(d.get("notes_for_taskgen", "")).strip(),
            )
        )

    return ExperimentalDirectionsArtifact(
        inputs=inputs.to_dict_for_artifact(),
        directions=directions,
    )


def _write_experimental(
    artifact: ExperimentalDirectionsArtifact, out_dir: Path
) -> dict[str, Path]:
    json_path = out_dir / "experimental_directions.json"
    md_path = out_dir / "experimental_directions.md"
    json_path.write_text(artifact.to_json(), encoding="utf-8")
    md_path.write_text(_render_experimental_md(artifact), encoding="utf-8")
    return {"experimental_md": md_path, "experimental_json": json_path}


def _render_experimental_md(artifact: ExperimentalDirectionsArtifact) -> str:
    lines: list[str] = []
    lines.append("# Experimental Directions\n")
    lines.append(f"_Generated: {artifact.timestamp}_\n")
    if not artifact.directions:
        lines.append("_No directions produced._\n")
        return "\n".join(lines)

    for d in artifact.directions:
        lines.append(f"## {d.direction_id}: {d.thesis}\n")
        if d.why_orthogonal:
            lines.append(f"- **Why orthogonal:** {d.why_orthogonal}")
        if d.assumption_challenged:
            lines.append(f"- **Assumption challenged:** {d.assumption_challenged}")
        if d.strategy_family:
            lines.append(f"- **Strategy family:** {d.strategy_family}")
        if d.target_files_or_functions:
            lines.append(f"- **Targets:** {', '.join(d.target_files_or_functions)}")
        if d.expected_upside:
            lines.append(f"- **Expected upside:** {d.expected_upside}")
        if d.implementation_cost:
            lines.append(f"- **Cost:** {d.implementation_cost}")
        if d.kill_criteria:
            lines.append(f"- **Kill criteria:** {d.kill_criteria}")
        if d.notes_for_taskgen:
            lines.append(f"- **Notes for taskgen:** {d.notes_for_taskgen}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Per-question synthesis (shared between Stage 4 and Stage 6)
# ---------------------------------------------------------------------------


def _synthesize_one(
    *,
    model: Any,
    facts: Facts,
    question: str,
    hits: list[SearchHit],
    prior_run_ctx: str,
    source_stage: str,
) -> Answer:
    prompt = prompts.PER_QUESTION_SYNTH_PROMPT.format(
        question=question,
        facts_json=_json(facts.to_dict()),
        kb_chunks=_render_hits_for_prompt(hits) or "(no chunks retrieved)",
        prior_run_context=prior_run_ctx or "(none)",
    )
    payload = _llm_json(model, prompt, fallback={})
    status = payload.get("status", "open")
    if status not in ("prefer", "deprioritize", "reject", "open"):
        status = "open"

    return Answer(
        question=question,
        answer=str(payload.get("answer", "")).strip() or "(no answer produced)",
        evidence=_str_list(payload.get("evidence")) or [h.to_evidence_ref() for h in hits[:5]],
        affected=_str_list(payload.get("affected")),
        taskgen_implications=str(payload.get("taskgen_implications", "")).strip(),
        status=status,
        source_stage=source_stage,
    )


def _render_hits_for_prompt(hits: list[SearchHit], max_chunk_chars: int = 2000) -> str:
    """Render search hits compactly for inclusion in a synthesis prompt."""
    if not hits:
        return ""
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        body = h.content or ""
        if len(body) > max_chunk_chars:
            body = body[:max_chunk_chars] + " ...[truncated]"
        parts.append(
            f"### Chunk {i}: {h.title} (layer={h.layer}, source={h.source}, score={h.score:.3f})\n{body}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _llm_json(model: Any, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Issue a single LLM call expecting a JSON object back. Robust to noise.

    Returns the parsed object on success, or `fallback` on any failure.
    """
    try:
        response = model.query(
            [
                {"role": "system", "content": "You return valid JSON only."},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as exc:
        logger.warning("[DRA] LLM call failed: %s", exc)
        return dict(fallback)

    content = (response or {}).get("content", "") if isinstance(response, dict) else ""
    if not isinstance(content, str) or not content.strip():
        logger.warning("[DRA] LLM returned empty content")
        return dict(fallback)

    parsed = _extract_json_object(content)
    if parsed is None:
        logger.warning("[DRA] Could not extract JSON from LLM response (len=%d)", len(content))
        return dict(fallback)
    return parsed


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Try to pull a JSON object out of arbitrary model output."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    if text.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", text)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _facts_from_payload(payload: dict[str, Any]) -> Facts:
    return Facts(
        kernel_language=str(payload.get("kernel_language", "unknown")),
        kernel_backend=str(payload.get("kernel_backend", "unknown")),
        bottleneck_type=str(payload.get("bottleneck_type", "unknown")),
        hot_kernels=[h for h in (payload.get("hot_kernels") or []) if isinstance(h, dict)],
        benchmark_contract=str(payload.get("benchmark_contract", "")),
        correctness_constraints=_str_list(payload.get("correctness_constraints")),
        prior_successes=_str_list(payload.get("prior_successes")),
        prior_failures=_str_list(payload.get("prior_failures")),
        likely_targets=_str_list(payload.get("likely_targets")),
        notes=str(payload.get("notes", "")),
    )


def _str_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return []


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def _short_id() -> str:
    return uuid.uuid4().hex[:8]
