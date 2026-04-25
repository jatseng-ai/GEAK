"""Prompt templates for the DRA pipeline.

All LLM call sites in `runner.py` source their prompts from this module so
they can be tuned and diffed in one place.

Each prompt asks the model to return a single JSON object on stdout. The
runner is responsible for extracting and validating that JSON. We keep the
prompts deliberately schema-explicit because the staged pipeline depends on
machine-readable handoffs between stages.
"""

from __future__ import annotations

import textwrap

# ---------------------------------------------------------------------------
# Shared header used at the top of every DRA prompt.
# ---------------------------------------------------------------------------

_HEADER = textwrap.dedent(
    """\
    You are a research sub-agent inside the GEAK GPU kernel optimization system.
    Your job is to produce structured, evidence-backed analysis that a downstream
    task generator will use to plan optimization tasks.

    You MUST respond with a single JSON object that exactly matches the requested
    schema. No prose before or after. No code fences. Just the JSON.
    """
)


# ---------------------------------------------------------------------------
# Stage 0: Fact extraction
# ---------------------------------------------------------------------------

FACTS_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Extract a compact, structured view of the run context from the inputs below.

        ## Inputs
        ### Kernel source
        ```
        {kernel_text}
        ```

        ### Profile (profile.json)
        ```json
        {profile_json}
        ```

        ### Baseline metrics (baseline_metrics.json)
        ```json
        {baseline_json}
        ```

        ### Discovery (discovery.json)
        ```json
        {discovery_json}
        ```

        ### Codebase context (CODEBASE_CONTEXT.md)
        {codebase_context}

        ### Commandment (COMMANDMENT.md)
        {commandment}

        ### Prior round summary (may be empty)
        {previous_results}

        ### Round evaluations (may be empty)
        {round_evaluations}

        ## Required JSON schema
        {{
          "kernel_language": "<triton|hip|cuda|unknown>",
          "kernel_backend":  "<rocm|cuda|unknown>",
          "bottleneck_type": "<memory|compute|latency|balanced|unknown>",
          "hot_kernels":     [{{"name": "...", "share": 0.0}}],
          "benchmark_contract":     "<one-line description>",
          "correctness_constraints": ["..."],
          "prior_successes":         ["..."],
          "prior_failures":          ["..."],
          "likely_targets":          ["<file or function>"],
          "notes":                   "<short free text>"
        }}
        """
    )
)


# ---------------------------------------------------------------------------
# Stages 1+2: Question generation AND ranking in one call.
# Keeping these in one call (instead of two) saves a round-trip without
# losing the ranking lever -- rank_score is produced in the same response.
# ---------------------------------------------------------------------------

QUESTIONS_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Given the extracted facts below, generate research questions whose answers
        would meaningfully shape what optimization tasks GEAK should try next.
        Then rank them.

        Generate between 6 and 20 candidate questions. Score each on three axes
        (0-10 integers):
          - decision_impact:   how much the answer changes downstream task choice
          - actionability:     how concrete the answer can be made
          - kernel_relevance:  how tightly tied to this specific kernel/profile

        Compute rank_score = decision_impact + actionability + kernel_relevance.

        Return AT MOST {max_questions} of the highest-ranked questions. Do not
        include questions whose answers are already obvious from the facts.

        ## Facts
        ```json
        {facts_json}
        ```

        ## Required JSON schema
        {{
          "questions": [
            {{
              "question": "...",
              "rationale": "...",
              "decision_impact": 0,
              "actionability": 0,
              "kernel_relevance": 0,
              "rank_score": 0.0
            }}
          ]
        }}
        """
    )
)


# ---------------------------------------------------------------------------
# Stage 4: Per-question synthesis
# ---------------------------------------------------------------------------

PER_QUESTION_SYNTH_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Answer the research question below using ONLY the provided evidence:
          - the extracted facts
          - the retrieved knowledge-base chunks (origin "kb")
          - the fetched web sources (origin "web_arxiv", "web_github",
            "web_rocm_docs", "web_hn") -- these have a real `url` you must cite
          - the prior-run context (if any)

        Do not introduce facts that are not supported by the evidence. If the
        evidence is insufficient (fewer than {min_sources} distinct sources
        meaningfully back your claims), say so and set status to "open".

        ## Citation requirements (load-bearing)
        Cite at least {min_sources} DISTINCT sources in `evidence` if the
        retrieved material allows it. Each entry must point at a real chunk or
        URL from the inputs below; do NOT invent URLs or chunk titles. Prefer
        a mix of `kb` chunks AND `web_*` sources if both are available --
        single-origin answers are weaker than mixed ones.

        Pick a recommended status that the task generator can act on:
          - "prefer":       strong evidence (>= {min_sources} distinct sources agree); prioritize tasks here
          - "deprioritize": weak or contraindicated by evidence
          - "reject":       evidence shows this should NOT be tried again
          - "open":         evidence inconclusive or fewer than {min_sources} sources available

        ## Question
        {question}

        ## Facts
        ```json
        {facts_json}
        ```

        ## Retrieved knowledge-base + web chunks
        {kb_chunks}

        ## Prior-run context (may be empty)
        {prior_run_context}

        ## Required JSON schema
        {{
          "answer": "<the synthesized answer; weave citations inline like [1] [2] referring to the chunks above>",
          "evidence": [
            {{
              "source_type": "<kb|web_arxiv|web_github|web_rocm_docs|web_hn|facts|prior_run>",
              "title": "<short title or section name; copy from the chunk header>",
              "url": "<empty for kb / facts; the real URL for web_* sources>",
              "chunk_id": "<empty for web; the chunk id or title for kb sources>",
              "snippet": "<<= 400 chars excerpted from the chunk>",
              "score": 0.0
            }}
          ],
          "affected": ["<files or functions this answer points at>"],
          "taskgen_implications": "<one-line guidance for task generation>",
          "status": "<prefer|deprioritize|reject|open>"
        }}
        """
    )
)


# ---------------------------------------------------------------------------
# Query refinement (used by iterative_search.py when first try is weak)
# ---------------------------------------------------------------------------

QUERY_REFINEMENT_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        A previous web/KB search returned weak or off-topic results for the
        question below. Rewrite the search query so Google + arxiv + GitHub +
        AMD ROCm docs return more on-target hits.

        ## Hard constraints on the rewrite
        - Output SHORT: 3-7 keywords total. Long queries return ZERO hits on
          AMD's web_search tool. Drop everything that is not a noun, code
          identifier, or a critical disambiguator.
        - NO sentences. NO question marks. NO "what is", "how does", etc.
        - Keep code identifiers verbatim (e.g. ``knn_kernel``, ``__shfl_xor``).
        - Keep proper nouns (AMD, ROCm, CDNA, HIP, MI300X) when relevant.
        - Keep dimension constants when relevant (warp, wavefront 64, k=8).

        Pick ONE strategy:
          - swap a vague term for a specific one (e.g. "kernel" -> "knn_kernel")
          - shift to a sibling formulation (cause-and-effect -> the named pattern)
          - drop the qualifier and search for the bare concept
          - target a specific source family (e.g. "arxiv knn gpu" or
            "github rocm point cloud knn")

        ## Original question
        {question}

        ## Queries already tried
        {tried_queries}

        ## Sample of weak results (titles only)
        {weak_titles}

        ## Required JSON schema
        {{
          "refined_query": "<3-7 keywords, no quotes, no question mark>"
        }}
        """
    )
)


# ---------------------------------------------------------------------------
# Stage 5: Blindspot / meta critique
# ---------------------------------------------------------------------------

BLINDSPOT_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        You are running a skeptical pass (round {round_idx} of up to
        {max_rounds}) over the per-question answers gathered so far. Your job
        is to find weaknesses in the current thesis, NOT to agree.

        Look for:
          - assumptions that are weakly supported by the cited sources
          - strategy families that are underexplored
          - over-commitment to the bottleneck label
          - over-weighting of wrapper / layout changes
          - important files or dependencies that are missing
          - recommendations that conflict with each other
          - claims with fewer than 4 distinct cited sources (`status: open`
            answers are prime targets)

        Each blindspot must include a follow-up question that, if researched
        against fresh sources, would resolve the doubt.

        ## Avoid repeating prior blindspots
        The blindspots below were already raised in earlier rounds. Do NOT
        emit blindspots that are semantically duplicates of these. If you have
        nothing genuinely new and important to raise, return an empty list.

        ```json
        {prior_blindspots_json}
        ```

        Return AT MOST {max_blindspots} NEW blindspots, ranked by importance.

        ## Facts
        ```json
        {facts_json}
        ```

        ## All answers so far (first pass + any prior rounds)
        ```json
        {answers_json}
        ```

        ## Required JSON schema
        {{
          "blindspots": [
            {{
              "description": "<what is weakly supported or missing>",
              "why_it_matters": "<why this could change the conclusion>",
              "follow_up_question": "<a single concrete question to research next>"
            }}
          ]
        }}
        """
    )
)


# ---------------------------------------------------------------------------
# Stage 7: Final artifact synthesis
# ---------------------------------------------------------------------------

FINAL_SYNTH_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Produce the final convergent research artifact from the per-question
        answers (both first-pass and second-pass) and the blindspots.

        Resolve contradictions across answers. Group affected files/functions.
        Produce ranked hypotheses (most to least supported) and a structured
        `taskgen_guidance` block that the task generator will act on directly.

        ## Facts
        ```json
        {facts_json}
        ```

        ## All answers (first + second pass)
        ```json
        {answers_json}
        ```

        ## Blindspots
        ```json
        {blindspots_json}
        ```

        ## Required JSON schema
        {{
          "ranked_hypotheses": [
            "<one-line hypothesis, most-supported first>"
          ],
          "taskgen_guidance": {{
            "prefer_first":   ["<concrete direction the task generator should try first>"],
            "deprioritize":   ["<direction to deprioritize>"],
            "reject":         ["<direction to NOT try>"],
            "open_questions": ["<unresolved doubts the task generator should keep in mind>"]
          }},
          "executive_summary_md": "<concise markdown summary suitable for a prompt; sections allowed>"
        }}
        """
    )
)


# ---------------------------------------------------------------------------
# Experimental directions (separate pass, runs after deep_search)
# ---------------------------------------------------------------------------

EXPERIMENTAL_PROMPT = (
    _HEADER
    + textwrap.dedent(
        """\

        ## Task
        Propose orthogonal, deliberately-different optimization directions that
        challenge the dominant thesis represented by `deep_search` below.

        Orthogonal here means at least one of:
          - a different strategy family
          - a different target file or dependency
          - a different backend choice
          - a different optimization granularity
          - a different assumption about what is truly limiting performance

        Each direction MUST include `assumption_challenged` and `kill_criteria`.
        These are the load-bearing fields that keep this branch disciplined
        instead of random.

        Return between 2 and 5 directions.

        ## Facts
        ```json
        {facts_json}
        ```

        ## Deep-search summary (the thesis to challenge)
        ```json
        {deep_search_summary_json}
        ```

        ## Required JSON schema
        {{
          "directions": [
            {{
              "direction_id":  "<short slug, e.g. ed-1>",
              "thesis":        "<one-line statement of the alternative direction>",
              "why_orthogonal": "<which axis above this differs on>",
              "assumption_challenged": "<the deep_search assumption this probes>",
              "strategy_family":       "<algorithmic|fusion|tuning|wrapper|backend-swap|...>",
              "target_files_or_functions": ["..."],
              "expected_upside":     "<short>",
              "implementation_cost": "<low|medium|high + one-line reason>",
              "kill_criteria":       "<observable signal that says abandon this direction>",
              "notes_for_taskgen":   "<short>"
            }}
          ]
        }}
        """
    )
)
