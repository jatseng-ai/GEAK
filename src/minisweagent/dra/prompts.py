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
          - the retrieved knowledge-base chunks
          - the prior-run context (if any)

        Do not introduce facts that are not supported by the evidence. If the
        evidence is insufficient, say so and set status to "open".

        Pick a recommended status that the task generator can act on:
          - "prefer":       strong evidence; prioritize tasks in this direction
          - "deprioritize": weak or contraindicated; deprioritize tasks here
          - "reject":       evidence shows this should NOT be tried again
          - "open":         evidence inconclusive; needs more investigation

        ## Question
        {question}

        ## Facts
        ```json
        {facts_json}
        ```

        ## Retrieved knowledge-base chunks
        {kb_chunks}

        ## Prior-run context (may be empty)
        {prior_run_context}

        ## Required JSON schema
        {{
          "answer": "<the synthesized answer>",
          "evidence": ["<short citation strings, e.g. 'kb://...' or 'profile.json:hot_kernels[0]'>"],
          "affected": ["<files or functions this answer points at>"],
          "taskgen_implications": "<one-line guidance for task generation>",
          "status": "<prefer|deprioritize|reject|open>"
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
        You are running a skeptical pass over the first-pass answers below.
        Your job is to find weaknesses in the current thesis, NOT to agree.

        Look for:
          - assumptions that are weakly supported
          - strategy families that are underexplored
          - over-commitment to the bottleneck label
          - over-weighting of wrapper / layout changes
          - important files or dependencies that are missing
          - recommendations that conflict with each other

        Each blindspot must include a follow-up question that, if answered with
        the existing evidence sources, would resolve the doubt.

        Return AT MOST {max_blindspots} blindspots, ranked by importance.

        ## Facts
        ```json
        {facts_json}
        ```

        ## First-pass answers
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
