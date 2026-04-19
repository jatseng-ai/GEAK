# DRA Design Notes

Design notes for a future GEAK "DRA" feature set built around two related but separate capabilities:

1. `deep_search`: convergent, evidence-backed research for task generation
2. `experimental_directions`: orthogonal, higher-variance ideas that deliberately challenge the dominant thesis

This document captures the ideas discussed during early brainstorming and is intentionally a planning artifact, not an implementation guide for a finished feature.

## Why This Exists

GEAK already has strong ingredients for a deeper research workflow:

- preprocess artifacts such as `profile.json`, `baseline_metrics.json`, `discovery.json`, `CODEBASE_CONTEXT.md`, and `COMMANDMENT.md`
- a local knowledge base under `knowledge-base/`
- heterogeneous task generation that can already consume a `deep_search_path`
- prior-run artifacts and round evaluations that can be turned into evidence

At the same time, GEAK's current retrieval path is closer to single-shot RAG plus summarization than a full multi-pass research loop. DRA is meant to close that gap without overcomplicating the first version.

## Principles

- Build for GEAK, not for a generic consumer research product.
- Prefer repo-local and run-local evidence over web evidence.
- Keep the first version read-only and artifact-producing.
- Optimize for better task generation, not for a flashy report surface.
- Keep convergent and divergent analysis separate so task generation can use them differently.
- Do not copy the product UX of ChatGPT, Gemini, or Claude just because their user-facing surfaces look similar.

## Current GEAK Seams

The current codebase already exposes a useful insertion point for a future deep-research artifact:

- `src/minisweagent/agents/heterogeneous/task_generator.py`
  - accepts `deep_search_path`
  - forwards it into task generator template variables
- `src/minisweagent/agents/heterogeneous/prompts.py`
  - already references "Deep search findings" in the task-generator prompt
- `src/minisweagent/agents/heterogeneous/tools.py`
  - does not yet automatically pass a deep-search artifact from preprocess into task generation
- `src/minisweagent/mcp_integration/mcp_environment.py`
  - provides RAG-style query tools and a RAG filter sub-agent
  - is useful as an evidence source, but by itself is not a full deep-research loop
- `src/minisweagent/memory/integration.py`
  - currently leaves cross-session memory as a stub
  - creates a natural future home for durable research outcomes

## Problem Statement

GEAK's task generator should not rely only on:

- raw profiling files
- baseline metrics
- the kernel file
- a local optimization knowledge base

It should also have access to:

- synthesized research findings grounded in the current repo and run
- explicit blindspot analysis
- orthogonal ideas that are worth testing even when they are not the most likely winner

## Proposed DRA Split

### 1. `deep_search`

Purpose:

- converge on the best-supported hypotheses
- reduce wasted task slots
- bias the task generator toward high-value, kernel-relevant work

Output:

- `deep_search.md`
- `deep_search.json`

Behavior:

- evidence-backed
- explicit about what is well supported and what remains open
- optimized for exploitation and task prioritization

### 2. `experimental_directions`

Purpose:

- maintain exploration pressure
- avoid local optima
- surface ideas that deliberately challenge the dominant thesis

Output:

- `experimental_directions.md`
- `experimental_directions.json`

Behavior:

- intentionally orthogonal to `deep_search`
- may be more speculative, but should still be reasoned and falsifiable
- optimized for managed exploration rather than pure recommendation

These two artifacts should remain separate. Merging them would make the task generator treat "best current belief" and "useful contrarian probe" as the same kind of signal.

## Proposed Deep-Search Flow

The deep-search flow should be a small multi-pass pipeline of LLM calls plus retrieval, not necessarily a new always-on full agent runtime.

### Inputs

Primary inputs:

- kernel file
- `profile.json`
- `baseline_metrics.json`
- `discovery.json`
- `CODEBASE_CONTEXT.md`
- `COMMANDMENT.md`
- previous round task summaries
- previous round result summaries
- round-level evaluation artifacts
- local GEAK knowledge base

Optional inputs:

- web results
- external vendor docs
- uploaded notes or future private data sources

### Stage 0: Fact Extraction

Before asking broad questions, extract hard facts:

- kernel language and backend
- main bottleneck type
- hottest kernels or dependency paths
- benchmark contract and correctness constraints
- known prior successes and failures
- likely optimization targets in the dependency tree

This step creates a compact structured view of the run context.

### Stage 1: Question Generation

Generate many candidate questions from the facts, for example:

- Which file or dependency is the highest-value optimization target?
- Is the bottleneck label masking a larger structural issue?
- Are wrapper or layout transforms dominating end-to-end latency?
- Which strategy families best match the observed profile shape?
- What strategy families have already been tried and should be avoided?

The system can generate a large set of candidate questions, but it should not research all of them equally.

### Stage 2: Question Ranking

Rank candidate questions by:

- decision impact
- uncertainty
- actionability
- kernel-body relevance
- expected effect on task prioritization

Only the top questions proceed to full evidence gathering.

### Stage 3: First Evidence Pass

For each selected question, gather evidence in this order:

1. current run artifacts
2. repo files and codebase context
3. local GEAK knowledge base
4. prior GEAK run artifacts
5. external web or docs, only when needed

The goal is not "more searching"; it is better task-shaping evidence.

### Stage 4: Per-Question Synthesis

Each researched question should produce a structured answer containing:

- answer
- evidence
- affected files or functions
- task-generation implications
- recommended status: `prefer`, `deprioritize`, `reject`, or `open`

### Stage 5: Meta / Blindspot Analysis

Run a separate skeptical pass over the first-pass answers.

This pass should ask:

- What assumptions are weakly supported?
- Which strategy families are underexplored?
- Are we overcommitting to the bottleneck label?
- Are we overweighting wrapper changes?
- Which potentially important files or dependencies are missing?
- Which recommendations conflict with one another?

This is intentionally separate from first-pass synthesis so the system can critique its own current thesis.

### Stage 6: Second Targeted Evidence Pass

Use the blindspot report to trigger a second, narrower search round.

This pass should only chase unresolved or high-value doubts, not start another broad sweep.

### Stage 7: Final Artifact Synthesis

Produce:

- `deep_search.md` for human review and prompt consumption
- `deep_search.json` for structured downstream use

## `deep_search.md` Contents

Suggested sections:

- Executive summary
- Ranked hypotheses
- Highest-value target files and functions
- Prefer first
- Deprioritize or reject
- Blindspots still open

The Markdown output should be concise enough to remain useful to the task generator, not a long essay.

## `deep_search.json` Shape

Illustrative shape:

```json
{
  "inputs": {},
  "facts": {},
  "questions": [],
  "answers": [],
  "blindspots": [],
  "ranked_hypotheses": [],
  "taskgen_guidance": {
    "prefer_first": [],
    "deprioritize": [],
    "reject": [],
    "open_questions": []
  }
}
```

The most important section is `taskgen_guidance`, because that is what turns research into better task planning.

## Proposed Orthogonal / Experimental Flow

The experimental path should be its own pass after deep search, not a subsection mixed into the same artifact.

### Purpose

Generate ideas that are deliberately different along a meaningful axis, not random "crazy ideas."

Orthogonal should usually mean one of:

- a different strategy family
- a different target file or dependency
- a different backend choice
- a different optimization granularity
- a different assumption about what is truly limiting performance

### Examples of Orthogonal Directions

- Ignore micro-tuning and instead eliminate work with a different algorithmic decomposition.
- Stop optimizing the hottest kernel body and target an upstream layout transform that may dominate end-to-end latency.
- Rewrite a critical Triton path in HIP or CK when backend limitations are suspected.
- Replace one generic kernel with multiple shape-specialized variants plus minimal dispatch logic.

### Output Fields

Each direction should be short, testable, and structured. Suggested fields:

- `direction_id`
- `thesis`
- `why_orthogonal`
- `assumption_challenged`
- `strategy_family`
- `target_files_or_functions`
- `expected_upside`
- `implementation_cost`
- `kill_criteria`
- `notes_for_taskgen`

The two most important fields are:

- `assumption_challenged`
- `kill_criteria`

These keep the exploratory branch disciplined instead of random.

## Why Keep Orthogonal Analysis Separate

Adding orthogonal ideas directly into deep search has two bad outcomes:

- at worst, it creates conflict between recommended and intentionally contrarian advice
- at best, it creates redundancy that the task generator still has to untangle

Keeping separate artifacts preserves semantic clarity:

- `deep_search`: here is what we currently believe
- `experimental_directions`: here is what we should probe in case that belief is incomplete or wrong

## Task-Generator Integration

The task generator should consume both artifacts, but not in the same way.

### Primary Rule

Most tasks in a round should come from `deep_search`.

### Experimental Rule

A small number of tasks should come from `experimental_directions`, with explicit diversity relative to the primary batch.

Suggested prompt-level behavior:

- generate most tasks from `deep_search`
- generate at most one or a small number of tasks from `experimental_directions`
- experimental tasks in the same round should differ in strategy family from the primary tasks
- do not repeat failed experimental directions unless the underlying thesis changed

## Proposed Task Metadata Extension

A single scalar `priority` is not enough if GEAK wants both exploitation and exploration.

Suggested second axis:

- `mode = primary | experimental`

Alternative names:

- `track = exploit | explore`

This separates:

- how strongly GEAK believes in an idea
- whether the idea exists to exploit current evidence or to explore a deliberate alternative

Without this second axis, experimental ideas either:

- get ranked low and never run
- or get mixed into the main batch and pollute the exploitation path

## Scheduling Ideas

Illustrative scheduler policy:

- `>= 4 GPUs`: reserve one slot per round for `experimental`
- `2-3 GPUs`: reserve one experimental slot after round 1, or earlier if the search space looks narrow
- `1 GPU`: trigger experimental work after stagnation, or make every Nth round exploration-heavy

Adaptive rule:

- if verified improvement is happening, keep experimental share small
- if verified improvement stalls, increase experimental share

This gives GEAK a controlled exploitation versus exploration balance.

## Naming Notes

Working artifact names:

- `deep_search.md`
- `deep_search.json`
- `experimental_directions.md`
- `experimental_directions.json`

Possible future alias:

- `orthogonal_analysis`

For now, `experimental_directions` is the clearest name for the divergent artifact.

## MVP Recommendation

The first implementation should stay narrow:

1. add a preprocess-stage research runner
2. write `deep_search.md` and `deep_search.json`
3. add a second orthogonal-analysis pass
4. write `experimental_directions.md` and `experimental_directions.json`
5. plumb both artifacts into task generation
6. update task generation to bias toward primary ideas while preserving a small experimental lane

This avoids introducing a brand-new always-on runtime abstraction too early.

## Explicit Non-Goals for MVP

- recreating the UX of ChatGPT Deep Research, Gemini Deep Research, or Claude Research
- building a general-purpose consumer research product inside GEAK
- mixing convergent and orthogonal outputs into one report
- making web search the default primary source of truth
- creating a flashy long-form report that is hard for task generation to use

## Future Extensions

- refresh deep research after each round evaluation
- persist outcomes into future memory integration
- use round-level verified outcomes to reward good research hypotheses
- add optional private or MCP-backed evidence sources
- add a risk monitor when combining public web evidence with sensitive private data

## Open Questions

- Should preprocess generate both artifacts by default, or only when a flag is enabled?
- Should orthogonal directions always be generated, or only after the main thesis is well established?
- Should task generation reserve an experimental slot every round, or only when progress stalls?
- Should the final task schema include explicit exploration metadata from day one?
- How much of the final `deep_search` artifact should be optimized for prompt readability versus machine readability?

## Summary

The working design is:

- keep deep research convergent and evidence-backed
- keep orthogonal analysis separate and explicitly exploratory
- feed both into task generation with different semantics
- use repo-local and run-local evidence first
- start with a read-only artifact pipeline before investing in a more complex agent runtime

That split gives GEAK a cleaner path to both stronger exploitation and healthier exploration.
