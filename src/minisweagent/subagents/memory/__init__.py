"""Memory subagents — per-round runtime subagents for cross-session memory.

Populated by PR-3:
- CrossSessionMemoryAnalysisAgent — reads top-k KB entries + current kernel,
  writes structured `cross_session_memory_insights.md` (gated by
  GEAK_USE_CROSS_SESSION_MEMORY).
"""
