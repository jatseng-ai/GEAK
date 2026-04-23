"""Translation subagents — standalone multi-round subagent invoked by `geak translate`.

Populated by PR-2:
- TranslationLoop — `SubagentBase.loop()` pattern; rewrites a kernel from
  source language to target language, verified by golden-match tensor allclose.
  Uses max_attempts retry loop with feedback between attempts.
"""
