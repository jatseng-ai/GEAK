"""Evidence layer for DRA.

This is the *only* module in `dra/` that knows about the on-disk shape of
preprocess artifacts (profile.json, baseline_metrics.json, ...) and about
the RAG `HybridRetriever`. Everything else in the package goes through the
`EvidenceSource` to fetch facts and search the knowledge base.

Design intent:
  - Stage 3 ("First Evidence Pass") and Stage 6 ("Second Targeted Evidence
    Pass") both call `EvidenceSource.search` with a question string.
  - In-process retrieval (no MCP round-trip) so a many-question loop is
    cheap.
  - Reading run artifacts is centralized so the runner cannot accidentally
    couple itself to filesystem layout.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inputs container
# ---------------------------------------------------------------------------


@dataclass
class DRAInputs:
    """Filesystem inputs to the DRA pipeline.

    Only `kernel_path` is strictly required; every other path is optional.
    Missing paths are silently skipped at read time.
    """

    kernel_path: Path
    output_dir: Path

    profile_path: Path | None = None
    baseline_metrics_path: Path | None = None
    discovery_path: Path | None = None
    codebase_context_path: Path | None = None
    commandment_path: Path | None = None
    knowledge_base_path: Path | None = None

    previous_results_dir: Path | None = None
    previous_tasks_dir: Path | None = None
    round_evaluations: list[dict[str, Any]] = field(default_factory=list)
    current_round: int = 1

    def to_dict_for_artifact(self) -> dict[str, Any]:
        """A small dict suitable for embedding in the final artifact's `inputs`."""
        return {
            "kernel_path": str(self.kernel_path),
            "profile_path": str(self.profile_path) if self.profile_path else None,
            "baseline_metrics_path": (
                str(self.baseline_metrics_path) if self.baseline_metrics_path else None
            ),
            "discovery_path": str(self.discovery_path) if self.discovery_path else None,
            "codebase_context_path": (
                str(self.codebase_context_path) if self.codebase_context_path else None
            ),
            "commandment_path": str(self.commandment_path) if self.commandment_path else None,
            "knowledge_base_path": (
                str(self.knowledge_base_path) if self.knowledge_base_path else None
            ),
            "current_round": self.current_round,
            "has_previous_results": bool(
                self.previous_results_dir and Path(self.previous_results_dir).is_dir()
            ),
            "has_previous_tasks": bool(
                self.previous_tasks_dir and Path(self.previous_tasks_dir).is_dir()
            ),
            "num_round_evaluations": len(self.round_evaluations),
        }


# ---------------------------------------------------------------------------
# Evidence source
# ---------------------------------------------------------------------------


# Cap each text payload so we never ship multi-MB blobs into a prompt context.
_MAX_TEXT_PAYLOAD = 200_000


def _read_text_capped(path: Path, max_chars: int = _MAX_TEXT_PAYLOAD) -> str:
    """Read a text file, truncating to `max_chars` with a sentinel marker."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + f"\n\n... [truncated, original {len(raw)} chars] ..."


def _read_json_safely(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse JSON at %s: %s", path, exc)
        return None


@dataclass
class SearchHit:
    """A normalized search result from `EvidenceSource.search`."""

    title: str
    content: str
    score: float
    source: str  # "embedding" | "bm25" | "embedding+bm25"
    layer: str = "unknown"
    category: str = "unknown"
    origin: str = "kb"  # "kb" | "prior_run" | "repo" (extensible)

    def to_evidence_ref(self) -> str:
        """Compact citation string suitable for the `evidence` field of an Answer."""
        return f"{self.origin}://{self.title} (score={self.score:.3f})"


class EvidenceSource:
    """Centralized read-side adapter for DRA.

    Responsibilities:
      - Read preprocess artifacts (kernel, profile, baseline, discovery, ...).
      - Issue search calls to the local `HybridRetriever` (in-process).
      - Optionally surface prior-run experiences from cross_session memory.
    """

    def __init__(
        self,
        inputs: DRAInputs,
        retrieval_top_k: int = 8,
        index_path: str | None = None,
        use_prior_runs: bool = False,
    ):
        self.inputs = inputs
        self.retrieval_top_k = retrieval_top_k
        self.index_path = index_path
        self.use_prior_runs = use_prior_runs
        self._retriever = None  # lazy

    # ------------------------------------------------------------------
    # Artifact readers
    # ------------------------------------------------------------------

    def read_kernel(self) -> str:
        """The kernel source file, capped."""
        p = self.inputs.kernel_path
        if not p or not p.exists():
            logger.warning("Kernel file not found at %s", p)
            return ""
        return _read_text_capped(p)

    def read_profile(self) -> dict[str, Any] | None:
        p = self.inputs.profile_path
        if not p or not p.exists():
            return None
        data = _read_json_safely(p)
        return data if isinstance(data, dict) else None

    def read_baseline_metrics(self) -> dict[str, Any] | None:
        p = self.inputs.baseline_metrics_path
        if not p or not p.exists():
            return None
        data = _read_json_safely(p)
        return data if isinstance(data, dict) else None

    def read_discovery(self) -> dict[str, Any] | None:
        p = self.inputs.discovery_path
        if not p or not p.exists():
            return None
        data = _read_json_safely(p)
        return data if isinstance(data, dict) else None

    def read_codebase_context(self) -> str:
        p = self.inputs.codebase_context_path
        if not p or not p.exists():
            return ""
        return _read_text_capped(p)

    def read_commandment(self) -> str:
        p = self.inputs.commandment_path
        if not p or not p.exists():
            return ""
        return _read_text_capped(p)

    def summarize_previous_results(self) -> str:
        """Best-effort summary of prior round outputs, if available."""
        d = self.inputs.previous_results_dir
        if not d or not Path(d).is_dir():
            return ""
        lines = [f"## Previous results in {d}"]
        for child in sorted(Path(d).iterdir()):
            if child.is_file():
                lines.append(f"- {child.name} ({child.stat().st_size} bytes)")
        return "\n".join(lines)

    def round_evaluations_summary(self) -> str:
        """Compact view of the orchestrator's per-round evaluations."""
        if not self.inputs.round_evaluations:
            return ""
        chunks: list[str] = []
        for rev in self.inputs.round_evaluations:
            r = rev.get("round", "?")
            best_task = rev.get("best_task", "N/A")
            fb = rev.get("full_benchmark") or {}
            speedup = (
                fb.get("verified_speedup", "N/A")
                if isinstance(fb, dict) and fb
                else rev.get("benchmark_speedup", "N/A")
            )
            chunks.append(f"Round {r}: best_task={best_task}, speedup={speedup}x")
        return "\n".join(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @property
    def retriever(self):
        """Lazy-load the HybridRetriever, in-process (no MCP)."""
        if self._retriever is None:
            try:
                from rag_mcp.retrieval import DEFAULT_INDEX_PATH, HybridRetriever
            except ImportError as exc:
                raise RuntimeError(
                    "rag-mcp package not installed. "
                    "Install with: pip install -e mcp_tools/rag-mcp"
                ) from exc
            index = Path(self.index_path) if self.index_path else DEFAULT_INDEX_PATH
            if not index.exists():
                raise RuntimeError(
                    f"RAG index not found at {index}. "
                    "Build it with: python scripts/build_index.py --force"
                )
            self._retriever = HybridRetriever(index_path=index)
        return self._retriever

    def search(self, query: str, k: int | None = None) -> list[SearchHit]:
        """Hybrid search over the local knowledge base.

        Returns a list of normalized `SearchHit`s rather than the retriever's
        raw tuple format so callers in `runner.py` stay decoupled.
        """
        top_k = k if k is not None else self.retrieval_top_k
        try:
            raw = self.retriever.search(query, k=top_k)
        except Exception as exc:  # retrieval failures should not crash DRA
            logger.warning("RAG search failed for %r: %s", query, exc)
            return []

        hits: list[SearchHit] = []
        for doc, score, source, _orig_score in raw:
            md = getattr(doc, "metadata", {}) or {}
            title = md.get("section") or md.get("title") or md.get("source", "Unknown")
            hits.append(
                SearchHit(
                    title=str(title)[:120],
                    content=getattr(doc, "page_content", ""),
                    score=float(score),
                    source=source,
                    layer=str(md.get("layer", "unknown")),
                    category=str(md.get("category", "unknown")),
                    origin="kb",
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Optional: cross-session prior runs
    # ------------------------------------------------------------------

    def prior_run_context(self) -> str:
        """Pull a small block of prior-run context if cross_session is enabled."""
        if not self.use_prior_runs:
            return ""
        try:
            from minisweagent.memory.integration import assemble_memory_context  # type: ignore
        except ImportError:
            return ""

        try:
            bm = self.read_baseline_metrics() or {}
            ctx = assemble_memory_context(
                kernel_path=str(self.inputs.kernel_path),
                bottleneck_type=bm.get("bottleneck"),
                profiling_metrics=bm,
            )
        except Exception as exc:
            logger.warning("prior_run_context failed: %s", exc)
            return ""
        return ctx or ""
