#!/usr/bin/env python3
"""CI gate: only `src/minisweagent/cli.py` may define a Typer app.

Part of PR-1 (Foundation + Cleanup) per docs/refactor/EXECUTION_PLAN.md §7 Principle #8.

Runs as WARN-only until `cli.py` lands (still scheduled in PR-1), then FAIL-strict.
Today (pre-PR-1 cli.py): this script should emit WARN for the current `run/mini.py:app`
and the 9 other `:main` Typer entries — documenting that they exist and will be
consolidated.

After PR-1: this gate FAILS if anyone reintroduces a Typer app outside cli.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src" / "minisweagent"
CANONICAL_CLI = SRC_DIR / "cli.py"

# Patterns that declare a Typer app at module level
TYPER_DECL = re.compile(r"^\s*(?:\w+\s*=\s*)?typer\.Typer\s*\(", re.MULTILINE)
APP_COMMAND = re.compile(r"^\s*@app\.command\s*\(", re.MULTILINE)

# Files we allow to have Typer decls (the canonical one, once it lands)
ALLOWED = {CANONICAL_CLI.relative_to(REPO_ROOT)}

# Files that legitimately have a Typer app TODAY but will be consolidated or
# deleted in PR-1. These are WARN, not FAIL, until cli.py replaces them.
# Initial list determined by running this script pre-PR-1 (see commit message).
TRANSITIONAL = {
    # Primary & secondary CLIs — content moves into cli.py
    Path("src/minisweagent/run/mini.py"),            # → cli.py
    Path("src/minisweagent/run/orchestrator.py"),    # → `geak resume` subcommand in cli.py
    Path("src/minisweagent/run/mini_extra.py"),      # meta-dispatcher, deleted
    Path("src/minisweagent/run/inspector.py"),       # trajectory TUI, deleted
    Path("src/minisweagent/run/github_issue.py"),    # GitHub import, deleted
    # mini-swe-agent heritage — deleted wholesale in PR-1 heritage cleanup
    Path("src/minisweagent/run/extra/swebench.py"),
    Path("src/minisweagent/run/extra/swebench_single.py"),
    Path("src/minisweagent/run/extra/config.py"),
    # Debug-only Typer inside tools module; real-infrastructure tool, not a CLI entry
    Path("src/minisweagent/tools/strategy_manager.py"),
}


def main() -> int:
    violators: list[tuple[Path, str]] = []

    for py in SRC_DIR.rglob("*.py"):
        rel = py.relative_to(REPO_ROOT)
        if rel in ALLOWED:
            continue
        try:
            text = py.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        m = TYPER_DECL.search(text) or APP_COMMAND.search(text)
        if m:
            violators.append((rel, m.group(0).strip()))

    transitional_hits = [v for v in violators if v[0] in TRANSITIONAL]
    real_violations = [v for v in violators if v[0] not in TRANSITIONAL]

    if transitional_hits:
        print(f"[WARN] {len(transitional_hits)} transitional Typer entries (to be "
              f"consolidated into cli.py in PR-1):")
        for p, snippet in transitional_hits:
            print(f"  {p} :: {snippet}")

    if real_violations:
        print(f"[FAIL] {len(real_violations)} Typer apps outside the canonical "
              f"cli.py (and not on the transitional allowlist):")
        for p, snippet in real_violations:
            print(f"  {p} :: {snippet}")
        return 1

    if not transitional_hits and not real_violations:
        print("[OK] No Typer apps outside src/minisweagent/cli.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
