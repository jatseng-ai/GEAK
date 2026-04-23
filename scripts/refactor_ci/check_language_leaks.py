#!/usr/bin/env python3
"""CI gate: no literal `kernel_type == "triton"|"hip"` outside `kernel_languages/`.

Part of PR-1 (Foundation + Cleanup) per docs/refactor/EXECUTION_PLAN.md §9.1.

Semantic-site audit (see INVARIANTS.md + CODEBASE_AUDIT.md §9 for the 16 sites):
This script catches the narrow pattern (literal equality comparisons). The broader
semantic sites (dict dispatch, prompt templates, _LANGUAGE_GUIDANCE maps) are
tracked separately by inspection during PR-2 and PR-3.

Runs as WARN-only today; becomes FAIL-strict after PR-2 lands (language detection
and phase-based preprocess should eliminate these sites).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src" / "minisweagent"
KERNEL_LANGUAGES_DIR = SRC_DIR / "kernel_languages"

# The literal-equality patterns we forbid in core code.
LEAK_PATTERNS = [
    re.compile(r'kernel_type\s*==\s*["\'](?:triton|hip)["\']'),
    re.compile(r'kernel_language\s*==\s*["\'](?:triton|hip)["\']'),
    re.compile(r'["\'](?:triton|hip)["\']\s*==\s*kernel_type'),
    re.compile(r'["\'](?:triton|hip)["\']\s*==\s*kernel_language'),
]


def is_in_kernel_languages(path: Path) -> bool:
    try:
        path.relative_to(KERNEL_LANGUAGES_DIR)
        return True
    except ValueError:
        return False


def main() -> int:
    violations: list[tuple[Path, int, str]] = []

    for py in SRC_DIR.rglob("*.py"):
        if is_in_kernel_languages(py):
            continue
        try:
            text = py.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            for pat in LEAK_PATTERNS:
                if pat.search(line):
                    violations.append((py.relative_to(REPO_ROOT), i, line.strip()))

    if not violations:
        print("[OK] No literal kernel_type==\"triton\"|\"hip\" leaks outside kernel_languages/")
        return 0

    # Pre-PR-2: WARN only (these will be fixed as part of PR-2/PR-3 refactor).
    # Post-PR-2: flip to FAIL. Controlled by env var.
    import os as _os
    strict = _os.environ.get("GEAK_LANG_LEAK_STRICT", "0") == "1"
    level = "FAIL" if strict else "WARN"

    print(f"[{level}] {len(violations)} literal language-equality leaks "
          f"(will be FAIL after PR-2 if strict=1):")
    for p, lineno, line in violations:
        print(f"  {p}:{lineno}  {line}")

    return 1 if strict else 0


if __name__ == "__main__":
    sys.exit(main())
