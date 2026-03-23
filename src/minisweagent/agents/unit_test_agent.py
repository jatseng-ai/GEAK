# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Unit test subagent.

This agent searches for (or creates) a fixed test harness for a kernel and
returns a TEST_COMMAND string plus COMMANDMENT-ready commands.  Discovery
results are formatted into an enriched context that includes kernel analysis,
language-specific guidance, and extracted test patterns so the agent can make
informed decisions without re-scanning the repo.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.config import load_agent_config
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig


@dataclass
class UnitTestAgentConfig(AgentConfig):
    """Config loaded from mini_unit_test_agent.yaml (or provided via kwargs)."""


class UnitTestAgent(DefaultAgent):
    """Agent that creates a fixed test harness and returns TEST_COMMAND."""

    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=UnitTestAgentConfig, **kwargs)


def _extract_test_command(text: str) -> str:
    match = re.search(r"TEST_COMMAND:\s*(.+)\s*$", text.strip(), re.MULTILINE)
    if not match:
        raise ValueError(f"UnitTestAgent did not return TEST_COMMAND. Output was:\n{text}")
    return match.group(1).strip()


# ---------------------------------------------------------------------------
# Language-specific testing guidance (keyed by kernel_type)
# ---------------------------------------------------------------------------

_LANGUAGE_GUIDANCE: dict[str, str] = {
    "triton": (
        "This is a Triton kernel (JIT-compiled Python). No build step needed.\n"
        "- Import the kernel via its Python package path (do NOT use importlib.util).\n"
        "- Use `torch.testing.assert_close` for correctness validation.\n"
        "- Use `triton.testing.do_bench` or `torch.cuda.Event` for benchmarking.\n"
        "- Set `PYTHONPATH` before the process starts if the package is not installed.\n"
        "- Use fixed random seed (`torch.manual_seed(42)`) and fixed tensor sizes."
    ),
    "hip": (
        "This is a HIP kernel (C++ compiled with hipcc).\n"
        "- A build step is REQUIRED before running tests.\n"
        "- Use the project's build system (CMake/Makefile) or compile with `hipcc` directly.\n"
        "- Use host-side validation (compare GPU output against CPU reference).\n"
        "- Use `hipEventElapsedTime` or `torch.cuda.Event` for benchmarking.\n"
        "- NEVER use `sys.path.insert(0, '/absolute/path/...')`. "
        "Rely on PYTHONPATH set by the COMMANDMENT SETUP section."
    ),
    "cuda": (
        "This is a CUDA kernel (C++ compiled with nvcc).\n"
        "- A build step is REQUIRED before running tests.\n"
        "- Use the project's build system (CMake/Makefile) or compile with `nvcc` directly.\n"
        "- Use host-side validation (compare GPU output against CPU reference).\n"
        "- Use `cudaEventElapsedTime` or `torch.cuda.Event` for benchmarking."
    ),
    "ck": (
        "This is a Composable Kernel (CK) kernel (C++ compiled with hipcc + CK includes).\n"
        "CK harnesses use a 5-file architecture:\n"
        "\n"
        "  1. baseline.cpp   — frozen ground truth, compiled to libbaseline.so\n"
        "  2. optimized.cpp  — LLM edits TUNING PARAMETERS, compiled to liboptimized.so\n"
        "  3. test_harness.py — Python script that loads both .so via ctypes\n"
        "  4. compile.py     — auto-detects GPU arch via PyTorch, calls make\n"
        "  5. Makefile        — hipcc -shared -fPIC -O3 build rules\n"
        "\n"
        "Both .cpp files expose a single `extern \"C\" float run_kernel(...)` function.\n"
        "The Python harness loads each .so with ctypes.CDLL, sets argtypes/restype to\n"
        "mirror the C ABI, and calls run_kernel with PyTorch tensor data_ptr() values.\n"
        "\n"
        "Key rules:\n"
        "- The harness is Python-based (test_harness.py), NOT a C++ binary.\n"
        "- Kernels are compiled as standalone .so files — no dependency on CK's build system.\n"
        "- The COMMANDMENT SETUP section should run: ./compile.py && echo 'Build OK'\n"
        "- CORRECTNESS: python test_harness.py libbaseline.so liboptimized.so --correctness\n"
        "- PROFILE:     python test_harness.py libbaseline.so liboptimized.so --profile\n"
        "- BENCHMARK:   python test_harness.py libbaseline.so liboptimized.so --benchmark\n"
        "- NEVER use sys.path.insert or importlib.util. Use ctypes.CDLL for kernel loading.\n"
        "- Use torch.testing.assert_close for correctness validation between baseline and optimized.\n"
        "- A CK harness recipe (if available) provides the exact file contents to generate."
    ),
    "asm": (
        "This is a precompiled HSACO assembly kernel.\n"
        "- The assembly binary CANNOT be modified or recompiled.\n"
        "- Test ONLY via the Python wrapper that loads and launches it.\n"
        "- Use `torch.testing.assert_close` for correctness against a torch reference.\n"
        "- Benchmark the wrapper launch, not the assembly directly."
    ),
    "unknown": (
        "Kernel type could not be determined automatically.\n"
        "- Inspect the source file to determine if it is Triton, HIP, CUDA, or CK.\n"
        "- Apply the appropriate testing strategy based on your analysis."
    ),
}

_logger = logging.getLogger(__name__)

_RECIPE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "knowledge" / "ck_harness_recipes"


def _load_ck_recipe(kernel_info) -> str:
    """Load the CK harness recipe matching the kernel's device type.

    Reads ``_index.yaml`` for the device-type-to-recipe mapping, scans the
    kernel source for known device type class names, and returns the matching
    recipe markdown content.  Falls back to ``default_recipe`` when no device
    type is matched.
    """
    index_path = _RECIPE_DIR / "_index.yaml"
    if not index_path.exists():
        _logger.debug("CK recipe index not found at %s", index_path)
        return ""

    try:
        with open(index_path) as f:
            index = yaml.safe_load(f)
    except Exception:
        _logger.warning("Failed to parse CK recipe index", exc_info=True)
        return ""

    device_type_map: dict[str, str] = index.get("device_type_map", {})
    default_recipe: str = index.get("default_recipe", "")

    recipe_file = default_recipe
    source_text = ""

    file_path = getattr(kernel_info, "file_path", None)
    if file_path:
        try:
            source_text = Path(file_path).read_text(errors="replace")
        except Exception:
            _logger.debug("Could not read kernel source at %s", file_path)

    if source_text:
        for device_type, recipe_name in device_type_map.items():
            if re.search(rf"\b{re.escape(device_type)}\b", source_text):
                recipe_file = recipe_name
                _logger.info("Matched CK device type '%s' -> recipe '%s'", device_type, recipe_name)
                break

    if not recipe_file:
        return ""

    recipe_path = _RECIPE_DIR / recipe_file
    if not recipe_path.exists():
        _logger.warning("CK recipe file not found: %s", recipe_path)
        return ""

    try:
        return recipe_path.read_text()
    except Exception:
        _logger.warning("Failed to read CK recipe %s", recipe_path, exc_info=True)
        return ""


def format_discovery_for_agent(result) -> str:
    """Format a ``DiscoveryResult`` into an enriched context string for the UTA.

    Includes kernel analysis, language-specific testing guidance, discovered
    tests/benchmarks with confidence scores, and extracted test patterns.

    Formats an already-available ``DiscoveryResult`` for agent consumption.
    """
    if result is None:
        return ""

    lines: list[str] = []

    # --- Kernel analysis ---
    if result.kernels:
        k = result.kernels[0]
        lines.append("## Kernel Analysis")
        lines.append(f"- **Name**: {k.kernel_name}")
        lines.append(f"- **Type**: {k.kernel_type}")
        lines.append(f"- **Language**: {k.kernel_language}")
        lines.append(f"- **File**: `{k.file_path}`")
        lines.append(f"- **Functions**: {', '.join(k.function_names) if k.function_names else 'N/A'}")
        if k.inner_kernel_path:
            lines.append(f"- **Inner kernel**: `{k.inner_kernel_path}` ({k.inner_kernel_language or 'unknown'})")
        if k.build_info:
            bi = k.build_info
            if bi.compiler:
                lines.append(f"- **Compiler**: {bi.compiler}")
            if bi.build_system:
                lines.append(f"- **Build system**: {bi.build_system}")
            if bi.pybind_module:
                lines.append(f"- **Pybind module**: {bi.pybind_module}")
        lines.append("")

        # Language-specific testing guidance
        guidance = _LANGUAGE_GUIDANCE.get(k.kernel_type, "")
        if guidance:
            lines.append("## Language-Specific Testing Guidance")
            lines.append(guidance)
            lines.append("")

        # CK harness recipe injection
        if k.kernel_type == "ck":
            recipe = _load_ck_recipe(k)
            if recipe:
                lines.append("## CK Harness Recipe")
                lines.append("")
                lines.append("Use the following recipe as a template to generate the 5-file harness.")
                lines.append("Adapt the C ABI, ctypes argtypes, and shape lists as needed for this kernel.")
                lines.append("")
                lines.append(recipe)
                lines.append("")

    # --- Discovered tests ---
    if result.tests:
        lines.append("## Discovered Test Files (ranked by confidence)")
        for i, t in enumerate(result.tests[:5], 1):
            conf_pct = min(int(t.confidence * 100), 100)
            lines.append(f"  {i}. `{t.file_path}` — {t.test_type}, {conf_pct}% confidence")
            lines.append(f"     Suggested command: `{t.command}`")
        lines.append("")

    # --- Extracted test patterns (from top-confidence tests) ---
    patterns_found = False
    for t in result.tests[:3]:
        p = getattr(t, "patterns", None)
        if p is None:
            continue
        if not patterns_found:
            lines.append("## Extracted Test Patterns (reuse these in your harness)")
            patterns_found = True
        lines.append(f"From `{t.file_path.name}`:")
        if p.tolerances:
            lines.append(f"  Tolerances: {', '.join(p.tolerances)}")
        if p.input_shapes:
            lines.append(f"  Input shapes: {', '.join(p.input_shapes)}")
        if p.dtypes:
            lines.append(f"  Dtypes: {', '.join(p.dtypes)}")
        if p.reference_impls:
            lines.append(f"  Reference implementations: {', '.join(p.reference_impls)}")
        if p.import_patterns:
            lines.append("  Import patterns:")
            for imp in p.import_patterns[:5]:
                lines.append(f"    `{imp}`")
    if patterns_found:
        lines.append("")

    # --- Discovered benchmarks ---
    if result.benchmarks:
        lines.append("## Discovered Benchmark Files (ranked by confidence)")
        for i, b in enumerate(result.benchmarks[:5], 1):
            conf_pct = min(int(b.confidence * 100), 100)
            lines.append(f"  {i}. `{b.file_path}` — {b.bench_type}, {conf_pct}% confidence")
            lines.append(f"     Suggested command: `{b.command}`")
        lines.append("")

    # --- Dependency graph summary ---
    if result.kernels and result.dependency_graphs:
        k = result.kernels[0]
        dep_graph = result.dependency_graphs.get(k.kernel_name)
        if dep_graph:
            lines.append("## Dependency Graph")
            lines.append(dep_graph.summary())
            lines.append("")

    if not result.tests and not result.benchmarks:
        lines.append("No existing tests or benchmarks were found by the automated scan.")
        lines.append("You will need to create them from scratch.")
        lines.append("")

    return "\n".join(lines)



def run_unit_test_agent(
    *,
    model: Model,
    repo: Path,
    kernel_name: str,
    log_dir: Path | None = None,
    discovery_context: str = "",
) -> str:
    """Run UnitTestAgent in ``repo`` and return the extracted test command string.

    If *discovery_context* is provided (e.g. from :func:`format_discovery_for_agent`),
    it is appended to the task prompt so the agent starts with pre-scanned results
    instead of exploring from scratch.
    """
    agent_config, _ = load_agent_config("mini_unit_test_agent")

    env = LocalEnvironment(**LocalEnvironmentConfig(cwd=str(repo)).__dict__)
    agent = UnitTestAgent(model, env, **agent_config)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        agent.log_file = log_dir / "unit_test_agent.log"

    task = (
        f"Create a fixed test harness for kernel: {kernel_name}\n"
        f"Repository: {repo}\n\n"
        f"IMPORTANT: Read INSTRUCTIONS.md in the repository for test harness requirements\n"
        f"and COMMANDMENT format rules before creating the harness."
    )
    if discovery_context:
        task += f"\n\n{discovery_context}"

    exit_status, result = agent.run(task)
    if exit_status != "Submitted":
        raise RuntimeError(f"UnitTestAgent did not finish successfully: {exit_status}\n{result}")

    return _extract_test_command(result)
