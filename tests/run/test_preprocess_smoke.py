from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_ROOT = _REPO_ROOT / "src"


def test_preprocess_modules_import() -> None:
    modules = [
        "minisweagent.run.preprocess",
        "minisweagent.run.preprocess.repo_paths",
        "minisweagent.run.preprocess.resolve_kernel_url",
        "minisweagent.run.preprocess.codebase_context",
        "minisweagent.run.preprocess.discovery_types",
        "minisweagent.run.preprocess.run_harness",
        "minisweagent.run.preprocess.harness_utils",
        "minisweagent.run.preprocess.unit_test_agent",
        "minisweagent.run.preprocess.shape_fixer_agent",
        "minisweagent.run.preprocess.testcase_cache",
        "minisweagent.run.preprocess.kernel_profile",
        "minisweagent.run.preprocess.baseline",
        "minisweagent.run.preprocess.benchmark_parsing",
        "minisweagent.run.preprocess.commandment",
        "minisweagent.run.preprocess.validate_commandment",
        "minisweagent.run.preprocess.preprocessor",
    ]
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None


def test_preprocess_owned_prompt_assets_exist() -> None:
    config_dir = _SRC_ROOT / "minisweagent" / "run" / "preprocess" / "config"
    assert (config_dir / "mini_unit_test_agent.yaml").is_file()
    assert (config_dir / "mini_shape_fixer.yaml").is_file()


def test_preprocess_repo_root_helper_points_to_repo_root() -> None:
    from minisweagent.run.preprocess.repo_paths import get_preprocess_repo_root

    repo_root = get_preprocess_repo_root()
    assert repo_root == _REPO_ROOT


def test_preprocess_cli_help() -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_SRC_ROOT}:{existing}" if existing else str(_SRC_ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "minisweagent.run.preprocess.preprocessor", "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0
    assert "GEAK preprocessor" in result.stdout
