#!/usr/bin/env python3
"""Test harness for fused_qkv_split_qk_rope kernel."""

import os
import sys
import argparse
import math

# Resolve repo root
REPO_ROOT = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get(
        "GEAK_REPO_ROOT",
        "/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf",
    ),
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
torch.manual_seed(42)

# Import kernel under test
from aiter.ops.triton.rope.fused_qkv_split_qk_rope import fused_qkv_split_qk_rope

# Import test helpers
from op_tests.triton_tests.fusions.test_fused_qk_concat import generate_rope_cached_freqs
from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle

# Import test helper functions from the test file
from op_tests.triton_tests.rope.test_fused_qkv_split_qk_rope import (
    generate_qkv_inputs,
    run_torch,
)

# Constants
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# Build ordered full case stream (pytest order)
# Pytest applies decorators bottom-to-top; bottom decorator = outermost loop.
# Decorator order (top to bottom in test file):
#   B, QH_PER_KH, KH, D, rotate_style, max_embed_positions,
#   (nope, nope_first), reuse_freqs_front_part, dtype
# Iteration order (outermost to innermost):
#   dtype, reuse_freqs_front_part, (nope, nope_first), max_embed_positions,
#   rotate_style, D, KH, QH_PER_KH, B

B_VALS = [1, 4, 8, 16, 32]
QH_PER_KH_VALS = [1, 2, 4, 8, 16]
KH_VALS = [1, 4]
D_VALS = [64, 128]
ROTATE_STYLE_VALS = [RotateStyle.GPTJ, RotateStyle.NEOX]
MAX_EMBED_POSITIONS_VALS = [131072]
NOPE_NOPE_FIRST_VALS = [(False, False), (True, False), (True, True)]
REUSE_FREQS_FRONT_PART_VALS = [False, True]
DTYPE_VALS = [torch.bfloat16]


def _build_all_configs():
    configs = []
    for dtype in DTYPE_VALS:
        for reuse_freqs_front_part in REUSE_FREQS_FRONT_PART_VALS:
            for nope, nope_first in NOPE_NOPE_FIRST_VALS:
                for max_embed_positions in MAX_EMBED_POSITIONS_VALS:
                    for rotate_style in ROTATE_STYLE_VALS:
                        for D in D_VALS:
                            for KH in KH_VALS:
                                for QH_PER_KH in QH_PER_KH_VALS:
                                    for B in B_VALS:
                                        configs.append((
                                            B, QH_PER_KH, KH, D,
                                            rotate_style, max_embed_positions,
                                            nope, nope_first,
                                            reuse_freqs_front_part, dtype,
                                        ))
    return configs


ALL_CONFIGS = _build_all_configs()


def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


def _config_str(cfg):
    B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype = cfg
    rs = "NEOX" if rotate_style == RotateStyle.NEOX else "GPTJ"
    return (
        "B={} QH_PER_KH={} KH={} D={} rotate={} nope={} nope_first={} reuse_front={}".format(
            B, QH_PER_KH, KH, D, rs, nope, nope_first, reuse_freqs_front_part
        )
    )


def _build_inputs(cfg):
    B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype = cfg
    torch.manual_seed(42)
    qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype)
    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, (D // 2) if reuse_freqs_front_part else D, dtype
    )
    ref_freqs = freqs[pos].squeeze(-2)
    head_dim = D * (2 if nope else 1)
    qh = QH_PER_KH * KH
    is_neox = (rotate_style == RotateStyle.NEOX)
    return {
        "qkv": qkv, "cos": cos, "sin": sin, "pos": pos,
        "ref_freqs": ref_freqs,
        "qh": qh, "kvh": KH, "head_dim": head_dim,
        "is_neox": is_neox,
        "reuse_freqs_front_part": reuse_freqs_front_part,
        "nope_first": nope_first,
        "nope": nope,
        "QH_PER_KH": QH_PER_KH, "KH": KH, "D": D,
        "rotate_style": rotate_style,
    }


def _run_kernel(inputs):
    return fused_qkv_split_qk_rope(
        inputs["qkv"],
        inputs["cos"],
        inputs["sin"],
        inputs["pos"],
        inputs["qh"],
        inputs["kvh"],
        inputs["head_dim"],
        is_neox=inputs["is_neox"],
        offsets=None,
        reuse_freqs_front_part=inputs["reuse_freqs_front_part"],
        nope_first=inputs["nope_first"],
    )


def _run_reference(inputs):
    return run_torch(
        inputs["qkv"],
        inputs["QH_PER_KH"],
        inputs["KH"],
        inputs["D"] * (2 if inputs["nope"] else 1),
        inputs["ref_freqs"],
        inputs["reuse_freqs_front_part"],
        inputs["nope"],
        inputs["nope_first"],
        inputs["rotate_style"],
    )


def run_correctness(configs):
    print("Running correctness on {} configs...".format(len(configs)))
    failures = 0
    for i, cfg in enumerate(configs):
        inputs = _build_inputs(cfg)
        q_tri, k_tri, v_tri = _run_kernel(inputs)
        q_ref, k_ref, v_ref = _run_reference(inputs)
        try:
            torch.testing.assert_close(q_ref, q_tri)
            torch.testing.assert_close(k_ref, k_tri)
            torch.testing.assert_close(v_ref, v_tri)
            print("  [{}/{}] PASS  {}".format(i + 1, len(configs), _config_str(cfg)))
        except AssertionError as e:
            print("  [{}/{}] FAIL  {}".format(i + 1, len(configs), _config_str(cfg)))
            print("    {}".format(e))
            failures += 1

    print("\nGEAK_SHAPES_USED={}".format(sorted([cfg[:9] for cfg in configs])))
    if failures > 0:
        print("\nFAILED: {}/{} configs failed correctness.".format(failures, len(configs)))
        sys.exit(1)
    else:
        print("\nPASSED: All {} configs passed correctness.".format(len(configs)))


def run_benchmark(configs, label="benchmark"):
    print("Running {} on {} configs...".format(label, len(configs)))
    latencies = []

    for i, cfg in enumerate(configs):
        inputs = _build_inputs(cfg)

        # Warmup
        for _ in range(WARMUP):
            _run_kernel(inputs)
        torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(ITERATIONS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _run_kernel(inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        median_ms = sorted(times)[len(times) // 2]
        latencies.append(median_ms)
        print("  {}  {:.4f}ms".format(_config_str(cfg), median_ms))

    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))

    print("\nGEAK_SHAPES_USED={}".format(sorted([cfg[:9] for cfg in configs])))
    print("GEAK_RESULT_LATENCY_MS={:.4f}".format(geo_mean))


def run_profile(configs):
    print("Running profile on {} configs...".format(len(configs)))
    for i, cfg in enumerate(configs):
        inputs = _build_inputs(cfg)
        # Warmup
        for _ in range(3):
            _run_kernel(inputs)
        torch.cuda.synchronize()
        # Single run for profiling
        _run_kernel(inputs)
        torch.cuda.synchronize()
        print("  [{}/{}] {}".format(i + 1, len(configs), _config_str(cfg)))

    print("\nGEAK_SHAPES_USED={}".format(sorted([cfg[:9] for cfg in configs])))


def main():
    parser = argparse.ArgumentParser(description="Test harness for fused_qkv_split_qk_rope")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.correctness:
        configs = _pick(ALL_CONFIGS, 25)
        run_correctness(configs)
    elif args.benchmark:
        configs = _pick(ALL_CONFIGS, 25)
        run_benchmark(configs, "benchmark")
    elif args.full_benchmark:
        run_benchmark(ALL_CONFIGS, "full-benchmark")
    elif args.profile:
        configs = _pick(ALL_CONFIGS, 5)
        run_profile(configs)


if __name__ == "__main__":
    main()
