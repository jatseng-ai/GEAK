#!/usr/bin/env python3
"""
Test harness for the Triton topk kernel.
Modes: --correctness, --profile, --benchmark, --full-benchmark
"""
from __future__ import annotations
import argparse
import math
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Resolve imports: prefer GEAK_WORK_DIR, then GEAK_REPO_ROOT + kernel subdir,
# then the original kernel directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get(
        "GEAK_REPO_ROOT",
        "/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf",
    ),
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aiter.ops.triton.topk import topk as triton_topk  # noqa: E402

# ---------------------------------------------------------------------------
# Config space - taken verbatim from bench_topk.py
# ---------------------------------------------------------------------------
BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
DIM2S = (16, 128, 256, 128256)  # row length M
KS = (2, 8)  # top-k values

DEVICE = "cuda"
DTYPE = torch.float32

WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))


def _build_all_configs():
    """
    Build the ordered full case stream.
    Order: outer DIM2S, middle KS, inner BATCH_SIZES
    (matches the benchmark file's itertools.product(DIM2S, KS) with BATCH_SIZES as x_vals).
    """
    configs = []
    for M in DIM2S:
        for K in KS:
            for B in BATCH_SIZES:
                configs.append((B, M, K))
    return configs


ALL_CONFIGS = _build_all_configs()


def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}


def run_correctness(configs):
    """Run correctness checks against torch.topk."""
    torch.manual_seed(42)
    print(f"Running correctness on {len(configs)} configs...")
    failures = 0
    for idx, (B, M, K) in enumerate(configs):
        # Build input: arange + per-row shuffle (matches test_topk.py)
        x = torch.arange(M, dtype=DTYPE, device=DEVICE).repeat(B, 1)
        for b in range(B):
            x[b] = x[b, torch.randperm(M, device=DEVICE)]

        ref_value, ref_index = torch.topk(x, K, largest=True)
        res_value, res_index = triton_topk(x, K, largest=True)

        # Check values
        atol = 1e-4  # reduce_dim=1 from test
        rtol = RESOLUTION[DTYPE]
        try:
            torch.testing.assert_close(
                res_value.cpu(), ref_value.cpu().to(DTYPE), atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                res_index.cpu(), ref_index.cpu(), atol=0, rtol=0
            )
            status = "PASS"
        except AssertionError as e:
            status = "FAIL"
            failures += 1
            print(f"  [{idx+1}/{len(configs)}] B={B} M={M} K={K} -> {status}: {e}")
            continue

        print(f"  [{idx+1}/{len(configs)}] B={B} M={M} K={K} -> {status}")

    print(f"\nCorrectness: {len(configs) - failures}/{len(configs)} passed")
    shapes_used = sorted(configs)
    print(f"GEAK_SHAPES_USED={shapes_used}")
    if failures > 0:
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(configs, label="benchmark"):
    """Benchmark using GPU events."""
    torch.manual_seed(42)
    print(f"Running {label} on {len(configs)} configs...")
    latencies = []

    for idx, (B, M, K) in enumerate(configs):
        x = torch.rand(B, M, device=DEVICE, dtype=DTYPE)

        # Warmup
        for _ in range(WARMUP):
            triton_topk(x, K, largest=True)
        torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(ITERATIONS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            triton_topk(x, K, largest=True)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        median_ms = sorted(times)[len(times) // 2]
        latencies.append(median_ms)
        print(f"  B={B} M={M} K={K}  {median_ms:.4f}ms")

    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))

    shapes_used = sorted(configs)
    print(f"GEAK_SHAPES_USED={shapes_used}")
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.6f}")


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
def run_profile(configs):
    """Run a few configs for profiling (no correctness)."""
    torch.manual_seed(42)
    print(f"Running profile on {len(configs)} configs...")
    for idx, (B, M, K) in enumerate(configs):
        x = torch.rand(B, M, device=DEVICE, dtype=DTYPE)
        # Warmup
        for _ in range(3):
            triton_topk(x, K, largest=True)
        torch.cuda.synchronize()
        # One timed run
        triton_topk(x, K, largest=True)
        torch.cuda.synchronize()
        print(f"  [{idx+1}/{len(configs)}] B={B} M={M} K={K} done")

    shapes_used = sorted(configs)
    print(f"GEAK_SHAPES_USED={shapes_used}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Topk kernel test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--profile", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    if args.correctness:
        configs = _pick(ALL_CONFIGS, 25)
        run_correctness(configs)
    elif args.profile:
        configs = _pick(ALL_CONFIGS, 5)
        run_profile(configs)
    elif args.benchmark:
        configs = _pick(ALL_CONFIGS, 25)
        run_benchmark(configs, label="benchmark")
    elif args.full_benchmark:
        run_benchmark(ALL_CONFIGS, label="full-benchmark")


if __name__ == "__main__":
    main()
