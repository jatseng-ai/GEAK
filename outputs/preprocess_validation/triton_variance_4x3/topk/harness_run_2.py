#!/usr/bin/env python3
"""
Test harness for topk kernel.
Modes: --correctness, --profile, --benchmark, --full-benchmark
"""
from __future__ import annotations
import argparse
import itertools
import math
import os
import sys
import torch

# ---------------------------------------------------------------------------
# Resolve imports: prefer GEAK_WORK_DIR, then GEAK_REPO_ROOT + repo-relative,
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

from aiter.ops.triton.topk import topk as triton_topk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = "cuda"
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ---------------------------------------------------------------------------
# Ordered full case stream — matches bench_topk.py exactly:
#   for M, K in itertools.product(DIM2S, KS):
#       for B in BATCH_SIZES:
#           (B, M, K)
# ---------------------------------------------------------------------------
BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
DIM2S = (16, 128, 256, 128256)
KS = (2, 8)

ALL_CONFIGS = [
    (B, M, K)
    for M, K in itertools.product(DIM2S, KS)
    for B in BATCH_SIZES
]

# ---------------------------------------------------------------------------
# Subsetting helper
# ---------------------------------------------------------------------------
def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


# ---------------------------------------------------------------------------
# Correctness reference (from test_topk.py)
# ---------------------------------------------------------------------------
RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}

def check_correctness(configs):
    torch.manual_seed(42)
    dtype = torch.float32
    failures = 0
    for idx, (B, M, K) in enumerate(configs):
        # Build input exactly as test_topk.py does
        x = torch.arange(M, dtype=dtype, device=DEVICE).repeat(B, 1)
        for b in range(B):
            x[b] = x[b, torch.randperm(M, device=DEVICE)]

        ref_value, ref_index = torch.topk(x, K, largest=True)
        res_value, res_index = triton_topk(x, K, largest=True)

        # Compare values
        atol = 1e-4 * 1  # reduce_dim=1 as in test
        rtol = RESOLUTION[dtype]
        try:
            torch.testing.assert_close(
                res_value.cpu(), ref_value.cpu().to(dtype), atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                res_index.cpu(), ref_index.cpu(), atol=0, rtol=0
            )
            print(f"  PASS  B={B} M={M} K={K}")
        except AssertionError as e:
            print(f"  FAIL  B={B} M={M} K={K}: {e}")
            failures += 1
    return failures


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def benchmark_configs(configs):
    torch.manual_seed(42)
    latencies = []
    for B, M, K in configs:
        x = torch.rand(B, M, device=DEVICE, dtype=torch.float32)

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

        times.sort()
        median_ms = times[len(times) // 2]
        latencies.append(median_ms)
        print(f"B={B} M={M} K={K}  {median_ms:.4f}ms")

    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))
    return geo_mean


# ---------------------------------------------------------------------------
# Profile helper
# ---------------------------------------------------------------------------
def profile_configs(configs):
    torch.manual_seed(42)
    for B, M, K in configs:
        x = torch.rand(B, M, device=DEVICE, dtype=torch.float32)
        # Warmup
        for _ in range(3):
            triton_topk(x, K, largest=True)
        torch.cuda.synchronize()
        # Single run for profiling
        triton_topk(x, K, largest=True)
        torch.cuda.synchronize()
        print(f"B={B} M={M} K={K}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TopK test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.correctness:
        configs = _pick(ALL_CONFIGS, 25)
        print(f"Running correctness on {len(configs)} configs...")
        failures = check_correctness(configs)
        print(f"GEAK_SHAPES_USED={sorted(configs)}")
        if failures > 0:
            print(f"FAILED: {failures} configs failed correctness")
            sys.exit(1)
        print("All correctness checks passed.")

    elif args.benchmark:
        configs = _pick(ALL_CONFIGS, 25)
        print(f"Running benchmark on {len(configs)} configs...")
        geo_mean = benchmark_configs(configs)
        print(f"GEAK_SHAPES_USED={sorted(configs)}")
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")

    elif args.full_benchmark:
        configs = ALL_CONFIGS
        print(f"Running full benchmark on {len(configs)} configs...")
        geo_mean = benchmark_configs(configs)
        print(f"GEAK_SHAPES_USED={sorted(configs)}")
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")

    elif args.profile:
        configs = _pick(ALL_CONFIGS, 5)
        print(f"Running profile on {len(configs)} configs...")
        profile_configs(configs)
        print(f"GEAK_SHAPES_USED={sorted(configs)}")


if __name__ == "__main__":
    main()
