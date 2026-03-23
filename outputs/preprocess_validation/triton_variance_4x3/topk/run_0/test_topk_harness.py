#!/usr/bin/env python3
"""Test harness for topk kernel: correctness, benchmark, profile, full-benchmark."""
from __future__ import annotations
import argparse
import itertools
import math
import os
import sys
import torch
import triton

# ---------------------------------------------------------------------------
# Resolve imports: prefer GEAK_WORK_DIR, then GEAK_REPO_ROOT + kernel subdir,
# then the original kernel directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = "/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf"
_work_dir = os.environ.get("GEAK_WORK_DIR")
_repo_root = os.environ.get("GEAK_REPO_ROOT", _REPO_ROOT)
_kernel_rel = "aiter/ops/triton"

_candidates = []
if _work_dir:
    _candidates.append(_work_dir)
    _candidates.append(os.path.join(_work_dir, _kernel_rel))
_candidates.append(_repo_root)
_candidates.append(os.path.join(_repo_root, _kernel_rel))
_candidates.append(os.path.join(_REPO_ROOT, _kernel_rel))

for _p in _candidates:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Also ensure the repo root is on sys.path for op_tests imports
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
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
# Configs from benchmark file: bench_topk.py
# Ordered case stream: itertools.product(DIM2S, KS) outer, BATCH_SIZES inner
# ---------------------------------------------------------------------------
BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
DIM2S = (16, 128, 256, 128256)
KS = (2, 8)

# Build ordered full case stream: (batch, M, K)
ALL_CONFIGS = []
for M, K in itertools.product(DIM2S, KS):
    for batch in BATCH_SIZES:
        ALL_CONFIGS.append((batch, M, K))

# ---------------------------------------------------------------------------
# Correctness tolerances from test_topk.py
# ---------------------------------------------------------------------------
RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}
DTYPE = torch.float32


def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


def _make_input(batch, M, dtype=DTYPE, seed=42):
    """Create input tensor matching test_topk.py pattern."""
    torch.manual_seed(seed)
    x = torch.arange(M, dtype=dtype, device=DEVICE).repeat(batch, 1)
    for b in range(batch):
        x[b] = x[b, torch.randperm(M, device=DEVICE)]
    return x


def run_correctness(configs):
    """Run correctness checks. Exit non-zero on failure."""
    print(f"Running correctness on {len(configs)} configs...")
    for i, (batch, M, K) in enumerate(configs):
        torch.manual_seed(42)
        x = _make_input(batch, M)
        ref_value, ref_index = torch.topk(x, K, largest=True)
        res_value, res_index = triton_topk(x, K, largest=True)

        # Move to CPU for comparison
        res_value_cpu = res_value.cpu()
        res_index_cpu = res_index.cpu()
        ref_value_cpu = ref_value.cpu()
        ref_index_cpu = ref_index.cpu()

        atol = 1e-4 * 1  # reduce_dim=1 from test
        rtol = RESOLUTION[DTYPE]
        try:
            torch.testing.assert_close(
                res_value_cpu, ref_value_cpu.to(DTYPE), atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                res_index_cpu, ref_index_cpu, atol=0, rtol=0
            )
        except AssertionError as e:
            print(f"FAIL config ({batch}, {M}, {K}): {e}")
            sys.exit(1)
        print(f"  [{i+1}/{len(configs)}] batch={batch} M={M} K={K} PASS")
    print("All correctness checks passed.")


def run_benchmark(configs, label="benchmark"):
    """Run benchmark, print per-config latency and geometric mean."""
    print(f"Running {label} on {len(configs)} configs...")
    latencies = []
    for i, (batch, M, K) in enumerate(configs):
        torch.manual_seed(42)
        x = torch.rand(batch, M, device=DEVICE, dtype=DTYPE)

        fn = lambda: triton_topk(x, K, largest=True)

        # Warmup
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(ITERATIONS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median_ms = times[len(times) // 2]
        latencies.append(median_ms)
        print(f"  batch={batch} M={M} K={K}  {median_ms:.4f}ms")

    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.6f}")
    return latencies


def run_profile(configs):
    """Run a few configs for profiling (no correctness)."""
    print(f"Running profile on {len(configs)} configs...")
    for batch, M, K in configs:
        torch.manual_seed(42)
        x = torch.rand(batch, M, device=DEVICE, dtype=DTYPE)
        # Just run the kernel
        triton_topk(x, K, largest=True)
        torch.cuda.synchronize()
    print("Profile runs complete.")


def main():
    parser = argparse.ArgumentParser(description="Test harness for topk kernel")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if not any([args.correctness, args.benchmark, args.full_benchmark, args.profile]):
        parser.print_help()
        sys.exit(1)

    if args.correctness:
        configs = _pick(ALL_CONFIGS, 25)
        run_correctness(configs)
        print(f"GEAK_SHAPES_USED={sorted(configs)}")

    if args.profile:
        configs = _pick(ALL_CONFIGS, 5)
        run_profile(configs)
        print(f"GEAK_SHAPES_USED={sorted(configs)}")

    if args.benchmark:
        configs = _pick(ALL_CONFIGS, 25)
        run_benchmark(configs, "benchmark")
        print(f"GEAK_SHAPES_USED={sorted(configs)}")

    if args.full_benchmark:
        run_benchmark(ALL_CONFIGS, "full-benchmark")
        print(f"GEAK_SHAPES_USED={sorted(ALL_CONFIGS)}")


if __name__ == "__main__":
    main()
