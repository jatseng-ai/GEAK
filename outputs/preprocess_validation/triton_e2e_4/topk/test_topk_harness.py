#!/usr/bin/env python3
"""
Test harness for the Triton topk kernel.
Modes: --correctness, --benchmark, --full-benchmark, --profile
"""
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
_REPO_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".geak_resolved", "ROCm_aiter", "main-e15b75d464bf",
)

_work_dir = os.environ.get("GEAK_WORK_DIR")
_repo_root = os.environ.get("GEAK_REPO_ROOT", _REPO_ROOT)

_candidates = []
if _work_dir:
    _candidates.append(_work_dir)
_candidates.append(_repo_root)
_candidates.append(os.path.join(_repo_root, "aiter", "ops", "triton"))

for _p in _candidates:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aiter.ops.triton.topk import topk as triton_topk  # noqa: E402

# ---------------------------------------------------------------------------
# Config stream -- mirrors bench_topk.py exactly
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
    Ordered full case stream: outer product (M, K) x BATCH_SIZES.
    Each config is (batch, M, K).
    """
    configs = []
    for M, K in itertools.product(DIM2S, KS):
        for B in BATCH_SIZES:
            configs.append((B, M, K))
    return configs


ALL_CONFIGS = _build_all_configs()


def _pick(configs, count):
    if len(configs) <= count:
        return list(configs)
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


# ---------------------------------------------------------------------------
# Correctness helpers (from test_topk.py)
# ---------------------------------------------------------------------------
RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}


def _check_correctness(batch, M, K):
    """Run correctness check for one config. Raises on failure."""
    torch.manual_seed(42)
    x = torch.arange(M, dtype=DTYPE, device=DEVICE).repeat(batch, 1)
    # Per-row shuffle
    for b in range(batch):
        x[b] = x[b, torch.randperm(M, device=DEVICE)]

    ref_value, ref_index = torch.topk(x, K, largest=True)
    res_value, res_index = triton_topk(x, K, largest=True)

    # Values check
    res_v_cpu = res_value.cpu()
    ref_v_cpu = ref_value.cpu().to(DTYPE)
    atol = 1e-4  # reduce_dim=1
    rtol = RESOLUTION[DTYPE]
    torch.testing.assert_close(res_v_cpu, ref_v_cpu, atol=atol, rtol=rtol)

    # Indices check
    res_i_cpu = res_index.cpu()
    ref_i_cpu = ref_index.cpu()
    torch.testing.assert_close(res_i_cpu, ref_i_cpu, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def _bench_one(batch, M, K):
    """Benchmark one config, return median latency in ms."""
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
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------
def run_correctness(configs):
    print(f"Running correctness on {len(configs)} configs...")
    for i, (B, M, K) in enumerate(configs):
        try:
            _check_correctness(B, M, K)
            print(f"  [{i+1}/{len(configs)}] PASS  B={B} M={M} K={K}")
        except Exception as e:
            print(f"  [{i+1}/{len(configs)}] FAIL  B={B} M={M} K={K}: {e}")
            print(f"GEAK_SHAPES_USED={sorted(set(configs))}")
            sys.exit(1)
    print(f"GEAK_SHAPES_USED={sorted(set(configs))}")
    print("All correctness checks passed.")


def run_benchmark(configs, label="benchmark"):
    print(f"Running {label} on {len(configs)} configs...")
    latencies = []
    for i, (B, M, K) in enumerate(configs):
        ms = _bench_one(B, M, K)
        latencies.append(ms)
        print(f"  B={B} M={M} K={K}  {ms:.4f}ms")

    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))

    print(f"GEAK_SHAPES_USED={sorted(set(configs))}")
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")


def run_profile(configs):
    print(f"Running profile on {len(configs)} configs...")
    for i, (B, M, K) in enumerate(configs):
        x = torch.rand(B, M, device=DEVICE, dtype=DTYPE)
        triton_topk(x, K, largest=True)
        print(f"  [{i+1}/{len(configs)}] B={B} M={M} K={K}")
    print(f"GEAK_SHAPES_USED={sorted(set(configs))}")


def main():
    parser = argparse.ArgumentParser(description="Test harness for topk kernel")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)

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
