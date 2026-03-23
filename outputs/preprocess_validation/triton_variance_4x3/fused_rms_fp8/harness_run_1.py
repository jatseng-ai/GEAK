#!/usr/bin/env python3
"""
Test harness for fused_fp8_quant kernel.
Modes: --correctness, --profile, --benchmark, --full-benchmark
Shape source: op_tests/triton_tests/quant/test_fused_fp8_quant.py
"""
import os
import sys
import argparse
import math

import torch
import torch.nn.functional as F

# Ensure the repo root is on sys.path so 'aiter' can be imported
REPO_ROOT = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get(
        "GEAK_REPO_ROOT",
        "/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf",
    ),
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import aiter
from aiter.ops.triton.quant.fused_fp8_quant import (
    fused_rms_fp8_group_quant,
)

# ── Constants ──────────────────────────────────────────────────────────
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

torch.manual_seed(42)

fp8_dtype = aiter.dtypes.fp8

# ── Reference implementations (from test file) ────────────────────────

def rmsnorm(input, weight, eps=1e-6):
    row_norm = input * input
    row_norm = torch.sum(row_norm, dim=-1)
    norm_factor = torch.rsqrt((row_norm / input.shape[1]) + eps)
    rms_norm = input * norm_factor[:, None] * weight[None, :]
    return rms_norm


def per_token_fp8_group_quant(x, dtype_quant, group_size=128):
    DTYPE_MAX = torch.finfo(dtype_quant).max
    M, N = x.shape
    if N % group_size > 0:
        num_pad = group_size - (N % group_size)
        x_reshape = F.pad(x, (0, num_pad, 0, 0), "constant", 0)
        x_reshape = x_reshape.reshape(
            M, (N + group_size - 1) // group_size, group_size
        ).to(torch.float32)
    else:
        x_reshape = x.reshape(M, N // group_size, group_size).to(torch.float32)
    x_max = torch.max(torch.abs(x_reshape), dim=-1, keepdim=True)[0]
    x_max = torch.where(x_max < 1e-10, 1e-10, x_max).to(torch.float32)
    x_scale = x_max / DTYPE_MAX
    scale_recip = 1.0 / x_scale
    x_quant = torch.clamp(x_reshape * scale_recip, -DTYPE_MAX, DTYPE_MAX).to(
        dtype_quant
    )
    x_quant = x_quant.reshape(M, (N + group_size - 1) // group_size * group_size)[:, :N]
    x_scale = x_scale.squeeze(-1)
    return x_quant, x_scale


def upcast(x, s, dtype, group_size=128):
    x_N = x.shape[1]
    x = x.reshape(-1, x_N // group_size, group_size).to(torch.float32) * s.reshape(
        -1, s.shape[1], 1
    )
    x = x.reshape(-1, x_N)
    return x.to(dtype=dtype)


def run_torch_rms_fp8_group_quant(
    x1, w1, eps1, x2, w2, eps2, res1, dtype_quant, group_size
):
    s = x1 + res1
    y1 = rmsnorm(s, w1, eps1)
    y2 = rmsnorm(x2, w2, eps2)
    y1_q, y1_s = per_token_fp8_group_quant(y1, dtype_quant, group_size)
    return (y1_q, y1_s), y1.to(x1.dtype), y2.to(x1.dtype), s.to(x1.dtype)


def generate_fused_rms_quant_data(M, N1, N2, dtype=torch.bfloat16):
    x1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    x2 = torch.randn((M, N2), dtype=dtype, device="cuda") / 10
    w1 = torch.ones((N1,), dtype=torch.float32, device="cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cuda")
    res1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    return x1, w1, x2, w2, res1


# ── Config list (from test_fused_rms_fp8_group_quant parametrize) ──────
# Parametrize order in pytest (outermost first):
#   dtype: [torch.float16, torch.bfloat16]
#   N1, N2: [(128, 128), (128, 7168), (7168, 7168)]
#   M: [1, 32, 256]
# pytest expands outermost first, so the order is:
# for dtype in [...]: for (N1,N2) in [...]: for M in [...]:
ALL_CONFIGS = []
for dtype in [torch.float16, torch.bfloat16]:
    for (N1, N2) in [(128, 128), (128, 7168), (7168, 7168)]:
        for M in [1, 32, 256]:
            ALL_CONFIGS.append((M, N1, N2, dtype))


def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


def config_str(cfg):
    M, N1, N2, dtype = cfg
    dtype_name = "fp16" if dtype == torch.float16 else "bf16"
    return f"M={M} N1={N1} N2={N2} dtype={dtype_name}"


# ── Correctness ────────────────────────────────────────────────────────

def run_correctness(configs):
    print(f"Running correctness on {len(configs)} configs...")
    group_size = 128
    dtype_quant = fp8_dtype
    all_pass = True

    for cfg in configs:
        M, N1, N2, dtype = cfg
        torch.manual_seed(42)
        x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)

        # Reference
        (y1_q_torch, y1_s_torch), y1_torch, y2_torch, y1_res_torch = \
            run_torch_rms_fp8_group_quant(
                x1, w1, 1e-6, x2, w2, 1e-6, res1, dtype_quant, group_size
            )

        # Triton kernel
        (y1_q_triton, y1_s_triton), y1_triton, y2_triton, y1_res_triton = \
            fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size, dtype_quant=dtype_quant,
                res1=res1, output_unquantized_inp1=True,
            )

        try:
            torch.testing.assert_close(y1_torch, y1_triton, atol=0.1, rtol=0.1)
            torch.testing.assert_close(y2_torch, y2_triton, atol=0.1, rtol=0.1)
            torch.testing.assert_close(y1_res_torch, y1_res_triton, atol=0.1, rtol=0.1)

            y1_upcast_torch = upcast(
                y1_q_torch, y1_s_torch, dtype=torch.float32, group_size=group_size
            )
            y1_upcast_triton = upcast(
                y1_q_triton, y1_s_triton, dtype=torch.float32, group_size=group_size
            )
            torch.testing.assert_close(y1_upcast_torch, y1_upcast_triton, atol=0.1, rtol=0.1)
            print(f"  PASS: {config_str(cfg)}")
        except AssertionError as e:
            print(f"  FAIL: {config_str(cfg)}: {e}")
            all_pass = False

    shapes_used = sorted([(M, N1, N2, str(d)) for M, N1, N2, d in configs])
    print(f"GEAK_SHAPES_USED={shapes_used}")

    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


# ── Benchmark ──────────────────────────────────────────────────────────

def run_benchmark(configs):
    print(f"Running benchmark on {len(configs)} configs...")
    group_size = 128
    dtype_quant = fp8_dtype
    latencies = []

    for cfg in configs:
        M, N1, N2, dtype = cfg
        torch.manual_seed(42)
        x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)

        def run_kernel():
            fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size, dtype_quant=dtype_quant,
                res1=res1, output_unquantized_inp1=True,
            )

        # Warmup
        for _ in range(WARMUP):
            run_kernel()
        torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(ITERATIONS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_kernel()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median_ms = times[len(times) // 2]
        latencies.append(median_ms)
        print(f"  {config_str(cfg)}  {median_ms:.4f}ms")

    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))

    shapes_used = sorted([(M, N1, N2, str(d)) for M, N1, N2, d in configs])
    print(f"GEAK_SHAPES_USED={shapes_used}")
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")


# ── Profile ────────────────────────────────────────────────────────────

def run_profile(configs):
    print(f"Running profile on {len(configs)} configs...")
    group_size = 128
    dtype_quant = fp8_dtype

    for cfg in configs:
        M, N1, N2, dtype = cfg
        torch.manual_seed(42)
        x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)

        # Warmup
        for _ in range(5):
            fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size, dtype_quant=dtype_quant,
                res1=res1, output_unquantized_inp1=True,
            )
        torch.cuda.synchronize()

        # Profile run
        fused_rms_fp8_group_quant(
            x1, w1, 1e-6,
            inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
            group_size=group_size, dtype_quant=dtype_quant,
            res1=res1, output_unquantized_inp1=True,
        )
        torch.cuda.synchronize()
        print(f"  Profiled: {config_str(cfg)}")

    shapes_used = sorted([(M, N1, N2, str(d)) for M, N1, N2, d in configs])
    print(f"GEAK_SHAPES_USED={shapes_used}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test harness for fused_fp8_quant")
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
        run_benchmark(configs)
    elif args.full_benchmark:
        run_benchmark(ALL_CONFIGS)
    elif args.profile:
        configs = _pick(ALL_CONFIGS, 5)
        run_profile(configs)


if __name__ == "__main__":
    main()
