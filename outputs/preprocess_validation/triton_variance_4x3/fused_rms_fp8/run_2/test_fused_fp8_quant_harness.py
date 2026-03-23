#!/usr/bin/env python3
"""
Test harness for fused_fp8_quant kernel.
Modes: --correctness, --benchmark, --full-benchmark, --profile
"""
import os
import sys
import argparse
import math

# Resolve imports: prefer GEAK_WORK_DIR, then GEAK_REPO_ROOT + kernel's repo-relative dir, then original kernel dir
_REPO_ROOT = "/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf"
_work_dir = os.environ.get("GEAK_WORK_DIR", "")
_repo_root = os.environ.get("GEAK_REPO_ROOT", "")
_kernel_rel_dir = "aiter/ops/triton/quant"

if _work_dir and os.path.isdir(_work_dir):
    sys.path.insert(0, _work_dir)
elif _repo_root and os.path.isdir(_repo_root):
    sys.path.insert(0, _repo_root)
else:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

torch.manual_seed(42)

import aiter
from aiter.ops.triton.quant.fused_fp8_quant import (
    fused_rms_fp8_per_tensor_static_quant,
    fused_rms_fp8_group_quant,
    fused_flatten_fp8_group_quant,
    fused_reduce_act_mul_fp8_group_quant,
    fused_reduce_rms_fp8_group_quant,
    fused_silu_mul_fp8_per_tensor_static_quant,
)

fp8_dtype = aiter.dtypes.fp8

WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ============================================================
# Reference implementations from test file
# ============================================================

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


def per_tensor_fp8_static_quant(x, dtype_quant, x_scale):
    DTYPE_MAX = torch.finfo(dtype_quant).max
    scale_recip = 1.0 / x_scale
    x_quant = torch.clamp(x * scale_recip, -DTYPE_MAX, DTYPE_MAX).to(dtype_quant)
    return x_quant


def upcast(x, s, dtype, group_size=128):
    x_N = x.shape[1]
    x = x.reshape(-1, x_N // group_size, group_size).to(torch.float32) * s.reshape(
        -1, s.shape[1], 1
    )
    x = x.reshape(-1, x_N)
    return x.to(dtype=dtype)


def generate_fused_rms_quant_data(M, N1, N2, dtype=torch.bfloat16):
    x1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    x2 = torch.randn((M, N2), dtype=dtype, device="cuda") / 10
    w1 = torch.ones((N1,), dtype=torch.float32, device="cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cuda")
    res1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    return x1, w1, x2, w2, res1


def run_torch_rms_fp8_group_quant(
    x1, w1, eps1, x2, w2, eps2, res1, dtype_quant, group_size
):
    s = x1 + res1
    y1 = rmsnorm(s, w1, eps1)
    y2 = rmsnorm(x2, w2, eps2)
    y1_q, y1_s = per_token_fp8_group_quant(y1, dtype_quant, group_size)
    return (y1_q, y1_s), y1.to(x1.dtype), y2.to(x1.dtype), s.to(x1.dtype)


# ============================================================
# Config list from test_fused_rms_fp8_group_quant parametrize
# Order: dtype (outermost) -> N1,N2 -> M (innermost)
# ============================================================

def _build_all_configs():
    """Build ordered config list from test parametrize decorators."""
    configs = []
    Ms = [1, 32, 256]
    N1N2s = [(128, 128), (128, 7168), (7168, 7168)]
    dtypes = [torch.float16, torch.bfloat16]
    for dtype in dtypes:
        for n1, n2 in N1N2s:
            for M in Ms:
                configs.append((M, n1, n2, dtype))
    return configs


ALL_CONFIGS = _build_all_configs()


def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


# ============================================================
# Benchmark helper
# ============================================================

def _make_benchmark_fn(M, N1, N2, dtype):
    """Create a callable that runs fused_rms_fp8_group_quant."""
    group_size = 128
    dtype_quant = fp8_dtype
    x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)

    def fn():
        return fused_rms_fp8_group_quant(
            x1, w1, 1e-6,
            inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
            group_size=group_size, dtype_quant=dtype_quant,
            res1=res1, output_unquantized_inp1=True,
        )
    return fn


def _benchmark_one(fn):
    """Benchmark a single callable using GPU events. Returns median ms."""
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


# ============================================================
# Correctness check
# ============================================================

def _check_correctness_one(M, N1, N2, dtype):
    """Run correctness check for one config. Returns True if passed."""
    torch.manual_seed(42)
    group_size = 128
    dtype_quant = fp8_dtype
    x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)

    # Reference
    (y1_q_torch, y1_s_torch), y1_torch, y2_torch, y1_res_torch = \
        run_torch_rms_fp8_group_quant(x1, w1, 1e-6, x2, w2, 1e-6, res1, dtype_quant, group_size)

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

        y1_upcast_torch = upcast(y1_q_torch, y1_s_torch, dtype=torch.float32, group_size=group_size)
        y1_upcast_triton = upcast(y1_q_triton, y1_s_triton, dtype=torch.float32, group_size=group_size)
        torch.testing.assert_close(y1_upcast_torch, y1_upcast_triton, atol=0.1, rtol=0.1)
        return True
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return False


# ============================================================
# CLI modes
# ============================================================

def _config_str(cfg):
    M, N1, N2, dtype = cfg
    dt = "fp16" if dtype == torch.float16 else "bf16"
    return f"M={M} N1={N1} N2={N2} dtype={dt}"


def _config_tuple_str(cfg):
    M, N1, N2, dtype = cfg
    dt = "torch.float16" if dtype == torch.float16 else "torch.bfloat16"
    return f"({M}, {N1}, {N2}, {dt})"


def run_correctness(configs):
    print(f"Running correctness on {len(configs)} configs...")
    all_pass = True
    for cfg in configs:
        passed = _check_correctness_one(*cfg)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {_config_str(cfg)}")
        if not passed:
            all_pass = False
    shapes_str = "[" + ", ".join(_config_tuple_str(c) for c in configs) + "]"
    print(f"GEAK_SHAPES_USED={shapes_str}")
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


def run_benchmark(configs):
    print(f"Running benchmark on {len(configs)} configs...")
    latencies = []
    for cfg in configs:
        fn = _make_benchmark_fn(*cfg)
        lat = _benchmark_one(fn)
        latencies.append(lat)
        print(f"  {_config_str(cfg)}  {lat:.4f}ms")
    # Geometric mean
    log_sum = sum(math.log(l) for l in latencies)
    geo_mean = math.exp(log_sum / len(latencies))
    shapes_str = "[" + ", ".join(_config_tuple_str(c) for c in configs) + "]"
    print(f"GEAK_SHAPES_USED={shapes_str}")
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")


def run_profile(configs):
    print(f"Running profile on {len(configs)} configs...")
    for cfg in configs:
        fn = _make_benchmark_fn(*cfg)
        # Just run the kernel a few times for profiling
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        print(f"  Profiled: {_config_str(cfg)}")
    shapes_str = "[" + ", ".join(_config_tuple_str(c) for c in configs) + "]"
    print(f"GEAK_SHAPES_USED={shapes_str}")


def main():
    parser = argparse.ArgumentParser(description="Test harness for fused_fp8_quant")
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
