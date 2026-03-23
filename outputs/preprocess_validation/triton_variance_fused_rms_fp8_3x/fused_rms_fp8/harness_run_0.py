#!/usr/bin/env python3
"""
Test harness for fused_fp8_quant kernel.
Modes: --correctness, --benchmark, --full-benchmark, --profile
"""
import os
import sys
import argparse
import math

# Resolve imports: prefer GEAK_WORK_DIR, then GEAK_REPO_ROOT + kernel subdir, then original kernel dir
_REPO_ROOT = "/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf"
_work_dir = os.environ.get("GEAK_WORK_DIR", "")
_repo_root = os.environ.get("GEAK_REPO_ROOT", "")
_kernel_rel_dir = "aiter/ops/triton/quant"

if _work_dir and os.path.isdir(_work_dir):
    sys.path.insert(0, _work_dir)
elif _repo_root and os.path.isdir(_repo_root):
    sys.path.insert(0, _repo_root)
    _kdir = os.path.join(_repo_root, _kernel_rel_dir)
    if os.path.isdir(_kdir):
        sys.path.insert(0, _kdir)
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
# Reference implementations (from test file)
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


def checkAllclose(a, b, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05):
    """Matches the test file's checkAllclose: allows up to tol_err_ratio fraction of mismatches."""
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    if isClose.all():
        return True
    mask = ~isClose
    num = mask.sum()
    percent = (num / a.numel()).item()
    if percent > tol_err_ratio:
        raise AssertionError(
            f"checkAllclose failed: {percent*100:.2f}% elements differ "
            f"(threshold {tol_err_ratio*100:.1f}%), "
            f"max abs diff={torch.max(torch.abs(a[mask] - b[mask])).item():.4f}"
        )
    return True


# ============================================================
# Config generation from test file parametrize decorators
# ============================================================

def build_all_configs():
    """
    Build configs from the test file. Each config is a tuple:
    (test_name, param_dict)
    
    Ordered as pytest would: outermost parametrize varies slowest.
    pytest processes parametrize decorators bottom-to-top, so:
    - dtype (bottom) varies fastest
    - N1,N2 (middle) varies next
    - M (top) varies slowest
    """
    configs = []
    
    # test_fused_rms_fp8_per_tensor_static_quant configs (source line 105 - first)
    M_vals = [1, 32, 256]
    N_vals = [(128, 128), (128, 7168), (7168, 7168)]
    dtype_vals = [torch.float16, torch.bfloat16]
    
    for M in M_vals:
        for N1, N2 in N_vals:
            for dtype in dtype_vals:
                configs.append(("per_tensor", {"M": M, "N1": N1, "N2": N2, "dtype": dtype}))
    
    # test_fused_rms_fp8_group_quant configs (source line 149 - second)
    for M in M_vals:
        for N1, N2 in N_vals:
            for dtype in dtype_vals:
                configs.append(("group_quant", {"M": M, "N1": N1, "N2": N2, "dtype": dtype}))
    
    # test_rmsnorm_quant_fuse configs
    rmsnorm_M = [1, 2, 4, 8, 256, 1024, 8192]
    rmsnorm_N = [128, 4096, 8192]
    for m in rmsnorm_M:
        for n in rmsnorm_N:
            configs.append(("rmsnorm_quant", {"M": m, "N": n, "dtype": torch.bfloat16}))
    
    # test_fused_reduce_rms_fp8_group_quant configs
    reduce_M = [1, 32, 256, 8192]
    reduce_N = [(128, 128, 128), (1536, 512, 64), (7168, 7168, 7168)]
    reduce_SPK = [1, 4, 14]
    reduce_dtype = [torch.float16, torch.bfloat16]
    for M in reduce_M:
        for N1, N2, N3 in reduce_N:
            for SPK in reduce_SPK:
                for dtype in reduce_dtype:
                    configs.append(("reduce_rms", {"M": M, "N1": N1, "N2": N2, "N3": N3, "SPK": SPK, "dtype": dtype}))
    
    # test_silu_mul_quant_fuse configs
    silu_M = [1, 2, 4, 8, 256, 1024, 8192]
    silu_N = [128, 4096, 8192]
    for m in silu_M:
        for n in silu_N:
            configs.append(("silu_mul", {"M": m, "N": n, "dtype": torch.bfloat16}))
    
    return configs


def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


def config_label(name, params):
    parts = [name]
    for k, v in params.items():
        if k == "dtype":
            parts.append(f"{k}={'fp16' if v == torch.float16 else 'bf16'}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def config_tuple(name, params):
    """Return a hashable tuple for GEAK_SHAPES_USED."""
    items = []
    items.append(name)
    for k, v in params.items():
        if k == "dtype":
            items.append("fp16" if v == torch.float16 else "bf16")
        else:
            items.append(v)
    return tuple(items)


# ============================================================
# Data generation helpers
# ============================================================

def generate_fused_rms_quant_data(M, N1, N2, dtype=torch.bfloat16):
    x1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    x2 = torch.randn((M, N2), dtype=dtype, device="cuda") / 10
    w1 = torch.ones((N1,), dtype=torch.float32, device="cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cuda")
    res1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    return x1, w1, x2, w2, res1


def generate_fused_reduce_rms_quant_data(M, N1, N2, N3, SPK, dtype=torch.bfloat16):
    if SPK > 1:
        x1 = torch.randn((SPK, M, N1), dtype=torch.float32, device="cuda") / 10
        x2 = torch.randn((SPK, M, N2), dtype=torch.float32, device="cuda") / 10
        x3 = torch.randn((SPK, M, N3), dtype=torch.float32, device="cuda") / 10
    else:
        x1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
        x2 = torch.randn((M, N2), dtype=dtype, device="cuda") / 10
        x3 = None
    w1 = torch.ones((N1,), dtype=torch.float32, device="cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cuda")
    res1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    return x1, w1, x2, w2, res1, x3


# ============================================================
# Kernel invocation wrappers
# ============================================================

def run_kernel(name, params):
    """Run the kernel for a given config and return a callable for benchmarking."""
    torch.manual_seed(42)
    
    if name == "group_quant":
        M, N1, N2, dtype = params["M"], params["N1"], params["N2"], params["dtype"]
        group_size = 128
        x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)
        
        def fn():
            return fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size, dtype_quant=fp8_dtype,
                res1=res1, output_unquantized_inp1=True,
            )
        return fn
    
    elif name == "per_tensor":
        M, N1, N2, dtype = params["M"], params["N1"], params["N2"], params["dtype"]
        x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)
        scale = torch.randn(1, dtype=torch.float32, device="cuda")
        
        def fn():
            return fused_rms_fp8_per_tensor_static_quant(
                x1, w1, 1e-6, scale,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                dtype_quant=fp8_dtype, res1=res1,
                output_unquantized_inp1=True,
            )
        return fn
    
    elif name == "rmsnorm_quant":
        M, N = params["M"], params["N"]
        dtype = params["dtype"]
        x = torch.randn((M, N), dtype=dtype, device="cuda")
        w = torch.ones(N, dtype=dtype).cuda()
        eps = 0.0012
        
        DTYPE_MAX = torch.finfo(fp8_dtype).max
        rms_out = rmsnorm(x.to(torch.float32), w.to(torch.float32), eps)
        rms_out_abs_max = torch.max(torch.abs(rms_out))
        scale_val = rms_out_abs_max / DTYPE_MAX
        x_scale = scale_val.clone().detach().to(dtype=torch.float32, device="cuda")
        
        def fn():
            return fused_rms_fp8_per_tensor_static_quant(
                x, w, eps, x_scale,
                None, None, eps,
                dtype_quant=fp8_dtype, res1=None,
                output_unquantized_inp1=True,
                rmsnorm_convert_to_inp1_type=True,
            )
        return fn
    
    elif name == "reduce_rms":
        M, N1, N2, N3 = params["M"], params["N1"], params["N2"], params["N3"]
        SPK, dtype = params["SPK"], params["dtype"]
        group_size = 128
        x1, w1, x2, w2, res1, x3 = generate_fused_reduce_rms_quant_data(M, N1, N2, N3, SPK, dtype)
        
        def fn():
            return fused_reduce_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                inp3=x3, group_size=group_size,
                dtype_quant=fp8_dtype, dtype=dtype,
                res1=res1, output_unquantized_inp1=True,
            )
        return fn
    
    elif name == "silu_mul":
        M, N = params["M"], params["N"]
        dtype = params["dtype"]
        x = torch.randn((M, 2 * N), dtype=dtype, device="cuda")
        
        DTYPE_MAX = torch.finfo(fp8_dtype).max
        x1_half, x2_half = x.split([N, N], dim=-1)
        silu_out = (F.silu(x1_half.to(torch.float32)) * x2_half.to(torch.float32)).to(x.dtype).to(torch.float32)
        silu_out_abs_max = torch.max(torch.abs(silu_out))
        scale_val = silu_out_abs_max / DTYPE_MAX
        x_scale = scale_val.clone().detach().to(dtype=torch.float32, device="cuda")
        
        def fn():
            return fused_silu_mul_fp8_per_tensor_static_quant(
                x, x_scale, dtype_quant=fp8_dtype,
                silu_convert_to_inp_type=True,
            )
        return fn
    
    else:
        raise ValueError(f"Unknown config name: {name}")


# ============================================================
# Correctness checks
# ============================================================

def check_correctness(name, params):
    """Run correctness check for a given config. Returns True if passed."""
    torch.manual_seed(42)
    
    if name == "group_quant":
        M, N1, N2, dtype = params["M"], params["N1"], params["N2"], params["dtype"]
        group_size = 128
        x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)
        
        # Reference
        s = x1 + res1
        y1_ref = rmsnorm(s, w1, 1e-6)
        y2_ref = rmsnorm(x2, w2, 1e-6)
        y1_q_ref, y1_s_ref = per_token_fp8_group_quant(y1_ref, fp8_dtype, group_size)
        
        # Triton
        (y1_q_tri, y1_s_tri), y1_tri, y2_tri, y1_res_tri = fused_rms_fp8_group_quant(
            x1, w1, 1e-6,
            inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
            group_size=group_size, dtype_quant=fp8_dtype,
            res1=res1, output_unquantized_inp1=True,
        )
        
        torch.testing.assert_close(y1_ref.to(x1.dtype), y1_tri, atol=0.1, rtol=0.1)
        torch.testing.assert_close(y2_ref.to(x1.dtype), y2_tri, atol=0.1, rtol=0.1)
        torch.testing.assert_close(s.to(x1.dtype), y1_res_tri, atol=0.1, rtol=0.1)
        
        y1_up_ref = upcast(y1_q_ref, y1_s_ref, dtype=torch.float32, group_size=group_size)
        y1_up_tri = upcast(y1_q_tri, y1_s_tri, dtype=torch.float32, group_size=group_size)
        torch.testing.assert_close(y1_up_ref, y1_up_tri, atol=0.1, rtol=0.1)
        return True
    
    elif name == "per_tensor":
        M, N1, N2, dtype = params["M"], params["N1"], params["N2"], params["dtype"]
        x1, w1, x2, w2, res1 = generate_fused_rms_quant_data(M, N1, N2, dtype)
        scale = torch.randn(1, dtype=torch.float32, device="cuda")
        
        # Reference
        s = x1 + res1
        y1_ref = rmsnorm(s, w1, 1e-6)
        y2_ref = rmsnorm(x2, w2, 1e-6)
        y1_q_ref = per_tensor_fp8_static_quant(y1_ref, fp8_dtype, scale)
        
        # Triton
        y1_q_tri, y1_tri, y2_tri, y1_res_tri = fused_rms_fp8_per_tensor_static_quant(
            x1, w1, 1e-6, scale,
            inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
            dtype_quant=fp8_dtype, res1=res1,
            output_unquantized_inp1=True,
        )
        
        torch.testing.assert_close(y1_ref.to(x1.dtype), y1_tri, atol=0.1, rtol=0.1)
        torch.testing.assert_close(y2_ref.to(x1.dtype), y2_tri, atol=0.1, rtol=0.1)
        torch.testing.assert_close(s.to(x1.dtype), y1_res_tri, atol=0.1, rtol=0.1)
        
        y1_up_ref = y1_q_ref.to(torch.float32) * scale
        y1_up_tri = y1_q_tri.to(torch.float32) * scale
        torch.testing.assert_close(y1_up_ref, y1_up_tri, atol=0.1, rtol=0.1)
        return True
    
    elif name == "rmsnorm_quant":
        M, N = params["M"], params["N"]
        dtype = params["dtype"]
        x = torch.randn((M, N), dtype=dtype, device="cuda")
        w = torch.ones(N, dtype=dtype).cuda()
        eps = 0.0012
        
        DTYPE_MAX = torch.finfo(fp8_dtype).max
        rms_out = rmsnorm(x.to(torch.float32), w.to(torch.float32), eps)
        rms_out_abs_max = torch.max(torch.abs(rms_out))
        scale_val = rms_out_abs_max / DTYPE_MAX
        x_scale = scale_val.clone().detach().to(dtype=torch.float32, device="cuda")
        
        # Reference
        rms_ref = rmsnorm(x.to(torch.float32), w.to(torch.float32), eps).to(x.dtype)
        fp8_ref = per_tensor_fp8_static_quant(rms_ref.to(torch.float32), fp8_dtype, x_scale.to(torch.float32))
        
        # Triton
        fp8_tri, rms_tri, _, _ = fused_rms_fp8_per_tensor_static_quant(
            x, w, eps, x_scale,
            None, None, eps,
            dtype_quant=fp8_dtype, res1=None,
            output_unquantized_inp1=True,
            rmsnorm_convert_to_inp1_type=True,
        )
        
        # Use checkAllclose matching test file (allows 5% mismatch at atol=0.01, rtol=0.01)
        checkAllclose(rms_tri.to(torch.float32), rms_ref.to(torch.float32))
        checkAllclose(fp8_tri.to(torch.float32), fp8_ref.to(torch.float32))
        return True
    
    elif name == "reduce_rms":
        M, N1, N2, N3 = params["M"], params["N1"], params["N2"], params["N3"]
        SPK, dtype = params["SPK"], params["dtype"]
        group_size = 128
        x1, w1, x2, w2, res1, x3 = generate_fused_reduce_rms_quant_data(M, N1, N2, N3, SPK, dtype)
        
        # Reference
        out_dtype = dtype
        if x1.dim() == 3:
            x1_sum = torch.sum(x1, dim=0)
            x2_sum = torch.sum(x2, dim=0)
            x3_sum = torch.sum(x3, dim=0).to(out_dtype) if x3 is not None else None
        else:
            x1_sum = x1
            x2_sum = x2
            x3_sum = None
        s = x1_sum + res1
        y1_ref = rmsnorm(s, w1, 1e-6)
        y2_ref = rmsnorm(x2_sum, w2, 1e-6)
        y1_q_ref, y1_s_ref = per_token_fp8_group_quant(y1_ref, fp8_dtype, group_size)
        
        # Triton
        (y1_q_tri, y1_s_tri), y1_tri, y2_tri, y1_res_tri, y3_tri = fused_reduce_rms_fp8_group_quant(
            x1, w1, 1e-6,
            inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
            inp3=x3, group_size=group_size,
            dtype_quant=fp8_dtype, dtype=dtype,
            res1=res1, output_unquantized_inp1=True,
        )
        
        torch.testing.assert_close(y1_ref.to(out_dtype), y1_tri, atol=0.1, rtol=0.1)
        torch.testing.assert_close(y2_ref.to(out_dtype), y2_tri, atol=0.1, rtol=0.1)
        
        y1_up_ref = upcast(y1_q_ref, y1_s_ref, dtype=torch.float32, group_size=group_size)
        y1_up_tri = upcast(y1_q_tri, y1_s_tri, dtype=torch.float32, group_size=group_size)
        torch.testing.assert_close(y1_up_ref, y1_up_tri, atol=0.1, rtol=0.1)
        
        if x3_sum is not None and y3_tri is not None:
            torch.testing.assert_close(x3_sum, y3_tri, atol=0.1, rtol=0.1)
        return True
    
    elif name == "silu_mul":
        M, N = params["M"], params["N"]
        dtype = params["dtype"]
        x = torch.randn((M, 2 * N), dtype=dtype, device="cuda")
        
        DTYPE_MAX = torch.finfo(fp8_dtype).max
        x1_half, x2_half = x.split([N, N], dim=-1)
        silu_out = (F.silu(x1_half.to(torch.float32)) * x2_half.to(torch.float32)).to(x.dtype).to(torch.float32)
        silu_out_abs_max = torch.max(torch.abs(silu_out))
        scale_val = silu_out_abs_max / DTYPE_MAX
        x_scale = scale_val.clone().detach().to(dtype=torch.float32, device="cuda")
        
        # Reference
        fp8_ref = per_tensor_fp8_static_quant(silu_out, fp8_dtype, x_scale)
        
        # Triton
        fp8_tri = fused_silu_mul_fp8_per_tensor_static_quant(
            x, x_scale, dtype_quant=fp8_dtype,
            silu_convert_to_inp_type=True,
        )
        
        # Use checkAllclose matching test file (allows 5% mismatch)
        checkAllclose(fp8_tri.to(torch.float32), fp8_ref.to(torch.float32))
        return True
    
    else:
        raise ValueError(f"Unknown config name: {name}")


# ============================================================
# Benchmark timing
# ============================================================

def benchmark_config(name, params):
    """Benchmark a single config. Returns median latency in ms."""
    fn = run_kernel(name, params)
    
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
    median = times[len(times) // 2]
    return median


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    
    all_configs = build_all_configs()
    
    if args.correctness:
        configs = _pick(all_configs, 25)
        print(f"Running correctness on {len(configs)} configs...")
        for i, (name, params) in enumerate(configs):
            label = config_label(name, params)
            try:
                check_correctness(name, params)
                print(f"  [{i+1}/{len(configs)}] PASS: {label}")
            except Exception as e:
                print(f"  [{i+1}/{len(configs)}] FAIL: {label}: {e}")
                shapes_used = sorted([config_tuple(n, p) for n, p in configs])
                print(f"GEAK_SHAPES_USED={shapes_used}")
                sys.exit(1)
        shapes_used = sorted([config_tuple(n, p) for n, p in configs])
        print(f"GEAK_SHAPES_USED={shapes_used}")
        print("All correctness checks passed.")
    
    elif args.benchmark:
        configs = _pick(all_configs, 25)
        print(f"Running benchmark on {len(configs)} configs...")
        latencies = []
        for i, (name, params) in enumerate(configs):
            label = config_label(name, params)
            lat = benchmark_config(name, params)
            latencies.append(lat)
            print(f"  {label}  {lat:.4f}ms")
        
        # Geometric mean
        log_sum = sum(math.log(l) for l in latencies)
        geo_mean = math.exp(log_sum / len(latencies))
        
        shapes_used = sorted([config_tuple(n, p) for n, p in configs])
        print(f"GEAK_SHAPES_USED={shapes_used}")
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")
    
    elif args.full_benchmark:
        configs = all_configs
        print(f"Running full benchmark on {len(configs)} configs...")
        latencies = []
        for i, (name, params) in enumerate(configs):
            label = config_label(name, params)
            lat = benchmark_config(name, params)
            latencies.append(lat)
            print(f"  {label}  {lat:.4f}ms")
        
        # Geometric mean
        log_sum = sum(math.log(l) for l in latencies)
        geo_mean = math.exp(log_sum / len(latencies))
        
        shapes_used = sorted([config_tuple(n, p) for n, p in configs])
        print(f"GEAK_SHAPES_USED={shapes_used}")
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")
    
    elif args.profile:
        configs = _pick(all_configs, 5)
        print(f"Running profile on {len(configs)} configs...")
        for i, (name, params) in enumerate(configs):
            label = config_label(name, params)
            fn = run_kernel(name, params)
            # Warmup
            for _ in range(WARMUP):
                fn()
            torch.cuda.synchronize()
            # Run once for profiling
            fn()
            torch.cuda.synchronize()
            print(f"  [{i+1}/{len(configs)}] {label}")
        
        shapes_used = sorted([config_tuple(n, p) for n, p in configs])
        print(f"GEAK_SHAPES_USED={shapes_used}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
