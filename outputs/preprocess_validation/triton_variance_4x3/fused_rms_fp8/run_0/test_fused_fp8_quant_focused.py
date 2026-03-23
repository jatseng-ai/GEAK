#!/usr/bin/env python3
import sys
import torch
import torch.nn.functional as F

# Add kernel path
sys.path.insert(0, '/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf')

try:
    from aiter.ops.triton.quant.fused_fp8_quant import (
        fused_rms_fp8_per_tensor_static_quant,
        fused_rms_fp8_group_quant,
        fused_flatten_fp8_group_quant,
        fused_reduce_act_mul_fp8_group_quant,
        fused_reduce_rms_fp8_group_quant,
        fused_silu_mul_fp8_per_tensor_static_quant,
    )
    import aiter
except ImportError as e:
    print(f"FAIL: Import error - {e}")
    sys.exit(1)

torch.manual_seed(42)

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

def test_fused_rms_fp8_per_tensor_static_quant():
    """Test fused_rms_fp8_per_tensor_static_quant function"""
    M, N1, N2 = 32, 128, 128
    dtype = torch.bfloat16
    dtype_quant = aiter.dtypes.fp8
    
    x1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    x2 = torch.randn((M, N2), dtype=dtype, device="cuda") / 10
    w1 = torch.ones((N1,), dtype=torch.float32, device="cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cuda")
    res1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    scale = torch.randn(1, dtype=torch.float32, device="cuda")
    
    # Torch reference
    s = x1 + res1
    y1_torch = rmsnorm(s, w1, 1e-6)
    y2_torch = rmsnorm(x2, w2, 1e-6)
    y1_q_torch = per_tensor_fp8_static_quant(y1_torch, dtype_quant, scale)
    
    # Triton kernel
    y1_q_triton, y1_triton, y2_triton, y1_res_triton = fused_rms_fp8_per_tensor_static_quant(
        x1, w1, 1e-6, scale,
        inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
        dtype_quant=dtype_quant, res1=res1, output_unquantized_inp1=True,
    )
    
    # Validate
    y1_upcast_torch = y1_q_torch.to(torch.float32) * scale
    y1_upcast_triton = y1_q_triton.to(torch.float32) * scale
    
    try:
        torch.testing.assert_close(y1_upcast_torch, y1_upcast_triton, atol=0.1, rtol=0.1)
        torch.testing.assert_close(y1_torch, y1_triton, atol=0.1, rtol=0.1)
        torch.testing.assert_close(y2_torch, y2_triton, atol=0.1, rtol=0.1)
        return True
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        return False

def test_fused_rms_fp8_group_quant():
    """Test fused_rms_fp8_group_quant function"""
    M, N1, N2 = 32, 128, 128
    dtype = torch.bfloat16
    dtype_quant = aiter.dtypes.fp8
    group_size = 128
    
    x1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    x2 = torch.randn((M, N2), dtype=dtype, device="cuda") / 10
    w1 = torch.ones((N1,), dtype=torch.float32, device="cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cuda")
    res1 = torch.randn((M, N1), dtype=dtype, device="cuda") / 10
    
    # Torch reference
    s = x1 + res1
    y1_torch = rmsnorm(s, w1, 1e-6)
    y2_torch = rmsnorm(x2, w2, 1e-6)
    y1_q_torch, y1_s_torch = per_token_fp8_group_quant(y1_torch, dtype_quant, group_size)
    
    # Triton kernel
    (y1_q_triton, y1_s_triton), y1_triton, y2_triton, y1_res_triton = fused_rms_fp8_group_quant(
        x1, w1, 1e-6,
        inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
        group_size=group_size, dtype_quant=dtype_quant,
        res1=res1, output_unquantized_inp1=True,
    )
    
    # Validate
    try:
        torch.testing.assert_close(y1_torch, y1_triton, atol=0.1, rtol=0.1)
        torch.testing.assert_close(y2_torch, y2_triton, atol=0.1, rtol=0.1)
        return True
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        return False

def test_fused_silu_mul_fp8_per_tensor_static_quant():
    """Test fused_silu_mul_fp8_per_tensor_static_quant function"""
    M, N = 32, 128
    dtype = torch.bfloat16
    dtype_quant = aiter.dtypes.fp8
    
    inp = torch.randn((M, 2 * N), dtype=dtype, device="cuda") / 10
    scale = torch.randn(1, dtype=torch.float32, device="cuda")
    
    # Triton kernel
    try:
        out_fp8 = fused_silu_mul_fp8_per_tensor_static_quant(
            inp, scale, dtype_quant=dtype_quant
        )
        assert out_fp8.shape == (M, N)
        assert out_fp8.dtype == dtype_quant
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        sys.exit(0)
    
    tests = [
        ("fused_rms_fp8_per_tensor_static_quant", test_fused_rms_fp8_per_tensor_static_quant),
        ("fused_rms_fp8_group_quant", test_fused_rms_fp8_group_quant),
        ("fused_silu_mul_fp8_per_tensor_static_quant", test_fused_silu_mul_fp8_per_tensor_static_quant),
    ]
    
    all_passed = True
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                print(f"PASS: {name}")
            else:
                print(f"FAIL: {name}")
                all_passed = False
        except Exception as e:
            print(f"FAIL: {name} - {e}")
            all_passed = False
    
    if all_passed:
        print("\nAll tests PASSED")
        sys.exit(0)
    else:
        print("\nSome tests FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
