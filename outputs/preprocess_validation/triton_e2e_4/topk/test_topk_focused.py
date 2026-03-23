#!/usr/bin/env python3
import sys
import torch

# Add the kernel path to sys.path if needed
# sys.path.insert(0, '/home/upandey/GEAK-v3-codebase-reorganization/outputs/preprocess_validation/triton_e2e_4/topk/.geak_resolved/ROCm_aiter/main-e15b75d464bf')

from aiter.ops.triton.topk import topk as triton_topk

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _to_cpu(res: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Move `res` to CPU so it matches `ref`'s device."""
    if res.device.type != "cpu":
        res = res.cpu()
    return res

def _assert_close(
    res: torch.Tensor,
    ref: torch.Tensor,
    dtype: torch.dtype,
    *,
    equal_nan: bool = False,
    reduce_dim: int = 1,
) -> None:
    RESOLUTION = {
        torch.float16: 1e-3,
        torch.float32: 1.3e-6,
        torch.bfloat16: 0.016,
    }
    res = _to_cpu(res, ref)
    assert res.dtype == dtype, f"dtype mismatch: {res.dtype} vs {dtype}"
    ref = ref.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol, equal_nan=equal_nan)

def _assert_equal(
    res: torch.Tensor, ref: torch.Tensor, *, equal_nan: bool = False
) -> None:
    res = _to_cpu(res, ref)
    torch.testing.assert_close(res, ref, atol=0, rtol=0, equal_nan=equal_nan)

def test_topk_small():
    """Test topk with small input (1-stage kernel path)."""
    torch.manual_seed(42)
    batch_size = 4
    hiddensize = 128
    k = 8
    dtype = torch.float32
    
    x = torch.arange(hiddensize, dtype=dtype, device=DEVICE).repeat(batch_size, 1)
    for b in range(batch_size):
        x[b] = x[b, torch.randperm(hiddensize, device=DEVICE)]
    
    ref_value, ref_index = torch.topk(x, k, largest=True)
    res_value, res_index = triton_topk(x, k, largest=True)
    
    _assert_close(res_value, ref_value.cpu(), dtype)
    _assert_equal(res_index.cpu(), ref_index.cpu())
    print("PASS: test_topk_small")

def test_topk_large():
    """Test topk with large input (2-stage kernel path)."""
    torch.manual_seed(42)
    batch_size = 2
    hiddensize = 128256
    k = 8
    dtype = torch.float32
    
    x = torch.arange(hiddensize, dtype=dtype, device=DEVICE).repeat(batch_size, 1)
    for b in range(batch_size):
        x[b] = x[b, torch.randperm(hiddensize, device=DEVICE)]
    
    ref_value, ref_index = torch.topk(x, k, largest=True)
    res_value, res_index = triton_topk(x, k, largest=True)
    
    _assert_close(res_value, ref_value.cpu(), dtype)
    _assert_equal(res_index.cpu(), ref_index.cpu())
    print("PASS: test_topk_large")

def test_topk_edge_case():
    """Test topk with edge case: k=2, batch_size=1."""
    torch.manual_seed(42)
    batch_size = 1
    hiddensize = 16
    k = 2
    dtype = torch.float32
    
    x = torch.randn(batch_size, hiddensize, dtype=dtype, device=DEVICE)
    
    ref_value, ref_index = torch.topk(x, k, largest=True)
    res_value, res_index = triton_topk(x, k, largest=True)
    
    _assert_close(res_value, ref_value.cpu(), dtype)
    _assert_equal(res_index.cpu(), ref_index.cpu())
    print("PASS: test_topk_edge_case")

def main():
    try:
        if DEVICE == "cpu":
            print("WARNING: CUDA not available, tests may not run correctly")
            return 1
        
        test_topk_small()
        test_topk_large()
        test_topk_edge_case()
        
        print("\n=== ALL TESTS PASSED ===")
        return 0
    except Exception as e:
        print(f"\n=== TEST FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
