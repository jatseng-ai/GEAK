#!/usr/bin/env python3
import sys
import torch

# Add the kernel path to sys.path
sys.path.insert(0, '/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf')

from aiter.ops.triton.rope.fused_qkv_split_qk_rope import fused_qkv_split_qk_rope

# Helper functions from the test file
class RotateStyle:
    GPTJ = 0
    NEOX = 1

def generate_rope_cached_freqs(B, max_embed_positions, D, dtype):
    """Generate RoPE cached frequencies."""
    pos = torch.arange(B, dtype=torch.long, device='cuda').unsqueeze(-1)
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32, device='cuda') / D))
    freqs = torch.outer(torch.arange(max_embed_positions, dtype=torch.float32, device='cuda'), inv_freq)
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return pos, freqs, cos, sin

def ref_rope_sbhd_fwd(x, freqs, rotate_style, reuse_freqs_front_part, nope_first):
    """Reference RoPE implementation."""
    B, H, D = x.shape
    
    if reuse_freqs_front_part:
        # Only apply RoPE to first half of dimensions
        rope_dim = D // 2
        x_rope = x[..., :rope_dim]
        x_nope = x[..., rope_dim:]
    else:
        rope_dim = D
        x_rope = x
        x_nope = None
    
    # Get cos and sin for the rope dimensions
    cos = freqs.cos()[..., :rope_dim // 2]
    sin = freqs.sin()[..., :rope_dim // 2]
    
    if rotate_style == RotateStyle.NEOX:
        # Split into two halves and rotate
        x1 = x_rope[..., :rope_dim // 2]
        x2 = x_rope[..., rope_dim // 2:rope_dim]
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
    else:  # GPTJ
        # Interleaved rotation
        x1 = x_rope[..., 0::2]
        x2 = x_rope[..., 1::2]
        rotated_pairs = torch.stack([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
        rotated = rotated_pairs.flatten(start_dim=-2)
    
    if x_nope is not None:
        if nope_first:
            return torch.cat([x_nope, rotated], dim=-1)
        else:
            return torch.cat([rotated, x_nope], dim=-1)
    else:
        return rotated

def run_torch(qkv, QH_PER_KH, KH, D, ref_freqs, reuse_freqs_front_part, nope, nope_first, rotate_style):
    """Reference torch implementation."""
    q_size = QH_PER_KH * KH * D
    kv_size = KH * D
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(-1, QH_PER_KH * KH, D).contiguous()
    k = k.view(-1, KH, D).contiguous()
    v = v.view(-1, KH, D).contiguous()

    q = ref_rope_sbhd_fwd(
        q,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    k = ref_rope_sbhd_fwd(
        k,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )

    return q, k, v

def test_case(B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype):
    """Run a single test case."""
    # Generate inputs
    qkv = torch.randn(
        (B, (QH_PER_KH * KH + 2 * KH) * (D * (2 if nope else 1))),
        dtype=dtype,
        device="cuda",
    )
    
    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, (D // 2) if reuse_freqs_front_part else D, dtype
    )
    ref_freqs = freqs[pos].squeeze(-2)

    # Run triton kernel
    q_triton, k_triton, v_triton = fused_qkv_split_qk_rope(
        qkv,
        cos,
        sin,
        pos,
        QH_PER_KH * KH,
        KH,
        (D * (2 if nope else 1)),
        is_neox=(rotate_style == RotateStyle.NEOX),
        offsets=None,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    
    # Run torch reference
    q_torch, k_torch, v_torch = run_torch(
        qkv,
        QH_PER_KH,
        KH,
        (D * (2 if nope else 1)),
        ref_freqs,
        reuse_freqs_front_part,
        nope,
        nope_first,
        rotate_style,
    )

    # Compare results
    torch.testing.assert_close(q_torch, q_triton, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k_torch, k_triton, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_torch, v_triton, rtol=1e-3, atol=1e-3)

def main():
    torch.manual_seed(42)
    
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0
    
    try:
        # Test a few representative cases
        test_cases = [
            # B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype
            (4, 8, 4, 64, RotateStyle.NEOX, 131072, False, False, False, torch.bfloat16),
            (8, 4, 4, 128, RotateStyle.GPTJ, 131072, False, False, True, torch.bfloat16),
            (16, 2, 1, 64, RotateStyle.NEOX, 131072, True, False, False, torch.bfloat16),
            (32, 1, 4, 128, RotateStyle.NEOX, 131072, True, True, True, torch.bfloat16),
        ]
        
        for i, params in enumerate(test_cases):
            print(f"Running test case {i+1}/{len(test_cases)}...", end=" ")
            test_case(*params)
            print("PASS")
        
        print("\nAll tests PASSED")
        return 0
        
    except Exception as e:
        print(f"\nFAIL: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
