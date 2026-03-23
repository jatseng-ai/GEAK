#!/usr/bin/env python3
import sys
import torch

# Add the kernel path to sys.path
sys.path.insert(0, '/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf')

from aiter.ops.triton.rope.fused_qkv_split_qk_rope import fused_qkv_split_qk_rope

# Helper functions from the test file
def generate_rope_cached_freqs(B, max_embed_positions, D, dtype):
    """Generate RoPE cached frequencies."""
    pos = torch.arange(B, dtype=torch.long, device='cuda').unsqueeze(-1)
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32, device='cuda') / D))
    freqs = torch.outer(torch.arange(max_embed_positions, dtype=torch.float32, device='cuda'), inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return pos, freqs, cos, sin

def ref_rope_sbhd_fwd(x, freqs, rotate_style, reuse_freqs_front_part, nope_first):
    """Reference RoPE implementation."""
    B, H, D = x.shape
    
    if reuse_freqs_front_part:
        # Only apply RoPE to first half of head_dim
        rope_dim = D // 2
        x_rope = x[..., :rope_dim]
        x_nope = x[..., rope_dim:]
    else:
        rope_dim = D
        x_rope = x
        x_nope = None
    
    # Apply rotation
    if rotate_style == 0:  # GPTJ
        x1 = x_rope[..., :rope_dim//2]
        x2 = x_rope[..., rope_dim//2:]
        cos_freq = freqs[..., :rope_dim//2].cos().unsqueeze(1)
        sin_freq = freqs[..., :rope_dim//2].sin().unsqueeze(1)
        x_rope_out = torch.cat([x1 * cos_freq - x2 * sin_freq, x2 * cos_freq + x1 * sin_freq], dim=-1)
    else:  # NEOX
        x1 = x_rope[..., 0::2]
        x2 = x_rope[..., 1::2]
        cos_freq = freqs[..., :rope_dim//2].cos().unsqueeze(1)
        sin_freq = freqs[..., :rope_dim//2].sin().unsqueeze(1)
        x_rope_out = torch.empty_like(x_rope)
        x_rope_out[..., 0::2] = x1 * cos_freq - x2 * sin_freq
        x_rope_out[..., 1::2] = x2 * cos_freq + x1 * sin_freq
    
    if x_nope is not None:
        if nope_first:
            return torch.cat([x_nope, x_rope_out], dim=-1)
        else:
            return torch.cat([x_rope_out, x_nope], dim=-1)
    return x_rope_out

def run_torch(qkv, QH_PER_KH, KH, D, ref_freqs, reuse_freqs_front_part, nope, nope_first, rotate_style):
    """Torch reference implementation."""
    q_size = QH_PER_KH * KH * D
    kv_size = KH * D
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(-1, QH_PER_KH * KH, D).contiguous()
    k = k.view(-1, KH, D).contiguous()
    v = v.view(-1, KH, D).contiguous()
    
    q = ref_rope_sbhd_fwd(q, ref_freqs, rotate_style, reuse_freqs_front_part, nope_first)
    k = ref_rope_sbhd_fwd(k, ref_freqs, rotate_style, reuse_freqs_front_part, nope_first)
    
    return q, k, v

def test_basic():
    """Test basic functionality with simple parameters."""
    torch.manual_seed(42)
    
    B = 8
    QH_PER_KH = 4
    KH = 2
    D = 64
    rotate_style = 1  # NEOX
    max_embed_positions = 131072
    nope = False
    nope_first = False
    reuse_freqs_front_part = True
    dtype = torch.bfloat16
    
    # Generate inputs
    qkv = torch.randn(
        (B, (QH_PER_KH * KH + 2 * KH) * (D * (2 if nope else 1))),
        dtype=dtype,
        device='cuda',
    )
    
    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, (D // 2) if reuse_freqs_front_part else D, dtype
    )
    ref_freqs = freqs[pos].squeeze(-2)
    
    # Run kernel
    q_triton, k_triton, v_triton = fused_qkv_split_qk_rope(
        qkv,
        cos,
        sin,
        pos,
        QH_PER_KH * KH,
        KH,
        (D * (2 if nope else 1)),
        is_neox=(rotate_style == 1),
        offsets=None,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    
    # Run reference
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
    
    # Validate
    try:
        torch.testing.assert_close(q_torch, q_triton, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k_torch, k_triton, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v_torch, v_triton, rtol=1e-2, atol=1e-2)
        return True
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        return False

def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0
    
    print("Testing fused_qkv_split_qk_rope...")
    
    try:
        if test_basic():
            print("PASS: Basic test passed")
            return 0
        else:
            print("FAIL: Basic test failed")
            return 1
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
