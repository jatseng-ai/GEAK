#!/usr/bin/env python3
import sys
import torch

# Set seed for reproducibility
torch.manual_seed(42)

# Import the kernel function
from aiter.ops.triton.rope.fused_qkv_split_qk_rope import fused_qkv_split_qk_rope

# Import helper functions from test dependencies
try:
    from op_tests.triton_tests.fusions.test_fused_qk_concat import generate_rope_cached_freqs
    from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle
except ImportError:
    # Fallback implementations if imports fail
    class RotateStyle:
        GPTJ = 0
        NEOX = 1
    
    def generate_rope_cached_freqs(B, max_embed_positions, D, dtype):
        """Generate RoPE cached frequencies."""
        pos = torch.arange(B, dtype=torch.long, device='cuda').unsqueeze(-1)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float32, device='cuda') / D))
        freqs = torch.outer(torch.arange(max_embed_positions, dtype=torch.float32, device='cuda'), inv_freq)
        freqs = freqs.unsqueeze(0).unsqueeze(0)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return pos, freqs, cos, sin
    
    def ref_rope_sbhd_fwd(x, freqs, rotate_style, reuse_freqs_front_part, nope_first):
        """Reference RoPE forward implementation."""
        B, H, D = x.shape
        
        if reuse_freqs_front_part:
            rope_dim = D // 2
        else:
            rope_dim = D
        
        if nope_first:
            rope_x = x[:, :, rope_dim:]
            nope_x = x[:, :, :rope_dim]
        else:
            rope_x = x[:, :, :rope_dim]
            nope_x = x[:, :, rope_dim:] if reuse_freqs_front_part else None
        
        # Apply RoPE
        rope_dim_half = rope_dim // 2
        x1 = rope_x[:, :, :rope_dim_half]
        x2 = rope_x[:, :, rope_dim_half:rope_dim]
        
        cos_freq = freqs[:, :rope_dim_half].cos()
        sin_freq = freqs[:, :rope_dim_half].sin()
        
        if rotate_style == RotateStyle.NEOX:
            rotated = torch.cat([
                x1 * cos_freq - x2 * sin_freq,
                x2 * cos_freq + x1 * sin_freq
            ], dim=-1)
        else:  # GPTJ
            rotated = torch.cat([
                x1 * cos_freq - x2 * sin_freq,
                x1 * sin_freq + x2 * cos_freq
            ], dim=-1)
        
        if nope_first:
            return torch.cat([nope_x, rotated], dim=-1) if nope_x is not None else rotated
        else:
            return torch.cat([rotated, nope_x], dim=-1) if nope_x is not None else rotated

def generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype):
    """Generate QKV input tensor."""
    qkv = torch.randn(
        (B, (QH_PER_KH * KH + 2 * KH) * (D * (2 if nope else 1))),
        dtype=dtype,
        device='cuda',
    )
    return qkv

def run_torch(qkv, QH_PER_KH, KH, D, ref_freqs, reuse_freqs_front_part, nope, nope_first, rotate_style):
    """Run torch reference implementation."""
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

def test_fused_qkv_split_qk_rope():
    """Test fused_qkv_split_qk_rope kernel."""
    # Test parameters
    B = 8
    QH_PER_KH = 4
    KH = 4
    D = 64
    rotate_style = RotateStyle.NEOX
    max_embed_positions = 131072
    nope = False
    nope_first = False
    reuse_freqs_front_part = True
    dtype = torch.bfloat16

    print(f"Testing with B={B}, QH_PER_KH={QH_PER_KH}, KH={KH}, D={D}")
    print(f"rotate_style={rotate_style}, nope={nope}, nope_first={nope_first}")
    print(f"reuse_freqs_front_part={reuse_freqs_front_part}")

    # Generate inputs
    qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype)
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

    # Validate results
    try:
        torch.testing.assert_close(q_torch, q_triton, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k_torch, k_triton, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v_torch, v_triton, rtol=1e-2, atol=1e-2)
        print("PASS: All outputs match reference implementation")
        return True
    except AssertionError as e:
        print(f"FAIL: Output mismatch - {e}")
        return False

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        sys.exit(0)
    
    try:
        success = test_fused_qkv_split_qk_rope()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"FAIL: Exception occurred - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
