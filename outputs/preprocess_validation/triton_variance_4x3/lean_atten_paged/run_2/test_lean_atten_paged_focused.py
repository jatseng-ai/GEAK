#!/usr/bin/env python3
"""Focused test for lean_atten_paged kernel."""

import sys
import torch
import traceback

# Add the kernel path to sys.path
sys.path.insert(0, '/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf')

try:
    from aiter.ops.triton.attention.lean_atten_paged import persistent_lean_attention_paged
    from aiter.ops.triton.attention.lean_atten_paged import get_num_splits_and_buffer_sizes
except ImportError as e:
    print(f"FAIL: Could not import kernel: {e}")
    sys.exit(1)

def test_lean_atten_paged():
    """Test lean_atten_paged kernel with basic inputs."""
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 2
    num_heads = 4
    head_dim = 64
    seq_len_q = 1  # Decode phase typically has seq_len_q=1
    max_seq_len_k = 128
    BLOCK_M = 1
    BLOCK_N = 64
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("SKIP: CUDA not available, lean_atten_paged requires GPU")
        sys.exit(0)
    
    # Create input tensors
    # Q: [num_heads, batch_size * seq_len_q, head_dim]
    q = torch.randn(num_heads, batch_size * seq_len_q, head_dim, device=device, dtype=torch.float16)
    
    # For paged attention, K and V are stored in blocks
    # K, V: [num_heads, total_kv_blocks * BLOCK_N, head_dim]
    num_blocks_per_seq = (max_seq_len_k + BLOCK_N - 1) // BLOCK_N
    total_kv_blocks = batch_size * num_blocks_per_seq
    
    k = torch.randn(num_heads, total_kv_blocks * BLOCK_N, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(num_heads, total_kv_blocks * BLOCK_N, head_dim, device=device, dtype=torch.float16)
    
    # KV block tables: [batch_size, max_num_blocks_per_seq]
    kv_block_tables = torch.zeros(batch_size, num_blocks_per_seq, device=device, dtype=torch.int32)
    for i in range(batch_size):
        kv_block_tables[i] = torch.arange(i * num_blocks_per_seq, (i + 1) * num_blocks_per_seq, device=device, dtype=torch.int32)
    
    # batch_num_block_n: number of blocks per sequence in batch
    batch_num_block_n = torch.full((batch_size,), num_blocks_per_seq, device=device, dtype=torch.int32)
    
    # Calculate total programs and buffer sizes
    num_SMs = 120  # Typical for MI300X, adjust as needed
    (
        num_m_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        total_programs,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        seq_len_q, max_seq_len_k, num_heads, num_heads, head_dim, BLOCK_M, BLOCK_N, num_SMs
    )
    
    # Intermediate buffers for lean attention
    # Mp, Lp: [num_splits, num_heads, batch_size * seq_len_q, head_dim]
    Mp = torch.zeros(num_splits, num_heads, batch_size * seq_len_q, head_dim, device=device, dtype=torch.float32)
    Lp = torch.zeros(num_splits, num_heads, batch_size * seq_len_q, device=device, dtype=torch.float32)
    Op = torch.zeros(num_splits, num_heads, batch_size * seq_len_q, head_dim, device=device, dtype=torch.float32)
    
    # Locks for synchronization
    locks = torch.zeros(total_programs, device=device, dtype=torch.int32)
    
    # Softmax scale
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    num_warps = 4
    waves_per_eu = 1
    
    try:
        # Run the kernel
        output = persistent_lean_attention_paged(
            q=q,
            k=k,
            v=v,
            kv_block_tables=kv_block_tables,
            Mp=Mp,
            Lp=Lp,
            Op=Op,
            locks=locks,
            batch_num_block_n=batch_num_block_n,
            total_programs=total_programs,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            batch_size=batch_size,
            sm_scale=torch.tensor(sm_scale, dtype=torch.float16),
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
        )
        
        # Basic validation
        assert output.shape == q.shape, f"Output shape mismatch: {output.shape} vs {q.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        # Compute reference using standard attention (simplified for decode)
        # Reshape for standard attention computation
        q_ref = q.view(num_heads, batch_size, seq_len_q, head_dim).transpose(0, 1)  # [batch, heads, seq_q, dim]
        
        # For reference, we'll use a simple attention without paging
        k_ref = k[:, :max_seq_len_k, :].view(num_heads, batch_size, max_seq_len_k, head_dim).transpose(0, 1)
        v_ref = v[:, :max_seq_len_k, :].view(num_heads, batch_size, max_seq_len_k, head_dim).transpose(0, 1)
        
        # Compute attention scores
        scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * sm_scale
        attn_weights = torch.softmax(scores, dim=-1)
        output_ref = torch.matmul(attn_weights, v_ref)
        
        # Reshape reference output to match kernel output
        output_ref = output_ref.transpose(0, 1).reshape(num_heads, batch_size * seq_len_q, head_dim)
        
        # Compare outputs with tolerance
        rtol, atol = 1e-2, 1e-2  # Relaxed tolerance for FP16
        if torch.allclose(output, output_ref, rtol=rtol, atol=atol):
            print("PASS: lean_atten_paged output matches reference")
            return True
        else:
            max_diff = torch.max(torch.abs(output - output_ref)).item()
            print(f"FAIL: Output mismatch, max diff: {max_diff}")
            print(f"  Output range: [{output.min().item()}, {output.max().item()}]")
            print(f"  Reference range: [{output_ref.min().item()}, {output_ref.max().item()}]")
            return False
            
    except Exception as e:
        print(f"FAIL: Exception during kernel execution: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_lean_atten_paged()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
