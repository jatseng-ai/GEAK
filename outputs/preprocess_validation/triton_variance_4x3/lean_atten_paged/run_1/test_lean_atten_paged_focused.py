#!/usr/bin/env python3
"""Focused test for lean_atten_paged kernel."""

import sys
import torch
import traceback

# Set random seed for reproducibility
torch.manual_seed(42)

def test_lean_atten_paged():
    """Test the lean_atten_paged kernel function."""
    try:
        # Import the kernel function
        from aiter.ops.triton.attention.lean_atten_paged import (
            persistent_lean_attention_paged,
            get_num_splits_and_buffer_sizes,
        )
        
        print("Successfully imported lean_atten_paged kernel")
        
        # Set up test parameters
        batch_size = 2
        num_heads = 8
        head_dim = 64
        seq_len_q = 1  # Decode phase typically has seq_len_q = 1
        max_seq_len_k = 128
        block_size = 16
        BLOCK_M = 1
        BLOCK_N = 16
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("SKIP: CUDA not available, lean_atten_paged requires GPU")
            return True
        
        dtype = torch.float16
        
        # Create input tensors
        # Q: [num_heads, batch_size * seq_len_q, head_dim]
        q = torch.randn(num_heads, batch_size * seq_len_q, head_dim, device=device, dtype=dtype)
        
        # For paged attention, K and V are stored in blocks
        # K, V: [num_heads, num_blocks * block_size, head_dim]
        num_blocks_per_seq = (max_seq_len_k + block_size - 1) // block_size
        total_blocks = batch_size * num_blocks_per_seq
        
        k = torch.randn(num_heads, total_blocks * block_size, head_dim, device=device, dtype=dtype)
        v = torch.randn(num_heads, total_blocks * block_size, head_dim, device=device, dtype=dtype)
        
        # Block tables: [batch_size, max_num_blocks_per_seq]
        kv_block_tables = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(batch_size, num_blocks_per_seq)
        
        # batch_num_block_n: number of blocks per sequence in batch
        batch_num_block_n = torch.full((batch_size,), num_blocks_per_seq, device=device, dtype=torch.int32)
        
        # Calculate number of programs and buffer sizes
        num_SMs = 110  # Typical for MI300X, adjust as needed
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
        
        # Create intermediate buffers
        # Mp, Lp: [num_heads, num_splits, batch_size * seq_len_q]
        Mp = torch.zeros(num_heads, num_splits, batch_size * seq_len_q, device=device, dtype=torch.float32)
        Lp = torch.zeros(num_heads, num_splits, batch_size * seq_len_q, device=device, dtype=torch.float32)
        
        # Op: [num_heads, num_splits, batch_size * seq_len_q, head_dim]
        Op = torch.zeros(num_heads, num_splits, batch_size * seq_len_q, head_dim, device=device, dtype=torch.float32)
        
        # Locks for synchronization
        locks = torch.zeros(total_programs, device=device, dtype=torch.int32)
        
        # Scale factor
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Kernel parameters
        num_warps = 4
        waves_per_eu = 1
        
        print(f"Running lean_atten_paged with:")
        print(f"  batch_size={batch_size}, num_heads={num_heads}, head_dim={head_dim}")
        print(f"  seq_len_q={seq_len_q}, max_seq_len_k={max_seq_len_k}")
        print(f"  BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
        print(f"  total_programs={total_programs}, num_splits={num_splits}")
        
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
            sm_scale=sm_scale,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
        )
        
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        
        # Basic sanity checks
        assert output.shape == q.shape, f"Output shape mismatch: {output.shape} vs {q.shape}"
        assert output.dtype == v.dtype, f"Output dtype mismatch: {output.dtype} vs {v.dtype}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        # Check output is non-zero (attention should produce meaningful results)
        assert output.abs().max() > 0, "Output is all zeros"
        
        print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        print("PASS: lean_atten_paged kernel executed successfully")
        return True
        
    except Exception as e:
        print(f"FAIL: lean_atten_paged test failed with error:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_lean_atten_paged()
    sys.exit(0 if success else 1)
