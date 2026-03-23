#!/usr/bin/env python3
"""
Test harness for lean_atten_paged kernel.
Modes: --correctness, --benchmark, --full-benchmark, --profile
"""

import os
import sys
import argparse
import math
import random

# Resolve repo root
REPO_ROOT = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get(
        "GEAK_REPO_ROOT",
        "/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf"
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from aiter.ops.triton.attention.lean_atten_paged import persistent_lean_attention_paged

# ─── Fixed constants ───
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ─── Config list from benchmark file ───
# Source: op_tests/op_benchmarks/triton/bench_la_paged_decode.py
# Only LeanAttentionPaged configs, with BS=1, KV_BLK_SZ=16, num_blocks=64
# Each config: (BS, HQ, HK, SEQ_LEN, HEAD_DIM)
ALL_CONFIGS = []
BS = 1
for HQ in [32]:
    HK = HQ
    for SEQ_LEN in [512, 1024, 2*1024, 4*1024, 8*1024, 16*1024, 32*1024]:
        for HEAD_DIM in [128]:
            ALL_CONFIGS.append((BS, HQ, HK, SEQ_LEN, HEAD_DIM))


def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


def build_lean_attention_inputs(BS, HQ, HK, SEQ_LEN, HEAD_DIM, seed=42):
    """Build inputs for lean attention paged kernel, adapted from benchmark's input_la_helper."""
    torch.manual_seed(seed)
    random.seed(seed)

    KV_BLK_SZ = 16
    total_programs = 304
    BLOCK_M = 16
    BLOCK_N = KV_BLK_SZ
    N_CTX_Q = 16
    dtype = torch.float16

    n_ctx = [SEQ_LEN for _ in range(BS)]
    sum_n_ctx = sum(n_ctx)

    # Calculate batch_num_block_n (cumulative block counts)
    list_num_block_n = [(s + BLOCK_N - 1) // BLOCK_N for s in n_ctx]
    len_sum = 0
    list_sum_block_n = []
    for i in range(BS):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, device="cuda", dtype=torch.int32)

    sm_scale = 0.5

    # Allocate tensors
    q = torch.empty((HQ, N_CTX_Q * BS, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    k = torch.empty((HQ, sum_n_ctx, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)
    v = torch.empty((HQ, sum_n_ctx, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5)

    num_kv_blocks = sum_n_ctx // BLOCK_N + (1 if (sum_n_ctx % BLOCK_N != 0) else 0)

    # Build kv_block_tables and ref_block_tables for correctness
    block_tables = []
    ref_block_tables = []
    kv_n_ctx = []
    last = 0
    for s in n_ctx:
        kv_blocks = s // BLOCK_N + (1 if (s % BLOCK_N != 0) else 0)
        last += kv_blocks
        kv_n_ctx.append(last)

    for head in range(HQ):
        ref_b = []
        ref_b_ctx = []
        kv_n_ctx_idx = 0
        r = random.sample(range(0, num_kv_blocks), num_kv_blocks)
        for i in range(num_kv_blocks):
            ref_b.append(r[i])
            if i == kv_n_ctx[kv_n_ctx_idx] - 1:
                ref_b_ctx.append(ref_b)
                ref_b = []
                kv_n_ctx_idx += 1
        block_tables.append(r)
        ref_block_tables.append(ref_b_ctx)
    kv_block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    # LeanAttention specific parameters
    Mp = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, N_CTX_Q, HEAD_DIM), device=q.device, dtype=torch.float32)
    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    return {
        "q": q, "k": k, "v": v,
        "kv_block_tables": kv_block_tables,
        "Mp": Mp, "Lp": Lp, "Op": Op,
        "locks": locks,
        "batch_num_block_n": batch_num_block_n,
        "total_programs": total_programs,
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
        "BS": BS, "sm_scale": sm_scale,
        "N_CTX_Q": N_CTX_Q,
        "ref_block_tables": ref_block_tables,
        "n_ctx": n_ctx,
    }


def run_kernel(inputs):
    """Run the lean attention paged kernel."""
    return persistent_lean_attention_paged(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        kv_block_tables=inputs["kv_block_tables"],
        Mp=inputs["Mp"],
        Lp=inputs["Lp"],
        Op=inputs["Op"],
        locks=inputs["locks"],
        batch_num_block_n=inputs["batch_num_block_n"],
        total_programs=inputs["total_programs"],
        BLOCK_M=inputs["BLOCK_M"],
        BLOCK_N=inputs["BLOCK_N"],
        batch_size=inputs["BS"],
        sm_scale=inputs["sm_scale"],
        num_warps=4,
        waves_per_eu=2,
    )


def compute_reference(inputs):
    """Compute reference output using PyTorch, adapted from test_la_paged.py."""
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    ref_block_tables = inputs["ref_block_tables"]
    n_ctx = inputs["n_ctx"]
    n_ctx_q = inputs["N_CTX_Q"]
    BLOCK_N = inputs["BLOCK_N"]
    sm_scale = inputs["sm_scale"]
    batch = inputs["BS"]
    H = q.shape[0]

    ref_out = torch.empty_like(q, dtype=v.dtype)
    start = 0
    start_q = 0
    for h_idx in range(H):
        for b in range(batch):
            qb = q[h_idx, start_q:(start_q + n_ctx_q), :]
            idxs = [
                ref_block_tables[h_idx][b][kv_b_i] * BLOCK_N + b_i
                for kv_b_i in range(len(ref_block_tables[h_idx][b]))
                for b_i in range(BLOCK_N)
            ]
            idxs = torch.tensor(idxs, dtype=torch.int32, device="cuda")
            kb = torch.index_select(k[h_idx], dim=0, index=idxs)
            vb = torch.index_select(v[h_idx], dim=0, index=idxs)
            p = torch.matmul(qb, kb.transpose(0, 1)) * sm_scale
            p = torch.softmax(p.float(), dim=-1).to(q.dtype)
            refb = torch.matmul(p, vb)
            ref_out[h_idx, start_q:(start_q + n_ctx_q), :] = refb
            start += b
            start_q += n_ctx_q
        start = 0
        start_q = 0
    return ref_out


def run_correctness(configs):
    """Run correctness checks."""
    print(f"Running correctness on {len(configs)} configs...")
    all_passed = True
    for i, (BS, HQ, HK, SEQ_LEN, HEAD_DIM) in enumerate(configs):
        label = f"BS={BS} HQ={HQ} HK={HK} SEQ_LEN={SEQ_LEN} HEAD_DIM={HEAD_DIM}"
        try:
            inputs = build_lean_attention_inputs(BS, HQ, HK, SEQ_LEN, HEAD_DIM, seed=42)
            # Reset locks before each run
            inputs["locks"].zero_()
            la_out = run_kernel(inputs)
            ref_out = compute_reference(inputs)
            atol = 1e-2
            rtol = 3e-3
            torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)
            print(f"  [{i+1}/{len(configs)}] PASS  {label}")
        except Exception as e:
            print(f"  [{i+1}/{len(configs)}] FAIL  {label}: {e}")
            all_passed = False
    return all_passed


def run_benchmark(configs, label="benchmark"):
    """Run benchmark and return geometric mean latency."""
    print(f"Running {label} on {len(configs)} configs...")
    latencies = []
    for i, (BS, HQ, HK, SEQ_LEN, HEAD_DIM) in enumerate(configs):
        config_label = f"BS={BS} HQ={HQ} HK={HK} SEQ_LEN={SEQ_LEN} HEAD_DIM={HEAD_DIM}"
        inputs = build_lean_attention_inputs(BS, HQ, HK, SEQ_LEN, HEAD_DIM, seed=42)

        # Build the kernel call lambda
        def kernel_fn(inp=inputs):
            inp["locks"].zero_()
            return run_kernel(inp)

        # Warmup
        for _ in range(WARMUP):
            kernel_fn()
        torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(ITERATIONS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            kernel_fn()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

        times.sort()
        median_ms = times[len(times) // 2]
        latencies.append(median_ms)
        print(f"  {config_label}  {median_ms:.4f}ms")

    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")
    return geo_mean


def run_profile(configs):
    """Run profile mode (no correctness, just kernel execution)."""
    print(f"Running profile on {len(configs)} configs...")
    for i, (BS, HQ, HK, SEQ_LEN, HEAD_DIM) in enumerate(configs):
        config_label = f"BS={BS} HQ={HQ} HK={HK} SEQ_LEN={SEQ_LEN} HEAD_DIM={HEAD_DIM}"
        inputs = build_lean_attention_inputs(BS, HQ, HK, SEQ_LEN, HEAD_DIM, seed=42)
        inputs["locks"].zero_()
        run_kernel(inputs)
        torch.cuda.synchronize()
        print(f"  [{i+1}/{len(configs)}] Profiled {config_label}")


def main():
    parser = argparse.ArgumentParser(description="Test harness for lean_atten_paged")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)

    if args.correctness:
        configs = _pick(ALL_CONFIGS, 25)
        print(f"GEAK_SHAPES_USED={sorted(configs)}")
        passed = run_correctness(configs)
        if not passed:
            print("CORRECTNESS FAILED")
            sys.exit(1)
        print("ALL CORRECTNESS CHECKS PASSED")
    elif args.benchmark:
        configs = _pick(ALL_CONFIGS, 25)
        print(f"GEAK_SHAPES_USED={sorted(configs)}")
        run_benchmark(configs, "benchmark")
    elif args.full_benchmark:
        configs = ALL_CONFIGS
        print(f"GEAK_SHAPES_USED={sorted(configs)}")
        run_benchmark(configs, "full-benchmark")
    elif args.profile:
        configs = _pick(ALL_CONFIGS, 5)
        print(f"GEAK_SHAPES_USED={sorted(configs)}")
        run_profile(configs)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
