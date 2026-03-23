#!/usr/bin/env python3
"""
Test harness for lean_atten_paged kernel.
Modes: --correctness, --benchmark, --full-benchmark, --profile
"""

import argparse
import math
import os
import sys
import random

import torch

# Resolve imports
REPO_ROOT = os.environ.get(
    "GEAK_REPO_ROOT",
    "/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf",
)
WORK_DIR = os.environ.get("GEAK_WORK_DIR", "")
if WORK_DIR and WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aiter.ops.triton.attention.lean_atten_paged import persistent_lean_attention_paged

# ─── Constants ───────────────────────────────────────────────────────────────
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ─── Config list from benchmark file ────────────────────────────────────────
# Source: op_tests/op_benchmarks/triton/bench_la_paged_decode.py
# Only LeanAttentionPaged configs, BS=1 (default in benchmark)
# Format: (BS, HQ, SEQ_LEN, HEAD_DIM)
# The benchmark uses KV_BLK_SZ=16, num_blocks=64
ALL_CONFIGS = []
BS = 1
for HQ in [32]:
    for SEQ_LEN in [512, 1024, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024]:
        for HEAD_DIM in [128]:
            ALL_CONFIGS.append((BS, HQ, SEQ_LEN, HEAD_DIM))


def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


# ─── Input helper (from benchmark file's input_la_helper) ──────────────────
def setup_lean_attention_inputs(BS, H_Q, D, SEQ_LEN, KV_BLK_SZ=16, num_blocks=64):
    """
    Set up inputs for lean attention paged kernel.
    Adapted from bench_la_paged_decode.py::input_la_helper
    """
    total_programs = 304
    BLOCK_M = 16
    BLOCK_N = KV_BLK_SZ
    dtype = torch.float16

    n_ctx = [SEQ_LEN for _ in range(BS)]
    N_CTX_Q = 16

    sum_n_ctx = sum(int(n) for n in n_ctx)

    list_num_block_n = [(int(str(s).strip()) + BLOCK_N - 1) // BLOCK_N for s in n_ctx]
    len_sum = 0
    list_sum_block_n = []
    for i in range(BS):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, device="cuda", dtype=torch.int32)

    sm_scale = 0.5

    q = torch.empty((H_Q, N_CTX_Q * BS, D), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((H_Q, sum_n_ctx, D), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((H_Q, sum_n_ctx, D), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    num_kv_blocks = sum_n_ctx // BLOCK_N + (1 if (sum_n_ctx % BLOCK_N != 0) else 0)

    block_tables = []
    for head in range(H_Q):
        b = random.sample(range(0, num_kv_blocks), num_kv_blocks)
        block_tables.append(b)
    kv_block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    Mp = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
    Op = torch.empty(
        (total_programs, N_CTX_Q, D), device=q.device, dtype=torch.float32
    )
    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    return {
        "q": q,
        "k": k,
        "v": v,
        "kv_block_tables": kv_block_tables,
        "Mp": Mp,
        "Lp": Lp,
        "Op": Op,
        "locks": locks,
        "batch_num_block_n": batch_num_block_n,
        "total_programs": total_programs,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "batch_size": BS,
        "sm_scale": sm_scale,
        "n_ctx": n_ctx,
        "N_CTX_Q": N_CTX_Q,
        "sum_n_ctx": sum_n_ctx,
        "num_kv_blocks": num_kv_blocks,
    }


def run_kernel(inputs):
    """Run the lean attention paged kernel."""
    # Reset locks to zero before each call
    inputs["locks"].zero_()
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
        batch_size=inputs["batch_size"],
        sm_scale=inputs["sm_scale"],
        num_warps=4,
        waves_per_eu=2,
    )


def compute_reference(inputs):
    """
    Compute reference output using PyTorch.
    Adapted from test_la_paged.py::test_persistent_lean_attention
    """
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    n_ctx = inputs["n_ctx"]
    n_ctx_q = inputs["N_CTX_Q"]
    BLOCK_N = inputs["BLOCK_N"]
    sm_scale = inputs["sm_scale"]
    batch = inputs["batch_size"]
    H = q.shape[0]

    # Build ref_block_tables from kv_block_tables
    kv_block_tables_cpu = inputs["kv_block_tables"].cpu().tolist()

    # Compute kv_n_ctx (cumulative block counts per batch element)
    kv_n_ctx = []
    last = 0
    for s in n_ctx:
        kv_blocks = s // BLOCK_N + (1 if (s % BLOCK_N != 0) else 0)
        last += kv_blocks
        kv_n_ctx.append(last)

    ref_out = torch.empty_like(q, dtype=v.dtype)

    for h_idx in range(H):
        # Build ref_block_tables for this head
        ref_b_ctx = []
        ref_b = []
        kv_n_ctx_idx = 0
        num_kv_blocks = inputs["num_kv_blocks"]
        r = kv_block_tables_cpu[h_idx]

        for i in range(num_kv_blocks):
            ref_b.append(r[i])
            if i == kv_n_ctx[kv_n_ctx_idx] - 1:
                ref_b_ctx.append(ref_b)
                ref_b = []
                kv_n_ctx_idx += 1

        start_q = 0
        for b_idx in range(batch):
            qb = q[h_idx, start_q : start_q + n_ctx_q, :]

            idxs = [
                ref_b_ctx[b_idx][kv_b_i] * BLOCK_N + b_i
                for kv_b_i in range(len(ref_b_ctx[b_idx]))
                for b_i in range(BLOCK_N)
            ]
            idxs = torch.tensor(idxs, dtype=torch.int32, device="cuda")
            kb = torch.index_select(k[h_idx], dim=0, index=idxs)
            vb = torch.index_select(v[h_idx], dim=0, index=idxs)

            p = torch.matmul(qb, kb.transpose(0, 1)) * sm_scale
            p = torch.softmax(p.float(), dim=-1).to(q.dtype)
            refb = torch.matmul(p, vb)
            ref_out[h_idx, start_q : start_q + n_ctx_q, :] = refb
            start_q += n_ctx_q

    return ref_out


def benchmark_config(config):
    """Benchmark a single config, return median latency in ms."""
    BS, HQ, SEQ_LEN, HEAD_DIM = config
    torch.manual_seed(42)
    random.seed(42)
    inputs = setup_lean_attention_inputs(BS, HQ, HEAD_DIM, SEQ_LEN)

    # Warmup
    for _ in range(WARMUP):
        inputs["locks"].zero_()
        run_kernel(inputs)
    torch.cuda.synchronize()

    # Timed iterations
    times = []
    for _ in range(ITERATIONS):
        inputs["locks"].zero_()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        run_kernel(inputs)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times.sort()
    median_ms = times[len(times) // 2]
    return median_ms


def correctness_config(config):
    """Run correctness check for a single config. Returns True if passed."""
    BS, HQ, SEQ_LEN, HEAD_DIM = config
    torch.manual_seed(42)
    random.seed(42)
    inputs = setup_lean_attention_inputs(BS, HQ, HEAD_DIM, SEQ_LEN)

    # Run kernel
    la_out = run_kernel(inputs)

    # Compute reference
    ref_out = compute_reference(inputs)

    # Compare
    atol = 1e-2
    rtol = 3e-3
    try:
        torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Test harness for lean_atten_paged")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    configs = ALL_CONFIGS

    if args.correctness:
        selected = _pick(configs, 25)
        print(f"Running correctness on {len(selected)} configs...")
        all_passed = True
        for cfg in selected:
            BS, HQ, SEQ_LEN, HEAD_DIM = cfg
            label = f"BS={BS} HQ={HQ} SEQ_LEN={SEQ_LEN} HEAD_DIM={HEAD_DIM}"
            print(f"  Testing {label} ...", end=" ", flush=True)
            passed = correctness_config(cfg)
            if passed:
                print("PASSED")
            else:
                print("FAILED")
                all_passed = False
        print(f"\nGEAK_SHAPES_USED={sorted(selected)}")
        if not all_passed:
            print("CORRECTNESS FAILED", file=sys.stderr)
            sys.exit(1)
        print("ALL CORRECTNESS CHECKS PASSED")

    elif args.full_benchmark:
        selected = configs
        latencies = []
        for cfg in selected:
            BS, HQ, SEQ_LEN, HEAD_DIM = cfg
            label = f"BS={BS} HQ={HQ} SEQ_LEN={SEQ_LEN} HEAD_DIM={HEAD_DIM}"
            ms = benchmark_config(cfg)
            latencies.append(ms)
            print(f"{label}  {ms:.4f}ms")
        geo_mean = math.exp(sum(math.log(t) for t in latencies) / len(latencies))
        print(f"\nGEAK_SHAPES_USED={sorted(selected)}")
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")

    elif args.benchmark:
        selected = _pick(configs, 25)
        latencies = []
        for cfg in selected:
            BS, HQ, SEQ_LEN, HEAD_DIM = cfg
            label = f"BS={BS} HQ={HQ} SEQ_LEN={SEQ_LEN} HEAD_DIM={HEAD_DIM}"
            ms = benchmark_config(cfg)
            latencies.append(ms)
            print(f"{label}  {ms:.4f}ms")
        geo_mean = math.exp(sum(math.log(t) for t in latencies) / len(latencies))
        print(f"\nGEAK_SHAPES_USED={sorted(selected)}")
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")

    elif args.profile:
        selected = _pick(configs, 5)
        for cfg in selected:
            BS, HQ, SEQ_LEN, HEAD_DIM = cfg
            label = f"BS={BS} HQ={HQ} SEQ_LEN={SEQ_LEN} HEAD_DIM={HEAD_DIM}"
            torch.manual_seed(42)
            random.seed(42)
            inputs = setup_lean_attention_inputs(BS, HQ, HEAD_DIM, SEQ_LEN)
            # Warmup
            for _ in range(3):
                inputs["locks"].zero_()
                run_kernel(inputs)
            torch.cuda.synchronize()
            # Profile run
            inputs["locks"].zero_()
            run_kernel(inputs)
            torch.cuda.synchronize()
            print(f"Profiled: {label}")
        print(f"\nGEAK_SHAPES_USED={sorted(selected)}")


if __name__ == "__main__":
    main()
