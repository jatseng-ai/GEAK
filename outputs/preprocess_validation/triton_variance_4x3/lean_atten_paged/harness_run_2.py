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

import torch

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

from aiter.ops.triton.attention.lean_atten_paged import persistent_lean_attention_paged

# ─── Constants ───────────────────────────────────────────────────────────────
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ─── Config list from bench_la_paged_decode.py ───────────────────────────────
# Each config: (BS, HQ, HK, SEQ_LEN, HEAD_DIM)
# The benchmark uses BS=1, HQ=32, HK=32, HEAD_DIM=128, varying SEQ_LEN
# We only include LeanAttentionPaged configs (not PagedAttention)
ALL_CONFIGS = []
for HQ in [32]:
    HK = HQ
    for SEQ_LEN in [
        512,
        1024,
        2 * 1024,
        4 * 1024,
        8 * 1024,
        16 * 1024,
        32 * 1024,
    ]:
        for HEAD_DIM in [128]:
            ALL_CONFIGS.append((1, HQ, HK, SEQ_LEN, HEAD_DIM))


def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


def setup_inputs(BS, HQ, HK, D, SEQ_LEN, dtype=torch.float16):
    """
    Set up inputs for lean_atten_paged kernel.
    Adapted from bench_la_paged_decode.py::input_la_helper and test_la_paged.py.
    """
    torch.manual_seed(42)
    random.seed(42)

    total_programs = 304
    BLOCK_M = 16
    KV_BLK_SZ = 16
    BLOCK_N = KV_BLK_SZ

    n_ctx = [SEQ_LEN for _ in range(BS)]
    N_CTX_Q = 16
    sum_n_ctx = sum(n_ctx)

    # Compute batch_num_block_n (cumulative block counts)
    list_num_block_n = [(s + BLOCK_N - 1) // BLOCK_N for s in n_ctx]
    len_sum = 0
    list_sum_block_n = []
    for i in range(BS):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, device="cuda", dtype=torch.int32)

    sm_scale = 0.5

    # Allocate Tensors
    q = torch.empty((HQ, N_CTX_Q * BS, D), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((HQ, sum_n_ctx, D), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((HQ, sum_n_ctx, D), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    num_kv_blocks = sum_n_ctx // BLOCK_N + (1 if (sum_n_ctx % BLOCK_N != 0) else 0)

    # Build block tables
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
        kv_n_ctx_idx = 0
        r = random.sample(range(0, num_kv_blocks), num_kv_blocks)
        ref_b_list = []
        for i in range(num_kv_blocks):
            ref_b_list.append(r[i])
            if i == kv_n_ctx[kv_n_ctx_idx] - 1:
                ref_block_tables_entry = ref_b_list[:]
                ref_b.append(ref_block_tables_entry)
                ref_b_list = []
                kv_n_ctx_idx += 1
        block_tables.append(r)
        ref_block_tables.append(ref_b)

    kv_block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    # LeanAttention Specific Parameters
    Mp = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, N_CTX_Q, D), device=q.device, dtype=torch.float32)
    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    return {
        "q": q, "k": k, "v": v,
        "kv_block_tables": kv_block_tables,
        "Mp": Mp, "Lp": Lp, "Op": Op,
        "locks": locks,
        "batch_num_block_n": batch_num_block_n,
        "total_programs": total_programs,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "batch_size": BS,
        "sm_scale": sm_scale,
        "n_ctx_q": N_CTX_Q,
        "n_ctx": n_ctx,
        "ref_block_tables": ref_block_tables,
        "D": D,
        "HQ": HQ,
    }


def run_kernel(inputs):
    """Run the lean_atten_paged kernel and return output."""
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
    """Compute reference output using PyTorch (from test_la_paged.py)."""
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    n_ctx_q = inputs["n_ctx_q"]
    n_ctx = inputs["n_ctx"]
    BLOCK_N = inputs["BLOCK_N"]
    sm_scale = inputs["sm_scale"]
    ref_block_tables = inputs["ref_block_tables"]
    batch = inputs["batch_size"]
    HQ = inputs["HQ"]

    ref_out = torch.empty_like(q, dtype=v.dtype)
    for h_idx in range(HQ):
        start_q = 0
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
            start_q += n_ctx_q
    return ref_out


def benchmark_config(inputs):
    """Benchmark a single config, return median latency in ms."""
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
    return times[len(times) // 2]  # median


def run_correctness(configs):
    print(f"Running correctness check on {len(configs)} configs...")
    all_pass = True
    for cfg in configs:
        BS, HQ, HK, SEQ_LEN, HEAD_DIM = cfg
        label = f"BS={BS} HQ={HQ} HK={HK} SEQ_LEN={SEQ_LEN} HEAD_DIM={HEAD_DIM}"
        try:
            inputs = setup_inputs(BS, HQ, HK, HEAD_DIM, SEQ_LEN)
            la_out = run_kernel(inputs)
            ref_out = compute_reference(inputs)
            atol = 1e-2
            rtol = 3e-3
            torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)
            print(f"  PASS: {label}")
        except Exception as e:
            print(f"  FAIL: {label} - {e}")
            all_pass = False
        finally:
            torch.cuda.empty_cache()

    print(f"GEAK_SHAPES_USED={sorted(configs)}")
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


def run_benchmark_mode(configs, mode_name="benchmark"):
    print(f"Running {mode_name} on {len(configs)} configs...")
    latencies = []
    for cfg in configs:
        BS, HQ, HK, SEQ_LEN, HEAD_DIM = cfg
        label = f"BS={BS} HQ={HQ} HK={HK} SEQ_LEN={SEQ_LEN} HEAD_DIM={HEAD_DIM}"
        try:
            inputs = setup_inputs(BS, HQ, HK, HEAD_DIM, SEQ_LEN)
            ms = benchmark_config(inputs)
            latencies.append(ms)
            print(f"  {label}  {ms:.4f}ms")
        except Exception as e:
            print(f"  {label}  ERROR: {e}")
        finally:
            torch.cuda.empty_cache()

    print(f"GEAK_SHAPES_USED={sorted(configs)}")
    if latencies:
        geo_mean = math.exp(sum(math.log(t) for t in latencies) / len(latencies))
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")
    else:
        print("GEAK_RESULT_LATENCY_MS=N/A")
        sys.exit(1)


def run_profile(configs):
    print(f"Running profile on {len(configs)} configs...")
    for cfg in configs:
        BS, HQ, HK, SEQ_LEN, HEAD_DIM = cfg
        label = f"BS={BS} HQ={HQ} HK={HK} SEQ_LEN={SEQ_LEN} HEAD_DIM={HEAD_DIM}"
        inputs = setup_inputs(BS, HQ, HK, HEAD_DIM, SEQ_LEN)
        inputs["locks"].zero_()
        run_kernel(inputs)
        torch.cuda.synchronize()
        print(f"  {label}  done")
        torch.cuda.empty_cache()
    print(f"GEAK_SHAPES_USED={sorted(configs)}")


def main():
    parser = argparse.ArgumentParser(description="Test harness for lean_atten_paged")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)

    if args.correctness:
        configs = _pick(ALL_CONFIGS, 25)
        run_correctness(configs)
    elif args.benchmark:
        configs = _pick(ALL_CONFIGS, 25)
        run_benchmark_mode(configs, "benchmark")
    elif args.full_benchmark:
        run_benchmark_mode(ALL_CONFIGS, "full-benchmark")
    elif args.profile:
        configs = _pick(ALL_CONFIGS, 5)
        run_profile(configs)


if __name__ == "__main__":
    main()
