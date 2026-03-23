#!/usr/bin/env python3
"""Test harness for fused_qkv_split_qk_rope kernel."""

import sys
import os
import argparse
import math

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
torch.manual_seed(42)

# Imports from the repo
from aiter.ops.triton.rope.fused_qkv_split_qk_rope import fused_qkv_split_qk_rope
from op_tests.triton_tests.fusions.test_fused_qk_concat import generate_rope_cached_freqs
from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle

# ─── Constants ───
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ─── Build ordered full case stream from test parametrize ───
# pytest parametrize order: last decorator varies fastest (outermost loop)
# Decorators top-to-bottom: B, QH_PER_KH, KH, D, rotate_style, max_embed_positions,
#   (nope, nope_first), reuse_freqs_front_part, dtype
# pytest iterates: dtype (fastest) -> reuse_freqs_front_part -> (nope, nope_first) ->
#   max_embed_positions -> rotate_style -> D -> KH -> QH_PER_KH -> B (slowest)

B_vals = [1, 4, 8, 16, 32]
QH_PER_KH_vals = [1, 2, 4, 8, 16]
KH_vals = [1, 4]
D_vals = [64, 128]
rotate_style_vals = [RotateStyle.GPTJ, RotateStyle.NEOX]
max_embed_positions_vals = [131072]
nope_nope_first_vals = [(False, False), (True, False), (True, True)]
reuse_freqs_front_part_vals = [False, True]
dtype_vals = [torch.bfloat16]

ALL_CONFIGS = []
for B in B_vals:
    for QH_PER_KH in QH_PER_KH_vals:
        for KH in KH_vals:
            for D in D_vals:
                for rotate_style in rotate_style_vals:
                    for max_embed_positions in max_embed_positions_vals:
                        for (nope, nope_first) in nope_nope_first_vals:
                            for reuse_freqs_front_part in reuse_freqs_front_part_vals:
                                for dtype in dtype_vals:
                                    ALL_CONFIGS.append((
                                        B, QH_PER_KH, KH, D,
                                        rotate_style, max_embed_positions,
                                        nope, nope_first,
                                        reuse_freqs_front_part, dtype
                                    ))


def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


def config_label(cfg):
    B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype = cfg
    rs = "NEOX" if rotate_style == RotateStyle.NEOX else "GPTJ"
    return (f"B={B} QH_PER_KH={QH_PER_KH} KH={KH} D={D} "
            f"rotate={rs} nope={nope} nope_first={nope_first} "
            f"reuse_front={reuse_freqs_front_part}")


def generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype):
    qkv = torch.randn(
        (B, (QH_PER_KH * KH + 2 * KH) * (D * (2 if nope else 1))),
        dtype=dtype, device="cuda",
    )
    return qkv


def run_torch(qkv, QH_PER_KH, KH, D, ref_freqs, reuse_freqs_front_part, nope, nope_first, rotate_style):
    q_size = QH_PER_KH * KH * D
    kv_size = KH * D
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(-1, QH_PER_KH * KH, D).contiguous()
    k = k.view(-1, KH, D).contiguous()
    v = v.view(-1, KH, D).contiguous()

    q = ref_rope_sbhd_fwd(
        q, ref_freqs, rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part, nope_first=nope_first,
    )
    k = ref_rope_sbhd_fwd(
        k, ref_freqs, rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part, nope_first=nope_first,
    )
    return q, k, v


def build_inputs(cfg):
    B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype = cfg
    torch.manual_seed(42)
    qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype)
    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, (D // 2) if reuse_freqs_front_part else D, dtype
    )
    ref_freqs = freqs[pos].squeeze(-2)
    return qkv, pos, freqs, cos, sin, ref_freqs


def run_kernel(qkv, cos, sin, pos, QH_PER_KH, KH, D, nope, nope_first, rotate_style, reuse_freqs_front_part):
    return fused_qkv_split_qk_rope(
        qkv, cos, sin, pos,
        QH_PER_KH * KH, KH,
        (D * (2 if nope else 1)),
        is_neox=(rotate_style == RotateStyle.NEOX),
        offsets=None,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )


def run_correctness(configs):
    print(f"Running correctness on {len(configs)} configs...")
    failures = 0
    for i, cfg in enumerate(configs):
        B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype = cfg
        qkv, pos, freqs, cos, sin, ref_freqs = build_inputs(cfg)

        q_triton, k_triton, v_triton = run_kernel(
            qkv, cos, sin, pos, QH_PER_KH, KH, D, nope, nope_first,
            rotate_style, reuse_freqs_front_part
        )
        q_torch, k_torch, v_torch = run_torch(
            qkv, QH_PER_KH, KH, (D * (2 if nope else 1)),
            ref_freqs, reuse_freqs_front_part, nope, nope_first, rotate_style
        )

        try:
            torch.testing.assert_close(q_torch, q_triton)
            torch.testing.assert_close(k_torch, k_triton)
            torch.testing.assert_close(v_torch, v_triton)
            print(f"  [{i+1}/{len(configs)}] PASS  {config_label(cfg)}")
        except AssertionError as e:
            print(f"  [{i+1}/{len(configs)}] FAIL  {config_label(cfg)}")
            print(f"    {e}")
            failures += 1

    print(f"\nCorrectness: {len(configs) - failures}/{len(configs)} passed")
    if failures > 0:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("CORRECTNESS PASSED")


def run_benchmark(configs, label="benchmark"):
    print(f"Running {label} on {len(configs)} configs...")
    latencies = []
    for i, cfg in enumerate(configs):
        B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype = cfg
        qkv, pos, freqs, cos, sin, ref_freqs = build_inputs(cfg)

        # Warmup
        for _ in range(WARMUP):
            run_kernel(qkv, cos, sin, pos, QH_PER_KH, KH, D, nope, nope_first,
                       rotate_style, reuse_freqs_front_part)
        torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(ITERATIONS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_kernel(qkv, cos, sin, pos, QH_PER_KH, KH, D, nope, nope_first,
                       rotate_style, reuse_freqs_front_part)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median_ms = times[len(times) // 2]
        latencies.append(median_ms)
        print(f"  {config_label(cfg)}  {median_ms:.4f}ms")

    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")


def run_profile(configs):
    print(f"Running profile on {len(configs)} configs...")
    for i, cfg in enumerate(configs):
        B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype = cfg
        qkv, pos, freqs, cos, sin, ref_freqs = build_inputs(cfg)
        # Warmup
        for _ in range(3):
            run_kernel(qkv, cos, sin, pos, QH_PER_KH, KH, D, nope, nope_first,
                       rotate_style, reuse_freqs_front_part)
        torch.cuda.synchronize()
        # Single profiled run
        run_kernel(qkv, cos, sin, pos, QH_PER_KH, KH, D, nope, nope_first,
                   rotate_style, reuse_freqs_front_part)
        torch.cuda.synchronize()
        print(f"  [{i+1}/{len(configs)}] profiled  {config_label(cfg)}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.correctness:
        configs = _pick(ALL_CONFIGS, 25)
        run_correctness(configs)
        print(f"GEAK_SHAPES_USED={sorted([(c[0],c[1],c[2],c[3],int(c[4]),c[6],c[7],c[8]) for c in configs])}")
    elif args.benchmark:
        configs = _pick(ALL_CONFIGS, 25)
        run_benchmark(configs, "benchmark")
        print(f"GEAK_SHAPES_USED={sorted([(c[0],c[1],c[2],c[3],int(c[4]),c[6],c[7],c[8]) for c in configs])}")
    elif args.full_benchmark:
        run_benchmark(ALL_CONFIGS, "full-benchmark")
        print(f"GEAK_SHAPES_USED={sorted([(c[0],c[1],c[2],c[3],int(c[4]),c[6],c[7],c[8]) for c in ALL_CONFIGS])}")
    elif args.profile:
        configs = _pick(ALL_CONFIGS, 5)
        run_profile(configs)
        print(f"GEAK_SHAPES_USED={sorted([(c[0],c[1],c[2],c[3],int(c[4]),c[6],c[7],c[8]) for c in configs])}")


if __name__ == "__main__":
    main()
