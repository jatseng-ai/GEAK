#!/usr/bin/env python3
"""Test harness for fused_qkv_split_qk_rope kernel."""

import sys
import os
import argparse
import math
import itertools

# Resolve repo root
REPO_ROOT = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get(
        "GEAK_REPO_ROOT",
        "/tmp/preproc_debug_topk2/.geak_resolved/ROCm_aiter/main-e15b75d464bf",
    ),
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
torch.manual_seed(42)

from aiter.ops.triton.rope.fused_qkv_split_qk_rope import fused_qkv_split_qk_rope
from op_tests.triton_tests.fusions.test_fused_qk_concat import generate_rope_cached_freqs
from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle
from op_tests.triton_tests.rope.test_fused_qkv_split_qk_rope import (
    generate_qkv_inputs,
    run_torch,
)

# ── Fixed constants ──
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ── Build the ordered full case stream ──
# pytest parametrize: last decorator = outermost, first decorator = innermost.
# Decorators top-to-bottom: B, QH_PER_KH, KH, D, rotate_style, max_embed_positions,
#   (nope,nope_first), reuse_freqs_front_part, dtype
# So outermost = dtype, innermost = B.
_dtype_vals = [torch.bfloat16]
_reuse_freqs_front_part_vals = [False, True]
_nope_nope_first_vals = [(False, False), (True, False), (True, True)]
_max_embed_positions_vals = [131072]
_rotate_style_vals = [RotateStyle.GPTJ, RotateStyle.NEOX]
_D_vals = [64, 128]
_KH_vals = [1, 4]
_QH_PER_KH_vals = [1, 2, 4, 8, 16]
_B_vals = [1, 4, 8, 16, 32]

ALL_CONFIGS = []
for dtype in _dtype_vals:
    for reuse_freqs_front_part in _reuse_freqs_front_part_vals:
        for (nope, nope_first) in _nope_nope_first_vals:
            for max_embed_positions in _max_embed_positions_vals:
                for rotate_style in _rotate_style_vals:
                    for D in _D_vals:
                        for KH in _KH_vals:
                            for QH_PER_KH in _QH_PER_KH_vals:
                                for B in _B_vals:
                                    ALL_CONFIGS.append((
                                        B, QH_PER_KH, KH, D,
                                        rotate_style, max_embed_positions,
                                        nope, nope_first,
                                        reuse_freqs_front_part, dtype,
                                    ))


def _pick(configs, count):
    if len(configs) <= count:
        return configs
    n = len(configs)
    return [configs[round(i * (n - 1) / (count - 1))] for i in range(count)]


def _config_label(cfg):
    B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype = cfg
    rs = "NEOX" if rotate_style == RotateStyle.NEOX else "GPTJ"
    return (f"B={B} QH_PER_KH={QH_PER_KH} KH={KH} D={D} "
            f"rotate={rs} nope={nope} nope_first={nope_first} "
            f"reuse_front={reuse_freqs_front_part}")


def _build_inputs(cfg):
    """Build inputs for a single config. Returns (qkv, cos, sin, pos, ref_freqs, cfg_params)."""
    B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype = cfg

    torch.manual_seed(42)
    qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype)
    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, max_embed_positions, (D // 2) if reuse_freqs_front_part else D, dtype
    )
    ref_freqs = freqs[pos].squeeze(-2)

    return qkv, cos, sin, pos, ref_freqs, cfg


def _run_kernel(qkv, cos, sin, pos, cfg):
    """Run the triton kernel."""
    B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype = cfg
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
    passed = 0
    failed = 0
    for i, cfg in enumerate(configs):
        label = _config_label(cfg)
        B, QH_PER_KH, KH, D, rotate_style, max_embed_positions, nope, nope_first, reuse_freqs_front_part, dtype = cfg
        try:
            qkv, cos, sin, pos, ref_freqs, _ = _build_inputs(cfg)
            q_triton, k_triton, v_triton = _run_kernel(qkv, cos, sin, pos, cfg)
            q_torch, k_torch, v_torch = run_torch(
                qkv, QH_PER_KH, KH,
                (D * (2 if nope else 1)),
                ref_freqs, reuse_freqs_front_part,
                nope, nope_first, rotate_style,
            )
            torch.testing.assert_close(q_torch, q_triton)
            torch.testing.assert_close(k_torch, k_triton)
            torch.testing.assert_close(v_torch, v_triton)
            passed += 1
        except Exception as e:
            print(f"FAIL [{i}] {label}: {e}")
            failed += 1

    print(f"Correctness: {passed}/{passed+failed} passed")
    shapes_used = [_config_label(c) for c in configs]
    print(f"GEAK_SHAPES_USED={sorted(shapes_used)}")
    if failed > 0:
        sys.exit(1)
    print("ALL CORRECTNESS TESTS PASSED")


def run_benchmark(configs, label="benchmark"):
    print(f"Running {label} on {len(configs)} configs...")
    latencies = []
    for i, cfg in enumerate(configs):
        clabel = _config_label(cfg)
        qkv, cos, sin, pos, ref_freqs, _ = _build_inputs(cfg)

        # Build the callable
        def fn(qkv=qkv, cos=cos, sin=sin, pos=pos, cfg=cfg):
            return _run_kernel(qkv, cos, sin, pos, cfg)

        # Warmup
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(ITERATIONS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median_ms = times[len(times) // 2]
        latencies.append(median_ms)
        print(f"{clabel}  {median_ms:.4f}ms")

    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))

    shapes_used = [_config_label(c) for c in configs]
    print(f"GEAK_SHAPES_USED={sorted(shapes_used)}")
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")


def run_profile(configs):
    print(f"Running profile on {len(configs)} configs...")
    for cfg in configs:
        qkv, cos, sin, pos, ref_freqs, _ = _build_inputs(cfg)
        _run_kernel(qkv, cos, sin, pos, cfg)
    torch.cuda.synchronize()
    shapes_used = [_config_label(c) for c in configs]
    print(f"GEAK_SHAPES_USED={sorted(shapes_used)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.correctness:
        run_correctness(_pick(ALL_CONFIGS, 25))
    elif args.benchmark:
        run_benchmark(_pick(ALL_CONFIGS, 25), "benchmark")
    elif args.full_benchmark:
        run_benchmark(ALL_CONFIGS, "full-benchmark")
    elif args.profile:
        run_profile(_pick(ALL_CONFIGS, 5))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
