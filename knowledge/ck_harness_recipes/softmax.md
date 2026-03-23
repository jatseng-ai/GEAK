# CK Harness Recipe: Softmax

## Kernel Family

- Device type: `DeviceSoftmaxImpl`
- Header: `ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp`
- CK examples: `23_softmax/softmax_blockwise.cpp`
- API style: `MakeArgumentPointer` (pointer-based)
- Needs workspace: No

## Architecture

The harness consists of 5 files. The C++ kernels are compiled to shared libraries
(`.so`) with `hipcc` and loaded into a Python test script via `ctypes`. The baseline
kernel output is ground truth; the optimized kernel is verified against it.

| File | Role | Mutable? |
|------|------|----------|
| `baseline.cpp` | Compiled to `libbaseline.so` -- frozen ground truth | Never |
| `optimized.cpp` | Compiled to `liboptimized.so` -- LLM edits TUNING PARAMETERS | Each iteration |
| `test_harness.py` | Loads both `.so` via ctypes, verifies, benchmarks | Never |
| `compile.py` | Auto-detects GPU arch from PyTorch, calls `make` | Never |
| `Makefile` | `hipcc -shared -fPIC -O3` build rules | Never |

## Tunable Template Parameters

| Parameter | Position | Type | Constraints |
|-----------|----------|------|-------------|
| BlockSize | 7 | int | Power of 2: 64, 128, 256, 512, 1024 |
| ClusterM | 8 | int | ClusterM * ClusterK must equal BlockSize |
| ClusterK | 9 | int | ClusterM * ClusterK must equal BlockSize |
| SliceM | 10 | int | >= 1 |
| SliceK | 11 | int | Must align with vector size |
| SrcVecDim | 12 | int | 0 = vectorize along M, 1 = vectorize along K |
| SrcScalarPerVector | 13 | int | Power of 2: 1, 2, 4, 8; must divide tensor dim |
| OutScalarPerVector | 14 | int | Power of 2: 1, 2, 4, 8; must divide tensor dim |

Not all combinations are valid. `IsSupportedArgument` returns false for invalid
configs; the harness reports these as UNSUPPORTED (return value -1.0f).

## C ABI Signature

```cpp
extern "C" __attribute__((visibility("default"))) float run_kernel(
    const void* in_dev,
    void* out_dev,
    const int64_t* lengths,
    const int64_t* strides,
    int ndims,
    const int* reduce_dims,
    int n_reduce_dims,
    double alpha,
    double beta,
    bool time_kernel,
    int warmup,
    int nrepeat)
```

Returns elapsed time in ms when `time_kernel=true`, 0.0 otherwise, or -1.0f if
the configuration is unsupported.

## baseline.cpp

```cpp
// GEAK Test Harness - Softmax Baseline Kernel
// This file is compiled as a standalone shared library (.so).
// It does NOT depend on CK's build system — only CK headers + HIP runtime.
//
// Build: ./compile.py baseline  (or: make ARCH=gfx950 baseline)

#include <cstdint>
#include <vector>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// ============================================================================
// TUNING PARAMETERS — baseline tile configuration from the original example
// ============================================================================
using KernelInstance =
    ck::tensor_operation::device::DeviceSoftmaxImpl<ck::half_t,  // InDataType
                                                    float,       // AccDataType
                                                    ck::half_t,  // OutDataType
                                                    PassThrough, // InElementwiseOperation
                                                    PassThrough, // AccElementwiseOperation
                                                    3,           // Rank
                                                    1,           // NumReduceDim
                                                    256,         // BlockSize
                                                    8,           // ClusterM
                                                    32,          // ClusterK
                                                    1,           // SliceM
                                                    8,           // SliceK
                                                    1,           // SrcVecDim (0=M, 1=K)
                                                    8,           // SrcScalarPerVector
                                                    8            // OutScalarPerVector
                                                    >;
// ============================================================================

extern "C" __attribute__((visibility("default"))) float run_kernel(const void* in_dev,
                                                                   void* out_dev,
                                                                   const int64_t* lengths,
                                                                   const int64_t* strides,
                                                                   int ndims,
                                                                   const int* reduce_dims,
                                                                   int n_reduce_dims,
                                                                   double alpha,
                                                                   double beta,
                                                                   bool time_kernel,
                                                                   int warmup,
                                                                   int nrepeat)
{
    KernelInstance op;

    std::vector<ck::index_t> ck_lengths(lengths, lengths + ndims);
    std::vector<ck::index_t> ck_strides(strides, strides + ndims);
    std::vector<int> ck_reduce_dims(reduce_dims, reduce_dims + n_reduce_dims);

    auto arg = op.MakeArgumentPointer(ck_lengths,
                                      ck_strides,
                                      ck_reduce_dims,
                                      alpha,
                                      beta,
                                      in_dev,
                                      out_dev,
                                      PassThrough{},
                                      PassThrough{});

    if(!op.IsSupportedArgument(arg.get()))
        return -1.0f;

    StreamConfig config;
    config.stream_id_   = nullptr;
    config.time_kernel_ = time_kernel;
    config.cold_niters_ = warmup;
    config.nrepeat_     = nrepeat;

    auto invoker = op.MakeInvokerPointer();
    float ms     = invoker->Run(arg.get(), config);

    (void)hipDeviceSynchronize();
    return ms;
}
```

## optimized.cpp

```cpp
// GEAK Test Harness - Softmax Optimized Kernel
// This file is compiled as a standalone shared library (.so).
// The LLM modifies the TUNING PARAMETERS section during optimization.
//
// Build: ./compile.py optimized  (or: make ARCH=gfx950 optimized)

#include <cstdint>
#include <vector>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// ============================================================================
// TUNING PARAMETERS — modify these to optimize the kernel
//
// BlockSize:           Number of threads per block (64, 128, 256, 512, 1024)
// ClusterM, ClusterK:  Thread cluster dimensions (must multiply to BlockSize)
// SliceM, SliceK:      Work per thread (elements each thread processes)
// SrcVecDim:           Vectorized load dimension (0=M, 1=K)
// SrcScalarPerVector:  Elements per vectorized load (1, 2, 4, 8)
// OutScalarPerVector:  Elements per vectorized store (1, 2, 4, 8)
// ============================================================================
using KernelInstance =
    ck::tensor_operation::device::DeviceSoftmaxImpl<ck::half_t,  // InDataType
                                                    float,       // AccDataType
                                                    ck::half_t,  // OutDataType
                                                    PassThrough, // InElementwiseOperation
                                                    PassThrough, // AccElementwiseOperation
                                                    3,           // Rank
                                                    1,           // NumReduceDim
                                                    256,         // BlockSize
                                                    8,           // ClusterM
                                                    32,          // ClusterK
                                                    1,           // SliceM
                                                    8,           // SliceK
                                                    1,           // SrcVecDim (0=M, 1=K)
                                                    8,           // SrcScalarPerVector
                                                    8            // OutScalarPerVector
                                                    >;
// ============================================================================

extern "C" __attribute__((visibility("default"))) float run_kernel(const void* in_dev,
                                                                   void* out_dev,
                                                                   const int64_t* lengths,
                                                                   const int64_t* strides,
                                                                   int ndims,
                                                                   const int* reduce_dims,
                                                                   int n_reduce_dims,
                                                                   double alpha,
                                                                   double beta,
                                                                   bool time_kernel,
                                                                   int warmup,
                                                                   int nrepeat)
{
    KernelInstance op;

    std::vector<ck::index_t> ck_lengths(lengths, lengths + ndims);
    std::vector<ck::index_t> ck_strides(strides, strides + ndims);
    std::vector<int> ck_reduce_dims(reduce_dims, reduce_dims + n_reduce_dims);

    auto arg = op.MakeArgumentPointer(ck_lengths,
                                      ck_strides,
                                      ck_reduce_dims,
                                      alpha,
                                      beta,
                                      in_dev,
                                      out_dev,
                                      PassThrough{},
                                      PassThrough{});

    if(!op.IsSupportedArgument(arg.get()))
        return -1.0f;

    StreamConfig config;
    config.stream_id_   = nullptr;
    config.time_kernel_ = time_kernel;
    config.cold_niters_ = warmup;
    config.nrepeat_     = nrepeat;

    auto invoker = op.MakeInvokerPointer();
    float ms     = invoker->Run(arg.get(), config);

    (void)hipDeviceSynchronize();
    return ms;
}
```

## test_harness.py

```python
#!/usr/bin/env python3
"""GEAK Test Harness - Softmax PoC

Loads baseline and optimized CK softmax kernel .so files via ctypes,
verifies the optimized kernel against the baseline (ground truth), and
reports bandwidth and speedup.

Modes:
    --correctness     Verify optimized against baseline on HARNESS_SHAPES
    --profile         Run optimized kernel once per PROFILE_SHAPE (for rocprofv3)
    --benchmark       Benchmark both kernels on HARNESS_SHAPES, report speedup
    --full-benchmark  Benchmark both kernels on ALL_SHAPES, report speedup

Usage:
    python test_harness.py libbaseline.so liboptimized.so --correctness
    python test_harness.py libbaseline.so liboptimized.so --benchmark
    python test_harness.py libbaseline.so liboptimized.so --benchmark --iterations 50
    python test_harness.py libbaseline.so liboptimized.so --full-benchmark
    python test_harness.py libbaseline.so liboptimized.so --profile
"""

import argparse
import ctypes
import os
import statistics
import sys
from pathlib import Path

import torch

# -- Shape lists (sorted by element count) ------------------------------------
# 3D softmax shapes: (Batch, SeqLen, Hidden) with reduction on last dim.

ALL_SHAPES: list[list[int]] = [
    [1, 8, 256],
    [1, 8, 1024],
    [1, 8, 4096],
    [1, 32, 512],
    [4, 16, 1024],
    [1, 32, 4096],
    [4, 32, 1024],
    [2, 64, 1024],
    [4, 32, 2048],
    [8, 32, 1024],
    [4, 64, 2048],
    [8, 64, 1024],
    [8, 128, 1024],
    [8, 128, 2048],
    [16, 64, 2048],
    [8, 128, 4096],
    [16, 128, 2048],
    [32, 64, 2048],
    [16, 128, 4096],
    [32, 128, 2048],
    [32, 64, 4096],
    [32, 128, 4096],
    [64, 128, 2048],
    [64, 128, 4096],
    [128, 128, 4096],
]

HARNESS_SHAPES: list[list[int]] = ALL_SHAPES

PROFILE_SHAPES: list[list[int]] = [ALL_SHAPES[i] for i in range(0, len(ALL_SHAPES), len(ALL_SHAPES) // 5)][:5]


# -- Kernel loading via ctypes ------------------------------------------------


def load_kernel(path: str):
    """Load a kernel .so and set up the run_kernel function signature."""
    lib = ctypes.CDLL(path)
    lib.run_kernel.restype = ctypes.c_float
    lib.run_kernel.argtypes = [
        ctypes.c_void_p,  # in_dev
        ctypes.c_void_p,  # out_dev
        ctypes.POINTER(ctypes.c_int64),  # lengths
        ctypes.POINTER(ctypes.c_int64),  # strides
        ctypes.c_int,  # ndims
        ctypes.POINTER(ctypes.c_int),  # reduce_dims
        ctypes.c_int,  # n_reduce_dims
        ctypes.c_double,  # alpha
        ctypes.c_double,  # beta
        ctypes.c_bool,  # time_kernel
        ctypes.c_int,  # warmup
        ctypes.c_int,  # nrepeat
    ]
    return lib


def call_kernel(
    lib,
    x: torch.Tensor,
    y: torch.Tensor,
    reduce_dims: list[int],
    alpha: float = 1.0,
    beta: float = 0.0,
    time_kernel: bool = False,
    warmup: int = 5,
    nrepeat: int = 50,
) -> float:
    """Call run_kernel from a loaded .so. Returns time in ms (0 if not timing)."""
    ndims = x.ndim
    lengths = (ctypes.c_int64 * ndims)(*x.shape)
    strides = (ctypes.c_int64 * ndims)(*x.stride())
    rdims = (ctypes.c_int * len(reduce_dims))(*reduce_dims)

    torch.cuda.synchronize()
    ms = lib.run_kernel(
        x.data_ptr(),
        y.data_ptr(),
        lengths,
        strides,
        ndims,
        rdims,
        len(reduce_dims),
        alpha,
        beta,
        time_kernel,
        warmup,
        nrepeat,
    )
    return ms


# -- Test logic ---------------------------------------------------------------


def run_kernel_output(
    lib, x: torch.Tensor, reduce_dim: int
) -> torch.Tensor | None:
    """Run kernel on input x. Returns output tensor, or None if unsupported."""
    y = torch.empty_like(x)
    ms = call_kernel(lib, x, y, [reduce_dim], alpha=1.0, beta=0.0, time_kernel=False)
    if ms < 0:
        return None
    return y


def benchmark_kernel(
    lib,
    shape: list[int],
    reduce_dim: int,
    warmup: int = 5,
    nrepeat: int = 20,
) -> float:
    """Time the kernel. Returns average time in ms, or -1 if unsupported."""
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    y = torch.empty_like(x)

    ms = call_kernel(
        lib,
        x,
        y,
        [reduce_dim],
        alpha=1.0,
        beta=0.0,
        time_kernel=True,
        warmup=warmup,
        nrepeat=nrepeat,
    )
    return ms


def compute_bandwidth(shape: list[int], time_ms: float) -> float:
    """Compute effective bandwidth in GB/s (read input + write output, FP16)."""
    numel = 1
    for s in shape:
        numel *= s
    bytes_moved = numel * 2 * 2  # 2 bytes per FP16, read + write
    return bytes_moved / time_ms / 1e6  # bytes / ms = KB/s, / 1e6 = GB/s


# -- Mode implementations ----------------------------------------------------


def mode_correctness(base_lib, opt_lib, shapes: list[list[int]], reduce_dim_arg: int) -> bool:
    """Verify optimized kernel against baseline on each shape. Returns True if all pass."""
    torch.manual_seed(42)
    any_failed = False

    for shape in shapes:
        reduce_dim = reduce_dim_arg if reduce_dim_arg >= 0 else len(shape) - 1
        x = torch.randn(shape, dtype=torch.float16, device="cuda")

        y_base = run_kernel_output(base_lib, x, reduce_dim)
        if y_base is None:
            print(f"  SKIP  {shape}  (baseline: UNSUPPORTED)")
            continue

        y_opt = run_kernel_output(opt_lib, x, reduce_dim)
        if y_opt is None:
            print(f"  SKIP  {shape}  (optimized: UNSUPPORTED)")
            continue

        try:
            torch.testing.assert_close(y_opt, y_base, atol=1e-3, rtol=1e-3)
            max_err = (y_opt.float() - y_base.float()).abs().max().item()
            print(f"  PASS  {shape}  max_err={max_err:.2e}")
        except AssertionError as e:
            max_err = (y_opt.float() - y_base.float()).abs().max().item()
            print(f"  FAIL  {shape}  max_err={max_err:.2e}")
            print(f"        {e}")
            any_failed = True

    return not any_failed


def mode_profile(opt_lib, shapes: list[list[int]], reduce_dim_arg: int) -> None:
    """Run optimized kernel once per shape for profiler capture."""
    for shape in shapes:
        reduce_dim = reduce_dim_arg if reduce_dim_arg >= 0 else len(shape) - 1
        x = torch.randn(shape, dtype=torch.float16, device="cpu").to("cuda")
        y = torch.empty_like(x)
        ms = call_kernel(opt_lib, x, y, [reduce_dim], alpha=1.0, beta=0.0, time_kernel=False)
        if ms < 0:
            print(f"  SKIP  {shape}  (UNSUPPORTED)")


def mode_benchmark(
    base_lib, opt_lib, shapes: list[list[int]], reduce_dim_arg: int,
    warmup: int, nrepeat: int, base_label: str, opt_label: str,
) -> bool:
    """Benchmark both kernels on shapes. Returns True if all shapes ran."""
    opt_times_us: list[float] = []
    base_times_us: list[float] = []
    speedups: list[float] = []

    print(f"{'Shape':>30s}  {'Baseline ms':>11s}  {'Optimized ms':>12s}  {'BW GB/s':>8s}  {'Speedup':>7s}")
    print("-" * 80)

    for shape in shapes:
        reduce_dim = reduce_dim_arg if reduce_dim_arg >= 0 else len(shape) - 1

        ms_base = benchmark_kernel(base_lib, shape, reduce_dim, warmup=warmup, nrepeat=nrepeat)
        if ms_base < 0:
            print(f"{str(shape):>30s}  {'UNSUPPORTED':>11s}")
            continue

        ms_opt = benchmark_kernel(opt_lib, shape, reduce_dim, warmup=warmup, nrepeat=nrepeat)
        if ms_opt < 0:
            print(f"{str(shape):>30s}  {ms_base:11.4f}  {'UNSUPPORTED':>12s}")
            continue

        bw_opt = compute_bandwidth(shape, ms_opt)
        speedup = ms_base / ms_opt if ms_opt > 0 else float("inf")

        base_times_us.append(ms_base * 1000)
        opt_times_us.append(ms_opt * 1000)
        speedups.append(speedup)

        print(f"{str(shape):>30s}  {ms_base:11.4f}  {ms_opt:12.4f}  {bw_opt:8.1f}  {speedup:6.2f}x")

    if not opt_times_us:
        print("\nNo shapes were supported by both kernels.")
        return False

    median_opt = statistics.median(opt_times_us)
    median_base = statistics.median(base_times_us)
    mean_speedup = statistics.mean(speedups)
    median_speedup = median_base / median_opt if median_opt > 0 else float("inf")

    print()
    print(f"Shapes benchmarked: {len(opt_times_us)}")
    print(f"median_wall_time_us: {median_opt:.2f}")
    print(f"median_baseline_us:  {median_base:.2f}")
    print(f"median_speedup:      {median_speedup:.4f}")
    print(f"mean_speedup:        {mean_speedup:.4f}")

    return True


# -- Main ---------------------------------------------------------------------


def get_iterations(args_iterations: int | None) -> int:
    """Resolve iteration count: CLI flag -> env var -> default 20."""
    if args_iterations is not None:
        return args_iterations
    env_val = os.environ.get("GEAK_BENCHMARK_ITERATIONS")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return 20


def main():
    parser = argparse.ArgumentParser(description="GEAK Softmax Test Harness")
    parser.add_argument("baseline", help="Path to baseline kernel .so (ground truth)")
    parser.add_argument("optimized", help="Path to optimized kernel .so")

    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--correctness", action="store_true", help="Verify optimized against baseline")
    modes.add_argument("--profile", action="store_true", help="Run optimized kernel once per shape (for rocprofv3)")
    modes.add_argument("--benchmark", action="store_true", help="Benchmark on HARNESS_SHAPES")
    modes.add_argument("--full-benchmark", action="store_true", help="Benchmark on ALL_SHAPES")

    parser.add_argument("--iterations", type=int, default=None, help="Timed iterations (default: env GEAK_BENCHMARK_ITERATIONS or 20)")
    parser.add_argument("--reduce-dim", type=int, default=-1, help="Reduction dimension (default: last dim)")
    args = parser.parse_args()

    for path in [args.baseline, args.optimized]:
        if not Path(path).exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)

    base_label = Path(args.baseline).name
    opt_label = Path(args.optimized).name
    base_lib = load_kernel(str(Path(args.baseline).resolve()))
    opt_lib = load_kernel(str(Path(args.optimized).resolve()))

    if args.correctness:
        print(f"Correctness check: {opt_label} vs {base_label} (ground truth)")
        print(f"Shapes: {len(HARNESS_SHAPES)}")
        print()
        ok = mode_correctness(base_lib, opt_lib, HARNESS_SHAPES, args.reduce_dim)
        print()
        print("RESULT: PASS" if ok else "RESULT: FAIL")
        sys.exit(0 if ok else 1)

    elif args.profile:
        print(f"Profile mode: {opt_label}")
        print(f"Shapes: {len(PROFILE_SHAPES)}")
        print()
        mode_profile(opt_lib, PROFILE_SHAPES, args.reduce_dim)

    elif args.benchmark:
        nrepeat = get_iterations(args.iterations)
        print(f"Benchmark: {opt_label} vs {base_label}")
        print(f"Shapes: {len(HARNESS_SHAPES)}, iterations: {nrepeat}")
        print()
        ok = mode_benchmark(base_lib, opt_lib, HARNESS_SHAPES, args.reduce_dim,
                            warmup=5, nrepeat=nrepeat, base_label=base_label, opt_label=opt_label)
        sys.exit(0 if ok else 1)

    elif args.full_benchmark:
        nrepeat = get_iterations(args.iterations)
        print(f"Full benchmark: {opt_label} vs {base_label}")
        print(f"Shapes: {len(ALL_SHAPES)}, iterations: {nrepeat}")
        print()
        ok = mode_benchmark(base_lib, opt_lib, ALL_SHAPES, args.reduce_dim,
                            warmup=5, nrepeat=nrepeat, base_label=base_label, opt_label=opt_label)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

## compile.py

```python
#!/usr/bin/env python3
"""Build kernel .so files with GPU architecture auto-detected from PyTorch.

Usage:
    ./compile.py                # build all
    ./compile.py optimized      # rebuild only optimized kernel
    ./compile.py clean          # clean build artifacts
    ./compile.py --arch gfx942  # override auto-detection
"""

import argparse
import subprocess
import sys


def detect_arch() -> str:
    """Detect GPU architecture from PyTorch."""
    try:
        import torch

        if not torch.cuda.is_available():
            sys.exit("Error: No GPU available (torch.cuda.is_available() = False)")
        props = torch.cuda.get_device_properties(0)
        arch = props.gcnArchName.split(":")[0]
        return arch
    except ImportError:
        sys.exit(
            "Error: PyTorch not found. Install it or use: make ARCH=gfx950 directly"
        )


def main():
    parser = argparse.ArgumentParser(description="Build GEAK kernel .so files")
    parser.add_argument(
        "targets", nargs="*", default=["all"], help="Make targets (default: all)"
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="GPU architecture (default: auto-detect from PyTorch)",
    )
    args = parser.parse_args()

    arch = args.arch or detect_arch()
    print(f"GPU architecture: {arch}")

    cmd = ["make", f"ARCH={arch}"] + args.targets
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
```

## Makefile

```makefile
# GEAK Test Harness - Kernel .so build rules
# Standalone build (independent of CK's CMake).
#
# Preferred: use compile.py which auto-detects GPU arch from PyTorch.
# Direct usage: make ARCH=gfx950 [target]

HIPCC      ?= /opt/rocm/bin/hipcc
CK_INCLUDE  = ../../../include

ifndef ARCH
  $(error ARCH not set. Use compile.py for auto-detection, or: make ARCH=gfx950)
endif

HIPCC_FLAGS = -shared -fPIC -O3 \
              --offload-arch=$(ARCH) \
              -I$(CK_INCLUDE) \
              -std=c++17

.PHONY: all baseline optimized clean

all: baseline optimized

baseline: libbaseline.so
optimized: liboptimized.so

libbaseline.so: baseline.cpp
	@echo "Building baseline for $(ARCH)..."
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

liboptimized.so: optimized.cpp
	@echo "Building optimized for $(ARCH)..."
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

clean:
	rm -f libbaseline.so liboptimized.so
```

## Adapting for Other Kernels

When adapting this recipe for a different CK kernel family, change three things:

1. **C ABI signature** in both `.cpp` files -- match the kernel's input/output arguments
2. **`load_kernel()` and `call_kernel()`** in `test_harness.py` -- mirror the new C ABI with ctypes
3. **Shape lists** -- replace with shapes appropriate for the kernel family

Everything else (`compile.py`, `Makefile`, 4-mode structure, baseline-vs-optimized
comparison, benchmark reporting) stays identical.
