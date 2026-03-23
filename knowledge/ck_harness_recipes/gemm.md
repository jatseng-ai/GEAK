# CK Harness Recipe: GEMM

## Kernel Family

- Device types: `DeviceGemm_Xdl_CShuffle`, `DeviceGemmXdl`
- Header: `ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp`
- CK examples: `01_gemm/gemm_xdl_fp16.cpp`
- API style: `MakeArgument` (value-based, not pointer)
- Needs workspace: No

## Key Differences from Softmax

- Uses **value-based** `MakeArgument` + `MakeInvoker()` instead of pointer variants.
- `IsSupportedArgument(argument)` not `IsSupportedArgument(argument.get())`.
- `invoker.Run(argument, StreamConfig{...})` not `invoker->Run(arg.get(), config)`.
- Three tensor pointers (A, B, C) with scalar M, N, K dimensions and strides.
- Row-major layout assumed for all matrices (RowMajor: stride = N for A, stride = N for B, stride = N for C).

## Tunable Template Parameters

`DeviceGemm_Xdl_CShuffle` has many template parameters. The main tuning knobs:

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| NumGemmKPrefetchStage | Prefetch pipeline depth | 1 |
| BlockSize | Threads per block | 128, 256 |
| MPerBlock | M tile size | 64, 128, 256 |
| NPerBlock | N tile size | 64, 128, 256 |
| KPerBlock | K tile size | 16, 32, 64 |
| AK1 | A vector load width | 2, 4, 8 |
| BK1 | B vector load width | 2, 4, 8 |
| MPerXDL | M per XDL instruction | 16, 32 |
| NPerXDL | N per XDL instruction | 16, 32 |
| MXdlPerWave | M XDLs per wave | 2, 4, 8 |
| NXdlPerWave | N XDLs per wave | 2, 4, 8 |
| ABlockTransfer* | A tile transfer config | ThreadCluster sizes, access order, vector dims |
| BBlockTransfer* | B tile transfer config | ThreadCluster sizes, access order, vector dims |
| CShuffle* | C shuffle config | MXdlPerWave, NXdlPerWave cluster lengths |

Constraints: MPerBlock, NPerBlock must be multiples of MPerXDL, NPerXDL respectively.
BlockSize must accommodate the thread cluster configurations.

## C ABI Signature

```cpp
extern "C" __attribute__((visibility("default"))) float run_kernel(
    const void* a_dev,
    const void* b_dev,
    void* c_dev,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool time_kernel,
    int warmup,
    int nrepeat)
```

## baseline.cpp

```cpp
// GEAK Test Harness - GEMM Baseline Kernel
// Compiled as a standalone shared library (.so).
// Uses CK headers + HIP runtime only, no CK build system dependency.

#include <cstdint>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using CDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::RowMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault =
    ck::tensor_operation::device::GemmSpecialization::Default;

// ============================================================================
// TUNING PARAMETERS — baseline configuration from CK example
// ============================================================================
using KernelInstance = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
    ALayout, BLayout, CLayout,
    ADataType, BDataType, CDataType, AccDataType, CShuffleDataType,
    PassThrough, PassThrough, PassThrough,
    GemmDefault,
    1,              // NumGemmKPrefetchStage
    256,            // BlockSize
    256,            // MPerBlock
    128,            // NPerBlock
    32,             // KPerBlock
    8,              // AK1
    2,              // BK1
    16,             // MPerXDL
    16,             // NPerXDL
    8,              // MXdlPerWave
    4,              // NXdlPerWave
    ck::Sequence<4, 64, 1>,  // ABlockTransfer ThreadCluster Lengths
    ck::Sequence<1, 0, 2>,   // ABlockTransfer ThreadCluster ArrangeOrder
    ck::Sequence<1, 0, 2>,   // ABlockTransfer SrcAccessOrder
    2,              // ABlockTransfer SrcVectorDim
    8,              // ABlockTransfer SrcScalarPerVector
    8,              // ABlockTransfer DstScalarPerVector_K1
    1,              // ABlockLds AddExtraM
    ck::Sequence<8, 32, 1>,  // BBlockTransfer ThreadCluster Lengths
    ck::Sequence<0, 2, 1>,   // BBlockTransfer ThreadCluster ArrangeOrder
    ck::Sequence<0, 2, 1>,   // BBlockTransfer SrcAccessOrder
    1,              // BBlockTransfer SrcVectorDim
    4,              // BBlockTransfer SrcScalarPerVector
    2,              // BBlockTransfer DstScalarPerVector_K1
    0,              // BBlockLds AddExtraN
    1,              // CShuffle MXdlPerWavePerShuffle
    2,              // CShuffle NXdlPerWavePerShuffle
    ck::Sequence<1, 16, 1, 16>,  // CBlockTransferClusterLengths
    4,              // CBlockTransfer ScalarPerVector_NWaveNPerXdl
    ck::LoopScheduler::Interwave,
    ck::PipelineVersion::v1>;
// ============================================================================

extern "C" __attribute__((visibility("default"))) float run_kernel(
    const void* a_dev,
    const void* b_dev,
    void* c_dev,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool time_kernel,
    int warmup,
    int nrepeat)
{
    KernelInstance gemm;
    auto invoker = gemm.MakeInvoker();

    auto argument = gemm.MakeArgument(
        static_cast<const ADataType*>(a_dev),
        static_cast<const BDataType*>(b_dev),
        static_cast<CDataType*>(c_dev),
        M, N, K,
        lda, ldb, ldc,
        PassThrough{}, PassThrough{}, PassThrough{});

    if(!gemm.IsSupportedArgument(argument))
        return -1.0f;

    StreamConfig config;
    config.stream_id_   = nullptr;
    config.time_kernel_ = time_kernel;
    config.cold_niters_ = warmup;
    config.nrepeat_     = nrepeat;

    float ms = invoker.Run(argument, config);

    (void)hipDeviceSynchronize();
    return ms;
}
```

## optimized.cpp

Same as `baseline.cpp` but with the TUNING PARAMETERS comment changed to
"modify these to optimize the kernel" and documentation of each parameter.
The `extern "C" run_kernel` body is identical.

## test_harness.py

```python
#!/usr/bin/env python3
"""GEAK Test Harness - GEMM

Loads baseline and optimized CK GEMM kernel .so files via ctypes.
Verifies optimized against baseline (ground truth), reports TFLOPS and speedup.
"""

import argparse
import ctypes
import os
import statistics
import sys
from pathlib import Path

import torch

# -- Shape lists (sorted by element count) ------------------------------------
# GEMM shapes: [M, N, K] — row-major layout, fp16.

ALL_SHAPES: list[list[int]] = [
    [128, 128, 64],
    [256, 256, 128],
    [512, 512, 256],
    [512, 512, 512],
    [1024, 1024, 512],
    [1024, 1024, 1024],
    [2048, 1024, 512],
    [2048, 2048, 1024],
    [1024, 4096, 1024],
    [2048, 2048, 2048],
    [4096, 2048, 1024],
    [4096, 4096, 1024],
    [4096, 4096, 2048],
    [4096, 4096, 4096],
    [8192, 4096, 2048],
    [8192, 8192, 2048],
    [8192, 8192, 4096],
    [1024, 16384, 1024],
    [16384, 4096, 1024],
    [16384, 16384, 1024],
]

HARNESS_SHAPES: list[list[int]] = ALL_SHAPES

PROFILE_SHAPES: list[list[int]] = [ALL_SHAPES[i] for i in range(0, len(ALL_SHAPES), len(ALL_SHAPES) // 5)][:5]


def load_kernel(path: str):
    """Load a GEMM kernel .so and set up the run_kernel function signature."""
    lib = ctypes.CDLL(path)
    lib.run_kernel.restype = ctypes.c_float
    lib.run_kernel.argtypes = [
        ctypes.c_void_p,   # a_dev
        ctypes.c_void_p,   # b_dev
        ctypes.c_void_p,   # c_dev
        ctypes.c_int64,    # M
        ctypes.c_int64,    # N
        ctypes.c_int64,    # K
        ctypes.c_int64,    # lda
        ctypes.c_int64,    # ldb
        ctypes.c_int64,    # ldc
        ctypes.c_bool,     # time_kernel
        ctypes.c_int,      # warmup
        ctypes.c_int,      # nrepeat
    ]
    return lib


def call_kernel(
    lib,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    time_kernel: bool = False,
    warmup: int = 5,
    nrepeat: int = 50,
) -> float:
    """Call run_kernel from a loaded .so. Returns time in ms (0 if not timing)."""
    M, K = a.shape
    _, N = b.shape

    torch.cuda.synchronize()
    ms = lib.run_kernel(
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        M, N, K,
        K,  # lda (row-major: stride = K)
        N,  # ldb (row-major: stride = N)
        N,  # ldc (row-major: stride = N)
        time_kernel,
        warmup,
        nrepeat,
    )
    return ms


def run_kernel_output(lib, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | None:
    """Run kernel. Returns output C tensor, or None if unsupported."""
    M, _ = a.shape
    _, N = b.shape
    c = torch.empty(M, N, dtype=torch.float16, device="cuda")
    ms = call_kernel(lib, a, b, c, time_kernel=False)
    if ms < 0:
        return None
    return c


def benchmark_kernel(lib, shape: list[int], warmup: int = 5, nrepeat: int = 20) -> float:
    """Time the kernel. Returns average time in ms, or -1 if unsupported."""
    M, N, K = shape
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    c = torch.empty(M, N, dtype=torch.float16, device="cuda")
    return call_kernel(lib, a, b, c, time_kernel=True, warmup=warmup, nrepeat=nrepeat)


def compute_tflops(shape: list[int], time_ms: float) -> float:
    """Compute TFLOPS for GEMM (2*M*N*K flops)."""
    M, N, K = shape
    flops = 2.0 * M * N * K
    return flops / time_ms / 1e9


# -- Mode implementations (same structure as softmax) -------------------------
# mode_correctness, mode_profile, mode_benchmark follow the identical pattern
# as in the softmax recipe, with the GEMM-specific call_kernel / run_kernel_output
# and TFLOPS instead of bandwidth.

# ... (full implementations omitted for brevity — follow the softmax pattern
#      replacing bandwidth with TFLOPS and adapting tensor creation to A/B/C)
```

## compile.py and Makefile

Identical to the softmax recipe. No changes needed.
The `CK_INCLUDE` path in the Makefile should be adjusted to point at the CK
headers relative to the harness location.

## Shape Lists

GEMM shapes are `[M, N, K]` tuples. Row-major layout is assumed. For the Python
harness, tensors are: `a = torch.randn(M, K, ...)`, `b = torch.randn(K, N, ...)`,
`c = torch.empty(M, N, ...)`. Strides for row-major: `lda=K`, `ldb=N`, `ldc=N`.

## Correctness Notes

- GEMM accumulates in float (AccDataType), so fp16 outputs should match within
  `atol=1e-2, rtol=1e-2` (looser than softmax due to accumulation over K).
- For large K dimensions, consider `atol=5e-2` if exact match is difficult.
