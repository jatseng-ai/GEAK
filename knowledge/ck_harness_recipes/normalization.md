# CK Harness Recipe: Normalization (LayerNorm / GroupNorm / BatchNorm)

## Kernel Family

- Device types: `DeviceNormalizationFwdImpl`, `DeviceNormalizationBwdImpl`, `DeviceBatchNormFwdImpl`, `DeviceBatchNormBwdImpl`
- Header: `ck/tensor_operation/gpu/device/impl/device_normalization_fwd_impl.hpp`
- CK examples: `27_layernorm2d_fwd/layernorm2d_fwd_fp16.cpp`
- API style: `MakeArgumentPointer` (pointer-based, same as softmax)
- Needs workspace: **Yes** — `GetWorkSpaceSize()` + `SetWorkSpacePointer()` required

## Key Differences from Softmax

- **More tensor pointers**: x, gamma, beta, y, save_mean, save_inv_std (6 device pointers)
- **Workspace allocation**: Must call `GetWorkSpaceSize()` and `SetWorkSpacePointer()` between
  `MakeArgumentPointer()` and `Run()`.
- **Extra strides**: Separate strides for x, gamma, beta, y, save_mean, save_inv_std.
- **Epsilon parameter**: Normalization requires an epsilon (e.g. 1e-4).
- **Additional vector parameters**: GammaVecDim, GammaScalarPerVector, BetaVecDim, BetaScalarPerVector,
  SaveMeanInvStdScalarPerVector — in addition to the standard XY/Src/Out vectors.

## Tunable Template Parameters

| Parameter | Position | Type | Constraints |
|-----------|----------|------|-------------|
| BlockSize | 10 | int | Power of 2: 64, 128, 256, 512, 1024 |
| ClusterM | 11 | int | ClusterM * ClusterK must equal BlockSize |
| ClusterK | 12 | int | ClusterM * ClusterK must equal BlockSize |
| SliceM | 13 | int | >= 1 |
| SliceK | 14 | int | Must align with vector size |
| XYVectorDim | 15 | int | 0 = vectorize along M, 1 = vectorize along K |
| SrcScalarPerVector | 16 | int | Power of 2: 1, 2, 4, 8 |
| GammaVecDim | 17 | int | 0 or 1 |
| GammaScalarPerVector | 18 | int | Power of 2: 1, 2, 4, 8 |
| BetaVecDim | 19 | int | 0 or 1 |
| BetaScalarPerVector | 20 | int | Power of 2: 1, 2, 4, 8 |
| YScalarPerVector | 21 | int | Power of 2: 1, 2, 4, 8 |
| SaveMeanInvStdScalarPerVector | 22 | int | Power of 2: 1, 2, 4, 8 |

## C ABI Signature

```cpp
extern "C" __attribute__((visibility("default"))) float run_kernel(
    const void* x_dev,
    const void* gamma_dev,
    const void* beta_dev,
    void* y_dev,
    void* save_mean_dev,
    void* save_inv_std_dev,
    int64_t M,
    int64_t N,
    double epsilon,
    bool time_kernel,
    int warmup,
    int nrepeat)
```

Returns elapsed time in ms when `time_kernel=true`, 0.0 otherwise, or -1.0f if
the configuration is unsupported.

## baseline.cpp

```cpp
// GEAK Test Harness - Normalization Baseline Kernel
// Compiled as a standalone shared library (.so).

#include <cstdint>
#include <vector>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_fwd_impl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using XDataType              = ck::half_t;
using GammaDataType          = ck::half_t;
using BetaDataType           = ck::half_t;
using YDataType              = ck::half_t;
using SaveMeanInvStdDataType = float;
using ComputeDataType        = float;
using PassThrough            = ck::tensor_operation::element_wise::PassThrough;

constexpr int Rank         = 2;
constexpr int NumReduceDim = 1;

// ============================================================================
// TUNING PARAMETERS — baseline configuration from CK example
// ============================================================================
using KernelInstance =
    ck::tensor_operation::device::DeviceNormalizationFwdImpl<XDataType,
                                                             GammaDataType,
                                                             BetaDataType,
                                                             ComputeDataType,
                                                             YDataType,
                                                             SaveMeanInvStdDataType,
                                                             PassThrough,
                                                             Rank,
                                                             NumReduceDim,
                                                             256, // BlockSize
                                                             8,   // ClusterM
                                                             32,  // ClusterK
                                                             1,   // SliceM
                                                             8,   // SliceK
                                                             1,   // XYVectorDim (0=M, 1=K)
                                                             8,   // SrcScalarPerVector
                                                             1,   // GammaVecDim
                                                             8,   // GammaScalarPerVector
                                                             1,   // BetaVecDim
                                                             8,   // BetaScalarPerVector
                                                             8,   // YScalarPerVector
                                                             1>;  // SaveMeanInvStdScalarPerVector
// ============================================================================

extern "C" __attribute__((visibility("default"))) float run_kernel(
    const void* x_dev,
    const void* gamma_dev,
    const void* beta_dev,
    void* y_dev,
    void* save_mean_dev,
    void* save_inv_std_dev,
    int64_t M,
    int64_t N,
    double epsilon,
    bool time_kernel,
    int warmup,
    int nrepeat)
{
    KernelInstance op;

    // LayerNorm 2D: x is [M, N], gamma/beta are [N], y is [M, N], mean/invstd are [M]
    auto argument_ptr = op.MakeArgumentPointer(
        {static_cast<ck::index_t>(M), static_cast<ck::index_t>(N)},  // lengths
        {static_cast<ck::index_t>(N), 1},                             // x strides (row-major)
        {0, 1},                                                        // gamma strides (broadcast M, packed K)
        {0, 1},                                                        // beta strides
        {static_cast<ck::index_t>(N), 1},                             // y strides
        {1},                                                           // save_mean strides
        {1},                                                           // save_inv_std strides
        {1},                                                           // reduce dims
        epsilon,
        x_dev,
        gamma_dev,
        beta_dev,
        y_dev,
        save_mean_dev,
        save_inv_std_dev,
        PassThrough{});

    if(!op.IsSupportedArgument(argument_ptr.get()))
        return -1.0f;

    // Workspace allocation
    size_t workspace_sz = op.GetWorkSpaceSize(argument_ptr.get());
    void* workspace_ptr = nullptr;
    if(workspace_sz > 0)
    {
        (void)hipMalloc(&workspace_ptr, workspace_sz);
        op.SetWorkSpacePointer(argument_ptr.get(), workspace_ptr);
    }

    StreamConfig config;
    config.stream_id_   = nullptr;
    config.time_kernel_ = time_kernel;
    config.cold_niters_ = warmup;
    config.nrepeat_     = nrepeat;

    auto invoker_ptr = op.MakeInvokerPointer();
    float ms = invoker_ptr->Run(argument_ptr.get(), config);

    (void)hipDeviceSynchronize();

    if(workspace_ptr)
        (void)hipFree(workspace_ptr);

    return ms;
}
```

## optimized.cpp

Same as `baseline.cpp` but with the TUNING PARAMETERS comment changed to
"modify these to optimize the kernel" and documentation of each parameter.

## test_harness.py

```python
#!/usr/bin/env python3
"""GEAK Test Harness - Normalization (LayerNorm2D)

Loads baseline and optimized CK normalization kernel .so files via ctypes.
Verifies optimized against baseline (ground truth), reports bandwidth and speedup.
"""

import argparse
import ctypes
import os
import statistics
import sys
from pathlib import Path

import torch

# -- Shape lists: [M, N] where N is the normalization dimension ---------------

ALL_SHAPES: list[list[int]] = [
    [64, 256],
    [128, 256],
    [256, 512],
    [512, 512],
    [256, 1024],
    [512, 1024],
    [1024, 1024],
    [512, 2048],
    [1024, 2048],
    [2048, 1024],
    [2048, 2048],
    [1024, 4096],
    [2048, 4096],
    [4096, 2048],
    [4096, 4096],
    [8192, 1024],
    [8192, 2048],
    [8192, 4096],
    [16384, 1024],
    [16384, 4096],
]

HARNESS_SHAPES: list[list[int]] = ALL_SHAPES

PROFILE_SHAPES: list[list[int]] = [ALL_SHAPES[i] for i in range(0, len(ALL_SHAPES), len(ALL_SHAPES) // 5)][:5]


def load_kernel(path: str):
    """Load a normalization kernel .so and set up the run_kernel function signature."""
    lib = ctypes.CDLL(path)
    lib.run_kernel.restype = ctypes.c_float
    lib.run_kernel.argtypes = [
        ctypes.c_void_p,   # x_dev
        ctypes.c_void_p,   # gamma_dev
        ctypes.c_void_p,   # beta_dev
        ctypes.c_void_p,   # y_dev
        ctypes.c_void_p,   # save_mean_dev
        ctypes.c_void_p,   # save_inv_std_dev
        ctypes.c_int64,    # M
        ctypes.c_int64,    # N
        ctypes.c_double,   # epsilon
        ctypes.c_bool,     # time_kernel
        ctypes.c_int,      # warmup
        ctypes.c_int,      # nrepeat
    ]
    return lib


def call_kernel(
    lib,
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    save_mean: torch.Tensor,
    save_inv_std: torch.Tensor,
    epsilon: float = 1e-5,
    time_kernel: bool = False,
    warmup: int = 5,
    nrepeat: int = 50,
) -> float:
    """Call run_kernel from a loaded .so. Returns time in ms (0 if not timing)."""
    M, N = x.shape

    torch.cuda.synchronize()
    ms = lib.run_kernel(
        x.data_ptr(),
        gamma.data_ptr(),
        beta.data_ptr(),
        y.data_ptr(),
        save_mean.data_ptr(),
        save_inv_std.data_ptr(),
        M, N,
        epsilon,
        time_kernel,
        warmup,
        nrepeat,
    )
    return ms


def run_kernel_output(lib, x: torch.Tensor, gamma: torch.Tensor,
                      beta: torch.Tensor) -> torch.Tensor | None:
    """Run kernel. Returns output y tensor, or None if unsupported."""
    M, N = x.shape
    y = torch.empty_like(x)
    save_mean = torch.empty(M, dtype=torch.float32, device="cuda")
    save_inv_std = torch.empty(M, dtype=torch.float32, device="cuda")

    ms = call_kernel(lib, x, gamma, beta, y, save_mean, save_inv_std,
                     epsilon=1e-5, time_kernel=False)
    if ms < 0:
        return None
    return y


def benchmark_kernel(lib, shape: list[int], warmup: int = 5, nrepeat: int = 20) -> float:
    """Time the kernel. Returns average time in ms, or -1 if unsupported."""
    M, N = shape
    x = torch.randn(M, N, dtype=torch.float16, device="cuda")
    gamma = torch.ones(N, dtype=torch.float16, device="cuda")
    beta = torch.zeros(N, dtype=torch.float16, device="cuda")
    y = torch.empty_like(x)
    save_mean = torch.empty(M, dtype=torch.float32, device="cuda")
    save_inv_std = torch.empty(M, dtype=torch.float32, device="cuda")

    return call_kernel(lib, x, gamma, beta, y, save_mean, save_inv_std,
                       epsilon=1e-5, time_kernel=True, warmup=warmup, nrepeat=nrepeat)


def compute_bandwidth(shape: list[int], time_ms: float) -> float:
    """Compute effective bandwidth in GB/s (x + gamma + beta + y, FP16, + mean + invstd, FP32)."""
    M, N = shape
    bytes_moved = (M * N * 2 * 2) + (N * 2 * 2) + (M * 4 * 2)  # x/y fp16, gamma/beta fp16, mean/invstd fp32
    return bytes_moved / time_ms / 1e6


# -- Mode implementations follow the identical 4-mode pattern as softmax ------
# mode_correctness: compare y from baseline vs optimized with torch.testing.assert_close
# mode_profile: run optimized once per shape
# mode_benchmark: time both, report bandwidth + speedup
# mode_full_benchmark: same on ALL_SHAPES
#
# ... (full implementations follow the softmax recipe pattern, replacing
#      tensor creation and call_kernel arguments for normalization)
```

## compile.py and Makefile

Identical to the softmax recipe. No changes needed.

## Workspace Allocation Pattern

This is the critical difference from softmax. Between `MakeArgumentPointer()` and `Run()`:

```cpp
size_t workspace_sz = op.GetWorkSpaceSize(argument_ptr.get());
void* workspace_ptr = nullptr;
if(workspace_sz > 0)
{
    (void)hipMalloc(&workspace_ptr, workspace_sz);
    op.SetWorkSpacePointer(argument_ptr.get(), workspace_ptr);
}
```

After `Run()`, free the workspace:
```cpp
if(workspace_ptr)
    (void)hipFree(workspace_ptr);
```

## MakeArgumentPointer Signature

The normalization `MakeArgumentPointer` takes many stride vectors. The argument order is:

```
lengths, x_strides, gamma_strides, beta_strides, y_strides,
save_mean_strides, save_inv_std_strides, reduce_dims,
epsilon, x_dev, gamma_dev, beta_dev, y_dev,
save_mean_dev, save_inv_std_dev, PassThrough{}
```

For LayerNorm2D with shape `[M, N]`:
- x strides: `{N, 1}` (row-major)
- gamma strides: `{0, 1}` (broadcast over M, packed along K)
- beta strides: `{0, 1}`
- y strides: `{N, 1}`
- mean/invstd strides: `{1}` (1D, length M)
- reduce dims: `{1}` (reduce over N)

## Correctness Notes

- LayerNorm is well-conditioned; `atol=1e-3, rtol=1e-3` should work for fp16.
- Always verify all three outputs: y, save_mean, and save_inv_std.
