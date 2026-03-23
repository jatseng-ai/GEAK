# CK Harness Recipe: Reduction

## Kernel Family

- Device type: `DeviceReduceMultiBlock`
- Header: `ck/tensor_operation/gpu/device/impl/device_reduce_multiblock.hpp`
- CK examples: `12_reduce/reduce_blockwise.cpp`, `12_reduce/reduce_blockwise_impl.hpp`
- API style: `MakeArgumentPointer` (pointer-based, same as softmax)
- Needs workspace: No

## Key Differences from Softmax

- **Output shape differs from input**: Reduced dimensions are collapsed. The output tensor
  has `Rank - NumReduceDim` dimensions (or 1 if all dims are reduced).
- **Reduction operation template parameter**: `ReduceOperation` (AVG, MAX, MIN, ADD, etc.)
  and associated elementwise operations.
- **Two sets of lengths/strides**: Separate input lengths/strides and output lengths/strides.
- **`InMemoryDataOperationEnum`**: Controls how output is combined (Set, Add, etc.).
- **Optional index output**: Some ops (MIN, MAX, AMAX) can output indices.

## Tunable Template Parameters

| Parameter | Position | Type | Constraints |
|-----------|----------|------|-------------|
| BlockSize | 13 | int | Power of 2: 64, 128, 256, 512 |
| MThreadClusterSize | 14 | int | Invariant-dimension thread distribution |
| KThreadClusterSize | 15 | int | Reduce-dimension thread distribution |
| MThreadSliceSize | 16 | int | Elements per thread along invariant dim |
| KThreadSliceSize | 17 | int | Elements per thread along reduce dim |
| InSrcVectorDim | 18 | int | 0 = vectorize along invariant, 1 = vectorize along reduce |
| InSrcVectorSize | 19 | int | Power of 2: 1, 2, 4, 8 |
| OutDstVectorSize | 20 | int | Power of 2: 1, 2, 4, 8 |

Constraint: `MThreadClusterSize * KThreadClusterSize == BlockSize`.

## C ABI Signature

The reduce C ABI is very similar to softmax since both operate on generic
lengths/strides/reduce_dims:

```cpp
extern "C" __attribute__((visibility("default"))) float run_kernel(
    const void* in_dev,
    void* out_dev,
    const int64_t* in_lengths,
    const int64_t* in_strides,
    int in_rank,
    const int64_t* out_lengths,
    const int64_t* out_strides,
    int out_rank,
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
// GEAK Test Harness - Reduction Baseline Kernel
// Compiled as a standalone shared library (.so).

#include <cstdint>
#include <array>
#include <vector>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_multiblock.hpp"

using InOutDataType = ck::half_t;
using AccDataType   = float;

constexpr ck::ReduceTensorOp ReduceOpId = ck::ReduceTensorOp::AVG;
constexpr bool PropagateNan  = true;
constexpr bool OutputIndex   = false;

using ReduceOperation        = typename ck::reduce_binary_operator<ReduceOpId>::opType;
using InElementwiseOperation  = typename ck::reduce_unary_operator<ReduceOpId, true, true>::InElementwiseOperation;
using AccElementwiseOperation = typename ck::reduce_unary_operator<ReduceOpId, true, true>::AccElementwiseOperation;

// For this recipe, we fix Rank=4, NumReduceDim=3 as a common case.
// Adjust these for the specific reduction pattern in the kernel being optimized.
constexpr int Rank         = 4;
constexpr int NumReduceDim = 3;
constexpr int NumOutDim    = (Rank - NumReduceDim == 0) ? 1 : Rank - NumReduceDim;

// ============================================================================
// TUNING PARAMETERS — baseline configuration from CK example
// ============================================================================
using KernelInstance =
    ck::tensor_operation::device::DeviceReduceMultiBlock<InOutDataType,
                                                         AccDataType,
                                                         InOutDataType,
                                                         Rank,
                                                         NumReduceDim,
                                                         ReduceOperation,
                                                         InElementwiseOperation,
                                                         AccElementwiseOperation,
                                                         ck::InMemoryDataOperationEnum::Set,
                                                         PropagateNan,
                                                         OutputIndex,
                                                         false, // HaveIndexInputIfOutputIndex
                                                         256,   // BlockSize
                                                         4,     // MThreadClusterSize
                                                         64,    // KThreadClusterSize
                                                         1,     // MThreadSliceSize
                                                         1,     // KThreadSliceSize
                                                         0,     // InSrcVectorDim
                                                         1,     // InSrcVectorSize
                                                         1>;    // OutDstVectorSize
// ============================================================================

extern "C" __attribute__((visibility("default"))) float run_kernel(
    const void* in_dev,
    void* out_dev,
    const int64_t* in_lengths,
    const int64_t* in_strides,
    int in_rank,
    const int64_t* out_lengths,
    const int64_t* out_strides,
    int out_rank,
    const int* reduce_dims,
    int n_reduce_dims,
    double alpha,
    double beta,
    bool time_kernel,
    int warmup,
    int nrepeat)
{
    KernelInstance op;

    std::array<ck::index_t, Rank> arr_in_lengths;
    std::array<ck::index_t, Rank> arr_in_strides;
    std::array<ck::index_t, NumOutDim> arr_out_lengths;
    std::array<ck::index_t, NumOutDim> arr_out_strides;
    std::array<int, NumReduceDim> arr_reduce_dims;

    for(int i = 0; i < Rank; ++i)
    {
        arr_in_lengths[i] = static_cast<ck::index_t>(in_lengths[i]);
        arr_in_strides[i] = static_cast<ck::index_t>(in_strides[i]);
    }
    for(int i = 0; i < NumOutDim; ++i)
    {
        arr_out_lengths[i] = static_cast<ck::index_t>(out_lengths[i]);
        arr_out_strides[i] = static_cast<ck::index_t>(out_strides[i]);
    }
    for(int i = 0; i < NumReduceDim; ++i)
        arr_reduce_dims[i] = reduce_dims[i];

    // Compute reduce_total_length for elementwise ops (needed for AVG)
    int64_t reduce_total_length = 1;
    for(int i = 0; i < n_reduce_dims; ++i)
        reduce_total_length *= in_lengths[reduce_dims[i]];

    InElementwiseOperation in_elementwise_op;
    AccElementwiseOperation acc_elementwise_op;
    std::tie(in_elementwise_op, acc_elementwise_op) =
        ck::reduce_unary_operator<ReduceOpId, true, true>::GetElementwiseOperator(
            static_cast<int32_t>(reduce_total_length));

    auto argument_ptr = op.MakeArgumentPointer(
        arr_in_lengths,
        arr_in_strides,
        arr_out_lengths,
        arr_out_strides,
        arr_reduce_dims,
        alpha,
        beta,
        static_cast<const InOutDataType*>(in_dev),
        nullptr,  // in_index (not used when OutputIndex=false)
        static_cast<InOutDataType*>(out_dev),
        nullptr,  // out_index
        in_elementwise_op,
        acc_elementwise_op);

    if(!op.IsSupportedArgument(argument_ptr.get()))
        return -1.0f;

    StreamConfig config;
    config.stream_id_   = nullptr;
    config.time_kernel_ = time_kernel;
    config.cold_niters_ = warmup;
    config.nrepeat_     = nrepeat;

    auto invoker_ptr = op.MakeInvokerPointer();
    float ms = invoker_ptr->Run(argument_ptr.get(), config);

    (void)hipDeviceSynchronize();
    return ms;
}
```

## optimized.cpp

Same as `baseline.cpp` but with the TUNING PARAMETERS comment changed to
"modify these to optimize the kernel" and documentation of each parameter.

## test_harness.py

```python
#!/usr/bin/env python3
"""GEAK Test Harness - Reduction

Loads baseline and optimized CK reduction kernel .so files via ctypes.
Verifies optimized against baseline (ground truth), reports bandwidth and speedup.
"""

import argparse
import ctypes
import os
import statistics
import sys
from pathlib import Path

import torch

# -- Shape lists: [D0, D1, D2, D3] with reduce_dims = [0, 1, 2] -------------
# 4D reduction shapes. Output has 1 dimension (D3).

ALL_SHAPES: list[list[int]] = [
    [4, 8, 8, 64],
    [4, 16, 8, 128],
    [8, 16, 16, 256],
    [16, 32, 16, 256],
    [16, 32, 32, 512],
    [16, 64, 32, 960],
    [32, 32, 32, 960],
    [32, 64, 32, 960],
    [16, 64, 64, 960],
    [32, 64, 64, 960],
]

REDUCE_DIMS: list[int] = [0, 1, 2]

HARNESS_SHAPES: list[list[int]] = ALL_SHAPES

PROFILE_SHAPES: list[list[int]] = [ALL_SHAPES[i] for i in range(0, len(ALL_SHAPES), len(ALL_SHAPES) // 5)][:5]


def load_kernel(path: str):
    """Load a reduction kernel .so and set up the run_kernel function signature."""
    lib = ctypes.CDLL(path)
    lib.run_kernel.restype = ctypes.c_float
    lib.run_kernel.argtypes = [
        ctypes.c_void_p,                   # in_dev
        ctypes.c_void_p,                   # out_dev
        ctypes.POINTER(ctypes.c_int64),    # in_lengths
        ctypes.POINTER(ctypes.c_int64),    # in_strides
        ctypes.c_int,                      # in_rank
        ctypes.POINTER(ctypes.c_int64),    # out_lengths
        ctypes.POINTER(ctypes.c_int64),    # out_strides
        ctypes.c_int,                      # out_rank
        ctypes.POINTER(ctypes.c_int),      # reduce_dims
        ctypes.c_int,                      # n_reduce_dims
        ctypes.c_double,                   # alpha
        ctypes.c_double,                   # beta
        ctypes.c_bool,                     # time_kernel
        ctypes.c_int,                      # warmup
        ctypes.c_int,                      # nrepeat
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
    in_rank = x.ndim
    out_rank = y.ndim

    in_lengths = (ctypes.c_int64 * in_rank)(*x.shape)
    in_strides = (ctypes.c_int64 * in_rank)(*x.stride())
    out_lengths = (ctypes.c_int64 * out_rank)(*y.shape)
    out_strides = (ctypes.c_int64 * out_rank)(*y.stride())
    rdims = (ctypes.c_int * len(reduce_dims))(*reduce_dims)

    torch.cuda.synchronize()
    ms = lib.run_kernel(
        x.data_ptr(),
        y.data_ptr(),
        in_lengths,
        in_strides,
        in_rank,
        out_lengths,
        out_strides,
        out_rank,
        rdims,
        len(reduce_dims),
        alpha,
        beta,
        time_kernel,
        warmup,
        nrepeat,
    )
    return ms


def compute_out_shape(in_shape: list[int], reduce_dims: list[int]) -> list[int]:
    """Compute the output shape after reducing specified dimensions."""
    out_shape = [s for i, s in enumerate(in_shape) if i not in reduce_dims]
    return out_shape if out_shape else [1]


def run_kernel_output(
    lib, x: torch.Tensor, reduce_dims: list[int]
) -> torch.Tensor | None:
    """Run kernel. Returns output tensor, or None if unsupported."""
    out_shape = compute_out_shape(list(x.shape), reduce_dims)
    y = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    ms = call_kernel(lib, x, y, reduce_dims, alpha=1.0, beta=0.0, time_kernel=False)
    if ms < 0:
        return None
    return y


def benchmark_kernel(
    lib, shape: list[int], reduce_dims: list[int],
    warmup: int = 5, nrepeat: int = 20,
) -> float:
    """Time the kernel. Returns average time in ms, or -1 if unsupported."""
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    out_shape = compute_out_shape(shape, reduce_dims)
    y = torch.empty(out_shape, dtype=torch.float16, device="cuda")
    return call_kernel(lib, x, y, reduce_dims, alpha=1.0, beta=0.0,
                       time_kernel=True, warmup=warmup, nrepeat=nrepeat)


# -- Mode implementations follow the identical 4-mode pattern as softmax ------
# mode_correctness: compare outputs from baseline vs optimized
# mode_profile: run optimized once per shape
# mode_benchmark: time both, report bandwidth + speedup
# mode_full_benchmark: same on ALL_SHAPES
#
# ... (full implementations follow the softmax recipe pattern, replacing
#      tensor creation for the different in/out shapes)
```

## compile.py and Makefile

Identical to the softmax recipe. No changes needed.

## Adapting for Different Reduction Patterns

The `Rank` and `NumReduceDim` template parameters in the C++ kernel are compile-time
constants. When adapting for a different reduction pattern:

1. Change `Rank` and `NumReduceDim` in both `.cpp` files.
2. Change `NumOutDim` accordingly: `(Rank - NumReduceDim == 0) ? 1 : Rank - NumReduceDim`.
3. Update the `reduce_dims` list and shape lists in `test_harness.py`.
4. If using a different `ReduceOpId` (MAX, MIN, ADD, etc.), update the `ReduceOpId` constant
   and check `AccDataType` compatibility.

## Reduce Operation Variants

| ReduceOpId | Supports Indices | AccDataType for fp16 |
|------------|-----------------|---------------------|
| `AVG` | No | float |
| `ADD` | No | float |
| `MAX` | Yes | half_t |
| `MIN` | Yes | half_t |
| `AMAX` | Yes | half_t |

## Correctness Notes

- For AVG reduction over many elements, floating-point rounding accumulates. Use
  `atol=1e-2, rtol=1e-2` for fp16 reductions over large dimensions.
- The elementwise operators for AVG require `reduce_total_length` to compute the
  correct scale factor. Ensure this is passed correctly via `GetElementwiseOperator`.
