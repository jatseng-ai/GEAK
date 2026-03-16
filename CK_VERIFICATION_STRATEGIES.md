# CK Example Verification Strategies

Reference for building an automated system that generates verification tests for new Composable Kernel (CK) examples.

---

## 1. Architecture Overview

Every CK example follows a common verification architecture:

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Initialize  │────>│  Run GPU Kernel  │────>│  Copy Results  │
│  Host Data   │     │  (Device Under   │     │  Back to Host  │
│              │     │   Test)           │     │                │
└──────────────┘     └──────────────────┘     └───────┬────────┘
       │                                              │
       │             ┌──────────────────┐     ┌───────▼────────┐
       └────────────>│  Run Reference   │────>│  Compare With  │
                     │  (CPU or GPU)    │     │  check_err()   │
                     └──────────────────┘     └────────────────┘
```

The verification is controlled at runtime via a `do_verification` flag (CLI argument), which is typically the first positional argument or passed as `--verify=` / `-v`. This allows the same binary to be used for both benchmarking (no verification overhead) and correctness testing.

---

## 2. Verification Flag Conventions

### Legacy examples (01–69): `ExecutionConfig.do_verification`

The `do_verification` field is an integer supporting multiple modes:

| Value | Meaning |
|-------|---------|
| `0` | No verification |
| `1` | CPU reference verification |
| `2` | GPU reference verification |
| `3` | Both CPU and GPU reference verification |

```cpp
struct ExecutionConfig final {
    int do_verification = 1;  // default: CPU verification enabled
    int init_method     = 2;
    bool time_kernel    = false;
};
```

### ck_tile examples: `-v` flag

The `ck_tile` examples use an argument parser with `-v <0|1>` for a boolean verification toggle. The default is typically `1` (enabled).

```bash
./layernorm2d_fwd -m 128 -n 1024 -v 1
```

---

## 3. Data Initialization Strategies

Proper data initialization is critical. The choice affects numerical stability and the ability to detect bugs. CK examples use the `init_method` parameter to select from several strategies.

### 3.1 Legacy Examples: `init_method` Switch

Defined in `include/ck/library/utility/fill.hpp` and `include/ck/library/utility/host_tensor_generator.hpp`.

| `init_method` | Strategy | Typical Use |
|---------------|----------|-------------|
| `0` | Constant fill (all 1s) | Sanity check; quick smoke test |
| `1` | Uniform integer values in `[-5, 5]` | Exact integer arithmetic; avoids FP rounding |
| `2` | Uniform float values in `[-1, 1]` or `[-0.1, 0.1]` | General FP testing (default) |
| `3` | A = constant 1, B = integer `[-5, 5]` | Test asymmetric inputs |
| `4` | A = integer `[-5, 5]`, B = constant 1 | Test asymmetric inputs |
| `5` | Integer values in `[-2, 2]` | Narrow range integer testing |
| default | Uniform float `[-0.1, 0.1]` | Small values for numerical stability |

**Fill functors (legacy):**

| Functor | Description |
|---------|-------------|
| `FillConstant<T>{value}` | Every element = `value` |
| `FillUniformDistribution<T>{lo, hi}` | Uniform real distribution in `[lo, hi]` |
| `FillUniformDistributionIntegerValue<T>{lo, hi}` | Uniform real then `std::round()` |
| `FillMonotonicSeq<T>{init, step}` | Monotonic sequence: init, init+step, ... |

**Generator functors (used with `Tensor::GenerateTensorValue()`):**

| Functor | Description |
|---------|-------------|
| `GeneratorTensor_0<T>` | All zeros |
| `GeneratorTensor_1<T>{val}` | Constant value |
| `GeneratorTensor_2<T>{min, max}` | Random integer in `[min, max]` |
| `GeneratorTensor_3<T>{min, max}` | Random float in `[min, max]` |
| `GeneratorTensor_4<T>{mean, stddev, seed}` | Normal distribution |
| `GeneratorTensor_Checkboard` | Alternating +1/-1 checkerboard |
| `GeneratorTensor_Sequential<T, Dim>` | Sequential based on index |
| `GeneratorTensor_Diagonal<T>{val}` | Diagonal matrix |

### 3.2 ck_tile Examples

ck_tile examples typically use `ck_tile::FillUniformDistribution` directly with explicit ranges, or inline initialization logic. Seed control is usually exposed via `-seed` CLI argument.

---

## 4. Reference Computation Strategies

This is the most important component for an automated system to understand and replicate.

### 4.1 Strategy A: CK-Provided CPU Reference Implementation

The most common approach. CK provides pre-built host-side reference implementations that mirror the GPU kernel's mathematical operation.

**Location:** `library/include/ck/library/reference_tensor_operation/cpu/` (legacy) and `include/ck_tile/host/reference/` (ck_tile)

**Pattern:**

```cpp
// 1. Instantiate the reference implementation
using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<...>;
auto ref_gemm = ReferenceGemmInstance{};

// 2. Create arguments (mirrors the device API)
auto ref_argument = ref_gemm.MakeArgument(
    a_host, b_host, c_host_ref, a_element_op, b_element_op, c_element_op);

// 3. Run the reference
auto ref_invoker = ref_gemm.MakeInvoker();
ref_invoker.Run(ref_argument);
```

**Available reference implementations:**

| Reference Class | Kernel Type |
|-----------------|-------------|
| `ReferenceGemm` | General matrix multiply |
| `ReferenceConvFwd` | Convolution forward |
| `ReferenceConvBwdData` | Convolution backward data |
| `ReferenceConvBwdWeight` | Convolution backward weight |
| `ReferenceBatchNormFwd` | BatchNorm forward training |
| `ReferenceBatchNormInfer` | BatchNorm forward inference |
| `ReferenceBatchNormBwd` | BatchNorm backward |
| `ReferenceReduce` | Reduction operations |
| `ReferenceSoftmax` | Softmax |
| `reference_gemm` (ck_tile) | GEMM |
| `reference_layernorm2d_fwd` (ck_tile) | LayerNorm forward |
| `reference_softmax` (ck_tile) | Softmax |
| `reference_reduce` (ck_tile) | Reduction |
| `reference_moe_sorting` (ck_tile) | MoE token sorting |

### 4.2 Strategy B: CK-Provided GPU Reference Implementation

Some examples (primarily GEMM, example `01_gemm`) support verification against a simpler, known-correct GPU kernel. This is useful when CPU reference is too slow for large problem sizes.

```cpp
if ((config.do_verification == 2) || (config.do_verification == 3))
{
    auto ref_gemm_gpu = ReferenceGemmInstanceGPU{};
    auto ref_invoker_gpu = ref_gemm_gpu.MakeInvoker();
    auto ref_argument_gpu = ref_gemm_gpu.MakeArgument(
        static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
        static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
        static_cast<CDataType*>(c_ref_device_buf.GetDeviceBuffer()),
        M, N, K, a_element_op, b_element_op, c_element_op);
    ref_invoker_gpu.Run(ref_argument_gpu, StreamConfig{});
}
```

### 4.3 Strategy C: Inline CPU Reference (Manual Loops)

For kernels with fused element-wise operations or custom epilogues, the reference is computed in two stages: a base operation via CK reference, then a manual element-wise pass.

**Example from `30_grouped_conv_fwd_multiple_d` (conv + bias + relu + add):**

```cpp
// Stage 1: Run base convolution reference
auto ref_conv = HostConvFwdInstance<NDimSpatial>{};
ref_conv.MakeInvoker().Run(ref_conv.MakeArgument(
    in, wei, c_host, strides, dilations, pads_left, pads_right,
    InElementOp{}, WeiElementOp{}, PassThrough{}));

// Stage 2: Apply fused element-wise operations manually
out_host.ForEach([&](auto&, auto idx) {
    OutElementOp{}(out_host(idx), c_host(idx), bias(idx), residual(idx));
});
```

**Example from `02_layernorm2d` (ck_tile, layernorm + fused add + quantization):**

```cpp
// Base layernorm reference
ck_tile::reference_layernorm2d_fwd<...>(
    x_host, gamma_host, beta_host, y_host_ref, mean_ref, invStd_ref, epsilon);

// Fused operations applied manually on top
if (fused_quant != 0) {
    // manual smooth-quant, dynamic-quant, etc.
}
```

### 4.4 Strategy D: Pure Inline Reference (No CK Reference Class)

For simple operations like elementwise or permute, the reference is a simple host loop without any CK reference class.

**Example from `19_binary_elementwise`:**

```cpp
host_elementwise4D<...>(host_c, a, b, nchw, Add{});
```

---

## 5. Error Checking: `check_err()`

All examples converge on a single comparison utility. Two versions exist:

- **Legacy:** `ck::utils::check_err()` in `include/ck/library/utility/check_err.hpp`
- **ck_tile:** `ck_tile::check_err()` in `include/ck_tile/host/check_err.hpp`

### 5.1 Core Algorithm

Both versions use the same tolerance formula for floating-point types:

```
error = |output[i] - reference[i]|
pass  = (error <= atol + rtol * |reference[i]|) AND isfinite(output[i]) AND isfinite(reference[i])
```

For integer types, only absolute tolerance is used: `pass = (|output[i] - reference[i]| <= atol)`.

For FP8 types (ck_tile), an additional **rounding point distance** check is used alongside the standard tolerance check.

### 5.2 Type-Specific Default Tolerances

The functions are overloaded by data type, each with its own defaults:

| Data Type | Default `rtol` | Default `atol` | Notes |
|-----------|---------------|---------------|-------|
| `float` | `1e-5` | `3e-6` | Standard single precision |
| `float` (tf32 compute) | `5e-4` | `5e-4` | Reduced precision TF32 |
| `double` | `1e-6` | `1e-6` | (Legacy only, via `get_rtol/get_atol`) |
| `half_t` / `fp16` | `1e-3` | `1e-3` | Half precision |
| `bhalf_t` / `bf16` | `1e-1` / `5e-2` | `1e-3` / `5e-2` | Brain float; wider tolerance |
| `int8_t` | — | `0` | Exact match |
| `int32_t` | — | `0` | Exact match |
| `f8_t` / `fp8` | `1e-3` / 1 RPD | `1e-3` / `1e-1` | FP8; rounding-point distance in ck_tile |
| `bf8_t` | `1e-3` | `1e-3` | BF8 |
| `f4_t` | `0.5` | `0.5` | FP4; very wide tolerance |
| `pk_fp4_t` | — | exact | Packed FP4; bitwise comparison |

### 5.3 Automatic Threshold Computation

Both libraries provide `get_relative_threshold()` and `get_absolute_threshold()` functions that compute tolerances based on:

1. **Mantissa bits** of the compute, output, and accumulation data types
2. **Number of accumulations** (e.g., K dimension in GEMM)
3. **Maximum possible value** in the output (for absolute threshold)

```cpp
// Example: automatic threshold for GEMM verification
double rtol = get_relative_threshold<ComputeType, OutType, AccType>(K);
double atol = get_absolute_threshold<ComputeType, OutType, AccType>(max_val, K);
```

### 5.4 Per-Example Custom Tolerances (via `get_rtol` / `get_atol`)

Example `01_gemm` defines per-type tolerance functions in `common.hpp`:

```cpp
template <typename DataType, typename ComputeDataType = DataType>
constexpr double get_rtol() {
    if constexpr (std::is_same_v<DataType, float>)           return 1e-3;
    else if constexpr (std::is_same_v<DataType, double>)     return 1e-6;
    else if constexpr (std::is_same_v<DataType, half_t>)     return 1e-3;
    else if constexpr (std::is_same_v<DataType, bhalf_t>)    return 5e-2;
    else if constexpr (std::is_same_v<DataType, f8_t>)       return 1e-1;
    else if constexpr (std::is_same_v<DataType, bf8_t>)      return 1.5e-1;
    // ...
}
```

### 5.5 Error Reporting

When errors are detected, `check_err()` reports:
1. The **first N mismatched elements** (5 for legacy, 16 for ck_tile) with indices and values
2. **Maximum absolute error** across all elements
3. **Total error count** and **percentage** of wrong values

```
Error: Incorrect results! out[42] != ref[42]: 0.1234567 != 0.1234568
max err: 0.0001234, number of errors: 3, 0.01% wrong values
```

---

## 6. Multi-Output Verification

Kernels that produce multiple outputs verify each output tensor separately with descriptive error messages. The overall result is the logical AND of all individual checks.

**Example: BatchNorm forward (4 outputs):**

```cpp
pass = pass && check_err(y, y_ref, "Incorrect normalized output values");

if (updateMovingAverage) {
    pass = pass && check_err(runningMean, runningMean_ref, "Incorrect running mean values");
    pass = pass && check_err(runningVariance, runningVariance_ref, "Incorrect running variance values");
}

if (saveMeanAndInvVariance) {
    pass = pass && check_err(savedMean, savedMean_ref, "Incorrect saved mean values");
    pass = pass && check_err(savedInvVariance, savedInvVariance_ref, "Incorrect saved invvariance values");
}
```

**Example: MoE sorting (3 outputs + metadata):**

```cpp
rtn &= check_err(sorted_ids_host, sorted_ids_ref, "OUT Error: Incorrect ids!");
rtn &= check_err(sorted_weights_host, sorted_weights_ref, "OUT Error: Incorrect w!");
rtn &= check_err(sorted_expert_ids_host, sorted_expert_ids_ref, "OUT Error: Incorrect eid!");
```

**Example: LayerNorm with fused add and quantization (multiple fused outputs):**

```cpp
pass = check_err(y_host_dev, y_host_ref, "OUT Error: Incorrect results!", rtol, atol);
if (fused_add == 1) {
    pass &= check_err(y_residual_host_dev, x_host, "ADD Error: Incorrect results!", rtol, atol);
}
if (fused_quant == 1) {
    pass &= check_err(y_scale_host_dev, y_scale_host_ref, "SCALE Error: Incorrect results!", rtol, atol);
}
```

---

## 7. Data Transfer Pattern

All examples follow the same host ↔ device transfer pattern:

```cpp
// 1. Allocate device memory
DeviceMem a_device_buf(sizeof(DataType) * tensor.mDesc.GetElementSpaceSize());

// 2. Upload initialized host data to device
a_device_buf.ToDevice(tensor.mData.data());

// 3. Run GPU kernel (writes to output device buffer)
invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

// 4. Download results back to host for comparison
out_device_buf.FromDevice(out_device_result.mData.data());

// 5. Compare
pass = check_err(out_device_result, out_host_reference);
```

For `int4` types, an explicit type-conversion step is needed between host and device representations.

---

## 8. Checklist for Implementing Verification on a New Kernel

An automated system should follow these steps:

### Step 1: Identify the Mathematical Operation

Determine which category the kernel falls into:
- **GEMM-family:** Use `ReferenceGemm` or `reference_gemm`
- **Convolution-family:** Use `ReferenceConvFwd` / `ReferenceConvBwdData` / `ReferenceConvBwdWeight`
- **Normalization:** Use `reference_layernorm2d_fwd`, `ReferenceBatchNormFwd`, etc.
- **Reduction:** Use `ReferenceReduce` or `reference_reduce`
- **Elementwise/Permute:** Write inline host loops
- **Fused kernels:** Combine a base reference + manual epilogue

### Step 2: Determine Data Types and Tolerance

1. Extract `InDataType`, `OutDataType`, `AccDataType`, `ComputeDataType` from the kernel instance template parameters.
2. Use the tolerance table (Section 5.2) or automatic threshold functions (Section 5.3) to set `rtol` and `atol`.
3. For fused operations with low-precision accumulation, widen tolerances.

### Step 3: Select Initialization Strategy

- **Integer types:** Use `GeneratorTensor_2` or `FillUniformDistributionIntegerValue` for exact arithmetic
- **FP16/BF16:** Use small ranges like `[-0.5, 0.5]` to avoid overflow
- **FP32/FP64:** Use `[-1, 1]` or `[-0.1, 0.1]`
- **FP8/FP4:** Use very small ranges; be aware of limited representable values
- **Convolution weights:** Typically `[-0.5, 0.5]`

### Step 4: Implement the Verification Block

Follow this template:

```cpp
bool pass = true;

if (config.do_verification)
{
    // 1. Run reference computation on host
    auto ref_op = ReferenceOpInstance{};
    auto ref_invoker = ref_op.MakeInvoker();
    auto ref_argument = ref_op.MakeArgument(/* host tensors and params */);
    ref_invoker.Run(ref_argument);

    // 2. Apply any fused element-wise ops manually (if needed)
    out_host_ref.ForEach([&](auto&, auto idx) {
        EpilogueOp{}(out_host_ref(idx), intermediate(idx), ...);
    });

    // 3. Copy device results to host
    out_device_buf.FromDevice(out_device.mData.data());

    // 4. Compare
    pass &= ck::utils::check_err(
        out_device, out_host_ref,
        "Error: Incorrect results!",
        get_rtol<OutDataType, ComputeDataType>(),
        get_atol<OutDataType, ComputeDataType>());
}

return pass;
```

### Step 5: Handle Edge Cases

- **`IsSupportedArgument` check:** Always call before running. If unsupported, return `true` (not a failure).
- **int4 types:** Need explicit type conversion between host (`int4_t`) and device (`int8_t`) representations.
- **Packed types (`pk_fp4_t`, `pk_fp6x16_t`):** Use bitwise/element-wise unpacked comparison.
- **Strided outputs:** For non-contiguous layouts, compare row-by-row to avoid padding mismatches.

### Step 6: Wire Up the CLI

Ensure the binary accepts:
```
arg1: verification (0=no, 1=CPU, 2=GPU, 3=both)
arg2: initialization method
arg3: time kernel (0/1)
arg4+: problem dimensions
```

Return `0` for pass, non-zero for failure.

---

## 10. File Organization Patterns

### Legacy examples (01–69):

```
example/XX_kernel_name/
├── CMakeLists.txt
├── README.md
├── common.hpp                         # Types, tolerances, CLI parsing
├── run_kernel_example.inc             # Template: init, run, verify logic
├── kernel_variant_xdl_fp16.cpp        # Thin wrapper: typedefs + #include .inc
├── kernel_variant_xdl_fp32.cpp
└── kernel_variant_wmma_fp16.cpp
```

The `.inc` file contains the verification logic as a template function. Each `.cpp` file defines specific type aliases and includes the `.inc`.

### ck_tile examples:

```
example/ck_tile/XX_kernel_name/
├── CMakeLists.txt
├── README.md
├── kernel_name.cpp                    # Main: CLI parsing + verification
├── kernel_name_api.cpp                # GPU kernel dispatch
├── kernel_name_kernel.hpp             # Kernel definition
└── kernel_name_policy.hpp             # Tile/block configuration
```

Verification lives directly in the main `.cpp` file.

---

## 11. Summary of Key Source Files

| File | Purpose |
|------|---------|
| `include/ck/library/utility/check_err.hpp` | Legacy error comparison functions |
| `include/ck_tile/host/check_err.hpp` | ck_tile error comparison functions |
| `include/ck/library/utility/fill.hpp` | Legacy fill functors |
| `include/ck_tile/host/fill.hpp` | ck_tile fill functors |
| `include/ck/library/utility/host_tensor_generator.hpp` | Legacy generator functors |
| `include/ck/library/utility/device_memory.hpp` | `DeviceMem` for host↔device transfer |
| `include/ck/library/utility/host_tensor.hpp` | `Tensor<T>` host tensor class |
| `library/include/ck/library/reference_tensor_operation/cpu/` | All CPU reference implementations |
| `library/include/ck/library/reference_tensor_operation/gpu/` | GPU reference implementations |
| `include/ck_tile/host/reference/` | ck_tile CPU reference implementations |

---

## 12. Decision Tree for an Automated Agent

```
New kernel to verify
│
├── Is there a matching CK reference implementation?
│   ├── YES → Use Strategy A (Section 4.1)
│   │         Instantiate reference, run, compare with check_err()
│   │
│   └── NO → Does the kernel fuse a base op + epilogue?
│       ├── YES → Use Strategy C (Section 4.3)
│       │         Run base reference + manual epilogue loops
│       │
│       └── NO → Use Strategy D (Section 4.4)
│                Write simple host loops matching the kernel math
│
├── Determine data types from kernel template params
│   └── Look up tolerances from Section 5.2 table
│       or use get_rtol/get_atol from common.hpp
│       or use get_relative_threshold/get_absolute_threshold
│
├── Does the kernel produce multiple outputs?
│   ├── YES → Verify each output separately (Section 6)
│   └── NO → Single check_err() call
│
├── Select init_method based on data type (Section 3)
│
└── Follow the verification template (Section 9, Step 4)
```
