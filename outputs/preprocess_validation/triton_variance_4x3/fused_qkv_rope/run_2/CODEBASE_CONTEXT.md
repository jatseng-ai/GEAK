# Codebase Context

## Repository Layout

```
main-e15b75d464bf/
├── 3rdparty/
│   ├── ck_helper/
│   │   └── ck/
│   │       └── config.h
│   └── composable_kernel/
├── aiter/
│   ├── aot/
│   │   ├── test/
│   │   │   ├── matmul_fp16.py
│   │   │   ├── test.sh
│   │   │   ├── test_hsaco.py
│   │   │   └── test_matmul.cpp
│   │   ├── triton/
│   │   │   ├── decode_mla.py
│   │   │   └── norm.py
│   │   ├── __init__.py
│   │   ├── asm_mla_decode_fwd.py
│   │   ├── pa.py
│   │   ├── pa_ragged.py
│   │   ├── pa_v1.py
│   │   └── sampling.py
│   ├── configs/
│   │   ├── model_configs/
│   │   │   ├── a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv
│   │   │   ├── a8w8_blockscale_bpreshuffle_tuned_gemm_qwen3_235b.csv
│   │   │   ├── a8w8_blockscale_tuned_fmoe_qwen3_235b.csv
│   │   │   ├── a8w8_blockscale_tuned_gemm_ds_v3.csv
│   │   │   ├── a8w8_blockscale_tuned_gemm_qwen3_235b.csv
│   │   │   ├── a8w8_blockscale_untuned_fmoe_qwen3_235b.csv
│   │   │   ├── a8w8_blockscale_untuned_gemm_ds_v3.csv
│   │   │   ├── a8w8_blockscale_untuned_gemm_qwen3_235b.csv
│   │   │   ├── a8w8_bpreshuffle_tuned_gemm_dsv3.csv
│   │   │   ├── dsv3_bf16_tuned_gemm.csv
│   │   │   ├── dsv3_fp4_tuned_fmoe.csv
│   │   │   ├── gptoss_bf16_tuned_gemm.csv
│   │   │   ├── kimik2_bf16_tuned_gemm.csv
│   │   │   ├── kimik2_fp4_tuned_fmoe.csv
│   │   │   ├── llama405B_untuned_gemm.csv
│   │   │   ├── llama405B_untuned_gemm_bf16.csv
│   │   │   ├── llama70B_untuned_gemm.csv
│   │   │   ├── llama70B_untuned_gemm_bf16.csv
│   │   │   ├── qwen32B_untuned_gemm.csv
│   │   │   ├── qwen32B_untuned_gemm_bf16.csv
│   │   │   └── README.md
│   │   ├── __init__.py
│   │   ├── a4w4_blockscale_tuned_gemm.csv
│   │   ├── a4w4_blockscale_untuned_gemm.csv
│   │   ├── a8w8_blockscale_bpreshuffle_tuned_gemm.csv
│   │   ├── a8w8_blockscale_bpreshuffle_untuned_gemm.csv
│   │   ├── a8w8_blockscale_tuned_gemm.csv
│   │   ├── a8w8_blockscale_untuned_gemm.csv
│   │   ├── a8w8_bpreshuffle_tuned_gemm.csv
│   │   ├── a8w8_bpreshuffle_untuned_gemm.csv
│   │   ├── a8w8_tuned_batched_gemm.csv
│   │   ├── a8w8_tuned_gemm.csv
│   │   ├── a8w8_untuned_batched_gemm.csv
│   │   ├── a8w8_untuned_gemm.csv
│   │   ├── asm_a8w8_gemm.csv
│   │   ├── bf16_tuned_batched_gemm.csv
│   │   ├── bf16_tuned_gemm.csv
│   │   ├── bf16_untuned_batched_gemm.csv
│   │   ├── bf16_untuned_gemm.csv
│   │   ├── tuned_fmoe.csv
│   │   └── untuned_fmoe.csv
│   ├── jit/
│   │   ├── utils/
│   │   │   ├── hipify/
│   │   │   │   ... (5 items)
│   │   │   ├── __init__.py
│   │   │   ├── _cpp_extension_versioner.py
│   │   │   ├── chip_info.py
│   │   │   ├── cpp_extension.py
│   │   │   ├── file_baton.py
│   │   │   ├── mha_recipes.py
│   │   │   └── torch_guard.py
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── module_aiter_enum.so
│   │   └── optCompilerConfig.json
│   ├── ops/
│   │   ├── flydsl/
│   │   │   ├── kernels/
│   │   │   │   ... (6 items)
│   │   │   ├── __init__.py
│   │   │   ├── moe_kernels.py
│   │   │   └── utils.py
│   │   ├── triton/
│   │   │   ├── _triton_kernels/
│   │   │   │   ... (16 items)
│   │   │   ├── attention/
│   │   │   │   ... (22 items)
│   │   │   ├── comms/
│   │   │   │   ... (6 items)
│   │   │   ├── configs/
│   │   │   │   ... (12 items)
│   │   │   ├── fusions/
│   │   │   │   ... (6 items)
│   │   │   ├── gated_delta_net/
│   │   │   │   ... (2 items)
│   │   │   ├── gemm/
│   │   │   │   ... (5 items)
│   │   │   ├── gluon/
│   │   │   │   ... (7 items)
│   │   │   ├── moe/
│   │   │   │   ... (16 items)
│   │   │   ├── normalization/
│   │   │   │   ... (4 items)
│   │   │   ├── quant/
│   │   │   │   ... (6 items)
│   │   │   ├── rope/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── fused_qkv_split_qk_rope.py    ← TARGET KERNEL
│   │   │   │   └── rope.py
│   │   │   ├── utils/
│   │   │   │   ... (14 items)
│   │   │   ├── __init__.py
│   │   │   ├── activation.py
│   │   │   ├── causal_conv1d.py
│   │   │   ├── gather_kv_b_proj.py
│   │   │   ├── gmm.py
│   │   │   ├── README.md
│   │   │   ├── softmax.py
│   │   │   └── topk.py
│   │   ├── __init__.py
│   │   ├── activation.py
│   │   ├── aiter_operator.py
│   │   ├── attention.py
│   │   ├── batched_gemm_op_a8w8.py
│   │   ├── batched_gemm_op_bf16.py
│   │   ├── cache.py
│   │   ├── causal_conv1d.py
│   │   ├── communication.py
│   │   ├── custom.py
│   │   ├── custom_all_reduce.py
│   │   ├── deepgemm.py
│   │   ├── enum.py
│   │   ├── fused_qk_norm_mrope_cache_quant.py
│   │   ├── fused_qk_norm_rope_cache_quant.py
│   │   ├── gemm_op_a16w16.py
│   │   ├── gemm_op_a4w4.py
│   │   ├── gemm_op_a8w8.py
│   │   ├── gemm_op_common.py
│   │   ├── gradlib.py
│   │   ├── groupnorm.py
│   │   ├── mha.py
│   │   ├── mhc.py
│   │   ├── moe_op.py
│   │   ├── moe_sorting.py
│   │   ├── moe_sorting_opus.py
│   │   ├── norm.py
│   │   ├── pos_encoding.py
│   │   ├── quant.py
│   │   ├── quick_all_reduce.py
│   │   ├── rmsnorm.py
│   │   ├── rope.py
│   │   ├── sample.py
│   │   ├── sampling.py
│   │   ├── shuffle.py
│   │   ├── topk.py
│   │   ├── topk_plain.py
│   │   └── trans_ragged_layout.py
│   ├── utility/
│   │   ├── triton/
│   │   │   ├── README.md
│   │   │   └── triton_metadata_redirect.py
│   │   ├── aiter_types.py
│   │   ├── base_tuner.py
│   │   ├── dtypes.py
│   │   ├── fp4_utils.py
│   │   └── mp_tuner.py
│   ├── __init__.py
│   ├── _version.py
│   ├── bert_padding.py
│   ├── fused_moe.py
│   ├── fused_moe_bf16_asm.py
│   ├── fused_moe_dp_shared_expert.py
│   ├── install_mode
│   ├── int4_utils.py
│   ├── mla.py
│   ├── paged_attn.py
│   ├── rotary_embedding.py
│   ├── test_common.py
│   ├── test_mha_common.py
│   └── tuned_gemm.py
├── aiter_logs/
│   ├── readme.md
│   └── run.py
├── csrc/
│   ├── ck_batched_gemm_a8w8/
│   │   ├── include/
│   │   │   ├── batched_gemm_a8w8.h
│   │   │   └── batched_gemm_a8w8_common.cuh
│   │   ├── batched_gemm_a8w8.cu
│   │   ├── batched_gemm_a8w8_common.py
│   │   ├── batched_gemm_a8w8_tune.cu
│   │   ├── batched_gemm_a8w8_tune.py
│   │   ├── gen_instances.py
│   │   └── README.md
│   ├── ck_batched_gemm_bf16/
│   │   ├── include/
│   │   │   ├── batched_gemm_bf16.h
│   │   │   └── batched_gemm_bf16_common.cuh
│   │   ├── batched_gemm_bf16.cu
│   │   ├── batched_gemm_bf16_common.py
│   │   ├── batched_gemm_bf16_tune.cu
│   │   ├── batched_gemm_bf16_tune.py
│   │   ├── gen_instances.py
│   │   └── README.md
│   ├── ck_deepgemm/
│   │   ├── include/
│   │   │   ├── deepgemm.h
│   │   │   └── deepgemm_common.cuh
│   │   ├── deepgemm.cu
│   │   ├── deepgemm_common.py
│   │   └── gen_instances.py
│   ├── ck_gemm_a4w4_blockscale/
│   │   ├── include/
│   │   │   ├── gemm_a4w4_blockscale.h
│   │   │   └── gemm_a4w4_blockscale_common.cuh
│   │   ├── gemm_a4w4_blockscale.cu
│   │   ├── gemm_a4w4_blockscale_common.py
│   │   ├── gemm_a4w4_blockscale_tune.cu
│   │   ├── gemm_a4w4_blockscale_tune.py
│   │   ├── gen_instances.py
│   │   ├── README.md
│   │   └── validate_tuned_csv.py
│   ├── ck_gemm_a8w8/
│   │   ├── include/
│   │   │   ├── gemm_a8w8.h
│   │   │   └── gemm_a8w8_common.cuh
│   │   ├── gemm_a8w8.cu
│   │   ├── gemm_a8w8_common.py
│   │   ├── gemm_a8w8_tune.cu
│   │   ├── gemm_a8w8_tune.py
│   │   ├── gen_instances.py
│   │   └── README.md
│   ├── ck_gemm_a8w8_blockscale/
│   │   ├── include/
│   │   │   ├── gemm_a8w8_blockscale.h
│   │   │   ├── gemm_a8w8_blockscale_bpreshuffle_cktile.h
│   │   │   ├── gemm_a8w8_blockscale_cktile.h
│   │   │   ├── gemm_a8w8_blockscale_cktile_common.cuh
│   │   │   └── gemm_a8w8_blockscale_common.cuh
│   │   ├── gemm_a8w8_blockscale.cu
│   │   ├── gemm_a8w8_blockscale_cktile.cu
│   │   ├── gemm_a8w8_blockscale_cktile_instance.py
│   │   ├── gemm_a8w8_blockscale_cktile_tune.cu
│   │   ├── gemm_a8w8_blockscale_instance.py
│   │   ├── gemm_a8w8_blockscale_tune.cu
│   │   ├── gemm_a8w8_blockscale_tune.py
│   │   ├── gen_instances.py
│   │   ├── gen_instances_cktile.py
│   │   └── README.md
│   ├── ck_gemm_a8w8_blockscale_bpreshuffle/
│   │   ├── include/
│   │   │   ├── gemm_a8w8_blockscale_bpreshuffle.h
│   │   │   └── gemm_a8w8_blockscale_bpreshuffle_common.cuh
│   │   ├── gemm_a8w8_blockscale_bpreshuffle.cu
│   │   ├── gemm_a8w8_blockscale_bpreshuffle_common.py
│   │   ├── gemm_a8w8_blockscale_bpreshuffle_tune.cu
│   │   ├── gemm_a8w8_blockscale_bpreshuffle_tune.py
│   │   ├── gen_instances.py
│   │   └── README.md
│   ├── ck_gemm_a8w8_bpreshuffle/
│   │   ├── include/
│   │   │   ├── gemm_a8w8_bpreshuffle.h
│   │   │   └── gemm_a8w8_bpreshuffle_common.cuh
│   │   ├── gemm_a8w8_bpreshuffle.cu
│   │   ├── gemm_a8w8_bpreshuffle_common.py
│   │   ├── gemm_a8w8_bpreshuffle_tune.cu
│   │   ├── gemm_a8w8_bpreshuffle_tune.py
│   │   ├── gen_instances.py
│   │   └── README.md
│   ├── ck_gemm_moe_2stages_codegen/
│   │   ├── gemm_moe_ck2stages.cu
│   │   ├── gemm_moe_ck2stages.h
│   │   ├── gemm_moe_ck2stages_common.cuh
│   │   ├── gemm_moe_ck2stages_common.py
│   │   ├── gemm_moe_ck2stages_common_blockscale.cuh
│   │   ├── gemm_moe_ck2stages_common_mxfp4.cuh
│   │   ├── gemm_moe_ck2stages_common_mxfp4_bns.cuh
│   │   ├── gemm_moe_tune.py
│   │   ├── gen_instances.py
│   │   └── README.md
│   ├── ck_tile_gemm_moe_2stages/
│   │   ├── include/
│   │   │   ├── moe_cktile2stages.h
│   │   │   └── moe_cktile2stages_common.cuh
│   │   ├── gen_instances.py
│   │   ├── moe_cktile2stages.cu
│   │   └── moe_cktile2stages_common.py
│   ├── cktile_gemm_a8w8_bpreshuffle/
│   │   ├── include/
│   │   │   ├── gemm_a8w8_bpreshuffle_cktile.h
│   │   │   └── gemm_a8w8_bpreshuffle_cktile_common.cuh
│   │   ├── gemm_a8w8_bpreshuffle_cktile.cu
│   │   ├── gemm_a8w8_bpreshuffle_cktile_common.py
│   │   ├── gemm_a8w8_bpreshuffle_cktile_tune.cu
│   │   ├── gen_instances.py
│   │   └── ... (1 more)
│   └── ... (6 more)
└── ... (18 more)
```

## Kernel Dependency Tree

Target kernel: `aiter/ops/triton/rope/fused_qkv_split_qk_rope.py`

### Direct dependencies

| File | Imports | Description |
|------|---------|-------------|
| `aiter/ops/triton/_triton_kernels/rope/fused_qkv_split_qk_rope.py` | `_fused_qkv_split_qk_rope_kernel` | Triton kernel definitions (@triton.jit) |


### Transitive dependencies (depth 2)

Improving these may improve the target kernel's performance.

| File | Imports | Used by | Description |
|------|---------|---------|-------------|
| `aiter/ops/triton/rope/rope.py` | `_get_gptj_rotated_x`, `_get_neox_rotated_x` | `aiter/ops/triton/_triton_kernels/rope/fused_qkv_split_qk_rope.py` | Class definitions |


### Transitive dependencies (depth 3)

Improving these may improve the target kernel's performance.

| File | Imports | Used by | Description |
|------|---------|---------|-------------|
| `aiter/ops/triton/_triton_kernels/rope/rope.py` | `_rope_kernel_sbhd_fwd`, `_rope_kernel_sbhd_bwd`, `_rope_kernel_thd_fwd`, `_rope_kernel_thd_bwd`, `_rope_kernel_sbhd_cached_fwd`, `_rope_kernel_sbhd_cached_bwd`, `_rope_kernel_thd_cached_2c_fwd`, `_rope_kernel_thd_cached_2c_bwd`, `_rope_kernel_cached_thd_2c_gqa_fwd`, `_rope_kernel_cached_thd_2c_gqa_onehead_fwd`, `_rope_kernel_cached_thd_2c_gqa_bwd`, `_rope_kernel_cached_thd_2c_gqa_onehead_bwd`, `_rope_fwd_2d_kernel_neox`, `_rope_fwd_3d`, `_get_neox_rotated_x_1D`, `_get_gptj_rotated_x_1D`, `_get_neox_rotated_x`, `_get_gptj_rotated_x` | `aiter/ops/triton/rope/rope.py` | Triton kernel definitions (@triton.jit) |
| `aiter/ops/triton/utils/logger.py` | `AiterTritonLogger` | `aiter/ops/triton/rope/rope.py` | Class definitions |
