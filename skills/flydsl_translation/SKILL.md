---
name: flydsl_translation
description: Use when translating PyTorch GPU kernels to FlyDSL—preserving Model interface, inputs helpers, and numerical parity; choosing the right FlyDSL builders (GEMM, flash attention, norms) and avoiding forbidden PyTorch shortcuts.
---

# PyTorch GPU kernel → FlyDSL translation

## Role

You act as a **translation specialist**: turn a given PyTorch GPU kernel into an equivalent **FlyDSL** implementation while keeping the public Python surface compatible and results numerically aligned with the original (within tolerance).

## Hard constraints

1. **FlyDSL only for GPU logic** — Use the real FlyDSL API (`flydsl.compiler` / `flydsl.expr` / `arith`, `gpu`, `vector`). Do **not** use Triton, CUDA, or other GPU models. Do **not** add shims or mock modules for FlyDSL.
2. **Preserve the PyTorch module contract** — Same `Model(nn.Module)` shape: `__init__`, `forward` signature, output **shape** and **dtype** as the reference.
3. **Preserve harness entrypoints** — Keep `get_inputs()` and `get_init_inputs()` as required by the task.
4. **Numerical parity** — The translated path must match the PyTorch reference within the stated tolerance.
5. **No forbidden high-level PyTorch for covered ops** — Do **not** use `torch.matmul`, `F.linear`, `nn.Linear`, or `F.scaled_dot_product_attention` in the translated hot path; those have FlyDSL replacements. PyTorch fallback is **only** where there is no FlyDSL equivalent (e.g. Conv2d, MaxPool2d, BatchNorm2d per project rules).

## FlyDSL kernel structure (three layers)

Every translation should follow this layering:

- **`@flyc.kernel`** — Device kernel using layout algebra (`fx.logical_divide`, `fx.slice`, copy atoms, etc.).
- **`@flyc.jit`** — Host launcher: grid/block, then `kernel.launch(...)`.
- **`Model(nn.Module)`** — Allocates outputs and calls the `@flyc.jit` launcher.

## Translation strategy (preference order)

- **GEMM / Linear** — Prefer `compile_preshuffle_gemm_a8()` from `kernels.preshuffle_gemm`. The **B** matrix must be preshuffled with `shuffle_weight(B.contiguous(), layout=(16, 16))` from `tests.utils`. Tensor arguments passed in flattened form (e.g. `.view(-1)`). For fp16, use empty scale tensors as required by the chosen API (e.g. `torch.empty(0, device=dev, dtype=torch.float32)` where applicable).
- **Attention / SDPA** — When constraints match (e.g. head_dim ≥ 64, head_dim % 32 == 0, seq_len % 128 == 0), use `build_flash_attn_func_module()` from `kernels.flash_attn_func`. Do **not** decompose attention into separate GEMM + softmax + GEMM when flash attention applies (large regression). Do **not** drive attention with Python loops over batch×heads calling GEMM one call at a time. Builder bakes in `num_heads`; launcher receives views and `batch_size`, `seq_len`, `stream`.
- **Softmax** — `build_softmax_module(M, N, dtype_str)`; call shape per module docs (input, output, M, stream).
- **LayerNorm** — `build_layernorm_module(M, N, dtype_str)`; call with input, gamma, beta, output, M, stream.
- **RMSNorm** — `build_rmsnorm_module(M, N, dtype_str)`; call with input, gamma, output, M, stream.
- **Element-wise** (relu, sigmoid, tanh, clamp, …) — Custom `@flyc.kernel` with layout algebra.
- **Reductions** (sum, mean) — Manual block reduction with wave shuffle as appropriate.
- **Conv / Pool / BatchNorm** — Keep `torch.nn.functional` (or modules) only where there is no FlyDSL equivalent.
- **Larger models** — Use FlyDSL for every op that has a mapping; reserve PyTorch only for conv/pool/batchnorm-style gaps.

## Workflow

0. Must **Read** the all example translations one by one in docs/ folder before any futhre move. 
1. **Read** the source PyTorch kernel and any task metadata (paths, dtypes, shapes, tolerance).
2. **Plan** the FlyDSL mapping:
   - Classify the pattern: element-wise, reduction, GEMM, attention, norms, or mixed.
   - Pick builders/kernels from the strategy above; note weight preshuffle for GEMM and flash-attn eligibility for attention.
   - Decide what tiny fraction stays on PyTorch (conv/pool/batchnorm only).
3. **Implement** the translation at the specified output location: three-layer structure, correct launches, and preserved `Model` / `get_inputs` / `get_init_inputs`.
4. **Validate** using the project’s prescribed test harness (run, inspect failures, adjust types/layouts/launch args).
5. **Iterate** on failures until tests pass within tolerance; then treat the task as complete per the orchestrator’s completion rules.

## Task inputs you should expect

Typical task text includes: source path, output path, how to run tests, and sometimes a **FlyDSL knowledge base** excerpt—use it for API names, shapes, and project conventions.

## Knowledge base

When the task includes a **FlyDSL knowledge base** block (curated snippets or docs from the runner), treat it as authoritative for local FlyDSL patterns, file layout, and examples alongside this skill.
