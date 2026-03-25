# Harnessgen Toolkit

This folder contains the GEAK-side tooling for generating missing AIG-Eval
preprocess harnesses on a separate machine with two pre-created Docker
containers:

- `harness-gen`
- `harness-gen2`

The intended flow is:

1. Check out this `GEAK` branch on the new machine.
2. Ensure `AIG-Eval` is also present, including `external_repos/`.
3. Run the setup script to configure both containers.
4. Run the preprocess launcher from this folder.

## Files

- `setup_harnessgen_containers.sh`
  - Starts the two containers if needed
  - Runs `python3 -m pip install -e .` inside both containers
  - Persists `AMD_LLM_API_KEY`, `GEAK_MODEL`, and `MSWEA_MODEL_NAME`
  - Validates that GEAK preprocess imports work and the gateway is reachable

- `run_aigeval_preprocess_harnesses.sh`
  - Runs harness-only preprocess for the 16 kernels that still need harnesses
  - Uses anchored local file specs (`#L...`) for every kernel
  - Splits work across `harness-gen` and `harness-gen2`

## Quick Start

Preferred way to avoid shell history leakage:

```bash
export HARNESSGEN_KEY_A="your-key-for-harness-gen"
export HARNESSGEN_KEY_B="your-key-for-harness-gen2"
./scripts/harnessgen/setup_harnessgen_containers.sh
```

If the machine should only be validated, not modified:

```bash
./scripts/harnessgen/setup_harnessgen_containers.sh --validate-only
```

If the AMD gateway is expected to work and validation passes, print the exact
preprocess commands first:

```bash
./scripts/harnessgen/run_aigeval_preprocess_harnesses.sh --all --print-only
```

Then launch the real run:

```bash
./scripts/harnessgen/run_aigeval_preprocess_harnesses.sh --all
```

## Path Assumptions

By default the runner expects:

- `GEAK` at the current repo root
- `AIG-Eval` as a sibling checkout next to `GEAK`
- cloned source repos under `AIG-Eval/external_repos`

If your new machine uses different paths, pass:

- `--aig-eval-root`
- `--external-repos-root`
- `--geak-root`

## Notes

- The current GEAK config on this branch defaults preprocess model selection to
  `claude-opus-4.6`.
- The setup script validates gateway reachability by default. If the machine
  has restricted egress or you only want local validation, use
  `--skip-gateway-check`.
