#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEAK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_AIG_EVAL_ROOT=""
if [[ -d "${GEAK_ROOT}/../AIG-Eval" ]]; then
    DEFAULT_AIG_EVAL_ROOT="$(cd "${GEAK_ROOT}/../AIG-Eval" && pwd)"
fi

AIG_EVAL_ROOT="${HARNESSGEN_AIG_EVAL_ROOT:-$DEFAULT_AIG_EVAL_ROOT}"
EXTERNAL_REPOS_ROOT="${HARNESSGEN_EXTERNAL_REPOS_ROOT:-}"
OUTPUT_ROOT=""
MODEL="${HARNESSGEN_MODEL:-claude-opus-4.6}"
CONTAINER_A="${HARNESSGEN_CONTAINER_A:-harness-gen}"
CONTAINER_B="${HARNESSGEN_CONTAINER_B:-harness-gen2}"
CONTAINER_A_GPUS="${HARNESSGEN_CONTAINER_A_GPUS:-0,1,2}"
CONTAINER_B_GPUS="${HARNESSGEN_CONTAINER_B_GPUS:-3,4,5}"

LIST_MODE=false
PRINT_ONLY=false
RUN_ALL=false
SKIP_EXISTING=false

KERNELS_TO_RUN=()
RUN_PIDS=()
RUN_KERNELS=()
RUN_LOGS=()

readonly DEFAULT_KERNELS=(
    "mla_decode"
    "refk_identity"
    "mla_prefill_reduce"
    "rope"
    "gemm"
    "ff_backward"
    "fused_append_shared_experts"
    "refk_fp8_blockwise_mm"
    "gemm_a16w16_atomic"
    "batched_gemm_a16wfp4"
    "fused_qk_rope_cache_mla"
    "nsa_forward"
    "refk_mla_decode"
    "refk_moe"
    "fused_mxfp4_quant_moe_sort"
    "nsa_backward"
)

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run GEAK harness-only preprocess for the 16 AIG-Eval kernels that do not yet
have checked-in local harnesses.

This launcher:
  - lives inside the GEAK repo/branch
  - targets local cloned repos under AIG-Eval/external_repos
  - uses anchored local file specs (#L...)
  - splits work across two containers and six GPUs by default

Options:
  --all                     Run all 16 missing kernels
  --kernel NAME             Run a specific kernel (repeatable)
  --list                    List supported kernels, repo roots, and anchored specs
  --print-only              Print exact docker commands without executing
  --skip-existing           Skip kernels whose output already has harness_results.json
  --aig-eval-root PATH      AIG-Eval root (default: ${AIG_EVAL_ROOT:-<unset>})
  --external-repos-root PATH
                            External repos root (default: <AIG-Eval>/external_repos)
  --output-root PATH        Output root (default: <AIG-Eval>/tasks/geak_eval/preprocess_harness_outputs)
  --model NAME              Preprocess model (default: ${MODEL})
  --container-a NAME        First container (default: ${CONTAINER_A})
  --container-a-gpus IDS    CSV host GPU IDs for container A (default: ${CONTAINER_A_GPUS})
  --container-b NAME        Second container (default: ${CONTAINER_B})
  --container-b-gpus IDS    CSV host GPU IDs for container B (default: ${CONTAINER_B_GPUS})
  --geak-root PATH          GEAK root inside the containers (default: ${GEAK_ROOT})
  -h, --help                Show this help

Examples:
  ./$(basename "$0") --list
  ./$(basename "$0") --all --print-only
  ./$(basename "$0") --all --skip-existing
  ./$(basename "$0") --kernel mla_decode --kernel refk_identity
EOF
}

cleanup_jobs() {
    if [[ ${#RUN_PIDS[@]} -eq 0 ]]; then
        return
    fi
    for pid in "${RUN_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}

trap cleanup_jobs INT TERM

has_kernel() {
    local kernel="$1"
    local known
    for known in "${DEFAULT_KERNELS[@]}"; do
        if [[ "$kernel" == "$known" ]]; then
            return 0
        fi
    done
    return 1
}

parse_gpu_csv() {
    local csv="$1"
    local -n out_ref="$2"
    local old_ifs="$IFS"
    IFS=','
    read -r -a out_ref <<< "$csv"
    IFS="$old_ifs"
}

container_exists() {
    local container="$1"
    docker inspect "$container" >/dev/null 2>&1
}

ensure_container_running() {
    local container="$1"
    local running
    if ! container_exists "$container"; then
        echo "ERROR: Docker container '$container' does not exist."
        exit 1
    fi

    running="$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)"
    if [[ "$running" != "true" ]]; then
        echo "Starting container '$container'..."
        docker start "$container" >/dev/null
    fi
}

ensure_container_ready() {
    local container="$1"

    ensure_container_running "$container"

    docker exec "$container" bash -lc "
        source ~/.profile >/dev/null 2>&1 || true
        source ~/.bashrc >/dev/null 2>&1 || true
        test -d '$GEAK_ROOT'
        test -n \"\${AMD_LLM_API_KEY:-}\"
        cd '$GEAK_ROOT'
        python3 -m minisweagent.run.preprocess.preprocessor --help >/dev/null
    " >/dev/null
}

repo_basename() {
    local repo="$1"
    basename "$repo"
}

kernel_repo() {
    local kernel="$1"
    case "$kernel" in
        mla_decode|mla_prefill_reduce|rope|gemm|gemm_a16w16_atomic|batched_gemm_a16wfp4|fused_qk_rope_cache_mla|fused_mxfp4_quant_moe_sort)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/aiter"
            ;;
        ff_backward)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/Liger-Kernel"
            ;;
        nsa_forward|nsa_backward)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/native-sparse-attention"
            ;;
        fused_append_shared_experts)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/sglang"
            ;;
        refk_identity|refk_fp8_blockwise_mm|refk_mla_decode|refk_moe)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/reference-kernels"
            ;;
        *)
            return 1
            ;;
    esac
}

kernel_spec() {
    local kernel="$1"
    case "$kernel" in
        mla_decode)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/aiter/aiter/mla.py#L21"
            ;;
        mla_prefill_reduce)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/aiter/aiter/mla.py#L552"
            ;;
        rope)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/aiter/aiter/ops/triton/rope/rope.py#L106"
            ;;
        gemm)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/aiter/aiter/ops/triton/gemm/basic/gemm_a16w16.py#L18"
            ;;
        gemm_a16w16_atomic)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/aiter/aiter/ops/triton/gemm/basic/gemm_a16w16_atomic.py#L111"
            ;;
        batched_gemm_a16wfp4)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/aiter/aiter/ops/triton/gemm/batched/batched_gemm_a16wfp4.py#L222"
            ;;
        fused_qk_rope_cache_mla)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/aiter/aiter/ops/triton/fusions/fused_kv_cache.py#L72"
            ;;
        fused_mxfp4_quant_moe_sort)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/aiter/aiter/ops/triton/quant/fused_mxfp4_quant.py#L561"
            ;;
        ff_backward)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/Liger-Kernel/src/liger_kernel/ops/swiglu.py#L34"
            ;;
        nsa_forward)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/native-sparse-attention/native_sparse_attention/ops/parallel.py#L472"
            ;;
        nsa_backward)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/native-sparse-attention/native_sparse_attention/ops/parallel.py#L617"
            ;;
        fused_append_shared_experts)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py#L1081"
            ;;
        refk_identity)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/reference-kernels/problems/amd/identity/submission.py#L6"
            ;;
        refk_fp8_blockwise_mm)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/reference-kernels/problems/amd/fp8-mm/submission.py#L4"
            ;;
        refk_mla_decode)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/reference-kernels/problems/amd/mla-decode/submission.py#L156"
            ;;
        refk_moe)
            printf '%s\n' "${EXTERNAL_REPOS_ROOT}/reference-kernels/problems/amd/moe/submission.py#L102"
            ;;
        *)
            return 1
            ;;
    esac
}

list_kernels() {
    local kernel
    local repo
    local spec

    printf "  %-28s %-24s %s\n" "KERNEL" "REPO" "ANCHORED_LOCAL_SPEC"
    echo "  $(printf '%.0s-' {1..124})"
    for kernel in "${DEFAULT_KERNELS[@]}"; do
        repo="$(kernel_repo "$kernel")"
        spec="$(kernel_spec "$kernel")"
        printf "  %-28s %-24s %s\n" \
            "$kernel" \
            "$(repo_basename "$repo")" \
            "$spec"
    done
}

validate_paths() {
    local kernel
    local repo
    local spec
    local file_path

    if [[ -z "$AIG_EVAL_ROOT" || ! -d "$AIG_EVAL_ROOT" ]]; then
        echo "ERROR: AIG-Eval root not found. Use --aig-eval-root PATH."
        exit 1
    fi

    if [[ ! -d "$GEAK_ROOT" ]]; then
        echo "ERROR: GEAK root not found: $GEAK_ROOT"
        exit 1
    fi

    if [[ -z "$EXTERNAL_REPOS_ROOT" || ! -d "$EXTERNAL_REPOS_ROOT" ]]; then
        echo "ERROR: External repos root not found: ${EXTERNAL_REPOS_ROOT:-<unset>}"
        echo "       Use --external-repos-root PATH or ensure AIG-Eval/external_repos exists."
        exit 1
    fi

    for kernel in "${KERNELS_TO_RUN[@]}"; do
        repo="$(kernel_repo "$kernel")"
        spec="$(kernel_spec "$kernel")"
        file_path="${spec%%#*}"
        if [[ ! -d "$repo" ]]; then
            echo "ERROR: Repo root not found for $kernel: $repo"
            exit 1
        fi
        if [[ ! -f "$file_path" ]]; then
            echo "ERROR: Kernel file not found for $kernel: $file_path"
            exit 1
        fi
    done
}

build_inner_command() {
    local kernel="$1"
    local host_gpu="$2"
    local repo
    local spec
    local out_dir="${OUTPUT_ROOT}/${kernel}"
    local aiter_block

    repo="$(kernel_repo "$kernel")"
    spec="$(kernel_spec "$kernel")"

    if [[ "$(repo_basename "$repo")" == "aiter" ]]; then
        aiter_block="export AITER_ROOT='${repo}'"
    else
        aiter_block="unset AITER_ROOT || true"
    fi

    cat <<EOF
source ~/.profile >/dev/null 2>&1 || true
source ~/.bashrc >/dev/null 2>&1 || true
export GEAK_HARNESS_ONLY=1
export HIP_VISIBLE_DEVICES='${host_gpu}'
export ROCR_VISIBLE_DEVICES='${host_gpu}'
${aiter_block}
export PYTHONPATH='${repo}':"\${PYTHONPATH:-}"
cd '${GEAK_ROOT}'
python3 -m minisweagent.run.preprocess.preprocessor \
  '${spec}' \
  --repo '${repo}' \
  --gpu 0 \
  -m '${MODEL}' \
  -o '${out_dir}'
EOF
}

build_host_command_string() {
    local container="$1"
    local inner_cmd="$2"
    local quoted
    printf -v quoted 'docker exec %q bash -lc %q' "$container" "$inner_cmd"
    printf '%s\n' "$quoted"
}

should_skip_kernel() {
    local kernel="$1"
    local marker="${OUTPUT_ROOT}/${kernel}/harness_results.json"
    [[ "$SKIP_EXISTING" == "true" && -f "$marker" ]]
}

launch_kernel() {
    local kernel="$1"
    local container="$2"
    local host_gpu="$3"
    local inner_cmd
    local host_cmd
    local out_dir="${OUTPUT_ROOT}/${kernel}"
    local log_file="${OUTPUT_ROOT}/logs/${kernel}.log"

    mkdir -p "$out_dir" "${OUTPUT_ROOT}/logs"

    inner_cmd="$(build_inner_command "$kernel" "$host_gpu")"
    host_cmd="$(build_host_command_string "$container" "$inner_cmd")"

    printf '%s\n' "$host_cmd" > "${out_dir}/host_command.sh"
    printf '%s\n' "$inner_cmd" > "${out_dir}/container_command.sh"

    echo "Launching ${kernel} on ${container} (host GPU ${host_gpu})"
    echo "  output: ${out_dir}"
    echo "  log:    ${log_file}"

    (
        echo "=== Kernel: ${kernel}"
        echo "=== Container: ${container}"
        echo "=== Host GPU: ${host_gpu}"
        echo "=== Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=== Host command"
        echo "${host_cmd}"
        echo
        docker exec "$container" bash -lc "$inner_cmd"
    ) >"${log_file}" 2>&1 &

    RUN_PIDS+=("$!")
    RUN_KERNELS+=("$kernel")
    RUN_LOGS+=("$log_file")
}

wait_for_wave() {
    local failed=0
    local i
    local pid
    local kernel
    local log_file

    for i in "${!RUN_PIDS[@]}"; do
        pid="${RUN_PIDS[$i]}"
        kernel="${RUN_KERNELS[$i]}"
        log_file="${RUN_LOGS[$i]}"

        if wait "$pid"; then
            echo "  [ok]   ${kernel}"
        else
            echo "  [fail] ${kernel} (see ${log_file})"
            failed=1
        fi
    done

    RUN_PIDS=()
    RUN_KERNELS=()
    RUN_LOGS=()

    return "$failed"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            RUN_ALL=true
            shift
            ;;
        --kernel)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --kernel requires a kernel name"
                exit 1
            fi
            KERNELS_TO_RUN+=("$2")
            shift 2
            ;;
        --list)
            LIST_MODE=true
            shift
            ;;
        --print-only)
            PRINT_ONLY=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --aig-eval-root)
            AIG_EVAL_ROOT="$2"
            shift 2
            ;;
        --external-repos-root)
            EXTERNAL_REPOS_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --container-a)
            CONTAINER_A="$2"
            shift 2
            ;;
        --container-a-gpus)
            CONTAINER_A_GPUS="$2"
            shift 2
            ;;
        --container-b)
            CONTAINER_B="$2"
            shift 2
            ;;
        --container-b-gpus)
            CONTAINER_B_GPUS="$2"
            shift 2
            ;;
        --geak-root)
            GEAK_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$EXTERNAL_REPOS_ROOT" && -n "$AIG_EVAL_ROOT" ]]; then
    EXTERNAL_REPOS_ROOT="${AIG_EVAL_ROOT}/external_repos"
fi

if [[ -z "$OUTPUT_ROOT" && -n "$AIG_EVAL_ROOT" ]]; then
    OUTPUT_ROOT="${AIG_EVAL_ROOT}/tasks/geak_eval/preprocess_harness_outputs"
fi

if [[ "$LIST_MODE" == "true" ]]; then
    if [[ -z "$EXTERNAL_REPOS_ROOT" && -n "$AIG_EVAL_ROOT" ]]; then
        EXTERNAL_REPOS_ROOT="${AIG_EVAL_ROOT}/external_repos"
    fi
    echo ""
    echo "Supported preprocess kernels:"
    list_kernels
    echo ""
    exit 0
fi

if [[ "$RUN_ALL" == "true" ]]; then
    KERNELS_TO_RUN=("${DEFAULT_KERNELS[@]}")
fi

if [[ ${#KERNELS_TO_RUN[@]} -eq 0 ]]; then
    echo "No kernels specified. Use --all or --kernel NAME."
    echo ""
    usage
    exit 1
fi

for kernel in "${KERNELS_TO_RUN[@]}"; do
    if ! has_kernel "$kernel"; then
        echo "ERROR: Unknown kernel '$kernel'"
        echo ""
        list_kernels
        exit 1
    fi
done

validate_paths

declare -a GPU_POOL_A=()
declare -a GPU_POOL_B=()
declare -a SLOT_CONTAINERS=()
declare -a SLOT_GPUS=()

parse_gpu_csv "$CONTAINER_A_GPUS" GPU_POOL_A
parse_gpu_csv "$CONTAINER_B_GPUS" GPU_POOL_B

if [[ ${#GPU_POOL_A[@]} -eq 0 || ${#GPU_POOL_B[@]} -eq 0 ]]; then
    echo "ERROR: Both container GPU lists must contain at least one GPU."
    exit 1
fi

# Interleave slots across the two containers so partial waves also use both
# API keys instead of front-loading all work onto container A.
max_slots=${#GPU_POOL_A[@]}
if (( ${#GPU_POOL_B[@]} > max_slots )); then
    max_slots=${#GPU_POOL_B[@]}
fi

for ((slot_idx = 0; slot_idx < max_slots; slot_idx++)); do
    if (( slot_idx < ${#GPU_POOL_A[@]} )); then
        SLOT_CONTAINERS+=("$CONTAINER_A")
        SLOT_GPUS+=("${GPU_POOL_A[$slot_idx]}")
    fi
    if (( slot_idx < ${#GPU_POOL_B[@]} )); then
        SLOT_CONTAINERS+=("$CONTAINER_B")
        SLOT_GPUS+=("${GPU_POOL_B[$slot_idx]}")
    fi
done

if [[ "$PRINT_ONLY" != "true" ]]; then
    ensure_container_ready "$CONTAINER_A"
    ensure_container_ready "$CONTAINER_B"
fi

mkdir -p "${OUTPUT_ROOT}/logs"

echo ""
echo "=== AIG-Eval Harness Preprocess Runner ==="
echo "  GEAK root:     ${GEAK_ROOT}"
echo "  AIG-Eval root: ${AIG_EVAL_ROOT}"
echo "  model:         ${MODEL}"
echo "  output root:   ${OUTPUT_ROOT}"
echo "  container A:   ${CONTAINER_A} (GPUs: ${CONTAINER_A_GPUS})"
echo "  container B:   ${CONTAINER_B} (GPUs: ${CONTAINER_B_GPUS})"
echo "  kernels:       ${KERNELS_TO_RUN[*]}"
echo "  skip existing: ${SKIP_EXISTING}"
echo "  mode:          $([[ "$PRINT_ONLY" == "true" ]] && echo 'print-only' || echo 'execute')"
echo ""

BATCH_SIZE=${#SLOT_CONTAINERS[@]}
TOTAL_KERNELS=${#KERNELS_TO_RUN[@]}
TOTAL_WAVES=$(( (TOTAL_KERNELS + BATCH_SIZE - 1) / BATCH_SIZE ))

overall_failed=0

for ((offset = 0; offset < TOTAL_KERNELS; offset += BATCH_SIZE)); do
    wave_num=$(( offset / BATCH_SIZE + 1 ))
    wave_count=$(( TOTAL_KERNELS - offset ))
    if (( wave_count > BATCH_SIZE )); then
        wave_count=$BATCH_SIZE
    fi

    echo "=== Wave ${wave_num}/${TOTAL_WAVES} (${wave_count} slot(s)) ==="

    RUN_PIDS=()
    RUN_KERNELS=()
    RUN_LOGS=()

    actual_jobs=0
    for ((idx = 0; idx < wave_count; idx++)); do
        kernel="${KERNELS_TO_RUN[$((offset + idx))]}"

        if should_skip_kernel "$kernel"; then
            echo "Skipping ${kernel} (existing harness_results.json found)"
            continue
        fi

        container="${SLOT_CONTAINERS[$idx]}"
        gpu="${SLOT_GPUS[$idx]}"
        inner_cmd="$(build_inner_command "$kernel" "$gpu")"
        host_cmd="$(build_host_command_string "$container" "$inner_cmd")"

        if [[ "$PRINT_ONLY" == "true" ]]; then
            echo "--- ${kernel} | ${container} | host GPU ${gpu} ---"
            echo "${host_cmd}"
            echo ""
        else
            launch_kernel "$kernel" "$container" "$gpu"
        fi
        actual_jobs=$((actual_jobs + 1))
    done

    if [[ "$PRINT_ONLY" != "true" ]]; then
        if (( actual_jobs == 0 )); then
            echo "No jobs launched in this wave."
        elif ! wait_for_wave; then
            overall_failed=1
        fi
        echo ""
    fi
done

if [[ "$PRINT_ONLY" == "true" ]]; then
    echo "Printed all commands without executing them."
    exit 0
fi

echo "=== Summary ==="
if [[ "$overall_failed" -eq 0 ]]; then
    echo "All launched preprocess jobs completed successfully."
else
    echo "One or more preprocess jobs failed. Check logs under ${OUTPUT_ROOT}/logs/."
    exit 1
fi
