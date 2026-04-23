#!/bin/bash
# CI benchmark script — runs GEAK on AgentKernelArena tasks in parallel.
# Adapted from batch_test.sh for GitHub Actions (4 GPUs, no skip, hardcoded tasks).

set -euo pipefail

ARENA_ROOT="${ARENA_ROOT:-../arena}"
GPUS_PER_TASK=2
TOTAL_GPUS=4
LOG_DIR="logs"

# 16 benchmark tasks: repo_path,kernel_path (relative to ARENA_ROOT)
TASKS=(
  "tasks/hip2hip/others/assign_score_withk,tasks/hip2hip/others/assign_score_withk/assign_score_withk_wrapper.py"
  "tasks/hip2hip/others/ball_query,tasks/hip2hip/others/ball_query/ball_query_wrapper.py"
  "tasks/hip2hip/others/furthest_point_sample,tasks/hip2hip/others/furthest_point_sample/furthest_point_sample_wrapper.py"
  "tasks/hip2hip/others/gather_points,tasks/hip2hip/others/gather_points/gather_points_wrapper.py"
  "tasks/hip2hip/others/knn,tasks/hip2hip/others/knn/knn_wrapper.py"
  "tasks/hip2hip/others/matrix_multiplication,tasks/hip2hip/others/matrix_multiplication/main.hip"
  "tasks/hip2hip/others/points_in_boxes,tasks/hip2hip/others/points_in_boxes/points_in_boxes_wrapper.py"
  "tasks/hip2hip/others/roiaware_pool3d,tasks/hip2hip/others/roiaware_pool3d/roiaware_pool3d_wrapper.py"
  "tasks/hip2hip/others/roipoint_pool3d,tasks/hip2hip/others/roipoint_pool3d/roipoint_pool3d_wrapper.py"
  "tasks/hip2hip/others/silu,tasks/hip2hip/others/silu/silu.hip"
  "tasks/hip2hip/others/three_interpolate,tasks/hip2hip/others/three_interpolate/three_interpolate_wrapper.py"
  "tasks/hip2hip/others/three_nn,tasks/hip2hip/others/three_nn/three_nn_wrapper.py"
  "tasks/repository/rocprim/block_radix_rank,tasks/repository/rocprim/block_radix_rank/rocPRIM/rocprim/include/rocprim/block/block_radix_rank.hpp"
  "tasks/repository/rocprim/device_binary_search,tasks/repository/rocprim/device_binary_search/rocPRIM/rocprim/include/rocprim/device/device_binary_search.hpp"
  "tasks/repository/rocprim/device_merge_sort,tasks/repository/rocprim/device_merge_sort/rocPRIM/rocprim/include/rocprim/device/device_merge_sort.hpp"
  "tasks/repository/rocprim/device_search_n,tasks/repository/rocprim/device_search_n/rocPRIM/rocprim/include/rocprim/device/device_search_n"
)

mkdir -p "$LOG_DIR"

# Build GPU slot queue
declare -a free_gpu_slots=()
idx=0
while (( idx + GPUS_PER_TASK <= TOTAL_GPUS )); do
    gpu_ids="$idx"
    for ((j = 1; j < GPUS_PER_TASK; j++)); do
        gpu_ids+=",$((idx + j))"
    done
    free_gpu_slots+=("$gpu_ids")
    idx=$((idx + GPUS_PER_TASK))
done

MAX_PARALLEL=${#free_gpu_slots[@]}
echo "Max parallel tasks: $MAX_PARALLEL"
echo "GPU slots: ${free_gpu_slots[*]}"
echo ""

declare -A pid_to_gpus=()
task_idx=0
failed=0

for entry in "${TASKS[@]}"; do
    IFS=',' read -r rel_repo rel_kernel <<< "$entry"
    repo_path="${ARENA_ROOT}/${rel_repo}"
    kernel_url="${ARENA_ROOT}/${rel_kernel}"
    repo_name=$(basename "$rel_repo")

    # Wait for a free GPU slot
    while (( ${#free_gpu_slots[@]} == 0 )); do
        echo "No free GPU slots, waiting for a task to finish..."
        wait -n || true
        for pid in "${!pid_to_gpus[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" || { echo "Task PID $pid failed"; failed=$((failed + 1)); }
                echo "Task with PID $pid finished, freeing GPUs: ${pid_to_gpus[$pid]}"
                free_gpu_slots+=("${pid_to_gpus[$pid]}")
                unset pid_to_gpus["$pid"]
            fi
        done
    done

    gpu_ids="${free_gpu_slots[0]}"
    free_gpu_slots=("${free_gpu_slots[@]:1}")

    echo "=========================================="
    echo "Task $((task_idx + 1))/${#TASKS[@]}: $repo_name"
    echo "Kernel: $kernel_url"
    echo "GPU IDs: $gpu_ids"
    echo "=========================================="

    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="${LOG_DIR}/${repo_name}_${timestamp}.log"

    geak --kernel-url "$kernel_url" \
         --repo "$repo_path" \
         --task "Optimize the repository ${repo_path}, test command is python3 scripts/task_runner.py compile && python3 scripts/task_runner.py correctness && python3 scripts/task_runner.py performance" \
         --num-parallel "$GPUS_PER_TASK" \
         --gpu-ids "$gpu_ids" \
         > "$log_file" 2>&1 &

    pid=$!
    pid_to_gpus[$pid]="$gpu_ids"
    echo "Started task $((task_idx + 1)) with PID $pid on GPUs $gpu_ids"
    echo ""

    task_idx=$((task_idx + 1))
done

# Wait for remaining jobs
echo "=========================================="
echo "Waiting for all remaining tasks to complete..."
echo "=========================================="
for pid in "${!pid_to_gpus[@]}"; do
    wait "$pid" || { echo "Task PID $pid failed"; failed=$((failed + 1)); }
done

echo ""
echo "All ${#TASKS[@]} tasks completed. Failures: $failed"

if (( failed > 0 )); then
    exit 1
fi
