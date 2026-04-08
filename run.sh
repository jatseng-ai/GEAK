#!/bin/bash

TASK_FILE="GEAK_tasks.txt"
GPUS_PER_TASK=1
TOTAL_GPUS=1
START_GPU=1  # 起始 GPU ID
LOG_DIR="logs"  # 日志目录
export AMD_LLM_API_KEY="bdecd15bd6a348d9bb8f7daf6cd023b1"
export KERNEL_LLM_VISIBLE_DEVICES=4,5
export USE_KERNEL_LLM=1
# 创建日志目录
mkdir -p "$LOG_DIR"

# 计算最大并发任务数
available_gpus=1 #$((TOTAL_GPUS - START_GPU))
MAX_PARALLEL=1 #$((available_gpus / GPUS_PER_TASK))

echo "Max parallel tasks: $MAX_PARALLEL"
echo ""

# 存储后台进程 PID
declare -a pids=()
task_idx=0

while IFS=',' read -r repo_path kernel_url || [[ -n "$repo_path" ]]; do
    # 去除前后空格
    repo_path=$(echo "$repo_path" | xargs)
    kernel_url=$(echo "$kernel_url" | xargs)

    # 跳过空行
    [[ -z "$repo_path" ]] && continue

    # 从路径中提取 repo_name (最后一个目录名)
    repo_name=$(basename "$repo_path")

    offset=$((task_idx % MAX_PARALLEL * GPUS_PER_TASK))
    start_gpu=$((START_GPU + offset))
    end_gpu=$((start_gpu + GPUS_PER_TASK - 1))
    gpu_ids=$(seq -s',' $start_gpu $end_gpu)

    echo "=========================================="
    echo "Task $((task_idx + 1)): $repo_name"
    echo "Kernel URL: $kernel_url"
    echo "GPU IDs: $gpu_ids"
    echo "=========================================="

    # 获取当前时间戳
    timestamp=$(date +"%Y%m%d_%H%M%S")
    # 日志文件路径
    log_file="${LOG_DIR}/${repo_name}_${timestamp}.log"


    # 后台运行任务
    geak --kernel-url "$kernel_url" \
         --repo "$repo_path" \
         --task "Optimize the repository ${repo_path}, test command is python3 scripts/task_runner.py compile && python3 scripts/task_runner.py correctness && python3 scripts/task_runner.py performance" \
         --num-parallel 1 \
         --gpu-ids "$gpu_ids" \
         > "$log_file" 2>&1 &

    pids+=($!)
    echo "Started task $((task_idx + 1)) with PID ${pids[-1]}"
    echo ""

    task_idx=$((task_idx + 1))

    # 如果达到最大并发数，等待任意一个任务完成
    if (( ${#pids[@]} >= MAX_PARALLEL )); then
        echo "Reached max parallel ($MAX_PARALLEL), waiting for a task to finish..."
        wait -n  # 等待任意一个后台任务完成
        # 清理已完成的 PID
        new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        pids=("${new_pids[@]}")
    fi
done < "$TASK_FILE"

# 等待所有剩余任务完成
echo "=========================================="
echo "Waiting for all remaining tasks to complete..."
echo "=========================================="
wait

echo ""
echo "All tasks completed!"
