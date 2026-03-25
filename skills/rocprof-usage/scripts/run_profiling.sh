#!/bin/bash
# GPU Kernel Profiling Script using rocprof
# Usage: ./run_profiling.sh <test_script.py> [kernel_name]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <test_script.py> [kernel_name]"
    echo "  test_script.py - Python script that runs the GPU kernel"
    echo "  kernel_name    - Optional: filter to specific kernel name"
    exit 1
fi

TEST_SCRIPT=$1
KERNEL_NAME=${2:-""}
OUTPUT_DIR="rocprof_results_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "GPU Kernel Profiling with rocprof"
echo "=========================================="
echo "Test script: $TEST_SCRIPT"
echo "Kernel filter: ${KERNEL_NAME:-'(all kernels)'}"
echo "Output directory: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Basic timing and stats
echo "[1/6] Running basic profiling (timing stats)..."
rocprof --stats -o "$OUTPUT_DIR/timing.csv" python3 "$TEST_SCRIPT" 2>&1 | tee "$OUTPUT_DIR/profiling.log"
echo ""

# Step 2: Instruction metrics
echo "[2/6] Collecting instruction metrics..."
cat > "$OUTPUT_DIR/instr_input.txt" << METRICS
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR SQ_INSTS_LDS
METRICS
if [ -n "$KERNEL_NAME" ]; then
    echo "kernel: $KERNEL_NAME" >> "$OUTPUT_DIR/instr_input.txt"
fi
rocprof -i "$OUTPUT_DIR/instr_input.txt" -o "$OUTPUT_DIR/instr_metrics.csv" python3 "$TEST_SCRIPT" 2>&1 | tee -a "$OUTPUT_DIR/profiling.log"
echo ""

# Step 3: Utilization metrics
echo "[3/6] Collecting utilization metrics..."
cat > "$OUTPUT_DIR/util_input.txt" << METRICS
pmc: VALUUtilization VALUBusy SALUBusy MemUnitStalled
METRICS
if [ -n "$KERNEL_NAME" ]; then
    echo "kernel: $KERNEL_NAME" >> "$OUTPUT_DIR/util_input.txt"
fi
rocprof -i "$OUTPUT_DIR/util_input.txt" -o "$OUTPUT_DIR/util_metrics.csv" python3 "$TEST_SCRIPT" 2>&1 | tee -a "$OUTPUT_DIR/profiling.log"
echo ""

# Step 4: Occupancy and memory bandwidth metrics
echo "[4/6] Collecting occupancy and memory bandwidth metrics..."
cat > "$OUTPUT_DIR/occup_input.txt" << METRICS
pmc: MeanOccupancyPerCU SIMD_UTILIZATION FETCH_SIZE WRITE_SIZE
METRICS
if [ -n "$KERNEL_NAME" ]; then
    echo "kernel: $KERNEL_NAME" >> "$OUTPUT_DIR/occup_input.txt"
fi
rocprof -i "$OUTPUT_DIR/occup_input.txt" -o "$OUTPUT_DIR/occup_metrics.csv" python3 "$TEST_SCRIPT" 2>&1 | tee -a "$OUTPUT_DIR/profiling.log"
echo ""

# Step 5: Cache metrics (L2 cache hit/miss)
echo "[5/6] Collecting cache metrics..."
cat > "$OUTPUT_DIR/cache_input.txt" << METRICS
pmc: TCC_HIT_sum TCC_MISS_sum TCP_TOTAL_CACHE_ACCESSES_sum
METRICS
if [ -n "$KERNEL_NAME" ]; then
    echo "kernel: $KERNEL_NAME" >> "$OUTPUT_DIR/cache_input.txt"
fi
rocprof -i "$OUTPUT_DIR/cache_input.txt" -o "$OUTPUT_DIR/cache_metrics.csv" python3 "$TEST_SCRIPT" 2>&1 | tee -a "$OUTPUT_DIR/profiling.log"
echo ""

# Step 6: LDS bank conflict metrics
echo "[6/6] Collecting LDS bank conflict metrics..."
cat > "$OUTPUT_DIR/lds_input.txt" << METRICS
pmc: SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS SQ_WAIT_INST_LDS
METRICS
if [ -n "$KERNEL_NAME" ]; then
    echo "kernel: $KERNEL_NAME" >> "$OUTPUT_DIR/lds_input.txt"
fi
rocprof -i "$OUTPUT_DIR/lds_input.txt" -o "$OUTPUT_DIR/lds_metrics.csv" python3 "$TEST_SCRIPT" 2>&1 | tee -a "$OUTPUT_DIR/profiling.log"
echo ""

echo "=========================================="
echo "Profiling Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Files generated:"
ls -la "$OUTPUT_DIR/"*.csv 2>/dev/null || echo "  (check $OUTPUT_DIR for output files)"
echo ""
echo "=== REGISTER/SCRATCH USAGE (from timing.csv) ==="
echo "Check columns: arch_vgpr, accum_vgpr, sgpr, scr (scratch), lds"
echo "  - scr > 0 indicates register spilling"
head -2 "$OUTPUT_DIR/timing.csv" 2>/dev/null | cut -d',' -f11-16
echo ""
echo "Quick analysis:"
echo "  python3 scripts/analyze_rocprof.py --stats $OUTPUT_DIR/timing.stats.csv"
