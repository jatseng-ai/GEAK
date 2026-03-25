---
name: rocprof-usage
description: This skill describes how to use AMD's `rocprof` tool to analyze GPU kernel performance and identify bottlenecks on AMD GPUs.
---

# GPU Kernel Profiling Skill with rocprof

This document describes how to use AMD's `rocprof` tool to analyze GPU kernel performance and identify bottlenecks on AMD GPUs.

## Prerequisites

- AMD ROCm installed (tested with ROCm 6.4.3)
- HIP kernel code to analyze
- Python 3.x for analysis scripts

## Quick Start

1. Run basic profiling with kernel stats:
   
       rocprof --stats -o output.csv python3 your_test_script.py

2. Run detailed metrics collection:
   
       rocprof -i scripts/rocprof_input.txt -o metrics.csv python3 your_test_script.py

3. Analyze results:
   
       python3 scripts/analyze_rocprof.py --stats output.stats.csv --metrics metrics.csv

## Profiling Commands

### 1. Basic Kernel Timing and Statistics

Command:

    rocprof --stats -o rocprof_output.csv python3 test_script.py

Output files:
- rocprof_output.csv - Per-kernel timing data (DispatchNs, BeginNs, EndNs, DurationNs)
- rocprof_output.stats.csv - Aggregated statistics (Calls, TotalDurationNs, AverageNs, Percentage)
- rocprof_output.json - Chrome tracing format (can be viewed in chrome://tracing)

### 2. Hardware Counter Metrics

Create an input file `rocprof_input.txt` with content:

    # Group 1: Instruction metrics
    pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR SQ_INSTS_LDS
    
    # Group 2: Utilization metrics  
    pmc: VALUUtilization VALUBusy SALUBusy MemUnitStalled
    
    # Group 3: Occupancy metrics
    pmc: MeanOccupancyPerCU SIMD_UTILIZATION
    
    # Optional: Filter to specific kernel
    kernel: your_kernel_name

Run with:

    rocprof -i rocprof_input.txt -o rocprof_metrics.csv python3 test_script.py

### 3. Memory Bandwidth Metrics

Create `rocprof_memory.txt` with content:

    pmc: FETCH_SIZE WRITE_SIZE
    kernel: your_kernel_name

Run with:

    rocprof -i rocprof_memory.txt -o rocprof_memory.csv python3 test_script.py

### 4. HIP/HSA API Tracing

    # Trace HIP API calls
    rocprof --hip-trace -o hip_trace.csv python3 test_script.py
    
    # Trace HSA API calls  
    rocprof --hsa-trace -o hsa_trace.csv python3 test_script.py
    
    # Full system trace
    rocprof --sys-trace -o sys_trace.csv python3 test_script.py

## Key Metrics Explained

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| VALUUtilization | % of active threads in a wave | 100% (no divergence) |
| VALUBusy | % of time VALU is processing | Higher = compute-bound |
| SALUBusy | % of time SALU is processing | - |
| MemUnitStalled | % of time waiting for memory | Lower = better |
| MeanOccupancyPerCU | Active waves per compute unit | Higher = better latency hiding |
| SIMD_UTILIZATION | % of time with active warps | Higher = better |
| FETCH_SIZE | KB read from memory | - |
| WRITE_SIZE | KB written to memory | - |
| **TCC_HIT_sum** | L2 cache hits | Higher = better |
| **TCC_MISS_sum** | L2 cache misses | Lower = better |
| **SQ_LDS_BANK_CONFLICT** | Cycles stalled on LDS bank conflicts | 0 = ideal |

### Register and Memory Usage (from CSV columns)

| Column | Description | Warning Threshold |
|--------|-------------|-------------------|
| arch_vgpr | Vector GPRs used per kernel | >64 may limit occupancy |
| accum_vgpr | Accumulator VGPRs used | - |
| sgpr | Scalar GPRs used | >48 may limit occupancy |
| scr | Scratch memory (bytes) | >0 = register spilling! |
| lds | LDS memory used (bytes) | Depends on workgroup count |


## Bottleneck Classification

Based on metrics, classify your kernel:

1. **COMPUTE-BOUND**: High VALUBusy (>50%), low MemUnitStalled
   - Optimize: Reduce ALU operations, use faster math

2. **MEMORY-BOUND**: High MemUnitStalled (>20%), low VALUBusy
   - Optimize: Coalesce memory access, use caching, reduce bandwidth

3. **LATENCY-BOUND**: Low VALUBusy AND low MemUnitStalled
   - Optimize: Increase occupancy, kernel fusion, reduce launch overhead

## Available Metrics

List all available metrics:

    rocprof --list-basic    # Basic hardware counters
    rocprof --list-derived  # Derived/computed metrics

## Helper Scripts

See the `scripts/` folder for:
- analyze_rocprof.py - Parse and analyze rocprof output
- run_profiling.sh - Automated profiling script
- rocprof_input.txt - Template input file for metrics

## Example Workflow

    # Step 1: Create test script that runs your kernel
    # Step 2: Run basic profiling
    rocprof --stats -o output.csv python3 test_script.py
    
    # Step 3: Identify the kernel name from output.stats.csv
    # Step 4: Create input file with kernel filter
    # Step 5: Collect detailed metrics
    rocprof -i rocprof_input.txt -o metrics.csv python3 test_script.py
    
    # Step 6: Analyze results
    python3 scripts/analyze_rocprof.py --metrics metrics.csv --stats output.stats.csv

---

## Important Considerations for Different Kernels

### 1. GPU Architecture-Specific Metrics

Different AMD GPU architectures support different metrics. Always check available metrics first:

    rocprof --list-derived 2>&1 | grep -i "gpu-agent"

**Known architecture differences:**
- **gfx908 (MI100)**: Supports most metrics
- **gfx90a (MI200)**: Supports most metrics  
- **gfx942 (MI300)**: Some metrics like `L2CacheHit` may not be supported
- **gfx1030/gfx1100 (RDNA)**: Different metric names and availability

If you get "metric not supported" errors, remove that metric from your input file.

### 2. Kernel Name Filtering

Kernel names in rocprof output are mangled C++ names. To find the exact name:

    # First run without filter to see all kernel names
    rocprof --stats -o output.csv python3 test.py
    
    # Check the stats file for kernel names
    cat output.stats.csv | head -5

Use partial matching - rocprof matches substrings:

    kernel: my_kernel    # Matches "my_kernel_v1", "my_kernel_float", etc.

### 3. Multiple Kernel Analysis

If your application has multiple kernels:

    # Profile all kernels first
    rocprof --stats -o all_kernels.csv python3 test.py
    
    # Identify hotspots from stats file (sorted by percentage)
    # Then profile specific kernels with detailed metrics

### 4. Metric Collection Limitations

**Important:** rocprof can only collect ~6-8 metrics per run due to hardware counter limitations.

Group your metrics logically:

    # Run 1: Instruction metrics
    pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR SQ_INSTS_LDS
    
    # Run 2: Utilization metrics (separate run)
    pmc: VALUUtilization VALUBusy SALUBusy MemUnitStalled
    
    # Run 3: Occupancy metrics (separate run)
    pmc: MeanOccupancyPerCU SIMD_UTILIZATION

### 5. Warmup and Iteration Count

For accurate profiling:
- **Warmup runs**: Execute kernel 3-5 times before measurement to warm caches
- **Multiple iterations**: Run kernel 50-100+ times for stable averages
- **Synchronization**: Always call `torch.cuda.synchronize()` or `hipDeviceSynchronize()`

Example test script pattern:

    # Warmup
    for _ in range(5):
        output = my_kernel(inputs)
    torch.cuda.synchronize()
    
    # Timed runs
    for _ in range(100):
        output = my_kernel(inputs)
    torch.cuda.synchronize()

### 6. Profiling Overhead

Be aware that profiling adds overhead:
- **--stats only**: Minimal overhead (~5-10%)
- **Hardware counters (pmc)**: Moderate overhead (~2-5x slower)
- **API tracing**: Significant overhead (~10-50x slower)

For performance-critical measurements, use `--stats` only.

### 7. Common Kernel Types and What to Look For

**Matrix Multiplication / GEMM:**
- Look for: High VALUBusy, good occupancy
- Common issues: Memory bandwidth, tile size

**Reduction Kernels:**
- Look for: Atomic operation overhead, warp divergence
- Common issues: Low occupancy due to synchronization

**Element-wise Kernels:**
- Look for: Memory bandwidth utilization
- Common issues: Memory-bound, low arithmetic intensity

**Convolution Kernels:**
- Look for: Cache hit rates, memory access patterns
- Common issues: Strided access, bank conflicts

**Sparse Operations:**
- Look for: Thread divergence (VALUUtilization < 100%)
- Common issues: Irregular memory access, load imbalance

### 8. Interpreting Low Utilization

If both VALUBusy and MemUnitStalled are low:

1. **Kernel is too small**: Not enough work to saturate GPU
   - Solution: Batch operations, increase problem size

2. **Launch overhead dominates**: Kernel executes in microseconds
   - Solution: Kernel fusion, persistent kernels

3. **Synchronization barriers**: Threads waiting on each other
   - Solution: Reduce barriers, better work distribution

4. **Occupancy limited**: Not enough waves to hide latency
   - Solution: Reduce register/LDS usage

### 9. Register Pressure Analysis

Check register usage in the CSV output:
- `arch_vgpr`: Vector general-purpose registers used
- `sgpr`: Scalar general-purpose registers used
- `lds`: Local data share memory used

**Occupancy limits (approximate for MI200/MI300):**
- 256 VGPRs max per wave → <64 VGPRs for good occupancy
- 128 SGPRs max per wave → <48 SGPRs for good occupancy
- 64KB LDS per workgroup → depends on workgroup count

### 10. Comparing Kernel Versions

When optimizing, compare metrics between versions:

    # Version 1
    rocprof --stats -o v1_stats.csv python3 test_v1.py
    
    # Version 2  
    rocprof --stats -o v2_stats.csv python3 test_v2.py
    
    # Compare average duration
    grep "your_kernel" v1_stats.csv v2_stats.csv

---

## Troubleshooting

1. **Metric not supported**: Some metrics are GPU-architecture specific. 
   - Check `rocprof --list-derived` for your GPU
   - Remove unsupported metrics from input file

2. **Context create failed**: Too many metrics in one group.
   - Reduce to max 6-8 metrics per pmc line
   - Split into multiple runs

3. **No kernel data**: Kernel didn't execute or program exited early.
   - Add synchronization before program exit
   - Ensure kernel actually runs (check for errors)

4. **Inconsistent results**: Profiling overhead affecting measurements.
   - Use more iterations for averaging
   - Compare relative changes, not absolute values

5. **rocprof hangs**: Large trace files or many kernel invocations.
   - Use kernel filter to reduce data
   - Limit iteration count during profiling

## References

- ROCm Profiler Documentation: https://rocm.docs.amd.com/projects/rocprofiler/en/latest/
- ROCm Performance Guidelines: https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch.html
- AMD GPU Architecture: https://gpuopen.com/learn/
