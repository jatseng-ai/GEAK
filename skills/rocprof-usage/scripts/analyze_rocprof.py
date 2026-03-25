#!/usr/bin/env python3
"""
Analyze rocprof output to identify GPU kernel bottlenecks

Usage:
    python3 analyze_rocprof.py [--metrics METRICS_CSV] [--stats STATS_CSV] [--memory MEMORY_CSV] [--dir DIR]
"""

import csv
import statistics
import argparse
import os
import glob

def read_csv_data(filepath):
    """Read CSV file and return list of dicts"""
    if not filepath or not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return list(csv.DictReader(f))

def analyze_rocprof_metrics(stats_file=None, metrics_file=None, memory_file=None, 
                            cache_file=None, lds_file=None, output_dir=None):
    print("=" * 80)
    print("GPU KERNEL PROFILING ANALYSIS REPORT")
    print("=" * 80)
    
    # If output_dir specified, look for files there
    if output_dir:
        if not stats_file:
            stats_file = os.path.join(output_dir, 'timing.stats.csv')
        if not metrics_file:
            # Try to find metrics files
            for name in ['instr_metrics.csv', 'util_metrics.csv', 'metrics.csv', 'rocprof_metrics.csv']:
                path = os.path.join(output_dir, name)
                if os.path.exists(path):
                    metrics_file = path
                    break
        if not cache_file:
            cache_file = os.path.join(output_dir, 'cache_metrics.csv')
        if not lds_file:
            lds_file = os.path.join(output_dir, 'lds_metrics.csv')
        if not memory_file:
            memory_file = os.path.join(output_dir, 'occup_metrics.csv')
    
    # Read stats file for overall timing
    if stats_file and os.path.exists(stats_file):
        print("\n" + "=" * 80)
        print("1. KERNEL EXECUTION TIME SUMMARY")
        print("=" * 80)
        
        stats_data = read_csv_data(stats_file)
        for row in stats_data:
            name = row.get('Name', '')
            # Skip PyTorch internal kernels
            if 'at::native' in name or 'void at::' in name:
                continue
                
            calls = int(row.get('Calls', 0))
            total_ns = int(row.get('TotalDurationNs', 0))
            avg_ns = float(row.get('AverageNs', 0))
            pct = float(row.get('Percentage', 0))
            
            short_name = name.split('(')[0] if '(' in name else name
            
            print(f"\nKernel: {short_name}")
            print(f"  Total Calls: {calls}")
            print(f"  Total Duration: {total_ns/1e6:.3f} ms")
            print(f"  Average Duration: {avg_ns/1e3:.3f} µs ({avg_ns:.0f} ns)")
            print(f"  Percentage of GPU Time: {pct:.2f}%")
    
    # Read metrics files
    metrics_data = read_csv_data(metrics_file) if metrics_file else []
    
    # Also try to read util_metrics if available
    util_data = []
    if output_dir:
        util_file = os.path.join(output_dir, 'util_metrics.csv')
        if os.path.exists(util_file):
            util_data = read_csv_data(util_file)
    
    # Combine metrics data
    if not metrics_data and util_data:
        metrics_data = util_data
    
    if metrics_data:
        print("\n" + "=" * 80)
        print("2. KERNEL CONFIGURATION & REGISTER USAGE")
        print("=" * 80)
        
        sample = metrics_data[0]
        grid_size = int(sample.get('grd', 0))
        workgroup_size = int(sample.get('wgr', 0))
        lds_size = int(sample.get('lds', 0))
        scratch_size = int(sample.get('scr', 0))
        arch_vgpr = int(sample.get('arch_vgpr', 0))
        accum_vgpr = int(sample.get('accum_vgpr', 0))
        sgpr = int(sample.get('sgpr', 0))
        wave_size = int(sample.get('wave_size', 64))
        
        print(f"\n  Grid Size (total threads): {grid_size}")
        print(f"  Workgroup Size (threads per block): {workgroup_size}")
        if workgroup_size > 0:
            print(f"  Number of Workgroups: {grid_size // workgroup_size}")
        print(f"  Wave Size: {wave_size}")
        
        print(f"\n  === REGISTER USAGE ===")
        print(f"  Arch VGPRs: {arch_vgpr}")
        print(f"  Accumulator VGPRs: {accum_vgpr}")
        print(f"  SGPRs: {sgpr}")
        print(f"  LDS Usage: {lds_size} bytes")
        print(f"  Scratch Memory: {scratch_size} bytes")
        
        if scratch_size > 0:
            print(f"\n  ⚠️  WARNING: Scratch memory > 0 indicates REGISTER SPILLING!")
            print(f"      This can significantly hurt performance.")
            print(f"      Consider reducing register pressure.")
        else:
            print(f"\n  ✓  No register spilling (scratch = 0)")
    
    # Instruction mix analysis
    if metrics_data and 'SQ_WAVES' in metrics_data[0]:
        print("\n" + "=" * 80)
        print("3. INSTRUCTION MIX ANALYSIS")
        print("=" * 80)
        
        sq_waves = int(metrics_data[0].get('SQ_WAVES', 0))
        sq_insts_valu = int(metrics_data[0].get('SQ_INSTS_VALU', 0))
        sq_insts_salu = int(metrics_data[0].get('SQ_INSTS_SALU', 0))
        sq_insts_vmem_rd = int(metrics_data[0].get('SQ_INSTS_VMEM_RD', 0))
        sq_insts_vmem_wr = int(metrics_data[0].get('SQ_INSTS_VMEM_WR', 0))
        sq_insts_lds = int(metrics_data[0].get('SQ_INSTS_LDS', 0))
        
        total_insts = sq_insts_valu + sq_insts_salu + sq_insts_vmem_rd + sq_insts_vmem_wr + sq_insts_lds
        
        if total_insts > 0:
            print(f"\n  Total Waves: {sq_waves}")
            print(f"\n  Instruction Counts:")
            print(f"    VALU (Vector ALU):     {sq_insts_valu:>10} ({100*sq_insts_valu/total_insts:.1f}%)")
            print(f"    SALU (Scalar ALU):     {sq_insts_salu:>10} ({100*sq_insts_salu/total_insts:.1f}%)")
            print(f"    VMEM Read:             {sq_insts_vmem_rd:>10} ({100*sq_insts_vmem_rd/total_insts:.1f}%)")
            print(f"    VMEM Write:            {sq_insts_vmem_wr:>10} ({100*sq_insts_vmem_wr/total_insts:.1f}%)")
            print(f"    LDS:                   {sq_insts_lds:>10} ({100*sq_insts_lds/total_insts:.1f}%)")
    
    # Utilization metrics
    data_for_util = util_data if util_data else metrics_data
    if data_for_util and 'VALUUtilization' in data_for_util[0]:
        print("\n" + "=" * 80)
        print("4. UTILIZATION METRICS")
        print("=" * 80)
        
        valu_util = [float(d['VALUUtilization']) for d in data_for_util if d.get('VALUUtilization')]
        valu_busy = [float(d['VALUBusy']) for d in data_for_util if d.get('VALUBusy')]
        salu_busy = [float(d['SALUBusy']) for d in data_for_util if d.get('SALUBusy')]
        mem_stall = [float(d['MemUnitStalled']) for d in data_for_util if d.get('MemUnitStalled')]
        
        if valu_util:
            avg_valu_util = statistics.mean(valu_util)
            print(f"\n  VALU Utilization: {avg_valu_util:.2f}%")
            if avg_valu_util < 100:
                print(f"    ⚠️  Thread divergence detected ({100-avg_valu_util:.1f}% inactive threads)")
            else:
                print(f"    ✓  No thread divergence")
        
        if valu_busy:
            print(f"\n  VALU Busy: {statistics.mean(valu_busy):.2f}%")
        if salu_busy:
            print(f"  SALU Busy: {statistics.mean(salu_busy):.2f}%")
        if mem_stall:
            print(f"  Memory Unit Stalled: {statistics.mean(mem_stall):.2f}%")
    
    # Cache metrics
    cache_data = read_csv_data(cache_file) if cache_file else []
    if cache_data and 'TCC_HIT_sum' in cache_data[0]:
        print("\n" + "=" * 80)
        print("5. CACHE METRICS (L2 Cache)")
        print("=" * 80)
        
        tcc_hits = [float(d['TCC_HIT_sum']) for d in cache_data if d.get('TCC_HIT_sum')]
        tcc_misses = [float(d['TCC_MISS_sum']) for d in cache_data if d.get('TCC_MISS_sum')]
        
        if tcc_hits and tcc_misses:
            total_hits = sum(tcc_hits)
            total_misses = sum(tcc_misses)
            total_accesses = total_hits + total_misses
            
            if total_accesses > 0:
                hit_rate = 100 * total_hits / total_accesses
                print(f"\n  L2 Cache (TCC) Statistics:")
                print(f"    Total Hits: {total_hits:.0f}")
                print(f"    Total Misses: {total_misses:.0f}")
                print(f"    Hit Rate: {hit_rate:.2f}%")
                
                if hit_rate < 50:
                    print(f"\n    ⚠️  Low cache hit rate - consider improving data locality")
                elif hit_rate > 90:
                    print(f"\n    ✓  Excellent cache hit rate")
    
    # LDS bank conflict metrics
    lds_data = read_csv_data(lds_file) if lds_file else []
    if lds_data and 'SQ_LDS_BANK_CONFLICT' in lds_data[0]:
        print("\n" + "=" * 80)
        print("6. LDS BANK CONFLICT METRICS")
        print("=" * 80)
        
        bank_conflicts = [float(d['SQ_LDS_BANK_CONFLICT']) for d in lds_data if d.get('SQ_LDS_BANK_CONFLICT')]
        lds_insts = [float(d['SQ_INSTS_LDS']) for d in lds_data if d.get('SQ_INSTS_LDS')]
        
        if bank_conflicts:
            total_conflicts = sum(bank_conflicts)
            total_lds_insts = sum(lds_insts) if lds_insts else 0
            
            print(f"\n  LDS Bank Conflict Cycles: {total_conflicts:.0f}")
            print(f"  LDS Instructions: {total_lds_insts:.0f}")
            
            if total_lds_insts > 0:
                conflict_rate = total_conflicts / total_lds_insts
                print(f"  Conflicts per LDS Instruction: {conflict_rate:.2f}")
                
                if conflict_rate > 1:
                    print(f"\n    ⚠️  High LDS bank conflict rate!")
                    print(f"        Consider padding shared memory or changing access patterns")
                elif total_conflicts == 0:
                    print(f"\n    ✓  No LDS bank conflicts")
            elif total_conflicts == 0:
                print(f"\n    ✓  No LDS usage or conflicts")
    
    # Bottleneck analysis
    print("\n" + "=" * 80)
    print("7. BOTTLENECK SUMMARY")
    print("=" * 80)
    
    valu_busy_avg = statistics.mean(valu_busy) if 'valu_busy' in dir() and valu_busy else 0
    mem_stall_avg = statistics.mean(mem_stall) if 'mem_stall' in dir() and mem_stall else 0
    
    print("\n  Kernel Classification:")
    if valu_busy_avg < 10 and mem_stall_avg < 5:
        print("    → LATENCY BOUND: Kernel completes quickly, launch overhead may dominate")
    elif mem_stall_avg > 20:
        print("    → MEMORY BOUND: Memory access is the primary bottleneck")
    elif valu_busy_avg > 50:
        print("    → COMPUTE BOUND: ALU operations are the primary bottleneck")
    else:
        print("    → BALANCED: No single dominant bottleneck")
    
    print("\n" + "=" * 80)
    print("END OF ANALYSIS REPORT")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Analyze rocprof output for GPU kernel bottlenecks')
    parser.add_argument('--metrics', type=str, help='Path to metrics CSV file')
    parser.add_argument('--stats', type=str, help='Path to stats CSV file')
    parser.add_argument('--memory', type=str, help='Path to memory CSV file')
    parser.add_argument('--cache', type=str, help='Path to cache metrics CSV file')
    parser.add_argument('--lds', type=str, help='Path to LDS metrics CSV file')
    parser.add_argument('--dir', type=str, help='Directory containing profiling results')
    
    args = parser.parse_args()
    
    analyze_rocprof_metrics(
        stats_file=args.stats,
        metrics_file=args.metrics,
        memory_file=args.memory,
        cache_file=args.cache,
        lds_file=args.lds,
        output_dir=args.dir
    )

if __name__ == "__main__":
    main()
