import re
from collections import defaultdict

log_data = """
Max work-group size: 1024
Created queue on device: Intel(R) Data Center GPU Max 1550
Max compute units: 448
Max work-group size: 1024
Created queue on device: Intel(R) Data Center GPU Max 1550
Max compute units: 448
Max work-group size: 1024
Thread 168148, iteration 0, kernel1 started.
Thread 168160, iteration 0, kernel3 started.
Thread 168159, iteration 0, kernel2 started.
Thread 168161, iteration 0, kernel4 started.
Thread 168148, iteration 0, kernel1 executed in 139387 us. Kernel start: 1.71452e+08 us, end: 1.71591e+08
Thread 168160, iteration 0, kernel3 executed in 137552 us. Kernel start: 1.71707e+08 us, end: 1.71844e+08
Thread 168159, iteration 0, kernel2 executed in 136543 us. Kernel start: 1.71873e+08 us, end: 1.72009e+08
Thread 168161, iteration 0, kernel4 executed in 136374 us. Kernel start: 1.71998e+08 us, end: 1.72134e+08
Thread 168160, iteration 1, kernel3 started.
Thread 168148, iteration 1, kernel1 started.
Thread 168160, iteration 1, kernel3 executed in 137732 us. Kernel start: 1.72143e+08 us, end: 1.72281e+08
Thread 168159, iteration 1, kernel2 started.
Thread 168159, iteration 1, kernel2 executed in 137635 us. Kernel start: 1.72305e+08 us, end: 1.72443e+08
Thread 168148, iteration 1, kernel1 executed in 139876 us. Kernel start: 1.72447e+08 us, end: 1.72587e+08
Thread 168161, iteration 1, kernel4 started.
Thread 168161, iteration 1, kernel4 executed in 136052 us. Kernel start: 1.72573e+08 us, end: 1.72709e+08
Thread 168160, iteration 2, kernel3 started.
Thread 168160, iteration 2, kernel3 executed in 137470 us. Kernel start: 1.72713e+08 us, end: 1.7285e+08
Thread 168159, iteration 2, kernel2 started.
Thread 168159, iteration 2, kernel2 executed in 137470 us. Kernel start: 1.72876e+08 us, end: 1.73014e+08
Thread 168148, iteration 2, kernel1 started.
Thread 168161, iteration 2, kernel4 started.
Thread 168148, iteration 2, kernel1 executed in 140073 us. Kernel start: 1.73018e+08 us, end: 1.73158e+08
Thread 168160, iteration 3, kernel3 started.
Thread 168160, iteration 3, kernel3 executed in 137473 us. Kernel start: 1.73143e+08 us, end: 1.7328e+08
Thread 168161, iteration 2, kernel4 executed in 135701 us. Kernel start: 1.73285e+08 us, end: 1.73421e+08
Thread 168159, iteration 3, kernel2 started.
Thread 168159, iteration 3, kernel2 executed in 137411 us. Kernel start: 1.73446e+08 us, end: 1.73584e+08
Thread 168148, iteration 3, kernel1 started.
Thread 168148, iteration 3, kernel1 executed in 139820 us. Kernel start: 1.73588e+08 us, end: 1.73727e+08
Thread 168160, iteration 4, kernel3 started.
Thread 168160, iteration 4, kernel3 executed in 137526 us. Kernel start: 1.73712e+08 us, end: 1.7385e+08
Thread 168161, iteration 3, kernel4 started.
Thread 168161, iteration 3, kernel4 executed in 136369 us. Kernel start: 1.73856e+08 us, end: 1.73992e+08
Thread 168148, iteration 4, kernel1 started.
Thread 168159, iteration 4, kernel2 started.
Thread 168148, iteration 4, kernel1 executed in Thread 168160, iteration 5, kernel3 started.
140044 us. Kernel start: 1.74017e+08 us, end: 1.74157e+08
Thread 168160, iteration 5, kernel3 executed in 136603 us. Kernel start: 1.74142e+08 us, end: 1.74278e+08
Thread 168159, iteration 4, kernel2 executed in 136521 us. Kernel start: 1.74303e+08 us, end: 1.74439e+08
Thread 168161, iteration 4, kernel4 started.
Thread 168161, iteration 4, kernel4 executed in 135603 us. Kernel start: 1.74424e+08 us, end: 1.7456e+08
Thread 168160, iteration 6, kernel3 started.
Thread 168148, iteration 5, kernel1 started.
Thread 168160, iteration 6, kernel3 executed in 137191 us. Kernel start: 1.74567e+08 us, end: 1.74704e+08
Thread 168159, iteration 5, kernel2 started.
Thread 168159, iteration 5, kernel2 executed in 137471 us. Kernel start: 1.74728e+08 us, end: 1.74865e+08
Thread 168148, iteration 5, kernel1 executed in 139604 us. Kernel start: 1.74869e+08 us, end: 1.75009e+08
Thread 168161, iteration 5, kernel4 started.
Thread 168161, iteration 5, kernel4 executed in 136227 us. Kernel start: 1.74995e+08 us, end: 1.75131e+08
Thread 168159, iteration 6, kernel2 started.
Thread 168160, iteration 7, kernel3 started.
Thread 168159, iteration 6, kernel2 executed in 137592 us. Kernel start: 1.75157e+08 us, end: 1.75294e+08
Thread 168160, iteration 7, kernel3 executed in 137446 us. Kernel start: 1.75279e+08 us, end: 1.75416e+08
Thread 168161, iteration 6, kernel4 started.
Thread 168161, iteration 6, kernel4 executed in 136105 us. Kernel start: 1.75422e+08 us, end: 1.75558e+08
Thread 168148, iteration 6, kernel1 started.
Thread 168148, iteration 6, kernel1 executed in 139962 us. Kernel start: 1.75582e+08 us, end: 1.75722e+08
Thread 168159, iteration 7, kernel2 started.
Thread 168161, iteration 7, kernel4 started.
Thread 168160, iteration 8, kernel3 started.
Thread 168159, iteration 7, kernel2 executed in 137761 us. Kernel start: 1.75728e+08 us, end: 1.75866e+08
Thread 168161, iteration 7, kernel4 executed in 136969 us. Kernel start: 1.7585e+08 us, end: 1.75987e+08
Thread 168160, iteration 8, kernel3 executed in 136923 us. Kernel start: 1.75991e+08 us, end: 1.76128e+08
Thread 168148, iteration 7, kernel1 started.
Thread 168148, iteration 7, kernel1 executed in 140053 us. Kernel start: 1.76154e+08 us, end: 1.76294e+08
Thread 168160, iteration 9, kernel3 started.
Thread 168161, iteration 8, kernel4 started.
Thread 168159, iteration 8, kernel2 started.
Thread 168160, iteration 9, kernel3 executed in 136916 us. Kernel start: 1.7628e+08 us, end: 1.76417e+08
Thread 168148, iteration 8, kernel1 started.
Thread 168148, iteration 8, kernel1 executed in 139259 us. Kernel start: 1.76441e+08 us, end: 1.7658e+08
Thread 168159, iteration 8, kernel2 executed in 137291 us. Kernel start: 1.76585e+08 us, end: 1.76722e+08
Thread 168161, iteration 8, kernel4 executed in 135927 us. Kernel start: 1.76706e+08 us, end: 1.76842e+08
Thread 168161, iteration 9, kernel4 started.
Thread 168159, iteration 9, kernel2 started.
Thread 168148, iteration 9, kernel1 started.
Thread 168161, iteration 9, kernel4 executed in 136512 us. Kernel start: 1.7685e+08 us, end: 1.76987e+08
Thread 168159, iteration 9, kernel2 executed in 137192 us. Kernel start: 1.77011e+08 us, end: 1.77148e+08
Thread 168148, iteration 9, kernel1 executed in 139381 us. Kernel start: 1.77152e+08 us, end: 1.77291e+08
All threads have finished execution.
"""

from typing import List, Dict, Any

def parse_log(log_data: str) -> List[Dict[str, Any]]:
    # More flexible pattern to match various time formats
    pattern = r"Thread (\d+), iteration (\d+), (kernel\d+) executed in (?:approximately |about |~|roughly )?(\d+(?:\.\d+)?)\s*((?:milli)?seconds?|ms|(?:micro)?seconds?|µs|us)\. Kernel start: ([\d\.e\+]+)\s*(?:microseconds?|us|µs), end: ([\d\.e\+]+)"
    
    executions = []
    for match in re.finditer(pattern, log_data, re.IGNORECASE):
        thread_id, iteration, kernel_name, duration, time_unit, start, end = match.groups()
        
        # Convert duration to microseconds
        duration_us = float(duration)
        if any(unit in time_unit.lower() for unit in ['milli', 'ms']):
            duration_us *= 1000
        
        # Convert start and end times to float, handling potential scientific notation
        start_us = float(start.replace(',', ''))
        end_us = float(end.replace(',', ''))
        
        executions.append({
            "thread_id": int(thread_id),
            "iteration": int(iteration),
            "kernel_name": kernel_name,
            "duration": duration_us,
            "start": start_us,
            "end": end_us
        })
    
    return executions

def find_overlaps(executions):
    overlaps = []
    n = len(executions)
    for i in range(n):
        for j in range(i+1, n):
            if executions[i]["start"] < executions[j]["end"] and executions[i]["end"] > executions[j]["start"]:
                overlap_start = max(executions[i]["start"], executions[j]["start"])
                overlap_end = min(executions[i]["end"], executions[j]["end"])
                overlap_duration = overlap_end - overlap_start
                
                overlap_percent_i = (overlap_duration / (executions[i]["end"] - executions[i]["start"])) * 100
                overlap_percent_j = (overlap_duration / (executions[j]["end"] - executions[j]["start"])) * 100
                
                overlaps.append((executions[i], executions[j], overlap_duration, overlap_percent_i, overlap_percent_j))
    return overlaps

def analyze_kernel_performance(executions):
    kernel_stats = defaultdict(lambda: {"count": 0, "total_duration": 0, "min_duration": float('inf'), "max_duration": 0})
    
    for execution in executions:
        kernel_name = execution["kernel_name"]
        duration = execution["duration"]
        
        kernel_stats[kernel_name]["count"] += 1
        kernel_stats[kernel_name]["total_duration"] += duration
        kernel_stats[kernel_name]["min_duration"] = min(kernel_stats[kernel_name]["min_duration"], duration)
        kernel_stats[kernel_name]["max_duration"] = max(kernel_stats[kernel_name]["max_duration"], duration)
    
    for kernel_name, stats in kernel_stats.items():
        stats["avg_duration"] = stats["total_duration"] / stats["count"]
    
    return kernel_stats

def main(log_data):
    executions = parse_log(log_data)
    
    print("Kernel Execution Summary:")
    kernel_stats = analyze_kernel_performance(executions)
    for kernel_name, stats in kernel_stats.items():
        print(f"{kernel_name}:")
        print(f"  Count: {stats['count']}")
        print(f"  Average Duration: {stats['avg_duration']:.2f} us")
        print(f"  Min Duration: {stats['min_duration']} us")
        print(f"  Max Duration: {stats['max_duration']} us")
        print()
    
    overlapping_kernels = find_overlaps(executions)
    if overlapping_kernels:
        print("Overlapping kernels:")
        for overlap in overlapping_kernels:
            kernel1, kernel2, overlap_duration, overlap_percent1, overlap_percent2 = overlap
            print(f"{kernel1['kernel_name']} (Thread {kernel1['thread_id']}, Iteration {kernel1['iteration']}) "
                  f"overlaps with {kernel2['kernel_name']} (Thread {kernel2['thread_id']}, Iteration {kernel2['iteration']})")
            print(f"  Overlap duration: {overlap_duration:.2f} us")
            print(f"  Overlap percentage for {kernel1['kernel_name']}: {overlap_percent1:.2f}%")
            print(f"  Overlap percentage for {kernel2['kernel_name']}: {overlap_percent2:.2f}%")
            print()
    else:
        print("No overlapping kernels.")


main(log_data)