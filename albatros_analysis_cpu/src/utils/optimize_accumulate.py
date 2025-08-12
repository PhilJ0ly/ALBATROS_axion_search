import numpy as np
from numba import njit, prange
import time
import psutil
import sys, os

from os import path
# import json 
sys.path.insert(0,path.expanduser("~"))
from albatros_analysis.src.utils.accumulators import *

# import albatros_analysis.scripts.xcorr.helper as helper
# from albatros_analysis.src.correlations import baseband_data_classes as bdc
# from albatros_analysis.src.correlations import correlations as cr 
# from albatros_analysis.src.utils import pfb_cpu_utils as pu

def generate_test_data(lblock, ntap, sz, acclen):
      # """Generate test data for profiling"""
    print(f"Generating test data...")


    nblock  = sz//lblock - ntap +1
    ts= np.random.rand(sz).astype(np.float32).reshape(acclen, sz//acclen).copy()
    win = np.random.rand(ntap*lblock).astype(np.float32)

    # win2=win.reshape(ntap, 1, lblock).copy()
    
    actual_size = ts.nbytes / 1024 / 1024
    print(f"Generated ts: {ts.shape}, {actual_size:.1f}MB")
    print(f"Generated window: {win.shape}, {win.nbytes/1024/1024:.1f}MB")
    
    return ts, win




def benchmark_optimizations(ts, win, lblock, ntap, nblock, scratch_template, niter=10):
    """Benchmark all optimization variants"""
    
    functions = [
        ("Original", accumulate_lin_original),

        # ("Opt1: Restructured", accumulate_lin_opt1), 
        # ("Opt2: Vectorized", accumulate_lin_opt2),
        # ("Opt3: Block-wise", accumulate_lin_opt3),
        # ("Opt4: Reduced calc", accumulate_lin_opt4),
        # ("Opt5: Reordered", accumulate_lin_opt5),

        # ("opt6: hand-drawn", accumulate_lin_opt6),
        ("opt7: hand-drawn", accumulate_lin_opt7),
        # ("opt8: hand-drawn", accumulate_lin_opt8),
        # ("opt9: hand-drawn", accumulate_lin_opt9),
        # ("opt10: hand-drawn", accumulate_lin_opt10),

        ("v1: hand-drawn opt", accumulate_lin_opt_v1),
        ("v2: hand-drawn opt", accumulate_lin_opt_v2),
        ("v3: hand-drawn opt", accumulate_lin_opt_v3),
        ("v4: hand-drawn opt", accumulate_lin_opt_v4),
        ("v5: hand-drawn opt", accumulate_lin_opt_v5),
    ]
    
    results = {}

    ts_6 = ts.ravel()
    win_6 = win.ravel()
    rho = lblock//ts.shape[0]

    ts_og = ts.T 
    win_og = win.reshape(ntap, lblock).copy()

    not_flats = ["Original"]
    
    for name, func in functions:
        print(f"Testing {name}...")
        times = []
        
        # Warmup
        scratch = scratch_template.copy()
        if name in not_flats: 
            func(ts_og, win_og, lblock, ntap, nblock, scratch)
        else:
            func(ts_6, win_6, lblock, ntap, nblock, rho, scratch) 
        
        # Benchmark
        for _ in range(niter):
            scratch = scratch_template.copy()
            start = time.perf_counter()
            if name in not_flats: 
                func(ts_og, win_og, lblock, ntap, nblock, scratch)
            else:
                func(ts_6, win_6, lblock, ntap, nblock, rho, scratch) 
            end = time.perf_counter()
            times.append(end - start)
        
        results[name] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'speedup': times[0] / np.mean(times) if name != "Original" else 1.0
        }
        
        print(f"  Mean: {results[name]['mean']:.4f}s ± {results[name]['std']:.4f}s")
        if name != "Original":
            baseline = results["Original"]['mean']
            speedup = baseline / results[name]['mean']
            print(f"  Speedup: {speedup:.2f}x")
    
    return results

# Additional diagnostic function
@njit
def analyze_memory_access_pattern(ts, lblock, ntap, nblock, sample_size=1000):
    """Analyze the memory access pattern to understand cache behavior"""
    ts_cols = ts.shape[1]
    
    # Sample some index calculations to see the pattern
    indices = np.zeros((sample_size, 2), dtype=np.int64)
    
    count = 0
    for i in range(min(10, nblock)):
        for j in range(ntap):
            for k in range(min(100, lblock)):
                if count >= sample_size:
                    break
                    
                idx_num = i*lblock + j * lblock + k
                p = idx_num // ts_cols
                q = idx_num % ts_cols
                indices[count, 0] = p
                indices[count, 1] = q
                count += 1
            if count >= sample_size:
                break
        if count >= sample_size:
            break
    
    return indices[:count]

def print_optimization_recommendations():
    """Print specific recommendations based on the analysis"""
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. MEMORY ACCESS OPTIMIZATION:")
    print("   - Current pattern: Random access to ts[p,q] causes cache misses")
    print("   - Try reshaping data to improve spatial locality")
    print("   - Consider transposing ts if access pattern allows")
    
    print("\n2. COMPUTATIONAL OPTIMIZATION:")
    print("   - Pre-calculate division/modulo operations outside loops")
    print("   - Use bit shifts instead of division where possible")
    print("   - Consider lookup tables for repeated calculations")
    
    print("\n3. PARALLELIZATION OPTIMIZATION:")
    print("   - Current: 80 cores may be causing memory bandwidth saturation")
    print("   - Try reducing thread count to 16-32 cores")
    print("   - Use thread affinity to group threads by memory controller")
    
    print("\n4. ALGORITHMIC OPTIMIZATION:")
    print("   - Consider if the algorithm can be restructured")
    print("   - Look for opportunities to use BLAS/LAPACK operations")
    print("   - Evaluate if FFT-based convolution would be faster")
    
    print("\n5. SYSTEM-LEVEL OPTIMIZATION:")
    print("   - Monitor memory bandwidth utilization")
    print("   - Check NUMA topology and thread placement")
    print("   - Consider using fewer threads to reduce contention")

# Example usage function
def run_optimization_analysis(timestream, win, nchan=2049, ntap=4, niter=100):
    """Run the optimization analysis"""
    
    lblock = 2 * (nchan - 1)
    nblock = timestream.size // lblock - (ntap - 1)
    scratch_template = np.zeros((nblock, lblock), dtype=np.float32)
    
    print("Analyzing memory access pattern...")
    indices = analyze_memory_access_pattern(timestream, lblock, ntap, nblock)
    
    print(f"Sample access indices (first 10):")
    for i in range(min(10, len(indices))):
        print(f"  ts[{indices[i,0]}, {indices[i,1]}]")
    
    print(f"\nAccess pattern statistics:")
    print(f"  Row range: {indices[:,0].min()} - {indices[:,0].max()}")
    print(f"  Col range: {indices[:,1].min()} - {indices[:,1].max()}")
    print(f"  Row span: {indices[:,0].max() - indices[:,0].min() + 1}")
    print(f"  Unique rows accessed: {len(np.unique(indices[:,0]))}")
    
    print("\nBenchmarking optimizations...")
    results = benchmark_optimizations(
        timestream, win, lblock, ntap, nblock, scratch_template, niter=niter
    )
    
    # print_optimization_recommendations()
    
    return results

def main():
    """Main function to run comprehensive performance tests"""
    print("Comprehensive CPU Performance Monitor")
    print("="*50)
    print(f"System Info:")
    print(f"  CPU Cores: {psutil.cpu_count()}")
    print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"  Python Version: {sys.version}")
    
    # Get user input for number of iterations
    try:
        niter_input = input(f"\nNumber of iterations (default 1000): ").strip()
        niter = int(niter_input) if niter_input else 1000
    except ValueError:
        niter = 1000
    
    print(f"Running {niter} iterations per function...")

    # try:
    #     sz = input(f"\nData Size in (Mb): ").strip()
    #     sz = int(sz) if sz else 100
    # except ValueError:
    #     sz = 100
    
    # Generate test data
    print("\n" + "="*50)
    lblock, ntap, sz, acclen = 262144, 4, 14336*4096, 14336
    timestream, win = generate_test_data(lblock, ntap, sz, acclen)
    print(timestream.shape, win.shape)
    
    # Run comprehensive profiling
    print("\n" + "="*50)
    print("PFB accumulate blocks optimization analysis")
    print("="*50)

    run_optimization_analysis(timestream, win, nchan=2048*64+1, niter=niter)

if __name__=="__main__":
    main()
