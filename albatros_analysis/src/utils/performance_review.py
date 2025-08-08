#!/usr/bin/env python3
"""
Performance monitoring script for accumulate_lin and cpu_pfb functions.
Tests execution speed and CPU core usage with extensive statistics over many iterations.
"""

import numpy as np
import time
import psutil
import threading
import multiprocessing as mp
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads
from scipy.fft import rfft, set_workers
import contextlib
import sys
from statistics import stdev, mean
import warnings


# Suppress numba warnings for cleaner output
warnings.filterwarnings('ignore')

# Assuming these are the functions to test (you'll need to import or define them)
@njit(parallel=True)
def accumulate_lin(ts, win, lblock, ntap, nblock, scratch):
    # Clean scratch array
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    # i n, j ntap, k l
    for k in prange(lblock):
        for i in range(nblock):
            for j in range(ntap):
                idx_num = i*lblock + j * lblock + k
                p,q = idx_num//ts.shape[1], idx_num%ts.shape[1]
                scratch[i,k] += win[j, k]*ts[p, q]

# @contextlib.contextmanager
# def set_workers(n_workers):
#     """Context manager to set number of workers (placeholder for your implementation)"""
#     old_threads = set_num_threads(n_workers)
#     try:
#         yield
#     finally:
#         set_num_threads(old_threads)

def cpu_pfb(timestream, win, out=None, scratch=None, nchan=2049, ntap=4, n_workers=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    lblock = 2 * (nchan - 1)
    nblock = timestream.size // lblock - (ntap - 1)

    if scratch is None:
        scratch = np.zeros((nblock, lblock), dtype=np.float32)

    accumulate_lin(timestream, win, lblock, ntap, nblock, scratch)

    with set_workers(n_workers):
        out = rfft(scratch, axis=1)

    return out

class CPUMonitor:
    """Monitor CPU usage per core in real-time"""
    
    def __init__(self, interval=0.001):
        self.interval = interval
        self.monitoring = False
        self.cpu_data = defaultdict(list)
        self.timestamps = []
        self.thread = None
        
    def start(self):
        """Start monitoring CPU usage"""
        self.monitoring = True
        self.cpu_data.clear()
        self.timestamps.clear()
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.01)  # Give monitor time to start
        
    def stop(self):
        """Stop monitoring and return collected data"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
        return dict(self.cpu_data), self.timestamps
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        start_time = time.time()
        # Initial CPU call to initialize
        psutil.cpu_percent(interval=None, percpu=True)
        
        while self.monitoring:
            current_time = time.time() - start_time
            self.timestamps.append(current_time)
            
            # Get per-core CPU usage
            cpu_percents = psutil.cpu_percent(interval=None, percpu=True)
            for i, usage in enumerate(cpu_percents):
                self.cpu_data[f'core_{i}'].append(usage)
            
            # Overall CPU usage
            self.cpu_data['overall'].append(psutil.cpu_percent(interval=None))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.cpu_data['memory_percent'].append(memory.percent)
            self.cpu_data['memory_mb'].append(memory.used / 1024 / 1024)
            
            time.sleep(self.interval)

class ParallelizationAnalyzer:
    """Analyze parallelization efficiency"""
    
    @staticmethod
    def calculate_load_balance(core_usage_data):
        """Calculate how evenly work is distributed across cores"""
        if not core_usage_data:
            return 0.0, 0.0, 0
        
        num_cores = psutil.cpu_count()
        core_averages = []
        
        for i in range(num_cores):
            core_key = f'core_{i}'
            if core_key in core_usage_data and core_usage_data[core_key]:
                avg_usage = np.mean(core_usage_data[core_key])
                core_averages.append(avg_usage)
        
        if not core_averages:
            return 0.0, 0.0, 0
        
        # Calculate load balance metrics
        mean_usage = np.mean(core_averages)
        std_usage = np.std(core_averages)
        active_cores = sum(1 for usage in core_averages if usage > 5.0)  # Cores with >5% usage
        
        # Load balance score (lower std relative to mean is better)
        balance_score = 1.0 - (std_usage / (mean_usage + 1e-6))  # Add small epsilon to avoid division by zero
        balance_score = max(0.0, min(1.0, balance_score))  # Clamp to [0,1]
        
        return balance_score, mean_usage, active_cores
    
    @staticmethod
    def calculate_parallel_efficiency(execution_times, theoretical_cores):
        """Calculate parallel efficiency based on timing consistency"""
        if len(execution_times) < 2:
            return 0.0
        
        # Lower coefficient of variation indicates better parallel efficiency
        mean_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        cv = std_time / mean_time if mean_time > 0 else 1.0
        
        # Efficiency score (lower CV is better)
        efficiency = max(0.0, 1.0 - cv * 5)  # Scale CV by 5 for reasonable scoring
        return min(1.0, efficiency)

class PerformanceProfiler:
    """Profile function performance with detailed timing and resource usage over many iterations"""
    
    def __init__(self):
        self.results = {}
        self.analyzer = ParallelizationAnalyzer()
        
    def profile_function_multiple(self, func, args, kwargs, niter=1000, warmup=10):
        """Profile a function over multiple iterations with statistics"""
        print(f"  Warming up ({warmup} iterations)...")
        
        # Warmup runs
        for _ in range(warmup):
            func(*args, **kwargs)
        
        print(f"  Running {niter} iterations...")
        
        # Data collection
        execution_times = []
        memory_usage = []
        cpu_monitoring_data = []
        
        # Progress tracking
        progress_interval = max(1, niter // 20)  # Show progress 20 times
        
        for i in range(niter):
            if i % progress_interval == 0:
                print(f"    Progress: {i}/{niter} ({100*i/niter:.1f}%)")
            
            # Pre-execution measurements
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Start CPU monitoring for this iteration
            monitor = CPUMonitor(interval=0.0005)  # Very high frequency
            monitor.start()
            
            # Execute function
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            # Stop monitoring
            cpu_data, timestamps = monitor.stop()
            
            # Post-execution measurements
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Store results
            execution_times.append(end_time - start_time)
            memory_usage.append(final_memory - initial_memory)
            cpu_monitoring_data.append(cpu_data)
        
        print(f"    Completed {niter} iterations!")
        
        # Calculate statistics
        stats = self._calculate_statistics(execution_times, memory_usage, cpu_monitoring_data)
        stats['result_sample'] = result  # Keep one result sample
        
        return stats
    
    def _calculate_statistics(self, execution_times, memory_usage, cpu_data_list):
        """Calculate comprehensive statistics from collected data"""
        # Basic timing statistics
        timing_stats = {
            'mean': np.mean(execution_times),
            'std': np.std(execution_times),
            'min': np.min(execution_times),
            'max': np.max(execution_times),
            'median': np.median(execution_times),
            'cv': np.std(execution_times) / np.mean(execution_times) if np.mean(execution_times) > 0 else 0
        }
        
        # Memory statistics
        memory_stats = {
            'mean': np.mean(memory_usage),
            'std': np.std(memory_usage),
            'min': np.min(memory_usage),
            'max': np.max(memory_usage)
        }
        
        # CPU and parallelization analysis
        cpu_stats = self._analyze_cpu_usage(cpu_data_list)
        parallel_stats = self._analyze_parallelization(execution_times, cpu_data_list)
        
        return {
            'timing': timing_stats,
            'memory': memory_stats,
            'cpu': cpu_stats,
            'parallelization': parallel_stats,
            'raw_times': execution_times,
            'raw_memory': memory_usage
        }
    
    def _analyze_cpu_usage(self, cpu_data_list):
        """Analyze CPU usage across all iterations"""
        if not cpu_data_list:
            return {}
        
        num_cores = psutil.cpu_count()
        
        # Aggregate CPU data across all iterations
        all_core_usage = defaultdict(list)
        all_overall_usage = []
        
        for cpu_data in cpu_data_list:
            if not cpu_data:
                continue
                
            # Collect per-core data
            for i in range(num_cores):
                core_key = f'core_{i}'
                if core_key in cpu_data:
                    all_core_usage[core_key].extend(cpu_data[core_key])
            
            # Collect overall data
            if 'overall' in cpu_data:
                all_overall_usage.extend(cpu_data['overall'])
        
        # Calculate per-core statistics
        core_stats = {}
        for i in range(num_cores):
            core_key = f'core_{i}'
            if core_key in all_core_usage and all_core_usage[core_key]:
                core_data = all_core_usage[core_key]
                core_stats[f'core_{i}'] = {
                    'mean': np.mean(core_data),
                    'std': np.std(core_data),
                    'max': np.max(core_data)
                }
        
        # Overall CPU statistics
        overall_stats = {}
        if all_overall_usage:
            overall_stats = {
                'mean': np.mean(all_overall_usage),
                'std': np.std(all_overall_usage),
                'max': np.max(all_overall_usage)
            }
        
        return {
            'per_core': core_stats,
            'overall': overall_stats
        }
    
    def _analyze_parallelization(self, execution_times, cpu_data_list):
        """Analyze parallelization efficiency"""
        if not cpu_data_list:
            return {}
        
        num_cores = psutil.cpu_count()
        
        # Calculate load balance for each iteration
        balance_scores = []
        mean_usages = []
        active_cores_counts = []
        
        for cpu_data in cpu_data_list:
            if cpu_data:
                balance_score, mean_usage, active_cores = self.analyzer.calculate_load_balance(cpu_data)
                balance_scores.append(balance_score)
                mean_usages.append(mean_usage)
                active_cores_counts.append(active_cores)
        
        # Calculate parallel efficiency
        parallel_efficiency = self.analyzer.calculate_parallel_efficiency(execution_times, num_cores)
        
        # Statistics on parallelization metrics
        parallelization_stats = {}
        
        if balance_scores:
            parallelization_stats['load_balance'] = {
                'mean': np.mean(balance_scores),
                'std': np.std(balance_scores),
                'min': np.min(balance_scores),
                'max': np.max(balance_scores)
            }
        
        if active_cores_counts:
            parallelization_stats['active_cores'] = {
                'mean': np.mean(active_cores_counts),
                'std': np.std(active_cores_counts),
                'min': np.min(active_cores_counts),
                'max': np.max(active_cores_counts),
                'total_cores': num_cores
            }
        
        parallelization_stats['efficiency'] = parallel_efficiency
        
        return parallelization_stats
    
    def profile_cpu_pfb_components(self, timestream, win, nchan=2049, ntap=4, n_workers=None, niter=1000):
        """Profile cpu_pfb function components with extensive statistics"""
        if n_workers is None:
            n_workers = mp.cpu_count()
            
        lblock = 2 * (nchan - 1)
        nblock = timestream.size // lblock - (ntap - 1)
        
        results = {}
        
        # Profile accumulate_lin
        print("Profiling accumulate_lin...")
        scratch = np.zeros((nblock, lblock), dtype=np.float32)
        results['accumulate_lin'] = self.profile_function_multiple(
            accumulate_lin, 
            (timestream, win, lblock, ntap, nblock, scratch),
            {},
            niter=niter
        )
        
        # Profile rfft
        print("\nProfiling rfft...")
        def rfft_wrapper():
            with set_workers(n_workers):
                return rfft(scratch, axis=1)
        
        results['rfft'] = self.profile_function_multiple(
            rfft_wrapper, (), {}, niter=niter
        )
        
        # Profile complete function
        print("\nProfiling complete cpu_pfb...")
        results['complete'] = self.profile_function_multiple(
            cpu_pfb,
            (timestream, win),
            {'nchan': nchan, 'ntap': ntap, 'n_workers': n_workers},
            niter=niter
        )
        
        return results

def generate_test_data(size_mb=50, nchan=2049, ntap=4):
    """Generate test data for profiling"""
    print(f"Generating {size_mb}MB of test data...")
    
    # Calculate data size
    lblock = 2 * (nchan - 1)
    elements_needed = int(size_mb * 1024 * 1024 / 4)  # 4 bytes per float32
    
    # Create timestream data
    rows = elements_needed // lblock + ntap
    cols = lblock
    timestream = np.random.randn(rows, cols).astype(np.float32)
    
    # Create window function
    win = np.random.randn(ntap, lblock).astype(np.float32)
    
    actual_size = timestream.nbytes / 1024 / 1024
    print(f"Generated timestream: {timestream.shape}, {actual_size:.1f}MB")
    print(f"Generated window: {win.shape}, {win.nbytes/1024/1024:.1f}MB")
    
    return timestream, win

def print_detailed_performance_summary(results):
    """Print comprehensive performance summary with statistics"""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE SUMMARY")
    print("="*80)
    
    for component, data in results.items():
        print(f"\n{component.upper()}:")
        print("-" * 40)
        
        # Timing statistics
        timing = data['timing']
        print(f"  Execution Time:")
        print(f"    Mean:     {timing['mean']:.6f} ± {timing['std']:.6f} seconds")
        print(f"    Median:   {timing['median']:.6f} seconds")
        print(f"    Range:    {timing['min']:.6f} - {timing['max']:.6f} seconds")
        print(f"    CV:       {timing['cv']:.4f} ({timing['cv']*100:.2f}%)")
        
        # Memory statistics
        memory = data['memory']
        print(f"  Memory Usage:")
        print(f"    Mean:     {memory['mean']:.2f} ± {memory['std']:.2f} MB")
        print(f"    Range:    {memory['min']:.2f} - {memory['max']:.2f} MB")
        
        # CPU statistics
        if 'cpu' in data and data['cpu']:
            cpu = data['cpu']
            if 'overall' in cpu and cpu['overall']:
                overall = cpu['overall']
                print(f"  CPU Usage (Overall):")
                print(f"    Mean:     {overall['mean']:.1f} ± {overall['std']:.1f}%")
                print(f"    Peak:     {overall['max']:.1f}%")
        
        # Parallelization analysis
        if 'parallelization' in data and data['parallelization']:
            parallel = data['parallelization']
            print(f"  Parallelization:")
            
            if 'load_balance' in parallel:
                lb = parallel['load_balance']
                print(f"    Load Balance: {lb['mean']:.3f} ± {lb['std']:.3f} (0=poor, 1=perfect)")
            
            if 'active_cores' in parallel:
                ac = parallel['active_cores']
                print(f"    Active Cores: {ac['mean']:.1f} ± {ac['std']:.1f} / {ac['total_cores']}")
                efficiency_pct = (ac['mean'] / ac['total_cores']) * 100
                print(f"    Core Usage:   {efficiency_pct:.1f}%")
            
            if 'efficiency' in parallel:
                print(f"    Efficiency:   {parallel['efficiency']:.3f} (0=poor, 1=excellent)")

def plot_performance_distributions(results, name):
    """Plot distributions of execution times and parallelization metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Distributions', fontsize=16)
    
    components = list(results.keys())
    colors = ['blue', 'red', 'green', 'orange']
    
    # Execution time distributions
    ax1 = axes[0, 0]
    for i, (component, data) in enumerate(results.items()):
        times = data['raw_times']
        ax1.hist(times, bins=50, alpha=0.7, label=component, color=colors[i % len(colors)])
    ax1.set_xlabel('Execution Time (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Execution Time Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Load balance comparison
    ax2 = axes[0, 1]
    load_balances = []
    labels = []
    for component, data in results.items():
        if 'parallelization' in data and 'load_balance' in data['parallelization']:
            lb_data = [data['parallelization']['load_balance']['mean']]
            load_balances.extend(lb_data)
            labels.append(component)
    
    if load_balances:
        bars = ax2.bar(labels, load_balances, color=colors[:len(labels)])
        ax2.set_ylabel('Load Balance Score')
        ax2.set_title('Load Balance Comparison')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, load_balances):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Active cores comparison
    ax3 = axes[1, 0]
    active_cores = []
    total_cores = psutil.cpu_count()
    for component, data in results.items():
        if 'parallelization' in data and 'active_cores' in data['parallelization']:
            ac_data = data['parallelization']['active_cores']['mean']
            active_cores.append(ac_data)
        else:
            active_cores.append(0)
    
    bars = ax3.bar(labels, active_cores, color=colors[:len(labels)])
    ax3.axhline(y=total_cores, color='red', linestyle='--', label=f'Total Cores ({total_cores})')
    ax3.set_ylabel('Active Cores')
    ax3.set_title('Core Utilization')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, active_cores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}', ha='center', va='bottom')
    
    # Coefficient of variation comparison
    ax4 = axes[1, 1]
    cvs = []
    for component, data in results.items():
        cvs.append(data['timing']['cv'])
    
    bars = ax4.bar(labels, cvs, color=colors[:len(labels)])
    ax4.set_ylabel('Coefficient of Variation')
    ax4.set_title('Timing Consistency (Lower is Better)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, cvs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(name)

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

    try:
        sz = input(f"\nData Size in (Mb): ").strip()
        sz = int(sz) if sz else 100
    except ValueError:
        sz = 100
    
    # Generate test data
    print("\n" + "="*50)
    timestream, win = generate_test_data(size_mb=sz)
    
    # Run comprehensive profiling
    print("\n" + "="*50)
    print("STARTING COMPREHENSIVE PROFILING")
    print("="*50)
    
    profiler = PerformanceProfiler()
    results = profiler.profile_cpu_pfb_components(timestream, win, niter=niter)
    
    # Print detailed results
    print_detailed_performance_summary(results)
    
    # Create visualizations
    print("\nGenerating performance plots...")
    plot_performance_distributions(results, "/scratch/s/sievers/philj0ly/performance.png")
    
    # Summary insights
    print("\n" + "="*80)
    print("PARALLELIZATION INSIGHTS")
    print("="*80)
    
    for component, data in results.items():
        if 'parallelization' in data:
            parallel = data['parallelization']
            print(f"\n{component.upper()}:")
            
            if 'load_balance' in parallel and 'active_cores' in parallel:
                lb_score = parallel['load_balance']['mean']
                active_cores = parallel['active_cores']['mean']
                total_cores = parallel['active_cores']['total_cores']
                efficiency = parallel.get('efficiency', 0)
                
                # Parallelization assessment
                if lb_score > 0.8 and active_cores > total_cores * 0.7:
                    assessment = "EXCELLENT - Well parallelized"
                elif lb_score > 0.6 and active_cores > total_cores * 0.5:
                    assessment = "GOOD - Reasonably parallelized"
                elif active_cores > total_cores * 0.3:
                    assessment = "MODERATE - Some parallelization"
                else:
                    assessment = "POOR - Limited parallelization"
                
                print(f"  Assessment: {assessment}")
                print(f"  Recommendations:")
                
                if lb_score < 0.6:
                    print(f"    - Load imbalance detected (score: {lb_score:.3f})")
                    print(f"    - Consider work redistribution or different chunking")
                
                if active_cores < total_cores * 0.7:
                    print(f"    - Using only {active_cores:.1f}/{total_cores} cores")
                    print(f"    - May benefit from increased parallelism")
                
                if efficiency < 0.7:
                    print(f"    - Timing inconsistency detected (efficiency: {efficiency:.3f})")
                    print(f"    - May have synchronization overhead")

if __name__ == "__main__":
    main()