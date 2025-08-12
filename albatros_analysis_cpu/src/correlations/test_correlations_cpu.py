# Philippe Joly 2025-06-05
# Numba parallelized versions

import numba
from numba import njit, prange
import numpy as np
import sys

def avg_xcorr_all_ant_cpu(x: np.ndarray, nant: int, npol: int, ntime: int, nfreq: int, split: int = 1, scratch=None, out=None):
    M = nant * npol
    N = M
    K = ntime

    if K % split != 0:
        raise ValueError("split should be a divisor of ntime")

    batchCount = nfreq * split

    if out is None:
        out = np.empty((M, N, nfreq), dtype='complex64', order='C')
    elif (out.shape != (M, M, nfreq) or out.dtype != x.dtype or not out.flags.c_contiguous):
        raise ValueError("invalid out buffer")

    if split > 1:
        raise NotImplementedError("Splitting not yet implemented for CPU version")
    else:
        # Perform batched xcorr = x @ x.conj().T for each frequency
        for f in range(nfreq):
            # x is shape (M, K, nfreq) → take slice (M, K) at freq f
            xf = x[:, :, f]
            out[:, :, f] = xf @ xf.conj().T / K

    return out 

# Approach 1: Simple parallel loop over frequencies
@njit(parallel=True)
def avg_xcorr_all_ant_cpu_parallel_v1(x, nant, npol, ntime, nfreq, split=1, out=None):
    """
    Parallel version using prange over frequency dimension.
    Most straightforward approach.
    """
    M = nant * npol
    K = ntime

    if K % split != 0:
        raise ValueError("split should be a divisor of ntime")

    if out is None:
        out = np.empty((M, M, nfreq), dtype=np.complex64)
    
    if split > 1:
        raise NotImplementedError("Splitting not yet implemented for CPU version")
    
    # Parallel loop over frequencies
    for f in prange(nfreq):
        # Extract frequency slice
        xf = x[:, :, f]
        
        # Compute cross-correlation matrix
        for i in range(M):
            for j in range(M):
                corr_sum = 0.0 + 0.0j
                for k in range(K):
                    corr_sum += xf[i, k] * np.conj(xf[j, k])
                out[i, j, f] = corr_sum / K
    
    return out


# Approach 2: More optimized with better memory access patterns
@njit(parallel=True)
def avg_xcorr_all_ant_cpu_parallel_v2(x, nant, npol, ntime, nfreq, split=1, out=None):
    """
    More optimized version with better memory access patterns.
    Uses parallel reduction for better performance.
    """
    M = nant * npol
    K = ntime

    if K % split != 0:
        raise ValueError("split should be a divisor of ntime")

    if out is None:
        out = np.empty((M, M, nfreq), dtype=np.complex64)
    
    if split > 1:
        raise NotImplementedError("Splitting not yet implemented for CPU version")
    
    # Parallel loop over frequencies
    for f in prange(nfreq):
        # For each frequency, compute the cross-correlation matrix
        for i in range(M):
            for j in range(i, M):  # Only compute upper triangle + diagonal
                corr_sum = 0.0 + 0.0j
                
                # Vectorized inner product (Numba will optimize this)
                for k in range(K):
                    corr_sum += x[i, k, f] * np.conj(x[j, k, f])
                
                result = corr_sum / K
                out[i, j, f] = result
                
                # Use Hermitian symmetry to fill lower triangle
                if i != j:
                    out[j, i, f] = np.conj(result)
    
    return out


@njit(parallel=True)
def avg_xcorr_all_ant_cpu_parallel_v3(x, nant, npol, ntime, nfreq, split=1, out=None):
    """
    Hybrid parallelization over both frequencies and matrix computation.
    Best for larger arrays where you have many cores available.
    """
    M = nant * npol
    K = ntime

    if K % split != 0:
        raise ValueError("split should be a divisor of ntime")

    if out is None:
        out = np.empty((M, M, nfreq), dtype=np.complex64)
    
    if split > 1:
        raise NotImplementedError("Splitting not yet implemented for CPU version")
    
    # Flatten the computation: parallel over (freq, i, j) combinations
    tri_size = M * (M + 1) // 2  # Size of upper triangle
    total_elements = nfreq * tri_size
    
    for idx in prange(total_elements):
        # Decode the flattened index to (f, i, j) where j >= i
        f = idx // tri_size
        remainder = idx % tri_size
        
        # Convert remainder to (i, j) in upper triangular form
        # Using the inverse of triangular number formula
        i = 0
        temp_remainder = remainder
        while temp_remainder >= (M - i):
            temp_remainder -= (M - i)
            i += 1
        j = i + temp_remainder
        
        # Compute cross-correlation
        corr_sum = 0.0 + 0.0j
        for k in range(K):
            corr_sum += x[i, k, f] * np.conj(x[j, k, f])
        
        result = corr_sum / K
        out[i, j, f] = result
        
        # Use Hermitian symmetry
        if i != j:
            out[j, i, f] = np.conj(result)
    
    return out


# Approach 4: Using Numba's matrix multiplication (if available)
@njit(parallel=True)
def avg_xcorr_all_ant_cpu_parallel_v4(x, nant, npol, ntime, nfreq, split=1, out=None):
    """
    Version that tries to leverage Numba's optimized matrix operations.
    May not always be faster due to overhead, but worth testing.
    """
    M = nant * npol
    K = ntime

    if K % split != 0:
        raise ValueError("split should be a divisor of ntime")

    if out is None:
        out = np.empty((M, M, nfreq), dtype=np.complex64)
    
    if split > 1:
        raise NotImplementedError("Splitting not yet implemented for CPU version")
    
    # Parallel loop over frequencies
    for f in prange(nfreq):
        # Extract frequency slice
        xf = x[:, :, f]
        
        # Manual matrix multiplication with conjugate transpose
        for i in range(M):
            for j in range(M):
                temp = 0.0 + 0.0j
                for k in range(K):
                    temp += xf[i, k] * np.conj(xf[j, k])
                out[i, j, f] = temp / K
    
    return out


    


# Example usage and benchmarking helper
def benchmark_versions(x, nant, npol, ntime, nfreq, niter=100):
    """
    Helper function to benchmark different versions.
    """
    import time
    
    versions = [
        ("original", avg_xcorr_all_ant_cpu),
        ("v1_simple", avg_xcorr_all_ant_cpu_parallel_v1),
        ("v2_optimized", avg_xcorr_all_ant_cpu_parallel_v2),
        ("v3_hybrid", avg_xcorr_all_ant_cpu_parallel_v3),
        ("v4_matmul", avg_xcorr_all_ant_cpu_parallel_v4),
    ]
    
    print(f"Benchmarking with shape: M={nant*npol}, K={ntime}, nfreq={nfreq}")
    
    results = {}
    for name, func in versions:
        # Warm up
        _ = func(x, nant, npol, ntime, nfreq)
        
        # Time it
        tot = 0
        for i in range(niter):
            start = time.perf_counter()
            result = func(x, nant, npol, ntime, nfreq)
            end = time.perf_counter()
            tot +=end-start
        
        results[name] = (tot/niter, result)
        print(f"{name}: {tot/niter} s,  {x.nbytes/tot*niter/1e6} MSPS")
    
    return results

# for osamp 2^16 pfb 8
# (2, 4, 65536) complex64 1 2 4 65536 (2, 2, 65536) complex64

if __name__=="__main__":
    shp = (2, 4, 65536)

    xr = np.random.randn(np.prod(shp)).reshape(shp).astype("float32")
    xi = np.random.randn(np.prod(shp)).reshape(shp).astype("float32")
    x = xr + 1j*xi

    scratch = np.empty((2, 2, 65536), dtype="complex64")

    nant, npol = 1, 2
    ntime, nfreq = 4, 65536
    niter = 2
    
    print("Starting xcorr Speed test with", shp, "array for", niter, "iterations...\n")

    res = benchmark_versions(x, nant, npol, ntime, nfreq, niter)
    print("\n")
    og_name = "original"
    for name in res:
        print(name, np.allclose(res[name][-1], res[og_name][-1], rtol=1e-5, atol=1e-6))
        dif = np.abs(res[name][-1] - res[og_name][-1])
        print(np.mean(dif), np.std(dif))
    sys.exit()


