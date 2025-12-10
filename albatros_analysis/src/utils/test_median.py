# Philippe Joly 2025-12-09

# This is a parallel median functon

import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from os import path
from numba import njit, prange

sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.utils.pfb_utils import tiled_transpose

def compute_median_batch(args):
    """Compute median of real and imaginary parts for a batch of rows"""

    batch_start, batch_end, batch = args
    dtype = batch.dtype

    if np.issubdtype(dtype, np.complexfloating) == False:
        result = np.median(batch, axis=1).astype(dtype)
    else:
        real_medians = np.median(batch.real, axis=1)
        imag_medians = np.median(batch.imag, axis=1)
        result = (real_medians+1j*imag_medians).astype(dtype)
    
    return batch_start, batch_end, result


def batched_median_from_disk(arr, batch_size: int, n_workers=os.cpu_count(), out=None):
    
    nfreq = arr.shape[-1]

    arr = tiled_transpose(arr)

    batch_args = []
    t_batch = time.time()
    for batch_start in range(0, nfreq, batch_size):
        batch_end = min(batch_start + batch_size, nfreq)
        batch_args.append((batch_start, batch_end, arr[batch_start:batch_end]))
    t_batch = time.time() - t_batch
    # print(f"Prepared {len(batch_args)} batches in {t_batch:.3f} s")
    
    # Result array
    if out is None:
        out = np.empty(nfreq, dtype=arr.dtype)
    
    # print(f"Computing median in batches of {batch_size} across {n_workers} workers...")
    t_med = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(compute_median_batch, batch_args)
        for batch_start, batch_end, batch_medians in results:
            out[batch_start:batch_end] = batch_medians
    t_med = time.time() - t_med
    # print(f"Computed median across {nfreq} frequencies in {t_med:.2f} s - {arr.nbytes/t_med/1e9} GB/s")
    
    return out

def simple_median(arr):
    """Compute median of real and imaginary parts for entire array"""
    dtype = arr.dtype

    if np.issubdtype(dtype, np.complexfloating) == False:
        result = np.median(arr, axis=0).astype(dtype)
    else:
        real_medians = np.median(arr.real, axis=0)
        imag_medians = np.median(arr.imag, axis=0)
        result = (real_medians+1j*imag_medians).astype(dtype)
    
    return result


@njit(parallel=True, fastmath=True)
def numba_median(arr, is_complex, out=None):
    """
    Computes the median using Numba for acceleration/parallelization.
    Median calculated separately for real and imaginary parts.
    Median calculated along axis 1 (time).
    Median calculation is done manually for better performance.
    Designed for nfreqs >> ntimes
    
    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (nfreqs, ntimes) time-contiguous.
    is_complex : bool
        Whether the input array is complex-valued.
    out : np.ndarray, optional
        Pre-allocated output array of shape (nfreqs,). If None, a new array is created.
        
    Returns
    -------
    medians : np.ndarray
        Array of shape (nfreqs,)
    """

    n_freqs, n_times = arr.shape
    dtype = arr.dtype
    # out_type_constructor = dtype.type

    
    if out is None:
        out = np.empty(n_freqs, dtype=dtype)
    else:
        assert out.shape == (n_freqs,) and out.dtype == dtype, f"Output array has incorrect shape {out.shape} or dtype {out.dtype}, expected {(n_freqs,)} {dtype}"
    
    mid_idx = n_times // 2
    is_even = (n_times % 2 == 0)
    
    if is_complex:
        for i in prange(n_freqs):
            # Extract row (this is a view)
            row = arr[i, :]
            
            # Copy to a temporary array so we can sort without modifying input
            # n_times is small, this allocation is cheap
            r_temp = row.real.copy()

            # O(nlogn) sort cheap for small n_times
            r_temp.sort()
            
            if is_even:
                r_med = (r_temp[mid_idx - 1] + r_temp[mid_idx]) * 0.5
            else:
                r_med = r_temp[mid_idx]

            i_temp = row.imag.copy()
            i_temp.sort()
            
            if is_even:
                i_med = (i_temp[mid_idx - 1] + i_temp[mid_idx]) * 0.5
            else:
                i_med = i_temp[mid_idx]
                
            # Combine into output
            out[i] = r_med + 1j * i_med
    else:
        for i in prange(n_freqs):
            row = arr[i, :]
            r_temp = row.copy()
            r_temp.sort()
            
            if is_even:
                out[i] =(r_temp[mid_idx - 1] + r_temp[mid_idx]) * 0.5
            else:
                out[i] = r_temp[mid_idx]
    
    return out.astype(dtype)


def test_median_speed(niter=1):
    ntimes = 400
    nfreqs = 10_000*ntimes
    x_re = np.ones((ntimes,nfreqs), dtype=np.float32)
    x = (x_re + 1j*x_re).astype(np.complex64)
    print(os.cpu_count(), "CPU cores detected")

    batch_sizes = [500, 1000, 2000, 3000]

    for batch_size in batch_sizes:
        medians = batched_median_from_disk(x, batch_size)
        t_start = time.time()
        for i in range(niter):
            medians = batched_median_from_disk(x, batch_size)
        t_end = time.time()
        print(f"Computed batched ({batch_size}) median of array shape {x.shape} in {(t_end - t_start)/niter:.2f} s - {x.nbytes/((t_end - t_start)/niter)/1e9} GB/s")


    t_start = time.time()
    medians = simple_median(x)
    for i in range(niter):
        medians = simple_median(x)
    t_end = time.time()
    print(f"Computed np median of array shape {x.shape} in {(t_end - t_start)/niter:.2f} s - {x.nbytes/((t_end - t_start)/niter)/1e9} GB/s")

    t_start = time.time()
    medians = numba_median(x)
    for i in range(niter):
        medians = numba_median(x)
    t_end = time.time()
    print(f"Computed numba median of array shape {x.shape} in {(t_end - t_start)/niter:.2f} s - {x.nbytes/((t_end - t_start)/niter)/1e9} GB/s")


if __name__ == "__main__":
    test_median_speed()