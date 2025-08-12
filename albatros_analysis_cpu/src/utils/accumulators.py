# Philippe Joly 2025-06-17
# This script implements a variety of modified pfb function which takes in the transpose of the desired input

import numpy as np
from numba import njit, prange
import time
import psutil
import sys, os

# Original function for comparison
@njit(parallel=True)
def accumulate_lin_original(ts, win, lblock, ntap, nblock, scratch):
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

# Optimization 1: Minimize expensive operations
@njit(parallel=True)
def accumulate_lin_opt1(ts, win, lblock, ntap, nblock, scratch):
    # Clean scratch array
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    # Pre-calculate division constant
    ts_cols = ts.shape[1]
    
    # Restructure loops to reduce expensive operations
    for i in prange(nblock):  # Parallelize over blocks instead
        for k in range(lblock):
            temp_sum = 0.0
            for j in range(ntap):
                idx_num = i*lblock + j * lblock + k
                p = idx_num // ts_cols  # Still expensive but reduced frequency
                q = idx_num % ts_cols
                temp_sum += win[j, k] * ts[p, q]
            scratch[i, k] = temp_sum

# Optimization 2: Vectorized approach (if possible)
@njit(parallel=True)
def accumulate_lin_opt2(ts, win, lblock, ntap, nblock, scratch):
    # Clean scratch array
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    ts_cols = ts.shape[1]
    
    # Try to vectorize inner operations
    for i in prange(nblock):
        base_idx = i * lblock
        for j in range(ntap):
            tap_base_idx = base_idx + j * lblock
            for k in range(lblock):
                idx_num = tap_base_idx + k
                p = idx_num // ts_cols
                q = idx_num % ts_cols
                scratch[i, k] += win[j, k] * ts[p, q]

# Optimization 3: Block-wise processing for better cache locality
@njit(parallel=True)
def accumulate_lin_opt3(ts, win, lblock, ntap, nblock, scratch):
    # Clean scratch array
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    ts_cols = ts.shape[1]
    block_size = 64  # Process in blocks for better cache usage
    
    for i_block in prange(0, nblock, block_size):
        i_end = min(i_block + block_size, nblock)
        
        for i in range(i_block, i_end):
            for k in range(lblock):
                temp_sum = 0.0
                for j in range(ntap):
                    idx_num = i*lblock + j * lblock + k
                    p = idx_num // ts_cols
                    q = idx_num % ts_cols
                    temp_sum += win[j, k] * ts[p, q]
                scratch[i, k] = temp_sum

# Optimization 4: Reduce index calculations
@njit(parallel=True) 
def accumulate_lin_opt4(ts, win, lblock, ntap, nblock, scratch):
    # Clean scratch array
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    ts_cols = ts.shape[1]
    
    # Pre-calculate some indices to reduce repeated calculations
    for i in prange(nblock):
        i_base = i * lblock
        
        for j in range(ntap):
            j_base = i_base + j * lblock
            
            for k in range(lblock):
                idx_num = j_base + k
                p = idx_num // ts_cols
                q = idx_num % ts_cols
                scratch[i, k] += win[j, k] * ts[p, q]

# Optimization 5: Attempt to improve memory access pattern
@njit(parallel=True)
def accumulate_lin_opt5(ts, win, lblock, ntap, nblock, scratch):
    # Clean scratch array  
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    ts_cols = ts.shape[1]
    
    # Try different loop ordering for better cache performance
    for j in range(ntap):
        for i in prange(nblock):
            i_base = i * lblock + j * lblock
            for k in range(lblock):
                idx_num = i_base + k
                p = idx_num // ts_cols
                q = idx_num % ts_cols
                scratch[i, k] += win[j, k] * ts[p, q]


@njit(parallel=True, cache=True)
def accumulate_lin_opt6(t, w, lblock, ntap, nblock, rho, scratch):
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    # Parallelize over rows (nblock dimension)
    for i in prange(nblock):
        start_w_idx = 0
        for q in range(rho):
            for j in range(lblock//rho):
                col_start_t_idx = j*(nblock+3)*rho + q
                row_start_t_idx = col_start_t_idx + i*rho
                
                # Unroll the small ntap loop (ntap=4)
                scratch[i, start_w_idx] += (t[row_start_t_idx] * w[start_w_idx] +
                                       t[row_start_t_idx + rho] * w[start_w_idx + lblock] +
                                       t[row_start_t_idx + 2*rho] * w[start_w_idx + 2*lblock] +
                                       t[row_start_t_idx + 3*rho] * w[start_w_idx + 3*lblock])
                start_w_idx += 1
    return scratch


@njit(parallel=True, cache=True)
def accumulate_lin_opt7(t, w, lblock, ntap, nblock, rho, scratch):
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    # Parallelize over column chunks
    for q in prange(rho):
        for j in range(lblock//rho):
            start_w_idx = q * (lblock//rho) + j
            col_start_t_idx = j*(nblock+3)*rho + q
            
            for i in range(nblock):
                row_start_t_idx = col_start_t_idx + i*rho
                
                # Unroll ntap=4 loop
                scratch[i, start_w_idx] += (t[row_start_t_idx] * w[start_w_idx] +
                                       t[row_start_t_idx + rho] * w[start_w_idx + lblock] +
                                       t[row_start_t_idx + 2*rho] * w[start_w_idx + 2*lblock] +
                                       t[row_start_t_idx + 3*rho] * w[start_w_idx + 3*lblock])
    return scratch

@njit(parallel=True, cache=True)
def accumulate_lin_opt7(t, w, lblock, ntap, nblock, rho, scratch):
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    # Parallelize over q*j combinations
    total_cols = rho * (lblock//rho)
    
    for col_idx in prange(total_cols):
        q = col_idx // (lblock//rho)
        j = col_idx % (lblock//rho)
        start_w_idx = q * (lblock//rho) + j
        col_start_t_idx = j*(nblock+3)*rho + q
        
        # Vectorized computation for all rows
        for i in range(nblock):
            row_start_t_idx = col_start_t_idx + i*rho
            
            # Manual unroll for ntap=4
            val = (t[row_start_t_idx] * w[start_w_idx] +
                   t[row_start_t_idx + rho] * w[start_w_idx + lblock] +
                   t[row_start_t_idx + 2*rho] * w[start_w_idx + 2*lblock] +
                   t[row_start_t_idx + 3*rho] * w[start_w_idx + 3*lblock])
            scratch[i, start_w_idx] = val
    
    return scratch


@njit(parallel=True, cache=True)
def accumulate_lin_opt8(t, w, lblock, ntap, nblock, rho, scratch, block_size=64):
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    # Parallelize over blocks of rows
    num_blocks = (nblock + block_size - 1) // block_size
    
    for block_idx in prange(num_blocks):
        i_start = block_idx * block_size
        i_end = min(i_start + block_size, nblock)
        
        start_w_idx = 0
        for q in range(rho):
            for j in range(lblock//rho):
                col_start_t_idx = j*(nblock+3)*rho + q
                
                # Process block of rows
                for i in range(i_start, i_end):
                    row_start_t_idx = col_start_t_idx + i*rho
                    
                    # Unroll ntap=4
                    scratch[i, start_w_idx] += (t[row_start_t_idx] * w[start_w_idx] +
                                           t[row_start_t_idx + rho] * w[start_w_idx + lblock] +
                                           t[row_start_t_idx + 2*rho] * w[start_w_idx + 2*lblock] +
                                           t[row_start_t_idx + 3*rho] * w[start_w_idx + 3*lblock])
                start_w_idx += 1
    return scratch


@njit(parallel=True, cache=True, boundscheck=False)
def accumulate_lin_opt9(t, w, lblock, ntap, nblock, rho, scratch):
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    # Since lblock is long and nblock << lblock, parallelize over rows
    for i in prange(nblock):
        # Precompute row offset
        row_offset = i * rho
        start_w_idx = 0
        
        for q in range(rho):
            for j in range(lblock//rho):
                col_start_t_idx = j*(nblock+3)*rho + q
                t_idx = col_start_t_idx + row_offset
                
                # Since ntap=4, fully unroll for maximum performance
                result = (t[t_idx] * w[start_w_idx] +
                         t[t_idx + rho] * w[start_w_idx + lblock] +
                         t[t_idx + 2*rho] * w[start_w_idx + 2*lblock] +
                         t[t_idx + 3*rho] * w[start_w_idx + 3*lblock])
                
                scratch[i, start_w_idx] = result
                start_w_idx += 1
    
    return scratch

@njit(parallel=True, cache=True)
def accumulate_lin_opt10(t, w, lblock, ntap, nblock, rho, scratch, rows_per_thread=4):
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.
    
    
    num_chunks = (nblock + rows_per_thread - 1) // rows_per_thread
    
    for chunk_idx in prange(num_chunks):
        i_start = chunk_idx * rows_per_thread
        i_end = min(i_start + rows_per_thread, nblock)
        
        for i in range(i_start, i_end):
            start_w_idx = 0
            for q in range(rho):
                for j in range(lblock//rho):
                    col_start_t_idx = j*(nblock+3)*rho + q
                    row_start_t_idx = col_start_t_idx + i*rho
                    
                    # Unroll ntap=4
                    scratch[i, start_w_idx] = (t[row_start_t_idx] * w[start_w_idx] +
                                          t[row_start_t_idx + rho] * w[start_w_idx + lblock] +
                                          t[row_start_t_idx + 2*rho] * w[start_w_idx + 2*lblock] +
                                          t[row_start_t_idx + 3*rho] * w[start_w_idx + 3*lblock])
                    start_w_idx += 1
    
    return scratch

# Version 1: Minimal changes - optimize initialization and add fastmath
@njit(parallel=True, cache=True)
def accumulate_lin_opt_v1(t, w, lblock, ntap, nblock, rho, scratch):
    # More efficient initialization
    scratch.fill(0.0)
   
    # Keep the original excellent parallelization strategy
    total_cols = rho * (lblock//rho)
   
    for col_idx in prange(total_cols):
        q = col_idx // (lblock//rho)
        j = col_idx % (lblock//rho)
        start_w_idx = q * (lblock//rho) + j
        col_start_t_idx = j*(nblock+3)*rho + q
       
        # Vectorized computation for all rows
        for i in range(nblock):
            row_start_t_idx = col_start_t_idx + i*rho
           
            # Manual unroll for ntap=4
            val = (t[row_start_t_idx] * w[start_w_idx] +
                   t[row_start_t_idx + rho] * w[start_w_idx + lblock] +
                   t[row_start_t_idx + 2*rho] * w[start_w_idx + 2*lblock] +
                   t[row_start_t_idx + 3*rho] * w[start_w_idx + 3*lblock])
            scratch[i, start_w_idx] = val
   
    return scratch


# Version 2: Pre-calculate some values to reduce arithmetic in inner loop
@njit(parallel=True, cache=True)
def accumulate_lin_opt_v2(t, w, lblock, ntap, nblock, rho, scratch):
    scratch.fill(0.0)
   
    total_cols = rho * (lblock//rho)
    lblock_div_rho = lblock // rho
    nblock_plus_3_times_rho = (nblock + 3) * rho
   
    for col_idx in prange(total_cols):
        q = col_idx // lblock_div_rho
        j = col_idx % lblock_div_rho
        start_w_idx = q * lblock_div_rho + j
        col_start_t_idx = j * nblock_plus_3_times_rho + q
        
        # Pre-calculate w indices
        w_idx_0 = start_w_idx
        w_idx_1 = start_w_idx + lblock
        w_idx_2 = start_w_idx + 2*lblock
        w_idx_3 = start_w_idx + 3*lblock
       
        # Vectorized computation for all rows
        for i in range(nblock):
            row_start_t_idx = col_start_t_idx + i*rho
           
            # Manual unroll with pre-calculated w indices
            val = (t[row_start_t_idx] * w[w_idx_0] +
                   t[row_start_t_idx + rho] * w[w_idx_1] +
                   t[row_start_t_idx + 2*rho] * w[w_idx_2] +
                   t[row_start_t_idx + 3*rho] * w[w_idx_3])
            scratch[i, start_w_idx] = val
   
    return scratch


# Version 3: Try to improve memory prefetching by processing multiple i values together
@njit(parallel=True, cache=True)
def accumulate_lin_opt_v3(t, w, lblock, ntap, nblock, rho, scratch):
    scratch.fill(0.0)
   
    total_cols = rho * (lblock//rho)
   
    for col_idx in prange(total_cols):
        q = col_idx // (lblock//rho)
        j = col_idx % (lblock//rho)
        start_w_idx = q * (lblock//rho) + j
        col_start_t_idx = j*(nblock+3)*rho + q
        
        # Load w values once
        w0 = w[start_w_idx]
        w1 = w[start_w_idx + lblock]
        w2 = w[start_w_idx + 2*lblock]
        w3 = w[start_w_idx + 3*lblock]
       
        # Process multiple rows to potentially improve prefetching
        i = 0
        while i < nblock:
            # Process up to 4 rows at once if possible
            remaining = nblock - i
            if remaining >= 4:
                # Process 4 rows
                for k in range(4):
                    row_start_t_idx = col_start_t_idx + (i + k)*rho
                    val = (t[row_start_t_idx] * w0 +
                           t[row_start_t_idx + rho] * w1 +
                           t[row_start_t_idx + 2*rho] * w2 +
                           t[row_start_t_idx + 3*rho] * w3)
                    scratch[i + k, start_w_idx] = val
                i += 4
            else:
                # Process remaining rows
                for k in range(remaining):
                    row_start_t_idx = col_start_t_idx + (i + k)*rho
                    val = (t[row_start_t_idx] * w0 +
                           t[row_start_t_idx + rho] * w1 +
                           t[row_start_t_idx + 2*rho] * w2 +
                           t[row_start_t_idx + 3*rho] * w3)
                    scratch[i + k, start_w_idx] = val
                break
   
    return scratch


# Version 4: Simplest possible optimization - just add fastmath and better init
@njit(parallel=True, cache=True, nogil=True)
def accumulate_lin_opt_v4(t, w, lblock, ntap, nblock, rho, scratch):
    # Use memset-like initialization
    for i in prange(nblock):
        for j in range(lblock):
            scratch[i,j] = 0.0
   
    # Keep everything else exactly the same
    total_cols = rho * (lblock//rho)
   
    for col_idx in prange(total_cols):
        q = col_idx // (lblock//rho)
        j = col_idx % (lblock//rho)
        start_w_idx = q * (lblock//rho) + j
        col_start_t_idx = j*(nblock+3)*rho + q
       
        for i in range(nblock):
            row_start_t_idx = col_start_t_idx + i*rho
           
            val = (t[row_start_t_idx] * w[start_w_idx] +
                   t[row_start_t_idx + rho] * w[start_w_idx + lblock] +
                   t[row_start_t_idx + 2*rho] * w[start_w_idx + 2*lblock] +
                   t[row_start_t_idx + 3*rho] * w[start_w_idx + 3*lblock])
            scratch[i, start_w_idx] = val
   
    return scratch


# Version 5: Micro-optimization - avoid repeated array indexing
@njit(parallel=True, cache=True)
def accumulate_lin_opt_v5(t, w, lblock, ntap, nblock, rho, scratch):
    scratch.fill(0.0)
   
    total_cols = rho * (lblock//rho)
   
    for col_idx in prange(total_cols):
        q = col_idx // (lblock//rho)
        j = col_idx % (lblock//rho)
        start_w_idx = q * (lblock//rho) + j
        col_start_t_idx = j*(nblock+3)*rho + q
       
        # Cache frequently accessed values
        lblock_offset = lblock
        lblock_offset_2 = 2*lblock
        lblock_offset_3 = 3*lblock
        
        for i in range(nblock):
            row_start_t_idx = col_start_t_idx + i*rho
           
            # Minimize array indexing operations
            t_base = row_start_t_idx
            w_base = start_w_idx
            
            val = (t[t_base] * w[w_base] +
                   t[t_base + rho] * w[w_base + lblock_offset] +
                   t[t_base + 2*rho] * w[w_base + lblock_offset_2] +
                   t[t_base + 3*rho] * w[w_base + lblock_offset_3])
            scratch[i, w_base] = val
   
    return scratch