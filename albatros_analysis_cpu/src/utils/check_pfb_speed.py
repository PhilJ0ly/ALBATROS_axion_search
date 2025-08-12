# Philippe Joly 2025-06-23
# This script tests CPU PFB/IPFB performance

import os
os.environ['NUMBA_OPT']='3'
os.environ['NUMBA_LOOP_VECTORIZE']='1'
os.environ['NUMBA_ENABLE_AVX']='1'
os.environ['NUMBA_CACHE_DIR'] = '/scratch/s/sievers/philj0ly/tmp/my_numba_cache'
import numpy as np
import numba as nb
from scipy.fft import rfft, set_workers, irfft
import time

@nb.njit(parallel=True)
def _fpfb(timestream, win, scratch, lblock, ntap):
    nblock = timestream.size // lblock - (ntap - 1)
    scratch.fill(0.0)

    for i in nb.prange(nblock):
        for j in range(ntap):
            for k in range(lblock):
                scratch[i,k] += win[j*lblock + k]*timestream[i*lblock + j * lblock + k]

@nb.njit(parallel=True, cache=True)
def _fpfb_optimized(timestream, win, scratch, lblock, ntap):
    nblock = timestream.size // lblock - (ntap - 1)
    
    for i in nb.prange(nblock):
        for k in range(lblock):
            scratch[i,k] = 0.0
            for j in range(ntap):
                scratch[i,k] += win[j*lblock + k] * timestream[i*lblock + j*lblock + k]

# vector
@nb.njit(parallel=True, cache=True)
def _fpfb_vectorized(timestream, win, scratch, lblock, ntap):
    nblock = timestream.size // lblock - (ntap - 1)
    
    for i in nb.prange(nblock):
        # Zero out the row
        scratch[i,:] = 0.0
        
        # Vectorized accumulation
        for j in range(ntap):
            start_idx = i*lblock + j*lblock
            end_idx = start_idx + lblock
            win_start = j*lblock
            win_end = win_start + lblock
            scratch[i,:] += win[win_start:win_end] * timestream[start_idx:end_idx]


# more efficient memory access
@nb.njit(parallel=True, cache=True)
def _fpfb_cache_local(timestream, win, scratch, lblock, ntap):
    nblock = timestream.size // lblock - (ntap - 1)
    
    scratch.fill(0.0)
    
    for j in range(ntap):
        win_offset = j * lblock
        for i in nb.prange(nblock):
            ts_offset = i * lblock + j * lblock
            for k in range(lblock):
                scratch[i,k] += win[win_offset + k] * timestream[ts_offset + k]


def pfb(timestream, win, scratch, lblock, ntap):
    _fpfb_optimized(timestream, win, scratch, lblock, ntap)

    with set_workers(80):
        spectra=rfft(scratch,axis=1)

    return spectra

def fft_holder(timestream, win, scratch, lblock, ntap):
    with set_workers(80):
        spectra=rfft(scratch,axis=1)
    return spectra


def main():
    osamp =64
    lblock=4096*osamp
    ntap=4
    N = lblock * ntap
    w = np.arange(0, N) - N // 2
    window='hamming'
    timestream = np.random.randn(14336*4096)
    nblock = len(timestream)//lblock - (ntap - 1)
    win = np.__dict__[window](N) * np.sinc(w / lblock)
    scratch = np.empty((nblock, lblock), dtype="float64")

    niter=50
    print(f"Testing FPFB Speeds on {timestream.size} Timestream with {niter} Iterations")
    print(nblock, lblock)
    
    
    pfb_win_funcs = [
        # ["_fpfb", _fpfb, 0],
        # ["_fpfb_cache", _fpfb_cache_local, 0],
        ["_fpfb_opt", _fpfb_optimized, 0],
        # ["_fpfb_vector", _fpfb_vectorized, 0],
        ["rfft", fft_holder, 0],
        ["PFB", pfb, 0]
    ]

    for func in pfb_win_funcs:
        name = func[0]
        fpfb = func[1]

        # warm up
        for i in range(min(10, niter//10)):
            y = fpfb(timestream, win, scratch, lblock, ntap)

        for i in range(niter):
            t1=time.time()
            y=fpfb(timestream, win, scratch, lblock, ntap)
            t2=time.time()
            func[2]+=t2-t1

        func[2] /= niter
        print(name)
        print("\t", func[2], timestream.nbytes/func[2]/(1e6))

    
    

if __name__=='__main__':
    main()