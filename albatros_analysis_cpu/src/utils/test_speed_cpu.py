# Philippe Joly 2025-06-12

"""
Test script to test the speed of different CPU versions of PFB and IPFB.
"""

import os
os.environ['NUMBA_OPT']='3'
os.environ['NUMBA_LOOP_VECTORIZE']='1'
os.environ['NUMBA_ENABLE_AVX']='1'
os.environ['NUMBA_CACHE_DIR'] = '/scratch/s/sievers/philj0ly/tmp/my_numba_cache'
import sys
import numpy as np
import numba as nb
from scipy.fft import rfft, set_workers, irfft
import time
import multiprocessing as mp
from os import path
sys.path.append(path.expanduser('~'))

from albatros_analysis.src.utils.pfb_cpu_utils import get_matft_cpu

def generate_test_data(nspec, nchan, seed=42):
    np.random.seed(seed)
    
    # Generate complex input data (nspec x nchan)
    dat_real = np.random.randn(nspec, nchan).astype(np.float32)
    dat_imag = np.random.randn(nspec, nchan).astype(np.float32)
    dat_cpu = np.asarray(dat_real + 1j * dat_imag, dtype='complex64', order='C')
    
    return dat_cpu

@nb.njit(parallel=True)
def naive_transpose(x):
    y = np.empty((x.T.shape),dtype=x.dtype)
    nr,nc=y.shape
    for i in nb.prange(nr):
        for j in nb.prange(nc):
            y[i,j]=x[j,i]
    return y

@nb.njit(parallel=True)
def apply_thresh_filter_vectorized(ddft, matft, thresh, inv_matft):
    """Vectorized version processing rows in parallel"""
    if thresh > 0:
        thresh_sq = thresh * thresh
        scale = 1.0 + thresh_sq
        
        for i in nb.prange(ddft.shape[0]):
            row_ddft = ddft[i]
            row_matft = matft[i]
            row_inv = inv_matft[i]
            
            for j in range(ddft.shape[1]):
                abs2 = row_matft[j].real**2 + row_matft[j].imag**2
                filt = abs2 / (abs2 + thresh_sq) * scale
                row_ddft[j] *= filt * row_inv[j]
    else:
        for i in nb.prange(ddft.shape[0]):
            for j in range(ddft.shape[1]):
                ddft[i, j] *= inv_matft[i, j]


def ipfb(dat, matft, inv_matft=None, thresh=0.0, n_workers=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    if inv_matft is None:
        inv_matft = 1.0 / np.conj(matft)

    with set_workers(n_workers):
        dd = irfft(dat, axis=1) 
        
    dd2 = dd.T

    with set_workers(n_workers):  
        ddft = rfft(dd2, axis=1) 

        
    apply_thresh_filter_vectorized(ddft, matft, thresh, inv_matft) 

    with set_workers(n_workers): 
        res = irfft(ddft , axis=1) 

    return naive_transpose(res)

@nb.njit(parallel=True)
def apply_thresh_filter(ddft, matft, thresh, inv_matft):
  
    if thresh > 0:
        scale = 1.0 + thresh * thresh
        for i in nb.prange(ddft.shape[0]):
            for j in range(ddft.shape[1]):
                abs2 = matft[i, j].real**2 + matft[i, j].imag**2
                filt = abs2 / (abs2 + thresh**2) * scale
                ddft[i, j] *= filt * inv_matft[i,j]
    else:
        for i in nb.prange(ddft.shape[0]):
            for j in range(ddft.shape[1]):
                ddft[i, j] *= inv_matft[i, j]

def cpu_ipfb2(dat, matft, inv_matft=None, thresh=0.0, n_workers=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    if inv_matft is None:
        inv_matft = 1.0 / np.conj(matft)

    with set_workers(n_workers):
        dd = irfft(dat, axis=1) 
        
    dd2 = dd.T

    with set_workers(n_workers):  
        ddft = rfft(dd2, axis=1) 

        
    apply_thresh_filter_vectorized(ddft, matft, thresh, inv_matft) 

    with set_workers(n_workers): 
        res = irfft(ddft , axis=1) 

    # returns an untransposed version of the result (the desired value is transposed) this is done to optimize efficiency
    return res


@nb.njit(parallel=True, cache=True)
def accumulate_lin(t, w, lblock, scratch, ntap, rho):
    nblock = t.size // lblock - (ntap - 1)
    scratch.fill(0.0)
   
    total_cols = rho * (lblock//rho)

    # Cache frequently accessed values
    lblock_offset = lblock
    lblock_offset_2 = 2*lblock
    lblock_offset_3 = 3*lblock
   
    for col_idx in nb.prange(total_cols):
        q = col_idx // (lblock//rho)
        j = col_idx % (lblock//rho)
        start_w_idx = q * (lblock//rho) + j
        col_start_t_idx = j*(nblock+3)*rho + q
       
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

@nb.njit(parallel=True, cache=True)
def _fpfb_optimized(timestream, win, scratch, lblock, ntap):
    # Expects flat timestream and win (ntap*lblock), and a scratch that is of shape (nblock, lblock)
    # Writes into scratch
    nblock = timestream.size // lblock - (ntap - 1)
    # print(timestream.shape, win.shape, scratch.shape)
    
    for i in nb.prange(nblock):
        for k in range(lblock):
            scratch[i,k] = 0.0
            for j in range(ntap):
                scratch[i,k] += win[j*lblock + k] * timestream[i*lblock + j*lblock + k]


def pfb(timestream, win, out=None, scratch=None, nchan=2049, ntap=4, n_workers=None):
    # Expects flat timestream and win (ntap*lblock), and a scratch that is of shape (nblock, lblock)
    lblock = 2 * (nchan - 1)
    nblock = timestream.size // lblock - (ntap - 1)

    if n_workers is None:
        n_workers = mp.cpu_count()

    if scratch is None:
        scratch = np.zeros((nblock, lblock), dtype=timestream.dtype)

    _fpfb_optimized(timestream, win, scratch, lblock, ntap)

    with set_workers(n_workers):
        out=rfft(scratch,axis=1)

    return out

def cpu_pfb2(timestream, win, out=None, scratch=None, nchan=2049, ntap=4, n_workers=None):
    # Expects flat win (ntap*lblock) and a 2D timestream which is the transpose of the desired input,
    #  and a scratch that is of shape (nblock, lblock)
    # we require that lblock is a multiple of timestream.shape[0]

    lblock = 2 * (nchan - 1)
    nblock = timestream.size // lblock - (ntap - 1)

    assert lblock % timestream.shape[0] == 0, f"Lblock {lblock} is not a multiple of timestream.shape[0]: {timestream.shape}"

    if n_workers is None:
        n_workers = mp.cpu_count()

    if scratch is None:
        scratch = np.zeros((nblock, lblock), dtype=timestream.dtype)
    
    rho = lblock // timestream.shape[0]
    accumulate_lin(timestream.ravel(), win, lblock, scratch, ntap, rho)

    with set_workers(n_workers):
        spectra=rfft(scratch,axis=1)

    return spectra


def pipe_1(dat, matft, win, scratch, lblock, ntap, inv_matft=None, thresh=0.0, n_workers=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    raw = ipfb(dat, matft, inv_matft, thresh, n_workers)
    spectra = pfb(raw.ravel(), win, scratch, lblock, ntap, n_workers)

    return spectra

def pipe_2(dat, matft, win, scratch, lblock, ntap, inv_matft=None, thresh=0.0, n_workers=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    raw = cpu_ipfb2(dat, matft, inv_matft, thresh, n_workers)
    spectra = cpu_pfb2(raw, win, scratch, lblock, ntap, n_workers)

    return spectra

if __name__=="__main__":
    niter = 10

    thresh = 0.45
    test_cases = [
        {"osamp": 64, "pfb_mult": 256},
        # {"osamp": 512, "pfb_mult": 128},
        # {"osamp": 2**12, "pfb_mult": 32}
        # {"osamp": 2**16, "pfb_mult": 8} too much memory for GPU
    ]

    for tester in test_cases:
        tester["name"] = f"Osamp ({tester['osamp']}) pfb_mult ({tester['pfb_mult']})"
        tester["nslice"] = tester["osamp"]*tester["pfb_mult"]

    for test_case in test_cases:
        print(test_case)

        nspec = test_case["nslice"]

        print("Generating CPU matft...")
        matft_cpu = get_matft_cpu(
            nslice=nspec
        )
        inv_matft = 1.0 / np.conj(matft_cpu)
        
        dat_cpu = generate_test_data(nspec, nchan=2049, seed=41)

        print("Running CPU Pipeline...")

        
        
        subnames = ["irfft 1", "dd.T", "rfft", "new thresh","irfft 2", "res.T"]
        ipfb_times = np.zeros(6)
        tot_t = 0
        for i in range(niter):
            t0 = time.time()
            cpu_ipfb_result, times = cpu_ipfb(dat_cpu, matft_cpu, inv_matft=inv_matft, thresh=thresh)
            ipfb_times += times
            tot_t += time.time()-t0
        ipfb_t = tot_t/niter
        print(ipfb_t, "s per iteration.", 1/ipfb_t, "Hz")
        for i in range(len(subnames)):
            print(subnames[i], ipfb_times[i]/niter)


        cutsize = 16
        osamp = test_case["osamp"]
        pfb_mult = test_case["pfb_mult"]
        dwin = sinc_hamming(ntap=4,lblock=2048*2*osamp)
        cut = int(osamp*pfb_mult/cutsize)

        
        # print(cpu_ipfb_result[cut:-cut].size//262144 -3)

        print("Running CPU PFB...")
        subnames = ["polyphase", "rfft"]
        # cpu_win_big = np.asarray(dwin, dtype='float32', order='C').reshape(4,2*(2048*osamp+1-1))
        cpu_win_big = np.asarray(dwin, dtype='float32', order='C')
        tot_t = 0
        pfb_times = np.zeros(2)
        for i in range(niter):
            t0 = time.time()
            cpu_pfb_result, times = cpu_pfb(cpu_ipfb_result[cut:-cut],cpu_win_big, nchan=2048*osamp+1, ntap=4 )
            pfb_times += times
            tot_t += time.time()-t0
        pfb_t = tot_t/niter
        print(pfb_t, "s per iteration.", 1/pfb_t, "Hz")
        for i in range(len(subnames)):
            print(subnames[i], pfb_times[i]/niter)


        
                    