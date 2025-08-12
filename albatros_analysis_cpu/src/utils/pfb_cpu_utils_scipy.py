# Philippe Joly 2025-06-19

"""
Implementation of PFB and IPFB functions on CPU cores optimized for parallelization
"""

import sys
import os
os.environ['NUMBA_OPT']='3'
os.environ['NUMBA_LOOP_VECTORIZE']='1'
os.environ['NUMBA_ENABLE_AVX']='1'
os.environ['NUMBA_CACHE_DIR'] = '/scratch/s/sievers/philj0ly/tmp/my_numba_cache'

import numpy as np
import numba as nb
import scipy.fft as sfft
import multiprocessing as mp
import psutil

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src.utils.pfb_utils import *
from scipy.fft import rfft, irfft, set_workers


def print_cpu_mem(str_msg):
    print(str_msg)
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f'RSS Memory: {mem_info.rss / 1024**3:.2f} GB')
    print(f'VMS Memory: {mem_info.vms / 1024**3:.2f} GB')
    print(f'Available Memory: {psutil.virtual_memory().available / 1024**3:.2f} GB')

@nb.njit(parallel=True)
def nb_transpose_1d(x, out=None, cut=0):

    """
    Transposes a 2D array x in a parallelized, Numba-accelerated way.
    Creates an empty output array y, then fills it such that y[i, j] = x[j, i].
    Faster than x.T.copy()[cut:-cut].ravel() when used inside Numba-heavy workflows, especially with large matrices.
    """
    nr, nc = x.T.shape
    if out is None:
        out = np.empty((nr-2*cut)*nc,dtype=x.dtype)

    # RAVEL Output
    for i in nb.prange(nr-2*cut):
        for j in nb.prange(nc):
            out[i*nc+j]=x[j,i+cut]          
        
    return out

@nb.njit(parallel=True)
def nb_transpose_2d(x, out=None, cut=0):

    """
    Transposes a 2D array x in a parallelized, Numba-accelerated way.
    Creates an empty output array y, then fills it such that y[i, j] = x[j, i].
    Faster than x.T.copy()[cut:-cut] when used inside Numba-heavy workflows, especially with large matrices.
    """
    nr, nc = x.T.shape
    if out is None:
        out = np.empty((nr-2*cut, nc),dtype=x.dtype)
        
    for i in nb.prange(nr-2*cut):
        for j in nb.prange(nc):
            out[i,j]=x[j,i+cut]

    return out

@nb.njit(parallel=True)
def apply_thresh_filter(ddft, matft, thresh, inv_matft):

    """
    Applies an optional frequency-domain threshold filter to the Fourier-transformed data ddft,
    using the Fourier matrix matft and its inverse inv_matft.
    If thresh > 0, a soft filter is computed per coefficient based on the power spectrum.
    Otherwise, it just multiplies ddft by inv_matft.
    All operations are performed in parallel for efficiency.
    """

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


def cpu_ipfb(dat, matft, out, inv_matft=None, thresh=0.0, cut=0, n_workers=None):

    """
    CPU-based implementation of the inverse polyphase filterbank (iPFB).
    Performs:
        1. Inverse FFT on dat along axis 1 (assuming input is already in frequency domain).
        2. Transposes and applies forward FFT again.
        3. Applies an optional threshold filter.
        4. Final inverse FFT.
        5. Transposes the result back.

    Allows for CPU parallelism via scipy.fft.set_workers() and optional custom inverse matrix.
    """

    if n_workers is None:
        n_workers = mp.cpu_count()
    
    if inv_matft is None:
        inv_matft = 1.0 / np.conj(matft)

    with set_workers(n_workers):
        # print("dat irrft",dat.shape)
        dd = irfft(dat, axis=1) 
        
    dd2 = dd.T

    # print("dd2 rrft", dd2.shape)
    with set_workers(n_workers):  
        ddft = rfft(dd2, axis=1) 
        
    apply_thresh_filter(ddft, matft, thresh, inv_matft) 

    # print("ddft irfft", ddft.shape)
    with set_workers(n_workers): 
        res = irfft(ddft , axis=1) 

    nb_transpose_1d(res, out=out,  cut=cut)

@nb.njit(parallel=True)
def cpu_fpfb(timestream, win, scratch, lblock, ntap):

    """
    Core CPU function to compute the time-domain polyphase filterbank (PFB).
    Takes a flat timestream and a flat win (the filter), and computes
    filtered blocks using ntap taps of length lblock.
    Writes the results into scratch, a 2D array of shape (nblock, lblock).
    Highly optimized using Numba for JIT and parallelism.
    """

    nblock = timestream.size // lblock - (ntap - 1)
    
    for i in nb.prange(nblock):
        for k in range(lblock):
            scratch[i,k] = 0.0
            for j in range(ntap):
                scratch[i,k] += win[j*lblock + k] * timestream[i*lblock + j*lblock + k]


def cpu_pfb(timestream, win, out, scratch=None, nchan=2049, ntap=4, n_workers=None):

    """
    Full polyphase filterbank (PFB) pipeline on CPU.

    1. Computes filtered timestream using cpu_fpfb.
    2. Applies real FFT (rfft) along axis 1 to get frequency-domain representation.
    Returns the PFB spectrum.

    Supports multi-threading and reuse of pre-allocated memory (scratch, out).
    """
    
    # print(timestream.shape, timestream.dtype, win.shape, win.dtype)

    lblock = 2 * (nchan - 1)
    nblock = timestream.size // lblock - (ntap - 1)
    # print(nblock, lblock)

    if n_workers is None:
        n_workers = mp.cpu_count()

    if scratch is None:
        scratch = np.zeros((nblock, lblock), dtype=timestream.dtype)
    else:
        assert scratch.shape == (nblock, lblock)

    cpu_fpfb(timestream, win, scratch, lblock, ntap)
    # print("scratch shape", scratch.shape)
    with set_workers(n_workers):
        # print("pfb rfft", scratch.shape)
        out = sfft.rfft(scratch,axis=1)
    # print("out shape", out.shape, out.dtype)

    # return out

def get_matft_cpu(nslice,nchan=2049,ntap=4, n_workers=None):

    """
    1. Constructs the Fourier matrix (matft) used in the iPFB process on CPU.
    2. Creates a sinc-hamming window (dwin) over ntap taps and nchan channels.
    3. Fills the first ntap slices of mat with the window.
    4. Computes the FFT of the windowed matrix along axis 1.
    5. Returns the transformed matrix (matft).

    Intended to match nslice × nchan dimensionality for compatibility with iPFB.
    Assumes sinc_hamming is defined elsewhere in the module.
    """
    
    if n_workers is None:
        n_workers = mp.cpu_count()//2

    nn=2*(nchan-1)
    dwin=sinc_hamming(ntap,nn)

    cpu_win = np.asarray(dwin,dtype='float64',order='c')
    cpu_win= np.reshape(cpu_win,[ntap,len(cpu_win)//ntap])
    mat=np.zeros((nslice,nn),dtype='float64',order='c')
    mat[:ntap,:]=cpu_win
    
    mat=mat.T.copy()

    with set_workers(n_workers):
        matft = rfft(mat,axis=1)
    return matft
