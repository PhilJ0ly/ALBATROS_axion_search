# Philippe Joly 2025-06-02
# Splits pfb_utils between cpu and gpu functions


import numpy as np
import cupy as cp
from albatros_analysis.src.utils import pycufft
# from cupy.fft import rfft, pycufft.irfft
import time
from albatros_analysis.src.utils.pfb_utils import *
import os


def print_mem(stri):
    print('-'*20,stri, 'Memory', '-'*20)

    # free free memory
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    device = cp.cuda.Device()
    free_device_memory, total_device_memory = device.mem_info
    used_device_memory = total_device_memory - free_device_memory

    print(f"USED: {used_device_memory/ 1024**3:.3} GB")
    print(f"free: {free_device_memory/ 1024**3:.3} GB")
    print(f"total: {total_device_memory/ 1024**3:.3} GB")
    print('-'*60)

def calculate_filter(matft, thresh):
    filt = 1/cp.conj(matft)

    if thresh>0:
        abs_mat_sq = cp.abs(matft)**2
        filt *= abs_mat_sq/(thresh**2+abs_mat_sq)*(1.+thresh**2)
    
    return filt
    
def compute_filter(matft, thresh):
    """
    Direct filter computation.
    
    Args:
        matft: Complex array
        thresh: Threshold value
    
    Returns:
        Filter array
    """
    filt = 1/np.conj(matft)
    if thresh>0:
        matft_abs_sq = np.abs(matft) ** 2
        filt *= matft_abs_sq / (thresh_sq + matft_abs_sq) * (1.0 + thresh**2)
        
    return filt  

def cupy_ipfb_test(dat, filt):
    print(f"Input dat shape: {dat.shape}")
    print(f"Filter shape: {filt.shape}")
    
    
    dd = pycufft.irfft(dat, axis=1)
    print(f"After first IRFFT: {dd.shape}")
    
    dd2 = dd.T.copy()
    print(f"After transpose: {dd2.shape}")
    
    ddft = pycufft.rfft(dd2, axis=1)
    print(f"After second RFFT: {ddft.shape}")
    print(f"ddft energy before filter: {cp.sum(cp.abs(ddft)**2)}")
    
    # Check if shapes are compatible
    print(f"Filter shape: {filt.shape}")
    if ddft.shape != filt.shape:
        print(f"WARNING: Shape mismatch! ddft: {ddft.shape}, filt: {filt.shape}")
    
    ddft = ddft * filt
    print(f"ddft energy after filter: {cp.sum(cp.abs(ddft)**2)}")
    
    res = pycufft.irfft(ddft, axis=1)
    res = res.T
    print(f"Final result shape: {res.shape}")
    return res

def cupy_ipfb(dat,filt, complex_output=False):

    """On-device ipfb. Expects the data to be iPFB'd to live in GPU memory.

    Parameters
    ----------
    dat : cp.ndarray
        nspec x nchan array of complex64
    matft : cp.ndarray
        lblock x nspec array of float32. Note that this is transpose of Jon's original convention
        for speed reasons.
        [w0  wN  w2N  w3N  0  0  . . . ]
        [w1  .             0  0        ]
        [         .        . . .       ]
        [wN-1 . . .   w4N-1     .    0 ]
    thresh : float, optional
        Wiener filter threshold, by default 0.0

    Returns
    -------
    cp.ndarray
        nspec*nchan timestream values as a C-major matrix of shape (nspec x nchan)
    """

    dd=pycufft.irfft(dat,axis=1)

    # assert dd.flags.c_contiguous and dd.base is None

    dd2=dd.T.copy()
    ddft=pycufft.rfft(dd2,axis=1)

    ddft=ddft*filt

    # No need for complex output for RePFB as the PFB will use rfft anyway
    if complex_output:
        res = pycufft.ifft(ddft, axis=1)
    else:
        res = pycufft.irfft(ddft, axis=1)

    res=res.T
    return res

def cupy_pfb_old(timestream, win, out=None, nchan=2049, ntap=4):
    lblock = 2*(nchan-1)
    nblock = timestream.size // lblock - (ntap - 1)
    timestream=timestream.reshape(-1,lblock)
    if out is not None:
        assert out.shape == (nblock, nchan)
    win=win.reshape(ntap,lblock)
    y=timestream*win[:,cp.newaxis] # <-- takes a lot of memory
    y=y[0,:nblock,:]+y[1,1:nblock+1,:]+y[2,2:nblock+2,:]+y[3,3:nblock+3,:]
    out = pycufft.rfft(y,axis=1)
    return out

def cupy_pfb(timestream, win, out=None, nchan=2049, ntap=4, complex_input=False):
    # A more memory efficient version
    lblock = 2*(nchan-1)
    nblock = timestream.size // lblock - (ntap - 1)
    timestream=timestream.reshape(-1,lblock)
    if out is not None:
        assert out.shape == (nblock, nchan)
    win=win.reshape(ntap,lblock)
    
    y = timestream[0:nblock] * win[0]
    
    # Accumulate remaining taps in-place
    for i in range(1, ntap):
        y += timestream[i:nblock+i] * win[i]

    # in RePFB we always use rfft, but in iPFB we can use complex FFT if the input is complex
    if complex_input:
        # If the input is complex, we need to use the complex FFT
        out = pycufft.fft(y, axis=1)
    else:   
        out = pycufft.rfft(y,axis=1)
    return out

def get_matft(nslice,nchan=2049,ntap=4):
    nn=2*(nchan-1)
    dwin=sinc_hamming(ntap,nn)
    cupy_win=cp.asarray(dwin,dtype='float32',order='c')
    cupy_win=cp.reshape(cupy_win,[ntap,len(cupy_win)//ntap])
    mat=cp.zeros((nslice,nn),dtype='float32',order='c')
    mat[:ntap,:]=cupy_win

    # Frees memory
    cupy_win = None
    mat=mat.T.copy()
    matft=pycufft.rfft(mat,axis=1)
    
    return matft

# def sinc_hamming(ntap,lblock):
#     N=ntap*lblock
#     w=cp.arange(0,N)-N/2
#     return cp.hamming(ntap*lblock)*cp.sinc(w/lblock)

# def sinc_hanning(ntap,lblock):
#     N=ntap*lblock
#     w=cp.arange(0,N)-N/2
#     return cp.hanning(ntap*lblock)*cp.sinc(w/lblock)