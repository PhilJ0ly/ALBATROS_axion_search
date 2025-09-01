# Philippe Joly 2025-09-01


import numpy as np
import cupy as cp

import sys
from os import path
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.utils import pycufft
from albatros_analysis.src.utils.pfb_utils import *
from scipy.fft import rfft


def pfb_true_gpu(timestream,  window, nchan, ntap=4):
    # old cpu pfb from Steve.
    # Slow but true.

    # number of samples in a sub block
    # print(30*">")
    
    lblock = 2*(nchan-1)
    # print("lblock", lblock)
    # number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    # print("nblock", nblock)
    if nblock==int(nblock): nblock=int(nblock)
    else: raise Exception("nblock is {}, should be integer".format(nblock))

    # initialize array for spectrum 
    dtype=cp.complex64 if timestream.dtype=='float32' else cp.complex128
    spec = cp.zeros((nblock,nchan), dtype=dtype)
    # spec = np.zeros((nblock,lblock), dtype='float32')

    def s(ts_sec):
        return cp.sum(ts_sec.reshape(ntap,lblock),axis=0) # this is equivalent to sampling an ntap*lblock long fft - M

    win = window
    win = win.astype(timestream.dtype)
    # iterate over blocks and perform PFB
    for bi in range(nblock):
        # cut out the correct timestream section
        ts_sec = timestream[bi*lblock:(bi+ntap)*lblock].copy()
        to_pfb = ts_sec*win
        # print("Cpu will pfb", to_pfb.reshape(ntap, lblock))
        # print("Cpu will pfb", s(to_pfb))
        spec[bi] = pycufft.rfft(s(to_pfb), axis=0)
        # spec[bi] = s(to_pfb)

    return spec

def pfb_true(timestream,  window, nchan, ntap=4):
    # old cpu pfb from Steve.
    # Slow but true.

    # number of samples in a sub block
    # print(30*">")
    
    lblock = 2*(nchan-1)
    # print("lblock", lblock)
    # number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    # print("nblock", nblock)
    if nblock==int(nblock): nblock=int(nblock)
    else: raise Exception("nblock is {}, should be integer".format(nblock))

    # initialize array for spectrum 
    dtype=np.complex64 if timestream.dtype=='float32' else np.complex128
    spec = np.zeros((nblock,nchan), dtype=dtype)
    # spec = np.zeros((nblock,lblock), dtype='float32')

    def s(ts_sec):
        return np.sum(ts_sec.reshape(ntap,lblock),axis=0) # this is equivalent to sampling an ntap*lblock long fft - M

    win = window
    win = win.astype(timestream.dtype)
    # iterate over blocks and perform PFB
    for bi in range(nblock):
        # cut out the correct timestream section
        ts_sec = timestream[bi*lblock:(bi+ntap)*lblock].copy()
        to_pfb = ts_sec*win
        # print("Cpu will pfb", to_pfb.reshape(ntap, lblock))
        # print("Cpu will pfb", s(to_pfb))
        # spec[bi] = np.fft.rfft(s(to_pfb))
        spec[bi] = rfft(s(to_pfb))
        # spec[bi] = s(to_pfb)

    return spec

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

if __name__=="__main__":
    nblock = 39
    osamp = 1024
    
    ntap = 4

    dwin = sinc_hamming(ntap, 4096 * osamp).astype('float32')
    cupy_win_big = cp.asarray(dwin, dtype='float32', order='c')

    xin = np.random.randn((nblock+ntap-1)*4096*osamp).astype('float32')
    xin_gpu = cp.asarray(xin, dtype='float32', order='c')

    out = pfb_true(xin, dwin, 2048*osamp+1, ntap=ntap)
    out_gpu = cupy_pfb(xin_gpu, cupy_win_big, nchan=2048*osamp+1, ntap=ntap)
    # out_gpu2 = cupy_pfb(xin_gpu, cupy_win_big, nchan=2048*osamp+1, ntap=ntap)

    print("out, out_gpu shapes", out.shape, out_gpu.shape)
    print("out, out_gpu types", out.dtype, out_gpu.dtype)
    print("out, out_gpu mean abs diff", np.mean(np.abs(out - out_gpu.get())))
    print("out, out_gpu max abs diff", np.max(np.abs(out - out_gpu.get())))
    
    print("out, out_gpu mean rel diff", np.mean(np.abs(out - out_gpu.get())/np.abs(out)))
    print("out, out_gpu max rel diff", np.max(np.abs(out - out_gpu.get())/np.abs(out)))

    print("out, out_gpu all close", np.allclose(out, out_gpu.get()))
    # print("out_gpu, out_gpu2 all close", cp.allclose(out_gpu, out_gpu2))