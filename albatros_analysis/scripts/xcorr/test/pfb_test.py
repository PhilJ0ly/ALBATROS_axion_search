# Philippe Joly 2025-09-11

"""
This script is designed to test that the PFB is the same regardless of nblock

result: it is
"""

import numpy as np
import cupy as cp

import sys
from os import path
sys.path.insert(0, path.expanduser("~"))

from albatros_analysis.src.utils import pfb_gpu_utils as pu

def pfb_true(timestream,  window, nchan, ntap=4):
    # old cpu pfb from Steve.
    # Slow but true.

    # number of samples in a sub block
    print(30*">")
    
    lblock = 2*(nchan-1)
    print("lblock", lblock)
    # number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    print("nblock", nblock)
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
        spec[bi] = np.fft.rfft(s(to_pfb)).get()
        # spec[bi] = s(to_pfb)

    return spec

if __name__=="__main__":
    ntap = 4
    osamp = 64

    nblocks = [1,2,4,8,16,32,128,512]

    dwin = pu.sinc_hamming(ntap, 4096 * osamp)
    win = cp.asarray(dwin, dtype='float32', order='c')
    win

    ts = cp.random.rand(ntap-1+nblocks[-1], 4096*osamp).astype("float32")
    
    outs = []

    for i, nblock in enumerate(nblocks):
        outs.append(pu.cupy_pfb(
            ts[:ntap-1+nblock],
            win,
            nchan=2048 * osamp + 1, 
            ntap=4
        ))

    for i, out in enumerate(outs):
        if i < len(outs)-1:
            print(nblocks[i], outs[i].shape)
            for j in range(i, len(outs)):
                print(nblocks[i], nblocks[j], np.allclose(outs[i], outs[j][:nblocks[i]]))