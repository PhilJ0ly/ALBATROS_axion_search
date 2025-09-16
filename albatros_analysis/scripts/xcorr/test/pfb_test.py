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

def pfb_true(timestream, win, out=None, nchan=2049, ntap=4):
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
        
    out = cp.fft.rfft(y,axis=1)
    return out

def pfb_true_steve(timestream,  window, nchan, ntap=4):
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

    ts = cp.random.rand(ntap-1+nblocks[-1], 4096*osamp).astype("float32")
    
    outs = []
    slice_true = True

    for i, nblock in enumerate(nblocks):
        outs.append(pu.cupy_pfb(
            ts[:ntap-1+nblock],
            win,
            nchan=2048 * osamp + 1, 
            ntap=4
        ))
    
    nblocks.append(nblocks[-1])
    outs.append(pfb_true(
        ts[:ntap-1+nblock],
        win,
        nchan=2048 * osamp + 1, 
        ntap=4
    ))

    for i, out in enumerate(outs):
        if i < len(outs)-2:
            print(nblocks[i], outs[i].shape)
            for j in range(i, len(outs)):
                print(nblocks[i], nblocks[j], np.allclose(outs[i], outs[j][:nblocks[i]]))
                slice_true = slice_true and np.allclose(outs[i], outs[j][:nblocks[i]])
        else:
            break
    # print(outs[-1].shape, outs[-2].shape)    
    print(nblocks[-2], nblocks[1], np.allclose(outs[-1], outs[-2]))
    
    overlap_true = True
    print(f"\nEquivalence with overlap of ntap-1 blocks")
    print(50*'-')
    # check with equivalence iwth overlap
    iblocks = [10, 100, 150, 350, 500]
    for iblock in iblocks:
        out1 = pu.cupy_pfb(
            ts[:iblock],
            win,
            nchan=2048 * osamp + 1, 
            ntap=4
        ) 
        out2 = pu.cupy_pfb(
            ts[iblock-(ntap-1):], # <-- overlap
            win,
            nchan=2048 * osamp + 1, 
            ntap=4
        ) 
        overlap_true = overlap_true and np.allclose(np.concatenate((out1, out2), axis=0), outs[-1])
        print("iblock", iblock, np.allclose(np.concatenate((out1, out2), axis=0), outs[-1]))


print(50*'=')
print("slice true:", slice_true)
print("overlap true:", overlap_true)
print(50*'=')