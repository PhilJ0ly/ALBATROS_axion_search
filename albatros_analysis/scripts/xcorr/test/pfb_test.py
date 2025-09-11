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

if __name__=="__main__":
    ntap = 4
    osamp = 64

    nblocks = [1,2,4,8,16,32,128,512]

    dwin = pu.sinc_hamming(ntap, 4096 * osamp)
    win = cp.asarray(dwin, dtype='float32', order='c')

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