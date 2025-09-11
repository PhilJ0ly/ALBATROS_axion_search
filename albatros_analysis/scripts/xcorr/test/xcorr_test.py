# Philippe Joly 2025-09-11

"""
This script is designed to test the new avg_xcorr_all_ant_gpu version that allows for split >1
"""

import numpy as np
import cupy as cp
import ctypes

import sys
from os import path
sys.path.insert(0, path.expanduser("~"))

from albatros_analysis.src.correlations.correlations import avg_xcorr_all_ant_gpu as xcorr, avg_xcorr_all_ant_gpu_og as xcorr_og


# def set_arr(nblock, npol=2, nant=1 , nchan=4096*100):
#     return cp.random.rand(nant*npol, nblock, nchan).astype('complex64')

if __name__=="__main__":
    nblocks = np.arange(5)

    nant, npol = 1, 2
    nchan=4096*100

    for i in range(nblocks.shape[0]):
        nblock = 2**nblocks[i]
    
        xin = cp.random.rand(nant*npol, nblock, nchan).astype('complex64')

        print(50*'=')
        print(nblock, "nblock")
        print("input shape", xin.shape)

        out1 = xcorr(xin, nant, npol, nblock, nchan, split=1)
        print("xcorr out split=1 shape", out1.shape)

        out_og = xcorr_og(xin, nant, npol, nblock, nchan, split=1)
        print("xcorr_og  split=1 shape", out_og.shape)

        outn = xcorr(xin, nant, npol, nblock, nchan, split=nblock, stay_split=True)
        print("xcorr split=nblock shape", outn.shape)
        print("xcorr and xcorr_og the same split=1", cp.allclose(out1, out_og))

