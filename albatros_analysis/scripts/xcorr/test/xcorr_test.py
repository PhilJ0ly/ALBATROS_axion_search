# Philippe Joly 2025-09-11

"""
This script is designed to test the new avg_xcorr_all_ant_gpu version that allows for split >1

result: it works
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

def main():
    nblocks = np.arange(9)

    nant, npol = 1, 2
    nchan=4096*100

    for i in range(nblocks.shape[0]):
        nblock = 2**nblocks[i]
    
        xin = (cp.random.rand(nant*npol, nblock, nchan)+1j*cp.random.rand(nant*npol, nblock, nchan)).astype('complex64')

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

        idx = nblock//2
        out_og_n = xcorr_og(xin[:,idx,:], nant, npol, 1, nchan, split=1)
        print("xcorr and xcorr_og the same split=n and 1", cp.allclose(outn[:,:,:,idx], out_og_n))

        if nblock>3:
            outn1 =  xcorr(xin[:,:idx,:], nant, npol, idx, nchan, split=idx, stay_split=True)
            outn2 =  xcorr(xin[:,idx:,:], nant, npol, nblock-idx, nchan, split=nblock-idx, stay_split=True)
            # print(outn1.shape, outn2.shape)

            outnv2 = cp.concatenate((outn1, outn2), axis=-1)
            print("piecewise xcorr the same", cp.allclose(outn, outnv2))



if __name__=="__main__":
    nant, npol = 1, 2
    nchan=20
    nblock = 5

    xin = (cp.random.rand(nant*npol, nblock, nchan)+1j*cp.random.rand(nant*npol, nblock, nchan)).astype('complex64')

    xin_concat = cp.concatenate((xin,xin), axis=1)

    vis_concat = xcorr(xin_concat, nant, npol, 2*nblock, nchan, split=2*nblock, stay_split=True)
    vis_other = cp.zeros_like(vis_concat)
    vis_other[:,:,:,:nblock] = xcorr(xin, nant, npol, nblock, nchan, split=nblock, stay_split=True)
    vis_other[:,:,:,nblock:] = xcorr(xin, nant, npol, nblock, nchan, split=nblock, stay_split=True)

    print(cp.allclose(vis_concat, vis_other))
    # print(vis_concat)
    # print(50*'-')
    # print(vis_other)
    
    print(50*'-')
    diff = cp.abs(vis_other-vis_concat)
    print(cp.mean(diff, axis=2))
    # print(50*'-')
    # mask = diff > 1e-6
    # print(cp.unique((vis_other[mask] / vis_concat[mask]).get()))
    # print(50*'-')
    # print(vis_other)
    # print(cp.mean(diff, axis=3))