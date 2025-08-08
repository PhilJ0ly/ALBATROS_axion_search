# Philippe Joly 2025-06-26

""" 
GPU RePFB Stream

This script is desingned as an implementation of a RePFB algorithm to change the frequency resolution
of ALBATROS telescope data.

To reduce GPU memory requirements and PFB smoothness, this is written as a streaming solution
"""


import sys
from os import path
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr
import cupy as cp
from albatros_analysis.src.utils import pfb_gpu_utils as pu
import numpy as np
import time


import ctypes
lib = ctypes.CDLL('./libcgemm_batch.so')

lib.cgemm_strided_batched.argtypes = [
    ctypes.c_void_p,  # A.ptr
    ctypes.c_void_p,  # B.ptr
    ctypes.c_void_p,  # C.ptr
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int,     # K
    ctypes.c_int      # batchCount
]
lib.cgemm_strided_batched.restype = None


"""
RePFB Cross-Correlation Averaging

This function performs a reverse PFB and GPU-based cross-correlation on streaming baseband data from multiple antennas.

Parameters:
- idxs: List of antenna indices.
- files: List of baseband files per antenna.
- acclen: Accumulation length in units of 4096-sample IPFB output blocks.
- pfb_size: Size of the PFB transform.
- nchunks: Total number of IPFB output chunks to process.
- nblock: Number of PFB blocks per iteration (streamed).
- chanstart, chanend: Frequency channel bounds.
- osamp: Oversampling factor for RePFB.
- cut: Number of rows to cut from top and bottom of IPFB before PFB (to remove edge effects).
- filt_thresh: Regularization parameter for IPFB deconvolution filter.

Returns:
- vis: Averaged cross-correlation matrix per time and frequency.
- missing_fraction: Fraction of missing data per chunk and antenna.
- freqs: Array of final frequency bins processed.
"""

def repfb_xcorr_avg(idxs,files,acclen,nchunks, nblock, chanstart,chanend,osamp,cut=10,filt_thresh=0.45):

    nant = len(idxs)
    npol = 2
    ntap=4

    # nblock = (pfb_size-2*cut)*4096 // lblock - (ntap - 1)

    lblock = 4096*osamp
    szblock = int(nblock*lblock) # <-- this is to adjust for the pfb offset
    # should change such that the other way around

    lchunk = 4096*acclen # <-- length of chunk after IPFB (not sure)

    assert nblock >= ntap, "nblock must be at least 4"
    print("nblock", nblock, "lblock", lblock)
    print("ipfb_size", acclen)
    
    dwin=pu.sinc_hamming(ntap,lblock)
    cupy_win_big=cp.asarray(dwin,dtype='float32',order='c')
    print("win", cupy_win_big.shape)

    matft = cp.asnumpy(pu.get_matft(acclen))
    filt = cp.asarray(pu.compute_filter(matft, filt_thresh))
    print("filt", filt.shape)

    # pu.print_mem("WIN/FILT")

    pol = cp.empty((acclen+2*cut, 2049), dtype='complex64', order='C') #<-- not sure about shape
    cut_pol = cp.zeros((nant, npol, 2*cut, 2049), dtype='complex64', order='C')

    pfb_buf = cp.zeros((nblock, lblock), dtype='complex64', order='C')  # <- this assume that ipfb output is raveled
    rem_buf = cp.empty((nant, npol, lchunk), dtype='complex64', order='C')
    
    
    print("pol", pol.shape)
    print("pfb_buf", pfb_buf.shape, "rem_buf", rem_buf.shape)
    
    antenna_objs = []
    for i in range(nant):
        aa = bdc.BasebandFileIterator(
            files[i],
            0,
            idxs[i],
            acclen,
            nchunks=nchunks, # <-- we can make it very large such that it loads a very small amount every time
            chanstart=chanstart,
            chanend=chanend,
            type='float'
        )
        antenna_objs.append(aa)
    channels=np.asarray(aa.obj.channels,dtype='int64')
    nchan = (aa.obj.chanend - aa.obj.chanstart)*osamp 
    repfb_chanstart = channels[aa.obj.chanstart] * osamp
    repfb_chanend = channels[aa.obj.chanend] * osamp 


    start_specnums = [ant.spec_num_start for ant in antenna_objs] 
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    jobs = list(zip(*antenna_objs))
    print("chunky inputs:",nchunks, lchunk, nblock, lblock)
    job_chunks = get_chunky(nchunks, lchunk, nblock, lblock)
    
    # job_chunks = get_chunkier(nchunks, lchunk, nblock+(ntap-1)*lblock)
    print("job_chunks", len(job_chunks), job_chunks[:10])

    xin = cp.empty((nant*npol, nblock-(ntap-1), nchan),dtype='complex64',order='F') # <- needs to be FORTRAN for cuda xcorr func
    vis = np.zeros((nant*npol, nant*npol, nchan, len(job_chunks)), dtype="complex64", order="F") # should not be nchunks (should be n super chunks)      
    missing_fraction = np.zeros((nant, len(job_chunks)), dtype='float64', order='F') # <-- dont know when to put Fortran order

    print("vis", vis.shape, "xin", xin.shape, "missing", missing_fraction.shape)

    rem_idx = np.zeros((nant, npol)).astype("uint64")
    pfb_idx = np.zeros((nant, npol)).astype("uint64")
    # pfb_idx[:,:] = (nblock + (ntap-1) )*lblock
    # pfb_idx[:,:] = lblock
    # print(job_chunks)
    # raise ValueError

    time_count = 0
    pfb_idx[:,:] = 0

    jobs = zip(*antenna_objs)
    job_num = len(jobs)
    chunk_idx = 0

    while chunk_idx < job_num:
        for j in range(nant):
            for k in range(npol):
                while pfb_idx[j,k] <= szblock:
                    if pfb_idx[j,k]+rem_idx[j,k] > 0:

                    else:
                        pfb_buf[]


    for i, chunks in enumerate(zip(*antenna_objs)):
        t0 = time.perf_counter()
        if i%10==0: print(i)
        start_event.record()
        for j in range(nant):
            chunk=chunks[j]
            start_specnum = start_specnums[j]

            for k in range(npol):
                pol[:2*cut] = cut_pol[j,k,:,:]
                pol[2*cut:] = bdc.make_continuous_gpu(chunk[f"pol{k}"],chunk['specnums'],start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)

                pfb_scratch = pu.cupy_ipfb(pol, filt)[cut:-cut].ravel() # <-- Memory??

                if pfb_idx[j,k] + lchunk 

    
                


                        

    
    end_t = time.perf_counter()
    time_count += end_t - start_t
    if (s-1) % 10000 == 0:
        print(f"Job Chunk {s+1}/{len(job_chunks)}, avg time {time_count/(s+1):.4f} s")
    
    print(30*"=")
    print(f"Completed {len(job_chunks)}/{len(job_chunks)} Job Chunks\navg time per job: {time_count/len(job_chunks):.4f} s")
    print(30*"=")
    
    vis = np.ma.masked_invalid(vis)
    return vis, missing_fraction, np.arange(repfb_chanstart, repfb_chanend) 
