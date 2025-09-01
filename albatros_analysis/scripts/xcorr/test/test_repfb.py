# Philippe Joly 2025-09-01

import sys
from os import path
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr
import cupy as cp
from albatros_analysis.src.utils import pfb_gpu_utils as pu
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
# import ctypes
# lib = ctypes.CDLL('../libcgemm_batch.so')

from albatros_analysis.scripts.xcorr.helper_gpu_stream_cleanest import *



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

def process_dummy_chunk(ipfb_processor, ant_idx: int, pol_idx: int, gen_type: str='constant', specs: list = None, add_TS: bool = False):
    ts = cp.zeros(ipfb_processor.buffers.sizes.lchunk, dtype='float32')

    if gen_type =='constant':
        # specs = (value,t0)
        t0 = specs[1]
        ts = cp.ones(ipfb_processor.buffers.sizes.lchunk, dtype='float32') * specs[0]
    elif gen_type == 'uniform':
        # specs = (min, max,t0)
        t0 = specs[2]
        ts = cp.random.uniform(specs[0], specs[1], size=ipfb_processor.buffers.sizes.lchunk).astype('float32')
    elif gen_type == 'gaussian':
        # specs = (mean, stddev, t0)
        t0 = specs[2]
        ts = cp.random.normal(specs[0], specs[1], size=ipfb_processor.buffers.sizes.lchunk).astype('float32')
    elif gen_type == 'sine':
        # specs = (frequency, sample_rate, amp, t0) # <- t0 is the time step start offset
        t0 = specs[3]
        t = (cp.arange(ipfb_processor.buffers.sizes.lchunk) + t0) / specs[1]
        print("t", t[0], t[-1])
        ts = (specs[2]*cp.sin(2 * cp.pi * specs[0] * t)).astype('float32')
        # if specs[3]<ipfb_processor.buffers.sizes.szblock:
        #     ipfb_processor.time[specs[3]:specs[3]+ipfb_processor.buffers.sizes.lchunk] = t[:]
        #     ipfb_processor.ts[specs[3]:specs[3]+ipfb_processor.buffers.sizes.lchunk] = ts[:]
    else:
        ts = cp.zeros(ipfb_processor.buffers.sizes.lchunk, dtype='float32')
        t0 = specs[0]

    if add_TS:
        ipfb_processor.ts[t0:t0+ipfb_processor.buffers.sizes.lchunk] = ts[:]
        print("Added to ts at", t0, t0+ipfb_processor.buffers.sizes.lchunk)

    ipfb_processor.buffers.add_chunk_to_buffer(ant_idx, pol_idx, ts)



def repfb_test(acclen: int, nchunks: int, nblock: int,osamp: int, nant: int = 1, cut: int = 10, filt_thresh: float = 0.45, gen_type: str = 'constant') -> np.ndarray:

    """
    Perform reverse PFB and GPU-based cross-correlation on streaming baseband data.
    
    Args:
        acclen: Accumulation length in units of 4096-sample IPFB output blocks
        nchunks: Total number of IPFB output chunks to process
        nblock: Number of PFB blocks per iteration (streamed)
        osamp: Oversampling factor for RePFB
        cut: Number of rows to cut from top and bottom of IPFB
        filt_thresh: Regularization parameter for IPFB deconvolution filter
    
    Returns:
        vis: Averaged cross-correlation matrix per time and frequency
    """
     
    sr = 250e6  # Sample rate in Samples/second

    specs = {
        'constant': [4.0, 0.0], # value, t0
        'uniform': [0.0, 2.0, 0.0], # min, max, t0
        'gaussian': [0.0, 2.0, 0.0], # mean, stddev, t0
        'sine': [6169.08, sr, 4.0, 0.0]  # frequency, sample_rate, amp, t0
    }
    assert gen_type in specs, f"Invalid gen_type: {gen_type}. Must be one of {list(specs.keys())}."
    
    # Setup configuration
    config = ProcessingConfig(
        acclen=acclen, pfb_size=0, nchunks=nchunks, nblock=nblock,
        chanstart=0, chanend=1, osamp=osamp, nant=nant, cut=cut, filt_thresh=filt_thresh
    )
    sizes = BufferSizes.from_config(config)
    
    # Setup components
    cupy_win_big, _ = setup_filters_and_windows(config)
    # cupy_win_big = cp.ones_like(cupy_win_big)  # Use rectangular window for testing
    
    # Initialize managers
    buffer_mgr = BufferManager(config, sizes)
    ipfb_processor = IPFBProcessor(config, buffer_mgr, None, None)
    
    # Setup frequency ranges
    repfb_chanstart = 0 * osamp
    repfb_chanend = 1 * osamp


    # Initialize output arrays
    xin = cp.empty((config.nant * config.npol, nblock, sizes.nchan), dtype='complex64', order='F')
    vis = np.zeros((config.nant * config.npol, nant * config.npol, sizes.nchan, sizes.num_of_pfbs), dtype="complex64", order="F")

    print("Important Values:")
    print(f"nblock: {config.nblock}, lblock: {sizes.lblock}, szblock: {sizes.szblock}")
    print(f"ipfb_size: {config.acclen}, lchunk: {sizes.lchunk}\n")
    
    print('-'*20, ' ', 0, '/', nchunks, '-'*20)
    print('PFB Buffer', buffer_mgr.pfb_idx)
    print('Remaining Buffer', buffer_mgr.rem_idx)

    job_idx = 0
    for i in range(nchunks):

        for ant_idx in range(config.nant):
            for pol_idx in range(config.npol):
                print("sine t0", specs[gen_type][-1])
                process_dummy_chunk(ipfb_processor, ant_idx, pol_idx, gen_type=gen_type, specs=specs[gen_type], add_TS=(ant_idx==0 and pol_idx==0))

        if gen_type == 'sine':
            specs[gen_type][-1] += sizes.lchunk  # Increment time step offset for sine wave

        print('-'*20, ' ', i+1, '/', nchunks, '-'*20)
        print('PFB Buffer', buffer_mgr.pfb_idx)
        print('Remaining Buffer', buffer_mgr.rem_idx)
                
    
        # All PFB idices should be the same after processing all antennas
        assert (buffer_mgr.pfb_idx == buffer_mgr.pfb_idx[0,0]).all()

        # We know that all antennas/pols have the same PFB index after processing so is sufficient to check one
        while buffer_mgr.pfb_idx[0,0] == sizes.szblock:

            print('='*20, 'PFB Buffer Full', '='*20)
            print('PFB Buffer', buffer_mgr.pfb_idx)
            print('Remaining Buffer', buffer_mgr.rem_idx)
            # Generate PFB output for cross-correlation
            for ant_idx in range(config.nant):
                for pol_idx in range(config.npol):
                    output_start = ant_idx * config.npol + pol_idx # <-- still not sure
                    xin[output_start, :, :] = pu.cupy_pfb(
                        buffer_mgr.pfb_buf[ant_idx, pol_idx], 
                        cupy_win_big,
                        nchan=2048 * osamp + 1, 
                        ntap=4
                    )[:, repfb_chanstart:repfb_chanend]

            vis[:, :, :, job_idx] = cp.asnumpy(
                cr.avg_xcorr_all_ant_gpu(xin, config.nant, config.npol, nblock, sizes.nchan, split=1)
            )

            buffer_mgr.reset_overlap_region()
            for ant_idx in range(config.nant):
                for pol_idx in range(config.npol):
                    # Add remaining data to PFB buffer
                    buffer_mgr.add_remaining_to_pfb_buffer(ant_idx, pol_idx)

            
            print('PFB Buffer After', buffer_mgr.pfb_idx)
            print('Remaining Buffer After', buffer_mgr.rem_idx)

            job_idx += 1

        
        if  job_idx % 100 == 0:
            print(f"Job Chunk {job_idx + 1}/{sizes.num_of_pfbs} ")
    
    # Pad buffer if incomplete
    if buffer_mgr.pfb_idx[0,0] > (config.ntap - 1)* sizes.lblock:
        print(f"Job {job_idx + 1}/{sizes.num_of_pfbs}: ")
        print(f"Buffer not Full with only {buffer_mgr.pfb_idx[0,0]} < {sizes.szblock}")
        for ant_idx in range(config.nant):
            for pol_idx in range(config.npol):
                buffer_mgr.pad_incomplete_buffer(ant_idx, pol_idx)
                output_start = ant_idx * config.npol + pol_idx # <-- still not sure
                xin[output_start, :, :] = pu.cupy_pfb(
                    buffer_mgr.pfb_buf[ant_idx, pol_idx], 
                    cupy_win_big,
                    nchan=2048 * osamp + 1, 
                    ntap=4
                )[:, repfb_chanstart:repfb_chanend]

        vis[:, :, :, job_idx] = cp.asnumpy(
            cr.avg_xcorr_all_ant_gpu(xin, config.nant, config.npol, nblock, sizes.nchan, split=1)
        )
        print("PFB buffer padded and processed")
    
    vis = np.ma.masked_invalid(vis)

    true_out = pfb_true(ipfb_processor.ts, cupy_win_big, nchan=(2048*config.osamp+1), ntap=4)[:, repfb_chanstart:repfb_chanend]
    xin_true = cp.empty((config.nant * config.npol, nblock, true_out.shape[-1]), dtype='complex64', order='F')
    xin_true[:] = cp.asarray(true_out)
    vis_true = np.empty((config.nant * config.npol, nant * config.npol, true_out.shape[-1], 1), dtype="complex64", order="F")
    vis_true[:, :, :, 0] = cp.asnumpy(
        cr.avg_xcorr_all_ant_gpu(xin_true, config.nant, config.npol, nblock, true_out.shape[-1], split=1)
    )
    vis_true = np.ma.masked_invalid(vis_true)
    print("pfb_buf, ts shape", buffer_mgr.pfb_buf.shape, ipfb_processor.ts.shape)
    print("xin, xin_true shape", xin.shape, xin_true.shape)
    print(vis.shape, vis_true.shape, "vis, vis_true shapes")
    # checked that the timestreams are the same.
    # same =(ipfb_processor.ts == buffer_mgr.pfb_buf[0,0].ravel())
    # same = same.reshape(config.nblock+(config.ntap-1), sizes.lblock)
    # print("ts the same", same.all())
    # for i in range(same.shape[0]):
    #     print("ts the same %", i, np.mean(same[i])*100, np.mean(ipfb_processor.ts[i*sizes.lblock:(i+1)*sizes.lblock]), np.mean(buffer_mgr.pfb_buf[0,0,i,:]))

    print("window", cupy_win_big.shape)
    
    return vis, vis_true


if __name__=="__main__":

    outdir = "/scratch/philj0ly/test_08_25"

    print("Beginning Testing...\n")

    acclen = 1792
    nchunks = 24
    osamp = 1024
    cut = 19
    nblock = 39
    nant = 1
    gen_types = ['constant', 'uniform', 'gaussian', 'sine']
    # gen_type = 'uniform'  # Options: 'constant', 'uniform',  'gaussian', 'sine'
    # gen_types=['sine']

    for gen_type in gen_types: 
        print("\n", 50*"~")
        print(f"Testing gen_type: {gen_type}")
        print(50*"~")     
        pols, true_out =repfb_test(acclen,nchunks,nblock, osamp, nant=1, cut=cut,filt_thresh=0, gen_type=gen_type)

        fname = f"test_{gen_type}_{str(acclen)}_{str(osamp)}.npz"
        fpath = path.join(outdir,fname)
        np.savez(fpath,data=pols.data,mask=pols.mask, true_data=true_out, true_mask=true_out.mask)

        print("\nSaved with an oversampling rate of", osamp, "and an acclen of", acclen, "at")
        print(fpath)