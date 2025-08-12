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

def repfb_xcorr_avg(idxs,files,acclen,nchunks, nblock, chanstart,chanend,osamp,cut=10,filt_thresh=0.45, cupy_win_big=None, filt=None, verbose=False):

    nant = len(idxs)
    npol = 2
    ntap=4

    lblock = 4096*osamp
    szblock = int((nblock + (ntap-1) )*lblock) 

    lchunk = 4096*acclen # <-- length of chunk after IPFB 

    if cupy_win_big is None:
        dwin=pu.sinc_hamming(ntap,lblock)
        cupy_win_big=cp.asarray(dwin,dtype='float32',order='c')
    

    # CPU version if not enough memory
    # matft = cp.asnumpy(pu.get_matft(acclen+2*cut))
    # filt = cp.asarray(pu.compute_filter(matft, filt_thresh))

    if filt is None:
        matft = pu.get_matft(acclen+2*cut)
        filt = pu.calculate_filter(matft, filt_thresh)
    

    pol = cp.empty((acclen+2*cut, 2049), dtype='complex64', order='C') #<-- not sure about shape
    cut_pol = cp.zeros((nant, npol, 2*cut, 2049), dtype='complex64', order='C')

    pfb_buf = cp.zeros((nant, npol, nblock+(ntap-1), lblock), dtype='float32', order='C')  # <- this assume that ipfb output is raveled
    rem_buf = cp.empty((nant, npol, lchunk), dtype='float32', order='C')
    
    antenna_objs = []
    for i in range(nant):
        aa = bdc.BasebandFileIterator(
            files[i],
            0,
            idxs[i],
            acclen,
            nchunks=nchunks,
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
    job_chunks = get_chunkier(nchunks, lchunk, nblock, lblock, ntap)

    xin = cp.empty((nant*npol, nblock, nchan),dtype='complex64',order='F') # <- needs to be FORTRAN for cuda xcorr func
    vis = np.zeros((nant*npol, nant*npol, nchan, len(job_chunks)), dtype="complex64", order="F") # should not be nchunks (should be n super chunks)      
    missing_fraction = np.zeros((nant, len(job_chunks)), dtype='float64', order='F') # <-- dont know when to put Fortran order

    rem_idx = np.zeros((nant, npol)).astype("uint64")
    pfb_idx = np.zeros((nant, npol)).astype("uint64")
    missing_count = [[]] * nant

    if verbose:
        print("nblock", nblock, "lblock", lblock)
        print("ipfb_size", acclen)
        print("window shape", cupy_win_big.shape)
        print("filter shape", filt.shape)
        print("pol", pol.shape)
        print("pfb_buf", pfb_buf.shape, "rem_buf", rem_buf.shape)
        print("chunky inputs", nchunks, lchunk, nblock, lblock, ntap)
        print("vis", vis.shape, "xin", xin.shape, "missing", missing_fraction.shape)
        pu.print_mem("START ITER")
        print(f"starting {len(job_chunks)} PFB Jobs over {nchunks} IPFB chunks...")
    
    time_count = 0
    for s, (job_idx1, job_idx2) in enumerate(job_chunks):
                
        start_event.record()
        start_t = time.perf_counter()
        subjobs = jobs[job_idx1:job_idx2]

        # Leaving an (ntap-1) X lblock overlap to ensure continuity and avoid redundancy
        # This ensures that we can simply stitch the outputs together
        if s > 0:
            pfb_buf[:,:,:ntap-1,:] = pfb_buf[:,:,-(ntap-1):,:]
            pfb_idx[:,:] = (ntap-1)*lblock 


        for j in range(nant):
            start_specnum = start_specnums[j]

            for k in range(npol):
                if pfb_idx[j,k]+rem_idx[j,k] > szblock:
                    # if k == 0:
                    #     print("job_idx == start", pfb_idx[0,0])
                    assert job_idx1 == job_idx2
                    extra_idx = szblock-pfb_idx[j,k]
                    # print("szblock", szblock, "pfb_idx", pfb_idx[j,k], "extra", extra_idx)

                    # add remaining to complete the pfb buffer
                    pfb_buf[j,k].flat[:] = rem_buf[j,k,:extra_idx]

                    # shift remaining buffer
                    rem_buf[j,k, :rem_idx[j,k]-extra_idx] = rem_buf[j,k, extra_idx:rem_idx[j,k]]
                    rem_idx[j,k] -= extra_idx

                    pfb_idx[j,k] = szblock
                else:
                    assert job_idx1 != job_idx2 or (pfb_idx[j,k]+rem_idx[j,k]) == szblock

                    # Adding remaining buffer to PFB buffer
                    pfb_buf[j,k].flat[pfb_idx[j,k]:pfb_idx[j,k]+rem_idx[j,k]] = rem_buf[j,k, :rem_idx[j,k]]
                    pfb_idx[j,k] += rem_idx[j,k]
                    rem_idx[j,k] = 0


                    # Getting more IPFB Chuhnks to fill PFB buffer
                    # Note that subjobs is already designed such that the PFB buffer will be filled minimaly
                    for i, chunks in enumerate(subjobs):
                        
                        chunk=chunks[j]
                        final_idx = pfb_idx[j,k]+lchunk

                        # Only for first pol as the missing fraction should be the same across pols
                        if k == 0:
                            missing_count[j].append([
                                (1 - len(chunk["specnums"]) / acclen) * 100, # <-- Missing Fraction
                                (final_idx//lblock) % nblock + 1 # <-- Number of PFB jobs in which the chunk will be present 
                            ])

                        pol[:2*cut] = cut_pol[j,k,:,:]
                        pol[2*cut:] = bdc.make_continuous_gpu(chunk[f"pol{k}"],chunk['specnums'],start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)
                        cut_pol[j,k, :, :] = pol[-2*cut:]

                        pfb_scratch = pu.cupy_ipfb(pol, filt)[cut:-cut].ravel() 

                        if final_idx > szblock:
                            assert i >= len(subjobs)-1, "Job plan has an error"
                            extra_idx = szblock-pfb_idx[j,k]
                            rem_idx[j,k] = final_idx - szblock

                            pfb_buf[j,k].flat[pfb_idx[j,k]:] = pfb_scratch[:extra_idx]
                            rem_buf[j,k,:rem_idx[j,k]] = pfb_scratch[extra_idx:]

                            pfb_idx[j,k] = szblock
                        else:
                            pfb_buf[j,k].flat[pfb_idx[j,k]:final_idx] = pfb_scratch[:]

                            rem_idx[j,k] = 0
                            pfb_idx[j,k] = final_idx   

                assert pfb_idx[j,k] <= szblock, f"pfb_buf too big??: {pfb_idx[j,k]} instead of {szblock}"
                if pfb_idx[j,k] != szblock: 
                    if verbose:
                        print(f"{s+1}/{len(job_chunks)} (Ant {j}, pol {k}): Incomplete pfb_buffer with only {pfb_idx[j,k]} instead of {szblock}")
                    pfb_buf[j,k].flat[pfb_idx[j,k]:] = 0.

                xin[j*nant+k,:,:] = pu.cupy_pfb(pfb_buf[j,k],cupy_win_big,nchan=2048*osamp+1,ntap=4)[:, repfb_chanstart : repfb_chanend]

                if k == 0:
                    # Calculate the average missing percentage 
                    # assuming there is equally lchunk elements from every chunk
                    for miss in missing_count[j]:
                        missing_fraction[j, s] += miss[0]
                        miss[1] -= 1
                    missing_fraction[j, s] /= len(missing_count[j])

                    # Remove missing fractions which will not be present in the next PFB job
                    missing_count[j] = [miss for miss in missing_count[j] if miss[1] > 0]

    
        out=cr.avg_xcorr_all_ant_gpu(xin,nant,npol,nblock,nchan,split=1)
        end_event.record()
        end_event.synchronize()

        vis[:,:,:,s] = cp.asnumpy(out)

        end_t = time.perf_counter()
        time_count += end_t - start_t
        if verbose and s % 100 == 0:
            print(f"Job Chunk {s+1}/{len(job_chunks)}, avg time {time_count/(s+1):.4f} s")
    
    if verbose:
        print(30*"=")
        print(f"Completed {len(job_chunks)}/{len(job_chunks)} Job Chunks")
        print(f"avg time per job: {time_count/len(job_chunks):.4f} s")
        print(30*"=")
    
    vis = np.ma.masked_invalid(vis)
    return vis, missing_fraction, np.arange(repfb_chanstart, repfb_chanend), cupy_win_big, filt

def get_chunkier(nchunks, lchunk, nblock, lblock, ntap):
    ranges = []
    stride_sz = nblock * lblock
    overlap_sz = (ntap - 1) * lblock
    total_needed = stride_sz + overlap_sz

    remainder = 0
    cur_chunk = 0
    sample_offset = 0
    while True:
        start_chunk = cur_chunk
        added = 0

        # Accumulate until enough samples
        while remainder < total_needed and cur_chunk < nchunks:
            remainder += lchunk
            cur_chunk += 1
            added += 1

        if remainder >= total_needed:
            ranges.append((start_chunk, cur_chunk))
            remainder -= stride_sz  
            sample_offset += stride_sz
        else:
            if remainder > 0:
                ranges.append((start_chunk, None))
            break

    return ranges
