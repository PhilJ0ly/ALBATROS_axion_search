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
    szblock = int((nblock + (ntap-1) )*lblock) # <-- this is to adjust for the pfb offset
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

    pfb_buf = cp.zeros((nant, npol, nblock+(ntap-1), lblock), dtype='complex64', order='C')  # <- this assume that ipfb output is raveled
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
    print("chunky inputs:",nchunks, lchunk, nblock+(ntap-1), lblock)
    job_chunks = get_chunky(nchunks, lchunk, nblock+(ntap-1), lblock)
    
    # job_chunks = get_chunkier(nchunks, lchunk, nblock+(ntap-1)*lblock)
    print("job_chunks", len(job_chunks), job_chunks[:10])

    xin = cp.empty((nant*npol, nblock, nchan),dtype='complex64',order='F') # <- needs to be FORTRAN for cuda xcorr func
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

    print(f"starting {len(job_chunks)} Jobs over {nchunks} chunks...")
    for s, (job_idx1, job_idx2) in enumerate(job_chunks):
                
        start_event.record()
        start_t = time.perf_counter()
        subjobs = jobs[job_idx1:job_idx2]

        pfb_buf[:,:,:-1,:] = pfb_buf[:,:,1:,:]
        pfb_idx[:,:] = 0 

        for j in range(nant):
            start_specnum = start_specnums[j]
            missing_counter = 0

            for k in range(npol):
                if pfb_idx[j,k]+rem_idx[j,k] > szblock:
                    assert job_idx1 == job_idx2
                    extra_idx = szblock-pfb_idx[j,k]
                    # print("szblock", szblock, "pfb_idx", pfb_idx[j,k], "extra", extra_idx)

                    # add remaining to complete the pfb buffer
                    pfb_buf[j,k,-1,:] = rem_buf[j,k,:extra_idx]

                    # shift remaining buffer
                    rem_buf[j,k, :rem_idx[j,k]-extra_idx] = rem_buf[j,k, extra_idx:rem_idx[j,k]]
                    rem_idx[j,k] -= extra_idx

                    pfb_idx[j,k] = szblock
                else:
                    assert  job_idx1 != job_idx2
                    pfb_buf[j,k,-1,:rem_idx[j,k]] = rem_buf[j,k, :rem_idx[j,k]]
                    pfb_idx[j,k] += rem_idx[j,k]
                    rem_idx[j,k] = 0

                    for i, chunks in enumerate(subjobs):
                        
                        chunk=chunks[j]
                        missing_fraction[j, s] += (1 - len(chunk["specnums"]) / acclen) * 100 
                        missing_counter += 1

                        # pol[:2*cut] = cut_pol[j,k,:,:]
                        pol[:] = bdc.make_continuous_gpu(chunk[f"pol{k}"],chunk['specnums'],start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)
                        # cut_pol[j,k, :, :] = pol[-2*cut:]

                        pfb_scratch = pu.cupy_ipfb(pol, filt).ravel() # how to handle memory here??
                        # print(pfb_scratch.shape)

                        # pfb_scratch = pu.cupy_ipfb(pol, filt).ravel() 

                        final_idx = pfb_idx[j,k]+lchunk

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
                    print(f"{s+1}/{len(job_chunks)} (Ant {j}, pol {k}): Incomplete pfb_buffer with only {pfb_idx[j,k]} instead of {szblock}")
                    pfb_buf[j,k].flat[pfb_idx[j,k]:] = 0.

                xin[j*npol+k,:,:] = pu.cupy_pfb(pfb_buf[j,k],cupy_win_big,nchan=2048*osamp+1,ntap=4)[:, repfb_chanstart : repfb_chanend]


            if missing_counter != 0:
                missing_fraction[j,s] /= missing_counter
            else:
                missing_fraction[j,s] = missing_fraction[j,s-1]
                # here maybe a rolling count of the chunks (i) in use to know what is the exact missing percentage
            
        out=cr.avg_xcorr_all_ant_gpu(xin,nant,npol,nblock,nchan,split=1)
        end_event.record()
        end_event.synchronize()

        vis[:,:,:,s] = cp.asnumpy(out)

        end_t = time.perf_counter()
        time_count += end_t - start_t
        if (s-1) % 10000 == 0:
            print(f"Job Chunk {s+1}/{len(job_chunks)}, avg time {time_count/(s+1):.4f} s")
    
    print(30*"=")
    print(f"Completed {len(job_chunks)}/{len(job_chunks)} Job Chunks\navg time per job: {time_count/len(job_chunks):.4f} s")
    print(30*"=")
    
    vis = np.ma.masked_invalid(vis)
    return vis, missing_fraction, np.arange(repfb_chanstart, repfb_chanend) 

def get_chunky(nchunks, lchunk, nblock, lblock):
    ranges = []
    big_sz = nblock * lblock
    need = big_sz
    remainder = 0
    cur_chunk = 0

    while True:
        start_chunk = cur_chunk
        used = 0

        while remainder < need and cur_chunk < nchunks:
            remainder += lchunk
            cur_chunk += 1
            used += 1

        if remainder >= need:
            ranges.append((start_chunk, cur_chunk))
            remainder -= need
        else:
            # Not enough chunks left to satisfy need
            ranges.append((start_chunk, None))
            break

        # After the first full fill, just fill 1 block at a time
        need = lblock

        # Also handle the case where we can satisfy `need` from remainder only
        if remainder >= need and used == 0:
            # We just use remainder — no chunk consumed
            ranges.append((cur_chunk, cur_chunk))
            remainder -= need
            need = lblock

    return ranges

def get_chunkier(nchunks, lchunk, lblock):
    ranges = []
    remainder = 0
    cur_chunk = 0

    while True:
        start_chunk = cur_chunk
        used = 0
        need = lblock

        while remainder < need and cur_chunk < nchunks:
            remainder += lchunk
            cur_chunk += 1
            used += 1

        if remainder >= need:
            ranges.append((start_chunk, cur_chunk))
            remainder -= need
        else:
            # Not enough data left to fill lblock
            if remainder > 0:
                ranges.append((start_chunk, None))
            break

        # If we can satisfy the next block using remainder only (no new chunks)
        while remainder >= lblock:
            ranges.append((cur_chunk, cur_chunk))
            remainder -= lblock

    return ranges

    # for s, job_idx in enumerate(job_chunks):

    #     subjobs = jobs[job_chunks[0]:job_chunks[1]]

    #     for j in range(nant):
    #         start_specnum = start_specnums[j]
    #         for k in range(npol):

    #             # pol[:] = rem_buf[j,k, :, :]
    #             pol = cp.zeros((acclen+2*cut, 2049), dtype='complex64', order='C') #<-- Is it 4096?

                
                
    #             for i, chunks in enumerate(subjobs):
                    
    #                 chunk=chunks[j]
    #                 start_event.record()

    #                 # Figure this out later
    #                 missing_fraction[j, i] = (1 - len(chunk["specnums"]) / acclen) * 100

    #                 pol[:2*cut] = pol[-2*cut:] # Note that the first will be filled with zeros
    #                 pol[2*cut:] = bdc.make_continuous_gpu(chunk[f"pol{k}"],chunk['specnums'],start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)
                    
    #                 pfb_scratch = pu.cupy_ipfb(pol, filt)[cut:-cut]  # <-- preallocate memory? maybe not

    #                 final_idx = pfb_buf_idx+lchunk
    #                 extra_idx = szblock-pfb_buf_idx

    #                 # handles the logic if the acclen is larger compared to pfb_buf
    #                 max_l = max(1, final_idx//lblock-nblock+1)-1 
    #                 overload = final_idx%lblock
    #                 overload_flag = (overload != 0)

    #                 for l in range(max_l+1): 
    #                     last_flag = (l == max_l)
    #                     extra_start = extra_idx+l*lblock
                        
    #                     if not last_flag: 
    #                         extra_end = extra_start + lblock
    #                         pfb_buf.flat[pfb_buf_idx:] = pfb_scratch.flat[extra_start:extra_end]

    #                     else:
    #                         if overload_flag:
    #                             pfb_buf.flat[pfb_buf_idx:] = pfb_scratch.flat[extra_start:-overload]
    #                         else:
    #                             pfb_buf.flat[pfb_buf_idx:] = pfb_scratch.flat[extra_start:]
                            
                        
    #                     xin[j*nant+k,:,:] = pu.cupy_pfb(pfb_buf[:-1],cupy_win_big,nchan=2048*osamp+1,ntap=4)[:, repfb_chanstart : repfb_chanend]
                    

    #                     pfb_buf[:-1] = pfb_buf[1:]
    #                     pfb_buf_idx = (nblock-1)*lblock

    #                     if last_flag and overload_flag:
    #                         pfb_buf[-1, overload:] = pfb_scratch[-overload:]
    #                         pfb_buf_idx += overload

# should remove (i,i) indices when lchunk > szblock    
# def get_chunk_fill_ranges(nchunks, lchunk, big_sz):
#     ranges = []
#     cur_chunk = 0
#     remainder = 0

#     while cur_chunk < nchunks:
#         start_chunk = cur_chunk
#         total = remainder  # leftover from previous iteration

#         while cur_chunk < nchunks and total < big_sz:
#             total += lchunk
#             cur_chunk += 1

#         if total >= big_sz:
#             ranges.append((start_chunk, cur_chunk))
#             remainder = total - big_sz
#         else:
#             # Not enough data left to fill big_sz
#             ranges.append((start_chunk, None))
#             break

#     return ranges

    # for i, chunks in enumerate(zip(*antenna_objs)):
    #     pfb_buf_idx = 0
    #     for j in range(nant):
    #         for k in range(npol):   
    #             # reset array
    #             pol = cp.zeros((acclen+2*cut, 2049), dtype='complex64', order='C') 

    #             start_event.record()
    #             chunk=chunks[j]
    #             start_specnum = start_specnums[j]

    #             # Figure this out later
    #             missing_fraction[j, i] = (1 - len(chunk["specnums"]) / acclen) * 100

    #             pol[:2*cut] = pol[-2*cut:] # Note that the first will be filled with zeros
    #             pol[2*cut:] = bdc.make_continuous_gpu(chunk[f"pol{k}"],chunk['specnums'],start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)
    #             pfb_scratch = pu.cupy_ipfb(pol, filt)[cut:-cut]  # <-- preallocate memory? maybe not

    #             final_idx = pfb_buf_idx+acclen*4096
    #             extra_idx = szblock-pfb_buf_idx

    #             # handles the logic if the acclen is larger compared to pfb_buf
    #             max_l = max(1, final_idx//lblock-nblock+1)-1 
    #             overload = final_idx%lblock
    #             overload_flag = (overload != 0)

    #             for l in range(max_l+1): 
    #                 last_flag = (l == max_l)
    #                 extra_start = extra_idx+l*lblock
                    
    #                 if not last_flag: 
    #                     extra_end = extra_start + lblock
    #                     pfb_buf.flat[pfb_buf_idx:] = pfb_scratch.flat[extra_start:extra_end]

    #                 else:
    #                     if overload_flag:
    #                         pfb_buf.flat[pfb_buf_idx:] = pfb_scratch.flat[extra_start:-overload]
    #                     else:
    #                         pfb_buf.flat[pfb_buf_idx:] = pfb_scratch.flat[extra_start:]
                        
                    
    #                 xin[j*nant+k,:,:] = pu.cupy_pfb(pfb_buf[:-1],cupy_win_big,nchan=2048*osamp+1,ntap=4)[:, repfb_chanstart : repfb_chanend]
                

    #                 pfb_buf[:-1] = pfb_buf[1:]
    #                 pfb_buf_idx = (nblock-1)*lblock

    #                 if last_flag and overload_flag:
    #                     pfb_buf[-1, overload:] = pfb_scratch[-overload:]
    #                     pfb_buf_idx += overload