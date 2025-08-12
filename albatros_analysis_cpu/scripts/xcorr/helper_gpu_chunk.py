import sys
from os import path
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr
import cupy as cp
from albatros_analysis.src.utils import pfb_gpu_utils as pu
import numpy as np
import ctypes
import time

lib = ctypes.CDLL('./libcgemm_batch.so')

# 2) Declare the C function signature
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


def repfb_xcorr_avg(idxs, files, pfb_size, nchunks, chanstart, chanend, osamp, 
                              cutsize=16, filt_thresh=0.45, freq_chunk_size=512, output_file=None):
    """
    Memory-optimized version of REPFB cross-correlation with frequency chunking
    
    Parameters:
    -----------
    freq_chunk_size : int
        Number of frequency channels to process at once (reduces memory usage)
    """
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    pu.print_mem("helper start")
    nant = len(idxs)
    npol = 2

    # ----------------- START IPFB SETUP -----------------------#
    cut = int(pfb_size/cutsize)
    acclen = pfb_size - 2*cut
    ntap = 4
    nn = 2*2048*osamp
    assert acclen % osamp == 0

    re_pfb_size = acclen//osamp - ntap + 1
    assert re_pfb_size > 0 

    nchan = 1*osamp
    print("Memory usage estimates:")
    # pu.print_mem("before dwin")
    dwin = pu.sinc_hamming(ntap, nn)
    # pu.print_mem("dwin")
    cupy_win_big = cp.asarray(dwin, dtype='float32', order='c')
    # pu.print_mem("win")
    # print(f"cupy_win_big size: {np.prod(cupy_win_big.shape) * 4 / 1024**3:.2f} GB")

    

    matft = pu.get_matft(pfb_size)
    # matft = None
    # print(matft.shape, matft.dtype)
    # print(f"matft size: {np.prod(matft.shape) * 8 / 1024**3:.2f} GB")
    # raise ValueError
    to_ipfb_pol0 = cp.empty((pfb_size, 2049), dtype='complex64', order='C') 
    print(f"to_ipfb_pol0: {np.prod(to_ipfb_pol0.shape) * 8  / 1024**3:.2f} GB")


    to_ipfb_pol1 = to_ipfb_pol0.copy()
    print(f"to_ipfb_pol1: {np.prod(to_ipfb_pol1.shape) * 8  / 1024**3:.2f} GB")
    
    pu.print_mem("to_ipfb_pol1")
    cut_chunks = cp.zeros((nant, npol, 2*cut, 2049), dtype='complex64', order='C')
    print(f"cut_chunks size: {np.prod(cut_chunks.shape) * 8  / 1024**3:.2f} GB")
    
    # ----------------- END IPFB SETUP -------------------------#
    
    antenna_objs = []
    for i in range(nant):
        aa = bdc.BasebandFileIterator(
            files[i], 0, idxs[i], acclen, nchunks=nchunks,
            chanstart=chanstart, chanend=chanend, type='float'
        )
        antenna_objs.append(aa)
    
    channels = np.asarray(aa.obj.channels, dtype='int64')
    nchan = (aa.obj.chanend - aa.obj.chanstart) * osamp
    repfb_chanstart = channels[aa.obj.chanstart] * osamp
    repfb_chanend = channels[aa.obj.chanend] * osamp
    
    print(f"Processing {nchan} channels, {nchunks} time chunks")
    print(f"Using frequency chunks of size: {freq_chunk_size}")
    
    # Calculate frequency chunking parameters
    n_freq_chunks = (nchan + freq_chunk_size - 1) // freq_chunk_size
    
    # Setup output array (use memory mapping if requested)
    if output_file:
        vis = np.memmap(output_file, dtype='complex64', mode='w+',
                       shape=(nant*npol, nant*npol, nchan, nchunks), order='F')
        print(f"Using memory-mapped output file: {output_file}")
    else:
        vis = np.zeros((nant*npol, nant*npol, nchan, nchunks), dtype="complex64", order="F")
    
    # Allocate chunked buffers
    max_chunk_size = min(freq_chunk_size, nchan)
    xin_full = cp.empty((nant*npol, re_pfb_size, nchan), dtype='complex64', order='F')
    xin_chunk = cp.empty((nant*npol, re_pfb_size, max_chunk_size), dtype='complex64', order='F')
    scratch_chunk = cp.empty((nant*npol, nant*npol, max_chunk_size), dtype='complex64', order='F')
    
    missing_fraction = np.zeros((nant, nchunks), dtype='float64', order='F')
    
    print(f"Chunked buffer sizes:")
    print(f"xin_full size: {np.prod(xin_full.shape) * 8 / 1024**2:.2f} MB")
    print(f"xin_chunk size: {np.prod(xin_chunk.shape) * 8 / 1024**2:.2f} MB")
    print(f"scratch_chunk size: {np.prod(scratch_chunk.shape) * 8 / 1024**2:.2f} MB")

    
    # Memory pool for cleanup
    mempool = cp.get_default_memory_pool()
    
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    tot = 0
    count = 0
    
    for i, chunks in enumerate(zip(*antenna_objs)):
        t0 = time.perf_counter()
            
        start_event.record()
        
        # Process all antennas to fill xin_full
        for j in range(nant):
            chunk = chunks[j]
            start_specnum = start_specnums[j]
            pol0 = bdc.make_continuous_gpu(chunk['pol0'], chunk['specnums'], 
                                         start_specnum, channels[aa.obj.channel_idxs], 
                                         acclen, nchans=2049)
            pol1 = bdc.make_continuous_gpu(chunk['pol1'], chunk['specnums'], 
                                         start_specnum, channels[aa.obj.channel_idxs], 
                                         acclen, nchans=2049)
           
            perc_missing = (1 - len(chunk["specnums"]) / acclen) * 100
            missing_fraction[j, i] = perc_missing
            
            # IPFB processing
            to_ipfb_pol0[:2*cut] = cut_chunks[j, 0, :, :]
            to_ipfb_pol0[2*cut:] = pol0
            to_ipfb_pol1[:2*cut] = cut_chunks[j, 1, :, :]
            to_ipfb_pol1[2*cut:] = pol1

            raw_pol0 = pu.cupy_ipfb(to_ipfb_pol0, matft, thresh=filt_thresh)
            raw_pol1 = pu.cupy_ipfb(to_ipfb_pol1, matft, thresh=filt_thresh)

            pol0_new = pu.cupy_pfb(raw_pol0[cut:-cut], cupy_win_big, 
                                 nchan=2048*osamp+1, ntap=4)
            pol1_new = pu.cupy_pfb(raw_pol1[cut:-cut], cupy_win_big, 
                                 nchan=2048*osamp+1, ntap=4)
           
            cut_chunks[j, 0, :, :] = pol0[-2*cut:, :]
            cut_chunks[j, 1, :, :] = pol1[-2*cut:, :]
            xin_full[j*nant, :, :] = pol0_new[:, repfb_chanstart:repfb_chanend]
            xin_full[j*nant+1, :, :] = pol1_new[:, repfb_chanstart:repfb_chanend]
        
        # Process in frequency chunks
        for freq_idx in range(n_freq_chunks):
            start_freq = freq_idx * freq_chunk_size
            end_freq = min(start_freq + freq_chunk_size, nchan)
            actual_chunk_size = end_freq - start_freq
            
            # Copy chunk to smaller buffer
            xin_chunk[:, :, :actual_chunk_size] = xin_full[:, :, start_freq:end_freq]
            
            # Cross-correlate this frequency chunk
            out_chunk = cr.avg_xcorr_all_ant_gpu(
                xin_chunk[:, :, :actual_chunk_size], 
                nant, npol, re_pfb_size, actual_chunk_size, 
                split=1, out=scratch_chunk[:, :, :actual_chunk_size]
            )
            
            # Copy result back to host
            vis[:, :, start_freq:end_freq, i] = cp.asnumpy(out_chunk)
        
        end_event.record()
        end_event.synchronize()
        
        tot += time.perf_counter() - t0
        count += 1
        
        if count % 10 == 0:
            print(f"Processed {count} iterations, avg time: {tot/count:.3f}s")
            # Periodic memory cleanup
            mempool.free_all_blocks()
    
    print(f"Final stats - {count} iterations, avg time: {tot/count:.3f}s")
    
    # Final cleanup
    del xin_full, xin_chunk, scratch_chunk, cupy_win_big
    mempool.free_all_blocks()
    
    if not output_file:
        vis = np.ma.masked_invalid(vis)
    
    return vis, missing_fraction, np.arange(repfb_chanstart, repfb_chanend)
    


# Usage example with different memory optimization strategies
def get_memory_usage_estimate(nant, npol, nchan, nchunks, re_pfb_size, osamp, freq_chunk_size=None):
    """Estimate memory usage for different configurations"""
    
    # Original memory usage
    vis_size = nant*npol * nant*npol * nchan * nchunks * 8 / 1024**3
    xin_size = nant*npol * re_pfb_size * nchan * 8 / 1024**3
    scratch_size = nant*npol * nant*npol * nchan * 8 / 1024**3
    filter_size = (2*2048*osamp) * 4 / 1024**3  # float32
    
    print("Original memory estimates:")
    print(f"  Visibility array: {vis_size:.2f} GB")
    print(f"  Input buffer (xin): {xin_size:.2f} GB") 
    print(f"  Scratch buffer: {scratch_size:.2f} GB")
    print(f"  Filter window: {filter_size:.2f} GB")
    print(f"  Total GPU memory: {xin_size + scratch_size + filter_size:.2f} GB")
    
    if freq_chunk_size:
        # Chunked memory usage
        chunk_nchan = min(freq_chunk_size, nchan)
        xin_chunk_size = nant*npol * re_pfb_size * chunk_nchan * 8 / 1024**3
        scratch_chunk_size = nant*npol * nant*npol * chunk_nchan * 8 / 1024**3
        filter_half_size = (2*2048*osamp) * 2 / 1024**3  # float16
        
        print(f"\nChunked memory estimates (chunk_size={freq_chunk_size}):")
        print(f"  Input buffer (chunked): {xin_chunk_size:.2f} GB")
        print(f"  Scratch buffer (chunked): {scratch_chunk_size:.2f} GB") 
        print(f"  Filter window (half precision): {filter_half_size:.2f} GB")
        print(f"  Total GPU memory: {xin_size + xin_chunk_size + scratch_chunk_size + filter_half_size:.2f} GB")
        print(f"  Memory reduction: {((xin_size + scratch_size + filter_size) - (xin_size + xin_chunk_size + scratch_chunk_size + filter_half_size)):.2f} GB")

