# Philippe Joly 2025-06-02
# Splits pfb_utils between cpu and gpu functions

import numpy as np
import time
import psutil
import os
from scipy.fft import rfft, irfft, set_workers
import sys
from os import path

import multiprocessing as mp
from numba import njit, prange
from contextlib import contextmanager
import time

sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.utils.pfb_utils import *
from albatros_analysis.src.correlations import baseband_data_classes as bdc

def print_cpu_mem(str_msg):
    print(str_msg)
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f'RSS Memory: {mem_info.rss / 1024**3:.2f} GB')
    print(f'VMS Memory: {mem_info.vms / 1024**3:.2f} GB')
    print(f'Available Memory: {psutil.virtual_memory().available / 1024**3:.2f} GB')


@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter()-start:.4f} s")


@njit(parallel=True)
def accumulate_lin(ts, win, lblock, ntap, nblock, scratch):
    # i n, j ntap, k l
    for i in prange(nblock):
        for k in range(lblock):
            for j in range(ntap):
                idx_num = i*lblock + j * lblock + k
                p,q = idx_num//ts.shape[1], idx_num%ts.shape[1]

                scratch[i,k] += win[j, k]*ts[p, q]

    # for i in prange(nblock):
    #     for j in range(lblock):
    #         acc = 0.
    #         for k in range(ntap):
    #             acc += ts[i, ]


@njit(parallel=True, fastmath=True)
def compute_R(T, W):
    R = np.zeros((5, 8), dtype=T.dtype)
    for row in prange(5):  # parallel over rows
        for col in range(8):  # sliding window position
            acc = 0.0
            for k in range(4):  # rows in W
                for l in range(5):  # cols in W
                    acc += T[row, col + k * 5 + l] * W[k, l]
            R[row, col] = acc
    return R

@njit(parallel=True)
def accumulate_lin_opt6(t, w, lblock, ntap, nblock, rho, out):
    sz = t.size
    
    # Flatten the structure: we loop over start_w_idx directly
    for start_w_idx in prange(lblock):  # Outer loop
        q = start_w_idx % rho
        j = start_w_idx // rho
        col_start = j * (nblock + 3) * rho + q

        for i in range(nblock):  # Process each block
            base_idx = col_start + i * rho  # Starting t index for (i, start_w_idx)

            acc = 0.0
            for k in range(ntap):  # Inner loop (small range)
                acc += t[base_idx + k * rho] * w[start_w_idx + k * lblock]

            out[i, start_w_idx] = acc  # Write back
    return out

def cpu_pfb(timestream, win, out=None, scratch=None, nchan=2049, ntap=4, n_workers=None):
    # print('='*30)
    # print("ts shape",timestream.shape )
    # record_times = []
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    lblock = 2 * (nchan - 1)
    nblock = timestream.size // lblock - (ntap - 1)

    if scratch is None:
        scratch = np.zeros((nblock, lblock), dtype=np.float32)
    else:
        scratch.fill(0.)

    # t0 = time.time()
    # accumulate_lin_opt6(timestream.T.ravel(), win.ravel(), lblock, ntap, nblock, rho, scratch)
    # record_times.append(time.time()-t0)
    # print("opt6 time",time.time()-t0, "s")

    # t0 = time.time()
    accumulate_lin(timestream, win, lblock, ntap, nblock, scratch)
    # record_times.append(time.time()-t0)
    # print("og time",time.time()-t0, "s")

    # t0 = time.time()
    with set_workers(n_workers): # 0.12s
        out = rfft(scratch, axis=1)
    # record_times.append(time.time()-t0)

    return out #, np.array(record_times)


@njit(parallel=True)
def apply_thresh_filter(ddft, matft, thresh, inv_matft):
    # Applies filter and multipliplies by inv_matft
    if thresh > 0:
        scale = 1.0 + thresh * thresh
        for i in prange(ddft.shape[0]):
            for j in range(ddft.shape[1]):
                abs2 = matft[i, j].real**2 + matft[i, j].imag**2
                filt = abs2 / (abs2 + thresh**2) * scale
                ddft[i, j] *= filt * inv_matft[i,j]
    else:
        for i in prange(ddft.shape[0]):
            for j in range(ddft.shape[1]):
                ddft[i, j] *= inv_matft[i, j]

def cpu_ipfb(dat, matft, inv_matft=None, thresh=0.0, n_workers=None):
    record_times = []
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    if inv_matft is None:
        inv_matft = 1.0 / np.conj(matft)

    with set_workers(n_workers):
        t0 = time.time()
        dd = irfft(dat, axis=1) # 0.01 s
        record_times.append(time.time()-t0)

        if not (dd.flags['C_CONTIGUOUS'] and dd.base is None):
            dd = np.ascontiguousarray(dd)
        
        t0 = time.time()
        dd2 = dd.T #.copy() 
        record_times.append(time.time()-t0)
        
        
        t0 = time.time()
        ddft = rfft(dd2, axis=1) # 0.024 s
        record_times.append(time.time()-t0)

        
    t0 = time.time()
    apply_thresh_filter(ddft, matft, thresh, inv_matft) # 0.007
    record_times.append(time.time()-t0)

    # with timer("irfft 2"):
    with set_workers(n_workers): 
        t0 = time.time()
        res = irfft(ddft , axis=1) # 0.13s 
        record_times.append(time.time()-t0)

    return res.T #, np.array(record_times)



def get_matft_cpu(nslice,nchan=2049,ntap=4, n_workers=None):
    # Philippe Joly 2025-06-03
    # Implemented get_matft but for CPU
    
    if n_workers is None:
        n_workers = mp.cpu_count()

    nn=2*(nchan-1)
    dwin=sinc_hamming(ntap,nn)

    cpu_win = np.asarray(dwin,dtype='float32',order='c')
    cpu_win= np.reshape(cpu_win,[ntap,len(cpu_win)//ntap])
    mat=np.zeros((nslice,nn),dtype='float32',order='c')
    mat[:ntap,:]=cpu_win
    
    mat=mat.T.copy()
    # print("mat size is", mat.shape, np.prod(mat.shape)*4/1024**3, "GB")
    # print("doing matft axis=1", mat.shape, mat.base is None, mat.flags['C_CONTIGUOUS'])

    with set_workers(n_workers):
        matft = rfft(mat,axis=1)
    return matft


# def process_chunks(args):

#     (
#         i, chunks, start_specnums, channels, channel_idxs, acclen, cut_chunks, cut, repfb_chanstart, repfb_chanend, filt_thresh, osamp, pfb_size,
#         (matft_name, matft_shape, matft_dtype, win_name, win_shape, win_dtype)
#     ) = args

#     matft_shm = shared_memory.SharedMemory(name=matft_name)
#     matft = np.ndarray(matft_shape, dtype=matft_dtype, buffer=matft_shm.buf)

#     win_shm = shared_memory.SharedMemory(name=win_name)
#     cpu_win_big = np.ndarray(win_shape, dtype=win_dtype, buffer=win_shm.buf)
        
#     pol0_new_ants = []
#     pol1_new_ants = []
#     new_cut_chunks_ants = []
#     perc_missing_ants = []


#     for j, chunk in enumerate(chunks):
#         start_specnum = start_specnums[j]

#         pol0 = bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums'],start_specnum,channels[channel_idxs],acclen,nchans=2049)
#         pol1 = bdc.make_continuous_gpu(chunk['pol1'],chunk['specnums'],start_specnum,channels[channel_idxs],acclen,nchans=2049)

#         perc_missing = (1 - len(chunk["specnums"])/acclen)*100

#         # Inverse PFB Process
#         to_ipfb_pol0 = np.empty((pfb_size, 2049), dtype='complex64', order='C')
#         to_ipfb_pol1 = to_ipfb_pol0.copy()

#         # Check here if dimensions match
#         to_ipfb_pol0[:2*cut] = cut_chunks[j, 0,:,:] 
#         to_ipfb_pol0[2*cut:] = pol0

#         to_ipfb_pol1[:2*cut] = cut_chunks[j, 1,:,:] 
#         to_ipfb_pol1[2*cut:] = pol1

    
#         raw_pol0 = cpu_ipfb(to_ipfb_pol0, matft, thresh=filt_thresh)
#         raw_pol1 = cpu_ipfb(to_ipfb_pol1, matft, thresh=filt_thresh)

    
#         pol0_new = cpu_pfb(raw_pol0[cut:-cut],cpu_win_big,nchan=2048*osamp+1,ntap=4)
#         pol1_new = cpu_pfb(raw_pol1[cut:-cut],cpu_win_big,nchan=2048*osamp+1,ntap=4)

#         new_cut_chunks =np.array([pol0[-2*cut:,:], pol1[-2*cut:,:]]) 

#         pol0_new_ants.append(pol0_new[:,repfb_chanstart:repfb_chanend])
#         pol1_new_ants.append(pol1_new[:,repfb_chanstart:repfb_chanend])
#         new_cut_chunks_ants.append(new_cut_chunks) 
#         perc_missing_ants.append(perc_missing)

#     return i, pol0_new_ants, pol1_new_ants, new_cut_chunks_ants, perc_missing_ants
