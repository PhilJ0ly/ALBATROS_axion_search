# Philippe Joly 2025-09-11

import ctypes
import os
from .. import xp

lib = ctypes.cdll.LoadLibrary(
    os.path.realpath(__file__ + r"/..") + "/libcgemm_batch.so"
)
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


def avg_xcorr_all_ant_gpu(x: xp.ndarray, nant: int, npol: int, ntime: int, nfreq: int, split: int = 1, stay_split=False, scratch=None, out=None):
    #sgemm/cgemm callsign (m,n,k, ldA, strideA, ldB, strideB, ldC, strideC, nbatch)
    #A = m x k  | B = n x k  | C = m x n when B transpose enabled
    M = nant * npol
    N = M
    K = ntime
    if(K % split != 0):
        raise ValueError("split should be a divisor of ntime")
    batchCount = nfreq * split

    if out is None:
        if stay_split:    
            out = xp.empty((M, N, nfreq, split), dtype='complex64', order='F')
        else:
            out = xp.empty((M, N, nfreq), dtype='complex64', order='F')
    elif (
        ( out.shape != (M, M, nfreq) and not stay_split)
        or (stay_split and out.shape != (M, M, nfreq, split))
        or out.dtype != x.dtype 
        or not out.flags.f_contiguous
        ):
        raise ValueError("invalid out buffer")

    if split > 1:
        # Allocate scratch buffer if not provided
        if not stay_split:
            if scratch is None:
                scratch = xp.empty((M, N, nfreq, split), dtype='complex64', order='F')
            elif (
                scratch.shape != (M, N, nfreq, split)
                or scratch.dtype != x.dtype
                or not scratch.flags.f_contiguous
            ):
                raise ValueError("invalid scratch buffer")
        
        # Process each time chunk
        K_chunk = K // split
        for i in range(split):
            # Extract time chunk for this split
            start_time = i * K_chunk
            end_time = (i + 1) * K_chunk
            x_chunk = x[:, start_time:end_time, :]  # Shape: (M, 1, K_chunk, nfreq)
            
            # Perform batched matrix multiplication for this chunk
            # scratch[:, :, :, i] will store results for this time chunk
            if stay_split:
                scratch_slice = out[:, :, :, i]
            else:
                scratch_slice = scratch[:, :, :, i]
            
            lib.cgemm_strided_batched(
                ctypes.c_void_p(x_chunk.data.ptr),
                ctypes.c_void_p(x_chunk.data.ptr),
                ctypes.c_void_p(scratch_slice.data.ptr),
                M, N, K_chunk, nfreq
            )
            
            # Normalize by chunk size
            scratch_slice /= K_chunk
        
        if not stay_split:
            # Average across splits (time chunks)
            out = xp.mean(scratch, axis=3, out=out if out.shape == (M, N, nfreq) else None)            
    else:
        lib.cgemm_strided_batched(
            ctypes.c_void_p(x.data.ptr),
            ctypes.c_void_p(x.data.ptr),
            ctypes.c_void_p(out.data.ptr),
            M, N, K//split, batchCount
        )
        out /= K
    
    return out

def avg_xcorr_all_ant_gpu_og(x: xp.ndarray, nant: int,npol: int, ntime: int, nfreq: int, split : int = 1, scratch = None, out=None):
    #sgemm/cgemm callsign (m,n,k, ldA, strideA, ldB, strideB, ldC, strideC, nbatch)
    #A = m x k  | B = n x k  | C = m x n when B transpose enabled
    M=nant*npol
    N=M
    K=ntime
    if(K%split!=0):
        raise ValueError("split should be a divisor of ntime")
    batchCount=nfreq*split

    if out is None:
        if split >  1:    
            out = xp.empty((M,N,nfreq, split),dtype='complex64',order='F')
        else:
            out = xp.empty((M,N,nfreq),dtype='complex64',order='F')
    elif (
        (split==1 and out.shape != (M, M, nfreq))
        or (split>1 and out.shape != (M, M, nfreq, split))
        or out.dtype != x.dtype 
        or not out.flags.f_contiguous
        ):
        raise ValueError("invalid out buffer")

    if split > 1:
        raise NotImplementedError() #do a proper buffered solution later.
        # scratch.reshape(m,n,split,nbatch,order='F').sum(axis=2,out=out) #reduce along split time axis
        # print("reduced out", out.flags, out.shape)
    else:
        lib.cgemm_strided_batched(
            ctypes.c_void_p(x.data.ptr),
            ctypes.c_void_p(x.data.ptr),
            ctypes.c_void_p(out.data.ptr),
            M, N, K//split, batchCount
        )
        out/=K
    return out