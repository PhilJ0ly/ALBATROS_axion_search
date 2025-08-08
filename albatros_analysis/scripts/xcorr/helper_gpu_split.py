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


def repfb_xcorr_avg(idxs,files,pfb_size,nchunks,chanstart,chanend,osamp,cutsize=16,filt_thresh=0.45):
    pu.print_mem("Start Helper")
    nant = len(idxs)
    npol = 2

    # ----------------- START IPFB SETUP -----------------------#\
    
    cut=int(pfb_size/cutsize)
    acclen=pfb_size - 2*cut
    ntap=4
    nn=2*2048*osamp
    assert acclen%osamp == 0

    re_pfb_size = acclen//osamp - ntap + 1
    assert re_pfb_size > 0 
    
    dwin=pu.sinc_hamming(ntap,nn)
    cupy_win_big=cp.asarray(dwin,dtype='float32',order='c')

    matft = cp.asnumpy(pu.get_matft(pfb_size))
    filt = cp.asarray(pu.compute_filter(matft, filt_thresh))

    to_ipfb_pol = cp.empty((pfb_size,2049),dtype='complex64', order='C') 
    cut_chunks = cp.zeros((nant, npol, 2*cut, 2049),dtype='complex64', order='C') # need to maintain last chunks for each antenna
    pu.print_mem("Base Arr")

    print("cupy_win_big size", np.prod(cupy_win_big.shape)*4/1024**3, "GB")
    print("filt size", np.prod(filt.shape)*8/1024**3, "GB")
    print("to_ipfb size", np.prod(to_ipfb_pol.shape)*8/1024**3, "GB")
    print("cut_chunks size", np.prod(cut_chunks.shape)*8/1024**3, "GB")
    # print("acclen", acclen, "pfb_size", pfb_size)
    missing_flag=False
    # ----------------- END IPFB SETUP -------------------------#
    
    antenna_objs = []
    for i in range(nant):
        aa = bdc.BasebandFileIterator(
            files[i],
            0, #fileidx is 0 = start idx is inside the first file
            idxs[i],
            acclen,
            nchunks=nchunks,
            chanstart=chanstart,
            chanend=chanend,
            type='float'
        )
        antenna_objs.append(aa)
    channels=np.asarray(aa.obj.channels,dtype='int64')
    nchan = (aa.obj.chanend - aa.obj.chanstart)*osamp #nchan -> increases due to upchannelization
    repfb_chanstart = channels[aa.obj.chanstart] * osamp
    repfb_chanend = channels[aa.obj.chanend] * osamp # Removed a -1 here was:aa.obj.chanend-1
    # print("channels are", channels)
    # print("start and end chans are", repfb_chanstart, repfb_chanend)
    # sys.exit()
    # print("start and end are", )
    split = 1
    # print("nant", nant, "nchunks", nchunks, "nchan", nchan)
    
    vis = np.zeros((nant*npol, nant*npol, nchan, nchunks), dtype="complex64", order="F")
    xin = cp.empty((nant*npol, re_pfb_size, nchan),dtype='complex64',order='F')
    

    scratch = cp.empty((nant*npol,nant*npol,nchan*split),dtype='complex64',order='F')
    missing_fraction = np.zeros((nant, nchunks), dtype='float64', order='F')
    # missing_mask = cp.empty((nant,nchunks),dtype='float32',order='F') #all integers. float is ok. cuda doesnt like integers.

    print("vis size", np.prod(vis.shape)*8/1024**2, "MB")
    print("scratch size", np.prod(scratch.shape)*8/1024**2, "MB")
    print("xin size", np.prod(xin.shape)*8/1024**2, "MB")

    pu.print_mem("Before Processing")

    # rowcounts = np.empty(nchunks, dtype="int64")
    start_specnums = [ant.spec_num_start for ant in antenna_objs] #specnums are always on the host. so numpy is OK.
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    tot = 0
    count = 0
    pols = ["pol0", "pol1"]

    for i, chunks in enumerate(zip(*antenna_objs)):
        t0 = time.perf_counter()
        if i%10==0: print(i)
        start_event.record()
        for j in range(nant):
            chunk=chunks[j]
            start_specnum = start_specnums[j]

            perc_missing = (1 - len(chunk["specnums"]) / acclen) * 100
            missing_fraction[j, i] = perc_missing

            for k in range(len(pols)):
                pol_name = pols[k]

                pol=bdc.make_continuous_gpu(chunk[pol_name],chunk['specnums'],start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)
            
                to_ipfb_pol[:2*cut] = cut_chunks[j,k,:,:] # <---- zeros for CHUNK 0
                to_ipfb_pol[2*cut:] = pol

                raw_pol = pu.cupy_ipfb(to_ipfb_pol, filt)

                pol_new = pu.cupy_pfb(raw_pol[cut:-cut],cupy_win_big,nchan=2048*osamp+1,ntap=4)
           
                cut_chunks[j,k,:,:] = pol[-2*cut:,:]
                xin[j*nant+k,:,:] = pol_new[:, repfb_chanstart : repfb_chanend] 
                
        out=cr.avg_xcorr_all_ant_gpu(xin,nant,npol,re_pfb_size,nchan,split=1,out=scratch)
        end_event.record()
        end_event.synchronize()
        # print("CUDA IPFB+XCORR time ",cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        vis[:,:,:,i] = cp.asnumpy(out) #scratch should still be on the device

        tot += time.perf_counter()-t0
        count += 1
        if count %10==0:
            print("niter", count, "time", tot/count, "s BW", 2*nant*pol.nbytes/tot*count/1e6, "MSPS")

    print("niter", count, "time", tot/count, "s BW", 2*nant*pol.nbytes/tot*count/1e6, "MSPS")
    vis = np.ma.masked_invalid(vis)
    return vis, missing_fraction, np.arange(repfb_chanstart, repfb_chanend) 
    