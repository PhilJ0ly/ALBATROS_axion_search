# Philippe Joly 2025-06-03

# This script consists of helper functions for rebinning ALBATROS data within a CPU architecture in parallel.abs

import sys
from os import path
import traceback
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr 
from albatros_analysis.src.utils import pfb_cpu_utils as pu
import numpy as np
import multiprocessing as mp

import time
import os
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter()-start:.4f} s")

def repfb_xcorr_avg(idxs,files,pfb_size,nchunks,chanstart,chanend,osamp,cutsize=16,filt_thresh=0.45, n_cores=None):
    
    # if n_cores is None or n_cores > mp.cpu_count():
    #     n_cores = mp.cpu_count()

    # print("Using", n_cores, "CPU Cores")
    # os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
    # os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
    # os.environ['NUMEXPR_NUM_THREADS'] = str(mp.cpu_count())
    # os.environ['OPENBLAS_NUM_THREADS'] = str(mp.cpu_count())
    
    # Disable CPU affinity restrictions
    # os.environ['OMP_PROC_BIND'] = 'false'
    # os.environ['OMP_PLACES'] = 'cores'
    
    # # Enable dynamic thread adjustment
    # os.environ['OMP_DYNAMIC'] = 'true'
    
    print(f"Configured for {mp.cpu_count()} cores")

    nant = len(idxs)
    npol = 2

    
    cut=int(pfb_size/cutsize)
    acclen=pfb_size - 2*cut
    ntap=4
    nn=2*2048*osamp
    assert acclen%osamp == 0

    re_pfb_size = acclen//osamp - ntap + 1
    assert re_pfb_size > 0 

    dwin = pu.sinc_hamming(ntap, nn)
    cpu_win_big = np.asarray(dwin, dtype='float32', order='C') #.reshape(4,2*(2048*osamp+1-1)) # <---- note that 4=ntap
    
    matft = pu.get_matft_cpu(pfb_size)
    filt_arr = compute_filter(matft, filt_thresh)
    

    to_ipfb_pol0 = np.empty((pfb_size,2049),dtype='complex64', order='C') 
    to_ipfb_pol1 = to_ipfb_pol0.copy()

    cut_chunks = np.zeros((nant, npol, 2*cut, 2049),dtype='complex64', order='C') # need to maintain last chunks for each antenna

    print("to ipfb pol0 shape",to_ipfb_pol0.shape)
    print("matft shape", matft.shape)

    print("cpu_win_big size", np.prod(cpu_win_big.shape)*4/1024**3, "GB")
    print("matft size", np.prod(matft.shape)*8/1024**3, "GB")
    print("2x to_ipfb size", np.prod(to_ipfb_pol0.shape)*8*2/1024**3, "GB")
    print("cut_chunks size", np.prod(cut_chunks.shape)*8*2/1024**3, "GB")
    print("acclen", acclen, "pfb_size", pfb_size)
    missing_flag=False

    pu.print_cpu_mem("After setup arrays:")

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


    channels = np.asarray(aa.obj.channels, dtype='int64')
    nchan = (aa.obj.chanend - aa.obj.chanstart) * osamp
    repfb_chanstart = channels[aa.obj.chanstart] * osamp
    repfb_chanend = channels[aa.obj.chanend] * osamp
    
    print("start and end chans are", repfb_chanstart, repfb_chanend)
    print("nant", nant, "nchunks", nchunks, "nchan", nchan)

    vis = np.zeros((nant*npol, nant*npol, nchan, nchunks), dtype="complex64", order="C")
    xin = np.empty((nant*npol, re_pfb_size, nchan), dtype='complex64', order='C')

    split = 1
    scratch = np.empty((nant*npol,nant*npol,nchan*split),dtype='complex64',order='C')
    missing_fraction = np.zeros((nant, nchunks), dtype='float64', order='C')

    print("vis size", np.prod(vis.shape)*8/1024**3, "GB")
    print("scratch size", np.prod(scratch.shape)*8/1024**3, "GB")
    print("xin size", np.prod(xin.shape)*8/1024**3, "GB")

    lblock = 2 * (2048*osamp+1 - 1)
    nblock = (pfb_size-2*cut)*4096 // lblock - (ntap - 1)
    
    pfb_scratch = np.empty((nblock, lblock), dtype='float32',order='C')
    pol0_new = np.empty((nblock, 2048*osamp+1), dtype='complex64',order='C')
    pol1_new = pol0_new.copy()

    raw_pol0 = np.empty((pfb_size-2*cut)*4096, dtype='float32',order='C')
    raw_pol1 = raw_pol0.copy()

    print("to_ipfb_pol0", to_ipfb_pol0.shape, to_ipfb_pol0.dtype)
    print("raw_pol0", raw_pol0.shape, raw_pol0.dtype)
    print("cut", cut)
    print("matft", matft.shape)
    raise ValueError

    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    tot = 0
    count = 0
    for i, chunks in enumerate(zip(*antenna_objs)):
        t0 = time.perf_counter()
        if i%10==0: print(i)
        for j in range(nant):
            chunk=chunks[j]
            start_specnum = start_specnums[j]
            #t00 = time.perf_counter()
            pol0=bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums'],start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)
            pol1=bdc.make_continuous_gpu(chunk['pol1'],chunk['specnums'],start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)
            # print("continuous", time.perf_counter()-t00)
            
            #t00 = time.perf_counter()
            perc_missing = (1 - len(chunk["specnums"]) / acclen) * 100
            missing_fraction[j, i] = perc_missing
            # CHUNK 0 is going to have all zeros at the top. Ignore first chunk in the saved output.
            # with timer("to_ipfb setup"):
            to_ipfb_pol0[:2*cut] = cut_chunks[j,0,:,:] # <---- zeros for CHUNK 0
            to_ipfb_pol0[2*cut:] = pol0
            to_ipfb_pol1[:2*cut] = cut_chunks[j,1,:,:] # <---- zeros for CHUNK 0
            to_ipfb_pol1[2*cut:] = pol1
            # print("setp", time.perf_counter()-t00)
            #t00 = time.perf_counter()
            

            print("to_ipfb", to_ipfb_pol0.shape )
            print("raw_pol", raw_pol0.shape)
            print("matft", matft.shape)
            print("cut", cut)
            pu.cpu_ipfb(to_ipfb_pol0, raw_pol0, filt=filt_arr, cut=cut)
            pu.cpu_ipfb(to_ipfb_pol1, raw_pol1, filt=filt_arr, cut=cut)

            # print("IPFB", time.perf_counter()-t00)
            #t00 = time.perf_counter()
            # Info is lost here
            # This change here seems to have added time per iteration
            print("ts", raw_pol0.shape, raw_pol0.dtype)
            print("win", cpu_win_big.shape, cpu_win_big.dtype)
            print("scratch", pfb_scratch.shape, scratch.dtype)
            print("nchan", 2048*osamp+1)
            pol0_new = pu.cpu_pfb(raw_pol0,cpu_win_big, scratch=pfb_scratch, nchan=2048*osamp+1,ntap=4)
            pol1_new = pu.cpu_pfb(raw_pol1,cpu_win_big, scratch=pfb_scratch, nchan=2048*osamp+1,ntap=4)
            # print("PFB", time.perf_counter()-t00)

            #t00 = time.perf_counter()
            cut_chunks[j,0,:,:] = pol0[-2*cut:,:]
            cut_chunks[j,1,:,:] = pol1[-2*cut:,:]
            
            xin[j*nant,:,:] = pol0_new[:, repfb_chanstart : repfb_chanend] # BFI data is C-major for IPFB
            xin[j*nant+1,:,:] = pol1_new[:, repfb_chanstart : repfb_chanend]
            
            # print("saving", time.perf_counter()-t00)

        #t00 = time.perf_counter()
        vis[:,:,:,i] = cr.avg_xcorr_all_ant_cpu(xin,nant,npol,re_pfb_size,nchan,split=1,out=scratch)
        
        # print("avg xcorr", time.perf_counter()-t00)
        
        tot += time.perf_counter()-t0
        count += 1
        if count %10==0:
            print("niter", count, "time", tot/count, "s BW", 2*nant*pol0.nbytes/tot*count/1e6, "MSPS")

    print("niter", count, "time", tot/count, "s BW", 2*nant*pol0.nbytes/tot*count/1e6, "MSPS")

    vis = np.ma.masked_invalid(vis)
    return vis, missing_fraction, np.arange(repfb_chanstart, repfb_chanend) #TODO: for really large BW/delta-nu, we should probably store only start and end



    
    #     n_cores = mp.cpu_count()
    #     print(f"Running on {n_cores} CPU cores")
    #     args_list = [
    #         (
    #             i, chunks, start_specnums, channels, aa.obj.channel_idxs, acclen, cut_chunks, cut, repfb_chanstart, repfb_chanend, filt_thresh, osamp, pfb_size,
    #             (matft_shm.name, matft.shape, matft.dtype, win_shm.name, cpu_win_big.shape, cpu_win_big.dtype)
    #         ) for i, chunks in enumerate(zip(*antenna_objs)) 
    #     ]

    #     with ProcessPoolExecutor(
    #         max_workers=n_cores
    #     ) as executor:
    #         futures = [executor.submit(pu.process_chunks, args) for args in args_list]

    #         for future in as_completed(futures):
    #             i, pol0_out_ants, pol1_out_ants, new_cut_chunks_ants, perc_missing_ants = future.result()
    #             # print("xin", xin.shape, "pol0_out_ants", len(pol0_out_ants), "prec missing", len(perc_missing_ants), "new_cut_chunks_ants", len(new_cut_chunks_ants))

    #             # xin = np.empty((nant*npol, re_pfb_size, nchan), dtype='complex64', order='F')
    #             # missing_fraction = np.zeros((nant, nchunks), dtype='float64', order='F')
    #             for j in range(nant):
    #                 # Note that the sizes were cut-down accordingly in pfb_cpu_utils.py 
    #                 xin[j*nant, :, :] = pol0_out_ants[j]
    #                 xin[j*nant+1, :, :] = pol1_out_ants[j]
    #                 cut_chunks[j] = new_cut_chunks_ants[j]
    #                 missing_fraction[j, i] = perc_missing_ants[j]
                
    #             vis[:, :, :, i] = cr.avg_xcorr_all_ant_cpu(xin, nant, npol, re_pfb_size, nchan, split=1)

    #     vis = np.ma.masked_invalid(vis)
    #     return vis, missing_fraction, np.arange(repfb_chanstart, repfb_chanend)

    # except Exception as e:
    #     print(f"Error: {e}")
    #     traceback.print_exc(file=sys.stdout)
    #     sys.exit(1)

    # finally:
    #     # Discard Shared Memory
    #     matft_shm.close()
    #     win_shm.close()

    #     matft_shm.unlink() 
    #     win_shm.unlink()   
    #     print("Shared memory unlinked by main process.")
