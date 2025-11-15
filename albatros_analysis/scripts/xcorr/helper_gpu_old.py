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
    print("pfb_size", pfb_size, "nchunks", nchunks, "osamp", osamp)
    # pu.print_mem("Start Helper")
    nant = len(idxs)
    npol = 2

    # ----------------- START IPFB SETUP -----------------------#\
    
    cut=int(pfb_size/cutsize)
    acclen=pfb_size - 2*cut
    ntap=4
    nn=2*2048*osamp

    print("cut", cut,"acclen", acclen, "lblbock", nn)
    assert acclen%osamp == 0

    re_pfb_size = acclen//osamp - ntap + 1
    assert re_pfb_size > 0 
    print("re_pf_size", re_pfb_size)
    
    dwin=pu.sinc_hamming(ntap,nn)
    cupy_win_big=cp.asarray(dwin,dtype='float32',order='c')
    

    # CPU version if not enough memory
        # dwin=pu.sinc_hamming(ntap,nn)
        # cupy_win_big=cp.asarray(dwin,dtype='float32',order='c')

        # matft = cp.asnumpy(pu.get_matft(pfb_size))
        # filt = cp.asarray(pu.compute_filter(matft, filt_thresh))

    matft = pu.get_matft(pfb_size)
    filt = pu.calculate_filter(matft, filt_thresh)



    # matft=pu.get_matft(pfb_size)
    to_ipfb_pol0 = cp.empty((pfb_size,2049),dtype='complex64', order='C') 
    to_ipfb_pol1 = to_ipfb_pol0.copy()
    cut_chunks = cp.zeros((nant, npol, 2*cut, 2049),dtype='complex64', order='C') # need to maintain last chunks for each antenna
    print("to ipfb pol0 shape",to_ipfb_pol0.shape)
    # print("matft shape", matft.shape)

    # print("cupy_win_big size", np.prod(cupy_win_big.shape)*4/1024**3, "GB")
    # print("matft size", np.prod(matft.shape)*8/1024**3, "GB")
    # print("2x to_ipfb size", np.prod(to_ipfb_pol0.shape)*8*2/1024**3, "GB")
    # print("cut_chunks size", np.prod(cut_chunks.shape)*8*2/1024**3, "GB")
    # print("acclen", acclen, "pfb_size", pfb_size)

    print("win size", cupy_win_big.shape)
    print("filt shape", filt.shape)
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
    print("start and end chans are", repfb_chanstart, repfb_chanend)
    # sys.exit()
    # print("start and end are", )
    split = 1
    print("nant", nant, "nchunks", nchunks, "nchan", nchan)
    
    vis = np.zeros((nant*npol, nant*npol, nchan, nchunks), dtype="complex64", order="F")
    xin = cp.empty((nant*npol, re_pfb_size, nchan),dtype='complex64',order='F')
    

    scratch = cp.empty((nant*npol,nant*npol,nchan*split),dtype='complex64',order='F')
    missing_fraction = np.zeros((nant, nchunks), dtype='float64', order='F')
    # missing_mask = cp.empty((nant,nchunks),dtype='float32',order='F') #all integers. float is ok. cuda doesnt like integers.

    # print("vis size", np.prod(vis.shape)*8/1024**2, "MB")
    # print("scratch size", np.prod(scratch.shape)*8/1024**2, "MB")
    # print("xin size", np.prod(xin.shape)*8/1024**2, "MB")
    print("vis",vis.shape,"xin", xin.shape,"scratch", scratch.shape)

    # pu.print_mem("Before Processing")

    # rowcounts = np.empty(nchunks, dtype="int64")
    start_specnums = [ant.spec_num_start for ant in antenna_objs] #specnums are always on the host. so numpy is OK.
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    tot = 0
    count = 0
    for i, chunks in enumerate(zip(*antenna_objs)):
        t0 = time.perf_counter()
        # if i%10==0: print(i)
        start_event.record()
        for j in range(nant):
            chunk=chunks[j]
            start_specnum = start_specnums[j]
            pol0=bdc.make_continuous_gpu(chunk['pol0'],chunk['specnums'],start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)
            pol1=bdc.make_continuous_gpu(chunk['pol1'],chunk['specnums'],start_specnum,channels[aa.obj.channel_idxs],acclen,nchans=2049)
           
            perc_missing = (1 - len(chunk["specnums"]) / acclen) * 100
            missing_fraction[j, i] = perc_missing
            # CHUNK 0 is going to have all zeros at the top. Ignore first chunk in the saved output.
            to_ipfb_pol0[:2*cut] = cut_chunks[j,0,:,:] # <---- zeros for CHUNK 0
            to_ipfb_pol0[2*cut:] = pol0
            to_ipfb_pol1[:2*cut] = cut_chunks[j,1,:,:] # <---- zeros for CHUNK 0
            to_ipfb_pol1[2*cut:] = pol1

            # if count%100 == 0 : 
            #     print("to ipfb mean", np.mean(to_ipfb_pol0))

            raw_pol0 = pu.cupy_ipfb(to_ipfb_pol0, filt)
            raw_pol1 = pu.cupy_ipfb(to_ipfb_pol1, filt)

            pol0_new = pu.cupy_pfb(raw_pol0[cut:-cut],cupy_win_big,nchan=2048*osamp+1,ntap=4)
            pol1_new = pu.cupy_pfb(raw_pol1[cut:-cut],cupy_win_big,nchan=2048*osamp+1,ntap=4)
            
   
            cut_chunks[j,0,:,:] = pol0[-2*cut:,:]
            cut_chunks[j,1,:,:] = pol1[-2*cut:,:]
            xin[j*npol,:,:] = pol0_new[:, repfb_chanstart : repfb_chanend] # BFI data is C-major for IPFB
            xin[j*npol+1,:,:] = pol1_new[:, repfb_chanstart : repfb_chanend] #gotta support arbitrary chans
        
        out=cr.avg_xcorr_all_ant_gpu(xin,nant,npol,re_pfb_size,nchan,split=1,out=scratch)
        end_event.record()
        end_event.synchronize()
        # print("CUDA IPFB+XCORR time ",cp.cuda.get_elapsed_time(start_event, end_event)/1000)
        vis[:,:,:,i] = cp.asnumpy(out) #scratch should still be on the device

        tot += time.perf_counter()-t0
        count += 1
        if count %1000==0:
            print("niter", count, "time", tot/count, "s BW", 2*nant*pol0.nbytes/tot*count/1e6, "MSPS")
            # print(np.mean(xin))
            # print("to_ipfb", to_ipfb_pol0.dtype, "raw_pol", raw_pol0.dtype, "pol_new", pol0_new.dtype, "xin", xin.dtype)

    print("niter", count, "time", tot/count, "s BW", 2*nant*pol0.nbytes/tot*count/1e6, "MSPS")
    vis = np.ma.masked_invalid(vis)
    return vis, missing_fraction, np.arange(repfb_chanstart, repfb_chanend) 


def main():
    # Processing Unit
    pUnit = 'gpu' 

    config_fn = "config_axion_gpu.json"
    
    with open(config_fn, "r") as f:
        config = json.load(f)

    # Determine reference antenna
    ref_ant = min(
        config["antennas"].keys(),
        key=lambda ant: config["antennas"][ant]["clock_offset"],
    )
    dir_parents = []
    spec_offsets = []
    # Call get_starting_index for all antennas except reference
    for i, (ant, details) in enumerate(config["antennas"].items()):
        print(ref_ant, ant, details)
        dir_parents.append(details["path"])
        spec_offsets.append(details["clock_offset"])

    init_t = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    osamp = config["correlation"]["osamp"]
    pfb_mult = config["correlation"]["pfb_size_multiplier"]
    
    pfb_size = osamp*pfb_mult
    outdir = f"/project/s/sievers/philj0ly/xcorr_{pUnit}"
    
    # assert pfb_mult >= 8 

    
    cutsize = 16
    cut=int(pfb_size/cutsize)
    # cut = 10

    acclen=pfb_size - 2*cut

    acclen = 2048
    cut = 100

    nchunks = int(np.floor((end_t-init_t)*250e6/4096/acclen))
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    print("final idxs", idxs)
    print("nchunks", nchunks)
    print("IPFB ROWS", pfb_size)
    print("OSAMP", osamp)

    print("Processing Unit", pUnit)
    print("Beginning Processing...\n")

    t1=time.time()
    pols,missing_fraction,channels=repfb_xcorr_avg(idxs,files,pfb_size,nchunks,chanstart,chanend,osamp,cutsize,filt_thresh=0.45)

      
    t2=time.time()
    
    print("Processing took", t2-t1, "s")

    fname = f"stream_xcorr_all_ant_4bit_{str(init_t)}_{str(acclen)}_{str(osamp)}_{str(nchunks)}_{chanstart}_{chanend}_old.npz"
    fpath = path.join(outdir,fname)
    np.savez(fpath,data=pols.data,mask=pols.mask,missing_fraction=missing_fraction,chans=channels)

    print("\nSaved", chanend-chanstart, "channels over", end_t-init_t, "s")
    print("with an oversampling rate of", osamp, "at")
    print(fpath)

if __name__=="__main__":
    main()