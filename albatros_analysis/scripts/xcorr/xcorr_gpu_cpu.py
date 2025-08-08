# Philippe Joly 2025-06-03

# This script is designed to rebin frequency data from ALBATROS using either CPU cores (in parrallel) or a GPU
# Most of the Script was taken from the albatros_analysis repository. 
# I added the CPU functionality.

"""
Usage:
    First adjust the corresponding config files (config_{pUnit}_axion.json) with the desired parameters.
    Also, if the machine has both cpu as well as gpu and cpu wants to be used adjust the ./albatros_analysis/__init__.py
    such that xp = numpy
     
    Run
        python xcorr_gpu_cpu.py <processing unit (gpu/cpu)>
    [Defaults to cpu]
"""

# Functionality where sees no gpu and defaults to cpu could be added

import numpy as np
import time
import argparse
from os import path
import os
import sys
import helper
sys.path.insert(0,path.expanduser("~"))
import json

if __name__=="__main__":

    # Processing Unit
    pUnit = 'cpu' 
    if len(sys.argv) > 1 and sys.argv[1].lower()=='gpu':
        pUnit = 'gpu' 

    n_cores = None
    # if len(sys.argv) > 2 and sys.argv[1].lower()=='cpu':
    #     n_cores = int(sys.argv[2]) 

    config_fn = "config_axion_gpu.json"
    if len(sys.argv) > 2:
        config_fn = sys.argv[2]  
    
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
    if pUnit == "gpu":
        # from helper_gpu_stream import repfb_xcorr_avg
        from helper_gpu_stream_clean import repfb_xcorr_avg
        
        # pols,missing_fraction,channels=repfb_xcorr_avg(idxs,files,pfb_size,nchunks,chanstart,chanend,osamp,cutsize,filt_thresh=0.45)

        pols,missing_fraction,channels=repfb_xcorr_avg(idxs,files,acclen,nchunks,3, chanstart,chanend,osamp,cut=cut,filt_thresh=0.45)

        # raise ValueError
    else:
        import helper_cpu
        os.environ['NUMBA_OPT']='3'
        os.environ['NUMBA_LOOP_VECTORIZE']='1'
        os.environ['NUMBA_ENABLE_AVX']='1'
        os.environ['NUMBA_CACHE_DIR'] = '/scratch/s/sievers/philj0ly/tmp/my_numba_cache'

        pols,missing_fraction,channels=helper_cpu.repfb_xcorr_avg(idxs,files,pfb_size,nchunks,chanstart,chanend,osamp,cutsize,filt_thresh=0.45, n_cores=n_cores)

    t2=time.time()
    
    print("Processing took", t2-t1, "s")

    fname = f"stream_xcorr_all_ant_4bit_{str(init_t)}_{str(acclen)}_{str(osamp)}_{str(nchunks)}_{chanstart}_{chanend}.npz"
    fGPUname = f"stream_xcorr_all_ant_4bit_{str(init_t)}_{str(acclen)}_{str(osamp)}_{str(njobs)}_{chanstart}_{chanend}.npz"
    fpath = path.join(outdir,fname)
    np.savez(fpath,data=pols.data,mask=pols.mask,missing_fraction=missing_fraction,chans=channels)

    print("\nSaved", chanend-chanstart, "channels over", end_t-init_t, "s")
    print("with an oversampling rate of", osamp, "at")
    print(fpath)