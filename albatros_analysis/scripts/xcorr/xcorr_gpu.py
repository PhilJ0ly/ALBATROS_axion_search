# Philippe Joly 2025-09-10

# This script is designed to rebin frequency data from ALBATROS using the GPU

"""
Usage:
    First adjust the corresponding config files (config_gpu_axion.json) with the desired parameters.
     
    Ensure that environment variable USE_GPU is set to '1' for GPU usage.
    Run
        python xcorr_gpu.py <parameter file path> <out directory>
"""

import numpy as np
import time
import argparse
from os import path
import os
import sys
sys.path.insert(0,path.expanduser("~"))
import json

import helper
from helper_gpu import repfb_xcorr_avg        

def main():
    # Could add a check that GPU is there

    config_fn = f"config_axion_gpu.json"
    if len(sys.argv) > 1:
        config_fn = sys.argv[1]   
    
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

    osamp_ls = config["correlation"]["osamp"]
    ipfb_acclen = config["correlation"]["acclen"]
    nblock_ls = config["correlation"]["nblock"]
    cut = config["correlation"]["cut"]

    if len(sys.argv) > 2:
        outdir = sys.argv[2]
    else:
        outdir = config["misc"]["out_dir"]
    
    
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/ipfb_acclen))
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    
    print("final idxs", idxs)
    print("nchunks", nchunks)

    for i in range(len(osamp_ls)):
        osamp = osamp_ls[i]
        nblock = nblock_ls[i]
        print("OSAMP", osamp, "NBLOCK", nblock)

        # To do: update time measure
        t1=time.time()
        pols,missing_fraction,channels, _, _ =repfb_xcorr_avg(idxs,files,ipfb_acclen,nchunks,nblock, chanstart,chanend,osamp,cut=cut,filt_thresh=0.45)
        t2=time.time()
        
        print("Total Processing took", t2-t1, "s")

        fname = f"all_ant_4bit_{str(init_t)}_{str(ipfb_acclen)}_{str(osamp)}_{str(nblock)}_{chanstart}_{chanend}.npz"
        fpath = path.join(outdir,fname)
        np.savez(fpath,data=pols.data,mask=pols.mask,missing_fraction=missing_fraction,chans=channels)

        print("\nSaved", chanend-chanstart, "channels over", end_t-init_t, "s")
        print("with an oversampling rate of", osamp, "at")
        print(fpath)


if __name__=="__main__":
    main()