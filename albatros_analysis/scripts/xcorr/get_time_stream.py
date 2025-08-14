# Philippe Joly 2025-08-14

# This script is designed to get time streams from ALBATROS data.

import numpy as np
import time
import argparse
from os import path
import os
import sys
sys.path.insert(0,path.expanduser("~"))
import json

import helper
from helper_stream import get_time_stream

if __name__=="__main__": 

    config_fn = "config_read_test.json"
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
    
    outdir = f"/scratch/philj0ly/simple_test_data/timeStream"

    acclen = 2048*4
    cut = 100

    nchunks = int(np.floor((end_t-init_t)*250e6/4096/acclen))
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    
    # print("final idxs", idxs)
    print("nchunks", nchunks)
    print("Acclen", acclen)
    print("Channels", chanstart, 'to', chanend, 'for', end_t-init_t, 's')

    print("Beginning Processing on GPU...\n")

    t1=time.time()       

       
    time_stream, missing_fraction, _ = get_time_stream(idxs,files,acclen,nchunks, chanstart,chanend,cut=cut,filt_thresh=0.45)
    
    t2=time.time()
    
    print("Processing took", t2-t1, "s")

    fname = f"time_stream_{str(init_t)}-{str(end_t)}_{str(acclen)}_{str(nchunks)}_{chanstart}-{chanend}.npz"
    fpath = path.join(outdir,fname)
    np.savez(fpath,data=time_stream, missing_fraction=missing_fraction, chans=channels)

    print("\nSaved", chanend-chanstart, "channels over", end_t-init_t, "s")
    print("with an accumulation length of", acclen, "at")
    print(fpath)