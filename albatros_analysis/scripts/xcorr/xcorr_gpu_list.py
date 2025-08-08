# Philippe Joly 2025-08-06

# This script is designed to rebin frequency data from ALBATROS using a GPU
# This is meant to process a list of frequency bins and time intervals


import numpy as np
import pandas as pd
import time
import argparse
from os import path
import os
import sys
import helper
from helper_gpu_stream import repfb_xcorr_avg
sys.path.insert(0,path.expanduser("~"))
import json

if __name__=="__main__":
    config_fn = "config_axion_gpu.json"
    if len(sys.argv) > 2:
        config_fn = sys.argv[2]  
    
    with open(config_fn, "r") as f:
        config = json.load(f)

    ref_ant = min(
        config["antennas"].keys(),
        key=lambda ant: config["antennas"][ant]["clock_offset"],
    )
    dir_parents = []
    spec_offsets = []
    for i, (ant, details) in enumerate(config["antennas"].items()):
        print(ref_ant, ant, details)
        dir_parents.append(details["path"])
        spec_offsets.append(details["clock_offset"])
    
    osamp = config["correlation"]["osamp"]
    outdir = f"/project/s/sievers/philj0ly/xcorr_gp0"   
    acclen = config["correlation"]["ipfb_acclen"]
    cut = config["correlation"]["cut"]
    nblock = config["correlation"]["nblock"]

    albatros_time_bounds = [1721342139, 1721750623]


    to_process = pd.read_csv("/scratch/s/sievers/philj0ly/Jupyter/processing_group0_times_1721250090_1721411910.csv").to_numpy()

    process_num = to_process.shape[0]
    print("Starting to Process", process_num, "Processes...\n" )

    cupy_win_big = None
    filt = None

    t1=time.time()
    count = 0
    for i, process in enumerate(to_process):
        init_t = process[1]
        end_t = process[2]

        if init_t < albatros_time_bounds[0] or end_t > albatros_time_bounds[1]:
            continue

        nchunks = int(np.floor((end_t-init_t)*250e6/4096/acclen))
        init_t = str(init_t)
        end_t = str(end_t)

        chanstart = process[0]
        chanend = chanstart + 1
        print(init_t, end_t)

        idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)

        pols, missing_fraction, channels, cupy_win_big, filt = repfb_xcorr_avg(
            idxs,
            files,
            acclen,
            nchunks,
            nblock, 
            chanstart,
            chanend,
            osamp,
            cut=cut,
            filt_thresh=0.45,
            cupy_win_big=cupy_win_big,
            filt=filt
        )

        fname = f"stream_xcorr_all_ant_4bit_{str(init_t)}_{str(acclen)}_{str(osamp)}_{str(nchunks)}_{chanstart}_{chanend}.npz"

        fpath = path.join(outdir,fname)
        np.savez(fpath, data=pols.data, mask=pols.mask, missing_fraction=missing_fraction, chans=channels)

        print(f"{i+1}/{process_num} Processed {((chanstart+64)*61e3/1e6):.3f} MHz at {pd.to_datetime(init_t, unit='s')}")
        count +=1
        if count>=5:
            break

    t2=time.time()
    print(f"\nProcessing Completed in {(t2-t1):.2f} s")

