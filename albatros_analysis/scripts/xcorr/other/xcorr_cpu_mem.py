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

import numpy as np
import time
import argparse
import os
from os import path
import sys
import helper
import helper_cpu_mem
sys.path.insert(0,path.expanduser("~"))
import json

if __name__=="__main__":

    # Processing Unit
    pUnit = 'cpu' 

    node_id = int(os.environ['NODE_ID'])
    chunk_start = int(os.environ['CHUNK_START'])
    chunk_end = int(os.environ['CHUNK_END'])
    output_dir = os.environ['OUTPUT_DIR']
    config_fn = os.environ["CONFIG_FN"]
    
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
    cutsize = 16
    
    assert pfb_mult >= 8 


    cut=int(pfb_size/cutsize)
    acclen=pfb_size - 2*cut
    # nchunks = int(np.floor((end_t-init_t)*250e6/4096/acclen))
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    # print("final idxs", idxs)
    # print("nchunks", nchunks)
    # print("loaded files", files)
    # print("IPFB ROWS", pfb_size)
    # print("OSAMP", osamp)

    print(f"Python processing: Node {node_id}, chunks {chunk_start}-{chunk_end-1}")


    t1=time.time()

    try:
        output_file = helper_cpu_mem.process_node_range(
            idxs=idxs,
            files=files,
            pfb_size=pfb_size,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            chanstart=chanstart,
            chanend=chanend,
            osamp=osamp,
            cutsize=cutsize,
            filt_thresh=0.45,
            n_cores=None,  # Use all available cores
            node_id=node_id,
            output_dir=output_dir
        )
        
        print(f"Node {node_id} completed successfully!")
        print(f"Output files: {output_file}_*.npz")
        
    except Exception as e:
        print(f"Error on node {node_id}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    t2=time.time()
    
    print(f"Node {node_id} Processing took {t2-t1} s")