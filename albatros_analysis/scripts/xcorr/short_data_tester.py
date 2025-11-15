import numpy as np
import cupy as cp
import helper

import sys
from os import path
sys.path.insert(0, path.expanduser("~"))

from albatros_analysis.src.utils import pfb_gpu_utils as pu
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr

def setup_filters_and_windows(ntap, acclen, filt_thresh, osamp, cupy_win_big: Optional[cp.ndarray] = None, filt: Optional[cp.ndarray] = None) -> Tuple[cp.ndarray, cp.ndarray]:
    """Set up PFB window and IPFB filter"""
    if cupy_win_big is None:
        dwin = pu.sinc_hamming(ntap, 4096 * osamp)
        cupy_win_big = cp.asarray(dwin, dtype='float32', order='c')
    
    if filt is None:
        matft = pu.get_matft(acclen + 2*cut)
        filt = pu.calculate_filter(matft, filt_thresh)
    
    return cupy_win_big, filt

def main():
    config_fn = f"config_axion_gpu.json" 
    
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
    ipfb_acclen = config["correlation"]["acclen"]
    nblock = config["correlation"]["nblock"]
    cut = config["correlation"]["cut"]
    filt_thresh = config["correlation"]["filt_thresh"]

    if len(sys.argv) > 2:
        outdir = sys.argv[2]
    else:
        outdir = config["misc"]["out_dir"]
    
    
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/ipfb_acclen))
    idxs, files = helper.get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    
    antenna_objs = []
    for i, (idx, file) in enumerate(zip(idxs, files)):
        antenna = bdc.BasebandFileIterator(
            file, 0, idx, config.acclen, nchunks=config.nchunks,
            chanstart=config.chanstart, chanend=config.chanend, type='float'
        )
        antenna_objs.append(antenna)
    
    # Get channel information from first antenna
    channels = np.asarray(antenna_objs[0].obj.channels, dtype='int64')
    channel_idxs = antenna_objs[0].obj.channel_idxs

    window, filt = setup_filters_and_windows(4, ipfb_acclen, filt_thresh, osamp)

    assert nchunks > 20, "nchunks must be > 20 for this test"

    
    