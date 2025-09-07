# Philippe Joly 2025-09-07

# This script is designed to rebin frequency data from ALBATROS and integrate over times dictated by the plasma frequency bins
# This is meant to process a list of frequency bins and time intervals, and visualise the results


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from typing import List, Tuple, Optional

from os import path
import sys
sys.path.insert(0,path.expanduser("~"))

from albatros_analysis.scripts.xcorr.helper import get_init_info_all_ant
from albatros_analysis.scripts.xcorr.helper_gpu import *


global plasma_t_diff = 5*60 # (s) 5 minutes b/w measurements <-- to do: get from data

def get_plasma_freq(nmf2: np.ndarray) -> np.ndarray:
    eps = 8.8541878188 * 1e-12 # vacuum electric permittivity  [ F / m]
    m_e = 9.1093837139 * 1e-31 # electron mass [ kg ]
    q_e = 1.602176634  * 1e-19 # electron charge [ C ]

    w_p = np.sqrt( nmf2 * q_e**2 / (m_e * eps) )
    return w_p / (2*np.pi)

def get_plasma_data(plasma_path: str, obs_period: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        achaim = pd.read_csv(plasma_fn)
    except Exception as e:
        print(f"Error loading plasma frequency file {plasma_fn}: {e}")
        raise e
    
    achaim["datetime_fmt"] = pd.to_datetime(achaim["datetime"], format="%y%m%d_%H%M%S")
    achaim["datetime"] = achaim["datetime_fmt"].astype('int')//10**9
    achaim = achaim[(achaim['datetime'] >= obs_period[0]) & (achaim['datetime'] <= obs_period[1])].copy()
    print(f"Loaded {len(achaim)} Plasma Data points from {pd.to_datetime(obs_period[0], unit='s')} to {pd.to_datetime(obs_period[1], unit='s')}")

    achaim["plasma_freq"] = get_plasma_freq(achaim["nmf2"])
    plasma = achaim["plasma_freq"].to_numpy()
    time = achaim["datetime"].to_numpy() 

    return time, plasma

def bin_plasma_data(all_time: np.ndarray, all_plasma: np.ndarray, bin_num: int, plot_bins_path: str = None) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Bin plasma data into specified number of bins, and group by time intervals where we have continuous achaim data.
    returns 
        time = [ [t0, t1, t2, ...], [tN, tN+1, ...], ... ]  (list of lists of time intervals)
        plasma = [ [p0, p1, p2, ...], [pN, pN+1, ...], ... ] (list of lists of plasma bins corresponding to time intervals)
            this is such that plasma[i][j] is the plasma bin index from time[i][j] to time[i][j+1]
        bin_edges = [edge0, edge1, edge2, ...] (edges of the plasma bins)
    """

    plasma_t_diff = np.min(np.diff(all_time))
    assert plasma_t_diff == 5*60, "plasma data time difference is not 5 minutes, please check" # this is hardvcoded for now
    all_times -= plasma_t_diff/2 # shift times by half interval to have plasma value at the middle of the interval

    counts, bin_edges = np.histogram(all_plasma, bins=bin_num)
    plasma_idx = np.digitize(all_plasma, bin_edges, right=True) - 1 # -1 to make bins start at 0

    time  = []
    plasma = []
    for i, t in enumerate(all_time):
        if i==0 or t > time[-1][-1]:
            time.append([t, t+plasma_t_diff])
            plasma.append([plasma_idx[i]])
        else:
            plasma[-1].append(plasma_idx[i])
            time[-1].append(t+plasma_t_diff)

    if plot_bins_path is not None:
        plt.figure(figsize=(12,10))
        plt.hist(plasma, bins=bin_num, edgecolor='black', alpha=0.7)
        plt.title('Histogram of Plasma Frequencies')
        plt.xlabel('Plasma Frequency (Hz)')
        plt.ylabel('Count of 5 min intervals')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(plot_bins_path)
        plt.close()

    return time, plasma, bin_edges

def repfb_xcorr_bin_avg(init_t: int, time: List[int], plasma: List, bin_num: int, idxs: List[int], files: List[str], acclen: int,  nblock: int, 
                   chanstart: int, chanend: int, osamp: int, cut: int = 10, filt_thresh: float = 0.45,
                   window: Optional[cp.ndarray] = None, filt: Optional[cp.ndarray] = None, 
                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, cp.ndarray, cp.ndarray]:
    """
    Perform oversampling PFB, GPU-based cross-correlation on streaming baseband data, and average over plasma frequency bins.
    
    Args:
        time: List of time intervals (in seconds since epoch)
        plasma: List of plasma frequency bin indices corresponding to time intervals
        bin_num: Total number of plasma frequency bins
        idxs: List of antenna indices
        files: List of baseband files per antenna
        acclen: Accumulation length in units of 4096-sample IPFB output blocks
        nchunks: Total number of IPFB output chunks to process
        nblock: Number of PFB blocks per iteration (streamed)
        chanstart, chanend: Frequency channel bounds
        osamp: Oversampling factor for RePFB
        cut: Number of rows to cut from top and bottom of IPFB
        filt_thresh: Regularization parameter for IPFB deconvolution filter
        window: Precomputed PFB window (optional)
        filt: Precomputed IPFB filter (optional)
        verbose: Enable verbose logging
    
    Returns:
        vis: Averaged cross-correlation matrix per time and frequency
        missing_fraction: Fraction of missing data per chunk and antenna
        freqs: Array of final frequency bins processed
        window: PFB window used
        filt: IPFB filter used
    """

    # Setup configuration
    nant = len(idxs)
    config = ProcessingConfig(
        acclen=acclen, pfb_size=0, nchunks=nchunks, nblock=nblock,
        chanstart=chanstart, chanend=chanend, osamp=osamp, nant=nant, cut=cut, filt_thresh=filt_thresh
    )
    sizes = BufferSizes.from_config(config)
    
    # Setup components
    antenna_objs, channels = setup_antenna_objects(idxs, files, config)
    cupy_win_big, filt = setup_filters_and_windows(config, window, filt)
    
    # Initialize managers
    buffer_mgr = BufferManager(config, sizes)
    ipfb_processor = IPFBProcessor(config, buffer_mgr, channels, filt)
    missing_tracker = MissingDataTracker(nant, sizes.num_of_pfbs)
    
    # Setup frequency ranges
    repfb_chanstart = channels[config.chanstart] * osamp
    repfb_chanend = channels[config.chanend] * osamp
    
    # Initialize output arrays
    xin = cp.empty((config.nant * config.npol, nblock, sizes.nchan), dtype='complex64', order='F')
    vis = np.zeros((bin_num, config.nant * config.npol, nant * config.npol, sizes.nchan), dtype="complex64", order="F")
    
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    
    if verbose:
        _print_debug_info(config, sizes, buffer_mgr, sizes.num_of_pfbs, nchunks)
    
    time_track = init_t
    cur_t_interval_idx = 0
    
    job_idx = 0


    for i, chunks in enumerate(zip(*antenna_objs)):
        pfbed = False

        for ant_idx in range(config.nant):
            chunk = chunks[ant_idx]

            pfb_blocks_affected = ((buffer_mgr.pfb_idx[ant_idx, 0] + sizes.lchunk) // sizes.lblock) % config.nblock + 1
            missing_tracker.add_chunk_info(ant_idx, chunk, config.acclen, pfb_blocks_affected)

            for pol_idx in range(config.npol):
                ipfb_processor.process_chunk(chunk, ant_idx, pol_idx, start_specnums[ant_idx])
                
    
        # All PFB idices should be the same after processing all antennas
        assert (buffer_mgr.pfb_idx == buffer_mgr.pfb_idx[0,0]).all()

        # We know that all antennas/pols have the same PFB index after processing so is sufficient to check one
        while buffer_mgr.pfb_idx[0,0] == sizes.szblock:
            pfbed = True
            # Generate PFB output for cross-correlation
            for ant_idx in range(config.nant):
                
                missing_tracker.calculate_job_average(ant_idx, job_idx)

                for pol_idx in range(config.npol):
                    output_start = ant_idx * config.nant + pol_idx # <-- still not sure
                    xin[output_start, :, :] = pu.cupy_pfb(
                        buffer_mgr.pfb_buf[ant_idx, pol_idx], 
                        cupy_win_big,
                        nchan=2048 * osamp + 1, 
                        ntap=4
                    )[:, repfb_chanstart:repfb_chanend]

            vis[:, :, :, job_idx] = cp.asnumpy(
                cr.avg_xcorr_all_ant_gpu(xin, config.nant, config.npol, nblock, sizes.nchan, split=1)
            )
            buffer_mgr.reset_overlap_region()
            for ant_idx in range(config.nant):
                for pol_idx in range(config.npol):
                    # Add remaining data to PFB buffer
                    buffer_mgr.add_remaining_to_pfb_buffer(ant_idx, pol_idx)
            job_idx += 1

        
        if verbose and job_idx % 100 == 0:
            print(f"Job Chunk {job_idx + 1}/{sizes.num_of_pfbs} s")
     # Pad buffer if incomplete

    if buffer_mgr.pfb_idx[0,0] > (config.ntap - 1)* sizes.lblock:

        print(f"Job {job_idx + 1}: ")
        print(f"Buffer not Full with only {buffer_mgr.pfb_idx[0,0]} < {sizes.szblock}")

        for ant_idx in range(config.nant):
                
            missing_tracker.calculate_job_average(ant_idx, job_idx)

            for pol_idx in range(config.npol):
                buffer_mgr.pad_incomplete_buffer(ant_idx, pol_idx)
                output_start = ant_idx * config.nant + pol_idx # <-- still not sure
                xin[output_start, :, :] = pu.cupy_pfb(
                    buffer_mgr.pfb_buf[ant_idx, pol_idx], 
                    cupy_win_big,
                    nchan=2048 * osamp + 1, 
                    ntap=4
                )[:, repfb_chanstart:repfb_chanend]
        
        vis[:, :, :, job_idx] = cp.asnumpy(
            cr.avg_xcorr_all_ant_gpu(xin, config.nant, config.npol, nblock, sizes.nchan, split=1)
        )

        print("Paded and Processed Incomplete Buffer")
    
    if verbose:
        print("=" * 30)
        print(f"Completed {sizes.num_of_pfbs}/{sizes.num_of_pfbs} Job Chunks")
        print("=" * 30)
    
    vis = np.ma.masked_invalid(vis)
    freqs = np.arange(repfb_chanstart, repfb_chanend)
    
    return vis, missing_tracker.missing_fraction, freqs, cupy_win_big, filt

def main(plot=False):
    config_fn = "visual_config.json"
    if len(sys.argv) > 1:
        config_fn = sys.argv[1]  
    
    try:
        with open(config_fn, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_fn}: {e}")
        raise e

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
    acclen = config["correlation"]["acclen"]
    cut = config["correlation"]["cut"]
    nblock = config["correlation"]["nblock"]
    bin_num = config["correlation"]["bin_num"]

    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]

    plasma_path = config["misc"]["plasma_path"]
    outdir = config["misc"]["out_dir"]

    obs_period = (1721342139, 1721449881) # Full observation period
    # obs_period = [1721342139, 1721376339] # First 'continuous' chunk of observation period
    # obs_period = [1721376339, 1721449881] # Second 'continuous' chunk of observation period
    # To do: get from json

    all_time, all_plasma = get_plasma_data(plasma_path, obs_period)
    binned_time, binned_plasma, bin_edges = bin_plasma_data(all_time, all_plasma, bin_num, plot_bins_path=path.join(outdir, "plasma_bin_hist.png"))

    for i in range(len(binned_time)):
        print(f"Processing continuous chunk {i+1}/{len(binned_time)} over {len(binned_time[i])*5} minutes")
        window, filt = None, None
        avg_vis, missing_frac, counts, window, filt = repfb_xcorr_bin_avg(
            binned_time[i], binned_plasma[i], bin_num,
            list(range(len(dir_parents))), dir_parents, acclen, nblock,
            chanstart, chanend, osamp, cut=cut, filt_thresh=0.45,
            window=window, filt=filt, verbose=False
        )


if __name__=="__main__":

    all_time = np.append(np.arange(5)*5*60, np.arange(10)*5*60+1700000000)
    all_plasma = np.random.uniform(0, 100, size=len(all_time))
    bin_num = 10
    print("All times:", all_time)
    print("All plasma freqs:", all_plasma)
    print(30*'=')

    time, plasma, bin_edges = bin_plasma_data(all_time, all_plasma, bin_num, plot_bins=False)
    for i in range(len(time)):
        print("Time bins:", time[i])
        print("Plasma bins:", plasma[i])
        print(20*'-')
    print("Bin edges:", bin_edges)

    