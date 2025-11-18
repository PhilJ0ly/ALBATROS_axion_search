# Philippe Joly 2025-09-07

# This script is designed to rebin frequency data from ALBATROS and integrate over times dictated by the plasma frequency bins
# This is meant to process a list of frequency bins and time intervals, and visualise the results


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import json
import cupy as cp
from typing import List, Tuple, Optional
import time

from os import path
import sys
sys.path.insert(0,path.expanduser("~"))

from albatros_analysis.scripts.xcorr.helper import get_init_info_all_ant
from albatros_analysis.scripts.xcorr.helper_gpu import *


def get_plasma_freq(nmf2: np.ndarray) -> np.ndarray:
    eps = 8.8541878188 * 1e-12 # vacuum electric permittivity  [ F / m]
    m_e = 9.1093837139 * 1e-31 # electron mass [ kg ]
    q_e = 1.602176634  * 1e-19 # electron charge [ C ]

    w_p = np.sqrt( nmf2 * q_e**2 / (m_e * eps) )
    return w_p / (2*np.pi)

def get_plasma_data(plasma_path: str, obs_period: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        achaim = pd.read_csv(plasma_path)
    except Exception as e:
        print(f"Error loading plasma frequency file {plasma_path}: {e}")
        raise e
    
    achaim["datetime_fmt"] = pd.to_datetime(achaim["datetime"], format="%y%m%d_%H%M%S")
    achaim["datetime"] = achaim["datetime_fmt"].astype('int')//10**9
    achaim = achaim[(achaim['datetime'] >= obs_period[0]) & (achaim['datetime'] <= obs_period[1])].copy()
    
    assert len(achaim)>0, f"No Data points from {pd.to_datetime(obs_period[0], unit='s')} to {pd.to_datetime(obs_period[1], unit='s')}"
    print(f"Loaded {len(achaim)} Plasma Data points from {pd.to_datetime(obs_period[0], unit='s')} to {pd.to_datetime(obs_period[1], unit='s')}")
    
    achaim["plasma_freq"] = get_plasma_freq(achaim["nmf2"])
    plasma = achaim["plasma_freq"].to_numpy()
    time = achaim["datetime"].to_numpy() 

    return time, plasma

def split_gaps(times: List[np.int64], plasma: List[int], gaps: List[Tuple[int, int]]):
    """
    Split time and plasma lists into continuous chunks based on data gaps.
    
    Parameters:
    times : list
        List of timestamps 
    plasma : list
        List of plasma values where plasma[i] corresponds to interval times[i] -> times[i+1]
        Length should be len(times) - 1
    gaps : list of tuples
        List of (gap_start, gap_duration) tuples defining data gaps
        
    Returns:
    time_chunks : list of lists
        List of continuous time chunks
    plasma_chunks : list of lists
        List of corresponding plasma chunks
     
    """
    
    if len(plasma) != len(times) - 1:
        raise ValueError(f"Plasma length ({len(plasma)}) must be len(times) - 1 ({len(times) - 1})")
    
    # Track which indices to keep and where to split
    time_chunks = []
    current_chunk = []
    kept_indices = []  # indices in original times list that we keep
    
    for i, t in enumerate(times):
        # Check if this timestamp falls within any gap
        exclude = False
        for gap_start, gap_duration in gaps:
            gap_end = gap_start + gap_duration
            if gap_start <= t <= gap_end:
                exclude = True
                break
        
        if exclude:
            if current_chunk:
                time_chunks.append(current_chunk)
                current_chunk = []
        else:
            # Check if this timestamp comes after a gap
            should_break = False
            if current_chunk:
                for gap_start, gap_duration in gaps:
                    gap_end = gap_start + gap_duration
                    last_t = times[kept_indices[-1]]
                    if last_t < gap_start and t > gap_end:
                        should_break = True
                        break
            
            if should_break and current_chunk:
                time_chunks.append(current_chunk)
                current_chunk = [t]
                kept_indices.append(i)
            else:
                current_chunk.append(t)
                kept_indices.append(i)
    
    if current_chunk:
        time_chunks.append(current_chunk)
    
    # Now split plasma list based on intervals
    # plasma[i] corresponds to interval times[i] -> times[i+1]
    plasma_chunks = []
    chunk_start_idx = 0
    
    for chunk in time_chunks:
        chunk_len = len(chunk)
        # For each time chunk, we have chunk_len - 1 plasma intervals
        # Only keep chunks with at least 2 timestamps (1+ plasma intervals)
        if chunk_len > 1:
            plasma_chunk = []
            for i in range(chunk_len - 1):
                # Find the original index for this time point
                original_idx = kept_indices[chunk_start_idx + i]
                plasma_chunk.append(plasma[original_idx])
            
            plasma_chunks.append(plasma_chunk)
        
        chunk_start_idx += chunk_len
    
    # Filter out time chunks with only 1 timestamp
    time_chunks = [chunk for chunk in time_chunks if len(chunk) > 1]
    
    return time_chunks, plasma_chunks

def bin_plasma_data(all_time: np.ndarray, all_plasma: np.ndarray, bin_num: int, plot_bins_path: str = None, split_for_gaps: bool = False) -> Tuple[List[List[int]], List[List[float]]]:
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
    all_time -= int(plasma_t_diff/2) # shift times by half interval to have plasma value at the middle of the interval

    counts, bin_edges = np.histogram(all_plasma, bins=bin_num)

    bin_edges[0] -= 1e-6  # ensure min value is included
    bin_edges[-1] += 1e-6 # ensure max value is included

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
        plt.hist([item for sub_plasma in plasma for item in sub_plasma], bins=bin_num, align='right', edgecolor='black', alpha=0.7)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.xticks(np.arange(bin_num),labels=[f"{c/1e6:.2f}" for c in bin_centers], rotation=45)
        # plt.xticks(np.arange(bin_num+1))

        plt.title('Histogram of Plasma Frequencies')
        plt.xlabel('Plasma Frequency (MHz)')
        plt.ylabel('Count of 5 min intervals')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(plot_bins_path)
        plt.close()
    # print(plasma)
    # print(bin_num)
    gaps = [
        [
            (1721343875, 42),
            (1721344517, 43),
            (1721346596, 43),
            (1721351039, 42),
            (1721351881, 49),
            (1721360558, 43)
        ],
        [
            (1721381476, 43),
            (1721382716, 42),
            (1721385159, 44),
            (1721401206, 43),
            (1721404804, 206),
            (1721410495, 47),
            (1721411719, 43)
        ],
        # [
        #     (1721639204, 49),
        #     (1721640533, 51)
        # ]
    ]
    
    if split_for_gaps:
        new_time = []
        new_plasma = []

        for i in range(len(gaps)):
            cts_time_chunk, cts_plasma_chunk = split_gaps(time[i], plasma[i], gaps[i])

            new_time.extend(cts_time_chunk)
            new_plasma.extend(cts_plasma_chunk)
        
        time, plasma = new_time, new_plasma
    
    if plot_bins_path is not None:
        plt.figure(figsize=(12,10))
        plt.hist([item for sub_plasma in plasma for item in sub_plasma], bins=bin_num, align='right', edgecolor='black', alpha=0.7)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.xticks(np.arange(bin_num),labels=[f"{c/1e6:.2f}" for c in bin_centers], rotation=45)
        # plt.xticks(np.arange(bin_num+1))

        plt.title('Histogram of Plasma Frequencies')
        plt.xlabel('Plasma Frequency (MHz)')
        plt.ylabel('Count of 5 min intervals')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(plot_bins_path)
        plt.close()
   
   
    return time, plasma, bin_edges

def get_plots(avg_data, bin_edges, missing_fraction, total_counts, chans, 
              t_chunk, osamp, ncols, outdir, 
              log=False, all_stokes=False, band_per_plot=None, bin_counts=None, sharey='none'):

    df_record = 125e6 / 2048  # (Hz) frequency range / # of channels
    df = df_record / osamp    # (Hz) / rechannelization factor
    freqs = chans * df        # frequencies for each channel # for the data we have error slipped in with chans 64 *osamp ahead
    # bw = (bin_edges[1] - bin_edges[0])/2
    scale = 'log' if log else 'linear'

    pdf_name = f"stokes_subbands_log_{log}_{str(osamp)}.pdf"
    pdf_path = path.join(outdir, pdf_name)

    if band_per_plot is None:
        band_limits = [(freqs[0], freqs[-1])]
    else:
        fmin, fmax = freqs[0], freqs[-1]
        band_edges = np.arange(fmin-band_per_plot/2, fmax + band_per_plot, band_per_plot) # <- -bbp/2 to ensure that center of bins (bin numbers) are centered with graph when bbp=61kHz
        band_limits = list(zip(band_edges[:-1], band_edges[1:]))

    n_bins = len(avg_data)
    nrows = int(np.ceil(n_bins / ncols))
    height = min(6*nrows, 20)

    with PdfPages(pdf_path) as pdf:
        # loop over subbands first
        for (f_low, f_high) in band_limits:
            bw = (f_high-f_low)/2
            print(50*'-')
            print(f"Spectrum Band {(f_low+bw)/1e3:.0f}+–{bw/1e3:.0f} kHz")
            mask = (freqs >= f_low) & (freqs < f_high)
            if not np.any(mask):
                continue

            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, sharey=sharey, figsize=(25,height), squeeze=False
            )
            fig.suptitle(
                f"Stokes Parameter I for Spectrum Band {(f_low+bw)/1e3:.0f}+–{bw/1e3:.0f} kHz\n", fontsize=24
            )

            for i, pols in enumerate(avg_data):
                ax = axes[i // ncols, i % ncols]

                I_stokes = (pols[0,0] + pols[1,1]).real.astype("float32")[mask]

                ax.plot(freqs[mask], I_stokes, label='I Stokes', color='blue')

                if all_stokes:
                    Q_stokes = (pols[0,0] - pols[1,1]).real.astype("float32")[mask]
                    U_stokes = (pols[1,0] + pols[0,1])[mask]
                    V_stokes = (pols[1,0] - pols[0,1])[mask]
                    ax.plot(freqs[mask], Q_stokes, label='Q Stokes', color='orange')
                    ax.plot(freqs[mask], np.abs(U_stokes), label='|U Stokes|', color='green')
                    ax.plot(freqs[mask], np.abs(V_stokes), label='|V Stokes|', color='red')

                total_time = total_counts[i] * t_chunk

                ax.set_title(
                    f"Plasma Frequency [{i}] {(bin_edges[i])/1e3:.0f}-{bin_edges[i+1]/1e3:.0f} kHz\n"
                    f"{total_time/60:.1f} min | " 
                    f"Missing fraction: {missing_fraction[i]*100:.1f}%"
                )

                ax.set_xlabel("Spectrum Frequency [Hz]")
                if i%ncols ==0:
                    ax.set_ylabel(f"Stokes I ({scale} scale)")
                ax.set_yscale(scale)

            # remove unused subplots
            for j in range(n_bins, nrows*ncols):
                fig.delaxes(axes[j // ncols, j % ncols])

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved graphs to {pdf_path}")

def plot_from_data(data_path, ncols, outdir, log=False, all_stokes=False, band_per_plot=None):
    with np.load(data_path) as f:
        avg_data = f["data"]
        missing_fraction = f["missing_fraction"]
        total_counts = f["total_counts"]
        bin_edges = f["bin_edges"]
        t_chunk = f["t_chunk"]
        chans = f["chans"]
        osamp = f["osamp"]

    get_plots(avg_data, bin_edges, missing_fraction, total_counts, chans, t_chunk, osamp, ncols, outdir, log=log, all_stokes=all_stokes, band_per_plot=band_per_plot)

def repfb_xcorr_bin_avg(time: List[int], plasma: List[int], avg_vis: MedianTracker, dir_parents: str, spec_offsets: List[float], acclen: int,  nblock: int, chanstart: int, chanend: int, osamp: int, cut: int = 10, filt_thresh: float = 0.45, window: Optional[cp.ndarray] = None, filt: Optional[cp.ndarray] = None, verbose=False) -> Tuple[int, np.ndarray, cp.ndarray, cp.ndarray]:
    """
    Perform oversampling PFB, GPU-based cross-correlation on streaming baseband data, and average over plasma frequency bins.
    
    Args:
        time: List of time intervals (in seconds since epoch)
        plasma: List of plasma frequency bin indices corresponding to time intervals
        avg_vis: RunningMean object to average the output visibilities
        dir_parents: parent directory of ALBATROS data
        spec_offsets: clock offsets for each antenna
        acclen: Accumulation length in units of 4096-sample IPFB output blocks
        nblock: Number of PFB blocks per iteration (streamed)
        chanstart, chanend: Frequency channel bounds
        osamp: Oversampling factor for RePFB
        cut: Number of rows to cut from top and bottom of IPFB
        filt_thresh: Regularization parameter for IPFB deconvolution filter
        window: Precomputed PFB window (optional)
        filt: Precomputed IPFB filter (optional)
        verbose: Enable verbose logging
    
    Returns:
        t_chunk: The time of each pfb_output
        freqs: Array of final frequency bins processed
        window: PFB window used
        filt: IPFB filter used
    """

    init_t = time[0]
    end_t = time[-1]
    min_t_int = np.min(np.diff(time))
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/acclen))
    idxs, files = get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents) 

    # Setup configuration
    nant = len(idxs)
    config = ProcessingConfig(
        acclen=acclen, nchunks=nchunks, nblock=nblock,
        chanstart=chanstart, chanend=chanend, osamp=osamp, nant=nant, cut=cut, filt_thresh=filt_thresh
    )
    sizes = BufferSizes.from_config(config)

    # Note that the recording frequency is 250 MSPS which means that each sample corresponds to 1/250e6 s
    time_track = init_t + sizes.lblock * (config.ntap - 1) / 250e6 # this is to adjust for the additional (ntap-1)*lblock
    time_idx = 0
    time_max_idx = len(time)-1
    t_chunk = sizes.lblock * nblock / 250e6 # in seconds <-- already checked that this is correct
    avg_vis.t_chunk = t_chunk

    # Setup components
    antenna_objs, channels, channel_idxs = setup_antenna_objects(idxs, files, config)
    if len(channels) < chanend - chanstart:
        print(f"Warning: Not enough channels to cover requested range {chanstart}-{chanend-1}, available channels are {channel_idxs[0]}-{channel_idxs[-1]}")
        print(len(channels), len(channel_idxs), chanstart, chanend, channel_idxs[0], channel_idxs[-1])
        return t_chunk, None, window, filt # not enough channels to cover requested range

    window, filt = setup_filters_and_windows(config, window, filt)
    
    # Initialize managers
    buffer_mgr = BufferManager(config, sizes)
    ipfb_processor = IPFBProcessor(config, buffer_mgr, channels, channel_idxs-64, filt)
    missing_tracker = MissingDataTracker(nant, sizes.num_of_pfbs)
    
    # Setup frequency ranges
    repfb_chanstart = channels[config.chanstart-64] * osamp
    repfb_chanend = channels[config.chanend-64] * osamp
    
    # Initialize output arrays
    xin = cp.empty((config.nant * config.npol, nblock, sizes.nchan), dtype='complex64', order='F')
    
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    
    if verbose:
        _print_debug_info(config, sizes, buffer_mgr, sizes.num_of_pfbs, nchunks)

    job_idx = 0

    for i, chunks in enumerate(zip(*antenna_objs)):
        for ant_idx in range(config.nant):
            chunk = chunks[ant_idx]
            for pol_idx in range(config.npol):
                ipfb_processor.process_chunk(chunk, ant_idx, pol_idx, start_specnums[ant_idx])        

        assert (buffer_mgr.pfb_idx == buffer_mgr.pfb_idx[0,0]).all(), "The PFB idx are not equal across nant or npol"

        # We know that all antennas/pols have the same PFB index after processing so is sufficient to check one
        while buffer_mgr.pfb_idx[0,0] == sizes.szblock:
            # Generate PFB output for cross-correlation
            for ant_idx in range(config.nant):
                for pol_idx in range(config.npol):
                    output_start = ant_idx * config.nant + pol_idx # <-- still not sure
                    xin[output_start, :, :] = pu.cupy_pfb(
                        buffer_mgr.pfb_buf[ant_idx, pol_idx], 
                        window,
                        nchan=2048 * osamp + 1, 
                        ntap=4
                    )[:, repfb_chanstart:repfb_chanend]

            time_track += t_chunk
            while time_idx < time_max_idx and t_chunk/2 < time_track - time[time_idx+1]: # <- will land in the bin where most of the time stream comes from:
                time_idx += 1
            
            bin_idx = plasma[time_idx]
            avg_vis.add_to_buf(
                bin_idx, 
                cp.asnumpy(
                    cr.avg_xcorr_all_ant_gpu(xin, config.nant, config.npol, nblock, sizes.nchan, split=1)
                )
            )

            buffer_mgr.reset_overlap_region()
            for ant_idx in range(config.nant):
                for pol_idx in range(config.npol):
                    buffer_mgr.add_remaining_to_pfb_buffer(ant_idx, pol_idx)

            job_idx += 1

            if verbose and job_idx % 100 == 0:
                print(f"Job Chunk {job_idx + 1}/{sizes.num_of_pfbs} s")
        
    # Do not add the last incomplete pfb chunk to avoid tainting any results

    if verbose:
        print("=" * 30)
        print(f"Completed {sizes.num_of_pfbs}/{sizes.num_of_pfbs} Job Chunks")
        print("=" * 30)
    
    freqs = np.arange(repfb_chanstart, repfb_chanend)
    
    return t_chunk, freqs, window, filt

def mock_repfb_xcorr_bin_avg(time: List[int], plasma: List[int], avg_vis: MedianTracker, acclen: int,  nblock: int, osamp: int) -> int:
    """
    Mock function to simulate counting the number of outputs per plasma bin for median tracker.
    This function does not perform actual cross-correlation but updates the bin counts in avg_vis.
    
    Args:
        time: List of time intervals (in seconds since epoch)
        plasma: List of plasma frequency bin indices corresponding to time intervals
        avg_vis: MedianTracker object to track counts per plasma bin
        acclen: Accumulation length in units of 4096-sample IPFB output blocks
        nblock: Number of PFB blocks per iteration (streamed)
    """


    init_t = time[0]
    end_t = time[-1]
    min_t_int = np.min(np.diff(time))
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/acclen))

    config = ProcessingConfig(
        acclen=acclen, nchunks=nchunks, nblock=nblock,
        chanstart=0, chanend=1, osamp=osamp, nant=1, cut=1, filt_thresh=0
    )
    sizes = BufferSizes.from_config(config)
    mock_buffer_mgr = MockBufferManager(config, sizes)

    # Note that the recording frequency is 250 MSPS which means that each sample corresponds to 1/250e6 s
    time_track = init_t + sizes.lblock * (config.ntap - 1) / 250e6 # this is to adjust for the additional (ntap-1)*lblock
    time_idx = 0
    time_max_idx = len(time)-1
    t_chunk = sizes.lblock * nblock / 250e6 # in seconds <-- already checked that this is correct
    avg_vis.t_chunk = t_chunk

    for i in range(nchunks): # <- check that this is correct TO DO
        # This loop goes over each chunk of data for all antennas to simulate the streaming process and calculate the number of outputs per bin we will have
        for ant_idx in range(config.nant):
            for pol_idx in range(config.npol):
                mock_buffer_mgr.add_chunk_to_buffer(ant_idx, pol_idx)  

        while mock_buffer_mgr.pfb_idx[0,0] == sizes.szblock:
            time_track += t_chunk
            while time_idx < time_max_idx and t_chunk/2 < time_track - time[time_idx+1]: 
                time_idx += 1 # <- will land in the bin where most of the time stream comes from:
            
            bin_idx = plasma[time_idx]
            avg_vis.add_to_bin_tot_count(bin_idx)

            mock_buffer_mgr.reset_overlap_region()
            for ant_idx in range(config.nant):
                for pol_idx in range(config.npol):
                    mock_buffer_mgr.add_remaining_to_pfb_buffer(ant_idx, pol_idx)
    
    return t_chunk


def main(plot_cols=None, band_per_plot=None, median_batch_size=10000):
    timer1 = time.time()

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
    tmp_dir = config["misc"]["tmp_dir"]
    graphs_dir = path.join(outdir, "plots")
    outdir = path.join(outdir, "out")

    obs_period = (1721342774+5*60, 1721449881) # Full observation (first 2 chunks) period (+5min just for buffer times will be shifted back by 2.5min)

    # obs_period = (1721342250+5*60, 1721368700) # First 'continuous' chunk of observation period
    # # Note that there are some holes in the albatros data towards the beginning
    # obs_period = (1721379700, 1721411900) # Second 'continuous' chunk of observation period

    # # To do: get from json
    # obs_period = (1721379700, 1721379700+30*60) # 30 minute test

    # obs_period = (1721342774+5*60, 1721666031) # Full observation period adjusted for holes in data at start
    # # 1721360558.
    # obs_period = (1721411900+5*60, 1721666031) # Third 'continuous' chunk of observation period
    
    all_time, all_plasma = get_plasma_data(plasma_path, obs_period)
    binned_time, binned_plasma, bin_edges = bin_plasma_data(all_time, all_plasma, bin_num, plot_bins_path=path.join(graphs_dir, "plasma_bin_hist.png"), split_for_gaps=True)
    
    # Initialize mean tracker for each bin
    avg_vis = MedianTracker(bin_num, tmp_dir, bin_edges, obs_period[0], obs_period[1])

    # This will predict how many output rows we will have per plasma bin for median calculation
    for i in range(len(binned_time)):
        t_chunk = mock_repfb_xcorr_bin_avg(binned_time[i], binned_plasma[i], avg_vis, acclen, nblock, osamp)
    # Seems to work fine
    
    # maybe add a buffer amount to disk files to avoid issues with exact sizes

    # print("Total bins counts:")
    # k = 1
    # for i in range(bin_num):
    #     n = avg_vis.bin_tot_count[i]
    #     print(f"bin {k}:", n, "intervals,", n*t_chunk/60, "minutes")
    #     k+=1

    channels = None
    window, filt = None, None # window and filter can be reused between calls as same acclen, nblock, osamp
    for i in range(len(binned_time)):
        print(f"Processing continuous chunk {i+1}/{len(binned_time)} over {len(binned_time[i])*5} minutes")
        t_chunk, new_channels, window, filt = repfb_xcorr_bin_avg(
            binned_time[i], binned_plasma[i], avg_vis, dir_parents, spec_offsets, acclen, nblock,
            chanstart, chanend, osamp, cut=cut, filt_thresh=0.45,
            window=window, filt=filt
        )
        channels = new_channels if new_channels is not None else channels

    np.savez(path.join(tmp_dir, "params_checkpoint.npz", bin_count=avg_vis.bin_count, fine_counter=avg_vis.fine_counter, bin_num=avg_vis.bin_num, shape=avg_vis.shape, dtype=avg_vis.dtype, bin_edges=bin_edges, t_chunk=t_chunk, chans=channels, osamp=osamp)) # Save in case median fails
    print("RAW processing complete, getting MEAN/MEDIAN and saving...")

    median, fine_counter, bin_count = avg_vis.get_median(median_batch_size) 
    missing_fraction = 1.-fine_counter.mean(axis=tuple(range(1,count.ndim)))/bin_count

    fname = f"average_plasma_bins_{str(bin_num)}_{str(osamp)}_{obs_period[0]}_{obs_period[1]}_{chanstart}_{chanend}.npz"
    fpath = path.join(outdir,fname)
    np.savez(fpath, data=median, missing_fraction=missing_fraction, total_counts=bin_count, bin_edges=bin_edges, t_chunk=t_chunk, chans=channels, osamp=osamp)

    print(f"Saved ALBATROS data with an oversampling rate of {osamp} in {bin_num} plasma frequency bins at")
    print(fpath)

    timer2 = time.time()
    print(f"Processing took {timer2-timer1} s")

    if plot_cols is not None:
        print("Printing Graph...")
        get_plots(median, bin_edges, missing_fraction, bin_count, channels, t_chunk, osamp, plot_cols, graphs_dir, log=True, all_stokes=False, band_per_plot=band_per_plot)


if __name__=="__main__":
    bbw = 125e6/2048
    # If you want to run the full processing and plotting, just call main()
    main(plot_cols=4, band_per_plot=bbw,median_batch_size=1000)


    # If you just want to plot from existing data, use plot_from_data()
    # data_path = '/scratch/philj0ly/vis_plasma/average_plasma_bins_20_65536_1721343074_1721449881_64_183_v2.npz'
    # outdir = '/scratch/philj0ly/vis_plasma/plots/'
    # outdir = '/home/philj0ly/'
    # plot_from_data(data_path, 4, outdir, log=True, all_stokes=False, band_per_plot=bbw)
