# Philippe Joly 2025-09-03

# This script is designed to rebin frequency data from ALBATROS and integrate over times dictated by the plasma frequency bins
# This is meant to process a list of frequency bins and time intervals, and visualise the results


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from os import path
import sys
sys.path.insert(0,path.expanduser("~"))

from albatros_analysis.scripts.xcorr.helper import get_init_info_all_ant
from albatros_analysis.scripts.xcorr.helper_gpu import repfb_xcorr_avg

t_diff = 5*60 # (s) 5 minutes b/w measurements <-- to do: get from data

def get_plasma_freq(nmf2):
    eps = 8.8541878188 * 1e-12 # vacuum electric permittivity  [ F / m]
    m_e = 9.1093837139 * 1e-31 # electron mass [ kg ]
    q_e = 1.602176634  * 1e-19 # electron charge [ C ]

    w_p = np.sqrt( nmf2 * q_e**2 / (m_e * eps) )
    return w_p / (2*np.pi)


def get_time_bins(plasma_fn, obs_period, bin_num):
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

    counts, bin_edges = np.histogram(plasma, bins=bin_num)
    bin_indices = np.digitize(plasma, bin_edges)

    binned_data = {}
    for i in range(1, bin_num + 1):
        binned_data[i] = time[bin_indices == i]

    binned_time = {}
    for i in binned_data:
        times_out = []
        timestamps = np.sort(binned_data[i])

        for t in timestamps:
            if len(times_out) != 0 and t == times_out[-1][-1]:
                times_out[-1][-1] += t_diff
            else:
                times_out.append([t, t+t_diff])

        binned_time[i] = {
            "band": [bin_edges[i-1], bin_edges[i]], 
            "total_time": binned_data[i].shape[0]*t_diff,
            "times": times_out
        }  
    
    return binned_time


def read_repfb_data(init_t, end_t, chanstart, chanend, osamp, acclen, cut, nblock, spec_offsets, dir_parents, window=None, filt=None):
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/acclen))
    init_t, end_t = int(init_t), int(end_t)

    print(init_t, end_t, end_t-init_t, nchunks)
    idxs, files = get_init_info_all_ant(init_t, end_t, spec_offsets, dir_parents)
    
    pols,missing_fraction,channels, window, filt = repfb_xcorr_avg(idxs,files,acclen,nchunks,nblock,chanstart,chanend,osamp,cut=cut,filt_thresh=0.45, window=window, filt=filt)

    return pols, missing_fraction, channels, window, filt

    
def get_avg_data(dir_parents, spec_offsets, chanstart, chanend, osamp, acclen, cut, nblock, times):
    # to do: handle missing fraciton
    window, filt = None, None

    avg_data = {}

    for i in times:
        print(f"Processing band {i}: {times[i]['band']} Hz with {times[i]['total_time']/60} minutes of data...")

        avg_data[i] = {
            "band": times[i]['band'],
            "total_time": times[i]['total_time'],
            "missing_fraction": 0.,
            "channels": None,
            "I_stokes": None,
            "Q_stokes": None,
            "U_stokes": None,
            "V_stokes": None,
        }

        count = 0

        for t in times[i]['times']:
            pols, missing_fraction, channels, window, filt = read_repfb_data(t[0], t[1], chanstart, chanend, osamp, acclen, cut, nblock, spec_offsets, dir_parents, window=window, filt=filt)

            count += pols[0,0].shape[1]  # Number of time samples processed

            if avg_data[i]["I_stokes"] is None:
                avg_data[i]["I_stokes"] = np.zeros(pols[0,0].shape[0], dtype='float32')
                avg_data[i]["Q_stokes"] = avg_data[i]["I_stokes"].copy()

                avg_data[i]["U_stokes"] = np.zeros(pols[0,0].shape[0], dtype='complex64')
                avg_data[i]["V_stokes"] = avg_data[i]["U_stokes"].copy()
            else:
                avg_data[i]["I_stokes"] += np.sum(pols[0,0] + pols[1,1], axis=1).real.astype("float32")  # I = XX + YY
                avg_data[i]["Q_stokes"] += np.sum(pols[0,0] - pols[1,1], axis=1).real.astype("float32")  # Q = XX - YY

                avg_data[i]["U_stokes"] += np.sum(pols[0,1] + pols[1,0], axis=1).astype("complex64")  # U = XY + YX
                avg_data[i]["V_stokes"] += 1j * np.sum(pols[1,0] - pols[0,1], axis=1).astype("complex64")  # V = i(YX - XY)

            # avg_data[i]["missing_fraction"] += missing_fraction
        
        # avg_data[i]["missing_fraction"] /= len(times[i]['total_time']/t_diff)  # Average missing fraction
        avg_data[i]["channels"] = channels  # channels should be the same for all
        avg_data[i]["I_stokes"] /= count  # Average over time intervals
        avg_data[i]["Q_stokes"] /= count
        avg_data[i]["U_stokes"] /= count
        avg_data[i]["V_stokes"] /= count

        break  # to do: remove - for testing only

    return avg_data


def get_plot(avg_data, osamp, width, height, outdir, log=False, all_stokes=False):
    df_record = 125e6/2048 # (Hz) frequency range / # of channels
    df = df_record/osamp # (Hz) / rechannelization factor

    scale = 'linear'
    if log:
        scale = 'log'

    for i in avg_data:
        channels = avg_data[i]["channels"]
        I_stokes = avg_data[i]["I_stokes"]
        Q_stokes = avg_data[i]["Q_stokes"]
        U_stokes = avg_data[i]["U_stokes"]
        V_stokes = avg_data[i]["V_stokes"]
        total_time = avg_data[i]["total_time"]

        freqs = channels * df  # to do: center frequencies

        plt.figure(figsize=(width, height))
        plt.plot(channels, I_stokes, label='I Stokes', color='blue')

        if all_stokes:
            plt.plot(channels, Q_stokes, label='Q Stokes', color='orange')
            plt.plot(channels, np.abs(U_stokes), label='|U Stokes|', color='green')
            plt.plot(channels, np.abs(V_stokes), label='|V Stokes|', color='red')

        plt.title(f'Stokes Parameter - Band {avg_data[i]["band"]} Hz over {total_time/60} minutes')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Intensity')
        plt.yscale(scale)
        plt.grid()
        plt.legend()

        fname = f'graphs/stokes_{all_stokes}_plot_band_log_{log}_{avg_data[i]["band"][0]}_{avg_data[i]["band"][1]}_{str(osamp)}_{total_time}.png'
        fpath = path.join(outdir,fname)
        plt.savefig(fpath)
        plt.close()
    
    print(f"Saved plots to {outdir}")


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

    plasma_fn = config["misc"]["plasma_path"]
    outdir = config["misc"]["out_dir"]

    obs_period = [1721342139, 1721449881] # Full observation period
    # obs_period = [1721342139, 1721376339] # First 'continuous' chunk of observation period
    # obs_period = [1721376339, 1721449881] # Second 'continuous' chunk of observation period
    # To do: get from json

    binned_time = get_time_bins(plasma_fn, obs_period, bin_num)

    avg_data = get_avg_data(dir_parents, spec_offsets, chanstart, chanend, osamp, acclen, cut, nblock, binned_time)

    fname = f"average_stokes_{str(bin_num)}_{str(acclen)}_{str(osamp)}_{str(nblock)}_{chanstart}_{chanend}.npz"
    fpath = path.join(outdir,fname)
    np.savez(fpath, **avg_data)

    print(f"Saved averaged data to {fpath}")

    if plot == 'linear':
        get_plot(avg_data, osamp, width=20, height=10, outdir=outdir, log=False, all_stokes=True)
    elif plot == 'log':
        get_plot(avg_data, osamp, width=20, height=10, outdir=outdir, log=True, all_stokes=False)


if __name__ == "__main__":
    main(plot=True)