import numpy as np
import matplotlib.pyplot as plt
from os import path
import sys
from matplotlib.backends.backend_pdf import PdfPages


def get_plots_og(avg_data, bin_edges, missing_fraction, total_counts, chans, 
              t_chunk, osamp, width, height, outdir, 
              log=False, all_stokes=False, band_per_plot=None, bin_counts=None):

    df_record = 125e6 / 2048  # (Hz) frequency range / # of channels
    df = df_record / osamp    # (Hz) / rechannelization factor
    freqs = chans * df        # frequencies for each channel

    scale = 'log' if log else 'linear'

    
    bin_num = len(bin_edges)-1
    pdf_name = f"stokes_log_{log}_{str(osamp)}_{str(bin_num)}.pdf"
    pdf_path = path.join(outdir, pdf_name)

    with PdfPages(pdf_path) as pdf:
        # loop over bins
        for i, pols in enumerate(avg_data):
            I_stokes = (pols[0,0] + pols[1,1]).real.astype("float32")

            total_time = total_counts[i] * t_chunk

            # add a separator/title page for this bin
            fig = plt.figure(figsize=(width, height))
            title_text = (
                f"Stokes Parameters Report (Bin {i})\n\n"
                f"Bin Frequency Span: {bin_edges[i]}–{bin_edges[i+1]} Hz\n"
                f"Frequency Range: {freqs[0]/1e6:.3f}–{freqs[-1]/1e6:.3f} MHz\n"
                f"Frequency Resolution: {freqs[1]-freqs[0]:.3f} Hz\n"
                f"Time Coverage: {total_time/60:.2f} minutes\n"
                f"Missing Fraction: {missing_fraction[i]:.2%}\n"
            )
            fig.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)

            # determine subband limits
            if band_per_plot is None:
                band_limits = [(freqs[0], freqs[-1])]
            else:
                fmin, fmax = freqs[0], freqs[-1]
                band_edges = np.arange(fmin, fmax + band_per_plot, band_per_plot)
                band_limits = list(zip(band_edges[:-1], band_edges[1:]))

            # loop over subbands for this bin
            for (f_low, f_high) in band_limits:
                mask = (freqs >= f_low) & (freqs < f_high)
                if not np.any(mask):
                    continue  # skip empty band

                fsel = freqs[mask]
                Isel = I_stokes[mask]

                fig, ax = plt.subplots(figsize=(width, height))
                ax.plot(fsel, Isel, label='I Stokes', color='blue')

                if all_stokes:
                    Q_stokes = (pols[0,0] - pols[1,1]).real.astype("float32")[mask]
                    U_stokes = (pols[1,0] + pols[0,1])[mask]
                    V_stokes = (pols[1,0] - pols[0,1])[mask]
                    ax.plot(fsel, Q_stokes, label='Q Stokes', color='orange')
                    ax.plot(fsel, np.abs(U_stokes), label='|U Stokes|', color='green')
                    ax.plot(fsel, np.abs(V_stokes), label='|V Stokes|', color='red')

                ax.set_title(f"{f_low/1e3:.0f}–{f_high/1e3:.0f} kHz")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Intensity")
                ax.set_yscale(scale)
                ax.grid()
                ax.legend()

                pdf.savefig(fig)
                plt.close(fig)

    print(f"Saved big PDF to {pdf_path}")


def get_plots(avg_data, bin_edges, missing_fraction, total_counts, chans, 
              t_chunk, osamp, ncols, outdir, 
              log=False, all_stokes=False, band_per_plot=None, bin_counts=None):

    df_record = 125e6 / 2048  # (Hz) frequency range / # of channels
    df = df_record / osamp    # (Hz) / rechannelization factor
    freqs = chans * df        # frequencies for each channel
    # bin_width = (bin_edges[1] - bin_edges[0])/2
    scale = 'log' if log else 'linear'

    # build PDF path
    pdf_name = f"stokes_subbands_log_{log}_{str(osamp)}.pdf"
    pdf_path = path.join(outdir, pdf_name)

    # set up subbands once
    if band_per_plot is None:
        band_limits = [(freqs[0], freqs[-1])]
    else:
        fmin, fmax = freqs[0], freqs[-1]
        band_edges = np.arange(fmin, fmax + band_per_plot, band_per_plot)
        band_limits = list(zip(band_edges[:-1], band_edges[1:]))

    
    n_bins = len(avg_data)
    nrows = int(np.ceil(n_bins / ncols))
    height = min(6*nrows, 20)
    with PdfPages(pdf_path) as pdf:
        # loop over subbands first
        for (f_low, f_high) in band_limits:
            print(50*'-')
            print(f_low, f_high)
            mask = (freqs >= f_low) & (freqs < f_high)
            if not np.any(mask):
                continue

            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, sharey='row', figsize=(25,height), squeeze=False
            )
            bw = (f_high-f_low)/2
            fig.suptitle(
                f"I Stokes Parameter for Spectrum Band {(f_low+bw)/1e3:.0f}+–{bw/1e3:.0f} kHz\n", fontsize=24
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
                    ax.set_ylabel(f"I Stokes ({scale} scsale)")
                ax.set_yscale(scale)

            # remove unused subplots
            for j in range(n_bins, nrows*ncols):
                fig.delaxes(axes[j // ncols, j % ncols])

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved graphs to {pdf_path}")


if __name__=="__main__":
    osamp = 2**8
    bin_num = 20
    bin_edges = np.arange(bin_num+1)*1e6
    counts = np.ones(bin_num)*10000
    missing_fraction = np.random.randint(1, high=5, size=bin_num)/10+0.0123
    t_chunk = 1.1 

    width=12
    height=6
    outdir = 'out'
    band_per_plot = None
    avg_vis = np.zeros((bin_num, 2, 2, 20*osamp))
    chans = np.arange(20*osamp)
    
    T = 3e3
    # s = np.sin(np.pi*chans/T/2)
    avg_vis[:] =  np.sin(np.pi*chans/T/2)
    print("Ploring")
    # avg_vis[2,:,:] = np.sin(np.pi*chans/T/2)
    ncols=4
    
 
    get_plots(avg_vis, bin_edges, missing_fraction, counts, chans, t_chunk, osamp, ncols, outdir, log=False, all_stokes=False, band_per_plot=band_per_plot)
    
   