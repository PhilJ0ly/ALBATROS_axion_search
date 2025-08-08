# Philippe Joly 2025-06-09
# Memory-optimized version of ALBATROS rebinning with chunked processing

import sys
from os import path
import traceback
import gc
import glob
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr 
from albatros_analysis.src.utils import pfb_cpu_utils as pu
import numpy as np

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import time
import os

def repfb_xcorr_avg_chunked(idxs, files, pfb_size, nchunks, chanstart, chanend, osamp, 
                           cutsize=16, filt_thresh=0.45, n_cores=None, 
                           chunk_batch_size=None, output_file=None):
    """
    Memory-optimized version that processes chunks in batches and optionally saves to disk
    
    Parameters:
    -----------
    chunk_batch_size : int, optional
        Number of chunks to process at once. If None, auto-calculate based on memory
    output_file : str, optional
        If provided, saves results to disk using memory mapping
    """
    
    if n_cores is None or n_cores > mp.cpu_count():
        n_cores = mp.cpu_count()

    print("Using", n_cores, "CPU Cores")

    nant = len(idxs)
    npol = 2
    
    cut = int(pfb_size/cutsize)
    acclen = pfb_size - 2*cut
    ntap = 4
    nn = 2*2048*osamp
    assert acclen % osamp == 0

    re_pfb_size = acclen//osamp - ntap + 1
    assert re_pfb_size > 0 

    # Calculate memory requirements and optimal batch size
    nchan = (chanend - chanstart) * osamp
    
    # Memory per chunk (in GB)
    vis_chunk_mem = (nant*npol)**2 * nchan * 8 / (1024**3)  # complex64 = 8 bytes
    xin_mem = nant*npol * re_pfb_size * nchan * 8 / (1024**3)
    
    print(f"Memory per chunk: vis={vis_chunk_mem:.2f}GB, xin={xin_mem:.2f}GB")
    
    # Auto-calculate batch size if not provided (target ~50GB per batch)
    if chunk_batch_size is None:
        target_batch_mem = 50  # GB
        chunk_batch_size = max(1, int(target_batch_mem / (vis_chunk_mem + xin_mem)))
        chunk_batch_size = min(chunk_batch_size, nchunks)
    
    print(f"Processing {nchunks} chunks in batches of {chunk_batch_size}")

    # Setup shared arrays for PFB matrices (read-only)
    dwin = pu.sinc_hamming(ntap, nn)
    cpu_win_big = np.asarray(dwin, dtype='float32', order='C')
    matft = pu.get_matft_cpu(pfb_size)

    try:
        # Shared memory for read-only data
        matft_shm = shared_memory.SharedMemory(create=True, size=matft.nbytes)
        matft_shared = np.ndarray(matft.shape, dtype=matft.dtype, buffer=matft_shm.buf)
        np.copyto(matft_shared, matft)

        win_shm = shared_memory.SharedMemory(create=True, size=cpu_win_big.nbytes)
        cpu_win_shared = np.ndarray(cpu_win_big.shape, dtype=cpu_win_big.dtype, buffer=win_shm.buf)
        np.copyto(cpu_win_shared, cpu_win_big)

        # Initialize antenna objects
        antenna_objs = []
        for i in range(nant):
            aa = bdc.BasebandFileIterator(
                files[i], 0, idxs[i], acclen, nchunks=nchunks,
                chanstart=chanstart, chanend=chanend, type='float'
            )
            antenna_objs.append(aa)

        channels = np.asarray(aa.obj.channels, dtype='int64')
        repfb_chanstart = channels[aa.obj.chanstart] * osamp
        repfb_chanend = channels[aa.obj.chanend] * osamp
        
        print("start and end chans are", repfb_chanstart, repfb_chanend)
        print("nant", nant, "nchunks", nchunks, "nchan", nchan)

        # Initialize output arrays or memory-mapped files
        if output_file:
            # Use memory-mapped arrays to reduce RAM usage
            vis = np.memmap(f"{output_file}_vis.dat", dtype='complex64', mode='w+',
                          shape=(nant*npol, nant*npol, nchan, nchunks), order='F')
            missing_fraction = np.memmap(f"{output_file}_missing.dat", dtype='float64', mode='w+',
                                       shape=(nant, nchunks), order='F')
        else:
            vis = np.zeros((nant*npol, nant*npol, nchan, nchunks), dtype="complex64", order="F")
            missing_fraction = np.zeros((nant, nchunks), dtype='float64', order='F')

        # Persistent state for cut_chunks
        cut_chunks = np.zeros((nant, npol, 2*cut, 2049), dtype='complex64', order='C')
        start_specnums = [ant.spec_num_start for ant in antenna_objs]

        # Process chunks in batches
        for batch_start in range(0, nchunks, chunk_batch_size):
            batch_end = min(batch_start + chunk_batch_size, nchunks)
            batch_size = batch_end - batch_start
            
            print(f"Processing batch {batch_start//chunk_batch_size + 1}/{(nchunks-1)//chunk_batch_size + 1}: "
                  f"chunks {batch_start}-{batch_end-1}")
            
            xin_batch = np.empty((nant*npol, re_pfb_size, nchan), dtype='complex64', order='F')
            vis_batch = np.zeros((nant*npol, nant*npol, nchan, batch_size), dtype="complex64", order="F")
            
            batch_chunks = []
            for chunk_idx in range(batch_start, batch_end):
                chunk_data = []
                for ant_obj in antenna_objs:
                    # Get the chunk_idx-th chunk from each antenna
                    chunk_data.append(next(ant_obj))
                batch_chunks.append((chunk_idx, chunk_data))
            
            args_list = [
                (
                    chunk_idx, chunks, start_specnums, channels, aa.obj.channel_idxs, 
                    acclen, cut_chunks.copy(), cut, repfb_chanstart, repfb_chanend, 
                    filt_thresh, osamp, pfb_size,
                    (matft_shm.name, matft.shape, matft.dtype, win_shm.name, cpu_win_big.shape, cpu_win_big.dtype)
                ) for chunk_idx, chunks in batch_chunks
            ]

            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                futures = [executor.submit(pu.process_chunks, args) for args in args_list]

                for future in as_completed(futures):
                    chunk_idx, pol0_out_ants, pol1_out_ants, new_cut_chunks_ants, perc_missing_ants = future.result()
                    batch_idx = chunk_idx - batch_start
                    
                    # Update xin for this chunk
                    for j in range(nant):
                        xin_batch[j*2, :, :] = pol0_out_ants[j]
                        xin_batch[j*2+1, :, :] = pol1_out_ants[j]
                        cut_chunks[j] = new_cut_chunks_ants[j]
                        missing_fraction[j, chunk_idx] = perc_missing_ants[j]
                    
                    # Compute correlations for this chunk
                    vis_batch[:, :, :, batch_idx] = cr.avg_xcorr_all_ant_cpu(
                        xin_batch, nant, npol, re_pfb_size, nchan, split=1
                    )
            
            # Copy batch results to main array
            vis[:, :, :, batch_start:batch_end] = vis_batch
            
            # Force garbage collection
            del xin_batch, vis_batch, batch_chunks, args_list
            gc.collect()
            
            pu.print_cpu_mem(f"After batch {batch_start//chunk_batch_size + 1}:")

        vis = np.ma.masked_invalid(vis)
        return vis, missing_fraction, np.arange(repfb_chanstart, repfb_chanend)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

    finally:
        # Clean up shared memory
        matft_shm.close()
        win_shm.close()
        matft_shm.unlink()
        win_shm.unlink()
        print("Shared memory unlinked by main process.")


# def repfb_xcorr_avg_distributed(idxs, files, pfb_size, nchunks, chanstart, chanend, osamp,
#                                cutsize=16, filt_thresh=0.45, n_nodes=1, chunks_per_node=None):
#     """
#     Distributed version that splits processing across multiple nodes
    
#     Parameters:
#     -----------
#     n_nodes : int
#         Number of compute nodes available
#     chunks_per_node : int, optional
#         Number of chunks each node should process. If None, auto-calculate
#     """
    
#     if chunks_per_node is None:
#         chunks_per_node = max(1, nchunks // n_nodes)
    
#     print(f"Distributing {nchunks} chunks across {n_nodes} nodes ({chunks_per_node} chunks per node)")
    
#     # Split chunk ranges for each node
#     node_ranges = []
#     for node_id in range(n_nodes):
#         start_chunk = node_id * chunks_per_node
#         end_chunk = min(start_chunk + chunks_per_node, nchunks)
#         if start_chunk < nchunks:
#             node_ranges.append((start_chunk, end_chunk))
    
#     print("Node chunk ranges:", node_ranges)
    
#     # This would typically involve job submission to a cluster scheduler
#     # For now, return the ranges that each node should process
#     return node_ranges


def process_node_range(idxs, files, pfb_size, chunk_start, chunk_end, chanstart, chanend, osamp,
                      cutsize=16, filt_thresh=0.45, n_cores=None, node_id=0, output_dir="./"):
    """
    Process a specific range of chunks on one node and save results
    """
    nchunks_node = chunk_end - chunk_start
    
    # Create node-specific output filename
    output_file = f"{output_dir}/node_{node_id}_chunks_{chunk_start}_{chunk_end}"
    
    print(f"Node {node_id} processing chunks {chunk_start}-{chunk_end-1}")
    t1=time.time()
    # Process this node's chunk range with memory optimization
    vis, missing_fraction, channels = repfb_xcorr_avg_chunked(
        idxs, files, pfb_size, nchunks_node, chanstart, chanend, osamp,
        cutsize, filt_thresh, n_cores, chunk_batch_size=30, output_file=output_file
    )

    t2=time.time()
    
    # Save results
    np.savez(f"{output_file}_vis.npz", vis)
    np.savez(f"{output_file}_missing.npz", missing_fraction)
    np.savez(f"{output_file}_channels.npz", channels)
    
    print(f"Node {node_id} completed. Results saved to {output_file}_*.npz")

    print("Processing took", t2-t1, "s")
    
    return output_file




# Example usage functions
# def run_single_node_optimized(idxs, files, pfb_size, nchunks, chanstart, chanend, osamp,
#                              cutsize=16, filt_thresh=0.45, n_cores=None):
#     """
#     Run on single node with memory optimization
#     """
#     return repfb_xcorr_avg_chunked(
#         idxs, files, pfb_size, nchunks, chanstart, chanend, osamp,
#         cutsize, filt_thresh, n_cores, chunk_batch_size=10  # Process 10 chunks at a time
#     )

# def run_multi_node_distributed(idxs, files, pfb_size, nchunks, chanstart, chanend, osamp,
#                               cutsize=16, filt_thresh=0.45, n_nodes=10, output_dir="./results"):
#     """
#     Run distributed across multiple nodes
#     """
#     # Get chunk ranges for each node
#     node_ranges = repfb_xcorr_avg_distributed(
#         idxs, files, pfb_size, nchunks, chanstart, chanend, osamp,
#         cutsize, filt_thresh, n_nodes
#     )
    
#     print("Submit these jobs to your cluster:")
#     for i, (start, end) in enumerate(node_ranges):
#         print(f"Node {i}: process_node_range(idxs, files, pfb_size, {start}, {end}, "
#               f"{chanstart}, {chanend}, {osamp}, {cutsize}, {filt_thresh}, "
#               f"n_cores=80, node_id={i}, output_dir='{output_dir}')")
    
#     return node_ranges