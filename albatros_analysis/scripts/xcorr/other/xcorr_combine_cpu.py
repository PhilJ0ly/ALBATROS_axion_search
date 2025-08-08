import sys
from os import path
import glob
sys.path.insert(0, path.expanduser("~"))
import numpy as np

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import time
import os

def combine_node_results(output_files, total_nchunks, nant, nchan):
    """
    Combine results from multiple nodes
    """
    npol = 2
    
    # Initialize combined arrays
    vis_combined = np.zeros((nant*npol, nant*npol, nchan, total_nchunks), dtype="complex64", order="F")
    missing_combined = np.zeros((nant, total_nchunks), dtype='float64', order='F')
    
    chunk_offset = 0
    for output_file in output_files:
        # Load node results
        vis_node = np.load(f"{output_file}_vis.npz")
        missing_node = np.load(f"{output_file}_missing.npz")
        
        nchunks_node = vis_node.shape[-1]
        
        # Copy to combined arrays
        vis_combined[:, :, :, chunk_offset:chunk_offset+nchunks_node] = vis_node
        missing_combined[:, chunk_offset:chunk_offset+nchunks_node] = missing_node
        
        chunk_offset += nchunks_node
    
    # Load channels from first file (should be same for all)
    channels = np.load(f"{output_files[0]}_channels.npz")
    
    return vis_combined, missing_combined, channels

if __name__=="main":
    output_dir = os.environ['OUTPUT_DIR']
    nnodes = int(os.environ['NNODES'])

    print(f"Combining results from {nnodes} nodes")

    # Find all node output files
    output_files = []
    for node_id in range(nnodes):
        # Find the output file pattern for this node
        pattern = f"{output_dir}/node_{node_id}_chunks_*"
        files = glob.glob(pattern)
        if files:
            base_file = files[0].replace('_vis.npz', '').replace('_missing.npz', '').replace('_channels.npz', '')
            output_files.append(base_file)
        else:
            print(f"Warning: No output files found for node {node_id}")

    print(f"Found output files: {output_files}")

    # Determine dimensions from first file
    vis_sample = np.load(f"{output_files[0]}_vis.npz")
    nant_npol, _, nchan, _ = vis_sample.shape
    nant = int(np.sqrt(nant_npol // 2))  # Assuming npol=2

    # Calculate total chunks
    total_nchunks = sum(np.load(f"{file}_vis.npz").shape[-1] for file in output_files)

    print(f"Combining data: nant={nant}, nchan={nchan}, total_nchunks={total_nchunks}")

    # Combine results
    try:
        vis_combined, missing_combined, channels = combine_node_results(
            output_files, total_nchunks, nant, nchan
        )
        
        # Save final combined results
        fname = f"xcorr_all_ant_4bit_{str(init_t)}_{str(acclen)}_{str(osamp)}_{str(nchunks)}_{chanstart}_{chanend}.npz"
        fpath = path.join(output_dir,fname)
        np.savez(fpath,data=vis_combined.data,mask=vis_combined.mask,missing_fraction=missing_combined,chans=channels)

        # final_output = f"{output_dir}/albatros_final_results"
        # np.save(f"{final_output}_vis.npz", vis_combined)
        # np.save(f"{final_output}_missing.npz", missing_combined)
        # np.save(f"{final_output}_channels.npz", channels)
        
        print(f"Final results saved to {fpath}")
        # print(f"Final vis shape: {vis_combined.shape}")
        
        # Optionally create summary statistics
        # vis_masked = np.ma.masked_invalid(vis_combined)
        # print(f"Data completeness: {(1 - vis_masked.mask.sum() / vis_masked.size) * 100:.2f}%")
        
    except Exception as e:
        print(f"Error combining results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)