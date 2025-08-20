# Philippe Joly 2025-08-14

""" 
GPU RePFB Stream 

This script implements a RePFB algorithm to change the frequency resolution
of ALBATROS.
"""

import sys
from os import path
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr
import cupy as cp
from albatros_analysis.src.utils import pfb_gpu_utils as pu
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import ctypes
lib = ctypes.CDLL('./libcgemm_batch.so')

lib.cgemm_strided_batched.argtypes = [
    ctypes.c_void_p,  # A.ptr
    ctypes.c_void_p,  # B.ptr
    ctypes.c_void_p,  # C.ptr
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int,     # K
    ctypes.c_int      # batchCount
]
lib.cgemm_strided_batched.restype = None


@dataclass
class ProcessingConfig:
    """Configuration for RePFB processing"""
    acclen: int  # Accumulation length for IPFB input
    pfb_size: int  # Size of the PFB transform
    nchunks: int  # Total number of IPFB output chunks to process
    nblock: int  # Number of PFB blocks per iteration (streamed)
    chanstart: int  # Start frequency channel
    chanend: int  # End frequency channel
    osamp: int  # Oversampling factor for RePFB
    nant: int # Number of antennas to process
    cut: int = 10  # Rows to cut from top/bottom of IPFB
    filt_thresh: float = 0.45  # Regularization parameter for IPFB deconvolution
    ntap: int = 4  # Number of taps
    npol: int = 2  # Number of polarizations

@dataclass
class BufferSizes:
    """Calculated buffer sizes"""
    lchunk: int  # Length of time stream chunk after IPFB
    szblock: int  # Total number of time stream samples 
    
    @classmethod
    def from_config(cls, config: ProcessingConfig) -> 'BufferSizes':
        lchunk = 4096 * config.acclen
        szblock = config.nchunks * lchunk
        return cls(lchunk, szblock)

class MissingDataTracker:
    """Tracks missing data fractions across processing"""
    
    def __init__(self, nant: int, n_job_chunks: int):
        self.missing_count = [[] for _ in range(nant)]
        self.missing_fraction = np.zeros((nant, n_job_chunks), dtype='float64')
    
    def add_chunk_info(self, ant_idx: int, chunk: dict, acclen: int, pfb_blocks_affected: int):
        """Add missing data info for a chunk"""
        missing_pct = (1 - len(chunk["specnums"]) / acclen) * 100
        self.missing_count[ant_idx].append([missing_pct, pfb_blocks_affected])
    
    def calculate_job_average(self, ant_idx: int, job_idx: int):
        """Calculate average missing fraction for a job"""
        total_missing = 0.0
        count = 0
        
        for miss_info in self.missing_count[ant_idx]:
            total_missing += miss_info[0]
            miss_info[1] -= 1  # Decrement blocks affected
            count += 1
        
        if count > 0:
            self.missing_fraction[ant_idx, job_idx] = total_missing / count
        
        # Remove entries that won't affect future jobs
        self.missing_count[ant_idx] = [info for info in self.missing_count[ant_idx] if info[1] > 0]


class TimeStreamManager:
    """Manages time stream output for IPFB-only processing"""
    
    def __init__(self, config: ProcessingConfig, sizes: BufferSizes):
        self.config = config
        self.sizes = sizes
        # Initialize output arrays for time stream data
        self.time_streams = np.zeros((config.nant, config.npol, sizes.szblock), dtype='complex64', order='F')
        self.cut_pol = cp.zeros((config.nant, 2, 2*config.cut, 2049), dtype='complex64', order='C')
        self.pol_buffer = cp.zeros((config.acclen + 2*config.cut, 2049), dtype='complex64', order='C')
        self.sample_idx = np.zeros((config.nant, config.npol), dtype=np.uint64)
    
    def add_chunk(self, ant_idx: int, pol_idx: int, processed_chunk: cp.ndarray):
        """Add processed time stream chunk to output array"""
        
        start_idx = self.sample_idx[ant_idx, pol_idx]
        end_idx = start_idx + self.sizes.lchunk
        
        # Convert to numpy and store
        self.time_streams[ant_idx, pol_idx, start_idx:end_idx] = cp.asnumpy(processed_chunk)
        self.sample_idx[ant_idx, pol_idx] = end_idx


def setup_antenna_objects(idxs: List[int], files: List[str], config: ProcessingConfig) -> Tuple[List, np.ndarray]:
    """Set up antenna file iterators"""
    antenna_objs = []
    for i, (idx, file) in enumerate(zip(idxs, files)):
        antenna = bdc.BasebandFileIterator(
            file, 0, idx, config.acclen, nchunks=config.nchunks,
            chanstart=config.chanstart, chanend=config.chanend, type='float'
        )
        antenna_objs.append(antenna)
    
    # Get channel information from first antenna
    channels = np.asarray(antenna_objs[0].obj.channels, dtype='int64')
    return antenna_objs, channels


def setup_filters_and_windows(config: ProcessingConfig, cupy_win_big: Optional[cp.ndarray] = None, filt: Optional[cp.ndarray] = None) -> Tuple[cp.ndarray, cp.ndarray]:
    """Set up PFB window and IPFB filter"""
    if cupy_win_big is None:
        dwin = pu.sinc_hamming(config.ntap, 4096 * config.osamp)
        cupy_win_big = cp.asarray(dwin, dtype='float32', order='c')
    
    if filt is None:
        matft = pu.get_matft(config.acclen + 2*config.cut)
        filt = pu.calculate_filter(matft, config.filt_thresh)
    
    return cupy_win_big, filt


def get_time_stream(idxs: List[int], files: List[str], acclen: int, nchunks: int,
                   chanstart: int, chanend: int, cut: int = 10, filt_thresh: float = 0.45, 
                   filt: Optional[cp.ndarray] = None, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, cp.ndarray]:
    """
    Perform inverse PFB to generate time streams from baseband data.
    
    Args:
        idxs: List of antenna indices
        files: List of baseband files per antenna
        acclen: Accumulation length in units of 4096-sample IPFB output blocks
        nchunks: Total number of IPFB output chunks to process
        chanstart, chanend: Frequency channel bounds
        cut: Number of rows to cut from top and bottom of IPFB
        filt_thresh: Regularization parameter for IPFB deconvolution filter
        filt: Precomputed IPFB filter (optional)
        verbose: Enable verbose logging
    
    Returns:
        time_streams: Time stream data [nant, npol, nsamples]
        missing_fraction: Fraction of missing data per chunk and antenna
        filt: IPFB filter used
    """
    
    nant = len(idxs)
    config = ProcessingConfig(
        acclen=acclen, pfb_size=0, nchunks=nchunks, nblock=0,  
        chanstart=chanstart, chanend=chanend, osamp=1, nant=nant, cut=cut, filt_thresh=filt_thresh
    )
    sizes = BufferSizes.from_config(config)
    
    antenna_objs, channels = setup_antenna_objects(idxs, files, config)
    _, filt = setup_filters_and_windows(config, None, filt)
    
    # Calculate total samples for output

    timestream_mgr = TimeStreamManager(config, sizes)
    missing_tracker = MissingDataTracker(nant, nchunks)
    
    
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    
    if verbose:
        print(f"Processing {nchunks} chunks for {nant} antennas")
        print(f"Accumulation length: {acclen}, Cut: {cut}")
        print(f"Total output samples per antenna/pol: {sizes.szblock}")
        print("Starting IPFB time stream processing...")
    
    total_time = 0.0
    for chunk_idx, chunks in enumerate(zip(*antenna_objs)):
        start_time = time.perf_counter()
        
        for ant_idx in range(nant):
            chunk = chunks[ant_idx]
            
            # Track missing data
            missing_pct = (1 - len(chunk["specnums"]) / acclen) * 100
            missing_tracker.missing_count[ant_idx].append([missing_pct, 1])
            missing_tracker.missing_fraction[ant_idx, chunk_idx] = missing_pct
            
            for pol_idx in range(config.npol):  # npol = 2
                timestream_mgr.pol_buffer[:2*cut] = timestream_mgr.cut_pol[ant_idx, pol_idx]
                
                timestream_mgr.pol_buffer[2*cut:] = bdc.make_continuous_gpu(
                    chunk[f"pol{pol_idx}"],
                    chunk['specnums'],
                    start_specnums[ant_idx],
                    channels[slice(None)],
                    acclen,
                    nchans=2049
                )
                
                timestream_mgr.add_chunk(ant_idx, pol_idx, pu.cupy_ipfb(timestream_mgr.pol_buffer, filt)[cut:-cut].ravel())
                
                # Update edge data for next iteration
                timestream_mgr.cut_pol[ant_idx, pol_idx, :, :] = timestream_mgr.pol_buffer[-2*cut:]
        
        end_time = time.perf_counter()
        total_time += end_time - start_time
        
        if verbose and (chunk_idx % 1000 == 0 or chunk_idx == nchunks - 1):
            print(f"Processed chunk {chunk_idx + 1}/{nchunks}, avg time {total_time/(chunk_idx + 1):.4f} s")
    
    if verbose:
        print("=" * 50)
        print(f"Completed {nchunks} chunks")
        print(f"Average time per chunk: {total_time/nchunks:.4f} s")
        print(f"Total processing time: {total_time:.2f} s")
        print("=" * 50)
    
    return timestream_mgr.time_streams, missing_tracker.missing_fraction, np.arange(chanstart, chanend)