
# Philippe Joly 2025-08-07

""" 
GPU RePFB Stream 

This script implements a RePFB algorithm to change the frequency resolution
of ALBATROS telescope data with improved modularity and readability.
"""

# this version uses while loop instead of a job planner.

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
    lblock: int  # Length of each PFB block
    szblock: int  # Size of streaming block buffer
    lchunk: int  # Length of time stream chunk after IPFB
    nchan: int  # Number of channels after oversampling
    num_of_pfbs: int  # Number of PFBs to process in total
    
    @classmethod
    def from_config(cls, config: ProcessingConfig) -> 'BufferSizes':
        lblock = 4096 * config.osamp
        szblock = int((config.nblock + (config.ntap - 1)) * lblock)
        lchunk = 4096 * config.acclen
        nchan = (config.chanend - config.chanstart) * config.osamp
        num_of_pfbs = int(np.ceil((lchunk*config.nchunks-(config.ntap-1)*lblock)/(config.nblock*lblock)))
        return cls(lblock, szblock, lchunk, nchan, num_of_pfbs)


class BufferManager:
    """Manages GPU buffers for streaming processing"""
    
    def __init__(self, config: ProcessingConfig, sizes: BufferSizes):
        self.config = config
        self.sizes = sizes
        
        # Initialize buffers
        self.pol = cp.empty((config.acclen + 2*config.cut, 2049), dtype='complex64', order='C')
        self.cut_pol = cp.zeros((config.nant, config.npol, 2*config.cut, 2049), dtype='complex64', order='C')
        self.pfb_buf = cp.zeros((config.nant, config.npol, config.nblock + (config.ntap-1), sizes.lblock),dtype='float32', order='C')
        self.rem_buf = cp.empty((config.nant, config.npol, sizes.lchunk), dtype='float32', order='C')
        
        # Buffer indices
        self.rem_idx = np.zeros((config.nant, config.npol), dtype=np.uint64)
        self.pfb_idx = np.zeros((config.nant, config.npol), dtype=np.uint64)
    
    def reset_overlap_region(self):
        """Set up overlap region for continuity between iterations"""
        ntap = self.config.ntap
        self.pfb_buf[:, :, :ntap-1, :] = self.pfb_buf[:, :, -(ntap-1):, :]
        self.pfb_idx[:, :] = (ntap - 1) * self.sizes.lblock
    
    def add_remaining_to_pfb_buffer(self, ant_idx: int, pol_idx: int):
        """
        Add remaining buffer data to PFB buffer.
        Returns True if buffer is full, False if more data needed.
        """
        rem_size = self.rem_idx[ant_idx, pol_idx]
        pfb_pos = self.pfb_idx[ant_idx, pol_idx]
        
        if pfb_pos + rem_size > self.sizes.szblock:
            # handle overflow
            available_space = self.sizes.szblock - pfb_pos
            self._handle_buffer_overflow(ant_idx, pol_idx, available_space)
        else:
            # add all remaining data
            self.pfb_buf[ant_idx, pol_idx].flat[pfb_pos:pfb_pos + rem_size] = self.rem_buf[ant_idx, pol_idx, :rem_size]
            self.pfb_idx[ant_idx, pol_idx] += rem_size
            self.rem_idx[ant_idx, pol_idx] = 0
        
    
    def _handle_buffer_overflow(self, ant_idx: int, pol_idx: int, available_space: int):
        """Handles case where remaining buffer exceed available PFB buffer space"""
        rem_size = self.rem_idx[ant_idx, pol_idx]
        
        # Fill remaining PFB space
        self.pfb_buf[ant_idx, pol_idx].flat[self.pfb_idx[ant_idx, pol_idx]:] = self.rem_buf[ant_idx, pol_idx, :available_space]
        
        # Shift remaining data
        overflow_size = rem_size - available_space
        self.rem_buf[ant_idx, pol_idx, :overflow_size] = self.rem_buf[ant_idx, pol_idx, available_space:rem_size]
        
        self.rem_idx[ant_idx, pol_idx] = overflow_size
        self.pfb_idx[ant_idx, pol_idx] = self.sizes.szblock
    
    def add_chunk_to_buffer(self, ant_idx: int, pol_idx: int, processed_chunk: cp.ndarray):
        """
        Add processed chunk to buffers.
        Returns True if PFB buffer is full.
        """
        chunk_size = len(processed_chunk)
        pfb_pos = self.pfb_idx[ant_idx, pol_idx]
        final_pos = pfb_pos + chunk_size
        
        if final_pos > self.sizes.szblock:
            # Handle overflow
            available_space = self.sizes.szblock - pfb_pos
            self.pfb_buf[ant_idx, pol_idx].flat[pfb_pos:] = processed_chunk[:available_space]
            
            # Store overflow in remaining buffer
            overflow_size = chunk_size - available_space
            # Assunmes the remaining buffer is already empty
            self.rem_buf[ant_idx, pol_idx, :overflow_size] = processed_chunk[available_space:]
            self.rem_idx[ant_idx, pol_idx] = overflow_size
            self.pfb_idx[ant_idx, pol_idx] = self.sizes.szblock
        else:
            # Normal case
            self.pfb_buf[ant_idx, pol_idx].flat[pfb_pos:final_pos] = processed_chunk
            self.pfb_idx[ant_idx, pol_idx] = final_pos

    
    def pad_incomplete_buffer(self, ant_idx: int, pol_idx: int):
        """Pad incomplete PFB buffer with zeros"""

        current_pos = self.pfb_idx[0, 0]
        if current_pos < self.sizes.szblock:
            self.pfb_buf[:, :].flat[current_pos:] = 0.0

class IPFBProcessor:
    """Handles IPFB processing of chunks to time stream"""
    
    def __init__(self, config: ProcessingConfig, buffer_mgr: BufferManager, channels: np.ndarray, filt: cp.ndarray):
        self.config = config
        self.channels = channels
        self.filt = filt
        self.buffers = buffer_mgr
    
    def process_chunk(self, chunk: dict, ant_idx: int, pol_idx: int, start_specnum: int):
        """Process a single chunk through inverse PFB"""
        # Prepare polarization data with edge handling
        # pol_data = cp.empty((self.config.acclen + 2*self.config.cut, 2049), dtype='complex64', order='C')
        self.buffers.pol[:2*self.config.cut] = self.buffers.cut_pol[ant_idx, pol_idx]
        
        # Make continuous data
        self.buffers.pol[2*self.config.cut:] = bdc.make_continuous_gpu(
            chunk[f"pol{pol_idx}"],
            chunk['specnums'],
            start_specnum,
            self.channels[slice(None)],  # Use all channels
            self.config.acclen,
            nchans=2049
        )
        # print("pol shape", self.buffers.pol.shape)
        # raise ValueError
        # Apply inverse PFB and remove edge effects and add chunk to PFB buffer
        self.buffers.add_chunk_to_buffer(ant_idx, pol_idx, pu.cupy_ipfb(self.buffers.pol, self.filt)[self.config.cut:-self.config.cut].ravel())

    def process_dummy_chunk(self, ant_idx: int, pol_idx: int, gen_type: str='constant', specs: list = None):
        ts = cp.zeros(self.buffers.sizes.lchunk, dtype='float32')

        if gen_type =='constant':
            # specs = (value,)
            ts = cp.ones(self.buffers.sizes.lchunk, dtype='float32') * specs[0]
        elif gen_type == 'uniform':
            # specs = (min, max)
            ts = cp.random.uniform(specs[0], specs[1], size=self.buffers.sizes.lchunk).astype('float32')
        elif gen_type == 'gaussian':
            # specs = (mean, stddev)
            ts = cp.random.normal(specs[0], specs[1], size=self.buffers.sizes.lchunk).astype('float32')
        elif gen_type == 'sine':
            # specs = (frequency, sample_rate, amp, t0) # <- t0 is the time step start offset
            t = (cp.arange(self.buffers.sizes.lchunk) + specs[3]) / specs[1]
            ts = (specs[2]*cp.sin(2 * cp.pi * specs[0] * t)).astype('float32')
        else:
            ts = cp.zeros(self.buffers.sizes.lchunk, dtype='float32')

        self.buffers.add_chunk_to_buffer(ant_idx, pol_idx, ts)



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


def repfb_xcorr_avg(idxs: List[int], files: List[str], acclen: int, nchunks: int, nblock: int, 
                   chanstart: int, chanend: int, osamp: int, cut: int = 10, filt_thresh: float = 0.45,
                   cupy_win_big: Optional[cp.ndarray] = None, filt: Optional[cp.ndarray] = None, 
                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, cp.ndarray, cp.ndarray]:
    """
    Perform reverse PFB and GPU-based cross-correlation on streaming baseband data.
    
    Args:
        idxs: List of antenna indices
        files: List of baseband files per antenna
        acclen: Accumulation length in units of 4096-sample IPFB output blocks
        nchunks: Total number of IPFB output chunks to process
        nblock: Number of PFB blocks per iteration (streamed)
        chanstart, chanend: Frequency channel bounds
        osamp: Oversampling factor for RePFB
        cut: Number of rows to cut from top and bottom of IPFB
        filt_thresh: Regularization parameter for IPFB deconvolution filter
        cupy_win_big: Precomputed PFB window (optional)
        filt: Precomputed IPFB filter (optional)
        verbose: Enable verbose logging
    
    Returns:
        vis: Averaged cross-correlation matrix per time and frequency
        missing_fraction: Fraction of missing data per chunk and antenna
        freqs: Array of final frequency bins processed
        cupy_win_big: PFB window used
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
    cupy_win_big, filt = setup_filters_and_windows(config, cupy_win_big, filt)
    
    # Initialize managers
    buffer_mgr = BufferManager(config, sizes)
    ipfb_processor = IPFBProcessor(config, buffer_mgr, channels, filt)
    missing_tracker = MissingDataTracker(nant, sizes.num_of_pfbs)
    
    # Setup frequency ranges
    repfb_chanstart = channels[config.chanstart] * osamp
    repfb_chanend = channels[config.chanend] * osamp

    
    
    # Initialize output arrays
    xin = cp.empty((config.nant * config.npol, nblock, sizes.nchan), dtype='complex64', order='F')
    vis = np.zeros((config.nant * config.npol, nant * config.npol, sizes.nchan, sizes.num_of_pfbs), dtype="complex64", order="F")
    
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    
    if verbose:
        _print_debug_info(config, sizes, buffer_mgr, sizes.num_of_pfbs, nchunks)
    
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
                    output_start = ant_idx * config.npol + pol_idx # <-- still not sure
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
                output_start = ant_idx * config.npol + pol_idx # <-- still not sure
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


def repfb_test(acclen: int, nchunks: int, nblock: int,osamp: int, nant: int = 1, cut: int = 10, filt_thresh: float = 0.45, gen_type: str = 'constant') -> np.ndarray:

    """
    Perform reverse PFB and GPU-based cross-correlation on streaming baseband data.
    
    Args:
        acclen: Accumulation length in units of 4096-sample IPFB output blocks
        nchunks: Total number of IPFB output chunks to process
        nblock: Number of PFB blocks per iteration (streamed)
        osamp: Oversampling factor for RePFB
        cut: Number of rows to cut from top and bottom of IPFB
        filt_thresh: Regularization parameter for IPFB deconvolution filter
    
    Returns:
        vis: Averaged cross-correlation matrix per time and frequency
    """
     
    sr = 250e6  # Sample rate in Samples/second

    specs = {
        'constant': [1.0,], # value
        'uniform': [0.0, 1.0], # min, max
        'gaussian': [0.0, 1.0], # mean, stddev
        'sine': [5e3, sr, 1.0, 0.0]  # frequency, sample_rate, amp, t0
    }
    assert gen_type in specs, f"Invalid gen_type: {gen_type}. Must be one of {list(specs.keys())}."
    
    # Setup configuration
    config = ProcessingConfig(
        acclen=acclen, pfb_size=0, nchunks=nchunks, nblock=nblock,
        chanstart=0, chanend=1, osamp=osamp, nant=nant, cut=cut, filt_thresh=filt_thresh
    )
    sizes = BufferSizes.from_config(config)
    
    # Setup components
    cupy_win_big, _ = setup_filters_and_windows(config)
    
    # Initialize managers
    buffer_mgr = BufferManager(config, sizes)
    ipfb_processor = IPFBProcessor(config, buffer_mgr, None, None)
    
    # Setup frequency ranges
    repfb_chanstart = 0 * osamp
    repfb_chanend = 1 * osamp

    
    
    # Initialize output arrays
    xin = cp.empty((config.nant * config.npol, nblock, sizes.nchan), dtype='complex64', order='F')
    vis = np.zeros((config.nant * config.npol, nant * config.npol, sizes.nchan, sizes.num_of_pfbs), dtype="complex64", order="F")

    print("Important Values:")
    print(f"nblock: {config.nblock}, lblock: {sizes.lblock}, szblock: {sizes.szblock}")
    print(f"ipfb_size: {config.acclen}, lchunk: {sizes.lchunk}\n")
    
    print('-'*20, ' ', 0, '/', nchunks, '-'*20)
    print('PFB Buffer', buffer_mgr.pfb_idx)
    print('Remaining Buffer', buffer_mgr.rem_idx)

    job_idx = 0
    for i in range(nchunks):

        for ant_idx in range(config.nant):
            for pol_idx in range(config.npol):
                ipfb_processor.process_dummy_chunk(ant_idx, pol_idx, gen_type=gen_type, specs=specs[gen_type])

                if gen_type == 'sine':
                    specs[gen_type][-1] += sizes.lchunk  # Increment time step offset for sine wave

        print('-'*20, ' ', i+1, '/', nchunks, '-'*20)
        print('PFB Buffer', buffer_mgr.pfb_idx)
        print('Remaining Buffer', buffer_mgr.rem_idx)
                
    
        # All PFB idices should be the same after processing all antennas
        assert (buffer_mgr.pfb_idx == buffer_mgr.pfb_idx[0,0]).all()

        # We know that all antennas/pols have the same PFB index after processing so is sufficient to check one
        while buffer_mgr.pfb_idx[0,0] == sizes.szblock:
            print('='*20, 'PFB Buffer Full', '='*20)
            print('PFB Buffer', buffer_mgr.pfb_idx)
            print('Remaining Buffer', buffer_mgr.rem_idx)
            # Generate PFB output for cross-correlation
            for ant_idx in range(config.nant):
                for pol_idx in range(config.npol):
                    output_start = ant_idx * config.npol + pol_idx # <-- still not sure
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

            
            print('PFB Buffer After', buffer_mgr.pfb_idx)
            print('Remaining Buffer After', buffer_mgr.rem_idx)

            job_idx += 1

        
        if  job_idx % 100 == 0:
            print(f"Job Chunk {job_idx + 1}/{sizes.num_of_pfbs} ")
    
    # Pad buffer if incomplete
    if buffer_mgr.pfb_idx[0,0] > (config.ntap - 1)* sizes.lblock:
        print(f"Job {job_idx + 1}/{sizes.num_of_pfbs}: ")
        print(f"Buffer not Full with only {buffer_mgr.pfb_idx[0,0]} < {sizes.szblock}")

        for ant_idx in range(config.nant):
            for pol_idx in range(config.npol):
                buffer_mgr.pad_incomplete_buffer(ant_idx, pol_idx)
                output_start = ant_idx * config.npol + pol_idx # <-- still not sure
                xin[output_start, :, :] = pu.cupy_pfb(
                    buffer_mgr.pfb_buf[ant_idx, pol_idx], 
                    cupy_win_big,
                    nchan=2048 * osamp + 1, 
                    ntap=4
                )[:, repfb_chanstart:repfb_chanend]

        vis[:, :, :, job_idx] = cp.asnumpy(
            cr.avg_xcorr_all_ant_gpu(xin, config.nant, config.npol, nblock, sizes.nchan, split=1)
        )
        print("PFB buffer padded and processed")
    
    vis = np.ma.masked_invalid(vis)
    
    return vis



def _print_debug_info(config: ProcessingConfig, sizes: BufferSizes, buffer_mgr: BufferManager, n_job_chunks: int, nchunks: int):
    """Print debug information if verbose mode is enabled"""
    print(f"nblock: {config.nblock}, lblock: {sizes.lblock}")
    print(f"ipfb_size: {config.acclen}")
    print(f"window shape: {buffer_mgr.pfb_buf.shape}")
    print(f"filter shape: Not shown")  # filt shape would need to be passed
    print(f"pol: {buffer_mgr.pol.shape}")
    print(f"pfb_buf: {buffer_mgr.pfb_buf.shape}, rem_buf: {buffer_mgr.rem_buf.shape}")
    print(f"chunky inputs: {nchunks}, {sizes.lchunk}, {config.nblock}, {sizes.lblock}, {config.ntap}")
    print(f"starting {n_job_chunks} PFB Jobs over {nchunks} IPFB chunks...")
    pu.print_mem("START ITER")