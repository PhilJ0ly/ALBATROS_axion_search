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
    lblock: int  # Length of each PFB block
    szblock: int  # Size of streaming block buffer
    lchunk: int  # Length of time stream chunk after IPFB
    nchan: int  # Number of channels after oversampling
    
    @classmethod
    def from_config(cls, config: ProcessingConfig) -> 'BufferSizes':
        lblock = 4096 * config.osamp
        szblock = int((config.nblock + (config.ntap - 1)) * lblock)
        lchunk = 4096 * config.acclen
        nchan = (config.chanend - config.chanstart) * config.osamp
        return cls(lblock, szblock, lchunk, nchan)


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
        if hasattr(self, '_first_iteration') and not self._first_iteration:
            # Keep overlap from previous iteration
            ntap = self.config.ntap
            self.pfb_buf[:, :, :ntap-1, :] = self.pfb_buf[:, :, -(ntap-1):, :]
            self.pfb_idx[:, :] = (ntap - 1) * self.sizes.lblock
        else:
            self._first_iteration = False
    
    def add_remaining_to_pfb_buffer(self, ant_idx: int, pol_idx: int) -> bool:
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
        
        return self.pfb_idx[ant_idx, pol_idx] == self.sizes.szblock
    
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
    
    def add_chunk_to_buffer(self, ant_idx: int, pol_idx: int, processed_chunk: cp.ndarray) -> bool:
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
            return True
        else:
            # Normal case
            self.pfb_buf[ant_idx, pol_idx].flat[pfb_pos:final_pos] = processed_chunk
            self.pfb_idx[ant_idx, pol_idx] = final_pos
            return final_pos == self.sizes.szblock
    
    def pad_incomplete_buffer(self, ant_idx: int, pol_idx: int):
        """Pad incomplete PFB buffer with zeros"""

        current_pos = self.pfb_idx[ant_idx, pol_idx]
        if current_pos < self.sizes.szblock:
            self.pfb_buf[ant_idx, pol_idx].flat[current_pos:] = 0.0

class IPFBProcessor:
    """Handles IPFB processing of chunks to time stream"""
    
    def __init__(self, config: ProcessingConfig, buffer_mgr: BufferManager, channels: np.ndarray, filt: cp.ndarray):
        self.config = config
        self.channels = channels
        self.filt = filt
        self.buffers = buffer_mgr
    
    def process_chunk(self, chunk: dict, ant_idx: int, pol_idx: int, start_specnum: int) -> bool:
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
        buffer_full = self.buffers.add_chunk_to_buffer(ant_idx, pol_idx, pu.cupy_ipfb(self.buffers.pol, self.filt)[self.config.cut:-self.config.cut].ravel())
        return buffer_full


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
    
    def __init__(self, config: ProcessingConfig, total_samples: int):
        self.config = config
        # Initialize output arrays for time stream data
        self.time_streams = np.zeros((config.nant, config.npol, total_samples), dtype='float32')
        self.sample_idx = np.zeros((config.nant, config.npol), dtype=np.uint64)
    
    def add_chunk(self, ant_idx: int, pol_idx: int, processed_chunk: cp.ndarray):
        """Add processed time stream chunk to output array"""
        chunk_size = len(processed_chunk)
        start_idx = self.sample_idx[ant_idx, pol_idx]
        end_idx = start_idx + chunk_size
        
        # Convert to numpy and store
        self.time_streams[ant_idx, pol_idx, start_idx:end_idx] = cp.asnumpy(processed_chunk)
        self.sample_idx[ant_idx, pol_idx] = end_idx


def plan_chunks(nchunks: int, lchunk: int, nblock: int, lblock: int, ntap: int) -> List[Tuple[int, Optional[int]]]:
    """
    Plan job chunks to efficiently use buffers.
    Returns list of (start_chunk, end_chunk) tuples.
    """
    ranges = []
    stride_size = nblock * lblock
    overlap_size = (ntap - 1) * lblock
    total_needed = stride_size + overlap_size
    
    remainder = 0
    cur_chunk = 0
    
    while cur_chunk < nchunks:
        start_chunk = cur_chunk
        
        # Accumulate chunks until we have enough samples
        while remainder < total_needed and cur_chunk < nchunks:
            remainder += lchunk
            cur_chunk += 1
        
        if remainder >= total_needed:
            ranges.append((start_chunk, cur_chunk))
            remainder -= stride_size
        else:
            # Handle final incomplete chunk
            if remainder > 0:
                ranges.append((start_chunk, cur_chunk))
            break
    
    return ranges


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


def fill_pfb_buffer(ant_idx: int, pol_idx: int, subjobs: List, buffer_mgr: BufferManager,
                        ipfb_processor: IPFBProcessor, missing_tracker: MissingDataTracker,
                        job_idx: int, start_specnum: int, channels: np.ndarray, 
                        config: ProcessingConfig, sizes: BufferSizes, verbose: bool):
    """Process a single antenna/polarization combination"""
    
    # Handle any remaining data from previous iteration
    buffer_full = buffer_mgr.add_remaining_to_pfb_buffer(ant_idx, pol_idx)
    
    if buffer_full:
        return  # Buffer already full, nothing more to do
    
    for chunk_idx, chunks in enumerate(subjobs):
        assert not buffer_full
        chunk = chunks[ant_idx]
        
        # Track missing data (only for first pol to avoid duplication)
        if pol_idx == 0:
            pfb_blocks_affected = ((buffer_mgr.pfb_idx[ant_idx, pol_idx] + sizes.lchunk) // sizes.lblock) % config.nblock + 1
            missing_tracker.add_chunk_info(ant_idx, chunk, config.acclen, pfb_blocks_affected)
        
        buffer_full = ipfb_processor.process_chunk(chunk, ant_idx, pol_idx, start_specnum)
        
    
    # Calculate missing data average for this job
    if pol_idx == 0:
        missing_tracker.calculate_job_average(ant_idx, job_idx)
    
    # Pad buffer if incomplete
    if buffer_mgr.pfb_idx[ant_idx, pol_idx] != sizes.szblock or not buffer_full:
        if verbose:
            pokjhgvcxcvbnm=1

        print(f"Job {job_idx + 1} (Ant {ant_idx}, pol {pol_idx}): ")
        print(f"Buffer Full: {buffer_full} with only {buffer_mgr.pfb_idx[ant_idx, pol_idx]} < {sizes.szblock}")\

        buffer_mgr.pad_incomplete_buffer(ant_idx, pol_idx)


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
    
    antenna_objs, channels = setup_antenna_objects(idxs, files, config)
    cupy_win_big, filt = setup_filters_and_windows(config, cupy_win_big, filt)
    
    # Initialize managers
    buffer_mgr = BufferManager(config, sizes)
    # print(buffer_mgr.pfb_buf.shape, "pfb_buf shape")
    # raise ValueError
    ipfb_processor = IPFBProcessor(config, buffer_mgr, channels, filt)
    job_chunks = plan_chunks(nchunks, sizes.lchunk, nblock, sizes.lblock, config.ntap)
    missing_tracker = MissingDataTracker(nant, len(job_chunks))
    
    repfb_chanstart = channels[config.chanstart] * osamp
    repfb_chanend = channels[config.chanend] * osamp
    
    xin = cp.empty((config.nant * config.npol, nblock, sizes.nchan), dtype='complex64', order='F')
    vis = np.zeros((config.nant * config.npol, nant * config.npol, sizes.nchan, len(job_chunks)), dtype="complex64", order="F")
    
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    jobs = list(zip(*antenna_objs))
    
    if verbose:
        _print_debug_info(config, sizes, buffer_mgr, len(job_chunks), nchunks)
    
    total_time = 0.0
    for job_idx, (chunk_start, chunk_end) in enumerate(job_chunks):
        start_time = time.perf_counter()
        
        buffer_mgr.reset_overlap_region()
        subjobs = jobs[chunk_start:chunk_end] if chunk_end is not None else jobs[chunk_start:] # check if this is right
        
        # Process each antenna and polarization
        for ant_idx in range(config.nant):
            for pol_idx in range(config.npol):
                fill_pfb_buffer(
                    ant_idx, pol_idx, subjobs, buffer_mgr, ipfb_processor,
                    missing_tracker, job_idx, start_specnums[ant_idx], 
                    channels, config, sizes, verbose
                )
                
    
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
        
        end_time = time.perf_counter()
        total_time += end_time - start_time
        
        if verbose and job_idx % 100 == 0:
            print(f"Job Chunk {job_idx + 1}/{len(job_chunks)}, avg time {total_time/(job_idx + 1):.4f} s")
    
    if verbose:
        print("=" * 30)
        print(f"Completed {len(job_chunks)}/{len(job_chunks)} Job Chunks")
        print(f"avg time per job: {total_time/len(job_chunks):.4f} s")
        print("=" * 30)
    
    vis = np.ma.masked_invalid(vis)
    freqs = np.arange(repfb_chanstart, repfb_chanend)
    
    return vis, missing_tracker.missing_fraction, freqs, cupy_win_big, filt


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
    
    antenna_objs, channels = setup_antenna_objects(idxs, files, config)
    _, filt = setup_filters_and_windows(config, None, filt)
    
    # Calculate total samples for output
    total_samples = nchunks * (acclen - 2*cut) * 4096  # samples per chunk after cutting edges
    
    timestream_mgr = TimeStreamManager(config, total_samples)
    missing_tracker = MissingDataTracker(nant, nchunks)
    
    pol_buffer = cp.empty((acclen + 2*cut, 2049), dtype='complex64', order='C')
    cut_pol = cp.zeros((nant, 2, 2*cut, 2049), dtype='complex64', order='C')
    
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    jobs = list(zip(*antenna_objs))
    
    if verbose:
        print(f"Processing {nchunks} chunks for {nant} antennas")
        print(f"Accumulation length: {acclen}, Cut: {cut}")
        print(f"Total output samples per antenna/pol: {total_samples}")
        print("Starting IPFB time stream processing...")
    
    total_time = 0.0
    for chunk_idx, chunks in enumerate(jobs):
        start_time = time.perf_counter()
        
        for ant_idx in range(nant):
            chunk = chunks[ant_idx]
            
            # Track missing data
            missing_pct = (1 - len(chunk["specnums"]) / acclen) * 100
            missing_tracker.missing_count[ant_idx].append([missing_pct, 1])
            missing_tracker.missing_fraction[ant_idx, chunk_idx] = missing_pct
            
            for pol_idx in range(2):  # npol = 2
                pol_buffer[:2*cut] = cut_pol[ant_idx, pol_idx]
                
                pol_buffer[2*cut:] = bdc.make_continuous_gpu(
                    chunk[f"pol{pol_idx}"],
                    chunk['specnums'],
                    start_specnums[ant_idx],
                    channels[slice(None)],
                    acclen,
                    nchans=2049
                )
                
                ipfb_result = pu.cupy_ipfb(pol_buffer, filt)
                processed_chunk = ipfb_result[cut:-cut].ravel()
                
                timestream_mgr.add_chunk(ant_idx, pol_idx, processed_chunk)
                
                # Update edge data for next iteration
                cut_pol[ant_idx, pol_idx, :, :] = pol_buffer[-2*cut:]
        
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
    
    return timestream_mgr.time_streams, missing_tracker.missing_fraction, filt


def _print_debug_info(config: ProcessingConfig, sizes: BufferSizes, buffer_mgr: BufferManager, n_job_chunks: int, nchunks: int):
    """Print debug information if verbose mode is enabled"""
    print(f"nblock: {config.nblock}, lblock: {sizes.lblock}")
    print(f"ipfb_size: {config.acclen}")
    print(f"window shape: {buffer_mgr.pfb_buf.shape}")
    print(f"filter shape: Not shown")  # filt shape would need to be passed
    print(f"pol: {buffer_mgr.pol.shape}")