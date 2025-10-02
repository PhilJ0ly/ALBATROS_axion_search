
# Philippe Joly 2025-08-07

""" 
GPU RePFB Stream 

This script implements a streaming RePFB algorithm to change the frequency resolution
of ALBATROS telescope data.
"""

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import List, Tuple, Optional
import ctypes

import sys
from os import path
sys.path.insert(0, path.expanduser("~"))

from albatros_analysis.src.utils import pfb_gpu_utils as pu
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr


@dataclass
class ProcessingConfig:
    """Configuration for RePFB processing"""
    acclen: int  # Accumulation length for IPFB inputProcessingConfig
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
    # last_pfb_nblock: int # nblock for the final PFB
    # num_of_spectra: int # output number of spectra
    
    @classmethod
    def from_config(cls, config: ProcessingConfig) -> 'BufferSizes':
        lblock = 4096 * config.osamp
        szblock = int((config.nblock + (config.ntap - 1)) * lblock)
        lchunk = 4096 * config.acclen
        nchan = (config.chanend - config.chanstart) * config.osamp
        num_of_pfbs = int(np.ceil((lchunk*config.nchunks-(config.ntap-1)*lblock)/(config.nblock*lblock)))

        # extra = (lchunk*config.nchunks-(config.ntap-1)*lblock)%(config.nblock*lblock)
        # last_pfb_nblock = int(np.ceil(extra/lblock))
        
        # if last_pfb_nblock == 0:
        #     num_of_spectra = num_of_pfbs*config.nblock
        # else:
        #     num_of_spectra = (num_of_pfbs-1)*config.nblock+last_pfb_nblock

        return cls(lblock, szblock, lchunk, nchan, num_of_pfbs)


class MeanTracker:
    """
    Handles averaging for numpy arrays

    self.sum: Sum of arrays added to each bin
    self.count: Count of valid entries per spectrum frequency bin per plasma bin
    self.counter: Count of arrays added to each bin
    """

    def __init__(self, bin_num: int, shape: Optional[Tuple[int]] = None, dtype: Optional[str] = None):
        self.bin_num = bin_num
        self.shape = shape
        self.dtype = dtype

        self.counter = np.zeros(bin_num, dtype="int64")
        self.sum = None
        self.count = None

        if shape is not None and dtype is not None:
            self.sum = np.zeros((bin_num,)+shape, dtype=dtype)
            self.count = np.zeros((bin_num,)+shape, dtype="int64")
        else:
            

    def add_to_mean(self, bin_idx, array: np.ndarray):
        assert bin_idx < self.bin_num, f"Invalid index {bin_idx} not < {self.bin_num}"

        if self.shape is None or self.dtype is None:
            # Initialize mean array according to first input
            self.shape = array.shape
            self.dtype = array.dtype

            self.mean = np.zeros((self.bin_num,)+self.shape, dtype=self.dtype)
            self.count = np.zeros((self.bin_num,)+self.shape, dtype="int64")
        else:
            # baby-proofing
            assert array.shape == self.shape, f"Invalid Mean Input. Expected {self.shape} array but got {array.shape}"
            assert array.dtype == self.dtype, f"Invalid Mean Input. Expected array of type {self.dtype} but got {array.dtype}"

        mask = ~np.ma.masked_invalid(array).mask

        #update only where valid entries
        self.count[bin_idx][mask] += 1
        self.sum[bin_idx][mask] += array[mask]
        self.counter[bin_idx] += 1

    def get_mean(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.sum/self.count, self.count, self.counter

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
    
    def reset_overlap_region(self, test_no_reset=False):
        """Set up overlap region for continuity between iterations"""
        ntap = self.config.ntap
        if not test_no_reset:
            self.pfb_buf[:, :, :ntap-1, :] = self.pfb_buf[:, :, -(ntap-1):, :].copy()
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
        self.rem_buf[ant_idx, pol_idx, :overflow_size] = self.rem_buf[ant_idx, pol_idx, available_space:rem_size].copy()
        
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
            self.pfb_buf[ant_idx, pol_idx].flat[pfb_pos:final_pos] = processed_chunk[:]
            self.pfb_idx[ant_idx, pol_idx] = final_pos

    
    def pad_incomplete_buffer(self, ant_idx: int, pol_idx: int):
        """Pad incomplete PFB buffer with zeros"""

        current_pos = self.pfb_idx[0, 0]
        if current_pos < self.sizes.szblock:
            self.pfb_buf[:, :].flat[current_pos:] = 0.0

class IPFBProcessor:
    """Handles IPFB processing of chunks to time stream"""
    
    def __init__(self, config: ProcessingConfig, buffer_mgr: BufferManager, channels: np.ndarray, channel_idxs: np.ndarray, filt: cp.ndarray):

        self.config = config
        self.channels = channels
        self.channel_idxs = channel_idxs
        self.filt = filt
        self.buffers = buffer_mgr
        
        # to test the time stream integrity of test_repfb.py
        # self.ts = cp.zeros(buffer_mgr.sizes.lblock*(buffer_mgr.sizes.num_of_pfbs*config.nblock+config.ntap-1), dtype='float32') # this was for testing ts continuity
        
    
    def process_chunk(self, chunk: dict, ant_idx: int, pol_idx: int, start_specnum: int):
        """Process a single chunk through inverse PFB"""
        self.buffers.pol[:2*self.config.cut] = self.buffers.cut_pol[ant_idx, pol_idx]
        
        # Make continuous data
        self.buffers.pol[2*self.config.cut:] = bdc.make_continuous_gpu(
            chunk[f"pol{pol_idx}"],
            chunk['specnums'],
            start_specnum,
            self.channels[self.channel_idxs],
            self.config.acclen,
            nchans=2049
        )
        
        self.buffers.add_chunk_to_buffer(ant_idx, pol_idx, pu.cupy_ipfb(self.buffers.pol, self.filt)[self.config.cut:-self.config.cut].ravel())

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
    channel_idxs = antenna_objs[0].obj.channel_idxs
    return antenna_objs, channels, channel_idxs

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
                   window: Optional[cp.ndarray] = None, filt: Optional[cp.ndarray] = None, 
                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, cp.ndarray, cp.ndarray]:
    """
    Perform oversampling PFB and GPU-based cross-correlation on streaming baseband data.
    
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
        acclen=acclen, nchunks=nchunks, nblock=nblock,
        chanstart=chanstart, chanend=chanend, osamp=osamp, nant=nant, cut=cut, filt_thresh=filt_thresh
    )
    sizes = BufferSizes.from_config(config)
    
    # Setup components
    antenna_objs, channels, channel_idxs = setup_antenna_objects(idxs, files, config)
    window, filt = setup_filters_and_windows(config, window, filt)
    
    assert config.chanstart == antenna_objs[0].obj.chanstart, "chan start issues"
    # Initialize managers
    buffer_mgr = BufferManager(config, sizes)
    ipfb_processor = IPFBProcessor(config, buffer_mgr, channels, channel_idxs, filt)
    missing_tracker = MissingDataTracker(nant, sizes.num_of_pfbs)
    
    # Setup frequency ranges
    repfb_chanstart = channels[config.chanstart] * osamp
    repfb_chanend = channels[config.chanend] * osamp
    
    # Initialize output arrays
    xin = cp.empty((config.nant * config.npol, nblock, sizes.nchan), dtype='complex64', order='F') 
    # Note that these HAVE TO be Fortran contiguous because of the hard-coded strides in the xcorr_avg
    vis = np.zeros((config.nant * config.npol, config.nant * config.npol, sizes.nchan, sizes.num_of_pfbs), dtype="complex64", order="F")
    
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    
    if verbose:
        _print_debug_info(config, sizes, buffer_mgr, sizes.num_of_pfbs, nchunks)
    
    job_idx = 0
    
    for i, chunks in enumerate(zip(*antenna_objs)):
        pfbed = False

        for ant_idx in range(config.nant):
            chunk = chunks[ant_idx]

            pfb_blocks_affected = ((buffer_mgr.pfb_idx[ant_idx, 0] + sizes.lchunk) // sizes.lblock) % nblock + 1
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
                        window,
                        nchan=2048 * osamp + 1, 
                        ntap=4
                    )[:, repfb_chanstart:repfb_chanend]

            vis[:, :, :, job_idx]= cp.asnumpy(
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

    # blocks_left = max(0,np.ceil((buffer_mgr.pfb_idx[0,0]-(config.ntap - 1)* sizes.lblock)/sizes.lblock))
    # assert blocks_left == sizes.last_pfb_nblock, f"last_pfb_nblock wrong: expected {sizes.last_pfb_nblock} but got {blocks_left}"

    if buffer_mgr.pfb_idx[0,0] > (config.ntap - 1)* sizes.lblock:
        if verbose:
            print(f"Job {job_idx + 1}: ")
        # print("Extra", buffer_mgr.pfb_idx[0,0] - (config.ntap-1+sizes.last_pfb_nblock-1)*sizes.lblock, "<=", sizes.lblock)

        for ant_idx in range(config.nant):
                
            missing_tracker.calculate_job_average(ant_idx, job_idx)

            for pol_idx in range(config.npol):
                buffer_mgr.pad_incomplete_buffer(ant_idx, pol_idx)
                output_start = ant_idx * config.npol + pol_idx 
                xin[output_start, :, :] = pu.cupy_pfb(
                    buffer_mgr.pfb_buf[ant_idx, pol_idx], 
                    window,
                    nchan=2048 * osamp + 1, 
                    ntap=4
                )[:, repfb_chanstart:repfb_chanend]
        
        vis[:, :, :, job_idx] = cp.asnumpy(
            cr.avg_xcorr_all_ant_gpu(xin, config.nant, config.npol, nblock, sizes.nchan, split=1)
        )

        print("Paded and Processed Incomplete Buffer")
    
    # if verbose:
    print("=" * 30)
    print(f"Completed {sizes.num_of_pfbs}/{sizes.num_of_pfbs} Job Chunks")
    print("=" * 30)
    
    vis = np.ma.masked_invalid(vis)
    freqs = np.arange(repfb_chanstart, repfb_chanend)

    # print(50*"=")
    # print("IPFB time count", ipfb_processor.time_counter)
    # print("nchunk x time chunk", nchunks*ipfb_processor.t_chunk)
    # print("Actual time", )
    
    return vis, missing_tracker.missing_fraction, freqs, window, filt


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