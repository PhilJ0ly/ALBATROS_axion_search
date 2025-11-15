# Philippe Joly 2025-11-15

import numpy as np
import os
import sys
from os import path

import numpy as np
from typing import List, Tuple, Optional


class MedianTracker:
    """
    Tracks arrays and calculates medians across huge datasets by storing data on disk.
    
    Arrays are stored in F-major order (Fortran order) so that values from subsequent
    arrays at the same frequency index are contiguous in memory for efficient median calculation.
    """
    
    def __init__(self, bin_num: int, temp_dir: str, max_size: int, shape: Optional[Tuple[int]] = None, dtype: Optional[str] = None):
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.bin_num = bin_num
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.temp_dir = temp_dir
        self.max_size = max_size

        self.bin_count = np.zeros(bin_num, dtype="int64")
        self.bin_tot_count = np.zeros(bin_num, dtype="int64")
        self.fine_counter = None
        
        if shape is not None and dtype is not None:
            self._setup_buf()
    
    def _setup_buf(self):
        self.fine_counter = np.zeros((bin_num,) + shape, dtype="int64")
        self._create_storage_files()
        
    def _create_storage_files(self):
        """Create pre-allocated binary files"""

        # clear temp directory Maybe change this
        if path.exists(self.temp_dir):
            os.remove("*.bin")
        else:
            os.makedirs(self.temp_dir)

        bytes_per_value = self.dtype.itemsize
        file_size = self.shape[-1] * self.bin_tot_count * bytes_per_value
        
        for bin_idx in range(self.bin_num):
            if self.bin_tot_count[bin_idx] > 0:

                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        filepath = self._get_filepath(bin_idx, i, j)
                        
                        with open(filepath, 'wb') as f:
                            # Write zeros to pre-allocate space
                            f.write(b'\x00' * file_size[bin_idx]) # maybe choose other carachter than 0? to mark unfilled spots
    
    def _get_filepath(self, bin_idx, i, j):
        """Get the filepath for a given (bin_idx, a, b) index."""
        return self.output_dir / f'median_bin_{bin_idx+1}_pol_{i}{j}.bin'
    
    def add_to_bin_tot_count(self, bin_idx: int):
        self.bin_tot_count[bin_idx] += 1   

    def add_to_buf(self, bin_idx, array: np.ndarray):
        assert bin_idx < self.bin_num and bin_idx >= 0, f"Invalid bin index {bin_idx}"

        if self.shape is None or self.dtype is None:
            # Initialize mean array according to first input
            self.shape = array.shape
            self.dtype = np.dtype(array.dtype)
            self._setup_buf()
        else:
            # baby-proofing
            assert array.shape == self.shape, f"Invalid Mean Input. Expected {self.shape} array but got {array.shape}"
            assert array.dtype == self.dtype, f"Invalid Mean Input. Expected array of type {self.dtype} but got {array.dtype}"


        mask = ~np.ma.masked_invalid(array).mask
        self.fine_counter[bin_idx][mask] += 1
        self.bin_count[bin_idx] += 1

        if self.bin_count[bin_idx] > self.bin_tot_count[bin_idx]:
            print(f"Trying to add more arrays to bin {bin_idx} than expected: {self.bin_count[bin_idx]} > {self.bin_tot_count[bin_idx]}")
            return 

        # Flush to Disk
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                filepath = self._get_filepath(i, j)
            
                # Calculate offset: we want to write at position array_index for each freq
                # In F-major order: freq varies slowest, array_index varies fastest
                # So data layout is: [arr0_freq0, arr1_freq0, ..., arr0_freq1, arr1_freq1, ...]
            
                freq_values = array[i, j, :]  # Shape: (nfreq,)
                
                with open(filepath, 'r+b') as f:
                    offset = self.bin_count[bin_idx] * self.dtype.itemsize
                    additional_offset = self.total_arrays[bin_idx]*self.dtype.itemsize
                    for freq_idx in range(self.nfreq):
                        # Calculate position in file (F-major ordering)
                        offset += additional_offset
                        f.seek(offset)
                        f.write(freq_values[freq_idx].tobytes())

    def get_median(self, batch_freq: Optional[int] = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        """
        Compute median for each frequency channel in a batch.
        Returns:  array of medians
        """
        nfreq = self.shape[-1]
        
        median_array = np.zeros((self.bin_num,) + self.shape, dtype=self.dtype)
        for bin_idx in range(self.bin_num):
            if self.bin_count[bin_idx] > 0:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        filepath = self._get_filepath(bin_idx, i, j)
                        
                        batched_median_from_disk(
                            filepath, self.bin_tot_count[bin_idx], nfreq, batch_freq, out=median_array[bin_idx, i, j, :] # <- not sure about this
                        )

    def batched_median_from_disk(filename, ntimes, nfreq, batch_size, n_workers=os.cpu_count(), out=None):
        """
        Compute median across time axis for a large (ntimes, nfreq) array
        stored in F-contiguous order on disk.
        Each batch runs on a separate core.
        
        Parameters:
        -----------
        filename : str
            Path to the binary file
        ntimes : int
            Number of time samples
        nfreq : int
            Number of frequency channels
        batch_size : int
            Number of frequency channels to process per batch
        n_workers : int, optional
            Number of parallel workers (cores). Defaults to number of CPUs.
        
        Returns:
        --------
        medians : ndarray of shape (nfreq,)
            Median value for each frequency channel
        """
        
        # Create list of batch arguments
        batch_args = []
        for batch_start in range(0, nfreq, batch_size):
            batch_end = min(batch_start + batch_size, nfreq)
            batch_args.append((filename, ntimes, nfreq, batch_start, batch_end))
        
        # Result array
        if out is None:
            out = np.empty(nfreq, dtype=self.dtype)
        
        # Process batches in parallel, one batch per core
        print(f"Processing {len(batch_args)} batches across {n_workers or os.cpu_count()} cores...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = executor.map(process_batch, batch_args)
            for batch_start, batch_end, batch_medians in results:
                out[batch_start:batch_end] = batch_medians
        
        return out

    def process_batch(args):
        """
        Process a single batch - designed to run on one core.
        """
        filename, ntimes, nfreq, batch_start, batch_end = args
        
        itemsize = self.dtype.itemsize
        batch_nfreq = batch_end - batch_start
        
        # Load this batch from disk
        batch_median = np.empty(batch_nfreq, dtype=self.dtype)
        
        with open(filename, 'rb') as f:
            offset = batch_start * ntimes * itemsize
            additional_offset = batch_nfreq * itemsize
            for freq_idx in range(batch_nfreq):
                f.seek(offset)
                batch_median[freq_idx] = np.median(np.fromfile(f, dtype=dtype, count=ntimes))
                offset += additional_offset

        return batch_start, batch_end, batch_median










# Example usage
if __name__ == "__main__":
    # Example: 2x3 grid with 100 frequency points, 5 total arrays
    shape = (2, 3, 100)
    dtype = np.float32
    total_arrays = 5
    
    tracker = MedianTracker(shape, dtype, total_arrays, output_dir='test_median')
    
    # Add arrays one by one
    for arr_idx in range(total_arrays):
        # Simulate an array (in practice, this would be your large array)
        array = np.random.randn(*shape).astype(dtype) + arr_idx
        tracker.add_array(array)
        print(f"Added array {arr_idx + 1}/{total_arrays}")
    
    # Calculate median for a specific position
    median_0_0 = tracker.calculate_median(0, 0)
    print(f"\nMedian at position (0, 0): shape={median_0_0.shape}")
    
    # Calculate all medians
    all_medians = tracker.calculate_all_medians()
    print(f"All medians: shape={all_medians.shape}")
    
    # Cleanup
    tracker.cleanup()
    print("\nCleanup complete")