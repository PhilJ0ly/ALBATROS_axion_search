# Philippe Joly 2025-11-15

import numpy as np
import os
import sys
from os import path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional
import struct
import mmap
import time


class MedianTracker:
    """
    Tracks arrays and calculates medians across huge datasets by storing data on disk.
    
    Arrays are stored in F-major order (Fortran order) so that values from subsequent
    arrays at the same frequency index are contiguous in memory for efficient median calculation.
    """
    
    def __init__(self, bin_num: int, temp_dir: str, plasma_bin_edges: np.ndarray, t_init: int, t_final: int, t_chunk: float = 0.0):
        
        self.bin_num = bin_num
        self.temp_dir = temp_dir
        self.shape = None
        self.dtype = None
        self.bin_count = np.zeros(bin_num, dtype="uint64")
        self.bin_tot_count = np.zeros(bin_num, dtype="uint64")
        self.fine_counter = None

        # Header info
        self.plasma_bin = plasma_bin_edges.astype("float64")
        self.t_init = np.uint64(t_init)
        self.t_final = np.uint64(t_final)
        self.t_chunk = np.float64(t_chunk)

        self.header_format = '<ddQQd4sQQ'  # Format: 2x float64 (plasma), 2x uint64 (time), 1x float64 (t_chunk), 1x 4-char string (dtype), 2x uint64 (bin counts)
        self.header_size = struct.calcsize(self.header_format)
        self.bin_count_offset  = self.header_size - self.bin_count.itemsize

    def read_file_header(self, bin_idx: int, i: int, j: int) -> dict:
        """Read data and header from a given file."""
        filepath = self._get_filepath(bin_idx, i, j)
        with open(filepath, 'rb') as f:
            header_bytes = f.read(self.header_size)
            header = struct.unpack(self.header_format, header_bytes)

            plasma_low = header[0]
            plasma_high = header[1]
            t_init = header[2]
            t_final = header[3]
            t_chunk = header[4]
            dtype_str = header[5].decode('ascii').strip('\x00')
            bin_tot_count = header[6]
            bin_count = header[7]

            dtype = np.dtype(dtype_str)

        header_info = {
            'plasma_low': plasma_low,
            'plasma_high': plasma_high,
            't_init': t_init,
            't_final': t_final,
            't_chunk': t_chunk,
            'dtype': dtype,
            'bin_tot_count': bin_tot_count,
            'bin_count': bin_count
        }

        return header_info

    def _setup_buf(self):
        self.fine_counter = np.zeros((self.bin_num,) + self.shape, dtype="int64")
        self._create_storage_files()

    def cleanup(self):
        # clear temp directory Maybe change this
        if path.exists(self.temp_dir):
            for bin_idx in range(self.bin_num*2):
                for i in range(self.shape[0]*2):
                    for j in range(self.shape[1]*2):
                        filepath = self._get_filepath(bin_idx, i, j)
                        if path.exists(filepath):
                            os.remove(filepath)
        else:
            os.makedirs(self.temp_dir)  

    def _create_storage_files(self):
        """Create pre-allocated binary files"""
        t1 = time.time()
        self.cleanup()

        size_of_file = self.header_size + self.shape[-1] * self.bin_tot_count * self.dtype.itemsize  # size in bytes

        # Too slow:
        #  Write the NaN pattern to pre-allocate
        # nan_value = np.array(np.nan, dtype=self.dtype)
        # nan_bytes = nan_value.tobytes()

        dtype_str_bytes = self.dtype.str.encode('ascii')

        for bin_idx in range(self.bin_num):
            if self.bin_tot_count[bin_idx] > 0:

                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        filepath = self._get_filepath(bin_idx, i, j)
                        
                        with open(filepath, 'wb') as f:
                            f.write(struct.pack('d', self.plasma_bin[bin_idx]))  # low plasma frequency bin edge
                            f.write(struct.pack('d', self.plasma_bin[bin_idx+1]))  # high plasma edge 

                            f.write(struct.pack('Q', self.t_init))  # t_init
                            f.write(struct.pack('Q', self.t_final))  # t_final

                            f.write(struct.pack('d', self.t_chunk))  # t_chunk

                            f.write(struct.pack('4s', dtype_str_bytes))  # dtype string

                            f.write(struct.pack('Q', self.bin_tot_count[bin_idx]))  # total expected count
                            f.write(struct.pack('Q', self.bin_count[bin_idx]))  # current count

                            f.seek(size_of_file[bin_idx] - 1)
                            f.write(b'\x00')  # Allocate file size

                            # too slow:
                            # f.write(b'\x00' * file_size[bin_idx]) # maybe choose other carachter than 0? to mark unfilled spots
                            # f.write(nan_bytes * items_per_file[bin_idx])

        print(f"Created storage files in {self.temp_dir} ({time.time()-t1:.2f} s)")
    
    def _get_filepath(self, bin_idx, i, j):
        """Get the filepath for a given (bin_idx, a, b) index."""
        return path.join(self.temp_dir, f'median_bin_{bin_idx+1}_pol_{i}{j}.bin')
    
    def add_to_bin_tot_count(self, bin_idx: int):
        self.bin_tot_count[bin_idx] += 1   

    def add_to_buf(self, bin_idx, array: np.ndarray):
        # print("Adding to bin", bin_idx)
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

        t1 = time.time()
        mask = ~np.ma.masked_invalid(array).mask
        self.fine_counter[bin_idx][mask] += 1
        # print("masked in", time.time()-t1, "s")

        if self.bin_count[bin_idx] >= self.bin_tot_count[bin_idx]:
            print(f"Trying to add more arrays to bin {bin_idx} than expected: {self.bin_count[bin_idx]} > {self.bin_tot_count[bin_idx]}")
            return 

        # Flush to Disk
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                filepath = self._get_filepath(bin_idx, i, j)
            
                # Calculate offset: we want to write at position array_index for each freq
                # In F-major order: freq varies slowest, array_index varies fastest
                # So data layout is: [arr0_freq0, arr1_freq0, ..., arr0_freq1, arr1_freq1, ...]
                
                t1 = time.time()
                # print("start writing...")
                with open(filepath, 'r+b') as f:
                    # Update header count
                    f.seek(self.bin_count_offset)
                    f.write(struct.pack('Q', self.bin_count[bin_idx]+1))  # Update current count

                    offset = self.bin_count[bin_idx] * self.dtype.itemsize + self.header_size
                    additional_offset = self.bin_tot_count[bin_idx]*self.dtype.itemsize
                    # print("adjusted header")
                    for freq_idx in range(self.shape[-1]):
                        # Calculate position in file (F-major ordering)
                        f.seek(offset)
                        f.write(array[i, j, freq_idx].tobytes())
                        offset += additional_offset

                t = time.time()-t1
                head_bin_count = self.read_file_header(bin_idx, i, j)['bin_count']
                # print(f"Added array to bin {bin_idx} {i}{j} (count={head_bin_count}) in {t:.2f} s")
                # raise NotImplementedError("Debugging - remove this line when working")
        
        self.bin_count[bin_idx] += 1

    def add_to_bufv2(self, bin_idx, array: np.ndarray):
        """Optimized version using memory-mapped files for maximum performance"""
        assert bin_idx < self.bin_num and bin_idx >= 0, f"Invalid bin index {bin_idx}"
        
        if self.shape is None or self.dtype is None:
            self.shape = array.shape
            self.dtype = np.dtype(array.dtype)
            self._setup_buf()
        else:
            assert array.shape == self.shape, f"Invalid Mean Input. Expected {self.shape} array but got {array.shape}"
            assert array.dtype == self.dtype, f"Invalid Mean Input. Expected array of type {self.dtype} but got {array.dtype}"
        
        # Update mask counter
        mask = ~np.ma.masked_invalid(array).mask
        self.fine_counter[bin_idx][mask] += 1
        
        if self.bin_count[bin_idx] >= self.bin_tot_count[bin_idx]:
            print(f"Trying to add more arrays to bin {bin_idx} than expected: {self.bin_count[bin_idx]} > {self.bin_tot_count[bin_idx]}")
            return
        
        # Pre-calculate common values
        current_count = self.bin_count[bin_idx]
        itemsize = self.dtype.itemsize
        freq_stride = self.bin_tot_count[bin_idx] * itemsize
        base_offset = current_count * itemsize + self.header_size
        new_bin_count_bytes = struct.pack('Q', current_count + 1)
        
        # Process each file using memory mapping
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                filepath = self._get_filepath(bin_idx, i, j)
                pixel_data = array[i, j, :]  # Extract frequency data for this pixel
                
                # Memory-map the file for fast random access
                with open(filepath, 'r+b') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
                        # Update header count
                        mm[self.bin_count_offset:self.bin_count_offset + 8] = new_bin_count_bytes
                        
                        # Write all frequencies in F-major order
                        # Layout: [arr0_freq0, arr1_freq0, ..., arr0_freq1, arr1_freq1, ...]
                        for freq_idx in range(self.shape[-1]):
                            offset = base_offset + freq_idx * freq_stride
                            value_bytes = pixel_data[freq_idx].tobytes()
                            mm[offset:offset + itemsize] = value_bytes
        
        self.bin_count[bin_idx] += 1


    def add_to_buf_vectorized(self, bin_idx, array: np.ndarray):
        """Alternative: Vectorized version that writes multiple frequencies at once"""
        assert bin_idx < self.bin_num and bin_idx >= 0, f"Invalid bin index {bin_idx}"
        
        if self.shape is None or self.dtype is None:
            self.shape = array.shape
            self.dtype = np.dtype(array.dtype)
            self._setup_buf()
        else:
            assert array.shape == self.shape, f"Invalid Mean Input. Expected {self.shape} array but got {array.shape}"
            assert array.dtype == self.dtype, f"Invalid Mean Input. Expected array of type {self.dtype} but got {array.dtype}"
        
        mask = ~np.ma.masked_invalid(array).mask
        self.fine_counter[bin_idx][mask] += 1
        
        if self.bin_count[bin_idx] >= self.bin_tot_count[bin_idx]:
            print(f"Trying to add more arrays to bin {bin_idx} than expected")
            return
        
        current_count = self.bin_count[bin_idx]
        itemsize = self.dtype.itemsize
        freq_stride = self.bin_tot_count[bin_idx] * itemsize
        base_offset = current_count * itemsize + self.header_size
        new_bin_count_bytes = struct.pack('Q', current_count + 1)
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                filepath = self._get_filepath(bin_idx, i, j)
                pixel_data = array[i, j, :]
                
                with open(filepath, 'r+b') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE) as mm:
                        # Update header
                        mm[self.bin_count_offset:self.bin_count_offset + 8] = new_bin_count_bytes
                        
                        # Convert all frequency data to contiguous bytes
                        pixel_bytes = pixel_data.tobytes()
                        
                        # Write each frequency value to its correct position
                        for freq_idx in range(self.shape[-1]):
                            offset = base_offset + freq_idx * freq_stride
                            start_byte = freq_idx * itemsize
                            mm[offset:offset + itemsize] = pixel_bytes[start_byte:start_byte + itemsize]
        
        self.bin_count[bin_idx] += 1

    @staticmethod
    def process_batch(args: Tuple[str, str, np.uint64, np.uint64, int, int, int]) -> Tuple[int, int, np.ndarray]:
        """
        Process a single batch - designed to run on one core.
        """
        filename, dtype, bin_count, bin_tot_count, batch_start, batch_end, header_size = args
        dtype = np.dtype(dtype)
        item_size = dtype.itemsize
        batch_nfreq = batch_end - batch_start
        
        # Load this batch from disk
        batch_median = np.empty(batch_nfreq, dtype=dtype)
        
        with open(filename, 'rb') as f:
            offset = batch_start * bin_tot_count * item_size + header_size
            additional_offset = bin_tot_count * item_size
            for freq_idx in range(batch_nfreq):
                f.seek(offset)
                batch_median[freq_idx] = np.nanmedian(np.abs(np.fromfile(f, dtype=dtype, count=bin_count))) # This gives back median of magnitude (Ask Mohan what to adjust for future)
                # print(np.fromfile(f, dtype=self.dtype, count=ntimes))
                # print(np.median(np.fromfile(f, dtype=self.dtype, count=ntimes)))
                # sys.exit(0)
                offset += additional_offset

        return batch_start, batch_end, batch_median

    def batched_median_from_disk(self, filename: str, bin_idx: int, batch_size: int, n_workers=os.cpu_count(), out=None):
        """
        Compute median across time axis for a large (ntimes, nfreq) array
        stored in F-contiguous order on disk.
        Each batch runs on a separate core.
        
        Parameters:
        -----------
        filename : str
            Path to the binary file
        batch_size : int
            Number of frequency channels to process per batch
        n_workers : int, optional
            Number of parallel workers (cores). Defaults to number of CPUs.
        
        Returns:
        --------
        out : ndarray of shape (nfreq,)
            Median value for each frequency channel
        """

        nfreq = self.shape[-1]
        
        # Create list of batch arguments
        batch_args = []
        for batch_start in range(0, nfreq, batch_size):
            batch_end = min(batch_start + batch_size, nfreq)
            batch_args.append((filename, self.dtype, self.bin_count[bin_idx], self.bin_tot_count[bin_idx], batch_start, batch_end, self.header_size))
        
        # Result array
        if out is None:
            out = np.empty(nfreq, dtype=self.dtype)
        
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = executor.map(self.process_batch, batch_args)
            for batch_start, batch_end, batch_medians in results:
                out[batch_start:batch_end] = batch_medians
        
        return out

    def get_median(self, batch_freq: Optional[int] = 10000, n_workers: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        """
        Compute median for each frequency channel in a batch.
        Returns:  array of medians
        """
        nfreq = self.shape[-1]
        
        median_array = np.zeros((self.bin_num,) + self.shape, dtype=self.dtype)
        print(f"Processing {nfreq//batch_freq+1} batches across {n_workers or os.cpu_count()} cores...")

        for bin_idx in range(self.bin_num):
            if self.bin_count[bin_idx] > 0:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        filepath = self._get_filepath(bin_idx, i, j)
                        
                        self.batched_median_from_disk(
                            filepath, bin_idx, batch_freq, out=median_array[bin_idx, i, j, :]
                        )
        
        return median_array, self.fine_counter, self.bin_count
    

def reader():
    tracker = MedianTracker(bin_num=20, temp_dir="/scratch/philj0ly/plasma_vis/tmp", plasma_bin_edges=np.array(np.arange(21)), t_chunk=1.0, t_init=56, t_final=1721411900)

    for bin_idx in range(20):
        for i in range(2):
            for j in range(2):
                try:
                    header = tracker.read_file_header(bin_idx, i, j)
                    print(f"Bin {bin_idx} Pol {i}{j}:", header["bin_count"], "out of", header["bin_tot_count"])
                except:
                    print(f"Bin {bin_idx} Pol {i}{j}: File not found")

def test_median_tracker_disk():
    # Simple test for MedianTracker
    #  shape=(2,2,20), dtype="float32"
    tracker = MedianTracker(bin_num=1, temp_dir="/home/philj0ly/tmp", plasma_bin_edges=np.array([0.2, 1.6]), t_chunk=1.0, t_init=56, t_final=1721411900)
    tracker.bin_tot_count += 105  # Expecting 1000 arrays in bin 0
    total = np.empty((2,2, 20, 101), dtype="complex64")

    for i in range(101):
        # arr = np.array([[np.arange(20)+i, (np.arange(20)+i+1)*1j],[np.arange(20)+i+20, (np.arange(20)+i+21)*1j]], dtype="complex64")
        arr = np.array([[np.random.rand(20)+i, (np.random.rand(20)+i+1)*1j],[np.random.rand(20)+i+20, (np.random.rand(20)+i+21)*1j]], dtype="complex64")
        total[..., i] = arr
        tracker.add_to_buf_vectorized(bin_idx=0, array=arr)
        # tracker.add_to_buf(bin_idx=1, array=arr*-1)

    median, count, counter = tracker.get_median(3)
    print("Median:\n", median)
    true_med = np.ma.median(np.ma.masked_invalid(np.abs(total)), axis=-1).filled(np.nan)
    print("Expected Median:\n", true_med)

    print("Success?", np.allclose(median[0], true_med, equal_nan=True))# and np.allclose(median[1], -true_med, equal_nan=True))

    print(tracker.read_file_header(0,0,0))

    tracker.cleanup()


def size_speed_test():
    tracker = MedianTracker(bin_num=1, temp_dir="/scratch/philj0ly/tmp", plasma_bin_edges=np.array([0.2, 1.6]), t_chunk=1.0, t_init=56, t_final=1721411900)
    tracker.bin_tot_count += 2000
    nfreq = 1000
    arr = np.array([[np.random.rand(nfreq), (np.random.rand(nfreq)+1)*1j],[np.random.rand(nfreq)+30, (np.random.rand(nfreq)+31)*1j]], dtype="complex64")
    tracker.add_to_bufv2(bin_idx=0, array=arr)
    t1 = time.time()

    for i in range(10):
        # arr = np.array([[np.arange(20)+i, (np.arange(20)+i+1)*1j],[np.arange(20)+i+20, (np.arange(20)+i+21)*1j]], dtype="complex64")
        tracker.add_to_buf_vectorized(bin_idx=0, array=arr)
        if i%10==0:
            print(i)
    t = time.time()-t1

    print(f"processed {arr.nbytes} in {t/1000} s")
    print(arr.nbytes/(t/1000))






if __name__ == "__main__":   
    # test_median_tracker_disk() 
    # reader()
    size_speed_test()
    
