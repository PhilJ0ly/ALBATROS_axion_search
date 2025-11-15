# Philippe Joly 2025-10-21

"""
Handles median tracking for numpy arrays
More specifically, it tracks the median of arrays added to different bins.
This is designed to handel large streamin processing where keeping all arrays in memory is not feasible.
It relies on sending memory to disk if needed.
"""

import sys
from os import path
import os

import numpy as np
from typing import List, Tuple, Optional


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

class MedianTracker:
    """
    Handles averaging (Median) for numpy arrays

    self.sum: Sum of arrays added to each bin
    self.count: Count of valid entries per spectrum frequency bin per plasma bin
    self.counter: Count of arrays added to each bin
    """

    def __init__(self, bin_num: int, shape: Optional[Tuple[int]] = None, dtype: Optional[str] = None):
        self.bin_num = bin_num
        self.shape = shape
        self.dtype = dtype

        self.counter = np.zeros(bin_num, dtype="int64")
        self.sum = [[] for _ in range(bin_num)]
        self.count = None
        if shape is not None and dtype is not None:
            self.count = np.zeros((bin_num,)+shape, dtype="int64")
            

    def add_to_mean(self, bin_idx, array: np.ndarray):
        assert bin_idx < self.bin_num and bin_idx >= 0, f"Invalid bin index {bin_idx}"

        if self.shape is None or self.dtype is None:
            # Initialize mean array according to first input
            self.shape = array.shape
            self.dtype = array.dtype
            self.count = np.zeros((self.bin_num,)+self.shape, dtype="int64")
        else:
            # baby-proofing
            assert array.shape == self.shape, f"Invalid Mean Input. Expected {self.shape} array but got {array.shape}"
            assert array.dtype == self.dtype, f"Invalid Mean Input. Expected array of type {self.dtype} but got {array.dtype}"


        mask = ~np.ma.masked_invalid(array).mask

        #update only where valid entries
        self.count[bin_idx][mask] += 1
        self.sum[bin_idx].append(array)
        self.counter[bin_idx] += 1

    def get_mean(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        median_array = np.zeros((self.bin_num,)+self.shape, dtype=self.dtype)
        for i in range(self.bin_num):
            if self.counter[i] > 0:
                stacked = np.ma.masked_invalid(np.stack(self.sum[i], axis=0))
                median_array[i] = np.ma.median(stacked, axis=0).filled(np.nan)
            else:
                median_array[i] = np.full(self.shape, np.nan, dtype=self.dtype)

        return median_array, self.count, self.counter

class MedianTrackerDiskOG:
    """
    Handles averaging (Median) for numpy arrays using disk-based storage.
    
    Data is stored one time step at a time to ensure proper layout on disk.
    File layout: each time step written sequentially, so file contains
    [t0_array, t1_array, t2_array, ...] which can be read as (ntimes, spatial_shape)
    
    self.sum: Buffer of arrays added to each bin  
    self.count: Count of valid entries per spectrum frequency bin per plasma bin
    self.counter: Count of arrays added to each bin
    """

    def __init__(self, bin_num: int, temp_dir: str, max_size: int, shape: Optional[Tuple[int]] = None, dtype: Optional[str] = None):
        self.bin_num = bin_num
        self.shape = shape
        self.dtype = dtype
        self.temp_dir = temp_dir
        self.max_size = max_size

        self.counter = np.zeros(bin_num, dtype="int64")
        self.bin_tot_count = np.zeros(bin_num, dtype="int64")
        self.count = None
        self.sum = None
        
        # for bin_idx in range(bin_num):
        #     fn = path.join(self.temp_dir, f"median_tracker_bin{bin_idx}.bin")
        #     if path.exists(fn):
        #         os.remove(fn)
        
        if shape is not None and dtype is not None:
            self.sum = [[] for _ in range(self.bin_num)]
            self.count = np.zeros((bin_num,) + shape, dtype="int64")

    def add_to_bin_tot_count(self, bin_idx: int):
        self.bin_tot_count[bin_idx] += 1   

    def add_to_mean(self, bin_idx, array: np.ndarray):
        assert bin_idx < self.bin_num and bin_idx >= 0, f"Invalid bin index {bin_idx}"

        if self.shape is None or self.dtype is None:
            # Initialize mean array according to first input
            self.shape = array.shape
            self.dtype = array.dtype

            self.sum = [[] for _ in range(self.bin_num)]
            self.count = np.zeros((self.bin_num,) + self.shape, dtype="int64")
        else:
            # baby-proofing
            assert array.shape == self.shape, f"Invalid Mean Input. Expected {self.shape} array but got {array.shape}"
            assert array.dtype == self.dtype, f"Invalid Mean Input. Expected array of type {self.dtype} but got {array.dtype}"


        mask = ~np.ma.masked_invalid(array).mask
        self.count[bin_idx][mask] += 1

        self.sum[bin_idx].append(array)
        self.counter[bin_idx] += 1

        # Flush buffer when it reaches max_size
        if len(self.sum[bin_idx]) >= self.max_size:
            fn = path.join(self.temp_dir, f"median_tracker_bin{bin_idx}.bin")
            mode = 'ab' if path.exists(fn) else 'wb'
            
            # Stack arrays and write: creates (buffer_size, *shape) array
            stacked = np.stack(self.sum[bin_idx], axis=0)
            
            with open(fn, mode) as f:
                # Write in C order - when appended, this creates valid (ntimes, *shape) array
                f.write(stacked.tobytes())
            
            self.sum[bin_idx] = []

    def get_mean(self, batch_freq: Optional[int] =1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        median_array = np.zeros((self.bin_num,) + self.shape, dtype=self.dtype)

        for bin_idx in range(self.bin_num):
            if self.counter[bin_idx] > 0:
                fn = path.join(self.temp_dir, f"median_tracker_bin{bin_idx}.bin")
                
                # Flush any remaining data in buffer
                if len(self.sum[bin_idx]) > 0:
                    mode = 'ab' if path.exists(fn) else 'wb'
                    stacked = np.stack(self.sum[bin_idx], axis=0)
                    
                    with open(fn, mode) as f:
                        f.write(stacked.tobytes())
                    
                    self.sum[bin_idx] = []

                assert path.exists(fn), f"Data file {fn} does not exist but counter is {self.counter[bin_idx]}"

                # Memory map with C order: (total_time_steps,) + original_shape
                data_map = np.memmap(fn, dtype=self.dtype, mode='r', 
                                    shape=(self.counter[bin_idx],) + self.shape)
                

                # Process in batches along the frequency dimension (last axis)
                for i in range(0, self.shape[-1], batch_freq):
                    end_idx = min(i + batch_freq, self.shape[-1])
                    # Slice: (ntimes, ..., freq_batch)
                    batch_data = data_map[:, ..., i:end_idx]
                    
                    # Compute median along the time axis (first dimension, axis=0)
                    # Note, this is less efficient as time is not contiguous in memory. However, data cant be stored contiguously in time in a straming fashion. Without knowing total size ahead of time, we cant preallocate a contiguous array with time last.
                    median_array[bin_idx, ..., i:end_idx] = np.ma.median(
                        np.ma.masked_invalid(batch_data), axis=0
                    ).filled(np.nan)

                print(f"Averaged bin {bin_idx}")
                
                del data_map

        return median_array, self.count, self.counter



class MedianTrackerDisk:
    """
    Handles averaging (Median) for numpy arrays using disk-based storage.
    
    Data is stored one time step at a time to ensure proper layout on disk.
    File layout: each time step written sequentially, so file contains
    [t0_array, t1_array, t2_array, ...] which can be read as (ntimes, spatial_shape)
    All this in F-Major order to allow for efficient reading along time axis.
    
    self.buf: Buffer of arrays added to each bin
    self.count: Count of valid entries per spectrum frequency bin per plasma bin
    self.counter: Count of arrays added to each bin
    """

    def __init__(self, bin_num: int, temp_dir: str, max_size: int, shape: Optional[Tuple[int]] = None, dtype: Optional[str] = None):
        self.bin_num = bin_num
        self.shape = shape
        self.dtype = dtype
        self.temp_dir = temp_dir
        self.max_size = max_size

        self.counter = np.zeros(bin_num, dtype="int64")
        self.bin_tot_count = np.zeros(bin_num, dtype="int64")
        self.count = None
        self.sum = None
        
        # for bin_idx in range(bin_num):
        #     fn = path.join(self.temp_dir, f"median_tracker_bin{bin_idx}.bin")
        #     if path.exists(fn):
        #         os.remove(fn)
        
        if shape is not None and dtype is not None:
            self.sum = [[] for _ in range(self.bin_num)]
            self.count = np.zeros((bin_num,) + shape, dtype="int64")

    def add_to_bin_tot_count(self, bin_idx: int):
        self.bin_tot_count[bin_idx] += 1   

    def add_to_mean(self, bin_idx, array: np.ndarray):
        assert bin_idx < self.bin_num and bin_idx >= 0, f"Invalid bin index {bin_idx}"

        if self.shape is None or self.dtype is None:
            # Initialize mean array according to first input
            self.shape = array.shape
            self.dtype = array.dtype

            self.sum = [[] for _ in range(self.bin_num)]
            self.count = np.zeros((self.bin_num,) + self.shape, dtype="int64")
        else:
            # baby-proofing
            assert array.shape == self.shape, f"Invalid Mean Input. Expected {self.shape} array but got {array.shape}"
            assert array.dtype == self.dtype, f"Invalid Mean Input. Expected array of type {self.dtype} but got {array.dtype}"


        mask = ~np.ma.masked_invalid(array).mask
        self.count[bin_idx][mask] += 1

        self.sum[bin_idx].append(array)
        self.counter[bin_idx] += 1

        # Flush buffer when it reaches max_size
        if len(self.sum[bin_idx]) >= self.max_size:
            fn = path.join(self.temp_dir, f"median_tracker_bin{bin_idx}.bin")
            mode = 'ab' if path.exists(fn) else 'wb'
            
            # Stack arrays and write: creates (buffer_size, *shape) array
            stacked = np.stack(self.sum[bin_idx], axis=0)
            
            with open(fn, mode) as f:
                # Write in C order - when appended, this creates valid (ntimes, *shape) array
                f.write(stacked.tobytes())
            
            self.sum[bin_idx] = []

    def get_mean(self, batch_freq: Optional[int] =1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        median_array = np.zeros((self.bin_num,) + self.shape, dtype=self.dtype)

        for bin_idx in range(self.bin_num):
            if self.counter[bin_idx] > 0:
                fn = path.join(self.temp_dir, f"median_tracker_bin{bin_idx}.bin")
                
                # Flush any remaining data in buffer
                if len(self.sum[bin_idx]) > 0:
                    mode = 'ab' if path.exists(fn) else 'wb'
                    stacked = np.stack(self.sum[bin_idx], axis=0)
                    
                    with open(fn, mode) as f:
                        f.write(stacked.tobytes())
                    
                    self.sum[bin_idx] = []

                assert path.exists(fn), f"Data file {fn} does not exist but counter is {self.counter[bin_idx]}"

                # Memory map with C order: (total_time_steps,) + original_shape
                data_map = np.memmap(fn, dtype=self.dtype, mode='r', 
                                    shape=(self.counter[bin_idx],) + self.shape)
                

                # Process in batches along the frequency dimension (last axis)
                for i in range(0, self.shape[-1], batch_freq):
                    end_idx = min(i + batch_freq, self.shape[-1])
                    # Slice: (ntimes, ..., freq_batch)
                    batch_data = data_map[:, ..., i:end_idx]
                    
                    # Compute median along the time axis (first dimension, axis=0)
                    # Note, this is less efficient as time is not contiguous in memory. However, data cant be stored contiguously in time in a straming fashion. Without knowing total size ahead of time, we cant preallocate a contiguous array with time last.
                    median_array[bin_idx, ..., i:end_idx] = np.ma.median(
                        np.ma.masked_invalid(batch_data), axis=0
                    ).filled(np.nan)

                print(f"Averaged bin {bin_idx}")
                
                del data_map

        return median_array, self.count, self.counter


def test_median_tracker_disk():
    # Simple test for MedianTrackerDisk
    tracker = MedianTrackerDisk(bin_num=1, temp_dir="/home/philj0ly/tmp", max_size=19, shape=(2,2,20), dtype="float32")
    total = np.empty((2,2, 20, 1000), dtype="float32")

    for i in range(1000):
        arr = np.array([[np.arange(20)+i, np.arange(20)+i+1],[np.arange(20)+i+20, np.arange(20)+i+21]], dtype="float32")
        total[..., i] = arr
        tracker.add_to_mean(bin_idx=0, array=arr)

    median, count, counter = tracker.get_mean(2)
    print("Median:\n", median)
    true_med = np.ma.median(np.ma.masked_invalid(total), axis=-1).filled(np.nan)
    print("Expected Median:\n", true_med)
    print("Count:\n", count)
    print("Counter:\n", counter)

    print("Success?", np.allclose(median, true_med, equal_nan=True))

if __name__ == "__main__":   
    test_median_tracker_disk() 
    
