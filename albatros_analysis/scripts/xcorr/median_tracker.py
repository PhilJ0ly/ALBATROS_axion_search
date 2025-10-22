# Philippe Joly 2025-10-21

"""
Handles median tracking for numpy arrays
More specifically, it tracks the median of arrays added to different bins.
This is designed to handel large streamin processing where keeping all arrays in memory is not feasible.
It relies on sending memory to disk if needed.
"""

import sys
from os import path

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

class MedianTrackerDisk:
    """
    Handles averaging (Median) for numpy arrays

    self.sum: Sum of arrays added to each bin
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
        self.count = None
        self.sum = None

        
        if shape is not None and dtype is not None:
            self.sum = np.zeros((self.bin_num,)+(self.max_size,)+self.shape, dtype=self.dtype)
            self.count = np.zeros((bin_num,)+shape, dtype="int64")
            

    def add_to_mean(self, bin_idx, array: np.ndarray):
        assert bin_idx < self.bin_num and bin_idx >= 0, f"Invalid bin index {bin_idx}"

        if self.shape is None or self.dtype is None:
            # Initialize mean array according to first input
            self.shape = array.shape
            self.dtype = array.dtype

            self.sum = np.zeros((self.bin_num,)+(self.max_size,)+self.shape, dtype=self.dtype)
            self.count = np.zeros((self.bin_num,)+self.shape, dtype="int64")
        else:
            # baby-proofing
            assert array.shape == self.shape, f"Invalid Mean Input. Expected {self.shape} array but got {array.shape}"
            assert array.dtype == self.dtype, f"Invalid Mean Input. Expected array of type {self.dtype} but got {array.dtype}"


        mask = ~np.ma.masked_invalid(array).mask
        self.count[bin_idx][mask] += 1

        self.sum[bin_idx][self.counter[bin_idx]%self.max_size] = array
        self.counter[bin_idx] += 1

        if self.counter[bin_idx]%self.max_size == 0:
            # Dump to disk
            np.savez(path.join(self.temp_dir, f"median_tracker_bin{bin_idx}_part{self.counter[bin_idx]//self.max_size-1}.npz"), data=self.sum[bin_idx])

    def get_mean(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        median_array = np.zeros((self.bin_num,)+self.shape, dtype=self.dtype)
        for i in range(self.bin_num):
            if self.counter[i] > 0:
                sum_array = self.sum[i][:self.counter[i]%self.max_size]

                for part in range(self.counter[i]//self.max_size):
                    data_path = path.join(self.temp_dir, f"median_tracker_bin{i}_part{part}.npz")
                    with np.load(data_path) as f:
                        loaded = f['data']
                    print(sum_array.shape, loaded.shape)
                    sum_array = np.concatenate((sum_array, loaded), axis=0)
                stacked = np.ma.masked_invalid(sum_array)
                median_array[i] = np.ma.median(stacked, axis=0).filled(np.nan)
            else:
                median_array[i] = np.full(self.shape, np.nan, dtype=self.dtype)

        return median_array, self.count, self.counter


def test_median_tracker_disk():
    # Simple test for MedianTrackerDisk
    tracker = MedianTrackerDisk(bin_num=1, temp_dir="/home/philj0ly/tmp", max_size=19, shape=(2,2, 10), dtype="float32")

    for i in range(101):
        arr = np.array([[np.arange(10)+i, np.arange(10)+i+1],[np.arange(10)+i+2, np.arange(10)+i+3]], dtype="float32")
        tracker.add_to_mean(bin_idx=0, array=arr)

    median, count, counter = tracker.get_mean()
    print("Median:\n", median)
    print("Count:\n", count)
    print("Counter:\n", counter)


if __name__ == "__main__":   
    test_median_tracker_disk() 
    
