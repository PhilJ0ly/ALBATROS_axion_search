# Philippe Joly 2025-09-17

"""
This script is designed to test the accuracy of running mean class in visualise)plasma_bins
"""

import sys
from os import path
sys.path.insert(0, path.expanduser("~"))

import numpy as np
from albatros_analysis.scripts.xcorr.helper_gpu import RunningMean

if __name__=="__main__":
    bin_num = 5
    means = [0,20,50,100,543]
    rm = RunningMean(bin_num)

    for i in range(10000):
        num_of_nan = np.random.randint(1,high=150,size=bin_num)
        for j in range(bin_num):
            nanidx = np.random.randint(0, high=1235, size=num_of_nan[0])
            arr = np.random.normal(loc=means[j], scale=1.0, size=1235)
            arr[nanidx] = np.nan
            rm.add_to_mean(j, arr)
        

    rm_mean, rm_count, rm_counter = rm.get_mean()
    print(f"Mean/std across channels/total count/counts\n")
    for j in range(bin_num):
        print("expected mean", means[j])
        print(np.mean(rm_mean[j]), np.std(rm_mean[j]), rm_counter[j], rm_count[j], "\n")