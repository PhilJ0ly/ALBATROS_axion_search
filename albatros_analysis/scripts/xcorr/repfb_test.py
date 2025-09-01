# Philippe Joly 2025-06-03

# This script is designed to test the rebining of frequency data from ALBATROS using either CPU cores (in parrallel) or a GPU


"""
Usage:
    First adjust the corresponding config files (config_{pUnit}_axion.json) with the desired parameters.
    Also, if the machine has both cpu as well as gpu and cpu wants to be used adjust the ./albatros_analysis/__init__.py
    such that xp = numpy
     
    Run
        python xcorr_gpu_cpu.py <processing unit (gpu/cpu)>
    [Defaults to cpu]
"""

# Functionality where sees no gpu and defaults to cpu could be added

import numpy as np
import time
import argparse
from os import path
import os
import sys
sys.path.insert(0,path.expanduser("~"))
import json

from helper_gpu_stream_cleanest import repfb_test

if __name__=="__main__":

    outdir = "/scratch/philj0ly/test_08_25"

    print("Beginning Testing...\n")

    acclen = 1792
    nchunks = 24
    osamp = 1024
    cut = 19
    nblock = 39
    nant = 1
    gen_types = ['constant', 'uniform', 'gaussian', 'sine']
    # gen_type = 'uniform'  # Options: 'constant', 'uniform',  'gaussian', 'sine'
    gen_types=['sine']

    for gen_type in gen_types: 
        print("\n", 50*"~")
        print(f"Testing gen_type: {gen_type}")
        print(50*"~")     
        pols, true_out =repfb_test(acclen,nchunks,nblock, osamp, nant=1, cut=cut,filt_thresh=0, gen_type=gen_type)

        fname = f"test_{gen_type}_{str(acclen)}_{str(osamp)}.npz"
        fpath = path.join(outdir,fname)
        np.savez(fpath,data=pols.data,mask=pols.mask, true_data=true_out, true_mask=true_out.mask)

        print("\nSaved with an oversampling rate of", osamp, "and an acclen of", acclen, "at")
        print(fpath)