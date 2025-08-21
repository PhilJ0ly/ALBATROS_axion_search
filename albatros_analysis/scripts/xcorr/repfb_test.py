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

    outdir = "/scratch/philj0ly/test_08_21"

    print("Beginning Testing...\n")

    acclen = 2100
    nchunks = 15
    osamp = 64
    cut = 19
    nblock = 10
    nant = 1
    gen_type = 'uniform'  # Options: 'constant', 'uniform',  'gaussian', 'sine'

           
    pols =repfb_test(acclen,nchunks,nblock, osamp, nant=1, cut=cut,filt_thresh=0.45, gen_type=gen_type)
    

    fname = f"test_{gen_type}_{str(acclen)}_{str(osamp)}.npz"
    fpath = path.join(outdir,fname)
    np.savez(fpath,data=pols.data,mask=pols.mask)

    print("\nSaved with an oversampling rate of", osamp, "and an acclen of", acclen, "at")
    print(fpath)