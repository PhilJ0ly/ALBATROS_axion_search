import numpy as np
import argparse
from os import path
import sys
sys.path.insert(0,path.expanduser("~"))
import json

if __name__=="__main__":

    config_fn = "/home/s/sievers/philj0ly/albatros_analysis/scripts/xcorr/config_axion.json"
    if len(sys.argv) > 1:
        config_fn = sys.argv[1]  
    
    with open(config_fn, "r") as f:
        config = json.load(f)

    init_t = config["correlation"]["start_timestamp"]
    end_t = config["correlation"]["end_timestamp"]
    chanstart = config["frequency"]["start_channel"]
    chanend = config["frequency"]["end_channel"]
    osamp = config["correlation"]["osamp"]
    pfb_mult = config["correlation"]["pfb_size_multiplier"]
    
    pfb_size = osamp*pfb_mult
    cutsize = 16
    
    assert pfb_mult >= 8 

    cut=int(pfb_size/cutsize)
    acclen=pfb_size - 2*cut
    nchunks = int(np.floor((end_t-init_t)*250e6/4096/acclen))
    print(nchunks)