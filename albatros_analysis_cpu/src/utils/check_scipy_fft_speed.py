# Philippe Joly 2025-06-18

import os
import numpy as np
from scipy.fft import rfft, set_workers, irfft
import time

def main():
    hs = [1000, 4096, 10000, 16384, 100000]
    # hs = [100000]
    cores = np.array([10, 20, 40, 50, 60, 75, 80])
    np.random.shuffle(cores)
    niter = 100

    for h in hs:
        x = np.random.rand(h*4096).reshape(h,4096)
        print(f"\nData shape {x.shape}, niter {niter}")
        for n_cores in cores:
            print(n_cores, "CPU Cores")
            tot = 0

            for i in range(niter):
                t0 =time.time()
                with set_workers(n_cores):
                    y = rfft(x, axis=1)
                tot += time.time()-t0
            tot /= niter
            print("\t SciPy RFFT:", tot, x.nbytes/tot/1e6)

            tot=0            
            for i in range(niter):
                t0 =time.time()
                with set_workers(n_cores):
                    y = irfft(x, axis=1)
                tot += time.time()-t0
            tot /= niter
            print("\t SciPy IRFFT:", tot, x.nbytes/tot/1e6)

            

if __name__=="__main__":
    main()   
