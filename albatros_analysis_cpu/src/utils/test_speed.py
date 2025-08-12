# Philippe Joly 2025-06-11

"""
Test script to compare the speed of CPU and GPU versions of PFB, IPFB.
"""

import numpy as np
import cupy as cp
import time
import json

import sys
from os import path
sys.path.append(path.expanduser('~'))


def generate_test_data(nspec, nchan, seed=42):
    np.random.seed(seed)
    # Generate complex input data (nspec x nchan)
    dat_real = np.random.randn(nspec*nchan).astype(np.float32)

    np.random.seed(seed+1)
    dat_imag = np.random.randn(nspec*nchan).astype(np.float32)
    
    dat_cpu = np.asarray(dat_real + 1j * dat_imag, dtype='complex64', order='C')
    
    # Convert to GPU array
    dat_gpu = cp.asarray(dat_cpu.reshape(nspec, nchan))
    
    return dat_cpu, dat_gpu

if __name__=="__main__":
    niter = 1000

    thresh = 0.45
    test_cases = [
        {"osamp": 64, "pfb_mult": 256},
        {"osamp": 512, "pfb_mult": 128},
        {"osamp": 2**12, "pfb_mult": 32}
        # {"osamp": 2**16, "pfb_mult": 8} too much memory for GPU
    ]

    for tester in test_cases:
        tester["name"] = f"Osamp ({tester['osamp']}) pfb_mult ({tester['pfb_mult']})"
        tester["nslice"] = tester["osamp"]*tester["pfb_mult"]

    test_case = test_cases[0]
    print(test_case)

    nspec = test_case["nslice"]
    cutsize = 16
    osamp = test_case["osamp"]
    pfb_mult = test_case["pfb_mult"]
    dwin = sinc_hamming(ntap=4,lblock=2048*2*osamp)
    cut = int(osamp*pfb_mult/cutsize)

    print("Generating CPU matft...")
    matft_cpu = get_matft_cpu(
        nslice=nspec
    )
    inv_matft = 1.0 / np.conj(matft_cpu)
    
    print("\nGenerating GPU matft...")
    matft_gpu = get_matft(
        nslice=nspec
    )

    dat_cpu, dat_gpu = generate_test_data(nspec, nchan=2049, seed=41)

    times = {
        "IPFB":{},
        "PFB":{}
    }

    print("Running CPU IPFB...")
    cpu_ipfb_result = np.empty((nspec-2*cut)*4096, dtype='float32',order='C')
    tot_t = 0
    for i in range(niter):
        t0 = time.time()
        cpu_ipfb(dat_cpu, matft_cpu, cpu_ipfb_result, inv_matft=inv_matft, thresh=thresh, cut=cut)
        tot_t += time.time()-t0
    times["IPFB"]["CPU"] = [tot_t/niter, niter/tot_t]

    print("Running GPU IPFB...")
    tot_t = 0
    for i in range(niter):
        t0 = time.time()
        gpu_ipfb_result = cupy_ipfb(dat_gpu, matft_gpu, thresh=thresh)
        tot_t += time.time()-t0
    times["IPFB"]["GPU"] = [tot_t/niter, niter/tot_t]

    ipfb_gpu_out = cp.asnumpy(gpu_ipfb_result)[cut:-cut].ravel()
    print(f"IPFB OUT: CPU {cpu_ipfb_result.shape} {cpu_ipfb_result.dtype} GPU {ipfb_gpu_out.shape} {ipfb_gpu_out.dtype}")
    print(f"Are the same?: {np.allclose(cpu_ipfb_result, ipfb_gpu_out, rtol=1e-5, atol=1e-6)}")
    dif = np.abs(cpu_ipfb_result - ipfb_gpu_out)
    print(np.mean(dif), np.std(dif))
        
    

    print("Running CPU PFB...")
    cpu_win_big = np.asarray(dwin, dtype='float32', order='C')
    tot_t = 0
    for i in range(niter):
        t0 = time.time()
        cpu_pfb_result = cpu_pfb(cpu_ipfb_result,cpu_win_big, nchan=2048*osamp+1, ntap=4 )
        tot_t += time.time()-t0
    times["PFB"]["CPU"] = [tot_t/niter, niter/tot_t]

    print("Running GPU PFB...")
    cupy_win_big = cp.asarray(dwin, dtype='float32', order='C').reshape(4,2*(2048*osamp+1-1))
    tot_t = 0
    for i in range(niter):
        t0 = time.time()
        gpu_pfb_result = cupy_pfb(gpu_ipfb_result[cut:-cut],cupy_win_big, nchan=2048*osamp+1, ntap=4 )
        tot_t += time.time()-t0
    times["PFB"]["GPU"] = [tot_t/niter, niter/tot_t]

    pfb_gpu_out = cp.asnumpy(gpu_pfb_result)
    print(f"IPFB OUT: CPU {cpu_ipfb_result.shape} {cpu_ipfb_result.dtype} GPU {ipfb_gpu_out.shape} {ipfb_gpu_out.dtype}")
    print(f"Are the same?: {np.allclose(cpu_ipfb_result, ipfb_gpu_out, rtol=1e-5, atol=1e-6)}")
    dif = np.abs(cpu_ipfb_result - ipfb_gpu_out)
    print(np.mean(dif), np.std(dif))

    # print("Done!\n")
    # print(json.dumps(times, indent=4))
          