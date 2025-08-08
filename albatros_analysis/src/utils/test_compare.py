# Philippe Joly 2025-06-11

"""
Test script to compare the speed of CPU and GPU versions of PFB, IPFB.
"""

import numpy as np
import time

import sys
from os import path
sys.path.append(path.expanduser('~'))
from albatros_analysis.src.utils.pfb_utils import sinc_hamming



def generate_test_data(nspec, nchan, seed=42):
    np.random.seed(seed)
    
    # Generate complex input data (nspec x nchan)
    dat_real = np.random.randn(nspec, nchan).astype(np.float32)
    np.random.seed(seed+1)
    dat_imag = np.random.randn(nspec, nchan).astype(np.float32)
    dat_cpu = np.asarray(dat_real + 1j * dat_imag, dtype='complex64', order='C')
    
    # Convert to GPU array
    # dat_gpu = cp.asarray(dat_cpu.reshape(nspec, nchan))
    
    return dat_cpu #, dat_gpu

def compare(a1, a2, op):
    print(f"{op}: CPU {a1.shape} {a1.dtype} GPU {a2.shape} {a2.dtype}")
    print(f"Are the same?: {np.allclose(a1, a2, rtol=1e-3, atol=1e-4)}")
    print(f"Max absolute difference: {np.max(np.abs(a1-a2))}")
    print(f"Max relative difference: {np.max(np.abs((a1-a2) / a2))}")
    
    print()


def main(pu, outdir, seed=41):

    thresh = 0.45
    test_case = {"osamp": 64, "pfb_mult": 256}
        # {"osamp": 512, "pfb_mult": 128},
        # {"osamp": 2**12, "pfb_mult": 32}
        # {"osamp": 2**16, "pfb_mult": 8} too much memory for GPU
    

    test_case["name"] = f"Osamp ({test_case['osamp']}) pfb_mult ({test_case['pfb_mult']})"
    test_case["nslice"] = test_case["osamp"]*test_case["pfb_mult"]

    print(test_case["name"])
    cpu_fn = path.join(outdir,f"{test_case['name']}_cpu.npz")
    gpu_fn = path.join(outdir,f"{test_case['name']}_gpu.npz")

    nspec = test_case["nslice"]
    cutsize = 16
    osamp = test_case["osamp"]
    pfb_mult = test_case["pfb_mult"]
    dwin = sinc_hamming(ntap=4,lblock=2048*2*osamp)
    cut = int(nspec/cutsize)

    if pu == "cpu" or pu == "gpu":
        dat_cpu = generate_test_data(nspec, nchan=2049, seed=seed)

    if pu=="cpu":
        from albatros_analysis.src.utils.pfb_cpu_utils import get_matft_cpu, cpu_ipfb, cpu_pfb, apply_filter, compute_filter, test_fft, cpu_ipfb2

        print("Generating CPU filt...")
        matft_cpu = get_matft_cpu(
            nslice=nspec
        )
        filt_arr = compute_filter(matft_cpu, thresh)

        

        cpu_ipfb_result = np.empty((nspec-2*cut)*4096, dtype='float32',order='C')
        cpu_win_big = np.asarray(dwin, dtype='float32', order='C')

        # print("to_ipfb_pol0", dat_cpu.shape, dat_cpu.dtype)
        # print("raw_pol0", cpu_ipfb_result.shape, cpu_ipfb_result.dtype)
        # print("cut", cut)
        # print("matft", matft_cpu.shape)

        # test_ipfb = cpu_ipfb2(dat_cpu, cpu_ipfb_result, filt=filt_arr, cut=cut)

        cpu_ipfb(dat_cpu, cpu_ipfb_result, filt=filt_arr, cut=cut)

        # np.random.seed(seed)
        # res = np.random.rand(cpu_ipfb_result.size).astype("float32")

        cpu_pfb_result = cpu_pfb(cpu_ipfb_result, cpu_win_big, nchan=2048*osamp+1, ntap=4 )

        np.random.seed(seed)
        ddftr = np.random.rand(4096, 8193).astype("float32")
        np.random.seed(seed+1)
        ddfti = np.random.rand(4096, 8193).astype("float32")

        ddft = ddftr+1j*ddfti
        apply_filter(ddft, filt=filt_arr) 
        
        # out_fft = test_fft(ddft)

        np.savez(
            cpu_fn, 
            # ipfb_in=dat_cpu,
            ipfb_out=cpu_ipfb_result, 
            # res=res,
            pfb_out=cpu_pfb_result, 
            filt=filt_arr,
            thresh_out=ddft,
            # matft=matft_cpu, 
            win=cpu_win_big, 
            # test_fft=out_fft,
            seed=seed
        )

    elif pu=="gpu":
        import cupy as cp
        from albatros_analysis.src.utils.pfb_gpu_utils import get_matft, cupy_ipfb, cupy_pfb, test_fft, get_filt, cupy_ipfb2

        print("\nGenerating GPU matft...")
        matft_gpu = get_matft(
            nslice=nspec
        )
        filt_arr = get_filt(matft_gpu, thresh)

        dat_gpu = cp.asarray(dat_cpu)
        # test_ipfb = cupy_ipfb2(dat_gpu, matft_gpu, thresh=thresh)
        gpu_ipfb_result = cupy_ipfb(dat_gpu, matft_gpu, thresh=thresh)

        cupy_win_big = cp.asarray(dwin, dtype='float32', order='C').reshape(4,2*(2048*osamp+1-1))

        
        # res1 = gpu_ipfb_result[cut:-cut]
        # np.random.seed(seed)
        # res = cp.asarray(np.random.rand(res1.size).reshape(res1.shape).astype("float32"))

        gpu_pfb_result = cupy_pfb(gpu_ipfb_result[cut:-cut],cupy_win_big, nchan=2048*osamp+1, ntap=4 )

        np.random.seed(seed)
        ddftr = np.random.rand(4096, 8193).astype("float32")
        np.random.seed(seed+1)
        ddfti = np.random.rand(4096, 8193).astype("float32")

        ddft = cp.asarray(ddftr+1j*ddfti)
        thresh_out = filt_arr*ddft

        # out_fft = test_fft(ddft)

        np.savez(
            gpu_fn,
            # ipfb_in=cp.asnumpy(dat_gpu),
            ipfb_out=cp.asnumpy(gpu_ipfb_result[cut:-cut].ravel()), 
            # res=cp.asnumpy(res).ravel(), 
            pfb_out=cp.asnumpy(gpu_pfb_result), 
            filt=cp.asnumpy(filt_arr),
            thresh_out=cp.asnumpy(thresh_out),
            # matft=matft_gpu, 
            # test_fft=cp.asnumpy(out_fft),
            win=cp.asnumpy(cupy_win_big.ravel()), 
            seed=seed
        )
    
    else:
        
        with np.load(cpu_fn) as f:
            # cpu_ipfb_in = f["ipfb_in"]
            # cpu_res = f["res"]
            cpu_ipfb_result = f["ipfb_out"]
            cpu_pfb_result = f["pfb_out"]
            # matft_cpu = f["matft"]
            cpu_thresh_result = f["thresh_out"]
            cpu_filter = f["filt"]
            cpu_win = f["win"]
            # cpu_fft = f['test_fft']

        with np.load(gpu_fn) as f:
            # gpu_ipfb_in = f["ipfb_in"]
            # gpu_res = f["res"]
            gpu_ipfb_result = f["ipfb_out"]
            gpu_pfb_result = f["pfb_out"]
            # matft_gpu = f["matft"]
            gpu_thresh_result = f["thresh_out"]
            gpu_filter = f["filt"]
            gpu_win = f["win"]
            # gpu_fft = f["test_fft"]

        compare(cpu_ipfb_result, gpu_ipfb_result, "IPFB OUT")
        compare(cpu_pfb_result, gpu_pfb_result, "PFB OUT")
        # compare(cpu_fft, gpu_fft, "IRFFT")
        # compare(cpu_win, gpu_win, "WIN")
        # compare(cpu_thresh_result, gpu_thresh_result, "Thresh filter OUT")
        # compare(cpu_filter, gpu_filter, "FILTER")



if __name__=="__main__":
    pUnit = 'compare' 
    if len(sys.argv) > 1:
        pUnit = sys.argv[1] 

    seed = 41
    if len(sys.argv) > 2:
        seed = sys.argv[2] 

    outdir = "/home/s/sievers/philj0ly/.$SCRATCH/test_utils"
    if len(sys.argv) > 3:
        outdir = sys.argv[3] 

    main(pUnit, outdir, seed)

    

        
    

          