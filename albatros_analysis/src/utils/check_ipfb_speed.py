# Philippe Joly 2025-06-18
# This script compares the speed of different implementations of the IPFB function
import os
os.environ['NUMBA_OPT']='3'
os.environ['NUMBA_LOOP_VECTORIZE']='1'
os.environ['NUMBA_ENABLE_AVX']='1'
os.environ['NUMBA_CACHE_DIR'] = '/scratch/s/sievers/philj0ly/tmp/my_numba_cache'
import numpy as np
import numba as nb
from scipy.fft import rfft, set_workers, irfft
import multiprocessing as mp
import time

from pfb_cpu_utils import get_matft_cpu

def generate_test_data(nspec, nchan, seed=42):
    np.random.seed(seed)
    
    dat_real = np.random.randn(nspec, nchan).astype(np.float32)
    dat_imag = np.random.randn(nspec, nchan).astype(np.float32)
    dat_cpu = np.asarray(dat_real + 1j * dat_imag, dtype='complex64', order='C')
    
    return dat_cpu

@nb.njit(parallel=True)
def naive_transpose(x):
    y = np.empty((x.T.shape),dtype=x.dtype)
    nr,nc=y.shape
    for i in nb.prange(nr):
        for j in nb.prange(nc):
            y[i,j]=x[j,i]
    return y

@nb.njit(parallel=True)
def apply_thresh_filter(ddft, matft, thresh, inv_matft):
  
    if thresh > 0:
        scale = 1.0 + thresh * thresh
        for i in nb.prange(ddft.shape[0]):
            for j in range(ddft.shape[1]):
                abs2 = matft[i, j].real**2 + matft[i, j].imag**2
                filt = abs2 / (abs2 + thresh**2) * scale
                ddft[i, j] *= filt * inv_matft[i,j]
    else:
        for i in nb.prange(ddft.shape[0]):
            for j in range(ddft.shape[1]):
                ddft[i, j] *= inv_matft[i, j]


@nb.njit(parallel=True)
def apply_thresh_filter_optimized(ddft, matft, thresh, inv_matft):
    """Optimized filter with better memory access patterns"""
    if thresh > 0:
        thresh_sq = thresh * thresh
        scale = 1.0 + thresh_sq
        
        for i in nb.prange(ddft.shape[0]):
            for j in range(ddft.shape[1]):
                # More efficient complex magnitude squared
                abs2 = matft[i, j].real * matft[i, j].real + matft[i, j].imag * matft[i, j].imag
                filt = abs2 / (abs2 + thresh_sq) * scale
                ddft[i, j] = ddft[i, j] * filt * inv_matft[i, j]
    else:
        # Vectorized multiplication when no threshold
        for i in nb.prange(ddft.shape[0]):
            for j in range(ddft.shape[1]):
                ddft[i, j] = ddft[i, j] * inv_matft[i, j]

# vectorized filter 
@nb.njit(parallel=True)
def apply_thresh_filter_vectorized(ddft, matft, thresh, inv_matft):
    """Vectorized version processing rows in parallel"""
    if thresh > 0:
        thresh_sq = thresh * thresh
        scale = 1.0 + thresh_sq
        
        for i in nb.prange(ddft.shape[0]):
            row_ddft = ddft[i]
            row_matft = matft[i]
            row_inv = inv_matft[i]
            
            for j in range(ddft.shape[1]):
                abs2 = row_matft[j].real**2 + row_matft[j].imag**2
                filt = abs2 / (abs2 + thresh_sq) * scale
                row_ddft[j] *= filt * row_inv[j]
    else:
        for i in nb.prange(ddft.shape[0]):
            for j in range(ddft.shape[1]):
                ddft[i, j] *= inv_matft[i, j]

def cpu_ipfb(dat, matft, inv_matft=None, thresh=0.0, n_workers=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    if inv_matft is None:
        inv_matft = 1.0 / np.conj(matft)

    with set_workers(n_workers):
        dd = irfft(dat, axis=1) 
        
    dd2 = naive_transpose(dd)

    with set_workers(n_workers):  
        ddft = rfft(dd2, axis=1) 

        
    apply_thresh_filter(ddft, matft, thresh, inv_matft) 

    with set_workers(n_workers): 
        res = irfft(ddft , axis=1) 

    return naive_transpose(res)

def cpu_ipfb2(dat, matft, inv_matft=None, thresh=0.0, n_workers=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    if inv_matft is None:
        inv_matft = 1.0 / np.conj(matft)

    with set_workers(n_workers):
        dd = irfft(dat, axis=1) 
        
    dd2 = dd.T

    with set_workers(n_workers):  
        ddft = rfft(dd2, axis=1) 

        
    apply_thresh_filter(ddft, matft, thresh, inv_matft) 

    with set_workers(n_workers): 
        res = irfft(ddft , axis=1) 

    return naive_transpose(res)


def main():

    n_workers = 80
    niter=100

    print(f"Testing IPFB Speeds with {niter} Iterations")

    thresh = 0.45
    test_cases = [
        {"osamp": 64, "pfb_mult": 256},
        {"osamp": 512, "pfb_mult": 128},
        {"osamp": 2**12, "pfb_mult": 32},
        # {"osamp": 2**16, "pfb_mult": 8}  # too much memory for GPU
    ]

    for tester in test_cases:
        tester["name"] = f"Osamp ({tester['osamp']}) pfb_mult ({tester['pfb_mult']})"
        tester["nslice"] = tester["osamp"]*tester["pfb_mult"]

    test_case = test_cases[0]
    for test_case in test_cases:
        print(test_case)

        nspec = test_case["nslice"]

        print("Generating CPU matft...")
        matft = get_matft_cpu(
            nslice=nspec
        )
        inv_matft = 1.0 / np.conj(matft)
        
        data = generate_test_data(nspec, nchan=2049, seed=41)
        with set_workers(n_workers):
            dd = irfft(data, axis=1)  
        dd2 = dd.T
        with set_workers(n_workers):  
            ddft = rfft(dd2, axis=1) 
        
        
        thresh_funcs = [
            ["thresh", apply_thresh_filter, 0],
            ["thresh optimized", apply_thresh_filter_optimized, 0],
            ["thresh vectorized", apply_thresh_filter_vectorized, 0]
        ]

        print("Testing threshold functions...\n")

        for func in thresh_funcs:
            name = func[0]
            thresh_fx = func[1]
            print(name)

            # warm up
            for i in range(min(10, niter//10)):
                y = thresh_fx(ddft, matft, thresh, inv_matft)

            for i in range(niter):
                t1=time.time()
                y=thresh_fx(ddft, matft, thresh, inv_matft)
                t2=time.time()
                func[2]+=t2-t1

            func[2] /= niter
            print("\t", func[2], ddft.nbytes/func[2]/(1e6))


        ipfb_funcs = [
            ["ipfb", cpu_ipfb, 0],
            ["ipfb.T", cpu_ipfb2, 0]
        ]
        print("Testing IPFB functions...\n")

        for func in ipfb_funcs:
            name = func[0]
            ipfb_fx = func[1]
            print(name)

            # warm up
            for i in range(min(10, niter//10)):
                y = ipfb_fx(data, matft, inv_matft, thresh, n_workers)

            for i in range(niter):
                t1=time.time()
                y=ipfb_fx(data, matft, inv_matft, thresh, n_workers)
                t2=time.time()
                func[2]+=t2-t1

            func[2] /= niter
            print("\t", func[2], data.nbytes/func[2]/(1e6))
        
        break



if __name__=='__main__':
    main()