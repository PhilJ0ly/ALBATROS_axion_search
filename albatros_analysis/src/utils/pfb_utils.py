import numpy as np
import time
import numba as nb

def sinc_hamming(ntap,lblock):
    N=ntap*lblock
    w=np.arange(0,N)-N/2
    return np.hamming(ntap*lblock)*np.sinc(w/lblock)

def sinc_hanning(ntap,lblock):
    N=ntap*lblock
    w=np.arange(0,N)-N/2
    return np.hanning(ntap*lblock)*np.sinc(w/lblock)

@nb.njit(parallel=True)
def tiled_transpose(x,bsc=128,bsr=32): 
    nr,nc= x.shape
    y = np.empty((nc,nr),dtype=x.dtype)

    br = (nr + bsr - 1)//bsr
    bc = (nc + bsc - 1)//bsc

    block_size = br*bc
    for i in nb.prange(block_size):
        i_start = (i//bc)*bsr
        j_start=(i%bc)*bsc
        
        i_end=min(i_start+bsr,nr)
        j_end=min(j_start+bsc,nc)
        
        for k in range(i_start, i_end):
            for l in range(j_start, j_end):
                y[l,k] = x[k,l]

    return y