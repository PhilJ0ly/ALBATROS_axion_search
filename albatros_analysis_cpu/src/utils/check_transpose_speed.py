import os
os.environ["NUMBA_OPT"] = "3"
os.environ["NUMBA_LOOP_VECTORIZE"] = "1"
os.environ["NUMBA_ENABLE_AVX"] = "1"
import numba as nb
import numpy as np
import time
import sys





@nb.njit(parallel=True)
def parallel_transpose_blocked(A, block_size=128):
    m, n = A.shape
    B = np.empty((n, m), dtype=A.dtype)
    
    # Number of blocks
    m_blocks = (m + block_size - 1) // block_size
    n_blocks = (n + block_size - 1) // block_size
    
    for bi in nb.prange(m_blocks):
        for bj in range(n_blocks):
            # Block boundaries
            i_start = bi * block_size
            i_end = min(i_start + block_size, m)
            j_start = bj * block_size
            j_end = min(j_start + block_size, n)
            
            # Transpose the block
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    B[j, i] = A[i, j]
    
    return B

@nb.njit(parallel=True)
def parallel_transpose_optimized(A):
    m, n = A.shape
    B = np.empty((n, m), dtype=A.dtype)
    
    # Use smaller blocks for better cache utilization
    block_size =256
    
    for i_block in nb.prange((m + block_size - 1) // block_size):
        i_start = i_block * block_size
        i_end = min(i_start + block_size, m)
        
        for j_block in range((n + block_size - 1) // block_size):
            j_start = j_block * block_size
            j_end = min(j_start + block_size, n)
            
            # Transpose the block
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    B[j, i] = A[i, j]
    
    return B

@nb.njit(parallel=True)
def naive_transpose(x):
    y = np.empty((x.T.shape),dtype=x.dtype)
    nr,nc=y.shape
    for i in nb.prange(nr):
        for j in nb.prange(nc):
            y[i,j]=x[j,i]
    return y

# @nb.njit(parallel=True)
# def naive_transpose2(x):
#     y = np.empty((x.T.shape),dtype=x.dtype)
#     nr,nc=y.shape
#     for i in nb.prange(nr):
#         for j in range(0, nr - 3, 4):
#             y[i, j]     = x[j, i]
#             y[i, j + 1] = x[j + 1, i]
#             y[i, j + 2] = x[j + 2, i]
#             y[i, j + 3] = x[j + 3, i]
#         for j in range(nr - nr % 4, nr):
#             y[i, j] = x[j, i]

#     return y

@nb.njit(parallel=True)
def naive_transpose2(x):
    M, N = x.shape  # x is (rows, cols)
    y = np.empty((N, M), dtype=x.dtype)
    for i in nb.prange(N):
        for j in range(0, M - 3, 4):
            y[i, j]     = x[j, i]
            y[i, j + 1] = x[j + 1, i]
            y[i, j + 2] = x[j + 2, i]
            y[i, j + 3] = x[j + 3, i]
        for j in range(M - M % 4, M):
            y[i, j] = x[j, i]
    return y

@nb.njit(parallel=True)
def tilled_transpose(x,bsc=32,bsr=128): 
    y = np.empty((x.T.shape),dtype=x.dtype)
    nr,nc=x.shape
    br = (nr + bsr - 1)//bsr
    bc = (nc + bsc - 1)//bsc
    totblocks = br*bc
    # print("totblocks", totblocks, br, bc)
    for i in nb.prange(totblocks):
        bi=i//bc
        bj=i%bc
        # print("bi, bj", bi, bj)
        imax=min(bi*bsr+bsr,nr)
        jmax=min(bj*bsc+bsc,nc)
        # print("imax, jmax",imax, jmax)
        for ii in range(bi*bsr, imax):
            for jj in range(bj*bsc, jmax):
                # print("putting ii,jj", ii, jj)
                y[jj,ii] = x[ii,jj]
    return y

@nb.njit(parallel=True)
def tiled_transpose_unroll(x, bsc=32, bsr=128):
    y = np.empty((x.T.shape), dtype=x.dtype)
    nr, nc = x.shape
    br = (nr + bsr - 1) // bsr
    bc = (nc + bsc - 1) // bsc
    totblocks = br * bc

    for i in nb.prange(totblocks):
        bi = i // bc
        bj = i % bc

        istart = bi * bsr
        jstart = bj * bsc
        imax = min(istart + bsr, nr)
        jmax = min(jstart + bsc, nc)

        for ii in range(istart, imax):
            full_range = (jmax - jstart) // 4 * 4
            for jj in range(jstart, jstart + full_range, 4):
                y[jj, ii]     = x[ii, jj]
                y[jj + 1, ii] = x[ii, jj + 1]
                y[jj + 2, ii] = x[ii, jj + 2]
                y[jj + 3, ii] = x[ii, jj + 3]
            for jj in range(jstart + full_range, jmax):
                y[jj, ii] = x[ii, jj]
    return y


def np_trans(x):
    return x.T.copy()

def main():
    nr=30000 # good tile shape bsc=32,bsr=128
    nc=4096

    # nc=10000 #good tile shape bsc=16, bsr=1024
    # nr=4096
    x_og = np.random.randn(nr*nc).reshape(nr,nc).astype("float64")
    x = x_og.copy()
    niter=50

    print(f"Testing Transpose Speeds on skinny {x.shape} and fat {x.T.shape} Arrays with {niter} Iterations\n")

    trans = [
        # ["np trans", np_trans, 0, 0],
        ["nb trans", naive_transpose, 0, 0],
        ["nb trans 2", naive_transpose2, 0, 0],
        # ["tiled trans", tilled_transpose, 0, 0],
        ["tiled unroll", tiled_transpose_unroll, 0, 0]
    ]

    for func in trans:
        name = func[0]
        transpo = func[1]

        # warm up
        for i in range(min(10, niter//10)):
            y = transpo(x)

        for i in range(niter):
            t1=time.time()
            y=transpo(x)
            t2=time.time()
            func[2]+=t2-t1

        func[2] /= niter
        print(name)
        print("\t", x.shape, func[2], x.nbytes/func[2]/(1024**2))

        for i in range(min(10, niter//10)):
            x = transpo(y)
            
        for i in range(niter):
            t1=time.time()
            x=transpo(y)
            t2=time.time()
            func[3]+=t2-t1

        func[3] /= niter
        print("\t", y.shape, func[3], x.nbytes/func[3]/(1024**2))


if __name__=="__main__":
    main()