# Philippe Joly 2025-12-09

# This is a numba implementation of a tiled transpose function

import numpy as np
import numba as nb
import time

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


def test_trans():
    nr=10000
    nc=3000
    x=np.random.randn(nr*nc).reshape(nr,nc)
    xT_ref = x.T.copy()
    xT = tiled_transpose(x)
    assert(np.allclose(xT, xT_ref))
    print("tiled_transpose test passed.")

def test_speed(nr,nc, niter=100, niter2=1000):
    # x_re =np.random.randn(nr*nc).reshape(nr,nc)
    # x_im =np.random.randn(nr*nc).reshape(nr,nc)
    x_re = np.ones((nr,nc), dtype=np.float32)
    x = (x_re + 1j*x_re).astype(np.complex64)

    tile_w = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
    ]
    # tile_sizes = [
    #     (32,128), 
    #     (64,256), 
    #     (512, 112),
    #     (1024, 256),
    #     (2048, 512),
    #     (2048, 64),
    #     (4096, 128),
    #     (64, 64)
    # ]

    time_count = []
    for bsc in tile_w:
        if bsc > nc:
            continue
        for bsr in tile_w:
            if bsr > nr:
                continue
            #warm up
            xT = tiled_transpose(x, bsc=bsc, bsr=bsr)

            tottime=0
            for i in range(niter):
                tt1=time.time()
                xT = tiled_transpose(x, bsc=bsc, bsr=bsr)
                tt2=time.time()
                tottime+=tt2-tt1

            time_per_it = tottime/niter
            time_count.append(((bsc, bsr), time_per_it, x.nbytes/(time_per_it*1e9)))
            print(f"avg time taken for tiled_transpose with bsc={bsc}, bsr={bsr}: {tottime/niter:5.3f}s")

    # xT = np.transpose(x).copy  # warm up
    # tottime=0
    # for i in range(niter):
    #     tt1=time.time()
    #     xT = np.transpose(x).copy()
    #     tt2=time.time()
    #     tottime+=tt2-tt1

    # time_per_it = tottime/niter
    # time_count.append(("numpy", time_per_it, x.nbytes/(time_per_it*1e9)))
    # print(f"avg time taken for numpy transpose {time_per_it:5.3f}s")

    new_list = sorted(time_count, key=lambda item: item[1])
    # print(new_list[:10])
    # second round testing
    for i in range(min(10, len(new_list))):
        xT = tiled_transpose(x, bsc=bsc, bsr=bsr)
        tottime=0
        for j in range(niter2):
            tt1=time.time()
            xT = tiled_transpose(x, bsc=bsc, bsr=bsr)
            tt2=time.time()
            tottime+=tt2-tt1

        time_per_it = tottime/niter2
        shp = new_list[i][0]
        new_list[i] = (shp, time_per_it, x.nbytes/(time_per_it*1e9))

    print(f"\nSummary of timings of {x.nbytes/1e9:.3f} GB array:")
    for bs, t, band in new_list[:10]:
        if bs == "numpy":
            print(f"Numpy transpose: {t:5.3f}s - {band:5.3f} GB/s")
        else:
            print(f"Tiled transpose (bsc={bs[0]}, bsr={bs[1]}): {t:5.3f}s - {band:5.3f} GB/s")

if __name__ == "__main__":

    # Those are the shapes of the arrays needing transpose for alabatros vis
    c_list = [167, 56, 278, 167, 280, 223, 335, 1117, 949, 780, 1006, 614, 725, 671, 893]
    nr = 120* 2**16
    # for 1000 columns -> we get ~7864 Rows per Column

    scale = 280
    test_speed(10001*scale, scale, niter=10)