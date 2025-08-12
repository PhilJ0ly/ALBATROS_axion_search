import mkfftw as mk
import mkfftw_float as mkf
import numpy as np
import time
from scipy import fft as sfft
import os
import sys

# osamp = 65336
# nc = 4096*osamp
# nr = 14336*4096//nc - (4 - 1)

niter=100
nr, nc= 221,262144
xr=np.random.randn(nr*nc).reshape(nr,nc).astype("float32")*20
xi=np.random.randn(nr*nc).reshape(nr,nc).astype("float32")*20

# xr = xr.T.copy()
# xi = xi.T.copy()

x=xr+1j*xi
print(f"starting speed test {x.shape[0]}x{x.shape[1]}, axis=1")

# tottime=0
# for i in range(niter):
#     tt1=time.time()
#     xf = np.fft.fft(x,axis=1)
#     tt2=time.time()
#     tottime+=tt2-tt1
#     # print("numpy exectime", tt2-tt1)
# print(f"avg time taken for numpy axis=1 {tottime/niter:5.3f}s")

tottime=0
xf2=mkf.many_fft_c2c_1d(x,axis=1) # for the FFTW_MEASURE to do its thing first
for i in range(niter):
    tt1=time.time()
    xf2 = mkf.many_fft_c2c_1d(x,axis=1)
    tt2=time.time()
    tottime+=tt2-tt1
print(f"avg time taken for fftw axis=1 {tottime/niter:5.3f}s, BW {x.nbytes/(tottime/niter)/1e6:5.3f} MSPS")


xf3=mkf.many_fft_r2c_1d(xr,axis=1) # for the FFTW_MEASURE to do its thing first
tottime=0
for i in range(niter):
    tt1=time.time()
    xf3 = mkf.many_fft_r2c_1d(xr,axis=1)
    tt2=time.time()
    tottime+=tt2-tt1
print(f"avg time taken for rfftw axis=1 {tottime/niter:5.3f}s, BW {xr.nbytes/(tottime/niter)/1e6:5.3f} MSPS")

# sys.exit()
xf3=mkf.many_fft_c2r_1d(x,axis=1) # for the FFTW_MEASURE to do its thing first
tottime=0
for i in range(niter):
    tt1=time.time()
    xf3 = mkf.many_fft_c2r_1d(x,axis=1)
    tt2=time.time()
    tottime+=tt2-tt1
print(f"avg time taken for irfftw axis=1 {tottime/niter:5.3f}s, BW {x.nbytes/(tottime/niter)/1e6:5.3f} MSPS")

# sys.exit()

# SCIPY

n_workers=os.cpu_count()
tottime=0
with sfft.set_workers(n_workers):
    xf3=sfft.fft(x,axis=1,workers=n_workers) # for the FFTW_MEASURE to do its thing first
    for i in range(niter):
        tt1=time.time()
        xf3=sfft.fft(x,axis=1,workers=n_workers)
        tt2=time.time()
        tottime+=tt2-tt1
    print(f"avg time taken for scipy fft axis=1 {tottime/niter:5.3f}s, BW {x.nbytes/(tottime/niter)/1e6:5.3f} MSPS")


tottime=0
with sfft.set_workers(n_workers):
    xf3=sfft.rfft(xr,axis=1,workers=n_workers) # for the FFTW_MEASURE to do its thing first
    for i in range(niter):
        tt1=time.time()
        xf3=sfft.rfft(xr,axis=1,workers=n_workers)
        tt2=time.time()
        tottime+=tt2-tt1
    print(f"avg time taken for scipy rfft axis=1 {tottime/niter:5.3f}s, BW {xr.nbytes/(tottime/niter)/1e6:5.3f} MSPS")


tottime=0
with sfft.set_workers(n_workers):
    xf3=sfft.irfft(x,axis=1,workers=n_workers) # for the FFTW_MEASURE to do its thing first
    for i in range(niter):
        tt1=time.time()
        xf3=sfft.irfft(x,axis=1,workers=n_workers)
        tt2=time.time()
        tottime+=tt2-tt1
    print(f"avg time taken for scipy irfft axis=1 {tottime/niter:5.3f}s, BW {x.nbytes/(tottime/niter)/1e6:5.3f} MSPS")


sys.exit()


# axis = 0
print("-------------------------------------------------------")
x=x.T.copy()
print(f"starting speed test {x.shape[0]}x{x.shape[1]}, axis=0")
tottime=0
for i in range(niter):
    tt1=time.time()
    xf = np.fft.fft(x,axis=0)
    tt2=time.time()
    tottime+=tt2-tt1
    # print("numpy exectime", tt2-tt1)
print(f"avg time taken for numpy axis=0 {tottime/niter:5.3f}s")
tottime=0
with mk.parallelize_fft():
    xf2=mk.many_fft_c2c_1d(x,axis=0) # for the FFTW_MEASURE to do its thing first
    for i in range(niter):
        tt1=time.time()
        xf2 = mk.many_fft_c2c_1d(x,axis=0)
        tt2=time.time()
        tottime+=tt2-tt1
        # print("FFTW exectime", tt2-tt1)
    print(f"avg time taken for fftw axis=0 {tottime/niter:5.3f}s")

n_workers=os.cpu_count()
tottime=0
with sfft.set_workers(n_workers):
    xf3=sfft.fft(x,axis=0,workers=n_workers) # for the FFTW_MEASURE to do its thing first
    for i in range(niter):
        tt1=time.time()
        xf3=sfft.fft(x,axis=0,workers=n_workers)
        tt2=time.time()
        tottime+=tt2-tt1
        # print("Scipy exectime", tt2-tt1)
    print(f"avg time taken for scipy axis=0 {tottime/niter:5.3f}s")


