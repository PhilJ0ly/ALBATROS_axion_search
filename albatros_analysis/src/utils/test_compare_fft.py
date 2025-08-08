import mkfftw_float as mkf
import mkfftw as mk
import numpy as np
import time
from scipy import fft as sfft
import os
import sys
n_workers=os.cpu_count()

def compare(a1, a2, op):
    print(f"{op}: FFTW {a1.shape} {a1.dtype} SciPy {a2.shape} {a2.dtype}")
    print(f"Are the same?: {np.allclose(a1, a2, rtol=1e-3, atol=1e-4)}")
    print(f"Max absolute difference: {np.max(np.abs(a1-a2))}")
    print(f"Max relative difference: {np.max(np.abs((a1-a2) / a2))}")
    
    print()

nr, nc= 221,262144
xr=np.random.randn(nr*nc).reshape(nr,nc).astype("float64")*20
xi=np.random.randn(nr*nc).reshape(nr,nc).astype("float64")*20

x=xr+1j*xi
print(f"starting equivalence test {x.shape[0]}x{x.shape[1]}, axis=1")

fftws = []
fftws.append(mk.many_fft_c2c_1d(x,axis=1))
fftws.append(mk.many_fft_r2c_1d(xr,axis=1))
fftws.append(mk.many_fft_c2r_1d(x,axis=1))

sci = []
with sfft.set_workers(n_workers):
    sci.append(sfft.fft(x,axis=1,workers=n_workers))
    sci.append(sfft.rfft(xr,axis=1,workers=n_workers))
    sci.append(sfft.irfft(x,axis=1,workers=n_workers))

names = ["fft", "rfft", "irfft"]

for i in range(3):
    compare(fftws[i], sci[i], names[i])