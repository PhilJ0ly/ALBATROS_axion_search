import ctypes
import os
import numpy as np
from contextlib import contextmanager
import atexit

NUM_CPU = os.cpu_count()
mylib = ctypes.cdll.LoadLibrary(
    os.path.realpath(__file__ + r"/..") + "/libfftwf.so"
)

fft_complex_c = mylib.fft_complex
fft_complex_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int
]
fft_complex_c.restype = None

rfft_c = mylib.rfft
rfft_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int
]
rfft_c.restype = None

irfft_c = mylib.irfft
irfft_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]
irfft_c.restype = None

ifft_complex_c = mylib.ifft_complex
ifft_complex_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]
ifft_complex_c.restype = None

@contextmanager
def parallelize_fft(nthreads=None):
    if not nthreads:
        nthreads=NUM_CPU
    mylib.init_fftw(nthreads)
    yield
    mylib.cleanup_fftw()

def fft_complex(dat, axis=1, out=None, backward=False):
    assert dat.dtype == "complex64"
    assert dat.ndim == 2, "Input must be 2D array"
    assert axis in [0, 1], "axis must be 0 or 1"

    if out is None:
        out = np.empty(dat.shape,dtype=dat.dtype) 

    assert out.shape == dat.shape, f"Expected output shape {dat.shape}, got {out.shape}"
    assert out.dtype == dat.dtype

    sign=2*int(backward==True)-1
    many_fft_c2c_1d_c(dat.ctypes.data, out.ctypes.data, dat.shape[0], dat.shape[1], axis, sign)
    return out

def rfft(dat, axis=1, out=None, backward=False):
    assert dat.dtype == "float32"
    assert dat.ndim == 2, "Array must be 2D"
    assert axis in [0, 1], "Axis is either 0 or 1"

    if axis == 1:
        # Transform along columns (last axis)
        out_shape = (dat.shape[0], dat.shape[1]//2 + 1)
    else:
        # Transform along rows (first axis)  
        out_shape = (dat.shape[0]//2 + 1, dat.shape[1])

    if out is None:
        out = np.empty(out_shape,dtype="complex64") 

    assert out.shape == out_shape, f"Expected output shape {out_shape}, got {out.shape}"
    assert out.dtype == "complex64"

    many_fft_r2c_1d_c(dat.ctypes.data, out.ctypes.data, dat.shape[0], dat.shape[1], axis)
    return out

def irfft(dat, axis=1, out=None,  n_output=-1):
    assert dat.dtype == "complex64"
    assert dat.ndim == 2, "Input must be 2D array"
    assert axis in [0, 1], "axis must be 0 or 1"

    if axis == 1:
        # Transform along columns (last axis)
        if n_output == -1:
            n_output = 2 * (dat.shape[1] - 1)  # scipy default
        out_shape = (dat.shape[0], n_output)
    else:
        # Transform along rows (first axis)
        if n_output == -1:
            n_output = 2 * (dat.shape[0] - 1)  # scipy default
        out_shape = (n_output, dat.shape[1])

    if out is None:
        out = np.empty(out_shape,dtype="float32") 
    assert out.shape == out_shape, f"Expected output shape {out_shape}, got {out.shape}"
    assert out.dtype == "float32"

    many_fft_c2r_1d_c(dat.ctypes.data, out.ctypes.data, dat.shape[0], dat.shape[1], axis, n_output)
    return out


read_wisdom_c = mylib.read_wisdom
read_wisdom_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

write_wisdom_c = mylib.write_wisdom
write_wisdom_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

mylib.init_fftw()
atexit.register(mylib.cleanup_fftw)

if __name__=="__main__": 
    x = np.random.rand(100).astype("float32")
    y = np.empty(x.shape, dtype="complex64")

    
