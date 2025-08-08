import ctypes
import os
import numpy as np
from contextlib import contextmanager
import atexit

NUM_CPU = os.cpu_count()
mylib = ctypes.cdll.LoadLibrary(
    os.path.realpath(__file__ + r"/..") + "/libfftw2.so"
)

many_fft_c2c_1d_c = mylib.many_fft_c2c_1d
many_fft_c2c_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
many_fft_c2c_1d_c.restype = None
# fftw_complex *dat, fftw_complex *datft, int nrows, int ncols, int axis, int sign

many_fft_r2c_1d_c = mylib.many_fft_r2c_1d
many_fft_r2c_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
many_fft_r2c_1d_c.restype = None
# double *dat, fftw_complex *datft, int nrows, int ncols, int axis

many_fft_c2r_1d_c = mylib.many_fft_c2r_1d
many_fft_c2r_1d_c.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
many_fft_c2r_1d_c.restype = None
# fftw_complex *datft, double *dat, int nrows, int ncols, int axis, int n_output

@contextmanager
def parallelize_fft(nthreads=None):
    if not nthreads:
        nthreads=NUM_CPU
    mylib.set_threads(nthreads)
    yield
    mylib.cleanup_threads()

def many_fft_c2c_1d(dat, axis=1, out=None, backward=False):
    assert dat.dtype == "complex128"
    assert dat.ndim == 2, "Input must be 2D array"
    assert axis in [0, 1], "axis must be 0 or 1"

    if out is None:
        out = np.empty(dat.shape,dtype=dat.dtype) 

    assert out.shape == dat.shape, f"Expected output shape {dat.shape}, got {out.shape}"
    assert out.dtype == dat.dtype

    sign=2*int(backward==True)-1
    many_fft_c2c_1d_c(dat.ctypes.data, out.ctypes.data, dat.shape[0], dat.shape[1], axis, sign)
    return out

def many_fft_r2c_1d(dat, axis=1, out=None, backward=False):
    assert dat.dtype == "float64"
    assert dat.ndim == 2, "Array must be 2D"
    assert axis in [0, 1], "Axis is either 0 or 1"

    if axis == 1:
        # Transform along columns (last axis)
        out_shape = (dat.shape[0], dat.shape[1]//2 + 1)
    else:
        # Transform along rows (first axis)  
        out_shape = (dat.shape[0]//2 + 1, dat.shape[1])

    if out is None:
        out = np.empty(out_shape,dtype="complex128") 

    assert out.shape == out_shape, f"Expected output shape {out_shape}, got {out.shape}"
    assert out.dtype == "complex128"

    many_fft_r2c_1d_c(dat.ctypes.data, out.ctypes.data, dat.shape[0], dat.shape[1], axis)
    return out

def many_fft_c2r_1d(dat, axis=1, out=None,  n_output=-1):
    assert dat.dtype == "complex128"
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
        out = np.empty(out_shape,dtype="float64") 
    assert out.shape == out_shape, f"Expected output shape {out_shape}, got {out.shape}"
    assert out.dtype == "float64"

    many_fft_c2r_1d_c(dat.ctypes.data, out.ctypes.data, dat.shape[0], dat.shape[1], axis, n_output)
    return out


read_wisdom_c = mylib.read_wisdom
read_wisdom_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

write_wisdom_c = mylib.write_wisdom
write_wisdom_c.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

mylib.set_threads(os.cpu_count())
atexit.register(mylib.cleanup_threads)