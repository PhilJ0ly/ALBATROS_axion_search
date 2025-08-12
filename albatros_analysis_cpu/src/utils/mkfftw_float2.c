#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>
#include <omp.h>

// gcc -std=c99 -O3 -march=native -shared -o libfftwf.so -fPIC -fopenmp mkfftw2.c -lfftw3_threads -lfftw3_omp -lm
// gcc -o libfftwf.so mkfftw_float2.c -lfftw3f -lfftw3f_threads -lm -fopenmp -O3

// gcc -std=c99 -O3 -march=native -shared -o libfftwf.so -fPIC -fopenmp mkfftw_float2.c -lfftw3_omp -lfftw3_threads -lm

// Thread-safe FFTW initialization
static int fftw_initialized = 0;
static omp_lock_t fftw_lock;

// Initialize FFTW with threading support (single precision)
void init_fftw() {
    if (!fftw_initialized) {
        omp_init_lock(&fftw_lock);
        omp_set_lock(&fftw_lock);
        
        if (!fftw_initialized) {
            fftwf_init_threads();
            fftwf_plan_with_nthreads(omp_get_max_threads());
            fftw_initialized = 1;
        }
        
        omp_unset_lock(&fftw_lock);
    }
}

// Cleanup FFTW resources
void cleanup_fftw() {
    if (fftw_initialized) {
        fftwf_cleanup_threads();
        fftwf_cleanup();
        omp_destroy_lock(&fftw_lock);
        fftw_initialized = 0;
    }
}

// Complex FFT - equivalent to scipy.fft.fft (single precision)
void fft_complex(const float complex* input, float complex* output, int n) {
    init_fftw();
    
    fftwf_complex* in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);
    
    // if (!in || !out) {
    //     if (in) fftwf_free(in);
    //     if (out) fftwf_free(out);
    //     return -1;
    // }
    
    // Copy input data - FFTW complex is array of float[2]
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        ((float*)in)[2*i] = crealf(input[i]);     // Real part
        ((float*)in)[2*i+1] = cimagf(input[i]);   // Imaginary part
    }
    
    // Create and execute plan
    fftwf_plan plan = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    
    // Copy output data
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        output[i] = ((float*)out)[2*i] + I * ((float*)out)[2*i+1];
    }
    
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}

// Real FFT - equivalent to scipy.fft.rfft (single precision)
void rfft(const float* input, float complex* output, int n) {
    init_fftw();
    
    int out_size = n / 2 + 1;
    
    float* in = (float*) fftwf_malloc(sizeof(float) * n);
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * out_size);
    
    // if (!in || !out) {
    //     if (in) fftwf_free(in);
    //     if (out) fftwf_free(out);
    //     return -1;
    // }
    
    // Copy input data
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        in[i] = input[i];
    }
    
    // Create and execute plan
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
    fftwf_execute(plan);
    
    // Copy output data
    #pragma omp parallel for
    for (int i = 0; i < out_size; i++) {
        output[i] = ((float*)out)[2*i] + I * ((float*)out)[2*i+1];
    }
    
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}

// Inverse Real FFT - equivalent to scipy.fft.irfft (single precision)
void irfft(const float complex* input, float* output, int n_output) {
    init_fftw();
    
    int input_size = n_output / 2 + 1;
    
    fftwf_complex* in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * input_size);
    float* out = (float*) fftwf_malloc(sizeof(float) * n_output);
    
    // if (!in || !out) {
    //     if (in) fftwf_free(in);
    //     if (out) fftwf_free(out);
    //     return -1;
    // }
    
    // Copy input data
    #pragma omp parallel for
    for (int i = 0; i < input_size; i++) {
        ((float*)in)[2*i] = crealf(input[i]);     // Real part
        ((float*)in)[2*i+1] = cimagf(input[i]);   // Imaginary part
    }
    
    // Create and execute plan
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(n_output, in, out, FFTW_ESTIMATE);
    fftwf_execute(plan);
    
    // FFTW doesn't normalize, so we need to divide by n
    float norm_factor = 1.0f / n_output;
    #pragma omp parallel for
    for (int i = 0; i < n_output; i++) {
        output[i] = out[i] * norm_factor;
    }
    
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}

// Inverse Complex FFT - equivalent to scipy.fft.ifft (single precision)
void ifft_complex(const float complex* input, float complex* output, int n) {
    init_fftw();
    
    fftwf_complex* in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n);
    
    // if (!in || !out) {
    //     if (in) fftwf_free(in);
    //     if (out) fftwf_free(out);
    //     return -1;
    // }
    
    // Copy input data
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        ((float*)in)[2*i] = crealf(input[i]);     // Real part
        ((float*)in)[2*i+1] = cimagf(input[i]);   // Imaginary part
    }
    
    // Create and execute plan (FFTW_BACKWARD for inverse)
    fftwf_plan plan = fftwf_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    
    // FFTW doesn't normalize, so we need to divide by n
    float norm_factor = 1.0f / n;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        output[i] = (((float*)out)[2*i] + I * ((float*)out)[2*i+1]) * norm_factor;
    }
    
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}

// Multi-dimensional FFT support (single precision)
void fft_complex_nd(const float complex* input, float complex* output, 
                   int ndim, const int* dimensions) {
    init_fftw();
    
    // Calculate total size
    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= dimensions[i];
    }
    
    fftwf_complex* in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * total_size);
    fftwf_complex* out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * total_size);
    
    // if (!in || !out) {
    //     if (in) fftwf_free(in);
    //     if (out) fftwf_free(out);
    //     return -1;
    // }
    
    // Copy input data
    #pragma omp parallel for
    for (int i = 0; i < total_size; i++) {
        ((float*)in)[2*i] = crealf(input[i]);     // Real part
        ((float*)in)[2*i+1] = cimagf(input[i]);   // Imaginary part
    }
    
    // Create and execute plan
    fftwf_plan plan = fftwf_plan_dft(ndim, dimensions, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    
    // Copy output data
    #pragma omp parallel for
    for (int i = 0; i < total_size; i++) {
        output[i] = ((float*)out)[2*i] + I * ((float*)out)[2*i+1];
    }
    
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);
}