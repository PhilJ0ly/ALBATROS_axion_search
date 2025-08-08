#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>
#include <omp.h>
#define MKFFTW_FLAG FFTW_ESTIMATE // MEASURE seems to be worth it for fat arrays but too much overhead for skinny arrays (where ESTIMATE is faster)
// module load python/3.9.10 gcc/11.3.0 fftw/3.3.10
// gcc -std=c99 -O3 -march=native -shared -o libfftwf.so -fPIC -fopenmp mkfftw2.c -lfftw3_omp -lm

int set_threads(int nthreads)
{
  if (nthreads < 0) {
    nthreads = omp_get_num_threads();
  }
  if (fftwf_init_threads()) {
    fftwf_plan_with_nthreads(nthreads);
  } else {
    printf("something went wrong during thread init");
  }
}


//gcc-4.9 -I/Users/sievers/local/include -fopenmp -std=c99 -O3 -shared -fPIC -o libmkfftw.so mkfftw.c -L/Users/sievers/local/lib -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3  -lm -lgomp
//gcc-9 -I/usr/local/include -fopenmp -std=c99 -O3 -shared -fPIC -o libmkfftw.so mkfftw.c -L/usr/local/lib -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3  -lm -lgomp


//gcc -I{HIPPO_FFTW_DIR}/include -fopenmp -std=c99 -O3 -shared -fPIC -o libmkfftw.so mkfftw.c -L${HIPPO_FFTW_DIR}/lib    -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3  -lm -lgomp
//gcc -fopenmp -std=c99 -O3 -shared -fPIC -o libmkfftw.so mkfftw.c -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3 -lgomp -lpthread
// gcc -std=c99 -O3 -march=native -shared -o libmkfftwf.so -fPIC -fopenmp mkfftw.c -lfftw3f_threads -lfftw3f -lm -lgomp


// FLOAT PRECISION FUNCTIONS (float32/complex64)

void many_fft_c2c_1d(fftwf_complex *dat, fftwf_complex *datft, int nrows, int ncols, int axis, int sign)
{
    int istride = 1, ostride = 1, idist = ncols, odist = ncols, ndata = ncols, ntrans = nrows;
    long int n = nrows * ncols;
    float nn;

    if (axis == 0) {
        istride = ncols;
        ostride = ncols;
        idist = 1;
        odist = 1;
        ndata = nrows;
        ntrans = ncols;
    }

    fftwf_plan plan = fftwf_plan_many_dft(1, &ndata, ntrans,
                                          dat, &ndata, istride, idist,
                                          datft, &ndata, ostride, odist,
                                          sign, MKFFTW_FLAG);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    if (sign == 1) {
        nn = 1.0f / ndata;
        #pragma omp parallel for
        for (long int i = 0; i < n; i++) {
            datft[i] *= nn;
        }
    }
}



// Real-to-complex FFT (equivalent to scipy.fft.rfft)
void many_fft_r2c_1d(float *dat, fftwf_complex *datft, int nrows, int ncols, int axis)
{
    int istride = 1, ostride = 1, idist, odist, ndata, ntrans;

    if (axis == 1) {
        istride = 1;
        ostride = 1;
        idist = ncols;
        odist = ncols / 2 + 1;
        ndata = ncols;
        ntrans = nrows;
    } else if (axis == 0) {
        istride = ncols;
        ostride = ncols / 2 + 1;
        idist = 1;
        odist = 1;
        ndata = nrows;
        ntrans = ncols;
    } else {
        printf("Error: axis must be 0 or 1\n");
        return;
    }

    fftwf_plan plan = fftwf_plan_many_dft_r2c(1, &ndata, ntrans,
                                              dat, &ndata, istride, idist,
                                              datft, &ndata, ostride, odist,
                                              MKFFTW_FLAG);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}



// Complex-to-real inverse FFT (equivalent to scipy.fft.irfft)
// n_output: if -1, defaults to 2*(input_length-1) to match scipy.fft.irfft behavior
void many_fft_c2r_1d(fftwf_complex *datft, float *dat, int nrows, int ncols, int axis, int n_output)
{
    int istride = 1, ostride = 1, idist, odist, ndata, ntrans;
    long int n_total;
    float nn;
    int actual_n_output;

    if (axis == 1) {
        actual_n_output = (n_output == -1) ? 2 * (ncols - 1) : n_output;
        istride = 1;
        ostride = 1;
        idist = ncols;
        odist = actual_n_output;
        ndata = actual_n_output;
        ntrans = nrows;
        n_total = nrows * actual_n_output;
    } else if (axis == 0) {
        actual_n_output = (n_output == -1) ? 2 * (nrows - 1) : n_output;
        istride = ncols;
        ostride = actual_n_output;
        idist = 1;
        odist = 1;
        ndata = actual_n_output;
        ntrans = ncols;
        n_total = actual_n_output * ncols;
    } else {
        printf("Error: axis must be 0 or 1\n");
        return;
    }

    fftwf_plan plan = fftwf_plan_many_dft_c2r(1, &ndata, ntrans,
                                              datft, &ndata, istride, idist,
                                              dat, &ndata, ostride, odist,
                                              MKFFTW_FLAG);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    nn = 1.0f / actual_n_output;
    #pragma omp parallel for
    for (long int i = 0; i < n_total; i++) {
        dat[i] *= nn;
    }
}


/*--------------------------------------------------------------------------------*/
void read_wisdom(char *double_file, char *single_file)
{
  printf("files are: .%s. and .%s.\n",double_file,single_file);
  int dd=fftwf_import_wisdom_from_filename(double_file);
  //int ss=fftwf_import_wisdom_from_filename(single_file);
  printf("return value is %d\n",dd);
}

/*--------------------------------------------------------------------------------*/
void write_wisdom(char *double_file, char *single_file)
{
  printf("files are: .%s. and .%s.\n",double_file,single_file);
  int dd=fftwf_export_wisdom_to_filename(double_file);
  //int ss=fftwf_export_wisdom_to_filename(single_file);
  printf("return value is %d\n",dd);
}

void cleanup_threads()
{
  fftwf_cleanup_threads();
}
