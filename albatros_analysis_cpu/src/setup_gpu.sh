#! /bin/bash
module load cuda/11.7.1 gcc/11.4.0
conda activate albatros
export USE_GPU=1
export LD_LIBRARY_PATH=/home/s/sievers/philj0ly/.conda/envs/albatros/lib/:$LD_LIBRARY_PATH
export CUPY_CACHE_DIR=/project/s/sievers/philj0ly/.cupy/kernel_cache
