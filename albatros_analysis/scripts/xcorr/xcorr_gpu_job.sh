#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00  


#SBATCH --job-name=xcorr_osamp(64)_pfb(256)
#SBATCH --output=/project/s/sievers/philj0ly/xcorr_gpu/logs/xcorr_output_%j_osamp(64)_pfb(256).txt
#SBATCH --mail-type=BEGIN,END,FAIL

module load MistEnv/2021a anaconda3/2021.05
source activate albatros
module load cuda/11.7.1
module load gcc/11.4.0

export USE_GPU=1
export LD_LIBRARY_PATH=/home/s/sievers/philj0ly/.conda/envs/albatros/lib/:$LD_LIBRARY_PATH
export CUPY_CACHE_DIR=/project/s/sievers/philj0ly/.cupy/kernel_cache

cd /home/s/sievers/philj0ly/albatros_analysis/scripts/xcorr
python xcorr_gpu_cpu.py gpu config_axion.json
