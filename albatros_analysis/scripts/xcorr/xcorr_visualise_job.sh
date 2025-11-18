#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00  


#SBATCH --job-name=visualise_osamp-65536_bins-20
#SBATCH --output=/scratch/philj0ly/plasma_vis/logs/vis_output_%j_osamp-65536_bins-20.out
#SBATCH --mail-type=BEGIN,END,FAIL

module load StdEnv/2023
module load python/3.11.5 gcc/12.3
module load cuda/12.6 fftw/3.3.10

source cd /home/philj0ly/albatros_analysis
source bin/activate

export USE_GPU=1
export CUPY_CACHE_DIR=/scratch/philj0ly/.cupy/kernel_cache

source cd /home/philj0ly/albatros_analysis/scripts/xcorr
python visualise_plasma_bins.py
