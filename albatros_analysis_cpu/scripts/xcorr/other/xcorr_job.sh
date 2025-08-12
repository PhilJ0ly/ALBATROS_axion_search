#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --time=00:15:00
#SBATCH --job-name=xcorr_job
#SBATCH --output=/project/s/sievers/mohanagr/cpu_all_antenna/xcorr_output_%j.txt
#SBATCH --mail-type=END,FAIL

source /home/s/sievers/mohanagr/.virtualenvs/base/bin/activate
module load NiaEnv/2022a
module load gcc/11.3.0 fftw/3.3.10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/s/sievers/mohanagr/albatros_analysis/scripts/xcorr

python xcorr_cpu2.py
