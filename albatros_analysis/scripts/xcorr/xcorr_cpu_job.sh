#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80              
#SBATCH --exclusive   
#SBATCH --time=01:00:00  

#SBATCH --job-name=xcorr_osamp(64)_pfb(256)
#SBATCH --output=/project/s/sievers/philj0ly/xcorr_cpu/logs/xcorr_output_%j_osamp(64)_pfb(256).txt
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/s/sievers/philj0ly/albatros_analysis/
module load NiaEnv/2022a
source bin/activate

cd /home/s/sievers/philj0ly/albatros_analysis/scripts/xcorr
python xcorr_gpu_cpu.py cpu config_axion.json

