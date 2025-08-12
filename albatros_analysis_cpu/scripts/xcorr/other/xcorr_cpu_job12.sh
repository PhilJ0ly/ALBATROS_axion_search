#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40              
#SBATCH --exclusive   
#SBATCH --time=02:00:00  

#SBATCH --job-name=xcorr_osamp(16)_pfb(8)_10
#SBATCH --output=/project/s/sievers/philj0ly/xcorr_cpu/logs/xcorr_output_%j_osamp(16)_pfb(8)_10.txt
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/s/sievers/philj0ly/albatros_analysis/
module load NiaEnv/2022a
source bin/activate

cd /home/s/sievers/philj0ly/albatros_analysis/scripts/xcorr
python xcorr_gpu_cpu.py cpu 10 config_axion_16.json



# Disregard
# 
# OSAMP=$(python -c 'import json,sys; print(json.load(sys.stdin)["correlation"]["osamp"])' < $CONFIGFN)
# PFB=$(python -c 'import json,sys; print(json.load(sys.stdin)["correlation"]["pfb_size_multiplier"])' < $CONFIGFN)
# NAME="osamp($OSAMP)_pfb($PFB)"
