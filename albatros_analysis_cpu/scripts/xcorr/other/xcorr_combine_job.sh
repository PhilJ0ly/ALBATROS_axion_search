#!/bin/bash
#SBATCH --job-name=xcorr_combine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=00:16:00
#SBATCH --partition=compute
#SBATCH --output=/scratch/s/sievers/philj0ly/logs/xcorr_combine_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# RESULTS COMBINATION JOB SCRIPT


cd /home/s/sievers/philj0ly/albatros_analysis/
module load NiaEnv/2022a
source bin/activate


echo "Job ID: $SLURM_JOB_ID"
echo "Combining results from $NNODES nodes"
echo "Output directory: $OUTPUT_DIR"

cd $SCRIPT_DIR

python xcorr_combine_cpu.py

echo "Results combination completed!"