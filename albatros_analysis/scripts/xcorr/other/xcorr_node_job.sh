#!/bin/bash
#SBATCH --job-name=xcorr_node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=00:16:00
#SBATCH --partition=compute
#SBATCH --output=/scratch/s/sievers/philj0ly/logs/xcorr_node_%j_%a.out

# NODE PROCESSING JOB SCRIPT

# Load required modules
cd /home/s/sievers/philj0ly/albatros_analysis/
module load NiaEnv/2022a
source bin/activate


# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node ID: $NODE_ID"
echo "Chunk range: $CHUNK_START to $((CHUNK_END-1))"
echo "Output directory: $OUTPUT_DIR"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"

cd $SCRIPT_DIR

python xcorr_cpu_mem.py $CONFIG_FN

echo "Node $NODE_ID processing completed!"