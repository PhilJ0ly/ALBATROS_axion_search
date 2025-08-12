#!/bin/bash

# SLURM Job Submission Scripts for XCORR Re-PFB distributed accross NIAGARA CPU Nodes


# Configuration
NNODES=1
OUTPUT_DIR="/scratch/s/sievers/philj0ly/cpu_results"
CONFIG_FN="config_axion.json"
SCRIPT_DIR="/home/s/sievers/philj0ly/albatros_analysis/scripts/xcorr"

cd /home/s/sievers/philj0ly/albatros_analysis/
module load NiaEnv/2022a
source bin/activate

cd $SCRIPT_DIR

# Calculate chunks per node
NCHUNKS=$(python get_chunks_num.py $SCRIPT_DIR/$CONFIG_FN)
CHUNKS_PER_NODE=$((NCHUNKS / NNODES))
REMAINDER=$((NCHUNKS % NNODES))

echo "Submitting ALBATROS distributed processing jobs:"
echo "Total chunks: $NCHUNKS"
echo "Number of nodes: $NNODES"
echo "Chunks per node: $CHUNKS_PER_NODE"
echo "Remainder chunks: $REMAINDER"


# Array to store job IDs for dependency management
JOB_IDS=()

# Submit individual node jobs
for ((node_id=0; node_id<NNODES; node_id++)); do\
    chunk_start=$((node_id * CHUNKS_PER_NODE))
    
    if [ $node_id -eq $((NNODES - 1)) ]; then
        # Last node gets remainder chunks
        chunk_end=$((chunk_start + CHUNKS_PER_NODE + REMAINDER))
    else
        chunk_end=$((chunk_start + CHUNKS_PER_NODE))
    fi
    
    echo "Node $node_id: chunks $chunk_start to $((chunk_end-1))"
    
    # Submit job for this node
    JOB_ID=$(sbatch --parsable \
        --export=NODE_ID=$node_id,CHUNK_START=$chunk_start,CHUNK_END=$chunk_end,OUTPUT_DIR=$OUTPUT_DIR,CONFIG_FN=$CONFIG_FN,SCRIPT_DIR=$SCRIPT_DIR \
        $SCRIPT_DIR/xcorr_node_job.sh)
    
    JOB_IDS+=($JOB_ID)
    echo "Submitted job $JOB_ID for node $node_id"
done

# Submit combination job that depends on all node jobs
DEPENDENCY_LIST=$(IFS=:; echo "${JOB_IDS[*]}")
COMBINE_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$DEPENDENCY_LIST \
    --export=OUTPUT_DIR=$OUTPUT_DIR,NNODES=$NNODES,$SCRIPT_DIR \
    $SCRIPT_DIR/xcorr_combine_job.sh)

echo "Submitted combination job $COMBINE_JOB_ID (depends on: $DEPENDENCY_LIST)"
echo "All jobs submitted successfully!"