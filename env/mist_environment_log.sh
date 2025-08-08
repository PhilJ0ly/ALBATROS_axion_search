#!/bin/bash

# Philippe Joly 22-05-2025

# This scripts simply logs all modules/dependencies of the current environment. 
# Note that teh script should be run while the nevironment is activated

ARG1=${1:-albatros}
if [ ! -d "./.env" ]; then
  mkdir "./.env"
fi
LOG_FILE="./.env/mist_${ARG1}_environment.txt"
echo "Environment: $ARG1" > $LOG_FILE
echo "Created on: $(date)" >> $LOG_FILE
echo "" >> $LOG_FILE

# Log conda environment
echo "=== Conda Environment Details ===" >> $LOG_FILE
conda list >> $LOG_FILE

# Log pip packages
echo "" >> $LOG_FILE
echo "=== Pip Packages ===" >> $LOG_FILE
pip list >> $LOG_FILE

# Log module versions
echo "" >> $LOG_FILE
echo "=== Loaded Modules ===" >> $LOG_FILE
module list 2>> $LOG_FILE

# Log system information
echo "" >> $LOG_FILE
echo "=== System Information ===" >> $LOG_FILE
echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}')" >> $LOG_FILE
echo "GCC Version: $(gcc --version | head -n 1)" >> $LOG_FILE
echo "Python Version: $(python --version)" >> $LOG_FILE

echo "Environment details saved to $LOG_FILE"

