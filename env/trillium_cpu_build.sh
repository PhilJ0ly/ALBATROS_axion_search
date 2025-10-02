#!/bin/bash

# Philippe Joly 2025-08-08

# This script is to setup a standard environment on the Trillium CPU machine compatible with the albatros_analysis repository

show_help() {
    cat << EOF
Usage: source trillium_cpu_build.sh [OPTIONS] <environment name>

Options:
    -h, --help  Shows help

Description:
    This script sets up a standard environment on the trillium GPU machine comparable with the albatros_analysis repository. 
    The environment is setup in the current directory.
    An environment log will be generated in ./.env/trillium_<name>_environment.txt outlining every pip package, modules loaded, and system properties.
EOF
}

REQ='/home/philj0ly/env/trillium_jupyter_requirements.txt'

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help 
else

    echo "Creating virtual environment..."
    module load StdEnv/2023 
    module load python/3.11.5
    module load gcc/12.3
    ARG1="${1:-pythonEnv}"
    
    # Create virtual environment and activate it
    python -m venv $ARG1
    source $ARG1/bin/activate

    echo "Installing packages with pip..."
    # Core packages
    pip install pip==25.2
    pip install -r $REQ


    echo "$ARG1 environment successfully created!"

    if [ ! -d "./env" ]; then
        mkdir "./env"
    fi
    LOG_FILE="./env/trillium_${ARG1}_environment.txt"
    echo "Environment: $ARG1" > $LOG_FILE
    echo "Created on: $(date)" >> $LOG_FILE
    echo "" >> $LOG_FILE

    # Log pip packages
    echo "=== Pip Packages ===" >> $LOG_FILE
    pip list >> $LOG_FILE

    # Log module versions
    echo "" >> $LOG_FILE
    echo "=== Loaded Modules ===" >> $LOG_FILE
    module list >> $LOG_FILE

    # Log system information
    echo "" >> $LOG_FILE
    echo "=== System Information ===" >> $LOG_FILE
    echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}')" >> $LOG_FILE
    echo "GCC Version: $(gcc --version | head -n 1)" >> $LOG_FILE
    echo "Python Version: $(python --version)" >> $LOG_FILE

    echo "Environment details saved to $LOG_FILE"

    deactivate

fi