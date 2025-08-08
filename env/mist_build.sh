#!/bin/bash

# Philippe Joly 2025-05-21

# This script is to setup a standard environment on the MIST machine compatible with the albatros_analysis repository

show_help() {
	cat << EOF
Usage: source mist_build.sh [OPTIONS] <environment name>

Options:
	-h, --help	Shows help

Description:
	This script sets up a standard environment on the MIST machine comparible with the albatros_analysis repository. 
	The environment is setup in the current directory.
	A environment log will be generated in ./.env/mist_<name>_environment.txt outlining every conda/pip packages, modules loaded, and system properties.
EOF
}


if [[ "$1" == "-h" || "$1" == "--help" ]]; then
	show_help 
else

	echo "Creating anaconda3 environment..."
	module load MistEnv/2021a 
	module load anaconda3/2021.05
	ARG1="${1:-pythonEnv}"
	conda create -n $ARG1 python=3.10
	source activate $ARG1

	conda config --add channels conda-forge
	conda config --set channel_priority strict

	conda install numpy=1.26.4 numba=0.59.1 matplotlib=3.8.3 astropy=6.1.7 sgp4=2.24 skyfield=1.53 pandas=2.2.2 psutil=7.0.0 scipy=1.15.2

	module load cuda/11.7.1
	module load gcc/11.4.0

	conda install -c conda-forge cupy=13.1.0 cudatoolkit=11.7
	
	conda clean -y --all
       	rm -rf $HOME/.conda/pkgs/*

	echo "$ARG1 environment successfully created!"

	if [ ! -d "./env" ]; then
	  mkdir "./env"
	fi
	LOG_FILE="./env/mist_${ARG1}_environment.txt"
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

	source deactivate

fi
