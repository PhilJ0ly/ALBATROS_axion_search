#!/bin/bash

sbatch --begin=00:01 xcorr_cpu_job.sh
sbatch --begin=00:31 xcorr_cpu_job2.sh

sbatch --begin=01:01 xcorr_cpu_job4.sh
sbatch --begin=01:31 xcorr_cpu_job5.sh

sbatch --begin=02:01 xcorr_cpu_job7.sh
sbatch --begin=02:31 xcorr_cpu_job8.sh

sbatch --begin=03:01 xcorr_cpu_job10.sh
sbatch --begin=04:01 xcorr_cpu_job11.sh
sbatch --begin=05:01 xcorr_cpu_job12.sh