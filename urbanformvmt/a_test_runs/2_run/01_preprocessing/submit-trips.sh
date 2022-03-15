#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=inrix_250k_lb_ub_weekday_nocom-%A_%a
#SBATCH --partition=standard
#SBATCH --account=vwproject
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=inrix_250k_lb_ub_weekday_nocom_out-%A_%a.txt
#SBATCH --error=inrix_250k_lb_ub_weekday_nocom_err-%A_%a.txt


pwd; hostname; date

module load anaconda

source activate inrix_env2

python -u clean_trips.py
