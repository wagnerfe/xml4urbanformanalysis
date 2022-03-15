#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=buildings-%A_%a
#SBATCH --partition=standard
#SBATCH --account=vwproject
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=buildings_out-%A_%a.txt
#SBATCH --error=buildings_err-%A_%a.txt
#SBATCH --array=0


pwd; hostname; date

module load anaconda

source activate inrix_env2

python -u trip_bldg_distance_based_features.py -i $SLURM_ARRAY_TASK_ID
