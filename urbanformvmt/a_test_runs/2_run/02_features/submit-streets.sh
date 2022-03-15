#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=streets-%A_%a
#SBATCH --partition=standard
#SBATCH --account=vwproject
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=streets_out-%A_%a.txt
#SBATCH --error=streets_err-%A_%a.txt
#SBATCH --array=19,28,30


pwd; hostname; date

module load anaconda

source activate inrix_env2

python -u trip_street_features.py -i $SLURM_ARRAY_TASK_ID	
