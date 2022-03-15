#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=test_blocks-%A_%a
#SBATCH --partition=standard
#SBATCH --account=metab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=finaltest_blocks_out-%A_%a.txt
#SBATCH --error=finaltest_blocks_err-%A_%a.txt
#SBATCH --array=27


pwd; hostname; date

module load anaconda

source activate inrix-env

python -u trip_block_distance_based_features.py -i $SLURM_ARRAY_TASK_ID 
