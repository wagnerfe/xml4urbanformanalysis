#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=ft_blocks-%A_%a
#SBATCH --partition=standard
#SBATCH --account=metab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=ft2_blocks_out-%A_%a.txt
#SBATCH --error=ft2_blocks_err-%A_%a.txt
#SBATCH --array=5


pwd; hostname; date

module load anaconda

source activate inrix_env2

python -u trip_block_distance_based_features.py -i $SLURM_ARRAY_TASK_ID 
