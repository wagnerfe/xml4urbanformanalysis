#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=ft_buildings-%A_%a
#SBATCH --partition=standard
#SBATCH --account=metab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=aa_buildings_out-%A_%a.txt
#SBATCH --error=aa_buildings_err-%A_%a.txt
#SBATCH --array=5,14,15,19,23,26,28,29,30


pwd; hostname; date

module load anaconda

source activate inrix-env

python -u trip_bldg_distance_based_features.py -i $SLURM_ARRAY_TASK_ID
