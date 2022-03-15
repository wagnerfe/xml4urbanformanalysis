#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=ft_bld-%A_%a
#SBATCH --partition=standard
#SBATCH --account=metab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=bld_out-%A_%a.txt
#SBATCH --error=bld_err-%A_%a.txt
#SBATCH --array=0-30


pwd; hostname; date

module load anaconda

source activate urban_form

python -u trip_bldg_distance_based_features.py -i $SLURM_ARRAY_TASK_ID	
