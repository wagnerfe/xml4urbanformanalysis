#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=ft_streets-%A_%a
#SBATCH --partition=standard
#SBATCH --account=metab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=ft_streets_out-%A_%a.txt
#SBATCH --error=ft_streets_err-%A_%a.txt
#SBATCH --array=0-30


pwd; hostname; date

module load anaconda

source activate inrix-env

python -u trip_street_features.py -i $SLURM_ARRAY_TASK_ID	
