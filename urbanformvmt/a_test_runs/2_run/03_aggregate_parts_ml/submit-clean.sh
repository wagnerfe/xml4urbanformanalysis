#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=clean-%A
#SBATCH --partition=standard
#SBATCH --account=vwproject
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=clean_out_%A.txt
#SBATCH --error=clean_err_%A.txt

pwd; hostname; date

module load anaconda

source activate inrix_env2

python -u clean_input.py 
