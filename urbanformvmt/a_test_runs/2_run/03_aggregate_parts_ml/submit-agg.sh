#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=agg-%A
#SBATCH --partition=standard
#SBATCH --account=vwproject
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --output=agg_out_%A.txt
#SBATCH --error=agg_err_%A.txt

pwd; hostname; date

module load anaconda

source activate inrix_env2

python -u aggreg_input.py 
