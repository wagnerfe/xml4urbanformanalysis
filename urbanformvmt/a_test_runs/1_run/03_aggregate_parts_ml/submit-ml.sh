#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=test_ml-%A
#SBATCH --partition=standard
#SBATCH --constraint=broadwell
#SBATCH --account=metab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=test_ml_out-%A.txt
#SBATCH --error=test_ml_err-%A.txt

pwd; hostname; date

module load anaconda

source activate inrix_env2

python -u ml.py 
