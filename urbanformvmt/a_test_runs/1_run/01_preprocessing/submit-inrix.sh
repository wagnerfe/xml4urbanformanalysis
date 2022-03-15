#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=inrix_parts-%A_%a
#SBATCH --partition=standard
#SBATCH --account=metab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=inrix_parts_test-%A_%a.txt
#SBATCH --error=inrix_parts_err-%A_%a.txt


pwd; hostname; date

module load anaconda

source activate building-env

python -u inrix-parts.py
