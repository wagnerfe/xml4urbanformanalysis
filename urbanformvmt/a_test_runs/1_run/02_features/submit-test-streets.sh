#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=finaltest_test_streets-%A_%a
#SBATCH --partition=standard
#SBATCH --account=metab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=finaltest_streets_out-%A_%a.txt
#SBATCH --error=finaltest_streets_err-%A_%a.txt
#SBATCH --array=27


pwd; hostname; date

module load anaconda

source activate inrix-env

python -u trip_street_features.py -i $SLURM_ARRAY_TASK_ID	
