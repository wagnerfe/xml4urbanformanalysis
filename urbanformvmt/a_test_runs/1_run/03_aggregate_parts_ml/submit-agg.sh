#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=ft_agg-%A
#SBATCH --partition=standard
#SBATCH --account=metab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=ft_agg_out.txt
#SBATCH --error=ft_agg_err.txt

pwd; hostname; date

module load anaconda

source activate inrix-env

python -u aggreg_input.py 
