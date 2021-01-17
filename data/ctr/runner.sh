#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
#SBATCH --mem=60G
#SBATCH --time=1440
#SBATCH --partition=short
#SBATCH --job-name=CTR_preprocess
#SBATCH --output=job_output
#SBATCH --error=job_error
module load python3/3.7.5

python3 continuous.py
