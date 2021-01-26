#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
model=$2
tune_frac=$3
scoring=$4

python3 scripts/experiments/gbdt.py --dataset $dataset \
    --tune_frac $tune_frac --scoring $scoring --model_type $model
