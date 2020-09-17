#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
n_estimators=$2
max_depth=$3
max_features=$4
tune_frac=$5
scoring=$6
criterion=$7

python3 scripts/experiments/topd_tuning.py --dataset $dataset \
    --n_estimators $n_estimators --max_depth $max_depth --max_features $max_features \
    --tune_frac $tune_frac --scoring $scoring --criterion $criterion
