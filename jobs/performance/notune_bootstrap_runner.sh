#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
n_estimators=$2
max_depth=$3
max_features=$4
model=$5
scoring=$6
criterion=$7
rs=$8

python3 scripts/experiments/performance.py --dataset $dataset \
    --n_estimators $n_estimators --max_depth $max_depth --max_features $max_features \
    --model $model --scoring $scoring --criterion $criterion --no_tune --bootstrap --rs $rs
