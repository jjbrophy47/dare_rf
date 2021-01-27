#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
model=$2
n_estimators=$3
max_depth=$4
k=$5
scoring=$6
criterion=$7
rs=$8

python3 scripts/experiments/performance.py \
    --dataset $dataset \
    --n_estimators $n_estimators \
    --max_depth $max_depth \
    --k $k \
    --model $model \
    --scoring $scoring \
    --criterion $criterion \
    --no_tune \
    --rs $rs
