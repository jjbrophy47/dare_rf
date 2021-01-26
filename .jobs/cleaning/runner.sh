#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
criterion=$2
n_estimators=$3
max_depth=$4
max_features=$5
method=$6
rs=$7

python3 scripts/experiments/cleaning.py \
  --dataset $dataset \
  --criterion $criterion \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --max_features $max_features \
  --method $method \
  --rs $rs