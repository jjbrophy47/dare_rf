#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
rs=$2
criterion=$3
n_estimators=$4
max_depth=$5
topd=$6
k=$7
subsample_size=$8

python3 scripts/experiments/delete.py \
  --dataset $dataset \
  --rs $rs \
  --criterion $criterion \
  --n_estimators $n_estimators \
  --max_depth $max_depth \
  --topd $topd \
  --K $k \
  --subsample_size $subsample_size
