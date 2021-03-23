#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

dataset=$1
criterion=$2
model=$3
rs=$4

python3 scripts/experiments/memory.py \
  --dataset $dataset \
  --criterion $criterion \
  --model $model \
  --rs $rs
