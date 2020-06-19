#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=lmbda
#SBATCH --output=jobs/logs/lmbda/flight_delays
#SBATCH --error=jobs/errors/lmbda/flight_delays
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.7.5

model_type=$1
criterion=$2
tune_frac=$3
step_size=$4
start_val=$5

dataset="flight_delays"
n_estimators=250
max_depth=20
max_features=0.25

data_dir="data/"
out_dir="output/lmbda/"
rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    python3 experiments/scripts/lmbda.py \
      --data_dir $data_dir \
      --out_dir $out_dir \
      --dataset $dataset \
      --start_val $start_val \
      --step_size $step_size \
      --model_type $model_type \
      --n_estimators $n_estimators \
      --max_depth $max_depth \
      --max_features $max_features \
      --criterion $criterion \
      --rs $rs
done
