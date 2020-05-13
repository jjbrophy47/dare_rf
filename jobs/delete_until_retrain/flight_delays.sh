#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=delete_until_retrain
#SBATCH --output=jobs/logs/delete_until_retrain/flight_delays_gi
#SBATCH --error=jobs/errors/delete_until_retrain/flight_delays_gi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --account=uoml
module load python3/3.7.5

dataset="flight_delays"
n_estimators=100
max_depth=10
max_features=0.25
lmbdas=(2000 2000 2500 2000 2500)
frac_remove=0.35
criterion="gini"

data_dir="data/"
out_dir="output/delete_until_retrain/"
adversaries=("random" "root")
rs_list=(1 2 3 4 5)
epsilons=(0.1 0.25 0.5 1.0 10.0)


for i in ${!rs_list[@]}; do
    for adversary in ${adversaries[@]}; do
        for epsilon in ${epsilons[@]}; do
            python3 experiments/scripts/delete_until_retrain.py \
              --data_dir $data_dir \
              --out_dir $out_dir \
              --dataset $dataset \
              --n_estimators $n_estimators \
              --max_depth $max_depth \
              --max_features $max_features \
              --adversary $adversary \
              --criterion $criterion \
              --lmbda ${lmbdas[$i]} \
              --frac_remove $frac_remove \
              --rs ${rs_list[$i]}
        done
    done
done
