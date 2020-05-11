#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=delete_until_retrain
#SBATCH --output=jobs/logs/delete_until_retrain/gas_sensor
#SBATCH --error=jobs/errors/delete_until_retrain/gas_sensor
#SBATCH --time=14-00:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --account=uoml
module load python3/3.7.5

dataset="gas_sensor"
n_estimators=100
max_depth=10
max_features=0.25
lmbdas=(320 340 260 340 360)
frac_remove 0.5
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
              --epsilon $epsilon \
              --rs ${rs_list[$i]}
        done
    done
done

