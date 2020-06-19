#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=delete_until_retrain
#SBATCH --output=jobs/logs/delete_until_retrain/higgs1_en
#SBATCH --error=jobs/errors/delete_until_retrain/higgs1_en
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --account=uoml
module load python3/3.7.5

dataset="higgs"
n_estimators=100
max_depth=10
max_features=0.25
lmbdas=(350 350 250)
frac_remove=0.35
criterion="entropy"

data_dir="data/"
out_dir="output/delete_until_retrain/"
adversaries=("root")
rs_list=(1 2 3)


for i in ${!rs_list[@]}; do
    for adversary in ${adversaries[@]}; do
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

