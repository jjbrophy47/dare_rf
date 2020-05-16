#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=addition
#SBATCH --output=jobs/logs/addition/bank_marketing_en
#SBATCH --error=jobs/errors/addition/bank_marketing_en
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=uoml
module load python3/3.7.5

dataset="bank_marketing"
n_estimators=100
max_depth=5
max_features=0.25
lmbdas=(75 50 50 50 50)
rs_list=(1 2 3 4 5)
criterion="entropy"

data_dir="data/"
out_dir="output/addition/"
adversaries=("root")
epsilons=(0.1 0.25 0.5 1.0)

for i in ${!rs_list[@]}; do
    for adversary in ${adversaries[@]}; do
        python3 experiments/scripts/addition.py --data_dir $data_dir --out_dir $out_dir \
          --dataset $dataset --naive --exact \
          --n_estimators $n_estimators --max_depth $max_depth \
          --max_features $max_features \
          --adversary $adversary --criterion $criterion --rs ${rs_list[$i]}

        for epsilon in ${epsilons[@]}; do
            python3 experiments/scripts/addition.py --data_dir $data_dir --out_dir $out_dir \
              --dataset $dataset --cedar --lmbda ${lmbdas[$i]} --epsilon $epsilon \
              --n_estimators $n_estimators --max_depth $max_depth \
              --max_features $max_features \
              --adversary $adversary --criterion $criterion --rs ${rs_list[$i]}
        done
    done
done

