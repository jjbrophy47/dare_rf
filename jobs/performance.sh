#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=performance
#SBATCH --output=jobs/logs/performance/gas_sensor
#SBATCH --error=jobs/errors/performance/gas_sensor
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --account=uoml
module load python3/3.7.5

# Higgs: use 0.05 tune_frac and reduce_search
dataset="gas_sensor"
criterion="gini"
tune_frac=1.0
model_type="forest"

scoring="roc_auc"
data_dir="data/"
out_dir="output/performance/"
verbose=2


python3 experiments/scripts/performance.py \
  --data_dir $data_dir \
  --out_dir $out_dir \
  --dataset $dataset \
  --tune_frac $tune_frac \
  --model_type $model_type \
  --scoring $scoring \
  --criterion $criterion \
  --verbose $verbose
  # --reduce_search
