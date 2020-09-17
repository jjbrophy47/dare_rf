dataset=$1
n_estimators=$2
max_depth=$3
max_features=$4
model=$5
scoring=$6
criterion=$7
mem=$8
time=$9
partition=${10}

rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=P_$dataset \
           --output=jobs/logs/performance/$dataset \
           --error=jobs/errors/performance/$dataset \
           jobs/performance/notune_bootstrap_runner.sh $dataset \
           $n_estimators $max_depth $max_features $model $scoring $criterion $rs
done
