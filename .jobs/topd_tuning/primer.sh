dataset=$1
criterion=$2
n_estimators=$3
max_depth=$4
k=$5
scoring=$6
tune_frac=$7
mem=$8
time=$9
partition=${10}

rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=TOPD_$dataset \
           --output=jobs/logs/topd_tuning/$dataset \
           --error=jobs/errors/topd_tuning/$dataset \
           jobs/topd_tuning/runner.sh $dataset \
           $n_estimators $max_depth $k $tune_frac $scoring $criterion $rs
done
