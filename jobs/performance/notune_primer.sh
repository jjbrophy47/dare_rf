dataset=$1
model=$2
n_estimators=$3
max_depth=$4
k=$5
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
           jobs/performance/notune_runner.sh $dataset \
           $model $n_estimators $max_depth $k $scoring $criterion $rs
done
