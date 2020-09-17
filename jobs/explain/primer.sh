dataset=$1
n_estimators=$2
max_depth=$3
max_features=$4
criterion=$5
mem=$6
time=$7
partition=$8

sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=EX_$dataset \
       --output=jobs/logs/explain/$dataset \
       --error=jobs/errors/explain/$dataset \
       jobs/explain/runner.sh $dataset $criterion \
       $n_estimators $max_depth $max_features
