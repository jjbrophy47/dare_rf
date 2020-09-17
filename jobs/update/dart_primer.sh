dataset=$1
n_estimators=$2
max_depth=$3
max_features=$4
criterion=$5
operation=$6
mem=$7
time=$8
partition=$9

rs_list=(1 2 3 4 5)
subsample_size_list=(1 1000)

for rs in ${rs_list[@]}; do
    for topd in $( seq 0 $max_depth ); do
        for subsample_size in ${subsample_size_list[@]}; do
            sbatch --mem=${mem}G \
                   --time=$time \
                   --partition=$partition \
                   --job-name=UD_$dataset \
                   --output=jobs/logs/update/$dataset \
                   --error=jobs/errors/update/$dataset \
                   jobs/update/dart_runner.sh $dataset $rs $criterion \
                   $n_estimators $max_depth $max_features $topd $subsample_size $operation
        done
    done
done
