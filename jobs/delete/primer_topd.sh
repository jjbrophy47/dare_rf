dataset=$1
criterion=$2
n_estimators=$3
max_depth=$4
k=$5
mem=$6
time=$7
partition=$8

out_dir='output/delete/'
rs_list=(1 2 3 4 5)
subsample_size_list=(1 1000)

for rs in ${rs_list[@]}; do
    for topd in $( seq 0 $max_depth ); do
        for subsample_size in ${subsample_size_list[@]}; do
            job_name=DEL_${dataset}_${criterion}_${rs}_${topd}_${subsample_size}
            sbatch --mem=${mem}G \
                   --time=$time \
                   --partition=$partition \
                   --job-name=$job_name \
                   --output=jobs/logs/delete/$dataset \
                   --error=jobs/errors/delete/$dataset \
                   jobs/delete/runner.sh $dataset $rs $criterion \
                   $n_estimators $max_depth $topd $k $subsample_size $out_dir
        done
    done
done
