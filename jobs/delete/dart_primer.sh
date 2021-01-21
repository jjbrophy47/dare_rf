dataset=$1
criterion=$2
n_estimators=$3
max_depth=$4
k=$5
mem=$6
time=$7
partition=$8

rs_list=(1 2 3 4 5)
subsample_size_list=(1 1000)
topd=0

for rs in ${rs_list[@]}; do
    # for topd in $( seq 0 $max_depth ); do
    for subsample_size in ${subsample_size_list[@]}; do
        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=DEL_{$dataset}_{$rs}_{$topd}_{$subsample_size} \
               --output=jobs/logs/delete/{$dataset}_{$rs}_{$topd}_{$subsample_size} \
               --error=jobs/errors/delete/{$dataset}_{$rs}_{$topd}_{$subsample_size} \
               jobs/delete/dart_runner.sh $dataset $rs $criterion \
               $n_estimators $max_depth $topd $k $subsample_size
    done
    # done
done
