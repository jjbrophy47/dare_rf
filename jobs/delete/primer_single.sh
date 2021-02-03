dataset=$1
criterion=$2
n_estimators=$3
max_depth=$4
k=$5
topd=$6
mem=$7
time=$8
partition=$9

rs_list=(1 2 3 4 5)
subsample_size_list=(1 1000)

for rs in ${rs_list[@]}; do
    for subsample_size in ${subsample_size_list[@]}; do
        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=DEL_$dataset \
               --output=jobs/logs/delete/$dataset \
               --error=jobs/errors/delete/$dataset \
               jobs/delete/runner.sh $dataset $rs $criterion \
               $n_estimators $max_depth $topd $k $subsample_size
    done
done
