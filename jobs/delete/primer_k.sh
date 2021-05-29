dataset=$1
criterion=$2
n_estimators=$3
max_depth=$4
k=$5
mem=$6
time=$7
partition=$8

out_dir='output/increase_k/'
topd=0
rs_list=(1 2 3 4 5)
k_list=(1 5 10 25 50 100)
subsample_size_list=(1 1000)

for rs in ${rs_list[@]}; do
    for k in ${k_list[@]}; do
        for subsample_size in ${subsample_size_list[@]}; do
            sbatch --mem=${mem}G \
                   --time=$time \
                   --partition=$partition \
                   --job-name=DEL_$dataset \
                   --output=jobs/logs/increase_k/$dataset \
                   --error=jobs/errors/increase_k/$dataset \
                   jobs/delete/runner.sh $dataset $rs $criterion \
                   $n_estimators $max_depth $topd $k $subsample_size $out_dir
        done
    done
done
