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
epsilon_list=(10)
lmbda_list=(0.1 0.0001)
topd=$max_depth

for rs in ${rs_list[@]}; do
    for subsample_size in ${subsample_size_list[@]}; do
        for epsilon in ${epsilon_list[@]}; do
                for lmbda in ${lmbda_list[@]}; do
                    sbatch --mem=${mem}G \
                           --time=$time \
                           --partition=$partition \
                           --job-name=UC_$dataset \
                           --output=jobs/logs/update/$dataset \
                           --error=jobs/errors/update/$dataset \
                           jobs/update/cedar_runner.sh $dataset $rs $criterion \
                           $n_estimators $max_depth $max_features $topd $subsample_size $operation \
                           $epsilon $lmbda
            done
        done
    done
done
