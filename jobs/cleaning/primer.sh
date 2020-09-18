dataset=$1
n_estimators=$2
max_depth=$3
max_features=$4
criterion=$5
mem=$6
time=$7
partition=$8

rs_list=(1 2 3 4 5)
method_list=('random' 'dart' 'dart_loss')

for rs in ${rs_list[@]}; do
    for method in ${method_list[@]}; do
        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=CL_$dataset \
               --output=jobs/logs/cleaning/$dataset \
               --error=jobs/errors/cleaning/$dataset \
               jobs/cleaning/runner.sh $dataset $criterion \
               $n_estimators $max_depth $max_features $method $rs
    done
done
