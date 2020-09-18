dataset=$1
n_estimators=$2
max_depth=$3
max_features=$4
criterion=$5
mem=$6
time=$7
partition=$8

rs_list=(1 2 3 4 5 6 7 8 9 10)
method_list=('random' 'dart' 'dart_loss')

for rs in ${rs_list[@]}; do
    for method in ${method_list[@]}; do
        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=RO_$dataset \
               --output=jobs/logs/roar/$dataset \
               --error=jobs/errors/roar/$dataset \
               jobs/roar/runner.sh $dataset $criterion \
               $n_estimators $max_depth $max_features $method $rs
    done
done
