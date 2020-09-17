dataset=$1
model=$2
tune_frac=$3
scoring=$4
criterion=$5
mem=$6
time=$7
partition=$8

rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=P_$dataset \
           --output=jobs/logs/performance/$dataset \
           --error=jobs/errors/performance/$dataset \
           jobs/performance/bootstrap_runner.sh $dataset $model $tune_frac $scoring $criterion $rs
done