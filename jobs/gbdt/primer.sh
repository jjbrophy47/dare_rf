dataset=$1
model=$2
tune_frac=$3
scoring=$4
mem=$5
time=$6
partition=$7

sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=GBDT_$dataset \
       --output=jobs/logs/gbdt/$dataset \
       --error=jobs/errors/gbdt/$dataset \
       jobs/gbdt/runner.sh $dataset $model $tune_frac $scoring
