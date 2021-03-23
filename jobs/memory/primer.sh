dataset=$1
criterion=$2
mem=$3
time=$4
partition=$5

sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=MEM_$dataset \
       --output=jobs/logs/memory/$dataset \
       --error=jobs/errors/memory/$dataset \
       jobs/memory/runner.sh $dataset $criterion
