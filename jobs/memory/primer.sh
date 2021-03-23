dataset=$1
criterion=$2
mem=$3
time=$4
partition=$5

rs_list=(1 2 3 4 5)

for rs in ${rs_list[@]}; do
    sbatch --mem=${mem}G \
           --time=$time \
           --partition=$partition \
           --job-name=MEM_$dataset \
           --output=jobs/logs/memory/$dataset \
           --error=jobs/errors/memory/$dataset \
           jobs/memory/runner.sh $dataset $criterion $rs
done
