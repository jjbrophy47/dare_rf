dataset=$1
criterion=$2
mem=$3
time=$4
partition=$5

rs_list=(1 2 3 4 5)
model_list=('sklearn' 'dare_0' 'dare_1' 'dare_2' 'dare_3' 'dare_4')

for rs in ${rs_list[@]}; do
    for model in ${model_list[@]}; do
        sbatch --mem=${mem}G \
               --time=$time \
               --partition=$partition \
               --job-name=MEM_$dataset \
               --output=jobs/logs/memory/$dataset \
               --error=jobs/errors/memory/$dataset \
               jobs/memory/runner.sh $dataset $criterion $model $rs
    done
done
