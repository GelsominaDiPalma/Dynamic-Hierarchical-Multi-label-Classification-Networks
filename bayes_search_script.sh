device=0               
max_parallel=1      
max_evals=40     
count=0

datasets=(
    "philosophy_ontology"
)
for dataset in "${datasets[@]}"; do
    python train_hyper.py \
        --dataset "$dataset" \
        --device "$device" \
        --max_evals "$max_evals" \
        --new_search 0 \
        --seed 0 &

    ((count++))
    if (( count % max_parallel == 0 )); then
        wait
    fi
done
wait
