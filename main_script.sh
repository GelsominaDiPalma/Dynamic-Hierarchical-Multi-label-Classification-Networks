device=0
max_parallel=10   
count=0

datasets=(
    "philosophy_ontology"
)
for dataset in "${datasets[@]}"
do
    for seed in {0..9}
    do 
        echo "Running: dataset=$dataset, seed=$seed, device=$device"
        python main.py --dataset "$dataset" --seed "$seed" --device "$device" &

        ((count++))
        if (( count % max_parallel == 0 )); then
            wait
        fi
    done
done
wait
