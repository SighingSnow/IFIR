
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model> <dataset> <batch_size> <device>"
    exit 1
fi

if [ "$#" -eq 4 ]; then
    python src/build_index.py \
        --model $1 \
        --dataset $2 \
        --batch_size $3 \
        --device $4
elif [ "$#" -eq 5 ]; then 
    python src/build_index.py \
        --model $1 \
        --dataset $2 \
        --batch_size $3 \
        --domain $4 \
        --device $5
fi