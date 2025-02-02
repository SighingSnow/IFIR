
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model> <dataset> <topk> <device> "
    exit 1
fi

if [ "$#" -eq 4 ]; then 
    python src/eval.py \
        --model $1 \
        --dataset $2 \
        --topk $3 \
        --device $4 
elif [ "$#" -eq 5 ]; then 
    python src/eval.py \
    --model $1 \
    --dataset $2 \
    --topk $3 \
    --domain $4 \
    --device $5 
fi