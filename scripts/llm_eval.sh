
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model> <dataset> <topk> <device> "
    exit 1
fi

if [ "$#" -eq 4 ]; then 
    python3 src/llm_eval.py \
        --model $1 \
        --dataset $2 \
        --topk $3 \
        --device $4 
elif [ "$#" -eq 5 ]; then 
    python3 src/llm_eval.py \
    --model $1 \
    --dataset $2 \
    --topk $3 \
    --domain $4 \
    --device $5 
elif [ "$#" -eq 6 ]; then 
    python3 src/llm_eval.py \
    --model $1 \
    --dataset $2 \
    --topk $3 \
    --domain $4 \
    --device $5 \
    --sample True 
fi