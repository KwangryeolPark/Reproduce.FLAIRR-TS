#!/bin/bash
if [ -e venv/bin/activate ]; then
    source ./venv/bin/activate
fi

# Default parameters
DATA=${1:-ETTh1}
MODEL=${2:-deepseek-r1:latest}

python main.py \
    --data $DATA \
    --data_path $DATA.csv \
    --model $MODEL \
    --max_iter 5 \
    --sample_size 3 \
    --seq_len 96 \
    --pred_len 96
