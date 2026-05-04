#!/bin/bash
if [ -e venv/bin/activate ]; then
    source ./venv/bin/activate
fi

# Default parameters
DATA=${1:-ETTh1}
MODEL=${2:-deepseek-r1:32b}

# python main.py \
#     --data $DATA \
#     --data_path $DATA.csv \
#     --model $MODEL \
#     --max_iter 3 \
#     --sample_size 10 \
#     --seq_len 96 \
#     --pred_len 96

python main.py \
    --data $DATA \
    --data_path $DATA.csv \
    --model $MODEL \
    --max_iter 3 \
    --sample_size 10 \
    --seq_len 96 \
    --pred_len 96 \
    --prompt_type dungeon-master \
    --test


# python main.py \
#     --data $DATA \
#     --data_path $DATA.csv \
#     --model $MODEL \
#     --max_iter 3 \
#     --sample_size all \
#     --seq_len 96 \
#     --pred_len 96 \
#     --prompt_type dungeon-master
