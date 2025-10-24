#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <version> <model>"
    exit 1
fi

# Assign the arguments to variables
version="$1" # change this to your desired version name
model="$2" # this can be either "gpt2*" or "opt-*", or leave it blank, which means using the default model.

# Check if the ./logs folder exists
if [ ! -d "./logs" ]; then
    # If the folder doesn't exist, create it
    mkdir "./logs"
    echo "Created ./logs folder"
else
    echo "./logs folder already exists"
fi

# Check if the ./logs/$version folder exists
if [ ! -d "./logs/$version" ]; then
    # If the folder doesn't exist, create it
    mkdir "./logs/$version"
    echo "Created ./logs/$version folder"
else
    echo "./logs/$version folder already exists"
fi

nvidia-smi

accelerate launch --config_file accelerate_configs/zero1.yaml train.py \
    --draft_model_name_or_path $model \
    --target_model_name_or_path "path/to/target/checkpoint-2802" \
    --ref_model_name_or_path "path/to/ref/checkpoint-7020" \
    \
    --data_name gsm8k \
    \
    --bf16 True \
    --output_dir "./checkpoints/$version" \
    --num_train_epochs 75 \
    --per_device_train_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 1 \
    --eval_strategy "steps" \
    --eval_steps 468 \
    --save_strategy "steps" \
    --save_steps 468 \
    --batch_eval_metrics True \
    --prediction_loss_only True \
    --eval_on_start \
    --save_only_model True \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --report_to tensorboard \
    --logging_dir "./logs/$version" \
    --gradient_checkpointing True \
    --k 0.4
