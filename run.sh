#!/bin/bash

# Define the lists
version="gsm8k-target-pythia-1.4b"
model="EleutherAI/pythia-1.4b"

remove_before_slash() {
    local input_string="$1"
    local output_string="${input_string##*/}"
    echo "$output_string"
}

sh run_train.sh "$version" "$model"