#!/bin/bash

# Define the lists
version="gsm8k-target-pythia-1.4b-draft-pythia-31m-best"
model="EleutherAI/pythia-31m"

remove_before_slash() {
    local input_string="$1"
    local output_string="${input_string##*/}"
    echo "$output_string"
}

sh run_train.sh "$version" "$model"
