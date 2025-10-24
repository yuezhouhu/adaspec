#!/bin/bash

# Define the lists
version="gsm8k-target-phi2"

model="microsoft/phi-2"

remove_before_slash() {
    local input_string="$1"
    local output_string="${input_string##*/}"
    echo "$output_string"
}

sh run_train.sh "$version" "$model"