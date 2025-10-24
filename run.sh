#!/bin/bash

# Define the lists
version="gsm8k-target-phi2-draft-codegen-350m-mono-best-3epoch"
model="Salesforce/codegen-350M-mono"

remove_before_slash() {
    local input_string="$1"
    local output_string="${input_string##*/}"
    echo "$output_string"
}

sh run_train.sh "$version" "$model"
