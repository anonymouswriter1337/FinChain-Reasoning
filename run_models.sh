#!/bin/bash

# Define an array of model names
models=("wiroai_finance_qwen_7b" "wiroai_finance_llama_8b"  "qwen_2p5_7b" "llama3p1_8b_instruct" "deepseek_r1_distill_llama_8b" "deepseek_r1_distill_qwen_7b" "gemma_2_9b_instruct" "mistral_7b_instruct_v0p3" "fin_r1")


# Loop through each model and run the Python script
for model in "${models[@]}"; do
    echo "Running model: $model"
    python generation.py --model "$model"
done
