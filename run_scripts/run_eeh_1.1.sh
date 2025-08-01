#!/bin/bash
# filepath: /cb/home/clairez/ws/depth_mup/lm-evaluation-harness/run_eeh_submit.sh

model_list=(
    # "allenai/OLMo-2-0425-1B"
    # "Qwen/Qwen3-0.6B-Base"
    # "microsoft/phi-1"
    # "Qwen/Qwen2.5-0.5B"
    # "Qwen/Qwen2.5-1.5B"
    # "Qwen/Qwen2.5-3B"
    # "google/gemma-3-1b-pt"
    # "google/gemma-3-4b-pt"
    # "google/gemma-2-2b"
    # "Zyphra/Zamba2-1.2B"
    # "Zyphra/Zamba2-2.7B"
    # "microsoft/phi-2"
    # "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B"
    # "cerebras/btlm-3b-8k-base"
    # "cerebras/Cerebras-GPT-111M"
    # "cerebras/Cerebras-GPT-256M"
    # "cerebras/Cerebras-GPT-590M"
    # "cerebras/Cerebras-GPT-1.3B"
    # "cerebras/Cerebras-GPT-2.7B"
    # "HuggingFaceTB/SmolLM2-135M"
    # "HuggingFaceTB/SmolLM2-360M"
    # "HuggingFaceTB/SmolLM2-1.7B"
    # "HuggingFaceTB/SmolVLM-Base"
    # "HuggingFaceTB/SmolLM-135M"
    # "HuggingFaceTB/SmolLM-360M"
    # "HuggingFaceTB/SmolLM-1.7B"
    # "HuggingFaceTB/SmolLM3-3B"
    # "HuggingFaceTB/SmolLM3-3B-Base"
    "bczhang/Celerity-300M-draft"
    "bczhang/Celerity-500M-draft"
    "bczhang/Celerity-900M-draft"
)

tasks=(
    # "arc_challenge"
    # "arc_easy"
    # "boolq"
    # "hellaswag"
    # "piqa"
    # "social_iqa"
    # "winogrande"
    # "race"
    # "lambada"
    "mmlu"
)

shots=(
    # 25
    # 0
    # 0
    # 10
    # 0
    # 0
    # 5
    # 0
    # 0
    0 #5
)

cd ~/ws/depth_mup/lm-evaluation-harness
source ~/ws/miniconda3/bin/activate
conda activate eeh_env

huggingface-cli login --token XXX

# Iterate through each model
for model in "${model_list[@]}"; do

    # Extract model name for the output directory
    model_name=$(echo "$model" | awk -F/ '{print $NF}')
    mkdir -p "output/${model_name}"
    
    # Iterate through each task
    for i in "${!tasks[@]}"; do
        task=${tasks[$i]}
        shot=${shots[$i]}
        
        echo "Submitting job for model=${model}, task=${task}, shots=${shot}"

        timestamp=$(date +"%Y%m%d_%H%M%S")
        log_file="logs/${model_name}/${model_name}_${task}_${timestamp}.txt"
        mkdir -p "logs/${model_name}"


        lm_eval \
        --model hf \
        --model_args pretrained=${model},dtype="bfloat16",parallelize=True,trust_remote_code=True \
        --tasks ${task} \
        --num_fewshot ${shot} \
        --device cuda \
        --batch_size auto \
        --output_path output/${model_name} \
        --log_samples \
        &> "$log_file"

    done
done