#!/bin/bash
# filepath: /cb/home/clairez/ws/depth_mup/lm-evaluation-harness/run_eeh_submit.sh

model_list=(
    # "Qwen/Qwen3-0.6B-Base"
    # "microsoft/phi-1"
    # "Qwen/Qwen2.5-0.5B"
    # "Qwen/Qwen2.5-1.5B"
    # "Qwen/Qwen2.5-3B"
    # "google/gemma-3-1b-pt"
    # "google/gemma-3-4b-pt"
    # "microsoft/phi-2"
    # "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B"
    # "cerebras/btlm-3b-8k-base"
    # "cerebras/Cerebras-GPT-111M"
    # "cerebras/Cerebras-GPT-256M"
    # "cerebras/Cerebras-GPT-590M"
    # "cerebras/Cerebras-GPT-1.3B"
    # "cerebras/Cerebras-GPT-2.7B"
    # "google/gemma-3-12b-pt"
    # "google/gemma-3-27b-pt"
    # "google/gemma-2-2b"
    # "google/gemma-2-9b"
    # "google/gemma-2-27b"
    # "Zyphra/Zamba2-1.2B"
    # "Zyphra/Zamba2-2.7B"
    # "allenai/OLMo-2-0425-1B"
    # "allenai/OLMo-2-1124-7B"
    # "allenai/OLMo-2-1124-13B"
    "allenai/OLMo-2-0325-32B"
    # "allenai/OLMo-1B-hf"
    # "allenai/OLMo-7B-hf"
)

tasks=(
    # "arc_challenge"
    # "arc_easy"
    # "boolq"
    "hellaswag"
    "piqa"
    "social_iqa"
    "winogrande"
    "race"
    "lambada"
    # "mmlu"
)

shots=(
    # 25
    # 0
    # 0
    10
    0
    0
    5
    0
    0
    # 5
)

cd ~/ws/depth_mup/lm-evaluation-harness
source ~/ws/miniconda3/bin/activate
conda activate eeh_vllm

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

        lm_eval --model vllm \
        --model_args pretrained=${model},trust_remote_code=True,tensor_parallel_size=8,gpu_memory_utilization=0.98 \
        --tasks ${task} \
        --batch_size auto \
        --num_fewshot ${shot} \
        --output_path output/${model_name} \
        --device cuda:0,1,2,3,4,5,6,7 &> "$log_file"

    done
done
