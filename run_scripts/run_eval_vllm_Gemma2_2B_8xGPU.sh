#!/bin/bash
# Add your venv setup
cd ~/ws/depth_mup/lm-evaluation-harness
source ~/ws/miniconda3/bin/activate
conda activate eeh_vllm

# Accept base_url and model_name as arguments
base_url=http://localhost:8193/v1
# tokenizer_name="Qwen/QwQ-32B"
tokenizer="google/gemma-2-2b"
model="google/gemma-2-2b"

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
    5
)

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

  # Run the command with the provided arguments
  OPENAI_API_KEY=serving-on-vllm \
  lm_eval \
  --model local-completions \
  --model_args model=${model},base_url=$base_url/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
  --tasks ${task} \
  --batch_size auto \
  --num_fewshot ${shot} \
  --output_path output/${model_name} \
  --log_samples &> "$log_file"  

done



