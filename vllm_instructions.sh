#!/bin/bash
# Add your venv setup
source /mnt/local/shared/michaelw/venvs/lmeval_venv/bin/activate

# Accept base_url and model_name as arguments
base_url=http://localhost:8190/v1
# tokenizer_name="Qwen/QwQ-32B"
tokenizer_name="meta-llama/Llama-3.3-70B-Instruct"
model_name="magpieskyt1backdsv2ceponopair_llama3p3_70b_ft"
# model_name="Qwen/QwQ-32B"
cache_dir="./results/mmlu_pro/cache/"


# Prompt the user to clear the cache and restart
read -p "Do you want to clear the cache and restart? (yes/no): " user_input

# Check if the user entered 'yes' or 'no'
if [[ "$user_input" == "yes" ]]; then
    echo "Clearing the cache..."
    rm -rf $cache_dir
elif [[ "$user_input" == "no" ]]; then
    echo "Proceeding without clearing the cache."
else
    # If the input is neither 'yes' nor 'no', exit the script
    echo "Invalid input. Please enter 'yes' or 'no'."
    exit 1
fi


# Run the command with the provided arguments
OPENAI_API_KEY=serving-on-vllm \
lm_eval \
--model local-completions \
--tasks mmlu_pro_math \
--model_args model=$model_name,tokenizer=$tokenizer_name,base_url=$base_url/chat/completions,num_concurrent=25,max_retries=3,tokenized_requests=False,tokenizer_backend=huggingface \
--output_path ./results/mmlu_pro/ \
--use_cache $cache_dir \
--apply_chat_template \
--log_samples

rm -rf $cache_dir
base_url=http://localhost:8290/v1
OPENAI_API_KEY=serving-on-vllm \
lm_eval \
--model local-completions \
--tasks mmlu_pro_math \
--model_args model=$model_name,tokenizer=$tokenizer_name,base_url=$base_url/chat/completions,num_concurrent=25,max_retries=3,tokenized_requests=False,tokenizer_backend=huggingface \
--output_path ./results/mmlu_pro/ \
--use_cache $cache_dir \
--apply_chat_template \
--log_samples





======

# Setup
source /mnt/local/shared/michaelw/venvs/miniconda3/etc/profile.d/conda.sh
conda activate vllm_venv
huggingface-cli login --token XXX
# MODEL_NAME="merged_ties_instruct_v1_llama-tok" # verifier_sft_0409
# MODEL_PATH="/mnt/local/shared/michaelw/models/${MODEL_NAME}"

# MODEL_NAME="agentica-org/DeepScaleR-1.5B-Preview"
# MODEL_PATH="agentica-org/DeepScaleR-1.5B-Preview"

# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_PATH="meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"

# MODEL_PATH="Qwen/Qwen3-8B"
# MODEL_NAME="Qwen/Qwen3-8B"

VLLM_PORT_NUMBER=8190



CUDA_VISIBLE_DEVICES=0,1,6,7 vllm serve \
  "$MODEL_PATH" \
  --port "$VLLM_PORT_NUMBER" \
  --served-model-name "$MODEL_NAME" \
  --tensor_parallel_size=4 \



# CUDA_VISIBLE_DEVICES=6,7 vllm serve \
#   "$MODEL_PATH" \
#   --dtype bfloat16 \
#   --api-key serving-on-vllm \
#   --tensor_parallel_size=2 \
#   --port $VLLM_PORT_NUMBER \
#   --enforce_eager \
#   --max_num_seqs=256 \
#   --gpu-memory-utilization 0.98 \
#   --enable-chunked-prefill \
#   --served-model-name "$MODEL_NAME" \
#   --max-model-len 24576 \
