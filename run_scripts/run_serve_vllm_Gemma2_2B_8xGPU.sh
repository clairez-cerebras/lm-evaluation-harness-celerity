#!/bin/bash

cd ~/ws/depth_mup/lm-evaluation-harness
source ~/ws/miniconda3/bin/activate
conda activate eeh_vllm
huggingface-cli login --token XXX

MODEL_PATH="google/gemma-2-2b"
MODEL_NAME="google/gemma-2-2b"

VLLM_PORT_NUMBER=8193

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve \
  "$MODEL_PATH" \
  --port "$VLLM_PORT_NUMBER" \
  --served-model-name "$MODEL_NAME" \
  --tensor_parallel_size=8 \
  --gpu-memory-utilization 0.98 \
  --enable-chunked-prefill \
  --enforce-eager \
  --max-num-seqs 1 \
  --max-model-len 8192


