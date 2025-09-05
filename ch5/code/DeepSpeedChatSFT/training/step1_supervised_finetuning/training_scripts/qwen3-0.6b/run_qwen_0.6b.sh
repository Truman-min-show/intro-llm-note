#!/bin/bash
# run_qwen_0.6b.sh - single GPU run using main.py that passes ds_config to deepspeed.initialize()
# Note: This script uses --deepspeed (flag) and DOES NOT pass --deepspeed_config to the launcher.
# Usage: bash run_qwen_0.6b.sh [OUTPUT_DIR] [MODEL_DIR] [DATA_PATH] [DS_CACHE_DIR]

OUTPUT_DIR=${1:-./outputs/qwen0.6b_step1}
MODEL_DIR=${2:-./models/Qwen3-0.6B}
DATA_PATH=${3:-./data/MyDataset/CVPR2024Summary.json}
DATA_CACHE_DIR=${4:-./data/cache}

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATA_CACHE_DIR}"

export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29501

deepspeed --master_port=${MASTER_PORT} main.py \
  --model_name_or_path "${MODEL_DIR}" \
  --data_path "${DATA_PATH}" \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 10 \
  --learning_rate 5e-6\
  --max_seq_len 512 \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42 \
  --dtype bf16 \
  --gradient_checkpointing \
  --deepspeed \
  --data_output_path "${DATA_CACHE_DIR}" \
  &> "${OUTPUT_DIR}/train.log"
