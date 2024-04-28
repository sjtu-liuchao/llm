#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
  --stage sft \
  --do_train True \
  --model_name_or_path /mnt/workspace/.cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct \
  --dataset alpaca_zh \
  --template llama3 \
  --lora_target q_proj,v_proj \
  --output_dir single_lora_llama3_checkpoint \
  --overwrite_cache \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --save_steps 100 \
  --learning_rate 5e-5 \
  --num_train_epochs 1.0 \
  --finetuning_type lora \
  --fp16 \
  --lora_rank 8