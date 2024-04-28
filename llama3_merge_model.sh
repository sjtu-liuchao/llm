#!/bin/bash

python src/export_model.py \
  --model_name_or_path /mnt/workspace/.cache/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct \
  --template llama3 \
  --finetuning_type lora \
  --export_dir  /mnt/workspace/.cache/modelscope/llama3_lora \
  --export_legacy_format false