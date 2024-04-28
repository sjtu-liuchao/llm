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
  
  
  
  
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: true
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false