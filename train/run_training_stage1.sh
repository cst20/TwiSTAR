#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_DIR=${ALIGN_MODEL_DIR:-"../basemodel/Qwen3-1-7B-expand"}
TRAIN_DATA=${ALIGN_TRAIN_DATA:-"../data/training_align_data_train.parquet"}
VAL_DATA=${ALIGN_VAL_DATA:-"../data/training_align_data_val.parquet"}

# Ensure user-installed CLIs (e.g., deepspeed) are discoverable
export PATH="${HOME}/.local/bin:${PATH}"

# Allow overrides via env vars
NUM_GPUS=${NUM_GPUS:-8}
# Optional: pin GPUs via deepspeed include string, e.g. "localhost:4,5,6,7"
ALIGN_INCLUDE=${ALIGN_INCLUDE:-""}
ALIGN_EPOCHS=${ALIGN_EPOCHS:-15}
ALIGN_OUTPUT_DIR=${ALIGN_OUTPUT_DIR:-"./results/beauty_align"}
ALIGN_LOGGING_DIR=${ALIGN_LOGGING_DIR:-"./logs/beauty_sid_align"}
ALIGN_MAX_LEN=${ALIGN_MAX_LEN:-2048}
ALIGN_BATCH_SIZE=${ALIGN_BATCH_SIZE:-8}
ALIGN_GRAD_ACCUM=${ALIGN_GRAD_ACCUM:-1}

# 避免显存碎片导致的极小分配失败（如 cuda calloc async 136 bytes）
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
# 单机 8 卡训练不依赖 SSH/hostfile
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

ds_args=(--master_port ${MASTER_PORT:-29500})
if [[ -n "${ALIGN_INCLUDE}" ]]; then
  # If user specifies --include, do NOT pass --num_gpus (deepspeed will select devices from include list)
  ds_args+=(--include "${ALIGN_INCLUDE}")
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Respect CUDA_VISIBLE_DEVICES without passing --num_gpus (otherwise deepspeed ignores it)
  :
else
  ds_args+=(--num_gpus ${NUM_GPUS})
fi

nohup deepspeed \
    "${ds_args[@]}" ./scripts/train_beauty_align.py \
    --model_dir "${MODEL_DIR}" \
    --train_data_path "${TRAIN_DATA}" \
    --val_data_path "${VAL_DATA}" \
    --max_length ${ALIGN_MAX_LEN} \
    --per_device_train_batch_size ${ALIGN_BATCH_SIZE} \
    --gradient_accumulation_steps ${ALIGN_GRAD_ACCUM} \
    --num_train_epochs ${ALIGN_EPOCHS} \
    --gradient_checkpointing True \
    --bf16 True \
    --deepspeed ./scripts/ds_config_zero2.json \
    --output_dir "${ALIGN_OUTPUT_DIR}" \
    --logging_dir "${ALIGN_LOGGING_DIR}" \
    --logging_steps 10 \
    --eval_strategy epoch \
    --eval_on_start False \
    --save_strategy epoch \
    --save_total_limit 15 \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --load_best_model_at_end True \
    --optim adamw_torch \
    --learning_rate 1e-4 \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 4 \
    --remove_unused_columns False >> beauty_align.log 2>&1 &
