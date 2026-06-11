#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

MODEL_DIR=${REC_MODEL_DIR:-"../basemodel/merged_beauty_model_1-1"}
TRAIN_DATA="../data/training_prediction_sid_data_train.parquet"
VAL_DATA="../data/training_prediction_sid_data_val.parquet"

# Ensure user-installed CLIs (e.g., deepspeed) are discoverable
export PATH="${HOME}/.local/bin:${PATH}"

# Allow overrides via env vars
NUM_GPUS=${NUM_GPUS:-8}
REC_INCLUDE=${REC_INCLUDE:-""}
REC_EPOCHS=${REC_EPOCHS:-6}
REC_OUTPUT_DIR=${REC_OUTPUT_DIR:-"./results/beauty_sid_rec"}
REC_LOGGING_DIR=${REC_LOGGING_DIR:-"./logs/beauty_sid_rec"}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
REC_GRAD_ACCUM=${REC_GRAD_ACCUM:-1}
REC_MAX_LEN=${REC_MAX_LEN:-2048}

# Fast checkpointing / debug controls
REC_SAVE_STRATEGY=${REC_SAVE_STRATEGY:-"epoch"}  # epoch|steps
REC_SAVE_STEPS=${REC_SAVE_STEPS:-500}
REC_MAX_STEPS=${REC_MAX_STEPS:-0}                 # 0 means disabled

# Eval/load-best controls
REC_EVAL_STRATEGY=${REC_EVAL_STRATEGY:-"epoch"}   # epoch|steps|no
REC_EVAL_STEPS=${REC_EVAL_STEPS:-${REC_SAVE_STEPS}}
REC_LOAD_BEST_MODEL_AT_END=${REC_LOAD_BEST_MODEL_AT_END:-"auto"}  # auto|true|false

# If save by steps, make eval match by default; otherwise HF will error with load_best_model_at_end
if [[ "${REC_SAVE_STRATEGY}" == "steps" && "${REC_EVAL_STRATEGY}" == "epoch" ]]; then
    REC_EVAL_STRATEGY="steps"
    REC_EVAL_STEPS="${REC_SAVE_STEPS}"
fi

if [[ "${REC_LOAD_BEST_MODEL_AT_END}" == "auto" ]]; then
    if [[ "${REC_SAVE_STRATEGY}" != "${REC_EVAL_STRATEGY}" ]]; then
        REC_LOAD_BEST_MODEL_AT_END="false"
    else
        REC_LOAD_BEST_MODEL_AT_END="true"
    fi
fi

USE_LORA=false

ds_args=(--master_port ${MASTER_PORT:-29500})
if [[ -n "${REC_INCLUDE}" ]]; then
  ds_args+=(--include "${REC_INCLUDE}")
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  :
else
  ds_args+=(--num_gpus ${NUM_GPUS})
fi

DEEPSPEED_CMD=(
    deepspeed
    "${ds_args[@]}"
    ./scripts/train_beauty_sid_rec.py
    --model_name_or_path "${MODEL_DIR}"
    --train_data_path "${TRAIN_DATA}"
    --val_data_path "${VAL_DATA}"
    --max_length ${REC_MAX_LEN}
)

if [ "${USE_LORA}" = "true" ]; then
    DEEPSPEED_CMD+=(
        --use_lora True
        --lora_r 128
        --lora_alpha 128
        --lora_dropout 0.05
        --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
else
    DEEPSPEED_CMD+=(--use_lora False)
fi

DEEPSPEED_CMD+=(
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE}
    --gradient_accumulation_steps ${REC_GRAD_ACCUM}
    --num_train_epochs ${REC_EPOCHS}
    --gradient_checkpointing True
    --bf16 True
    --deepspeed ./scripts/ds_config_zero2.json
    --output_dir "${REC_OUTPUT_DIR}"
    --logging_dir "${REC_LOGGING_DIR}"
    --logging_steps 10
    --eval_strategy ${REC_EVAL_STRATEGY}
    --eval_steps ${REC_EVAL_STEPS}
    --eval_on_start False
    --save_strategy ${REC_SAVE_STRATEGY}
    --save_steps ${REC_SAVE_STEPS}
    --save_total_limit 10
    --metric_for_best_model eval_loss
    --greater_is_better False
    --load_best_model_at_end ${REC_LOAD_BEST_MODEL_AT_END}
    --optim adamw_torch
    --learning_rate 1e-5
    --warmup_ratio 0.1
    --weight_decay 0.01
    --adam_beta1 0.9
    --adam_beta2 0.999
    --adam_epsilon 1e-8
    --max_grad_norm 1.0
    --dataloader_num_workers 4
    --remove_unused_columns False
)

if [[ "${REC_MAX_STEPS}" != "0" ]]; then
    DEEPSPEED_CMD+=(--max_steps ${REC_MAX_STEPS})
fi

nohup "${DEEPSPEED_CMD[@]}" >> beauty_sid_rec.log 2>&1 &
