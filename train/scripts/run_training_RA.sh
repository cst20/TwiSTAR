#!/bin/bash
# Allow overriding CONFIG_NAME to avoid overwriting previous runs.
export CONFIG_NAME=${CONFIG_NAME:-ReasoningActivation}
export TOKENIZERS_PARALLELISM=false

# Ensure user-installed CLIs (e.g., deepspeed) are discoverable
export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root (LLRM_eval)
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}/.."

NUM_TOTAL_EPOCHS=${RA_TOTAL_EPOCHS:-2}
INITIAL_MODEL_PATH=$1
INITIAL_DATA_PATH='../data/training_RA_train.parquet'
RA_PER_DEVICE_TRAIN_BATCH_SIZE=${RA_PER_DEVICE_TRAIN_BATCH_SIZE:-2}

# If you want GRPO (instead of SFT), set:
#   export RA_MODE=grpo
#   export NUM_GPUS=4
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
# and pass an initial model path (can be a stage1-aligned adapter checkpoint).
RA_MODE=${RA_MODE:-"sft"}

OUTPUT_DIR_BASE="./results/${CONFIG_NAME}"

 # Make model path absolute for torchrun/DDP.
 if [[ -n "${INITIAL_MODEL_PATH}" && "${INITIAL_MODEL_PATH}" != /* ]]; then
   # Caller might pass a path relative to repo root.
   CURRENT_MODEL_PATH="$(cd "${ROOT_DIR}" && realpath "${INITIAL_MODEL_PATH}")"
 else
   CURRENT_MODEL_PATH="${INITIAL_MODEL_PATH}"
 fi

 if [[ -z "${CURRENT_MODEL_PATH}" ]]; then
   echo "[error] empty CURRENT_MODEL_PATH; pass init model path as $1" >&2
   exit 2
 fi
 if [[ ! -e "${CURRENT_MODEL_PATH}" ]]; then
   echo "[error] CURRENT_MODEL_PATH not found: ${CURRENT_MODEL_PATH}" >&2
   exit 2
 fi
CURRENT_DATA_PATH=${INITIAL_DATA_PATH}

# 默认按 8 卡训练；允许通过环境变量 NUM_GPUS 覆盖
export NUM_GPUS=${NUM_GPUS:-8}

# Fast checkpointing / debug controls
RA_SAVE_STRATEGY=${RA_SAVE_STRATEGY:-"steps"}  # no|steps|epoch
RA_SAVE_STEPS=${RA_SAVE_STEPS:-200}
RA_MAX_STEPS=${RA_MAX_STEPS:-0}                 # 0 means disabled


# --- Training Loop ---
for (( i=1; i<=${NUM_TOTAL_EPOCHS}; i++ ))
do
    mkdir -p ${OUTPUT_DIR_BASE}/epoch_${i}
    LOG_FILE="${OUTPUT_DIR_BASE}/epoch_${i}/logs.log"
    echo "=================================" >> ${LOG_FILE}
    echo "          EPOCH ${i} / ${NUM_TOTAL_EPOCHS}" >> ${LOG_FILE}
    echo "=================================" >> ${LOG_FILE}

    echo "[$(date)] Starting training for epoch ${i}..." >> ${LOG_FILE}
    echo "--> Model Path: ${CURRENT_MODEL_PATH}" >> ${LOG_FILE}
    echo "--> Data Path: ${CURRENT_DATA_PATH}" >> ${LOG_FILE}

    if [[ "${RA_MODE}" == "grpo" ]]; then
      # Multi-GPU GRPO via torchrun (uses GPUs exposed in CUDA_VISIBLE_DEVICES)
      # Example:
      #   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 RA_MODE=grpo bash ./scripts/run_training_RA.sh <init_model>
      # Use full dataset:
      #   GRPO_NUM_SAMPLES=0 (<=0 means full parquet)
      TRAIN_CMD=(
        torchrun
        --nproc_per_node "${NUM_GPUS}"
        --master_port "${MASTER_PORT:-29500}"
        ./scripts/train_beauty_RA_grpo.py
        --mode train
        --model_name_or_path "${CURRENT_MODEL_PATH}"
        --data_path "${CURRENT_DATA_PATH}"
        --num_samples "${GRPO_NUM_SAMPLES:-256}"
        --num_generations "${GRPO_NUM_GENERATIONS:-32}"
        --steps "${GRPO_STEPS:-50}"
        --train_batch_rollouts "${GRPO_TRAIN_BATCH_ROLLOUTS:-8}"
        --lr "${GRPO_LR:-2e-5}"
        --beta_kl "${GRPO_BETA_KL:-0.01}"
        --save_every "${GRPO_SAVE_EVERY:-25}"
        --max_new_tokens "${GRPO_MAX_NEW_TOKENS:-128}"
        --max_prompt_tokens "${GRPO_MAX_PROMPT_TOKENS:-1024}"
        --max_seq_len "${GRPO_MAX_SEQ_LEN:-2048}"
        --temperature "${GRPO_TEMPERATURE:-0.9}"
        --top_p "${GRPO_TOP_P:-0.95}"
        --rollout_batch_size "${GRPO_ROLLOUT_BATCH_SIZE:-4}"
        --logp_batch_size "${GRPO_LOGP_BATCH_SIZE:-1}"
        --output_dir "${OUTPUT_DIR_BASE}/epoch_${i}"
      )
    else
      DS_CMD=(
          deepspeed
          --num_gpus "${NUM_GPUS}"
          --master_port "${MASTER_PORT:-29500}"
          ./scripts/train_beauty_RA.py
          --model_name_or_path "${CURRENT_MODEL_PATH}"
          --use_lora False
          --per_device_train_batch_size ${RA_PER_DEVICE_TRAIN_BATCH_SIZE}
          --num_train_epochs 1
          --gradient_checkpointing True
          --bf16 True
          --deepspeed ./scripts/ds_config_zero2.json
          --output_dir "${OUTPUT_DIR_BASE}/epoch_${i}"
          --logging_dir "${OUTPUT_DIR_BASE}/epoch_${i}"
          --data_path "${CURRENT_DATA_PATH}"
          --logging_steps 1
          --eval_strategy "no"
          --eval_on_start False
          --save_strategy "${RA_SAVE_STRATEGY}"
          --save_total_limit 2
          --load_best_model_at_end False
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
    fi

    if [[ "${RA_SAVE_STRATEGY}" == "steps" ]]; then
      DS_CMD+=(--save_steps "${RA_SAVE_STEPS}")
    fi

    if [[ "${RA_MAX_STEPS}" != "0" ]]; then
      DS_CMD+=(--max_steps "${RA_MAX_STEPS}")
    fi

    if [[ "${RA_MODE}" == "grpo" ]]; then
      nohup "${TRAIN_CMD[@]}" >> ${LOG_FILE} 2>&1 &
    else
      nohup "${DS_CMD[@]}" >> ${LOG_FILE} 2>&1 &
    fi

    TRAIN_PID=$!
    wait $TRAIN_PID

    # Check if training was successful
    if [ $? -ne 0 ]; then
        echo "[$(date)] Training for epoch ${i} FAILED. Check log for details." >> ${LOG_FILE}
        exit 1
    fi

    echo "[$(date)] Training for epoch ${i} finished successfully." >> ${LOG_FILE}

    CURRENT_MODEL_PATH=${OUTPUT_DIR_BASE}/epoch_${i}

    # --- Data Reconstruction Step ---
    if [ "$i" -eq ${NUM_TOTAL_EPOCHS} ]; then
        break
    fi

    python3 -u ./scripts/reconstruct_data_parallel.py \
        ${CURRENT_MODEL_PATH} \
        ${INITIAL_DATA_PATH} \
        ${CONFIG_NAME} \
        ${i} \
        ${NUM_GPUS} \
        2 >> ${LOG_FILE} 2>&1

    if [ $? -ne 0 ]; then
        echo "[$(date)] Data reconstruction FAILED." >> ${LOG_FILE}
        exit 1
    fi
    echo "[$(date)] Parallel data reconstruction with model ${CURRENT_MODEL_PATH} complete." >> ${LOG_FILE}
    CURRENT_DATA_PATH=${OUTPUT_DIR_BASE}/epoch_${i}/reconstructed_data.parquet
    echo "CURRENT_DATA_PATH: ${CURRENT_DATA_PATH}" >> ${LOG_FILE}

    echo "[$(date)] Data reconstruction complete." >> ${LOG_FILE}
    echo "---------------------------------\n" >> ${LOG_FILE}
done

echo "[$(date)] All ${NUM_TOTAL_EPOCHS} epochs completed." >> ${LOG_FILE}
echo "Final model is available at: ${CURRENT_MODEL_PATH}" >> ${LOG_FILE}
