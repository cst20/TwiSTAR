#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${TWISTAR_ROOT_DIR:-${SCRIPT_DIR}}"
LOGFILE="${ROOT_DIR}/stage2_pipeline.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "==============================================="
echo "   STARTING TWISTAR REC+RA TRAINING PIPELINE (S1+S2)"
echo "==============================================="
echo "Time: $(date)"
echo ""

cd "${ROOT_DIR}"

# 默认按 8 卡跑 alignment / fast_rec / slow reasoning activation；允许通过环境变量覆盖
export NUM_GPUS=${NUM_GPUS:-8}
echo "Using NUM_GPUS=${NUM_GPUS} for training stages."

wait_for_process_exit() {
  local process_pattern="$1"
  local sleep_seconds="${2:-60}"
  while pgrep -f "${process_pattern}" > /dev/null; do
    sleep "${sleep_seconds}"
  done
}

resolve_latest_checkpoint() {
  local checkpoint_glob="$1"
  ls -d ${checkpoint_glob} 2>/dev/null | sort -V | tail -n 1 || true
}

########################################
# Stage 0: 基础检查与数据准备
########################################

echo "[Stage 0] Checking base model & alignment data..."

BASEMODEL_DIR="${ROOT_DIR}/basemodel/Qwen3-1-7B"
EXPANDED_MODEL_DIR="${ROOT_DIR}/basemodel/Qwen3-1-7B-expand"

if [[ ! -d "${EXPANDED_MODEL_DIR}" ]]; then
  echo "  - Expanded model not found. Running basemodel download + vocab expand..."
  cd "${ROOT_DIR}/basemodel"
  if [[ ! -d "${BASEMODEL_DIR}" ]]; then
    echo "    > Downloading base model..."
    python3 download_basemodel.py
  fi
  echo "    > Expanding vocabulary..."
  python3 expand_vocab.py
  cd "${ROOT_DIR}"
else
  echo "  - Found expanded model at: ${EXPANDED_MODEL_DIR}"
fi

ALIGN_TRAIN_DATA="${ROOT_DIR}/data/training_align_data_train.parquet"
ALIGN_VAL_DATA="${ROOT_DIR}/data/training_align_data_val.parquet"

if [[ ! -f "${ALIGN_TRAIN_DATA}" || ! -f "${ALIGN_VAL_DATA}" ]]; then
  echo "  - Alignment parquet not found. Generating training data..."
  cd "${ROOT_DIR}/data"
  python3 generate_training_data.py
  cd "${ROOT_DIR}"
else
  echo "  - Found alignment data at: ${ALIGN_TRAIN_DATA} / ${ALIGN_VAL_DATA}"
fi

########################################
# Stage 1: Itemic Alignment 训练
########################################

echo ""
echo "[Stage 1] Running alignment training (run_training_stage1.sh)..."

ALIGN_RESULTS_DIR="${ROOT_DIR}/train/results/beauty_align"

cd "${ROOT_DIR}/train"
chmod +x ./run_training_stage1.sh

if [[ -d "${ALIGN_RESULTS_DIR}" ]]; then
  latest_align_checkpoint=$(resolve_latest_checkpoint "${ALIGN_RESULTS_DIR}/checkpoint-*")
  if [[ -n "${latest_align_checkpoint}" ]]; then
    echo "  - Found alignment checkpoint at ${latest_align_checkpoint}, skipping Stage1 training."
  else
    echo "  - Alignment directory exists but no checkpoint found. Restarting Stage1 training..."
    bash ./run_training_stage1.sh
    echo "  - Waiting for alignment training to complete..."
    sleep 10
    wait_for_process_exit "train_beauty_align.py"
  fi
else
  echo "  - No existing alignment results, starting Stage1 training..."
  bash ./run_training_stage1.sh
  echo "  - Waiting for alignment training to complete..."
  sleep 10
  wait_for_process_exit "train_beauty_align.py"
fi

latest_align_checkpoint=$(resolve_latest_checkpoint "${ALIGN_RESULTS_DIR}/checkpoint-*")
if [[ -z "${latest_align_checkpoint}" ]]; then
  echo "Error: no alignment checkpoint found under ${ALIGN_RESULTS_DIR}." >&2
  exit 1
fi
echo "  - Using alignment checkpoint: ${latest_align_checkpoint}"

########################################
# Stage 1.5: 合并 LoRA 权重到扩展基座
########################################

echo ""
echo "[Stage 1.5] Merging best LoRA checkpoint into expanded base model..."

MERGED_MODEL_DIR="${ROOT_DIR}/basemodel/merged_beauty_model_1-1"

cd "${ROOT_DIR}/basemodel"
echo "  - Refreshing merged model from alignment checkpoint..."
LORA_MODEL_PATH="${latest_align_checkpoint}" MERGED_OUTPUT_PATH="${MERGED_MODEL_DIR}" python3 merge_model.py
cd "${ROOT_DIR}"

########################################
# Stage 1.8: 生成 Stage2 所需训练数据
########################################

echo ""
echo "[Stage 1.8] Preparing recommendation & RA training data..."

PRED_TRAIN_DATA="${ROOT_DIR}/data/training_prediction_sid_data_train.parquet"
RA_TRAIN_DATA="${ROOT_DIR}/data/training_RA_train.parquet"

cd "${ROOT_DIR}/data"
if [[ ! -f "${PRED_TRAIN_DATA}" ]]; then
  echo "  - Generating SID prediction data..."
  python3 generate_sid_prediction_data.py
else
  echo "  - Found SID prediction data at: ${PRED_TRAIN_DATA}"
fi

if [[ ! -f "${RA_TRAIN_DATA}" ]]; then
  echo "  - Generating RA training data..."
  python3 generate_RA_data.py
else
  echo "  - Found RA training data at: ${RA_TRAIN_DATA}"
fi

cd "${ROOT_DIR}"

########################################
# Stage 2: Recommendation + CoT Reasoning Activation
########################################

echo ""
echo "[Stage 2] Executing Recommendation + Slow Reasoning Activation training..."

cd "${ROOT_DIR}/train"

# Ensure scripts are executable!
chmod +x ./scripts/run_training_rec.sh
chmod +x ./scripts/run_training_RA.sh
chmod +x ./run_training_stage2.sh

bash ./run_training_stage2.sh

echo ""
echo "========================================="
echo "   REC+RA PIPELINE (STAGE1 + STAGE2) DONE"
echo "   Next paper stages: train rank_candidates(m,n), then planner SFT + planner RL."
echo "========================================="
