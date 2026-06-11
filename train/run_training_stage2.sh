#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REC_SCRIPT="${SCRIPT_DIR}/scripts/run_training_rec.sh"
RA_SCRIPT="${SCRIPT_DIR}/scripts/run_training_RA.sh"
REC_RESULTS_DIR="${SCRIPT_DIR}/results/beauty_sid_rec"
REC_PROCESS_PATTERN="train_beauty_sid_rec.py"

# Ensure user-installed CLIs (e.g., deepspeed) are discoverable
export PATH="${HOME}/.local/bin:${PATH}"

# 默认按 8 卡串联 fast_rec 训练与 slow reasoning activation 训练；允许通过环境变量覆盖
export NUM_GPUS=${NUM_GPUS:-8}

if [[ ! -x "${REC_SCRIPT}" ]]; then
    echo "Error: ${REC_SCRIPT} not found or not executable." >&2
    exit 1
fi

if [[ ! -x "${RA_SCRIPT}" ]]; then
    echo "Error: ${RA_SCRIPT} not found or not executable." >&2
    exit 1
fi

echo "=== Stage 1: Starting fast_rec(k) training (run_training_rec.sh, NUM_GPUS=${NUM_GPUS}) ==="
bash "${REC_SCRIPT}"

echo "Waiting for recommendation training process to complete..."
sleep 10
while pgrep -f "${REC_PROCESS_PATTERN}" > /dev/null; do
    sleep 60
done
echo "fast_rec(k) training finished."

if [[ ! -d "${REC_RESULTS_DIR}" ]]; then
    echo "Error: results directory ${REC_RESULTS_DIR} not found." >&2
    exit 1
fi

# If another deepspeed job occupied default port, training may have failed early.
# Provide a more actionable message in that case.
if ! ls -d "${REC_RESULTS_DIR}"/checkpoint-* >/dev/null 2>&1; then
    echo "Error: no checkpoint directories found under ${REC_RESULTS_DIR}." >&2
    echo "Hint: If you saw EADDRINUSE on port 29500, set MASTER_PORT to a free port (e.g., export MASTER_PORT=29600) and rerun." >&2
    exit 1
fi

last_checkpoint=$(ls -d "${REC_RESULTS_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -z "${last_checkpoint}" ]]; then
    echo "Error: no checkpoint directories found under ${REC_RESULTS_DIR}." >&2
    exit 1
fi

echo "Identified final checkpoint for RA stage: ${last_checkpoint}"

echo "=== Stage 2: Starting think_and_rec(j) reasoning activation training (run_training_RA.sh, NUM_GPUS=${NUM_GPUS}) ==="
bash "${RA_SCRIPT}" "${last_checkpoint}"

echo "Rec+RA pipeline completed successfully. Train rank_candidates(m,n) and planner SFT/RL separately for the full paper pipeline."
