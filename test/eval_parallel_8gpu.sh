#!/usr/bin/env bash

# Usage: ./eval_parallel_8gpu.sh [EXTRA_ARGS]

echo "🚀 Starting 8-GPU Parallel Evaluation (non-CoT)..."

MERGED_MODEL_PATH="../train/results/beauty_sid_rec/checkpoint-8388"
ADDITIONAL_LORA_PATH=""
TEST_PARQUET="../data/training_prediction_sid_data_test.parquet"
GLOBAL_TRIE_FILE="./exact_trie.pkl"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_1010/parallel_eval_${TS}"
mkdir -p "$LOG_DIR"

echo "📝 Log directory: $LOG_DIR"
echo "⏰ Started at: $(date)"

TOTAL_SAMPLES=${TOTAL_SAMPLES:--1}          # -1 means use full parquet length
NUM_GPUS=${NUM_GPUS:-8}

# Note: To compute metrics up to @100, set NUM_BEAMS>=100 (can override via env/extra args)
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_BEAMS=${NUM_BEAMS:-10}
MAX_TOKENS=${MAX_TOKENS:-6}
THINK_TOKENS=${THINK_TOKENS:-0}
METRICS=${METRICS:-"hit@1,hit@5,hit@10,ndcg@5,ndcg@10"}
PRINT_GENERATIONS=${PRINT_GENERATIONS:-1}

if [ "$TOTAL_SAMPLES" -lt 0 ]; then
    TOTAL_SAMPLES=$(python3 - <<'PY'
import pandas as pd
print(len(pd.read_parquet('../data/training_prediction_sid_data_test.parquet')))
PY
)
fi

SAMPLES_PER_GPU=$(((TOTAL_SAMPLES + NUM_GPUS - 1) / NUM_GPUS))

echo "📋 Precomputing exact trie tree..."
if [ ! -f "$GLOBAL_TRIE_FILE" ]; then
    python3 precompute_global_trie.py \
        --test_parquet_file "$TEST_PARQUET" \
        --model_path "$MERGED_MODEL_PATH" \
        --output_file "$GLOBAL_TRIE_FILE"
    echo "✅ Exact trie precomputed: $GLOBAL_TRIE_FILE"
else
    echo "✅ Exact trie already exists: $GLOBAL_TRIE_FILE"
fi

echo "📊 8-GPU Parallel Configuration:"
echo "  Merged model: $MERGED_MODEL_PATH"
echo "  Additional LoRA: $ADDITIONAL_LORA_PATH"
echo "  Exact trie: $GLOBAL_TRIE_FILE"
echo "  Total samples: $TOTAL_SAMPLES"
echo "  GPUs: $NUM_GPUS"
echo "  Samples per GPU (ceil): $SAMPLES_PER_GPU"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Beam search: $NUM_BEAMS"
echo "  Metrics: $METRICS"
echo "  Max tokens: $MAX_TOKENS"
echo "  Think tokens: $THINK_TOKENS"
if [ "$PRINT_GENERATIONS" = "1" ]; then
  echo "  Print generations: ENABLED"
else
  echo "  Print generations: DISABLED"
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

pids=()
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    offset=$((gpu_id * SAMPLES_PER_GPU))
    log_file="${LOG_DIR}/gpu_${gpu_id}.log"
    
    echo "🔄 Starting GPU $gpu_id: samples $offset-$((offset + SAMPLES_PER_GPU - 1))"

    extra_print_args=()
    if [ "$PRINT_GENERATIONS" = "1" ]; then
        extra_print_args+=(--print_generations)
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 -u test_model_hitrate.py \
        --merged_model_path "${MERGED_MODEL_PATH}" \
        --additional_lora_path "${ADDITIONAL_LORA_PATH}" \
        --test_parquet_file "${TEST_PARQUET}" \
        --global_trie_file "${GLOBAL_TRIE_FILE}" \
        --test_batch_size ${BATCH_SIZE} \
        --num_beams ${NUM_BEAMS} \
        --metrics "${METRICS}" \
        --max_new_tokens ${MAX_TOKENS} \
        --temperature 0.6 \
        --top_p 1 \
        --think_max_tokens ${THINK_TOKENS} \
        "${extra_print_args[@]}" \
        --sample_num ${SAMPLES_PER_GPU} \
        --sample_offset ${offset} \
        --gpu_id ${gpu_id} \
        --log_file "$log_file" \
        "$@" > "$log_file" 2>&1 &

    pids+=($!)
    sleep 2
done

echo ""
echo "🔄 All ${NUM_GPUS} processes started:"
for i in $(seq 0 $((NUM_GPUS-1))); do
    echo "  GPU $i: PID ${pids[$i]} -> ${LOG_DIR}/gpu_${i}.log"
done

echo ""
echo "📋 Monitor commands:"
echo "  # View GPU status"
echo "  nvidia-smi"
echo "  # View specific GPU output (recommended)"
echo "  tail -f ${LOG_DIR}/gpu_0.log"
echo "  # View all GPU outputs"
echo "  tail -f ${LOG_DIR}/gpu_*.log"
echo "  # View process status"
echo "  ps aux | grep test_two_stage_model"

echo ""
echo "⏳ Waiting for all processes to complete..."
wait "${pids[@]}"

# Merge results and save to summary log
python3 -c "
import re
import os
from glob import glob

log_dir = '${LOG_DIR}'
metrics = '${METRICS}'.split(',')
total_metrics = {m: 0.0 for m in metrics}  # accumulate weighted sum (metric * samples)
total_samples = 0

summary_log = f'{log_dir}/summary_results.log'
with open(summary_log, 'w', encoding='utf-8') as f:
    f.write('🔍 8-GPU Parallel Evaluation Summary\\n')
    f.write('=' * 60 + '\\n')
    f.write(f'Timestamp: $(date)\\n')
    f.write(f'Log directory: {log_dir}\\n\\n')

    found_gpus = 0
    for gpu_id in range(int('${NUM_GPUS}')):
        log_file = f'{log_dir}/gpu_{gpu_id}.log'
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as gpu_f:
                content = gpu_f.read()

            final_results = re.search(r'🎯 Final Hit Rate Results:.*?={60}(.*?)={60}', content, re.DOTALL)
            if final_results:
                results_text = final_results.group(1)
                gpu_metrics = {}
                sample_match = re.search(r'Total samples: (\\d+)', content)
                if sample_match:
                    samples = int(sample_match.group(1))
                else:
                    # Fallback: if sample count missing, assume this shard processed SAMPLES_PER_GPU
                    samples = int('${SAMPLES_PER_GPU}')

                for metric in metrics:
                    match = re.search(rf'{metric}:\\s+([\\d\\.]+)', results_text)
                    if match:
                        value = float(match.group(1))
                        gpu_metrics[metric] = value
                        total_metrics[metric] += value * samples

                total_samples += samples
                found_gpus += 1
                f.write(f'✅ GPU {gpu_id}: {samples} samples\\n')
                for metric, value in gpu_metrics.items():
                    f.write(f'    {metric}: {value:.4f}\\n')
            else:
                f.write(f'❌ GPU {gpu_id}: No results found\\n')
        else:
            f.write(f'❌ GPU {gpu_id}: Log file not found\\n')

    f.write('\\n')
    if found_gpus > 0 and total_samples > 0:
        avg_metrics = {m: total_metrics[m] / total_samples for m in metrics}

        f.write('🎯 FINAL AVERAGED RESULTS:\\n')
        f.write('=' * 60 + '\\n')
        for metric, value in avg_metrics.items():
            f.write(f'{metric:>10}: {value:.4f}\\n')
        f.write('=' * 60 + '\\n')
        f.write(f'Total samples: {total_samples}\\n')
        f.write(f'Completed GPUs: {found_gpus}/${NUM_GPUS}\\n')
        f.write('✅ Evaluation completed successfully!\\n')

    else:
        f.write('❌ No valid results found!\\n')

print(f'Summary saved to: {summary_log}')
" >> "${LOG_DIR}/summary_results.log"
