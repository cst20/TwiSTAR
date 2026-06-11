# TwiSTAR

**TwiSTAR** (Think Fast, Think Slow, Then Act) is a generative recommendation framework with adaptive reasoning. It combines:

- a fast Semantic-ID (SID) generation model for low-latency retrieval;
- a ranking tool for candidate refinement;
- a slow reasoning model for difficult user histories;
- an agent/planner that decides which tool to invoke for each request.

This repository contains the code used to prepare Amazon-style sequential recommendation data, train SID-based LLM recommenders, train ranking/reasoning tools, and evaluate adaptive routing.

> Note: this repository intentionally excludes datasets, model checkpoints, logs, and other large artifacts. They should be generated or downloaded locally.

## Repository Structure

```text
TwiStar/
├── run_twistar.py                 # Lightweight, non-LLM TwiSTAR reproduction on Amazon Beauty
├── run_twistar_llm_1_7b.py         # Qwen3-1.7B rank_candidates(m,n) ablation
├── prepare_amazon_data.py          # Generates TwiSTAR training/evaluation corpora
├── agent_workflow.py               # TwiSTAR tools: fast_rec / rank_candidates / think_and_rec
├── run_full_stage2.sh              # End-to-end training pipeline: alignment -> rec -> reasoning
├── setup_conda_env.sh              # Environment setup helper
├── basemodel/                      # Base model download, SID vocabulary expansion, model merge
├── data/                           # Data conversion and I2I/ranking data generation scripts
├── train/                          # Training launchers and model training scripts
├── test/                           # Inference and evaluation scripts
└── scripts/                        # Extra helper scripts
```

`prepare_amazon_data.py` is a data-preparation entry point. Given:

1. a sequential interaction file, e.g. `user_id item_1 item_2 ...`, and
2. item metadata with titles/categories and optionally precomputed SIDs,

it generates the standard corpora required by the TwiSTAR pipeline:

- `training_align_data_{train,val,test}.parquet` for SID-text alignment;
- `training_prediction_sid_data_{train,val,test}.parquet` for fast SID generation;
- `training_RA_{train,val,test}.parquet` for reasoning activation;
- `i2i_swing_topK.jsonl` and `i2i_explain_prompts.jsonl` for collaborative commonsense injection;
- `ranking_recall_data_{train,val,test}.parquet` for ranking-tool training;
- `planner_sft_train.jsonl` for planner supervised warm-up;
- `sid2text.json` and `twistar_pipeline_manifest.json` that records generated artifacts and stage metadata.

If item metadata already contains a `sid` field, the script reuses it. Otherwise, it creates deterministic hash-based pseudo SIDs so the full pipeline can be smoke-tested before replacing them with RQ-VAE/residual-k-means SIDs.

## Agent Workflow and Paper Alignment

`agent_workflow.py` exposes the three tool calls described in the TwiSTAR paper:

1. `fast_rec(k)` retrieves top-`k` SIDs with the fast non-reasoning model.
2. `rank_candidates(m, n)` reranks `m` fast candidates and returns top-`n` items.
3. `think_and_rec(j)` invokes the slow reasoning model and directly recommends top-`j` SIDs.

The planner/controller path is represented as data + executor: `planner_sft_train.jsonl` provides supervised warm-up labels, and `agent_workflow.py` executes a trained planner's JSON tool calls at inference time. The paper's second stage is agentic RL (GRPO/PPO) with recommendation reward plus latency/tool-validity penalties; if no trained planner checkpoint is supplied, `--planner_policy fast_rank|fast_only|slow` is only a deterministic ablation, not the final paper planner.

## Installation

Python 3.10+ is recommended. Install the core dependencies:

```bash
pip install torch transformers datasets pandas pyarrow tqdm numpy scikit-learn
pip install accelerate peft deepspeed
```

Optional dependencies are needed for specific tools:

```bash
pip install vllm trl wandb
```

Alternatively, adapt and run:

```bash
bash setup_conda_env.sh
```

## Data Format

### Sequential interactions

The sequence file should be whitespace-separated:

```text
user_1 item_a item_b item_c item_d
user_2 item_e item_f item_g
```

### Item metadata

The item metadata can be JSON/JSONL/JSON.GZ. Each item should provide at least an item id and preferably title/category fields:

```json
{
  "B000000001": {
    "title": "Example item title",
    "categories": "Beauty > Hair Care",
    "sid": "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"
  }
}
```

## Quick Start

### 1. Prepare data

Place your data under `data/`, for example:

```text
data/sequential_data_processed.txt
data/Beauty.pretrain.json
```

Then run:

```bash
python prepare_amazon_data.py \
  --dataset Beauty \
  --seq data/sequential_data_processed.txt \
  --items data/Beauty.pretrain.json \
  --out_dir data/beauty_processed
```

For the existing training scripts, either copy/symlink the generated files back into `data/`, or pass explicit paths where scripts support them:

```bash
cp data/beauty_processed/training_* data/
cp data/beauty_processed/sid2text.json data/
cp data/beauty_processed/i2i_swing_top*.jsonl data/
```

### 2. Download and expand a base model

The default model is `Qwen/Qwen3-1.7B`:

```bash
cd basemodel
python download_basemodel.py --repo_id Qwen/Qwen3-1.7B --target_subdir Qwen3-1-7B
python expand_vocab.py \
  --base_model_dir ./Qwen3-1-7B \
  --save_dir ./Qwen3-1-7B-expand
cd ..
```

### 3. Run the rec + reasoning training pipeline

```bash
NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash run_full_stage2.sh
```

The script performs the model-training stages that are available in this repo:

1. SID/text alignment;
2. LoRA merge into the expanded base model;
3. fast SID recommendation training;
4. reasoning activation training.

The complete paper pipeline additionally trains the `rank_candidates(m,n)` tool and performs planner supervised warm-up + planner RL. The data artifacts for those stages are produced by `prepare_amazon_data.py`; `agent_workflow.py` is the inference executor for the resulting planner tool calls.

Useful overrides:

```bash
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=29617
export ALIGN_EPOCHS=3
export ALIGN_BATCH_SIZE=16
export ALIGN_MAX_LEN=1024
export REC_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export REC_MAX_LEN=1024
export RA_TOTAL_EPOCHS=1
export RA_PER_DEVICE_TRAIN_BATCH_SIZE=8
bash run_full_stage2.sh
```

### 4. Lightweight reproduction

If you only want to check the TwiSTAR routing idea without full LLM training:

```bash
python run_twistar.py \
  --seq data/sequential_data_processed.txt \
  --items data/Beauty.pretrain.json \
  --out_dir outputs
```

### 5. Qwen3-1.7B `rank_candidates(m,n)` ablation

This script evaluates the ranking tool path only (`fast_rec(k) -> rank_candidates(m,n)`). It does not run the full TwiSTAR planner or `think_and_rec(j)`.

```bash
python run_twistar_llm_1_7b.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --seq data/sequential_data_processed.txt \
  --items data/Beauty.pretrain.json \
  --sample_num 200 \
  --out_dir outputs_llm_1_7b
```

## Evaluation

Fast recommendation evaluation examples are in `test/`:

```bash
cd test
bash eval_parallel_8gpu.sh
bash eval_parallel_8gpu_cot.sh
```

Agent/routing analysis:

```bash
python test/analyze_agent_routing.py \
  --model_path train/results/beauty_sid_rec/checkpoint-XXXX \
  --val_parquet data/training_prediction_sid_data_val.parquet
```

NDCG evaluation for tool combinations:

```bash
python test/eval_agent_ndcg10.py \
  --mode fast_din \
  --onerec_model_path train/results/beauty_sid_rec/checkpoint-XXXX
```

## Notes for GitHub Release

Do **not** commit generated artifacts or large files, including:

- `basemodel/Qwen*/`
- `train/results/`
- `train/logs/`
- `wandb/`
- `outputs*/`
- `*.safetensors`, `*.bin`, `*.pt`, `*.pth`
- `*.parquet`, `*.jsonl`, `*.pkl`, `*.gz`, `*.log`

The current code directory is designed to be uploaded without those artifacts.

## Citation

If you use this code, please cite the TwiSTAR paper once the final citation information is available.

```bibtex
@misc{cao2026twistarthinkfastthinkslow,
      title={TwiSTAR:Think Fast, Think Slow, Then Act,Generative Recommendation with Adaptive Reasoning}, 
      author={Shiteng Cao and Kaian Jiang and Yunlong Gong and Zhiheng Li},
      year={2026},
      eprint={2605.11553},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2605.11553}, 
}
```
