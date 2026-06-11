import argparse
import json
import os
import torch
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import datetime
import re

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
    )
    return logging.getLogger("vllm_eval")

def extract_sid_from_text(text):
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    matches = re.findall(sid_pattern, text)
    if matches:
        return matches[-1]
    return "None"

def get_topk_results(cands, cand_scores, target, topk=10):
    ranked_results = []
    seen = set()
    for c, score in zip(cands, cand_scores):
        sid = extract_sid_from_text(c.split("</think>")[-1])
        if sid != "None" and sid not in seen:
            seen.add(sid)
            ranked_results.append(sid)
        if len(ranked_results) >= topk:
            break
            
    is_hit = False
    hit_rank = -1
    for idx, sid in enumerate(ranked_results):
        if sid == target:
            is_hit = True
            hit_rank = idx
            break
            
    return ranked_results, is_hit, hit_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--log_file", default="vllm_cot.log")
    parser.add_argument("--num_thinking_samples", type=int, default=5)
    parser.add_argument("--test_batch_size", type=int, default=100)
    args = parser.parse_args()
    
    logger = setup_logging(args.log_file)
    logger.info("Starting vLLM CoT Evaluation")
    
    # Load dataset
    df = pd.read_parquet(args.data_path)
    # Subset for faster evaluation just to get numbers quickly
    df = df.head(1000)
    logger.info(f"Loaded {len(df)} samples")
    
    # Prepare prompts and targets
    prompts = [desc + "<|im_end|>\n<|im_start|>assistant\n<think>\n" for desc in df["description"].tolist()]
    targets = [extract_sid_from_text(ans) for ans in df["groundtruth"].tolist()]
    
    # Initialize vLLM
    logger.info(f"Loading vLLM model from {args.model_path}")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9
    )
    
    # Stage 1: Generate Thinking
    logger.info("Stage 1: Generating CoT Thinking")
    think_sampling_params = SamplingParams(
        n=args.num_thinking_samples,
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
        stop=["</think>"]
    )
    
    think_outputs = llm.generate(prompts, think_sampling_params)
    
    # Stage 2: Generate SID constrained by thinking
    logger.info("Stage 2: Generating SIDs based on thinking")
    
    # We construct new prompts: Original + <think> + output_thought + </think>
    stage2_prompts = []
    mapping = [] # Map flattened prompts back to original sample index
    
    for idx, output in enumerate(think_outputs):
        for i, out in enumerate(output.outputs):
            thought = out.text
            new_prompt = prompts[idx] + thought + "</think>"
            stage2_prompts.append(new_prompt)
            mapping.append(idx)
            
    # For Stage 2 we just need deterministic generation since we already explored via thinking
    # But to get 10 candidates, we need beam search in Stage 2 or just sample more
    # We will use temperature sampling to get diversity within the same thought
    sid_sampling_params = SamplingParams(
        n=10, # Get 10 candidates per thought
        temperature=0.7, # Add temperature to get diversity
        top_p=0.9,
        max_tokens=20, # Give it enough room to finish the SID
        stop=["<|sid_end|>", "<|im_end|>"]
    )
    
    sid_outputs = llm.generate(stage2_prompts, sid_sampling_params)
    
    # Combine and evaluate
    logger.info("Evaluating Results")
    
    results_by_sample = [[] for _ in range(len(prompts))]
    for prompt_idx, out in zip(mapping, sid_outputs):
        for single_output in out.outputs:
            sid_text = single_output.text
            # It's possible the model outputs <|sid_begin|>...<|sid_end|>
            # Let's ensure we extract just the SID part correctly
            extracted = extract_sid_from_text(sid_text + "<|sid_end|>")
            if extracted != "None":
                sid_text = extracted
                
            # Add safety for None cumulative_logprob
            score = single_output.cumulative_logprob
            if score is None:
                score = -9999.0
                
            results_by_sample[prompt_idx].append((sid_text, score))
        
    hits = {1: 0, 5: 0, 10: 0}
    ndcgs = {5: 0.0, 10: 0.0}
    
    import math
    
    for i, target in enumerate(targets):
        cands_with_scores = sorted(results_by_sample[i], key=lambda x: x[1], reverse=True)
        cands = [x[0] for x in cands_with_scores]
        scores = [x[1] for x in cands_with_scores]
        
        ranked, is_hit, hit_rank = get_topk_results(cands, scores, target)
        
        if is_hit:
            if hit_rank < 1: hits[1] += 1
            if hit_rank < 5: 
                hits[5] += 1
                ndcgs[5] += 1.0 / math.log2(hit_rank + 2)
            if hit_rank < 10: 
                hits[10] += 1
                ndcgs[10] += 1.0 / math.log2(hit_rank + 2)
                
    n = len(prompts)
    logger.info("=" * 40)
    logger.info("FINAL vLLM CoT RESULTS:")
    logger.info(f"Hit@1:  {hits[1]/n:.4f}")
    logger.info(f"Hit@5:  {hits[5]/n:.4f}")
    logger.info(f"Hit@10: {hits[10]/n:.4f}")
    logger.info(f"NDCG@5: {ndcgs[5]/n:.4f}")
    logger.info(f"NDCG@10:{ndcgs[10]/n:.4f}")
    logger.info("=" * 40)

if __name__ == "__main__":
    main()
