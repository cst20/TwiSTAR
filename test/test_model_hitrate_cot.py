#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import random
import datetime
import numpy as np
import re
from collections import defaultdict
from typing import List, Dict, Any, Callable


def extract_all_sids_from_text(text):
    import re
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    matches = re.findall(sid_pattern, text)
    return matches


def extract_sid_from_text(text):
    import re
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    match = re.search(sid_pattern, text)
    if match:
        return match.group(0)
    return text.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized CoT Model Hit Rate Test")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--merged_model_path", type=str,
                        default="../train/results/ReasoningActivation/epoch_2/checkpoint-125",
                        help="Merged model path")
    parser.add_argument("--additional_lora_path", type=str, default=None,
                        help="Optional additional LoRA path")

    parser.add_argument("--test_parquet_file", type=str, 
                        default="../data/training_prediction_sid_data_test.parquet",
                        help="Test parquet file path")

    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size")
    parser.add_argument("--num_beams", type=int, default=20, help="Number of beams for beam search")
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--sample_offset", type=int, default=0,
                        help="sample offset for multi-GPU parallel processing")
    parser.add_argument("--shard_mode", type=str, default="contiguous", choices=["contiguous", "stride"],
                        help="Dataset sharding mode across GPUs: contiguous slices or strided sampling")
    parser.add_argument("--sample_stride", type=int, default=1,
                        help="Stride for 'stride' sharding mode (usually num_gpus)")
    parser.add_argument("--sample_rank", type=int, default=0,
                        help="Rank for 'stride' sharding mode (0..stride-1)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID for logging purposes")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")

    parser.add_argument("--think_max_tokens", type=int, default=128,
                        help="max new tokens for thinking stage")
    parser.add_argument("--sid_max_tokens", type=int, default=20,
                        help="max new tokens for SID generation stage")

    parser.add_argument("--think_temperature", type=float, default=0.8, 
                        help="temperature for CoT thinking generation")
    parser.add_argument("--think_top_p", type=float, default=0.95, 
                        help="top_p for CoT thinking generation")

    parser.add_argument("--sid_temperature", type=float, default=0.6, 
                        help="temperature for SID generation")
    parser.add_argument("--sid_top_p", type=float, default=0.9, 
                        help="top_p for SID generation")
    parser.add_argument("--num_thinking_samples", type=int, default=4,
                        help="number of thinking samples to generate")
    parser.add_argument("--num_beams_per_sample", type=int, default=4,
                        help="number of beams for each thinking sample")

    parser.add_argument(
        "--sid_chunk_size",
        type=int,
        default=0,
        help="Stage2(SID) batched generate chunk size by prompt-count. 0 means auto (all prompts at once, fallback to smaller chunks on OOM).",
    )

    parser.add_argument("--print_generations", action=argparse.BooleanOptionalAction, default=True,
                        help="print prompts, think, and response candidates")
    parser.add_argument("--print_generations_limit", type=int, default=-1,
                        help="maximum number of samples to print when --print_generations is enabled; -1 prints all")
    parser.add_argument("--log_file", type=str,
                        default="./logs/cot_optimized_test.log",
                        help="all output log file path")
    parser.add_argument("--global_trie_file", type=str, default=None,
                        help="Pre-computed global trie file for parallel evaluation")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Inference prefers fast kernels; determinism is not required for this eval.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True


def setup_logging(log_file):
    """Setup logging to file"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('cot_test')
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


def format_chat_prompt_think_stage(user_content):
    system_message = "You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information."
    
    chat_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""
    return chat_prompt


def extract_thinking_content(generated_text, user_content):
    """Extract thinking content from generated text"""
    # Look for the thinking part after <think>
    if "<think>" in generated_text:
        # Split by the original user content to get only the assistant part
        if user_content in generated_text:
            assistant_part = generated_text.split(user_content)[-1]
        else:
            assistant_part = generated_text
        
        # Extract content after <think>
        if "<think>" in assistant_part:
            think_part = assistant_part.split("<think>")[-1]
            # Remove any trailing content that might be generated
            if "</think>" in think_part:
                think_part = think_part.split("</think>")[0]
            return think_part.strip()
    
    return ""


def format_chat_prompt_sid_stage(user_content, thinking_content):
    """Format input with pre-generated CoT content for direct SID generation"""
    system_message = "You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information."
    
    chat_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
<think>
{thinking_content}
</think>
"""
    return chat_prompt


def load_merged_model(model_path, additional_lora_path=None, logger=None):
    """Load pre-merged model and tokenizer, optionally with additional LoRA"""
    logger.info(f"Loading merged model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Set left padding for generation
    
    # Force GPU usage
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        logger.info(f"🔧 Forcing model to GPU: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        model = model.to(device)
        # Ensure KV cache is enabled for generation speed.
        if hasattr(model, "config"):
            model.config.use_cache = True
        logger.info(f"✅ Model moved to GPU")
        
        # Verify GPU placement
        first_param_device = next(model.parameters()).device
        if 'cuda' in str(first_param_device):
            logger.info(f"✅ Confirmed: Model is on {first_param_device}")
        else:
            logger.error(f"❌ Failed: Model is still on {first_param_device}")
            raise RuntimeError("Failed to move model to GPU")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )
    
    logger.info(f"Merged model loaded successfully, tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Optionally load additional LoRA
    if additional_lora_path and os.path.exists(additional_lora_path):
        logger.info(f"Loading additional LoRA from: {additional_lora_path}")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, additional_lora_path)
            logger.info("Additional LoRA loaded successfully")
        except Exception as e:
            logger.error(f"Error loading additional LoRA: {e}")
            logger.info("Continuing with merged model only")
    elif additional_lora_path:
        logger.warning(f"Additional LoRA path does not exist: {additional_lora_path}")
        logger.info("Continuing with merged model only")
    
    return model, tokenizer


class ParquetTestDataset(Dataset):
    """Dataset for loading test data from parquet files"""
    
    def __init__(self, parquet_file, sample_num=-1, sample_offset=0,
                 shard_mode: str = "contiguous", sample_stride: int = 1, sample_rank: int = 0,
                 logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Loading test data from parquet file: {parquet_file}")
        
        # Load parquet file
        self.df = pd.read_parquet(parquet_file)
        self.logger.info(f"Loaded {len(self.df)} samples from parquet")
        
        # Global offset then optional global limit
        if sample_offset > 0:
            self.df = self.df.iloc[sample_offset:].reset_index(drop=True)
            self.logger.info(f"Applied global offset {sample_offset}, remaining samples: {len(self.df)}")

        if sample_num > 0 and sample_num < len(self.df):
            self.df = self.df.iloc[:sample_num].reset_index(drop=True)
            self.logger.info(f"Limited to global first {sample_num} samples")

        # Sharding across GPUs
        if shard_mode == "stride":
            if sample_stride <= 0:
                raise ValueError("sample_stride must be > 0")
            if not (0 <= sample_rank < sample_stride):
                raise ValueError(f"sample_rank must be in [0, {sample_stride - 1}]")
            before = len(self.df)
            self.df = self.df.iloc[sample_rank::sample_stride].reset_index(drop=True)
            self.logger.info(
                f"Applied stride sharding: rank={sample_rank}/{sample_stride}, local samples: {len(self.df)} (from {before})"
            )
        elif shard_mode == "contiguous":
            # Backward compatible: sample_num means per-GPU limit when used with sample_offset slicing.
            # (Already applied above as global limit, so nothing else to do here.)
            pass
        else:
            raise ValueError(f"Unknown shard_mode: {shard_mode}")
        
        # Expected columns: ['description', 'groundtruth', 'user_id']
        required_cols = ['description', 'groundtruth']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in parquet file. Available: {list(self.df.columns)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'input_ids': row['description'],
            'labels': row['groundtruth'],
            'user_id': row.get('user_id', f'user_{idx}')
        }
    
    def get_prefix_allowed_tokens_fn(self, tokenizer, global_trie_file=None):
        """Create prefix allowed tokens function for SID constrained generation based on exact trie"""
        
        if not global_trie_file:
            raise ValueError("Global trie file path must be provided")
        
        if not os.path.exists(global_trie_file):
            raise FileNotFoundError(f"Global trie file not found: {global_trie_file}. Please run precompute_global_trie.py first.")
        
        # Load pre-computed exact trie
        self.logger.info(f"Loading pre-computed exact trie from: {global_trie_file}")
        import pickle
        with open(global_trie_file, 'rb') as f:
            trie_data = pickle.load(f)
        
        # Verify this is an exact trie
        trie_type = trie_data.get('trie_type', None)
        if trie_type != 'exact':
            raise ValueError(f"Expected exact trie file, but got trie_type='{trie_type}'. Please regenerate the trie file.")
        
        # Load exact trie structure
        allowed_tokens = trie_data['exact_trie']
        valid_sids = trie_data['valid_sids']
        search_space_size = trie_data.get('search_space_size', 0)
        max_length = trie_data.get('max_length', 0)
        
        self.logger.info(f"Loaded exact trie for CoT:")
        self.logger.info(f"  Total unique SIDs: {len(valid_sids)}")
        self.logger.info(f"  Search space size: {search_space_size:,} (exact match only)")
        self.logger.info(f"  Trie depth: {max_length}")
        
        for pos in range(min(6, max_length)):
            num_tokens = len(allowed_tokens.get(pos, {}))
            self.logger.info(f"  Position {pos}: {num_tokens} possible tokens")
        
        # Cache for fast prefix_allowed_tokens_fn (used by batched SID generation)
        self._allowed_tokens = allowed_tokens
        self._all_token_ids = list(range(len(tokenizer)))

        # Get "</think>" separator with newline (to match our prompt format)
        sep = tokenizer("</think>\n", add_special_tokens=False)["input_ids"]
        self._sep = sep
        
        self.logger.info(f"CoT SID constraint setup:")
        self.logger.info(f"  </think>\\n tokens: {sep}")
        self.logger.info(f"  Direct SID generation after </think>\\n")
        
        def find_last_sublist(lst, sub):
            """Find the last occurrence of sublist in list"""
            if not sub:
                return None
            n, m = len(lst), len(sub)
            for start in range(n - m, -1, -1):
                if lst[start:start + m] == sub:
                    return start
            return None
        
        def prefix_allowed_tokens_fn(batch_id, sentence):
            """Return allowed tokens based on current generation position using exact trie"""
            # NOTE: This generic implementation scans for the separator.
            # It is significantly slower than the fast prompt-length based version.
            sentence = sentence.tolist()
            
            # Find "</think>" position
            pos = find_last_sublist(sentence, sep)
            if pos is None:
                # Before "</think>", allow all tokens
                return self._all_token_ids
            
            # Calculate position after "</think>" - directly apply SID constraints
            pos_after_sep = pos + len(sep)
            generated_after_sep = sentence[pos_after_sep:]
            sid_pos = len(generated_after_sep)
            
            # Use exact trie: check what tokens are allowed at this SID position
            if sid_pos == 0:
                # First SID token position - should be <|sid_begin|>
                if 0 in allowed_tokens:
                    allowed = list(allowed_tokens[0].keys())
                    return allowed
                else:
                    # Allow all tokens if trie is not properly set up
                    return self._all_token_ids
            else:
                # Look up what's allowed based on previous SID tokens
                if sid_pos > 0 and len(generated_after_sep) >= sid_pos:
                    prev_token = generated_after_sep[sid_pos - 1]
                    prev_pos = sid_pos - 1
                    
                    if prev_pos in allowed_tokens and prev_token in allowed_tokens[prev_pos]:
                        allowed = allowed_tokens[prev_pos][prev_token]
                        return allowed
                
                # Fallback to allow all tokens if no valid continuation found
                return self._all_token_ids
        
        return prefix_allowed_tokens_fn


    def get_fast_prefix_allowed_tokens_fn(self, prompt_len: int):
        """Fast prefix_allowed_tokens_fn specialized for SID stage.

        Assumes the prompt already ends with "</think>\n" and SID generation starts immediately
        after the prompt. Avoids scanning the whole sequence for the separator on every step.
        """
        if not hasattr(self, "_allowed_tokens"):
            raise RuntimeError("Exact trie not loaded. Call get_prefix_allowed_tokens_fn() first.")

        allowed_tokens = self._allowed_tokens
        all_token_ids = self._all_token_ids

        def prefix_allowed_tokens_fn(batch_id, sentence):
            # sentence: 1D LongTensor (prompt + generated)
            sid_pos = int(sentence.numel() - prompt_len)
            if sid_pos <= 0:
                # First SID token position
                if 0 in allowed_tokens:
                    return list(allowed_tokens[0].keys())
                return all_token_ids

            prev_pos = sid_pos - 1
            prev_token = int(sentence[prompt_len + prev_pos])
            if prev_pos in allowed_tokens and prev_token in allowed_tokens[prev_pos]:
                return allowed_tokens[prev_pos][prev_token]
            return all_token_ids

        return prefix_allowed_tokens_fn


def batch_generate_sid_candidates(model, tokenizer, user_contents, all_thinking_contents,
                                 prefix_allowed_tokens_fn_factory, args, logger):
    """Batched SID beam search for all (sample, thinking) prompts.

    This removes Python-loop overhead and improves GPU utilization.
    """
    sid_prompts = []
    for sample_idx, user_content in enumerate(user_contents):
        for thinking_idx in range(args.num_thinking_samples):
            thinking_content = all_thinking_contents[sample_idx][thinking_idx]
            sid_prompts.append(format_chat_prompt_sid_stage(user_content, thinking_content))

    enc = tokenizer(
        sid_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])

    prefix_allowed_tokens_fn = None
    if prefix_allowed_tokens_fn_factory is not None:
        prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_factory(prompt_len)

    num_beams = args.num_beams_per_sample
    generate_kwargs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc.get("attention_mask", None),
        "max_new_tokens": args.sid_max_tokens,
        "num_beams": num_beams,
        "num_return_sequences": num_beams,
        "output_scores": True,
        "return_dict_in_generate": True,
        "early_stopping": True,
        "use_cache": True,
    }
    if prefix_allowed_tokens_fn is not None:
        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

    def _decode_output(output):
        output_ids = output["sequences"]
        scores = output.get("sequences_scores", None)
        decoded = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if scores is not None:
            if hasattr(scores, "detach"):
                scores_list = [float(s) for s in scores.detach().cpu().tolist()]
            else:
                scores_list = [float(s) for s in scores]
        else:
            scores_list = [0.0] * len(decoded)
        return decoded, scores_list

    def _generate_chunk(enc_slice):
        chunk_prompt_len = int(enc_slice["input_ids"].shape[1])
        chunk_kwargs = dict(generate_kwargs)
        chunk_kwargs["input_ids"] = enc_slice["input_ids"]
        chunk_kwargs["attention_mask"] = enc_slice.get("attention_mask", None)
        if prefix_allowed_tokens_fn_factory is not None:
            chunk_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn_factory(chunk_prompt_len)
        return model.generate(**chunk_kwargs)

    prompt_cnt = int(enc["input_ids"].shape[0])
    chunk_size = int(args.sid_chunk_size) if getattr(args, "sid_chunk_size", 0) else 0
    if chunk_size <= 0:
        chunk_size = prompt_cnt

    decoded_all, scores_all = [], []
    try:
        if chunk_size >= prompt_cnt:
            output = model.generate(**generate_kwargs)
            decoded_all, scores_all = _decode_output(output)
        else:
            for s in range(0, prompt_cnt, chunk_size):
                enc_slice = {k: v[s:s + chunk_size] for k, v in enc.items()}
                out = _generate_chunk(enc_slice)
                d, sc = _decode_output(out)
                decoded_all.extend(d)
                scores_all.extend(sc)
    except RuntimeError as e:
        # OOM fallback: split into smaller chunks automatically
        if "out of memory" not in str(e).lower():
            raise
        logger.warning(f"CUDA OOM in batched SID stage (prompts={prompt_cnt}, chunk_size={chunk_size}). Fallback to chunked generation.")
        torch.cuda.empty_cache()
        # start from half and go down to 1
        cs = max(1, min(prompt_cnt, chunk_size // 2 if chunk_size > 1 else prompt_cnt // 2 or 1))
        while True:
            try:
                decoded_all, scores_all = [], []
                for s in range(0, prompt_cnt, cs):
                    enc_slice = {k: v[s:s + cs] for k, v in enc.items()}
                    out = _generate_chunk(enc_slice)
                    d, sc = _decode_output(out)
                    decoded_all.extend(d)
                    scores_all.extend(sc)
                chunk_size = cs
                break
            except RuntimeError as e2:
                if "out of memory" not in str(e2).lower() or cs == 1:
                    raise
                torch.cuda.empty_cache()
                cs = max(1, cs // 2)

    logger.info(
        f"✅ Batched SID stage completed: prompts={prompt_cnt}, beams={num_beams}, chunk_size={chunk_size}, total={len(decoded_all)}"
    )
    return decoded_all, scores_all


class TestCollator:
    """Collator for test data"""
    
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
    
    def __call__(self, batch):
        targets = [d["labels"] for d in batch]
        user_contents = [d["input_ids"] for d in batch]
        
        return {
            "user_contents": user_contents,
            "targets": targets
        }


def batch_generate_thinking_optimized(model, tokenizer, user_contents, args, logger):
    logger.info("🚀 Optimized batch thinking generation started...")

    all_think_prompts = []
    batch_mapping = []
    
    for sample_idx, user_content in enumerate(user_contents):
        think_prompt = format_chat_prompt_think_stage(user_content)
        for thinking_idx in range(args.num_thinking_samples):
            all_think_prompts.append(think_prompt)
            batch_mapping.append((sample_idx, thinking_idx))
    
    logger.info(f"📊 Batch thinking: {len(all_think_prompts)} prompts for {len(user_contents)} samples × {args.num_thinking_samples} thinking samples")

    enc_think_batch = tokenizer(
        all_think_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    )
    enc_think_batch = {k: v.to(model.device) for k, v in enc_think_batch.items()}

    logger.info("🤔 Generating all thinking samples in parallel...")
    think_outputs = model.generate(
        input_ids=enc_think_batch["input_ids"],
        attention_mask=enc_think_batch.get("attention_mask", None),
        max_new_tokens=args.think_max_tokens,
        num_beams=1,
        do_sample=True,
        temperature=args.think_temperature,
        top_p=args.think_top_p,
        return_dict_in_generate=True,
        output_scores=False,
        early_stopping=False,
        use_cache=True,
        output_hidden_states=False
    )

    think_decoded_all = tokenizer.batch_decode(think_outputs["sequences"], skip_special_tokens=True)

    all_thinking_contents = [[] for _ in range(len(user_contents))]
    
    for i, (sample_idx, thinking_idx) in enumerate(batch_mapping):
        thinking = extract_thinking_content(think_decoded_all[i], user_contents[sample_idx])
        all_thinking_contents[sample_idx].append(thinking)
    
    logger.info(f"✅ Batch thinking generation completed: {len(user_contents)} samples × {args.num_thinking_samples} thinking samples")
    
    return all_thinking_contents


def generate_sid_standard(model, tokenizer, user_content, thinking_content,
                        prefix_allowed_tokens_fn, args, logger):
    sid_prompt = format_chat_prompt_sid_stage(user_content, thinking_content)
    
    enc_sid = tokenizer(
        [sid_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    )
    enc_sid = {k: v.to(model.device) for k, v in enc_sid.items()}

    num_beams = args.num_beams_per_sample
    
    generate_kwargs = {
        "input_ids": enc_sid["input_ids"],
        "attention_mask": enc_sid.get("attention_mask", None),
        "max_new_tokens": args.sid_max_tokens,
        "num_beams": num_beams,
        "num_return_sequences": num_beams,
        "output_scores": True,
        "return_dict_in_generate": True,
        "early_stopping": True,
        "use_cache": True,
    }
    
    # Add SID constrained generation
    if prefix_allowed_tokens_fn is not None:
        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
    
    try:
        output = model.generate(**generate_kwargs)
        
    except RuntimeError as e:
        err = str(e).lower()
        if "out of memory" in err or "cuda" in err:
            logger.warning(f"CUDA OOM with beam={num_beams}. Reducing beam size.")
            num_beams -= 1
            if num_beams < 1:
                raise RuntimeError("Beam search OOM even with beam=1") from e
            torch.cuda.empty_cache()
            generate_kwargs["num_beams"] = num_beams
            generate_kwargs["num_return_sequences"] = num_beams
            output = model.generate(**generate_kwargs)
        else:
            raise
    
    return output, num_beams


def process_unique_top10_candidates(predictions, scores, effective_num_beams):
    batch_size = len(predictions) // effective_num_beams
    new_predictions = []
    new_scores = []

    for b in range(batch_size):
        start = b * effective_num_beams
        end = start + effective_num_beams

        batch_preds = predictions[start:end]
        batch_scores = scores[start:end]

        sid_to_score = {}
        for pred, score in zip(batch_preds, batch_scores):
            sid_part = pred.split("</think>")[-1].strip().replace(" ", "")
            sid = extract_sid_from_text(sid_part)

            if sid in sid_to_score:
                sid_to_score[sid] = max(sid_to_score[sid], score)
            else:
                sid_to_score[sid] = score

        sorted_items = sorted(sid_to_score.items(), key=lambda x: x[1], reverse=True)
        top10_items = sorted_items[:10]

        sample_preds = []
        sample_scores = []

        for sid, score in top10_items:
            if batch_preds:
                prefix = batch_preds[0].split("</think>")[0] + "</think>"
                full_pred = prefix + "\n" + sid
            else:
                full_pred = sid
            sample_preds.append(full_pred)
            sample_scores.append(score)

        while len(sample_preds) < 10:
            if sample_preds:
                sample_preds.append(sample_preds[-1])
                penalty = (len(sample_preds) - len(top10_items)) * 0.1
                sample_scores.append(sample_scores[-1] - penalty)
            else:
                sample_preds.append("")
                sample_scores.append(-1000.0)
        
        new_predictions.extend(sample_preds)
        new_scores.extend(sample_scores)
    
    return new_predictions, new_scores


def get_topk_results(predictions, scores, targets, k, all_items=None):
    """Extract top-k results from predictions"""
    results = []
    B = len(targets)
    predictions = [_.split("</think>")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ", "") for _ in predictions]
    
    # Extract only SID parts from both predictions and targets
    predictions = [extract_sid_from_text(pred) for pred in predictions]
    
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000
    
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        
        pairs = [(seq, score) for seq, score in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        
        # Extract SID from target as well
        target_item = extract_sid_from_text(targets[b])
        
        one_results = []
        for pred_seq, pred_score in sorted_pairs:
            if pred_seq == target_item:
                one_results.append(1)
            else:
                one_results.append(0)
        
        results.append(one_results)
    
    return results


def hit_k(topk_results, k):
    """Calculate hit@k metric"""
    hit = 0.0
    for row in topk_results:
        if len(row) >= k and max(row[:k]) == 1:
            hit += 1
    return hit / len(topk_results)


def ndcg_k(topk_results, k):
    """Calculate ndcg@k metric"""
    ndcg = 0.0
    for row in topk_results:
        dcg = 0.0
        for i in range(min(k, len(row))):
            if row[i] == 1:
                dcg += 1.0 / np.log2(i + 2)
        idcg = 1.0 / np.log2(2)  # Best case: hit at position 1
        ndcg += dcg / idcg
    return ndcg / len(topk_results)


def get_metrics_results(topk_results, metrics):
    """Calculate evaluation metrics"""
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError(f"Metric {m} not implemented")
    
    return res


def run_evaluation(args):
    """Main evaluation function with CoT reasoning"""
    set_seed(args.seed)
    logger = setup_logging(args.log_file)
    logger.info(f"🚀 Starting CoT-Enhanced Model Hit Rate Evaluation [GPU {args.gpu_id}]")

    logger.info("=" * 80)
    logger.info("📋 EVALUATION CONFIGURATION")
    logger.info("=" * 80)

    logger.info("🔧 Model Configuration:")
    logger.info(f"  Merged Model Path: {args.merged_model_path}")
    logger.info(f"  Additional LoRA Path: {args.additional_lora_path or 'None'}")
    logger.info(f"  Test Parquet File: {args.test_parquet_file}")
    logger.info(f"  Global Trie File: {args.global_trie_file}")

    logger.info("📊 Data Configuration:")
    logger.info(f"  Test Batch Size: {args.test_batch_size}")
    logger.info(f"  Sample Number: {args.sample_num}")
    logger.info(f"  Sample Offset: {args.sample_offset}")
    logger.info(f"  GPU ID: {args.gpu_id}")
    logger.info(f"  Metrics: {args.metrics}")

    logger.info("🤔 CoT Reasoning Configuration:")
    logger.info(f"  Thinking Samples per Input: {args.num_thinking_samples}")
    logger.info(f"  Beams per Thinking Sample: {args.num_beams_per_sample}")
    logger.info(f"  Total Initial Candidates: {args.num_thinking_samples * args.num_beams_per_sample}")
    logger.info(f"  Final Unique Top-K: 10")

    logger.info("⚙️ Generation Parameters:")
    logger.info(f"  Think Max Tokens: {args.think_max_tokens}")
    logger.info(f"  SID Max Tokens: {args.sid_max_tokens}")
    logger.info(f"  Think Temperature: {args.think_temperature}")
    logger.info(f"  Think Top-p: {args.think_top_p}")
    logger.info(f"  SID Temperature: {args.sid_temperature}")
    logger.info(f"  SID Top-p: {args.sid_top_p}")

    logger.info("🔍 Other Configuration:")
    logger.info(f"  Print Generations: {args.print_generations}")
    logger.info(f"  Print Generations Limit: {args.print_generations_limit}")
    logger.info(f"  Random Seed: {args.seed}")
    logger.info(f"  Log File: {args.log_file}")

    logger.info("=" * 80)
    
    # 1. Load merged model
    logger.info("=" * 60)
    logger.info("Loading merged model...")
    final_model, tokenizer = load_merged_model(
        args.merged_model_path,
        args.additional_lora_path,
        logger
    )
    final_model.eval()

    logger.info("📊 Loading test dataset...")
    if not os.path.exists(args.test_parquet_file):
        raise FileNotFoundError(f"Parquet file not found: {args.test_parquet_file}")
    
    test_dataset = ParquetTestDataset(
        args.test_parquet_file,
        args.sample_num,
        args.sample_offset,
        shard_mode=args.shard_mode,
        sample_stride=args.sample_stride,
        sample_rank=args.sample_rank,
        logger=logger,
    )
    # Load trie + build generic prefix fn (also populates cache for fast version)
    _ = test_dataset.get_prefix_allowed_tokens_fn(tokenizer, args.global_trie_file)
    logger.info(f"Using parquet file: {args.test_parquet_file}")
    if args.global_trie_file:
        logger.info(f"✅ Global trie file: {args.global_trie_file}")
    logger.info("✅ CoT-Enhanced + SID constrained generation enabled")
    
    collator = TestCollator(args, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"📈 Test data size: {len(test_dataset)}")
    
    # 3. Start evaluation
    metrics = args.metrics.split(",")
    all_topk_results = []
    total = 0
    printed_generation_count = 0
    
    logger.info("🚀 Starting CoT-Enhanced evaluation...")
    
    import time
    start_time = time.time()
    
    with torch.inference_mode():
        progress_bar = tqdm(test_loader, desc="CoT Testing")
        for step, batch in enumerate(progress_bar):
            user_contents = batch["user_contents"]
            targets = batch["targets"]
            bs = len(targets)
            
            # Calculate progress information
            current_step = step + 1
            total_steps = len(test_loader)
            elapsed = time.time() - start_time
            if current_step > 0:
                avg_time = elapsed / current_step
                remaining_time = avg_time * (total_steps - current_step)
                
                # Format times
                elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
                remaining_str = f"{int(remaining_time//60):02d}:{int(remaining_time%60):02d}"
                
                # Create progress bar visual
                progress_pct = current_step / total_steps
                bar_length = 10
                filled = int(progress_pct * bar_length)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                progress_info = f"CoT-Enhanced Testing: {progress_pct*100:3.0f}%|{bar}| {current_step}/{total_steps} [{elapsed_str}<{remaining_str}, {avg_time:.2f}s/it]"
                logger.info(progress_info)

            all_thinking_contents = batch_generate_thinking_optimized(
                final_model, tokenizer, user_contents, args, logger
            )

            logger.info("🎯 Stage 2: Batched SID generation after </think> for all thinking samples...")
            decoded, scores_list = batch_generate_sid_candidates(
                final_model,
                tokenizer,
                user_contents,
                all_thinking_contents,
                test_dataset.get_fast_prefix_allowed_tokens_fn,
                args,
                logger,
            )
            effective_num_beams = args.num_thinking_samples * args.num_beams_per_sample

            logger.info(f"🔄 Applying unique deduplication and Top-10 selection...")
            decoded, scores_list = process_unique_top10_candidates(
                decoded, scores_list, effective_num_beams
            )
            effective_num_beams = 10
            logger.info(f"✅ After deduplication: {len(decoded)} total candidates (10 per sample)")
            
            # Print generations if requested
            if args.print_generations and (
                args.print_generations_limit < 0 or
                printed_generation_count < args.print_generations_limit
            ):
                remaining_to_print = bs if args.print_generations_limit < 0 else min(
                    bs, args.print_generations_limit - printed_generation_count
                )

                for i in range(remaining_to_print):
                    start = i * effective_num_beams
                    end = start + effective_num_beams
                    cands = decoded[start:end]
                    cand_scores = scores_list[start:end]
                    
                    logger.info(f"----- CoT-ENHANCED SAMPLE {printed_generation_count + 1} (shard_index={total + i}) -----")
                    logger.info(f"USER INPUT COMPLETE:")
                    logger.info(f"{user_contents[i]}")
                    logger.info(f"")

                    display_thinking_count = min(5, args.num_thinking_samples)
                    for thinking_idx in range(display_thinking_count):
                        logger.info(f"THINKING {thinking_idx+1}/{args.num_thinking_samples}:")
                        logger.info(f"{all_thinking_contents[i][thinking_idx]}")
                        logger.info(f"")
                    
                    if args.num_thinking_samples > 5:
                        logger.info(f"... (and {args.num_thinking_samples - 5} more thinking samples)")
                        logger.info(f"")
                    
                    logger.info("UNIQUE TOP-10 SID_CANDIDATES:")
                    for j, (c, sc) in enumerate(zip(cands, cand_scores)):
                        sid_result = extract_sid_from_text(c.split("</think>")[-1])
                        logger.info(f"  Rank {j+1}: score={sc:.4f} → {sid_result}")
                    logger.info(f"")
                    logger.info(f"TARGET:")
                    logger.info(f"{targets[i]}")
                    logger.info("-" * 80)

                printed_generation_count += remaining_to_print
            
            # Calculate topk results
            topk_res = get_topk_results(
                decoded, scores_list, 
                targets, effective_num_beams,
                all_items=None
            )
            
            # Accumulate results
            all_topk_results.extend(topk_res)
            total += bs
            
            # Progress report every 20 steps
            if (step + 1) % 20 == 0:
                temp_metrics_results = get_metrics_results(all_topk_results, metrics)
                logger.info("=" * 50)
                logger.info(f"📊 CoT-ENHANCED PROGRESS REPORT - Step {step+1}/{len(test_loader)}")
                logger.info(f"💾 Processed samples: {total}")
                logger.info("📈 Current Metrics:")
                for metric, value in temp_metrics_results.items():
                    logger.info(f"  {metric:>10}: {value:.4f}")
                logger.info("=" * 50)
    
    # 4. Final results
    final_metrics_results = get_metrics_results(all_topk_results, metrics)
    
    logger.info("=" * 60)
    logger.info("🎯 Final CoT Hit Rate Results:")
    logger.info("=" * 60)
    for metric, value in final_metrics_results.items():
        logger.info(f"{metric:>10}: {value:.4f}")
    logger.info("=" * 60)
    
    # 5. Test summary
    logger.info("\n📊 CoT-Enhanced Test Summary:")
    logger.info(f"Merged model: {args.merged_model_path}")
    if args.additional_lora_path:
        logger.info(f"Additional LoRA: {args.additional_lora_path}")
    logger.info(f"Test data: {args.test_parquet_file}")
    logger.info(f"Total samples: {total}")
    logger.info(f"Batch size: {args.test_batch_size}")
    logger.info(f"Beam size: {args.num_beams}")
    logger.info(f"SID max tokens: {args.sid_max_tokens}")
    
    logger.info("\n✅ CoT-Enhanced Evaluation completed successfully!")
    
    return final_metrics_results


def main():
    """Main function"""
    args = parse_args()
    
    try:
        results = run_evaluation(args)
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
