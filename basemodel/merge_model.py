#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_local_path(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return str(path)


def resolve_alignment_checkpoint() -> str:
    env_path = os.environ.get("LORA_MODEL_PATH", "").strip()
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(f"Configured alignment checkpoint not found: {path}")
        return str(path)

    align_dir = Path("../train/results/beauty_align")
    candidates = sorted(align_dir.glob("checkpoint-*"), key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(
            f"No alignment checkpoint found under {align_dir}. Run Stage1 alignment training first or set LORA_MODEL_PATH."
        )
    return str(candidates[-1])

def merge_and_save_models():
    base_model_path = resolve_local_path(os.environ.get('BASE_MODEL_PATH', './Qwen3-1-7B-expand'))
    lora_model_path = resolve_alignment_checkpoint()
    output_path = resolve_local_path(os.environ.get('MERGED_OUTPUT_PATH', './merged_beauty_model_1-1'))
    
    print("="*80)
    print("MERGING TWO MODELS INTO SINGLE DIRECTORY")
    print("="*80)
    
    if os.path.exists(output_path):
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    try:
        print(f"\n1. Loading base model from: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   Base model loaded successfully")
        print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
        
        print(f"\n2. Loading and merging alignment model from: {lora_model_path}")
        stage2_model = PeftModel.from_pretrained(base_model, lora_model_path)
        final_merged_model = stage2_model.merge_and_unload()
        print(f"   Alignment model merged successfully")
        
        del base_model, stage2_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\n3. Saving final merged model to: {output_path}")
        final_merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print(f"   ✓ Model saved successfully!")
        
        print(f"\n4. Verifying saved model...")
        saved_files = os.listdir(output_path)
        print(f"   Saved files: {saved_files}")
        
        test_model = AutoModelForCausalLM.from_pretrained(output_path)
        test_tokenizer = AutoTokenizer.from_pretrained(output_path)
        print(f"   ✓ Model verification successful!")
        print(f"   ✓ Model parameters: {test_model.num_parameters():,}")
        print(f"   ✓ Tokenizer vocab size: {test_tokenizer.vocab_size}")
        
        del test_model, test_tokenizer, final_merged_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\n" + "="*80)
        print(f"MODEL MERGE COMPLETED SUCCESSFULLY!")
        print(f"Merged model saved to: {output_path}")
        print("="*80)
        
        return output_path
        
    except Exception as e:
        print(f"\nError during model merging: {e}")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        raise

if __name__ == "__main__":
    merge_and_save_models()
