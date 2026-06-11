#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_special_tokens(max_range: int = 256) -> list[str]:
    special_tokens: list[str] = []

    special_tokens.append("<|sid_begin|>")
    special_tokens.append("<|sid_end|>")

    for prefix in ["s_a", "s_b", "s_c", "s_d"]:
        for idx in range(max_range):
            special_tokens.append(f"<{prefix}_{idx}>")

    return special_tokens


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be a positive integer")
    return ((value + multiple - 1) // multiple) * multiple


def expand_vocabulary(
    base_model_dir: Path,
    save_dir: Path,
) -> None:
    print(f"Loading model config from: {base_model_dir}")
    config = AutoConfig.from_pretrained(base_model_dir)

    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(base_model_dir)

    # 在集群上 GPU 可能被其他任务几乎占满，这里增加一个开关，允许强制使用 CPU，避免 CUDA OOM
    use_cpu_only = os.environ.get("ONEREC_EXPAND_CPU_ONLY", "0") in {"1", "true", "True"}
    if torch.cuda.is_available() and not use_cpu_only:
        device = "cuda"
    else:
        device = "cpu"

    print(f"Inference device: {device}")
    if device == "cuda":
        model = model.to("cuda")
    device_for_encoding = device

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    new_tokens = get_special_tokens(max_range=256)
    print(f"Preparing to add {len(new_tokens)} special tokens.")

    # transformers 版本兼容：部分版本不支持 replace_additional_special_tokens 参数
    try:
        tokens_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens},
            replace_additional_special_tokens=False,
        )
    except TypeError:
        tokens_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"Successfully added {tokens_added} tokens.")

    updated_vocab_size = len(tokenizer)
    target_vocab_size = round_up_to_multiple(updated_vocab_size, 256)

    print(
        f"Current vocab size: {updated_vocab_size}, adjusting to: {target_vocab_size} (nearest 256 multiple)."
    )
    model.resize_token_embeddings(target_vocab_size)

    # Some configs (e.g. Qwen3.5) keep text vocab_size under a nested text_config.
    # Make sure both top-level and nested vocab_size are consistent.
    config.vocab_size = target_vocab_size
    if hasattr(config, "text_config") and config.text_config is not None:
        try:
            config.text_config.vocab_size = target_vocab_size
        except Exception:
            if isinstance(config.text_config, dict):
                config.text_config["vocab_size"] = target_vocab_size

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving expanded model to: {save_dir}")
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    config.save_pretrained(save_dir)

    sample_text = "<|sid_begin|><s_a_0><s_b_0><s_c_0><s_d_0><|sid_end|>"
    sample_ids = tokenizer.encode(sample_text, return_tensors="pt").to(device_for_encoding)
    print(f"Sample tokens encoded shape: {sample_ids.shape}")


def main() -> None:
    import argparse

    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Expand tokenizer vocab with SID tokens")
    parser.add_argument(
        "--base_model_dir",
        type=str,
        default=str(base_dir / "Qwen3-1-7B"),
        help="Base HF model directory (under basemodel/) to expand",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(base_dir / "Qwen3-1-7B-expand"),
        help="Output directory for expanded model (under basemodel/)",
    )
    args = parser.parse_args()

    base_model_dir = Path(args.base_model_dir).resolve()
    save_dir = Path(args.save_dir).resolve()

    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")

    expand_vocabulary(base_model_dir=base_model_dir, save_dir=save_dir)


if __name__ == "__main__":
    main()
