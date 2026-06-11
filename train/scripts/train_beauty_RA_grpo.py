#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRPO-based Reasoning Activation (RA) for TwiSTAR sequential recommendation.

This replaces SFT on "assistant_content = <think>... </think> + groundtruth" with
reward optimization:
  Reward = format(thinking + item wrapper) + hit(groundtruth)

Format constraints (rewarded):
  1) Must include `<think>...</think>` with non-empty reasoning.
  2) After `</think>`, must output EXACTLY ONE item wrapped by:
       <|item_begin|><|sid_begin|>...<|sid_end|><|item_end|>
  3) No extra tool calls / special tokens / extra text.

Data
  Uses existing parquet format: columns ['user_id', 'description', 'groundtruth'].

Modes
  - dry_run: rollout candidates + score + save rollouts.jsonl
  - train: minimal GRPO-style update with HF Transformers + PEFT LoRA

Notes
  - This is a runnable baseline. For large-scale GRPO with vLLM engine and distributed
    training, integrate with a local `verl` backend if available.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SID_PATTERN = re.compile(r"<\|sid_begin\|>.*?<\|sid_end\|>")
# Strict SID = exactly 4-level tokenized code like <s_a_99><s_b_19><s_c_220><s_d_204>
SID_STRICT_PATTERN = re.compile(
    r"<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>"
)
SID_STRICT_PARTS_PATTERN = re.compile(
    r"<\|sid_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)><s_d_(\d+)><\|sid_end\|>"
)
ITEM_PATTERN = re.compile(r"<\|item_begin\|>(.*?)<\|item_end\|>", re.DOTALL)


def _is_peft_adapter_dir(path: str) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return False
    # PEFT adapter checkpoints typically contain adapter_config + adapter_model,
    # and do NOT contain a full base-model config / weights.
    has_adapter = (p / "adapter_config.json").exists() and (p / "adapter_model.safetensors").exists()
    has_full_model = (p / "config.json").exists() or (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
    return bool(has_adapter and (not has_full_model))


def load_tokenizer_and_model(
    model_path: str,
    *,
    device_map,
    dtype,
    trainable: bool,
):
    """Load either a full HF model dir or a PEFT adapter dir.

    - If `model_path` is a PEFT adapter checkpoint (beauty_align stage1), load its base
      model (from adapter_config.json) and attach the adapter.
    - Otherwise, load as a normal HF model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Tokenizer: prefer the provided path (adapter checkpoints often ship tokenizer.json)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        # For Qwen3, `<|im_end|>` is eos; we'll override generation eos later.
        tok.pad_token_id = tok.eos_token_id

    if _is_peft_adapter_dir(model_path):
        import json
        from peft import PeftModel

        cfg = json.loads(Path(model_path, "adapter_config.json").read_text(encoding="utf-8"))
        base_path = cfg.get("base_model_name_or_path")
        if not base_path:
            raise RuntimeError(f"PEFT adapter missing base_model_name_or_path: {model_path}")

        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path, is_trainable=bool(trainable))
        return tok, model

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return tok, model


@dataclass
class Sample:
    user_id: str
    description: str
    groundtruth: str


def load_parquet(path: Path, limit: int) -> List[Sample]:
    import pandas as pd

    df = pd.read_parquet(path)
    # limit <= 0 means load full dataset
    if int(limit) > 0:
        df = df.head(int(limit))
    out: List[Sample] = []
    for _, row in df.iterrows():
        out.append(
            Sample(
                user_id=str(row.get("user_id", "")),
                description=str(row.get("description", "")),
                groundtruth=str(row.get("groundtruth", "")),
            )
        )
    return out


def build_prompt(desc: str) -> str:
    # NOTE: Do NOT use Qwen chat tokens (<|im_start|>/<|im_end|>) here.
    # For Qwen3 tokenizer, `eos_token` == `<|im_end|>`, and the model tends to
    # immediately emit it when prompted in chat format.
    # Keep prompt plain-text to reduce degenerate `<|im_end|>` / chat-token copying.
    # Prime the model to start inside the think block to reduce the chance it emits
    # stray tokens or closes </think> immediately.
    return (
        "You are a professional recommendation expert. "
        "Predict the single most likely next item the user will purchase based on the purchase history.\n\n"
        f"{desc.strip()}\n\n"
        "Strict output requirements (MUST follow):\n"
        "1) Start your response with <think> and put reasoning ONLY inside <think>...</think>.\n"
        "   The <think> block MUST contain 1-2 short English sentences (at least 20 characters total).\n"
        "   Use plain English words; do NOT output any tokens like <|sid_begin|>, <s_a_*>, or angle-bracket tags.\n"
        "2) After </think>, output EXACTLY ONE SID string and NOTHING ELSE.\n"
        "   It MUST be strict: <|sid_begin|><s_a_#><s_b_#><s_c_#><s_d_#><|sid_end|>\n"
        "3) Do NOT output any other text, lists, titles, categories, punctuation, or extra lines.\n"
        "4) Do NOT output any chat-template tokens.\n"
        "\nRequired output format:\n"
        "<think>\n"
        "Because "
    )


def _is_natural_english_reasoning(think: str) -> bool:
    """Heuristic: reward 'human-like' short English reasoning."""
    t = (think or "").strip()
    if len(t) < 20:
        return False
    # Disallow angle-bracket tokens / tags
    if "<" in t or ">" in t:
        return False
    if "nan" in t.lower():
        return False
    # Require enough letters and at least a couple of words
    letters = sum(ch.isalpha() for ch in t)
    if letters < 15:
        return False
    words = re.findall(r"[A-Za-z]{2,}", t)
    if len(words) < 3:
        return False
    # Encourage sentence-like structure
    if not any(p in t for p in [".", ";", ","]):
        return False
    return True


def parse(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    think = ""
    after = t
    if "<think>" in t and "</think>" in t:
        before, after = t.split("</think>", 1)
        think = before.split("<think>", 1)[-1].strip()
        after = after.strip()
    elif "</think>" in t and "<think>" not in t:
        # If the prompt already contains the opening `<think>`, generations may start
        # directly with reasoning text and only include the closing `</think>`.
        before, after = t.split("</think>", 1)
        think = before.strip()
        after = after.strip()

    # Expect bare SID after </think>
    after_nospace = after.replace(" ", "")
    pred_sid = ""
    pred_sid_strict = ""
    pred_sid_parts: Optional[Tuple[int, int, int, int]] = None

    m_strict = SID_STRICT_PATTERN.search(after_nospace)
    if m_strict:
        pred_sid_strict = m_strict.group(0)
        pred_sid = pred_sid_strict
        m_parts = SID_STRICT_PARTS_PATTERN.search(pred_sid_strict)
        if m_parts:
            pred_sid_parts = tuple(int(x) for x in m_parts.groups())  # type: ignore[assignment]
    else:
        m = SID_PATTERN.search(after_nospace)
        if m:
            pred_sid = m.group(0)

    # legacy debug
    items = ITEM_PATTERN.findall(after)
    pred_item_raw = items[0].strip() if items else ""
    return {
        "think": think,
        "after": after,
        "items": items,
        "pred_item_raw": pred_item_raw,
        "pred_sid": pred_sid,
        "pred_sid_strict": pred_sid_strict,
        "pred_sid_parts": pred_sid_parts,
    }


def reward(sample: Sample, full_text: str) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    parsed = parse(full_text)
    text = (full_text or "")

    # When prompt includes `<think>`, generations may omit it; require `</think>`.
    think_ok = ("</think>" in text) and bool(parsed["think"]) and (len(parsed["think"]) >= 20)
    think_has_sid = (
        ("<|sid_begin|>" in parsed["think"]) or ("<s_a_" in parsed["think"]) or ("<s_b_" in parsed["think"]) or ("<s_c_" in parsed["think"]) or ("<s_d_" in parsed["think"])
    )
    think_nl_ok = _is_natural_english_reasoning(parsed["think"])
    sid_strict_ok = bool(parsed.get("pred_sid_strict"))
    sid_soft_ok = bool(parsed.get("pred_sid"))

    has_tool = ("<tool_call>" in text) or ("<tool_response>" in text)
    has_im = ("<|im_start|>" in text) or ("<|im_end|>" in text)

    has_item_wrappers = "<|item_begin|>" in text or "<|item_end|>" in text
    strict_ok = (not has_tool) and (not has_im) and (not has_item_wrappers)
    # Strictness: after `</think>` must be EXACTLY one strict SID, and nothing else.
    if sid_soft_ok:
        after = (parsed.get("after") or "").strip()
        after_compact = after.replace(" ", "").replace("\n", "")
        if not parsed.get("pred_sid_strict"):
            strict_ok = False
        elif after_compact != parsed["pred_sid_strict"]:
            strict_ok = False

    # Stronger shaping on reasoning quality so it doesn't get dominated by hit reward.
    r_think = 1.0 if think_ok else -1.0
    r_think_clean = 1.0 if (think_ok and (not think_has_sid) and ("<" not in parsed["think"]) and (">" not in parsed["think"])) else -1.0
    r_think_nl = 2.0 if think_nl_ok else -2.0
    # Reward shaping: strict SID gets full credit, loose SID gets small credit.
    if sid_strict_ok:
        r_sid = 1.0
    elif sid_soft_ok:
        r_sid = 0.2
    else:
        r_sid = -1.0
    r_strict = 1.0 if strict_ok else -1.0

    gt = (sample.groundtruth or "").strip().replace(" ", "")
    pred = (parsed.get("pred_sid_strict") or parsed.get("pred_sid") or "").strip().replace(" ", "")

    # Prefix hit shaping on strict SID parts:
    # - reward if s_a matches
    # - more reward if s_a+s_b matches
    # - more reward if s_a+s_b+s_c matches
    # - full reward if all 4 parts match
    gt_parts: Optional[Tuple[int, int, int, int]] = None
    m_gt = SID_STRICT_PARTS_PATTERN.search(gt)
    if m_gt:
        gt_parts = tuple(int(x) for x in m_gt.groups())  # type: ignore[assignment]
    pred_parts = parsed.get("pred_sid_parts")

    hit_a = 0.0
    hit_ab = 0.0
    hit_abc = 0.0
    hit_full = 0.0
    if gt_parts and pred_parts:
        if pred_parts[0] == gt_parts[0]:
            hit_a = 1.0
            if pred_parts[1] == gt_parts[1]:
                hit_ab = 1.0
                if pred_parts[2] == gt_parts[2]:
                    hit_abc = 1.0
                    if pred_parts[3] == gt_parts[3]:
                        hit_full = 1.0

    # Hit shaping (make exact hit much more valuable):
    # - only a: 0.5
    # - a+b: 1.0
    # - a+b+c: 2.0
    # - full: 10.0
    # Note: full implies a/ab/abc are also 1.0 (prefix ladder).
    r_hit = 0.5 * hit_a + 0.5 * hit_ab + 1.0 * hit_abc + 8.0 * hit_full

    total = r_think + r_think_clean + r_think_nl + r_sid + r_strict + r_hit
    breakdown = {
        "think": r_think,
        "think_clean": r_think_clean,
        "think_nl": r_think_nl,
        "sid_fmt": r_sid,
        "strict": r_strict,
        "hit": r_hit,
        "hit_a": hit_a,
        "hit_ab": hit_ab,
        "hit_abc": hit_abc,
        "hit_full": hit_full,
    }
    debug = {"parsed": parsed, "gt": gt, "pred": pred}
    return total, breakdown, debug


def rollout_transformers_iter(
    prompts: List[str],
    model_path: str,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int = 1,
    max_prompt_tokens: int = 1024,
    max_seq_len: int = 2048,
):
    import torch

    # If launched under `torchrun`, avoid `device_map=auto` (it can shard across all
    # visible GPUs and break PEFT adapter device placement).
    import os
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if ddp and torch.cuda.is_available():
        # One process per GPU; pin this worker to its local GPU.
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        device_map = "auto" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok, model = load_tokenizer_and_model(model_path, device_map=device_map, dtype=dtype, trainable=False)
    if torch.cuda.is_available() and (not ddp) and device_map == "cpu":
        model = model.to(torch.device("cuda", 0))

    # Qwen3-style tokenizers often set eos_token_id == <|im_end|>. We want to:
    #   1) avoid stopping at <|im_end|>
    #   2) allow banning <|im_end|> via bad_words_ids
    # So we use <|endoftext|> (if present) as pad/eos for generation.
    eot_id = tok.convert_tokens_to_ids("<|endoftext|>")
    unk_id = getattr(tok, "unk_token_id", None)
    if isinstance(eot_id, int) and eot_id >= 0 and (unk_id is None or eot_id != unk_id):
        tok.pad_token_id = int(eot_id)
    elif tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Ensure item wrapper tokens exist for GRPO format training.
    added = tok.add_special_tokens({"additional_special_tokens": ["<|item_begin|>", "<|item_end|>"]})
    if added > 0:
        try:
            model.resize_token_embeddings(len(tok))
        except Exception:
            pass
    model.eval()

    # Prevent the model from emitting Qwen chat tokens like `<|im_end|>` / `<|im_start|>`.
    # Qwen3 sets eos_token == `<|im_end|>`, so we override eos_token_id below and also ban it.
    bad_ids: List[List[int]] = []
    for t in ["<|im_end|>", "<|im_start|>", "<|item_begin|>", "<|item_end|>"]:
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            bad_ids.append([int(tid)])
    bad_words_ids = bad_ids or None

    eos_id = int(tok.pad_token_id)

    # Batch decoding works best with left padding (stable prompt length inside each batch).
    old_padding_side = getattr(tok, "padding_side", "right")
    old_trunc_side = getattr(tok, "truncation_side", "right")
    tok.padding_side = "left"
    tok.truncation_side = "left"  # keep the completion tail when truncating long prompts

    # Build token-id sets for constrained SID decoding in stage-2.
    def _tid(token: str) -> Optional[int]:
        v = tok.convert_tokens_to_ids(token)
        if v is None:
            return None
        try:
            v = int(v)
        except Exception:
            return None
        # Some tokenizers have no unk_token_id (None).
        if getattr(tok, "unk_token_id", None) is not None and v == tok.unk_token_id:
            return None
        return v if v >= 0 else None

    sid_begin_id = _tid("<|sid_begin|>")
    sid_end_id = _tid("<|sid_end|>")

    def _range_ids(prefix: str, max_range: int = 256) -> List[int]:
        out: List[int] = []
        for i in range(max_range):
            tid = _tid(f"<{prefix}_{i}>")
            if tid is not None:
                out.append(int(tid))
        return out

    s_a_ids = _range_ids("s_a")
    s_b_ids = _range_ids("s_b")
    s_c_ids = _range_ids("s_c")
    s_d_ids = _range_ids("s_d")

    # Stage-1: ban any angle-bracket tokens and all SID-related tokens.
    # We will append `</think>` ourselves and then generate SID with constraints.
    banned_stage1: List[int] = []
    for ch in ["<", ">"]:
        for tid in tok.encode(ch, add_special_tokens=False):
            if isinstance(tid, int) and tid >= 0:
                banned_stage1.append(int(tid))
    for tid in [sid_begin_id, sid_end_id]:
        if tid is not None:
            banned_stage1.append(int(tid))
    banned_stage1.extend(s_a_ids)
    banned_stage1.extend(s_b_ids)
    banned_stage1.extend(s_c_ids)
    banned_stage1.extend(s_d_ids)
    # Also ban chat tokens if present.
    for t in ["<|im_end|>", "<|im_start|>", "<|item_begin|>", "<|item_end|>"]:
        tid = _tid(t)
        if tid is not None:
            banned_stage1.append(int(tid))

    # Deduplicate
    banned_stage1 = sorted(set(banned_stage1))
    bad_words_stage1: List[List[int]] = [[i] for i in banned_stage1] if banned_stage1 else []
    # Also ban common tag strings as sequences (tokenizer may encode them without standalone '<' token).
    for s in ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]:
        ids = tok.encode(s, add_special_tokens=False)
        if ids and all(isinstance(i, int) and i >= 0 for i in ids):
            bad_words_stage1.append([int(i) for i in ids])
    # Ban the placeholder/missing-value token often produced by the model.
    for s in ["nan", "NaN"]:
        ids = tok.encode(s, add_special_tokens=False)
        if ids and all(isinstance(i, int) and i >= 0 for i in ids):
            bad_words_stage1.append([int(i) for i in ids])
    bad_words_stage1 = bad_words_stage1 or None

    # Stage-2: constrain decoding to exactly one strict SID.
    if sid_begin_id is None or sid_end_id is None or not (s_a_ids and s_b_ids and s_c_ids and s_d_ids):
        raise RuntimeError("Tokenizer missing SID special tokens; cannot run constrained SID decoding.")

    bs = max(1, int(batch_size))
    nn = max(1, int(n))
    for start in range(0, len(prompts), bs):
        ps = prompts[start : start + bs]
        if not ps:
            continue

        # -------- stage 1 (batched): generate natural-language reasoning
        enc0 = tok(
            ps,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_prompt_tokens),
        )
        enc0 = {k: v.to(model.device) for k, v in enc0.items()}
        prompt_len0 = int(enc0["input_ids"].shape[1])
        with torch.inference_mode():
            out1 = model.generate(
                **enc0,
                max_new_tokens=min(int(max_new_tokens), 96),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                num_beams=1,
                min_new_tokens=16,
                bad_words_ids=bad_words_stage1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_id,
                num_return_sequences=nn,
            )

        think_txts: List[str] = []
        prompt2_list: List[str] = []
        for i, p in enumerate(ps):
            for j in range(nn):
                idx = i * nn + j
                gen1_ids = out1[idx][prompt_len0:]
                think_txt = tok.decode(gen1_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                think_txt = (think_txt or "")
                think_txt = think_txt.replace("\r", " ").replace("\t", " ").replace("\n", " ")
                think_txt = re.sub(r"\s+", " ", think_txt).strip()
                if len(think_txt) > 220:
                    think_txt = think_txt[:220].rstrip()
                parts = re.split(r"(?<=[.!?])\s+", think_txt)
                if len(parts) >= 2:
                    think_txt = (parts[0] + " " + parts[1]).strip()
                think_txts.append(think_txt)
                prompt2_list.append(p + think_txt + "\n</think>\n")

        # -------- stage 2 (batched): force exactly one strict SID after </think>
        enc2 = tok(
            prompt2_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_seq_len),
        )
        enc2 = {k: v.to(model.device) for k, v in enc2.items()}
        prompt_len2 = int(enc2["input_ids"].shape[1])

        def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor):
            gen_pos = int(input_ids.shape[-1] - prompt_len2)
            if gen_pos <= 0:
                return [int(sid_begin_id)]
            if gen_pos == 1:
                return s_a_ids
            if gen_pos == 2:
                return s_b_ids
            if gen_pos == 3:
                return s_c_ids
            if gen_pos == 4:
                return s_d_ids
            if gen_pos == 5:
                return [int(sid_end_id)]
            return [eos_id]

        with torch.inference_mode():
            out2 = model.generate(
                **enc2,
                max_new_tokens=6,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=int(sid_end_id),
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )

        cand_batch: List[List[str]] = [[] for _ in range(len(ps))]
        for i in range(len(ps)):
            for j in range(nn):
                idx = i * nn + j
                gen2_ids = out2[idx][prompt_len2:]
                sid_txt = tok.decode(gen2_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                sid_txt = (sid_txt or "").strip().replace(" ", "").replace("\n", "")
                txt = (think_txts[idx].strip() + "\n</think>\n" + sid_txt).strip()
                cand_batch[i].append(txt)

        yield start, cand_batch

    tok.padding_side = old_padding_side
    tok.truncation_side = old_trunc_side
    return


def rollout_transformers(
    prompts: List[str],
    model_path: str,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int = 1,
    max_prompt_tokens: int = 1024,
    max_seq_len: int = 2048,
) -> List[List[str]]:
    grouped: List[List[str]] = []
    for _, cand_batch in rollout_transformers_iter(
        prompts=prompts,
        model_path=model_path,
        n=n,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        max_prompt_tokens=max_prompt_tokens,
        max_seq_len=max_seq_len,
    ):
        grouped.extend(cand_batch)
    return grouped


def generate_candidates_from_model(
    *,
    tok,
    model,
    prompts: List[str],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int = 1,
    max_prompt_tokens: int = 1024,
    max_seq_len: int = 2048,
) -> List[List[str]]:
    """Generate candidates using an already-loaded model/tokenizer.

    This avoids precomputing rollouts for every example when training on the full
    parquet; instead, each optimizer step samples from the full dataset and
    generates rollouts on-the-fly.

    IMPORTANT: this function must NOT mutate tokenizer vocab / model embeddings
    (e.g. add_special_tokens/resize_token_embeddings), because it is used after
    DDP wraps the policy.
    """
    import torch

    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()

        # Ensure we have a pad id for generation.
        if tok.pad_token_id is None:
            eot_id = tok.convert_tokens_to_ids("<|endoftext|>")
            unk_id = getattr(tok, "unk_token_id", None)
            if isinstance(eot_id, int) and eot_id >= 0 and (unk_id is None or eot_id != unk_id):
                tok.pad_token_id = int(eot_id)
            else:
                tok.pad_token_id = tok.eos_token_id

        eos_id = int(tok.pad_token_id)

        # Batch decoding works best with left padding.
        old_padding_side = getattr(tok, "padding_side", "right")
        old_trunc_side = getattr(tok, "truncation_side", "right")
        try:
            tok.padding_side = "left"
            tok.truncation_side = "left"
        except Exception:
            pass

        def _tid(token: str) -> Optional[int]:
            v = tok.convert_tokens_to_ids(token)
            if v is None:
                return None
            try:
                v = int(v)
            except Exception:
                return None
            if getattr(tok, "unk_token_id", None) is not None and v == tok.unk_token_id:
                return None
            return v if v >= 0 else None

        sid_begin_id = _tid("<|sid_begin|>")
        sid_end_id = _tid("<|sid_end|>")

        def _range_ids(prefix: str, max_range: int = 256) -> List[int]:
            out: List[int] = []
            for i in range(max_range):
                tid = _tid(f"<{prefix}_{i}>")
                if tid is not None:
                    out.append(int(tid))
            return out

        s_a_ids = _range_ids("s_a")
        s_b_ids = _range_ids("s_b")
        s_c_ids = _range_ids("s_c")
        s_d_ids = _range_ids("s_d")

        # Stage-1: ban angle brackets and SID tokens.
        banned_stage1: List[int] = []
        for ch in ["<", ">"]:
            for tid in tok.encode(ch, add_special_tokens=False):
                if isinstance(tid, int) and tid >= 0:
                    banned_stage1.append(int(tid))
        for tid in [sid_begin_id, sid_end_id]:
            if tid is not None:
                banned_stage1.append(int(tid))
        banned_stage1.extend(s_a_ids)
        banned_stage1.extend(s_b_ids)
        banned_stage1.extend(s_c_ids)
        banned_stage1.extend(s_d_ids)
        for t in ["<|im_end|>", "<|im_start|>", "<|item_begin|>", "<|item_end|>"]:
            tid = _tid(t)
            if tid is not None:
                banned_stage1.append(int(tid))
        banned_stage1 = sorted(set(banned_stage1))
        bad_words_stage1: List[List[int]] = [[i] for i in banned_stage1] if banned_stage1 else []
        for s in ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]:
            ids = tok.encode(s, add_special_tokens=False)
            if ids and all(isinstance(i, int) and i >= 0 for i in ids):
                bad_words_stage1.append([int(i) for i in ids])
        for s in ["nan", "NaN"]:
            ids = tok.encode(s, add_special_tokens=False)
            if ids and all(isinstance(i, int) and i >= 0 for i in ids):
                bad_words_stage1.append([int(i) for i in ids])
        bad_words_stage1 = bad_words_stage1 or None

        if sid_begin_id is None or sid_end_id is None or not (s_a_ids and s_b_ids and s_c_ids and s_d_ids):
            raise RuntimeError("Tokenizer missing SID special tokens; cannot run constrained SID decoding.")

        bs = max(1, int(batch_size))
        nn = max(1, int(n))
        grouped: List[List[str]] = []
        for start in range(0, len(prompts), bs):
            ps = prompts[start : start + bs]
            if not ps:
                continue

            enc0 = tok(
                ps,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_prompt_tokens),
            )
            enc0 = {k: v.to(model.device) for k, v in enc0.items()}
            prompt_len0 = int(enc0["input_ids"].shape[1])

            with torch.inference_mode():
                out1 = model.generate(
                    **enc0,
                    max_new_tokens=min(int(max_new_tokens), 96),
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    num_beams=1,
                    min_new_tokens=16,
                    bad_words_ids=bad_words_stage1,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=eos_id,
                    num_return_sequences=nn,
                )

            think_txts: List[str] = []
            prompt2_list: List[str] = []
            for i, p in enumerate(ps):
                for j in range(nn):
                    idx = i * nn + j
                    gen1_ids = out1[idx][prompt_len0:]
                    think_txt = tok.decode(gen1_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    think_txt = (think_txt or "")
                    think_txt = think_txt.replace("\r", " ").replace("\t", " ").replace("\n", " ")
                    think_txt = re.sub(r"\s+", " ", think_txt).strip()
                    if len(think_txt) > 220:
                        think_txt = think_txt[:220].rstrip()
                    parts = re.split(r"(?<=[.!?])\s+", think_txt)
                    if len(parts) >= 2:
                        think_txt = (parts[0] + " " + parts[1]).strip()
                    think_txts.append(think_txt)
                    prompt2_list.append(p + think_txt + "\n</think>\n")

            enc2 = tok(
                prompt2_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_seq_len),
            )
            enc2 = {k: v.to(model.device) for k, v in enc2.items()}
            prompt_len2 = int(enc2["input_ids"].shape[1])

            def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor):
                gen_pos = int(input_ids.shape[-1] - prompt_len2)
                if gen_pos <= 0:
                    return [int(sid_begin_id)]
                if gen_pos == 1:
                    return s_a_ids
                if gen_pos == 2:
                    return s_b_ids
                if gen_pos == 3:
                    return s_c_ids
                if gen_pos == 4:
                    return s_d_ids
                if gen_pos == 5:
                    return [int(sid_end_id)]
                return [eos_id]

            with torch.inference_mode():
                out2 = model.generate(
                    **enc2,
                    max_new_tokens=6,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    num_beams=1,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=int(sid_end_id),
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                )

            cand_batch: List[List[str]] = [[] for _ in range(len(ps))]
            for i in range(len(ps)):
                for j in range(nn):
                    idx = i * nn + j
                    gen2_ids = out2[idx][prompt_len2:]
                    sid_txt = tok.decode(gen2_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    sid_txt = (sid_txt or "").strip().replace(" ", "").replace("\n", "")
                    txt = (think_txts[idx].strip() + "\n</think>\n" + sid_txt).strip()
                    cand_batch[i].append(txt)
            grouped.extend(cand_batch)

        try:
            tok.padding_side = old_padding_side
            tok.truncation_side = old_trunc_side
        except Exception:
            pass
        return grouped
    finally:
        try:
            if was_training:
                model.train()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry_run", "train", "eval"], default="dry_run")
    ap.add_argument("--data_path", default=str(Path(__file__).resolve().parents[2] / "data" / "training_RA_train.parquet"))
    ap.add_argument(
        "--eval_data_path",
        default=str(Path(__file__).resolve().parents[2] / "data" / "training_RA_test.parquet"),
        help="Evaluation parquet path (default: training_RA_test.parquet)",
    )
    ap.add_argument(
        "--eval_num_samples",
        type=int,
        default=0,
        help="How many eval samples to use; <=0 means full eval parquet",
    )
    ap.add_argument(
        "--eval_save_examples",
        type=int,
        default=20,
        help="How many rollout examples (JSONL) to save per-rank during eval.",
    )
    # Default to an existing SFT ReasoningActivation checkpoint, so rollouts already
    # contain `<think>...</think>` + a single SID (better GRPO signal).
    ap.add_argument(
        "--model_name_or_path",
        default=str(Path(__file__).resolve().parents[1] / "results" / "ReasoningActivation" / "epoch_3" / "checkpoint-2796"),
    )
    ap.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / "grpo_runs" / f"ra_{int(time.time())}"))
    ap.add_argument("--num_samples", type=int, default=32, help="How many samples to use from data_path; <=0 means full dataset")
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=1024,
        help="Truncate the (potentially long) prompt to this many tokens for rollout stage-1 encoding.",
    )
    ap.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Truncate prompt+completion sequences to this many tokens for stage-2 / training logp.",
    )
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument(
        "--train_batch_rollouts",
        type=int,
        default=1,
        help="Number of rollout records per optimizer step (increases per-step compute).",
    )
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta_kl", type=float, default=0.01)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--rollout_batch_size",
        type=int,
        default=4,
        help="Batch size for rollout generation per rank (reduces Python overhead, improves GPU util).",
    )
    ap.add_argument(
        "--rollout_mode",
        choices=["precompute", "online"],
        default="precompute",
        help="Train mode rollout source. 'precompute' generates for all samples and writes rollouts.jsonl; 'online' samples from dataset each step.",
    )
    ap.add_argument(
        "--logp_batch_size",
        type=int,
        default=4,
        help="Micro-batch size for seq_logp() forward (reduces peak VRAM when num_generations is large).",
    )
    args = ap.parse_args()

    # -------- torch.distributed (DDP) support --------
    # Use `torchrun --nproc_per_node=4 ...` to utilize GPU0-3.
    # We only parallelize the TRAIN loop; rollout generation is done by rank0 then broadcast via filesystem.
    import os
    import torch

    # Better throughput on Ampere+ (safe for BF16 training).
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world_size > 1

    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")

        # Per-rank device sanity logging (helps debug "only GPU0 is computing").
        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
            dev = torch.device("cuda", local_rank)
            print(
                f"[ddp] rank={rank}/{world_size} local_rank={local_rank} "
                f"cuda_visible={visible} current_device={torch.cuda.current_device()} "
                f"device_name={torch.cuda.get_device_name(dev)}",
                flush=True,
            )
        except Exception:
            print(
                f"[ddp] rank={rank}/{world_size} local_rank={local_rank} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}",
                flush=True,
            )

    def r0_print(*a, **k):
        if (not ddp) or rank == 0:
            print(*a, **k)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"

    # ---------------- eval mode ----------------
    if args.mode == "eval":
        eval_path = Path(str(args.eval_data_path))
        samples_all = load_parquet(eval_path, limit=int(args.eval_num_samples))
        r0_print(f"eval_loaded_samples={len(samples_all)} eval_data_path={eval_path}")

        # Shard across ranks
        if ddp:
            idxs = [i for i in range(len(samples_all)) if (i % world_size) == rank]
        else:
            idxs = list(range(len(samples_all)))
        sub_samples = [samples_all[i] for i in idxs]
        sub_prompts = [build_prompt(s.description) for s in sub_samples]

        device_map = {"": local_rank} if (ddp and torch.cuda.is_available()) else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tok, model = load_tokenizer_and_model(str(args.model_name_or_path), device_map=device_map, dtype=dtype, trainable=False)
        if torch.cuda.is_available() and (not ddp):
            model = model.to(torch.device("cuda", 0))
        model.eval()

        # Local accumulators
        n_local = 0
        reward_sum = 0.0
        reward_min = 1e9
        reward_max = -1e9
        think_ok = 0
        sid_fmt_ok = 0
        strict_ok = 0
        hit_a = 0.0
        hit_ab = 0.0
        hit_abc = 0.0
        hit_full = 0.0

        ex_left = max(0, int(args.eval_save_examples))
        ex_path = out_dir / f"eval_rollouts_rank{rank}.jsonl"
        if ex_left > 0:
            ex_f = ex_path.open("w", encoding="utf-8")
        else:
            ex_f = None

        bs = max(1, int(args.rollout_batch_size))
        nn = max(1, int(args.num_generations))
        with torch.inference_mode():
            for start in range(0, len(sub_prompts), bs):
                if rank == 0 and (start % (bs * 64) == 0):
                    done = int(start)
                    total = int(len(sub_prompts))
                    # global estimate (approx, shards may be imbalanced by <= world_size)
                    est_global_done = done * (world_size if ddp else 1)
                    r0_print(f"[eval] progress_local={done}/{total} est_global_done~{est_global_done}/{len(samples_all)}", flush=True)
                ps = sub_prompts[start : start + bs]
                ss = sub_samples[start : start + bs]
                if not ps:
                    continue
                cand_groups = generate_candidates_from_model(
                    tok=tok,
                    model=model,
                    prompts=ps,
                    n=nn,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    batch_size=len(ps),
                    max_prompt_tokens=int(args.max_prompt_tokens),
                    max_seq_len=int(args.max_seq_len),
                )

                for j, cands in enumerate(cand_groups):
                    s = ss[j]
                    # pick best by reward among N candidates
                    scored = []
                    for t in cands:
                        r, br, dbg = reward(s, t)
                        scored.append((float(r), t, br, dbg))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    best_r, best_t, best_br, best_dbg = scored[0]

                    n_local += 1
                    reward_sum += float(best_r)
                    reward_min = min(reward_min, float(best_r))
                    reward_max = max(reward_max, float(best_r))
                    think_ok += 1 if float(best_br.get("think", 0.0)) > 0 else 0
                    sid_fmt_ok += 1 if float(best_br.get("sid_fmt", 0.0)) > 0 else 0
                    strict_ok += 1 if float(best_br.get("strict", 0.0)) > 0 else 0
                    hit_a += float(best_br.get("hit_a", 0.0))
                    hit_ab += float(best_br.get("hit_ab", 0.0))
                    hit_abc += float(best_br.get("hit_abc", 0.0))
                    hit_full += float(best_br.get("hit_full", 0.0))

                    if ex_f is not None and ex_left > 0:
                        obj = {
                            "global_idx": int(idxs[start + j]),
                            "user_id": s.user_id,
                            "groundtruth": s.groundtruth,
                            "prompt": ps[j],
                            "best": {
                                "text": best_t,
                                "reward": float(best_r),
                                "reward_breakdown": best_br,
                                "debug": best_dbg,
                            },
                        }
                        ex_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        ex_left -= 1

        if ex_f is not None:
            ex_f.close()

        # Reduce metrics
        if ddp:
            dev = torch.device("cuda", local_rank)
            t_sum = torch.tensor(
                [
                    float(n_local),
                    float(reward_sum),
                    float(think_ok),
                    float(sid_fmt_ok),
                    float(strict_ok),
                    float(hit_a),
                    float(hit_ab),
                    float(hit_abc),
                    float(hit_full),
                ],
                device=dev,
                dtype=torch.float32,
            )
            torch.distributed.all_reduce(t_sum, op=torch.distributed.ReduceOp.SUM)
            t_min = torch.tensor([float(reward_min)], device=dev, dtype=torch.float32)
            t_max = torch.tensor([float(reward_max)], device=dev, dtype=torch.float32)
            torch.distributed.all_reduce(t_min, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(t_max, op=torch.distributed.ReduceOp.MAX)
            if rank == 0:
                n = int(t_sum[0].item())
                rsum = float(t_sum[1].item())
                think = float(t_sum[2].item())
                sidfmt = float(t_sum[3].item())
                strict = float(t_sum[4].item())
                ha = float(t_sum[5].item())
                hab = float(t_sum[6].item())
                habc = float(t_sum[7].item())
                hfull = float(t_sum[8].item())
                r0_print(
                    f"eval_n={n} reward_mean={rsum/max(n,1):.6f} reward_min={float(t_min.item()):.3f} reward_max={float(t_max.item()):.3f} "
                    f"think_rate={think/max(n,1):.4f} sid_fmt_rate={sidfmt/max(n,1):.4f} strict_rate={strict/max(n,1):.4f} "
                    f"hit_a_rate={ha/max(n,1):.4f} hit_ab_rate={hab/max(n,1):.4f} hit_abc_rate={habc/max(n,1):.4f} hit_full_rate={hfull/max(n,1):.4f}",
                    flush=True,
                )
        else:
            n = max(1, int(n_local))
            r0_print(
                f"eval_n={n_local} reward_mean={reward_sum/n:.6f} reward_min={reward_min:.3f} reward_max={reward_max:.3f} "
                f"think_rate={think_ok/n:.4f} sid_fmt_rate={sid_fmt_ok/n:.4f} strict_rate={strict_ok/n:.4f} "
                f"hit_a_rate={hit_a/n:.4f} hit_ab_rate={hit_ab/n:.4f} hit_abc_rate={hit_abc/n:.4f} hit_full_rate={hit_full/n:.4f}",
                flush=True,
            )

        if ddp:
            torch.distributed.destroy_process_group()
        return 0

    # Rollout generation
    # - dry_run: always single-process
    # - train + DDP: shard rollouts across ranks to utilize GPU0-3, then merge on rank0
    if args.mode == "dry_run":
        samples = load_parquet(Path(args.data_path), limit=int(args.num_samples))
        r0_print(f"loaded_samples={len(samples)} data_path={args.data_path}")
        prompts = [build_prompt(s.description) for s in samples]

        # Stream rollouts to disk (avoid holding all generations in RAM for large datasets)
        with rollouts_path.open("w", encoding="utf-8") as f:
            for start, cand_batch in rollout_transformers_iter(
                prompts=prompts,
                model_path=str(args.model_name_or_path),
                n=int(args.num_generations),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                batch_size=int(args.rollout_batch_size),
                max_prompt_tokens=int(args.max_prompt_tokens),
                max_seq_len=int(args.max_seq_len),
            ):
                for j, cand in enumerate(cand_batch):
                    i = start + j
                    s = samples[i]
                    p = prompts[i]
                    scored = []
                    for t in cand:
                        r, br, dbg = reward(s, t)
                        scored.append({"text": t, "reward": r, "reward_breakdown": br, "debug": dbg})
                    scored.sort(key=lambda x: x["reward"], reverse=True)
                    f.write(
                        json.dumps(
                            {
                                "id": f"ra-{i:06d}",
                                "created_at": int(time.time()),
                                "user_id": s.user_id,
                                "description": s.description,
                                "groundtruth": s.groundtruth,
                                "prompt": p,
                                "candidates": scored,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    else:
        # train
        samples_all = load_parquet(Path(args.data_path), limit=int(args.num_samples))
        r0_print(f"loaded_samples={len(samples_all)} data_path={args.data_path}")
        prompts_all = [build_prompt(s.description) for s in samples_all]

        rollout_mode = str(getattr(args, "rollout_mode", "precompute"))
        # For full parquet, online rollouts are much faster than precomputing every example.
        if int(args.num_samples) <= 0 and rollout_mode == "precompute":
            rollout_mode = "online"
        r0_print(f"[rollout] mode={rollout_mode}")

        if rollout_mode == "precompute" and ddp:
            idxs = [i for i in range(len(samples_all)) if (i % world_size) == rank]
            part_path = out_dir / f"rollouts.part_rank{rank}.jsonl"
            sub_prompts = [prompts_all[i] for i in idxs]
            sub_samples = [samples_all[i] for i in idxs]
            # Stream per-rank rollouts to disk to avoid large RAM usage.
            with part_path.open("w", encoding="utf-8") as f:
                for start, cand_batch in rollout_transformers_iter(
                    prompts=sub_prompts,
                    model_path=str(args.model_name_or_path),
                    n=int(args.num_generations),
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    batch_size=int(args.rollout_batch_size),
                    max_prompt_tokens=int(args.max_prompt_tokens),
                    max_seq_len=int(args.max_seq_len),
                ):
                    for j, cand in enumerate(cand_batch):
                        off = start + j
                        i = idxs[off]
                        s = sub_samples[off]
                        p = sub_prompts[off]
                        scored = []
                        for t in cand:
                            r, br, dbg = reward(s, t)
                            scored.append({"text": t, "reward": r, "reward_breakdown": br, "debug": dbg})
                        scored.sort(key=lambda x: x["reward"], reverse=True)
                        f.write(
                            json.dumps(
                                {
                                    "id": f"ra-{i:06d}",
                                    "created_at": int(time.time()),
                                    "user_id": s.user_id,
                                    "description": s.description,
                                    "groundtruth": s.groundtruth,
                                    "prompt": p,
                                    "candidates": scored,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            # NOTE: Avoid NCCL barriers here.
            # During rollout generation, GPU memory can be temporarily high (KV cache), and
            # the *first* NCCL collective (e.g., barrier) may crash hard (no Python traceback),
            # causing other ranks to fail with TCPStore/NCCL setup errors.
            # We synchronize via filesystem instead: wait for all shard files, merge on rank0,
            # then wait for a marker.
            marker = out_dir / "rollouts.merged.ok"
            if rank == 0:
                # Wait for all shards.
                for r in range(world_size):
                    pp = out_dir / f"rollouts.part_rank{r}.jsonl"
                    while not pp.exists():
                        time.sleep(0.2)

                # Merge
                # IMPORTANT: avoid loading the full dataset into RAM on rank0.
                # For full parquet (22k+ samples), materializing all JSON objects can OOM
                # rank0 and crash NCCL/DDP later. Concatenate shard files streaming.
                with rollouts_path.open("w", encoding="utf-8") as out_f:
                    for r in range(world_size):
                        pp = out_dir / f"rollouts.part_rank{r}.jsonl"
                        with pp.open("r", encoding="utf-8") as in_f:
                            for line in in_f:
                                if line.strip():
                                    out_f.write(line if line.endswith("\n") else (line + "\n"))
                marker.write_text(str(int(time.time())), encoding="utf-8")
            else:
                while not marker.exists():
                    time.sleep(0.2)
        elif rollout_mode == "precompute":
            with rollouts_path.open("w", encoding="utf-8") as f:
                for start, cand_batch in rollout_transformers_iter(
                    prompts=prompts_all,
                    model_path=str(args.model_name_or_path),
                    n=int(args.num_generations),
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    batch_size=int(args.rollout_batch_size),
                    max_prompt_tokens=int(args.max_prompt_tokens),
                    max_seq_len=int(args.max_seq_len),
                ):
                    for j, cand in enumerate(cand_batch):
                        i = start + j
                        s = samples_all[i]
                        p = prompts_all[i]
                        scored = []
                        for t in cand:
                            r, br, dbg = reward(s, t)
                            scored.append({"text": t, "reward": r, "reward_breakdown": br, "debug": dbg})
                        scored.sort(key=lambda x: x["reward"], reverse=True)
                        f.write(
                            json.dumps(
                                {
                                    "id": f"ra-{i:06d}",
                                    "created_at": int(time.time()),
                                    "user_id": s.user_id,
                                    "description": s.description,
                                    "groundtruth": s.groundtruth,
                                    "prompt": p,
                                    "candidates": scored,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        else:
            # online rollouts: do not precompute rollouts.jsonl
            pass

    if args.mode == "dry_run":
        lines = rollouts_path.read_text(encoding="utf-8").splitlines()
        for i in range(min(3, len(lines))):
            rec = json.loads(lines[i])
            top = rec["candidates"][0]
            print(f"[{i}] best_reward={top['reward']:.3f} gt={rec['groundtruth']}")
            print(top["text"].strip())
            print("-")
        print(f"Saved rollouts: {rollouts_path}")
        return 0

    # Rollout generation can temporarily reserve a lot of CUDA memory (KV cache).
    # Free cached blocks before building the training graphs to reduce OOM risk.
    if torch.cuda.is_available():
        import gc

        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

    # Training loop: minimal GRPO update
    from peft import LoraConfig, get_peft_model
    try:
        from peft import PeftModel
    except Exception:  # pragma: no cover
        PeftModel = tuple()  # type: ignore

    random.seed(0 + int(rank))
    shard_idxs: List[int] = []
    roll_offsets: List[int] = []
    if rollout_mode == "precompute":
        # Avoid loading the entire rollouts.jsonl into memory (can be very large for full dataset).
        # Instead build a lightweight byte-offset index for random sampling.
        with rollouts_path.open("rb") as f:
            pos = f.tell()
            line = f.readline()
            while line:
                if line.strip():
                    roll_offsets.append(int(pos))
                pos = f.tell()
                line = f.readline()
        if not roll_offsets:
            raise SystemExit("No rollouts found.")
    else:
        # Online rollouts sample from the full dataset.
        if ddp:
            shard_idxs = [i for i in range(len(samples_all)) if (i % world_size) == rank]
        else:
            shard_idxs = list(range(len(samples_all)))
        if not shard_idxs:
            raise SystemExit("Empty shard for online rollouts.")

    device_map = {"": local_rank} if (ddp and torch.cuda.is_available()) else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok, policy = load_tokenizer_and_model(str(args.model_name_or_path), device_map=device_map, dtype=dtype, trainable=True)
    if torch.cuda.is_available() and not ddp:
        policy = policy.to(torch.device("cuda", 0))
    policy.train()
    try:
        policy.config.use_cache = False
    except Exception:
        pass
    try:
        tok.padding_side = "left"
        tok.truncation_side = "left"
    except Exception:
        pass
    try:
        policy.gradient_checkpointing_enable()
    except Exception:
        try:
            policy.get_base_model().gradient_checkpointing_enable()  # type: ignore[attr-defined]
        except Exception:
            pass

    if args.use_lora:
        if isinstance(policy, PeftModel):
            # Stage-1 aligned checkpoints (beauty_align) are already PEFT adapters.
            # Avoid stacking a new LoRA adapter on top by default.
            print("[warn] policy is already a PEFT adapter; skip adding a new LoRA adapter.")
        else:
            lora_cfg = LoraConfig(
                r=int(args.lora_r),
                lora_alpha=int(args.lora_alpha),
                lora_dropout=float(args.lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
            )
            policy = get_peft_model(policy, lora_cfg)

    _, ref = load_tokenizer_and_model(str(args.model_name_or_path), device_map=device_map, dtype=dtype, trainable=False)
    if torch.cuda.is_available() and not ddp:
        ref = ref.to(torch.device("cuda", 0))
    ref.eval()
    try:
        ref.config.use_cache = False
    except Exception:
        pass

    if ddp:
        policy = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=float(args.lr))

    def seq_logp(m, input_ids: torch.Tensor, prompt_len: int, micro_batch_size: int) -> torch.Tensor:
        """Sequence log-prob of completion tokens.

        IMPORTANT: Avoid `log_softmax(logits)` which materializes a full
        `[B, T, V]` tensor and can easily OOM. Compute token logp as:
          logp(token) = logit(token) - logsumexp(logits)
        """
        mb = max(1, int(micro_batch_size))
        outs: List[torch.Tensor] = []
        # Micro-batch along batch dimension to avoid OOM when num_generations is large.
        for start in range(0, int(input_ids.shape[0]), mb):
            sub = input_ids[start : start + mb]
            out = m(input_ids=sub)
            logits = out.logits[:, :-1, :]  # [b, T-1, V]
            target = sub[:, 1:]  # [b, T-1]
            # Gather target logits without building full softmax/log_softmax.
            tgt_logits = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [b, T-1]
            logz = torch.logsumexp(logits, dim=-1)  # [b, T-1]
            token_logp = tgt_logits - logz
            comp = token_logp[:, prompt_len - 1 :]
            outs.append(comp.sum(dim=-1))
        return torch.cat(outs, dim=0) if outs else torch.empty((0,), device=input_ids.device)

    for step in range(1, int(args.steps) + 1):
        device = next(policy.parameters()).device
        k = max(1, int(args.train_batch_rollouts))
        batch_recs: List[Dict[str, Any]] = []
        if rollout_mode == "precompute":
            # Randomly sample K records by offset.
            with rollouts_path.open("rb") as f:
                for _ in range(k):
                    off = int(random.choice(roll_offsets))
                    f.seek(off)
                    raw = f.readline()
                    if not raw:
                        continue
                    try:
                        batch_recs.append(json.loads(raw.decode("utf-8")))
                    except Exception:
                        continue
        else:
            # Online: sample K dataset indices, generate rollouts from current policy.
            chosen = [int(random.choice(shard_idxs)) for _ in range(k)]
            batch_prompts = [prompts_all[i] for i in chosen]
            batch_samples = [samples_all[i] for i in chosen]

            # DDP wrapper does not expose generate(); use the underlying module.
            gen_model = policy.module if hasattr(policy, "module") else policy
            cand_groups = generate_candidates_from_model(
                tok=tok,
                model=gen_model,
                prompts=batch_prompts,
                n=int(args.num_generations),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                batch_size=int(args.rollout_batch_size),
                max_prompt_tokens=int(args.max_prompt_tokens),
                max_seq_len=int(args.max_seq_len),
            )

            for p, s, cands in zip(batch_prompts, batch_samples, cand_groups):
                scored = []
                for t in cands:
                    r, br, dbg = reward(s, t)
                    scored.append({"text": t, "reward": r, "reward_breakdown": br, "debug": dbg})
                scored.sort(key=lambda x: x["reward"], reverse=True)
                batch_recs.append({"prompt": p, "candidates": scored})

        if not batch_recs:
            raise SystemExit("No valid rollout records could be sampled/generated.")

        losses = []
        reward_means = []

        for rec in batch_recs:
            prompt = rec["prompt"]
            cands = rec["candidates"][: int(args.num_generations)]
            texts = [c["text"] for c in cands]
            rewards = torch.tensor([float(c["reward"]) for c in cands], device=device)
            adv = rewards - rewards.mean()
            if float(adv.std().item()) > 1e-6:
                adv = adv / (adv.std() + 1e-6)

            seqs = [prompt + t for t in texts]
            prompt_len = int(
                tok(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_seq_len))["input_ids"].shape[1]
            )
            enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=int(args.max_seq_len))
            input_ids = enc["input_ids"].to(device)

            lp_pol = seq_logp(policy, input_ids, prompt_len, int(args.logp_batch_size))
            with torch.no_grad():
                lp_ref = seq_logp(ref, input_ids, prompt_len, int(args.logp_batch_size))
            kl = (lp_pol - lp_ref)
            loss_i = -torch.mean(adv * lp_pol) + float(args.beta_kl) * torch.mean(kl)
            losses.append(loss_i)
            reward_means.append(rewards.mean())

        loss = torch.mean(torch.stack(losses))
        reward_mean = torch.mean(torch.stack(reward_means))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        r0_print(f"step={step} loss={float(loss.item()):.4f} reward_mean={float(reward_mean.item()):.3f} batch_rollouts={k}")

        if (not ddp or rank == 0) and (step % int(args.save_every) == 0):
            save_dir = out_dir / f"checkpoint_step_{step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            to_save = policy.module if hasattr(policy, "module") else policy
            to_save.save_pretrained(str(save_dir))
            tok.save_pretrained(str(save_dir))
            r0_print(f"saved: {save_dir}")

    if ddp:
        torch.distributed.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
