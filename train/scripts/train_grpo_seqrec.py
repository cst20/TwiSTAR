#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRPO scaffold for sequential recommendation (format + hit-rate reward).

Target behavior
  - Model produces reasoning inside `<think>...</think>`.
  - Final answer contains exactly one item wrapped by:
      <|item_begin|> ... <|item_end|>
  - Reward includes:
      1) format correctness (think + item wrapper)
      2) whether prediction hits the ground truth item

This script supports:
  - dry_run: rollout candidates, score reward, save rollouts.jsonl
  - train: minimal GRPO-style update (HF Transformers + PEFT LoRA)

Notes
  - Optional vLLM rollout (`--use_vllm`). If vLLM cannot load the model, it falls back.
  - Ground truth in provided parquet uses SID tokens: <|sid_begin|>...<|sid_end|>.
    We treat the *inner content* inside <|item_begin|> as the predicted item, and extract
    a SID if present; otherwise compare raw string.
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
# We use SID format as the primary output format reward.
ITEM_PATTERN = re.compile(r"<\|item_begin\|>(.*?)<\|item_end\|>", re.DOTALL)
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class SeqRecSample:
    user_id: str
    description: str
    groundtruth: str


def load_seqrec_samples(path: Path, limit: int) -> List[SeqRecSample]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.lower() == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        # expected columns: user_id, description, groundtruth
        out: List[SeqRecSample] = []
        for _, row in df.head(limit).iterrows():
            out.append(
                SeqRecSample(
                    user_id=str(row.get("user_id", "")),
                    description=str(row.get("description", "")),
                    groundtruth=str(row.get("groundtruth", "")),
                )
            )
        return out

    if path.suffix.lower() == ".jsonl":
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(out) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out.append(
                    SeqRecSample(
                        user_id=str(obj.get("user_id", "")),
                        description=str(obj.get("description", "")),
                        groundtruth=str(obj.get("groundtruth", "")),
                    )
                )
        return out

    # single JSON
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        return [
            SeqRecSample(
                user_id=str(obj.get("user_id", "")),
                description=str(obj.get("description", "")),
                groundtruth=str(obj.get("groundtruth", "")),
            )
        ]
    raise ValueError(f"Unsupported input format: {path}")


def build_prompt(description: str) -> str:
    # Keep it minimal and strict.
    return (
        "You are a professional recommendation expert. "
        "Predict the single most likely next item the user will purchase based on the purchase history.\n\n"
        f"{description.strip()}\n\n"
        "Strict output requirements (MUST follow):\n"
        "1) Put your reasoning ONLY inside <think>...</think>.\n"
        "2) After </think>, output EXACTLY ONE SID string in the exact format: <|sid_begin|>...<|sid_end|>.\n"
        "3) Do NOT output any other text, lists, tool calls, wrappers, or extra lines.\n"
        "   Forbidden substrings include: <tool_call>, <tool_response>, <|im_start|>, <|im_end|>, <|item_begin|>, <|item_end|>.\n"
        "\nRequired output format:\n"
        "<think>\n"
        "...\n"
        "</think>\n"
        "<|sid_begin|>...<|sid_end|>\n"
    )


def parse_generation(text: str) -> Dict[str, Any]:
    t = (text or "").strip()

    think = ""
    after = t
    if THINK_OPEN in t and THINK_CLOSE in t:
        before, after = t.split(THINK_CLOSE, 1)
        think = before.split(THINK_OPEN, 1)[-1].strip()
        after = after.strip()

    # We expect a bare SID string after </think>
    pred_sid = ""
    m = SID_PATTERN.search(after.replace(" ", ""))
    if m:
        pred_sid = m.group(0)

    # Keep legacy fields for debugging
    items = ITEM_PATTERN.findall(after)
    pred_item_raw = items[0].strip() if items else ""

    return {
        "think": think,
        "after": after,
        "items": items,
        "pred_item_raw": pred_item_raw,
        "pred_sid": pred_sid,
    }


def compute_reward(sample: SeqRecSample, full_text: str) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    parsed = parse_generation(full_text)
    think = parsed["think"]
    pred_sid = parsed["pred_sid"]

    text = (full_text or "")
    has_tool_call = "<tool_call>" in text or "<tool_response>" in text
    has_im_tokens = "<|im_start|>" in text or "<|im_end|>" in text
    has_item_wrappers = "<|item_begin|>" in text or "<|item_end|>" in text

    # 1) think format: MUST have <think>...</think> with non-empty reasoning
    think_ok = (THINK_OPEN in text) and (THINK_CLOSE in text) and bool(think) and (len(think) >= 20)
    r_think = 1.0 if think_ok else -1.0

    # 2) SID format: MUST have exactly one SID after </think>
    sid_ok = bool(pred_sid)
    r_sid_fmt = 1.0 if sid_ok else -1.0

    # 3) strictness: no extra tokens/tool calls outside required blocks
    # Allow only whitespace outside the think block and the item wrapper.
    strict_ok = True
    if has_tool_call or has_im_tokens or has_item_wrappers:
        strict_ok = False
    if sid_ok:
        # After removing think block and the first SID, remaining should be whitespace only.
        stripped = text
        if THINK_OPEN in stripped and THINK_CLOSE in stripped:
            stripped = stripped.split(THINK_CLOSE, 1)[-1]
        stripped = SID_PATTERN.sub("", stripped, count=1)
        if stripped.strip():
            strict_ok = False
    r_strict = 1.0 if strict_ok else -1.0

    # 4) hit ground truth
    gt = (sample.groundtruth or "").strip().replace(" ", "")
    pred_norm = pred_sid.strip().replace(" ", "")

    r_hit = 0.0
    if gt and pred_norm and pred_norm == gt:
        r_hit = 2.0

    total = r_think + r_sid_fmt + r_strict + r_hit
    breakdown = {"think": r_think, "sid_fmt": r_sid_fmt, "strict": r_strict, "hit": r_hit}
    debug = {"parsed": parsed, "gt": gt, "pred_norm": pred_norm}
    return total, breakdown, debug


def _try_vllm_rollout(
    prompts: List[str],
    model_path: str,
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    tensor_parallel_size: int,
) -> Optional[List[List[str]]]:
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=max(1, int(tensor_parallel_size)),
        )
        sp = SamplingParams(
            n=int(n),
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
        )
        outs = llm.generate(prompts, sp)
        grouped: List[List[str]] = []
        for req in outs:
            grouped.append([o.text for o in req.outputs])
        return grouped
    except Exception:
        return None


def rollout_generate(
    prompts: List[str],
    model_path: str,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_vllm: bool,
    tensor_parallel_size: int,
) -> List[List[str]]:
    if use_vllm:
        outs = _try_vllm_rollout(
            prompts=prompts,
            model_path=model_path,
            n=n,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            tensor_parallel_size=tensor_parallel_size,
        )
        if outs is not None:
            return outs

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Ensure item wrapper tokens exist for training the required format.
    added = tok.add_special_tokens({"additional_special_tokens": ["<|item_begin|>", "<|item_end|>"]})

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    if added > 0:
        model.resize_token_embeddings(len(tok))
    model.eval()

    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    bad_words_ids = [[int(im_end_id)]] if isinstance(im_end_id, int) and im_end_id >= 0 else None

    grouped: List[List[str]] = []
    for p in prompts:
        cand: List[str] = []
        for _ in range(int(n)):
            enc = tok(p, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    num_beams=1,
                    min_new_tokens=32,
                    bad_words_ids=bad_words_ids,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.pad_token_id,
                    use_cache=True,
                )
            gen_ids = out[0][enc["input_ids"].shape[1] :]
            txt = tok.decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
            cand.append(txt)
        grouped.append(cand)
    return grouped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry_run", "train"], default="dry_run")
    ap.add_argument(
        "--input_path",
        default=str(Path(__file__).resolve().parents[2] / "data" / "training_prediction_sid_data_val.parquet"),
    )
    ap.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parents[2] / "train" / "grpo_runs" / f"seqrec_{int(time.time())}"),
    )
    ap.add_argument(
        "--model_path",
        default=str(Path(__file__).resolve().parents[2] / "basemodel" / "Qwen3-1-7B-expand"),
    )
    ap.add_argument("--ref_model_path", default=None)
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)

    # training
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta_kl", type=float, default=0.01)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_every", type=int, default=20)

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"

    samples = load_seqrec_samples(Path(args.input_path), limit=int(args.num_samples))
    if not samples:
        raise SystemExit("No samples loaded.")

    prompts = [build_prompt(s.description) for s in samples]
    gens = rollout_generate(
        prompts=prompts,
        model_path=str(args.model_path),
        n=int(args.num_generations),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        use_vllm=bool(args.use_vllm),
        tensor_parallel_size=int(args.tensor_parallel_size),
    )

    # score + save
    with rollouts_path.open("w", encoding="utf-8") as f:
        for i, (s, p, cand) in enumerate(zip(samples, prompts, gens)):
            scored = []
            for t in cand:
                r, br, dbg = compute_reward(s, t)
                scored.append({"text": t, "reward": r, "reward_breakdown": br, "debug": dbg})
            scored.sort(key=lambda x: x["reward"], reverse=True)
            rec = {
                "id": f"seqrec-{i:06d}",
                "created_at": int(time.time()),
                "user_id": s.user_id,
                "description": s.description,
                "groundtruth": s.groundtruth,
                "prompt": p,
                "candidates": scored,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.mode == "dry_run":
        # print first few
        lines = rollouts_path.read_text(encoding="utf-8").splitlines()
        for i in range(min(3, len(lines))):
            rec = json.loads(lines[i])
            top = rec["candidates"][0]
            print(f"[{i}] best_reward={top['reward']:.3f} gt={rec['groundtruth']}")
            print(top["text"].strip())
            print("-")
        print(f"Saved rollouts: {rollouts_path}")
        return 0

    # train: minimal GRPO loop (Transformers + LoRA)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(0)
    roll_lines = rollouts_path.read_text(encoding="utf-8").splitlines()
    if not roll_lines:
        raise SystemExit("No rollouts to train on.")

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tok = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    policy = AutoModelForCausalLM.from_pretrained(str(args.model_path), torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
    policy.train()

    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora_cfg)

    ref_path = str(args.ref_model_path) if args.ref_model_path else str(args.model_path)
    ref = AutoModelForCausalLM.from_pretrained(ref_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
    ref.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=float(args.lr))

    def seq_logp(m, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        out = m(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        target = input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        comp = token_logp[:, prompt_len - 1 :]
        return comp.sum(dim=-1)

    for step in range(1, int(args.steps) + 1):
        rec = json.loads(random.choice(roll_lines))
        prompt = rec["prompt"]
        cands = rec["candidates"][: int(args.num_generations)]
        texts = [c["text"] for c in cands]
        rewards = torch.tensor([float(c["reward"]) for c in cands], device=policy.device)
        adv = rewards - rewards.mean()
        if float(adv.std().item()) > 1e-6:
            adv = adv / (adv.std() + 1e-6)

        seqs = [prompt + t for t in texts]
        prompt_len = int(tok(prompt, return_tensors="pt")["input_ids"].shape[1])
        enc = tok(seqs, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(policy.device)

        lp_pol = seq_logp(policy, input_ids, prompt_len)
        with torch.no_grad():
            lp_ref = seq_logp(ref, input_ids, prompt_len)
        kl = (lp_pol - lp_ref)
        loss = -torch.mean(adv * lp_pol) + float(args.beta_kl) * torch.mean(kl)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        print(f"step={step} loss={float(loss.item()):.4f} reward_mean={float(rewards.mean().item()):.3f}")

        if step % int(args.save_every) == 0:
            save_dir = out_dir / f"checkpoint_step_{step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(str(save_dir))
            tok.save_pretrained(str(save_dir))
            print(f"saved: {save_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
