#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal GRPO scaffold for I2I explanation with <think>.

Goal
  - Encourage reasoning inside <think>...</think> using reward optimization (GRPO-style).
  - Keep the final answer short and strictly formatted.

Notes
  - This script is designed to be a runnable *framework*:
    - `dry_run`: generate a few candidates + compute reward, write rollouts JSONL.
    - `train`: do a minimal GRPO update loop using HF Transformers + PEFT (LoRA).
  - vLLM/Verl integration is optional:
    - vLLM is used only for rollout if available/compatible; otherwise fallback to Transformers.
    - Verl is not required to run this file; if you have Verl installed you can adapt the
      `rollouts.jsonl` output as RLHF training data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RE_CJK = re.compile(r"[\u4e00-\u9fff]")


ALLOWED_LABELS = [
    "Functional/Need Consistency",
    "Category & Key Attribute Consistency",
    "Usage Scenario & Audience Consistency",
    "Collaborative Relation Type",
]


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _tokenize_words(text: str) -> List[str]:
    text = text.lower()
    parts = re.split(r"[^a-z0-9]+", text)
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "for",
        "with",
        "on",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "it",
        "as",
        "at",
        "from",
        "both",
        "item",
        "items",
    }
    return [p for p in parts if len(p) >= 3 and p not in stop]


def _ascii_printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if 32 <= ord(ch) <= 126 or ch in "\n\t")
    return printable / max(1, len(text))


@dataclass
class PairSample:
    a_title: str
    a_category: str
    a_description: str
    b_title: str
    b_category: str
    b_description: str
    meta: Dict[str, Any]


def load_pairs(path: Path) -> List[PairSample]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        pairs: List[PairSample] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pairs.append(
                    PairSample(
                        a_title=str(obj.get("a_title", "")),
                        a_category=str(obj.get("a_category", "")),
                        a_description=str(obj.get("a_description", "")),
                        b_title=str(obj.get("b_title", "")),
                        b_category=str(obj.get("b_category", "")),
                        b_description=str(obj.get("b_description", "")),
                        meta={k: v for k, v in obj.items() if k not in {
                            "a_title",
                            "a_category",
                            "a_description",
                            "b_title",
                            "b_category",
                            "b_description",
                        }},
                    )
                )
        return pairs

    # single JSON object
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        out: List[PairSample] = []
        for it in obj:
            out.append(
                PairSample(
                    a_title=str(it.get("a_title", "")),
                    a_category=str(it.get("a_category", "")),
                    a_description=str(it.get("a_description", "")),
                    b_title=str(it.get("b_title", "")),
                    b_category=str(it.get("b_category", "")),
                    b_description=str(it.get("b_description", "")),
                    meta={k: v for k, v in it.items() if k not in {
                        "a_title",
                        "a_category",
                        "a_description",
                        "b_title",
                        "b_category",
                        "b_description",
                    }},
                )
            )
        return out

    return [
        PairSample(
            a_title=str(obj.get("a_title", "")),
            a_category=str(obj.get("a_category", "")),
            a_description=str(obj.get("a_description", "")),
            b_title=str(obj.get("b_title", "")),
            b_category=str(obj.get("b_category", "")),
            b_description=str(obj.get("b_description", "")),
            meta={k: v for k, v in obj.items() if k not in {
                "a_title",
                "a_category",
                "a_description",
                "b_title",
                "b_category",
                "b_description",
            }},
        )
    ]


def build_base_prompt(sample: PairSample) -> str:
    # Reuse the existing prompt builder to stay aligned.
    import runpy

    repo_root = Path(__file__).resolve().parents[3]
    prompt_mod = runpy.run_path(str(repo_root / "scripts" / "build_video_i2i_explain_prompt.py"))
    VideoItem = prompt_mod["VideoItem"]
    build_prompt = prompt_mod["build_prompt"]
    a = VideoItem(title=sample.a_title, category=sample.a_category, description=sample.a_description)
    b = VideoItem(title=sample.b_title, category=sample.b_category, description=sample.b_description)
    return build_prompt(a, b)


def build_rl_prompt(base_prompt: str) -> str:
    # Do not change base prompt content; only append RL-specific output requirements.
    extra = """

Additional RL requirement (MUST follow):
1) Put your full reasoning ONLY inside <think>...</think>.
2) After </think>, output ONLY the final answer lines.
3) Final answer must be 1-2 lines. Each line must start with exactly one label from:
   - Functional/Need Consistency
   - Category & Key Attribute Consistency
   - Usage Scenario & Audience Consistency
   - Collaborative Relation Type
4) You MUST refer strictly to Item A / Item B in the final answer (no A/B shorthand).
5) Do NOT copy any placeholders/examples literally (e.g., "<no more than 30 chars>").
6) Do NOT output any special tokens such as "<|im_end|>".

Required output format:
<think>
...your reasoning...
</think>
Functional/Need Consistency: <<=30 chars>
"""
    return base_prompt.rstrip() + "\n" + extra.strip() + "\n"


def parse_think_and_answer(text: str) -> Tuple[str, str]:
    # Keep it robust to templates.
    t = text.strip()
    if "<think>" in t and "</think>" in t:
        before, after = t.split("</think>", 1)
        think = before.split("<think>", 1)[-1]
        return think.strip(), after.strip()
    return "", t


def extract_labeled_lines(answer_text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for lab in ALLOWED_LABELS:
            if line.startswith(lab + ":"):
                expl = line.split(":", 1)[1].strip()
                if len(expl) > 30:
                    expl = expl[:30]
                lines.append(f"{lab}: {expl}")
                break
        if len(lines) >= 2:
            break
    return lines


def compute_reward(sample: PairSample, full_text: str) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    think, answer = parse_think_and_answer(full_text)
    labeled = extract_labeled_lines(answer)

    has_placeholders = ("no more than 30" in full_text.lower()) or ("<no more" in full_text.lower())
    has_special = "<|" in full_text

    # Format reward
    r_think = 0.0
    if think and len(think) >= 80:
        r_think += 1.0
    else:
        r_think -= 0.6
    if think and not RE_CJK.search(think):
        r_think += 0.2

    r_answer = 0.0
    if 1 <= len(labeled) <= 2 and (not has_placeholders) and (not has_special):
        r_answer += 1.0
    else:
        r_answer -= 0.6

    # Enforce Item A/B usage in final answer (not strict, but reward/penalize)
    ans_join = "\n".join(labeled) if labeled else answer
    r_item_ref = 0.0
    if "Item A" in ans_join and "Item B" in ans_join:
        r_item_ref += 0.6
    if re.search(r"\bA\b|\bB\b", ans_join):
        r_item_ref -= 0.4
    if "Video A" in ans_join or "Video B" in ans_join:
        r_item_ref -= 0.6

    # Length constraint check (after colon)
    r_len = 0.0
    if labeled:
        ok = True
        for ln in labeled:
            expl = ln.split(":", 1)[1].strip() if ":" in ln else ""
            if len(expl) > 30:
                ok = False
            if "<" in expl or ">" in expl:
                ok = False
        r_len = 0.4 if ok else -0.4

    # Content relevance reward (only from input strings, no external knowledge)
    input_text = "\n".join(
        [sample.a_title, sample.a_category, sample.a_description, sample.b_title, sample.b_category, sample.b_description]
    )
    inp_words = set(_tokenize_words(input_text))
    hyp_words = set(_tokenize_words(think + "\n" + ans_join))
    overlap = len(inp_words & hyp_words)
    r_overlap = min(1.0, overlap / 6.0)

    # Anti-gibberish
    ratio = _ascii_printable_ratio(full_text)
    r_clean = 0.0
    if ratio >= 0.95:
        r_clean += 0.4
    elif ratio < 0.85:
        r_clean -= 0.8

    # Hard penalty for any CJK in final answer (English-only training)
    r_no_cjk = 0.0
    if ans_join and RE_CJK.search(ans_join):
        r_no_cjk -= 1.0
    else:
        r_no_cjk += 0.2

    # Penalize copying placeholders or emitting special tokens
    r_no_copy = 0.0
    if has_placeholders:
        r_no_copy -= 1.0
    if has_special:
        r_no_copy -= 0.8

    breakdown = {
        "think": r_think,
        "answer": r_answer,
        "item_ref": r_item_ref,
        "len": r_len,
        "overlap": r_overlap,
        "clean": r_clean,
        "no_cjk": r_no_cjk,
        "no_copy": r_no_copy,
    }
    total = sum(breakdown.values())
    debug = {
        "think": think,
        "answer": answer,
        "labeled": labeled,
        "ascii_printable_ratio": ratio,
        "overlap": overlap,
    }
    return total, breakdown, debug


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _try_import_vllm() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except Exception:
        return False


def rollout_generate(
    prompts: List[str],
    model_path: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_vllm: bool,
    tensor_parallel_size: int,
) -> List[List[str]]:
    """Return generated texts grouped by prompt.

    Shape: outputs[prompt_idx][gen_idx] -> text
    """

    if use_vllm and _try_import_vllm():
        try:
            # Best-effort vLLM rollout. Some architectures may not be supported.
            # vLLM V1 may fail for some models; you can set env VLLM_USE_V1=0 when launching.
            from vllm import LLM, SamplingParams

            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype="bfloat16",
                tensor_parallel_size=max(1, int(tensor_parallel_size)),
            )
            sp = SamplingParams(
                n=int(num_generations),
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_new_tokens),
            )
            outs = llm.generate(prompts, sp)
            grouped: List[List[str]] = []
            for req_out in outs:
                grouped.append([o.text for o in req_out.outputs])
            return grouped
        except Exception:
            # fallback to transformers
            pass

    # Transformers rollout
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    grouped = []
    for p in prompts:
        texts = []
        for _ in range(num_generations):
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
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            gen_ids = out[0][enc["input_ids"].shape[1] :]
            txt = tok.decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
            texts.append(txt)
        grouped.append(texts)
    return grouped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry_run", "train"], default="dry_run")
    ap.add_argument("--input_path", default=str(Path(__file__).resolve().parents[2] / "sample.json"))
    ap.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[2] / "train" / "grpo_runs" / _now_ts()))
    ap.add_argument("--model_path", default=str(Path(__file__).resolve().parents[2] / "basemodel" / "Qwen3-1-7B"))
    ap.add_argument("--ref_model_path", default=None, help="Optional reference model path for KL (train mode)")
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--num_prompts", type=int, default=1)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)

    # training
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta_kl", type=float, default=0.01)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--save_every", type=int, default=10)

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)
    rollouts_path = out_dir / "rollouts.jsonl"

    pairs = load_pairs(Path(args.input_path))
    if not pairs:
        raise SystemExit("No samples loaded.")

    random.seed(0)
    if args.num_prompts > 0:
        pairs = pairs[: args.num_prompts]

    base_prompts = [build_base_prompt(s) for s in pairs]
    rl_prompts = [build_rl_prompt(p) for p in base_prompts]

    # Rollout
    gens = rollout_generate(
        prompts=rl_prompts,
        model_path=args.model_path,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_vllm=bool(args.use_vllm),
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Score + save rollouts
    with rollouts_path.open("w", encoding="utf-8") as f:
        for idx, (sample, prompt, candidates) in enumerate(zip(pairs, rl_prompts, gens)):
            scored = []
            for c in candidates:
                r, br, dbg = compute_reward(sample, c)
                scored.append({
                    "text": c,
                    "reward": r,
                    "reward_breakdown": br,
                    "debug": dbg,
                })
            scored.sort(key=lambda x: x["reward"], reverse=True)
            rec = {
                "id": f"pair-{idx:06d}",
                "created_at": int(time.time()),
                "input": {
                    "a_title": sample.a_title,
                    "a_category": sample.a_category,
                    "a_description": sample.a_description,
                    "b_title": sample.b_title,
                    "b_category": sample.b_category,
                    "b_description": sample.b_description,
                },
                "meta": sample.meta,
                "prompt": prompt,
                "candidates": scored,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.mode == "dry_run":
        # Print top-1 for each prompt
        for idx, sample in enumerate(pairs):
            top = json.loads(rollouts_path.read_text(encoding="utf-8").splitlines()[idx])["candidates"][0]
            print(f"[{idx}] best_reward={top['reward']:.3f}")
            print(top["text"].strip())
            print("-")
        print(f"Saved rollouts: {rollouts_path}")
        return 0

    # Minimal GRPO training loop (Transformers + LoRA)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    policy = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
    policy.train()

    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora_cfg)

    ref_path = args.ref_model_path or args.model_path
    ref = AutoModelForCausalLM.from_pretrained(ref_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
    ref.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=float(args.lr))

    def seq_logp(m, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        # logp of completion tokens only
        out = m(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        target = input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        # slice completion positions
        comp = token_logp[:, prompt_len - 1 :]
        return comp.sum(dim=-1)

    # Reload rollouts for training iterations
    roll_lines = rollouts_path.read_text(encoding="utf-8").splitlines()
    if not roll_lines:
        raise SystemExit("No rollouts to train on.")

    for step in range(1, int(args.steps) + 1):
        rec = json.loads(random.choice(roll_lines))
        prompt = rec["prompt"]
        cands = rec["candidates"]
        texts = [c["text"] for c in cands[: int(args.num_generations)]]
        rewards = torch.tensor([float(c["reward"]) for c in cands[: int(args.num_generations)]], device=policy.device)
        adv = rewards - rewards.mean()
        # stabilize
        if float(adv.std().item()) > 1e-6:
            adv = adv / (adv.std() + 1e-6)

        # Build input_ids for prompt+completion
        # Use raw concatenation; for best results, use consistent chat templates.
        seqs = [prompt + t for t in texts]
        enc_prompt = tok(prompt, return_tensors="pt")
        prompt_len = int(enc_prompt["input_ids"].shape[1])
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

        if step % 1 == 0:
            print(f"step={step} loss={float(loss.item()):.4f} reward_mean={float(rewards.mean().item()):.3f}")

        if step % int(args.save_every) == 0:
            save_dir = out_dir / f"checkpoint_step_{step}"
            _ensure_dir(save_dir)
            policy.save_pretrained(str(save_dir))
            tok.save_pretrained(str(save_dir))
            print(f"saved: {save_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
