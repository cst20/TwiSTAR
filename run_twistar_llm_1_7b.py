#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a Qwen3-1.7B ``rank_candidates`` ablation for TwiSTAR on Amazon Beauty.

This script is intentionally a ranking-tool ablation, not the full TwiSTAR
planner.  It keeps the paper-facing tool names for the fast/ranking path:

  fast_rec(k)          -> statistical candidate generation from run_twistar.py
  rank_candidates(m,n) -> Qwen3-1.7B scores candidate titles/categories by NLL

It does not implement ``think_and_rec(j)`` or planner SFT/RL.  If a fine-tuned
TwiSTAR ranking checkpoint exists, pass it via --model_name_or_path; otherwise it
runs zero-shot Qwen3-1.7B as a conditional-likelihood ranker over fast candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_twistar import (  # noqa: E402
    ItemMeta,
    Sample,
    TwiStarLite,
    load_items,
    load_sequences,
    make_splits,
    recall_ndcg,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="TwiSTAR rank_candidates(m,n) ablation with Qwen3-1.7B")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-1.7B")
    p.add_argument("--seq", default=str(SCRIPT_DIR / "data" / "sequential_data_processed.txt"))
    p.add_argument("--items", default=str(SCRIPT_DIR / "data" / "Beauty.pretrain.json"))
    p.add_argument("--out_dir", default=str(SCRIPT_DIR / "outputs_llm_1_7b"))
    p.add_argument("--sample_num", type=int, default=200, help="Number of test samples; -1 means full test set")
    p.add_argument("--sample_offset", type=int, default=0)
    p.add_argument("--candidate_k", type=int, default=50)
    p.add_argument("--rank_m", type=int, default=20, help="rank_candidates(m,n) 的 m：仅对 top-m fast_rec 候选打分以节省时间")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--max_history_items", type=int, default=12)
    p.add_argument("--max_pair_items", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8, help="candidate scoring batch size")
    p.add_argument("--save_every", type=int, default=100, help="Write partial metrics/predictions every N samples; 0 disables")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    return p.parse_args()


def item_text(meta: ItemMeta) -> str:
    cats = " > ".join(meta.categories[:4])
    if cats and meta.title:
        return f"{meta.title} (category: {cats})"
    return meta.title or cats or meta.item_id


def build_prompt(history: Sequence[str], items: Mapping[str, ItemMeta], max_history_items: int) -> str:
    hist = [iid for iid in history[-max_history_items:] if iid in items]
    lines = ["You are a recommendation expert."]
    lines.append("A user purchased these products in chronological order:")
    for idx, iid in enumerate(hist, 1):
        lines.append(f"{idx}. {item_text(items[iid])}")
    lines.append("Score whether the following candidate is the user's next purchase.")
    lines.append("Candidate: ")
    return "\n".join(lines)


def candidate_continuation(meta: ItemMeta) -> str:
    return item_text(meta) + "\nAnswer: likely"


class QwenCandidateScorer:
    def __init__(self, model_name_or_path: str, device: str, dtype: str, trust_remote_code: bool):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        if self.device.type == "cpu":
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # Do not require `accelerate`: load normally, then move to the selected device.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, prompt: str, continuations: Sequence[str], batch_size: int) -> List[float]:
        """Return average log-likelihood per continuation token; larger is better."""
        scores: List[float] = []
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)
        for start in range(0, len(continuations), batch_size):
            chunk = list(continuations[start : start + batch_size])
            texts = [prompt + c for c in chunk]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1536)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            outputs = self.model(**enc)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            shifted_mask = attention_mask[:, 1:]
            token_nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)

            # Because padding_side=left, prompt starts after left padding. For each row,
            # continuation labels begin at left_pad + prompt_len - 1 in shifted positions.
            seq_lens = attention_mask.sum(dim=1)
            total_len = input_ids.shape[1]
            for row_idx in range(input_ids.shape[0]):
                left_pad = int(total_len - seq_lens[row_idx].item())
                cont_start = max(0, left_pad + prompt_len - 1)
                valid = shifted_mask[row_idx].bool()
                cont_mask = torch.zeros_like(valid)
                cont_mask[cont_start:] = True
                mask = valid & cont_mask
                denom = int(mask.sum().item())
                if denom <= 0:
                    scores.append(float("-inf"))
                else:
                    scores.append(float(-token_nll[row_idx][mask].sum().item() / denom))
            del outputs, logits, labels, shifted_mask, token_nll, enc, input_ids, attention_mask
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return scores


def select_samples(samples: Sequence[Sample], sample_num: int, sample_offset: int) -> List[Sample]:
    data = list(samples[sample_offset:])
    if sample_num >= 0:
        data = data[:sample_num]
    return data


def evaluate_llm_rank_candidates(
    model: TwiStarLite,
    scorer: QwenCandidateScorer,
    samples: Sequence[Sample],
    items: Mapping[str, ItemMeta],
    candidate_k: int,
    rank_m: int,
    top_k: int,
    max_history_items: int,
    batch_size: int,
    out_dir: Path | None = None,
    save_every: int = 0,
) -> Tuple[Dict[str, Any], List[dict]]:
    total_recall = 0.0
    total_ndcg = 0.0
    fast_rec_bound = 0.0
    logs: List[dict] = []

    for idx, sample in enumerate(samples):
        fast = model.fast_rec(sample.history, candidate_k)
        candidates = list(fast.recs[:rank_m])
        # Evaluation must not inject ground truth. If GT is absent from fast@K2, LLM cannot recover it.
        if sample.target in fast.recs[:candidate_k]:
            fast_rec_bound += 1.0
        prompt = build_prompt(sample.history, items, max_history_items)
        conts = [candidate_continuation(items[iid]) for iid in candidates]
        llm_scores = scorer.score(prompt, conts, batch_size=batch_size) if conts else []
        ranked = [iid for iid, _ in sorted(zip(candidates, llm_scores), key=lambda kv: kv[1], reverse=True)]
        # Backfill with remaining fast candidates to keep top_k length.
        for iid in fast.recs:
            if iid not in ranked:
                ranked.append(iid)
            if len(ranked) >= top_k:
                break
        recs = ranked[:top_k]
        r, n = recall_ndcg(recs, sample.target, top_k)
        total_recall += r
        total_ndcg += n
        logs.append(
            {
                "index": idx,
                "user_id": sample.user_id,
                "target": sample.target,
                "hit": bool(r),
                "ndcg": n,
                "fast_bound_hit": sample.target in fast.recs[:candidate_k],
                "recs": recs,
            }
        )
        if (idx + 1) % 10 == 0:
            denom = idx + 1
            print(f"processed={denom} recall@10={total_recall/denom:.4f} ndcg@10={total_ndcg/denom:.4f}", flush=True)
        if out_dir is not None and save_every and save_every > 0 and (idx + 1) % save_every == 0:
            denom = idx + 1
            partial_metrics = {
                "processed": idx + 1,
                "num_samples": len(samples),
                "recall@10": total_recall / denom,
                "ndcg@10": total_ndcg / denom,
                "fast_rec_bound@50": fast_rec_bound / denom,
            }
            write_json(out_dir / "partial_metrics.json", partial_metrics)
            write_jsonl(out_dir / "partial_predictions.jsonl", logs)

    denom = max(1, len(samples))
    metrics = {
        "num_samples": len(samples),
        "recall@10": total_recall / denom,
        "ndcg@10": total_ndcg / denom,
        "fast_rec_bound@50": fast_rec_bound / denom,
    }
    return metrics, logs


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/5] Load data", flush=True)
    items = load_items(Path(args.items).resolve())
    sequences = load_sequences(Path(args.seq).resolve(), items, max_users=0)
    splits = make_splits(sequences)
    test_samples = select_samples(splits["test"], args.sample_num, args.sample_offset)
    print(f"items={len(items)} users={len(sequences)} test_samples={len(test_samples)}", flush=True)

    print("[2/5] Fit fast recall tool", flush=True)
    rec_model = TwiStarLite(items, max_pair_items=args.max_pair_items)
    rec_model.fit([s.history + (s.target,) for s in splits["train"]])

    print(f"[3/5] Load real LLM: {args.model_name_or_path}", flush=True)
    scorer = QwenCandidateScorer(args.model_name_or_path, args.device, args.dtype, args.trust_remote_code)

    print("[4/5] Evaluate LLM rank_candidates(m,n) ablation", flush=True)
    metrics, logs = evaluate_llm_rank_candidates(
        model=rec_model,
        scorer=scorer,
        samples=test_samples,
        items=items,
        candidate_k=args.candidate_k,
        rank_m=args.rank_m,
        top_k=args.top_k,
        max_history_items=args.max_history_items,
        batch_size=args.batch_size,
        out_dir=out_dir,
        save_every=args.save_every,
    )
    metrics.update(
        {
            "model_name_or_path": args.model_name_or_path,
            "model_size": "1.7B",
            "candidate_k": args.candidate_k,
            "rank_m": args.rank_m,
            "sample_offset": args.sample_offset,
            "elapsed_sec": round(time.time() - t0, 3),
        }
    )

    print("[5/5] Write outputs", flush=True)
    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "predictions.jsonl", logs)
    print("\nFinal TwiSTAR-Qwen3-1.7B Test Result")
    print(f"Recall@10 = {metrics['recall@10']:.6f}")
    print(f"NDCG@10   = {metrics['ndcg@10']:.6f}")
    print(f"fast_rec bound@50 = {metrics['fast_rec_bound@50']:.6f}")
    print(f"Metrics written to: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
