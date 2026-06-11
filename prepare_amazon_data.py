#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare Amazon sequential recommendation corpora for TwiSTAR.

This script turns Amazon sequential recommendation files into the standardized
training/evaluation corpora used by TwiSTAR. It does not launch expensive LLM
training; instead, it produces the files needed by the downstream
Itemic Alignment、fast_rec(k)、think_and_rec(j) reasoning、rank_candidates(m,n)
和 planner SFT/RL 数据支持 stages.

Design mapping:
  1. SID grounding / Itemic Alignment -> training_align_data_{split}.parquet
  2. Fast SID generation             -> training_prediction_sid_data_{split}.parquet
  3. Slow reasoning activation       -> training_RA_{split}.parquet
  4. Collaborative commonsense       -> i2i_swing_topK.jsonl + i2i_explain_prompts.jsonl
  5. Ranking tool                    -> ranking_recall_data_{split}.parquet
  6. Planner warm-up labels          -> planner_sft_train.jsonl

Typical usage:
  python prepare_amazon_data.py \
    --dataset Beauty \
    --seq data/sequential_data_processed.txt \
    --items data/Beauty.pretrain.json \
    --out_dir data/twistar_beauty

If the item metadata already contains a TwiSTAR-style `sid` field, this
script reuses it. Otherwise, it creates deterministic hash-based pseudo SIDs so
that the end-to-end pipeline can be smoke-tested before replacing them with
RQ-VAE/residual-k-means SIDs.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


SID_RE = re.compile(r"<\|sid_begin\|>.*?<\|sid_end\|>")


@dataclass(frozen=True)
class ItemMeta:
    item_id: str
    sid: str
    title: str = ""
    categories: str = ""
    brand: str = ""
    description: str = ""


@dataclass(frozen=True)
class SeqSplitSample:
    user_id: str
    history_item_ids: Tuple[str, ...]
    target_item_id: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Amazon reproduction corpora for TwiSTAR."
    )
    parser.add_argument("--dataset", default="Beauty", help="Amazon dataset name, used in manifest only.")
    parser.add_argument("--seq", required=True, help="sequential_data_processed.txt: user item1 item2 ...")
    parser.add_argument("--items", required=True, help="Amazon metadata/pretrain JSON or json.gz/jsonl.gz")
    parser.add_argument("--out_dir", required=True, help="Output directory for generated corpora")
    parser.add_argument("--sid_layers", type=int, default=3, choices=[3, 4], help="SID token depth")
    parser.add_argument("--codebook_size", type=int, default=256, help="SID codebook size per layer")
    parser.add_argument("--i2i_topk", type=int, default=50, help="Top-K I2I neighbors to export")
    parser.add_argument("--planner_k1", type=int, default=10, help="Small fast_rec candidate size for planner path-1")
    parser.add_argument("--planner_k2", type=int, default=50, help="Large recall candidate size for planner path-2")
    parser.add_argument("--planner_explore_prob", type=float, default=0.2, help="Randomly up-route easy samples in SFT labels")
    parser.add_argument("--ranking_negatives", type=int, default=49, help="Negatives sampled from fast_rec K2 candidates per ranking query")
    parser.add_argument(
        "--ranking_use_all_candidates",
        action="store_true",
        help="Use all fast_rec K2 candidates as ranking negatives instead of sampling --ranking_negatives",
    )
    parser.add_argument("--min_common_users", type=int, default=2, help="Min common users for I2I edge")
    parser.add_argument("--alpha", type=float, default=1.0, help="Swing user penalty denominator offset")
    parser.add_argument("--max_users", type=int, default=0, help="Smoke-test cap; 0 means all users")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--fast_rec_jsonl",
        default="",
        help=(
            "Optional real fast model recall results. JSONL fields: user_id, split, candidates. "
            "If absent, a deterministic I2I/popularity proxy is used to create planner warm-up labels."
        ),
    )
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def as_clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " > ".join(as_clean_text(x) for x in value if x)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).replace("\n", " ").strip()


def stable_sid(text: str, layers: int, codebook_size: int) -> str:
    """可复现 SID fallback；真实实验应替换为 RQ-VAE / residual k-means SID。"""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
    prefixes = ["s_a", "s_b", "s_c", "s_d"][:layers]
    toks = []
    for idx, prefix in enumerate(prefixes):
        raw = int.from_bytes(digest[idx * 4 : idx * 4 + 4], byteorder="little", signed=False)
        toks.append(f"<{prefix}_{raw % codebook_size}>")
    return "<|sid_begin|>" + "".join(toks) + "<|sid_end|>"


def normalize_sid(raw: Any, item_id: str, text: str, layers: int, codebook_size: int) -> str:
    sid = as_clean_text(raw)
    if SID_RE.fullmatch(sid.replace(" ", "")):
        return sid.replace(" ", "")
    if sid and "<s_" in sid:
        compact = sid.replace(" ", "")
        if compact.startswith("<|sid_begin|>") and compact.endswith("<|sid_end|>"):
            return compact
    return stable_sid(f"{item_id}\t{text}", layers=layers, codebook_size=codebook_size)


def iter_json_records(path: Path) -> Iterator[dict]:
    """支持 dict JSON、list JSON、Amazon json/json.gz line 格式。"""
    with open_text(path) as f:
        first = f.read(1)
        f.seek(0)
        if first in ("{", "[") and path.suffix != ".gz":
            obj = json.load(f)
            if isinstance(obj, dict):
                # TwiSTAR-style pretrain: item_id -> meta
                for key, value in obj.items():
                    meta = dict(value or {})
                    meta.setdefault("asin", key)
                    yield meta
            elif isinstance(obj, list):
                for value in obj:
                    if isinstance(value, dict):
                        yield value
            return

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = ast.literal_eval(line)
            if isinstance(obj, dict):
                yield obj


def load_items(path: Path, layers: int, codebook_size: int) -> Dict[str, ItemMeta]:
    items: Dict[str, ItemMeta] = {}
    for raw in iter_json_records(path):
        item_id = as_clean_text(raw.get("asin") or raw.get("item_id") or raw.get("id"))
        if not item_id:
            continue
        title = as_clean_text(raw.get("title"))
        categories = as_clean_text(raw.get("categories") or raw.get("category"))
        brand = as_clean_text(raw.get("brand"))
        description = as_clean_text(raw.get("description") or raw.get("feature"))
        text_for_sid = " | ".join(x for x in [title, categories, brand, description] if x) or item_id
        sid = normalize_sid(raw.get("sid"), item_id, text_for_sid, layers, codebook_size)
        items[item_id] = ItemMeta(
            item_id=item_id,
            sid=sid,
            title=title,
            categories=categories,
            brand=brand,
            description=description,
        )
    if not items:
        raise ValueError(f"No valid items loaded from {path}")
    return items


def load_sequences(path: Path, max_users: int = 0) -> List[Tuple[str, List[str]]]:
    rows: List[Tuple[str, List[str]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_users and line_idx >= max_users:
                break
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rows.append((parts[0], parts[1:]))
    if not rows:
        raise ValueError(f"No valid user sequences loaded from {path}")
    return rows


def filter_sequence(item_ids: Sequence[str], items: Mapping[str, ItemMeta]) -> List[str]:
    return [iid for iid in item_ids if iid in items]


def make_leave_one_out_splits(
    sequences: Sequence[Tuple[str, Sequence[str]]], items: Mapping[str, ItemMeta]
) -> Dict[str, List[SeqSplitSample]]:
    splits: Dict[str, List[SeqSplitSample]] = {"train": [], "val": [], "test": []}
    for user_id, raw_items in sequences:
        seq = filter_sequence(raw_items, items)
        if len(seq) < 3:
            continue
        # train: predict third-from-last from prefix before it; val/test follow leave-one-out.
        split_specs = {
            "train": (tuple(seq[:-3]), seq[-3]),
            "val": (tuple(seq[:-2]), seq[-2]),
            "test": (tuple(seq[:-1]), seq[-1]),
        }
        for split, (history, target) in split_specs.items():
            if history:
                splits[split].append(SeqSplitSample(user_id=user_id, history_item_ids=history, target_item_id=target, split=split))
    return splits


def item_desc(meta: ItemMeta) -> str:
    parts = [f"{meta.sid}"]
    if meta.title:
        parts.append(f'its title is "{meta.title}"')
    if meta.categories:
        parts.append(f'its categories are "{meta.categories}"')
    if meta.brand:
        parts.append(f'its brand is "{meta.brand}"')
    return ", ".join(parts)


def history_sid_desc(history_item_ids: Sequence[str], items: Mapping[str, ItemMeta]) -> str:
    sids = [items[iid].sid for iid in history_item_ids if iid in items]
    return "The user has purchased the following items: " + "; ".join(sids) + ";"


def history_text_desc(history_item_ids: Sequence[str], items: Mapping[str, ItemMeta]) -> str:
    descs = [item_desc(items[iid]) for iid in history_item_ids if iid in items]
    return "The user has purchased the following items: " + "; ".join(descs) + ";"


def rows_for_alignment(samples: Sequence[SeqSplitSample], items: Mapping[str, ItemMeta]) -> List[dict]:
    return [{"user_id": s.user_id, "description": history_text_desc(s.history_item_ids, items)} for s in samples]


def rows_for_fast_rec(samples: Sequence[SeqSplitSample], items: Mapping[str, ItemMeta]) -> List[dict]:
    return [
        {
            "user_id": s.user_id,
            "description": history_sid_desc(s.history_item_ids, items),
            "groundtruth": items[s.target_item_id].sid,
            "target_item_id": s.target_item_id,
        }
        for s in samples
    ]


def rows_for_reasoning_activation(samples: Sequence[SeqSplitSample], items: Mapping[str, ItemMeta]) -> List[dict]:
    rows = []
    for s in samples:
        target = items[s.target_item_id]
        rows.append(
            {
                "user_id": s.user_id,
                "description": history_text_desc(s.history_item_ids, items),
                "groundtruth": target.sid,
                "target_item_id": s.target_item_id,
                "title": target.title,
                "categories": target.categories,
            }
        )
    return rows


def write_table(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(list(rows)).to_parquet(path, engine="pyarrow", index=False)
    except Exception as exc:  # pragma: no cover - fallback for minimal environments
        fallback = path.with_suffix(path.suffix + ".jsonl")
        with fallback.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[WARN] Failed to write parquet {path}: {exc}. Wrote JSONL fallback: {fallback}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_sid_maps(items: Mapping[str, ItemMeta]) -> Tuple[Dict[str, dict], Dict[str, str]]:
    sid2text: Dict[str, dict] = {}
    item2sid: Dict[str, str] = {}
    for item_id, meta in items.items():
        item2sid[item_id] = meta.sid
        sid2text.setdefault(
            meta.sid,
            {
                "item_id": item_id,
                "title": meta.title,
                "categories": meta.categories,
                "brand": meta.brand,
                "description": meta.description,
            },
        )
    return sid2text, item2sid


def compute_i2i_swing(
    sequences: Sequence[Tuple[str, Sequence[str]]],
    items: Mapping[str, ItemMeta],
    alpha: float,
    min_common_users: int,
) -> Dict[str, Dict[str, float]]:
    pair_count: Counter[Tuple[str, str]] = Counter()
    pair_score: DefaultDict[Tuple[str, str], float] = defaultdict(float)

    for _, raw_seq in sequences:
        seq = list(dict.fromkeys(filter_sequence(raw_seq, items)))
        if len(seq) < 2:
            continue
        weight = 1.0 / (alpha + len(seq))
        for idx, left in enumerate(seq):
            for right in seq[idx + 1 :]:
                if left == right:
                    continue
                a, b = (left, right) if left < right else (right, left)
                pair_count[(a, b)] += 1
                pair_score[(a, b)] += weight

    sid_adj: DefaultDict[str, DefaultDict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (left, right), common_users in pair_count.items():
        if common_users < min_common_users:
            continue
        sid_l = items[left].sid
        sid_r = items[right].sid
        if sid_l == sid_r:
            continue
        score = pair_score[(left, right)]
        sid_adj[sid_l][sid_r] += score
        sid_adj[sid_r][sid_l] += score
    return {sid: dict(neighbors) for sid, neighbors in sid_adj.items()}


def write_i2i_outputs(
    sid_adj: Mapping[str, Mapping[str, float]],
    sid2text: Mapping[str, dict],
    out_i2i: Path,
    out_prompts: Path,
    topk: int,
) -> None:
    out_i2i.parent.mkdir(parents=True, exist_ok=True)
    with out_i2i.open("w", encoding="utf-8") as f_i2i, out_prompts.open("w", encoding="utf-8") as f_prompt:
        for sid, neighbors in sid_adj.items():
            top_items = sorted(neighbors.items(), key=lambda kv: kv[1], reverse=True)[:topk]
            f_i2i.write(
                json.dumps(
                    {"sid": sid, "topk": [{"sid": n_sid, "score": float(score)} for n_sid, score in top_items]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            src_meta = sid2text.get(sid, {})
            for n_sid, score in top_items[: min(5, topk)]:
                dst_meta = sid2text.get(n_sid, {})
                prompt = (
                    "In collaborative filtering, item i and item j are highly correlated. "
                    "Please explain why users who purchase item i also tend to purchase item j.\n"
                    f"Item i SID: {sid}\n"
                    f"Item i title: {src_meta.get('title', '')}\n"
                    f"Item i categories: {src_meta.get('categories', '')}\n"
                    f"Item j SID: {n_sid}\n"
                    f"Item j title: {dst_meta.get('title', '')}\n"
                    f"Item j categories: {dst_meta.get('categories', '')}\n"
                )
                f_prompt.write(
                    json.dumps(
                        {"source_sid": sid, "target_sid": n_sid, "swing_score": float(score), "prompt": prompt},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def sid_popularity(sequences: Sequence[Tuple[str, Sequence[str]]], items: Mapping[str, ItemMeta]) -> Counter[str]:
    cnt: Counter[str] = Counter()
    for _, raw_seq in sequences:
        for item_id in filter_sequence(raw_seq, items):
            cnt[items[item_id].sid] += 1
    return cnt


def load_fast_rec_candidates(path: Optional[Path]) -> Dict[Tuple[str, str], List[str]]:
    if not path or not path.exists():
        return {}
    out: Dict[Tuple[str, str], List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            user_id = str(obj.get("user_id", ""))
            split = str(obj.get("split", "test"))
            candidates = [str(x) for x in obj.get("candidates", [])]
            if user_id and candidates:
                out[(split, user_id)] = candidates
    return out


def proxy_fast_rec(
    sample: SeqSplitSample,
    items: Mapping[str, ItemMeta],
    sid_adj: Mapping[str, Mapping[str, float]],
    pop: Counter[str],
    k: int,
) -> List[str]:
    scores: DefaultDict[str, float] = defaultdict(float)
    history_sids = [items[iid].sid for iid in sample.history_item_ids if iid in items]
    n = len(history_sids)
    for pos, sid in enumerate(history_sids):
        recency = 0.85 ** max(0, n - pos - 1)
        for nbr_sid, score in sid_adj.get(sid, {}).items():
            scores[nbr_sid] += recency * float(score)
    for sid, freq in pop.most_common(max(k * 5, 200)):
        scores[sid] += 1e-4 * math.log1p(freq)
    for sid in history_sids:
        scores.pop(sid, None)
    return [sid for sid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def planner_label_from_candidates(gt_sid: str, candidates: Sequence[str], k1: int, k2: int) -> Tuple[str, List[dict]]:
    top1 = list(candidates[:k1])
    top2 = list(candidates[:k2])
    if gt_sid in top1:
        return "fast_only", [{"tool": "fast_rec", "arguments": {"k": k1}}]
    if gt_sid in top2:
        return "fast_rank", [
            {"tool": "fast_rec", "arguments": {"k": k2}},
            {"tool": "rank_candidates", "arguments": {"m": k2, "n": k1}},
        ]
    return "slow_reasoning", [{"tool": "think_and_rec", "arguments": {"j": k1}}]


def build_ranking_record(
    sample: SeqSplitSample,
    items: Mapping[str, ItemMeta],
    sid2text: Mapping[str, dict],
    candidate_sid: str,
    label: int,
    candidate_rank: int,
    in_recall_k: bool,
    injected_positive: bool,
    retrieve_k: int,
) -> dict:
    gt_sid = items[sample.target_item_id].sid
    candidate_meta = sid2text.get(candidate_sid, {})
    gt_meta = sid2text.get(gt_sid, {})
    history_sids = [items[iid].sid for iid in sample.history_item_ids if iid in items]
    return {
        "split": sample.split,
        "user_id": sample.user_id,
        "description": history_sid_desc(sample.history_item_ids, items),
        "history_sids_json": json.dumps(history_sids, ensure_ascii=False),
        "history_length": len(history_sids),
        "groundtruth_sid": gt_sid,
        "groundtruth_item_id": sample.target_item_id,
        "groundtruth_title": gt_meta.get("title", ""),
        "groundtruth_categories": gt_meta.get("categories", ""),
        "candidate_sid": candidate_sid,
        "candidate_item_id": candidate_meta.get("item_id", ""),
        "candidate_title": candidate_meta.get("title", ""),
        "candidate_categories": candidate_meta.get("categories", ""),
        "label": int(label),
        "candidate_rank": int(candidate_rank),
        "in_recall_k": bool(in_recall_k),
        "injected_positive": bool(injected_positive),
        "retrieve_k": int(retrieve_k),
    }


def rows_for_ranking_tool(
    samples: Sequence[SeqSplitSample],
    items: Mapping[str, ItemMeta],
    sid2text: Mapping[str, dict],
    sid_adj: Mapping[str, Mapping[str, float]],
    pop: Counter[str],
    fast_rec_candidates: Mapping[Tuple[str, str], List[str]],
    retrieve_k: int,
    negatives_per_query: int,
    use_all_candidates: bool,
    seed: int,
) -> List[dict]:
    """构造 ranker 的 point-wise 训练数据。

    与论文保持一致：先用 fast_rec 召回 K2 个候选；若 ground truth 不在召回集，
    则显式插入为正样本；负样本来自 fast_rec 召回结果。默认每个 query 采
    1 positive + 49 negatives，便于 DIN / cross-encoder / lightweight ranker 训练。
    """
    rng = random.Random(seed)
    rows: List[dict] = []
    for sample_index, sample in enumerate(samples):
        gt_sid = items[sample.target_item_id].sid
        raw_candidates = fast_rec_candidates.get((sample.split, sample.user_id))
        if not raw_candidates:
            raw_candidates = proxy_fast_rec(sample, items, sid_adj, pop, k=retrieve_k)

        # 去重并截断到 K2，避免同一个 SID 既作为正样本又作为负样本重复出现。
        candidates: List[str] = []
        seen = set()
        for sid in raw_candidates:
            if not sid or sid in seen:
                continue
            seen.add(sid)
            candidates.append(sid)
            if len(candidates) >= retrieve_k:
                break

        if not candidates:
            continue

        gt_in_recall = gt_sid in candidates
        negative_pool = [sid for sid in candidates if sid != gt_sid]
        if use_all_candidates:
            sampled_negatives = negative_pool
        else:
            sample_size = min(max(0, negatives_per_query), len(negative_pool))
            sampled_negatives = rng.sample(negative_pool, sample_size) if sample_size else []

        positive_rank = candidates.index(gt_sid) + 1 if gt_in_recall else 0
        query_rows = [
            build_ranking_record(
                sample=sample,
                items=items,
                sid2text=sid2text,
                candidate_sid=gt_sid,
                label=1,
                candidate_rank=positive_rank,
                in_recall_k=gt_in_recall,
                injected_positive=not gt_in_recall,
                retrieve_k=retrieve_k,
            )
        ]

        candidate_rank_map = {sid: idx + 1 for idx, sid in enumerate(candidates)}
        for neg_sid in sampled_negatives:
            query_rows.append(
                build_ranking_record(
                    sample=sample,
                    items=items,
                    sid2text=sid2text,
                    candidate_sid=neg_sid,
                    label=0,
                    candidate_rank=candidate_rank_map.get(neg_sid, 0),
                    in_recall_k=True,
                    injected_positive=False,
                    retrieve_k=retrieve_k,
                )
            )

        # 增加稳定 query id，方便 group-wise ranker 或 AUC/NDCG 分组评估。
        query_id = f"{sample.split}:{sample.user_id}:{sample_index}"
        for row in query_rows:
            row["query_id"] = query_id
            rows.append(row)
    return rows


def write_planner_sft(
    train_samples: Sequence[SeqSplitSample],
    items: Mapping[str, ItemMeta],
    sid_adj: Mapping[str, Mapping[str, float]],
    pop: Counter[str],
    fast_rec_candidates: Mapping[Tuple[str, str], List[str]],
    out_path: Path,
    k1: int,
    k2: int,
    explore_prob: float,
    seed: int,
) -> Dict[str, int]:
    rng = random.Random(seed)
    stats: Counter[str] = Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in train_samples:
            gt_sid = items[sample.target_item_id].sid
            candidates = fast_rec_candidates.get((sample.split, sample.user_id))
            if not candidates:
                candidates = proxy_fast_rec(sample, items, sid_adj, pop, k=max(k2, k1))
            path_name, calls = planner_label_from_candidates(gt_sid, candidates, k1, k2)
            if path_name == "fast_only" and rng.random() < explore_prob:
                if rng.random() < 0.5:
                    path_name = "fast_rank_explore"
                    calls = [
                        {"tool": "fast_rec", "arguments": {"k": k2}},
                        {"tool": "rank_candidates", "arguments": {"m": k2, "n": k1}},
                    ]
                else:
                    path_name = "slow_reasoning_explore"
                    calls = [{"tool": "think_and_rec", "arguments": {"j": k1}}]
            stats[path_name] += 1
            record = {
                "user_id": sample.user_id,
                "description": history_sid_desc(sample.history_item_ids, items),
                "groundtruth": gt_sid,
                "path": path_name,
                "tool_calls": calls,
                "candidate_probe": list(candidates[:k2]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dict(stats)


def write_pipeline_manifest(out_dir: Path, args: argparse.Namespace, counts: Mapping[str, Any], planner_stats: Mapping[str, int]) -> None:
    """Write a machine-readable TwiSTAR pipeline manifest aligned with the paper."""
    manifest = {
        "dataset": args.dataset,
        "generated_counts": counts,
        "planner_label_stats": planner_stats,
        "pipeline_stages": [
            {
                "stage": "itemic_alignment",
                "script": "train/scripts/train_beauty_align.py",
                "inputs": ["training_align_data_train.parquet", "training_align_data_val.parquet"],
                "note": "将脚本参数指向本 out_dir；只训练 SID token embedding/LoRA。",
            },
            {
                "stage": "fast_sid_rec",
                "script": "train/scripts/train_beauty_sid_rec.py",
                "inputs": ["training_prediction_sid_data_train.parquet", "training_prediction_sid_data_val.parquet"],
                "note": "训练 fast_rec(k)，推理时用 prefix trie 约束到合法 SID。",
            },
            {
                "stage": "collaborative_commonsense",
                "script": "train/scripts/train_grpo_i2i_explain.py",
                "inputs": ["i2i_explain_prompts.jsonl"],
                "note": "先用 teacher LLM 填充 explanation，再做 SFT/GRPO 注入 I2I 语言常识。",
            },
            {
                "stage": "slow_reasoning_seqrec",
                "script": "train/scripts/train_grpo_seqrec.py",
                "inputs": ["training_RA_train.parquet"],
                "note": "优先筛 fast_rec@50 miss 的 hard samples，奖励含 think 格式、SID 格式与层级命中。",
            },
            {
                "stage": "ranking_tool",
                "script": "train/scripts/train_ranking_tool.py",
                "inputs": [
                    "ranking_recall_data_train.parquet",
                    "ranking_recall_data_val.parquet",
                    "ranking_recall_data_test.parquet",
                ],
                "note": "DIN/轻量 ranker：对 fast_rec 的 K2 候选重排到 top-n。",
            },
            {
                "stage": "planner_supervised_warmup",
                "script": "planner SFT trainer (not launched by this data-preparation script)",
                "inputs": ["planner_sft_train.jsonl"],
                "note": "论文第一阶段：用 tool_calls 监督标签训练 planner 输出 fast_rec/rank_candidates/think_and_rec。",
            },
            {
                "stage": "planner_agentic_rl",
                "script": "planner GRPO/PPO trainer (paper stage; not launched by this data-preparation script)",
                "inputs": ["planner SFT checkpoint", "validation/test interaction environment"],
                "note": "论文第二阶段：用 NDCG@10 - beta*latency + valid-tool reward 做 GRPO/PPO；agent_workflow.py 是该 planner 的推理期 tool executor。",
            },
        ],
    }
    write_json(out_dir / "twistar_pipeline_manifest.json", manifest)


def main() -> int:
    args = parse_args()
    seq_path = Path(args.seq).resolve()
    items_path = Path(args.items).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Loading items: {items_path}")
    items = load_items(items_path, layers=args.sid_layers, codebook_size=args.codebook_size)
    print(f"      items={len(items)}")

    print(f"[2/7] Loading sequences: {seq_path}")
    sequences = load_sequences(seq_path, max_users=args.max_users)
    print(f"      users={len(sequences)}")

    print("[3/7] Building leave-one-out splits")
    splits = make_leave_one_out_splits(sequences, items)
    split_counts = {split: len(rows) for split, rows in splits.items()}
    print(f"      {split_counts}")

    print("[4/7] Writing alignment / fast-rec / RA corpora")
    serializable_items = {
        item_id: {
            "sid": meta.sid,
            "title": meta.title,
            "categories": meta.categories,
            "brand": meta.brand,
            "description": meta.description,
        }
        for item_id, meta in items.items()
    }
    write_json(out_dir / f"{args.dataset}.items_with_sid.json", serializable_items)
    sid2text, _ = build_sid_maps(items)
    write_json(out_dir / "sid2text.json", sid2text)
    for split, samples in splits.items():
        write_table(rows_for_alignment(samples, items), out_dir / f"training_align_data_{split}.parquet")
        write_table(rows_for_fast_rec(samples, items), out_dir / f"training_prediction_sid_data_{split}.parquet")
        write_table(rows_for_reasoning_activation(samples, items), out_dir / f"training_RA_{split}.parquet")

    print("[5/7] Building SID-level I2I Swing and explanation prompts")
    sid_adj = compute_i2i_swing(sequences, items, alpha=args.alpha, min_common_users=args.min_common_users)
    write_i2i_outputs(
        sid_adj=sid_adj,
        sid2text=sid2text,
        out_i2i=out_dir / f"i2i_swing_top{args.i2i_topk}.jsonl",
        out_prompts=out_dir / "i2i_explain_prompts.jsonl",
        topk=args.i2i_topk,
    )
    print(f"      sid_nodes_with_i2i={len(sid_adj)}")

    print("[6/8] Building ranking tool data")
    pop = sid_popularity(sequences, items)
    fast_rec_candidates = load_fast_rec_candidates(Path(args.fast_rec_jsonl).resolve() if args.fast_rec_jsonl else None)
    ranking_counts: Dict[str, int] = {}
    for split, samples in splits.items():
        ranking_rows = rows_for_ranking_tool(
            samples=samples,
            items=items,
            sid2text=sid2text,
            sid_adj=sid_adj,
            pop=pop,
            fast_rec_candidates=fast_rec_candidates,
            retrieve_k=args.planner_k2,
            negatives_per_query=args.ranking_negatives,
            use_all_candidates=args.ranking_use_all_candidates,
            seed=args.seed + len(split),
        )
        ranking_counts[split] = len(ranking_rows)
        write_table(ranking_rows, out_dir / f"ranking_recall_data_{split}.parquet")
    print(f"      ranking_rows={ranking_counts}")

    print("[7/8] Building planner SFT pseudo-labels")
    planner_stats = write_planner_sft(
        train_samples=splits["train"],
        items=items,
        sid_adj=sid_adj,
        pop=pop,
        fast_rec_candidates=fast_rec_candidates,
        out_path=out_dir / "planner_sft_train.jsonl",
        k1=args.planner_k1,
        k2=args.planner_k2,
        explore_prob=args.planner_explore_prob,
        seed=args.seed,
    )
    print(f"      planner_stats={planner_stats}")

    print("[8/8] Writing TwiSTAR pipeline manifest")
    counts = {
        "items": len(items),
        "users": len(sequences),
        "splits": split_counts,
        "ranking_rows": ranking_counts,
        "sid_nodes_with_i2i": len(sid_adj),
        "unique_sids": len(sid2text),
    }
    write_pipeline_manifest(out_dir, args, counts, planner_stats)
    print(f"Done. Outputs are under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
