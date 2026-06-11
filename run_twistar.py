#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight TwiSTAR reproduction on Amazon Beauty.

这个文件夹是独立的 TwiSTAR 复现入口。由于当前工作区可能没有
可直接加载的 TwiSTAR/Qwen 训练后 checkpoint，本脚本实现一个可运行的
结构等价轻量版本：

1. fast_rec：基于训练序列的 item-to-item 共现/转移召回；
2. rank_candidates：对 fast_rec@K2 候选做类别、标题、协同分数重排；
3. think_and_rec：面向“困难/兴趣发散”样本的慢思考代理，聚合全历史、多类别兴趣；
4. lite_router：轻量阈值路由器，仅用于无 LLM/checkpoint 时的 ablation；论文版 planner
   应使用 planner_sft_train.jsonl 做 supervised warm-up，再用 GRPO/PPO 做 RL；
5. evaluation：在 test leave-one-out 上输出 Recall@10 和 NDCG@10。

它还会写出 ranker 训练样本、轻量路由日志和最终 metrics，便于后续替换成
真正的 TwiSTAR fast/slow LLM checkpoint、ranking tool 和 GRPO/PPO planner。
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class ItemMeta:
    item_id: str
    sid: str
    title: str
    categories: Tuple[str, ...]
    title_tokens: Tuple[str, ...]


@dataclass(frozen=True)
class Sample:
    user_id: str
    history: Tuple[str, ...]
    target: str
    split: str


@dataclass
class ToolOutput:
    tool: str
    recs: List[str]
    scores: Dict[str, float]
    confidence: float
    gap: float
    diversity: int


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Run lightweight TwiSTAR on Amazon data")
    p.add_argument("--seq", default=str(SCRIPT_DIR / "data" / "sequential_data_processed.txt"))
    p.add_argument("--items", default=str(SCRIPT_DIR / "data" / "Beauty.pretrain.json"))
    p.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "outputs"))
    p.add_argument("--dataset", default="Beauty")
    p.add_argument("--recall_k", type=int, default=10)
    p.add_argument("--fast_k", type=int, default=10)
    p.add_argument("--candidate_k", type=int, default=50)
    p.add_argument("--max_users", type=int, default=0, help="0 means full data")
    p.add_argument("--max_pair_items", type=int, default=80, help="cap per-user train items for O(n^2) cooccur")
    p.add_argument("--write_ranking_data", dest="write_ranking_data", action="store_true", default=True)
    p.add_argument("--no_write_ranking_data", dest="write_ranking_data", action="store_false")
    return p.parse_args()


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(clean_text(v) for v in x if v)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x).replace("\n", " ").strip()


def parse_categories(x: Any) -> Tuple[str, ...]:
    if not x:
        return tuple()
    if isinstance(x, list):
        vals: List[str] = []
        for v in x:
            vals.extend(parse_categories(v))
        return tuple(dict.fromkeys(v for v in vals if v))
    s = clean_text(x)
    parts = [p.strip().lower() for p in re.split(r">|,|/|\|", s) if p.strip()]
    return tuple(dict.fromkeys(parts))


def tokenize_title(title: str) -> Tuple[str, ...]:
    stop = {"the", "and", "for", "with", "from", "this", "that", "you", "your", "set", "pack"}
    toks = [t.lower() for t in TOKEN_RE.findall(title or "")]
    return tuple(t for t in toks if len(t) > 2 and t not in stop)


def iter_item_records(path: Path) -> Iterator[dict]:
    with open_text(path) as f:
        first = f.read(1)
        f.seek(0)
        if first in ("{", "[") and path.suffix != ".gz":
            obj = json.load(f)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    row = dict(v or {})
                    row.setdefault("asin", k)
                    yield row
            elif isinstance(obj, list):
                for row in obj:
                    if isinstance(row, dict):
                        yield row
            return
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = ast.literal_eval(line)
            if isinstance(row, dict):
                yield row


def load_items(path: Path) -> Dict[str, ItemMeta]:
    out: Dict[str, ItemMeta] = {}
    for row in iter_item_records(path):
        item_id = clean_text(row.get("asin") or row.get("item_id") or row.get("id"))
        if not item_id:
            continue
        sid = clean_text(row.get("sid")) or item_id
        title = clean_text(row.get("title"))
        cats = parse_categories(row.get("categories") or row.get("category"))
        out[item_id] = ItemMeta(
            item_id=item_id,
            sid=sid,
            title=title,
            categories=cats,
            title_tokens=tokenize_title(title),
        )
    if not out:
        raise ValueError(f"no items loaded from {path}")
    return out


def load_sequences(path: Path, items: Mapping[str, ItemMeta], max_users: int) -> List[Tuple[str, List[str]]]:
    rows: List[Tuple[str, List[str]]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_users and idx >= max_users:
                break
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            seq = [x for x in parts[1:] if x in items]
            if len(seq) >= 3:
                rows.append((parts[0], seq))
    if not rows:
        raise ValueError(f"no valid sequences loaded from {path}")
    return rows


def make_splits(sequences: Sequence[Tuple[str, Sequence[str]]]) -> Dict[str, List[Sample]]:
    splits = {"train": [], "val": [], "test": []}
    for user_id, seq in sequences:
        if len(seq) < 3:
            continue
        specs = {
            "train": (tuple(seq[:-3]), seq[-3]),
            "val": (tuple(seq[:-2]), seq[-2]),
            "test": (tuple(seq[:-1]), seq[-1]),
        }
        for split, (hist, target) in specs.items():
            if hist:
                splits[split].append(Sample(user_id=user_id, history=hist, target=target, split=split))
    return splits


class TwiStarLite:
    def __init__(self, items: Mapping[str, ItemMeta], max_pair_items: int = 80):
        self.items = items
        self.max_pair_items = max_pair_items
        self.pop: Counter[str] = Counter()
        self.cooccur: DefaultDict[str, DefaultDict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.transition: DefaultDict[str, DefaultDict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.cat_pop: DefaultDict[str, Counter[str]] = defaultdict(Counter)
        self.global_popular: List[str] = []

    def fit(self, train_histories: Iterable[Sequence[str]]) -> None:
        for seq_raw in train_histories:
            seq = [x for x in seq_raw if x in self.items]
            if not seq:
                continue
            self.pop.update(seq)
            for a, b in zip(seq[:-1], seq[1:]):
                if a != b:
                    self.transition[a][b] += 1.0
            # 类别热度，用于 slow thinking proxy。
            for iid in seq:
                for cat in self.items[iid].categories:
                    self.cat_pop[cat][iid] += 1
            uniq = list(dict.fromkeys(seq[-self.max_pair_items :]))
            n = len(uniq)
            if n >= 2:
                w = 1.0 / math.sqrt(n)
                for i, a in enumerate(uniq):
                    for b in uniq[i + 1 :]:
                        if a == b:
                            continue
                        self.cooccur[a][b] += w
                        self.cooccur[b][a] += w
        self.global_popular = [iid for iid, _ in self.pop.most_common()]

    def history_diversity(self, history: Sequence[str]) -> int:
        cats = set()
        for iid in history:
            cats.update(self.items[iid].categories[:2])
        return len(cats)

    def _normalize_top(self, scores: Dict[str, float], topn: int, exclude: Sequence[str]) -> Tuple[List[str], Dict[str, float]]:
        for iid in exclude:
            scores.pop(iid, None)
        if len(scores) < topn:
            floor = 1e-8
            for iid in self.global_popular:
                if iid not in scores and iid not in exclude:
                    scores[iid] = floor * math.log1p(self.pop[iid])
                    if len(scores) >= topn * 5:
                        break
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:topn]
        recs = [iid for iid, _ in ranked]
        return recs, dict(ranked)

    def fast_rec(self, history: Sequence[str], topn: int) -> ToolOutput:
        scores: DefaultDict[str, float] = defaultdict(float)
        h = [x for x in history if x in self.items]
        m = len(h)
        for pos, iid in enumerate(h[-30:]):
            dist = m - pos - 1
            recency = 0.88 ** max(dist, 0)
            for nb, s in self.transition.get(iid, {}).items():
                scores[nb] += 2.5 * recency * math.log1p(s)
            for nb, s in self.cooccur.get(iid, {}).items():
                scores[nb] += recency * math.log1p(s)
        for iid in self.global_popular[: max(500, topn * 10)]:
            scores[iid] += 0.015 * math.log1p(self.pop[iid])
        recs, top_scores = self._normalize_top(dict(scores), topn, exclude=h)
        vals = [top_scores[x] for x in recs]
        conf = vals[0] if vals else 0.0
        gap = (vals[0] - vals[1]) if len(vals) > 1 else conf
        return ToolOutput("fast_rec", recs, top_scores, conf, gap, self.history_diversity(h))

    def rank_candidates(self, history: Sequence[str], candidates: Sequence[str], base_scores: Mapping[str, float], topn: int) -> ToolOutput:
        h = [x for x in history if x in self.items]
        recent = h[-10:]
        hist_cats = Counter(cat for iid in recent for cat in self.items[iid].categories)
        hist_tokens = Counter(tok for iid in recent for tok in self.items[iid].title_tokens)
        scores: Dict[str, float] = {}
        for cand in candidates:
            if cand in h or cand not in self.items:
                continue
            meta = self.items[cand]
            cat_score = sum(hist_cats.get(c, 0) for c in meta.categories) / max(1, len(recent))
            token_score = sum(hist_tokens.get(t, 0) for t in meta.title_tokens) / max(1, len(recent))
            pop_score = math.log1p(self.pop[cand])
            last_boost = 0.0
            if h:
                last = h[-1]
                last_boost = 1.2 * math.log1p(self.transition.get(last, {}).get(cand, 0.0))
                last_boost += 0.7 * math.log1p(self.cooccur.get(last, {}).get(cand, 0.0))
            scores[cand] = (
                1.00 * float(base_scores.get(cand, 0.0))
                + 0.75 * cat_score
                + 0.15 * token_score
                + 0.05 * pop_score
                + last_boost
            )
        recs, top_scores = self._normalize_top(scores, topn, exclude=h)
        vals = [top_scores[x] for x in recs]
        return ToolOutput("rank_candidates", recs, top_scores, vals[0] if vals else 0.0, vals[0] - vals[1] if len(vals) > 1 else 0.0, self.history_diversity(h))

    def think_and_rec(self, history: Sequence[str], topn: int) -> ToolOutput:
        """慢思考代理：对长历史/多兴趣用户，显式聚合类别与全历史协同证据。"""
        h = [x for x in history if x in self.items]
        scores: DefaultDict[str, float] = defaultdict(float)
        cat_interest = Counter(cat for iid in h for cat in self.items[iid].categories)
        for cat, weight in cat_interest.items():
            for iid, cnt in self.cat_pop.get(cat, Counter()).most_common(120):
                scores[iid] += 0.08 * math.sqrt(weight) * math.log1p(cnt)
        m = len(h)
        for pos, iid in enumerate(h[-60:]):
            # slow 模式比 fast 更看重较早的稳定兴趣，因此衰减更慢。
            recency = 0.96 ** max(m - pos - 1, 0)
            for nb, s in self.cooccur.get(iid, {}).items():
                scores[nb] += 1.25 * recency * math.log1p(s)
            for nb, s in self.transition.get(iid, {}).items():
                scores[nb] += 0.9 * recency * math.log1p(s)
        for iid in self.global_popular[:500]:
            scores[iid] += 0.01 * math.log1p(self.pop[iid])
        recs, top_scores = self._normalize_top(dict(scores), topn, exclude=h)
        vals = [top_scores[x] for x in recs]
        return ToolOutput("think_and_rec", recs, top_scores, vals[0] if vals else 0.0, vals[0] - vals[1] if len(vals) > 1 else 0.0, self.history_diversity(h))

    def route(self, history: Sequence[str], cfg: Mapping[str, float], candidate_k: int, topn: int) -> ToolOutput:
        fast50 = self.fast_rec(history, candidate_k)
        diversity = fast50.diversity
        if diversity >= cfg["slow_diversity"] and fast50.gap <= cfg["slow_gap_max"]:
            return self.think_and_rec(history, topn)
        if fast50.gap <= cfg["rank_gap_max"] or fast50.confidence <= cfg["rank_conf_max"]:
            return self.rank_candidates(history, fast50.recs, fast50.scores, topn)
        return ToolOutput("fast_rec", fast50.recs[:topn], {k: fast50.scores[k] for k in fast50.recs[:topn]}, fast50.confidence, fast50.gap, diversity)


def recall_ndcg(recs: Sequence[str], target: str, k: int) -> Tuple[float, float]:
    top = list(recs[:k])
    if target not in top:
        return 0.0, 0.0
    rank = top.index(target) + 1
    return 1.0, 1.0 / math.log2(rank + 1)


def evaluate_tool(
    model: TwiStarLite,
    samples: Sequence[Sample],
    tool: str,
    candidate_k: int,
    k: int,
    cfg: Optional[Mapping[str, float]] = None,
    keep_logs: bool = True,
) -> Dict[str, Any]:
    total_r = 0.0
    total_n = 0.0
    route_counter: Counter[str] = Counter()
    logs: List[dict] = []
    for s in samples:
        if tool == "fast":
            out = model.fast_rec(s.history, k)
        elif tool == "rank":
            fast = model.fast_rec(s.history, candidate_k)
            out = model.rank_candidates(s.history, fast.recs, fast.scores, k)
        elif tool == "slow":
            out = model.think_and_rec(s.history, k)
        elif tool == "lite_router":
            assert cfg is not None
            out = model.route(s.history, cfg, candidate_k, k)
        else:
            raise ValueError(tool)
        r, n = recall_ndcg(out.recs, s.target, k)
        total_r += r
        total_n += n
        route_counter[out.tool] += 1
        if keep_logs:
            logs.append({"user_id": s.user_id, "target": s.target, "tool": out.tool, "recs": out.recs, "hit": bool(r), "ndcg": n})
    denom = max(1, len(samples))
    return {
        "recall@10": total_r / denom,
        "ndcg@10": total_n / denom,
        "num_samples": len(samples),
        "routes": dict(route_counter),
        "logs": logs,
    }


def tune_lite_router(model: TwiStarLite, val_samples: Sequence[Sample], candidate_k: int, k: int) -> Tuple[Dict[str, float], Dict[str, Any]]:
    cached: List[Tuple[Sample, ToolOutput, ToolOutput, ToolOutput]] = []
    for s in val_samples:
        fast50 = model.fast_rec(s.history, candidate_k)
        fast10 = ToolOutput(
            "fast_rec",
            fast50.recs[:k],
            {iid: fast50.scores[iid] for iid in fast50.recs[:k]},
            fast50.confidence,
            fast50.gap,
            fast50.diversity,
        )
        rank10 = model.rank_candidates(s.history, fast50.recs, fast50.scores, k)
        slow10 = model.think_and_rec(s.history, k)
        cached.append((s, fast10, rank10, slow10))

    def eval_cfg(cfg: Mapping[str, float]) -> Dict[str, Any]:
        total_r = 0.0
        total_n = 0.0
        routes: Counter[str] = Counter()
        for s, fast10, rank10, slow10 in cached:
            if fast10.diversity >= cfg["slow_diversity"] and fast10.gap <= cfg["slow_gap_max"]:
                out = slow10
            elif fast10.gap <= cfg["rank_gap_max"] or fast10.confidence <= cfg["rank_conf_max"]:
                out = rank10
            else:
                out = fast10
            r, n = recall_ndcg(out.recs, s.target, k)
            total_r += r
            total_n += n
            routes[out.tool] += 1
        denom = max(1, len(cached))
        return {"recall@10": total_r / denom, "ndcg@10": total_n / denom, "num_samples": len(cached), "routes": dict(routes)}

    best_cfg: Dict[str, float] = {}
    best_eval: Dict[str, Any] = {"ndcg@10": -1.0}
    for slow_div in [3, 5, 8]:
        for slow_gap in [0.00, 0.10, 0.30]:
            for rank_gap in [0.10, 0.35, 0.80]:
                for rank_conf in [0.10, 0.40, 1.20]:
                    cfg = {
                        "slow_diversity": float(slow_div),
                        "slow_gap_max": float(slow_gap),
                        "rank_gap_max": float(rank_gap),
                        "rank_conf_max": float(rank_conf),
                    }
                    ev = eval_cfg(cfg)
                    if (ev["ndcg@10"], ev["recall@10"]) > (best_eval["ndcg@10"], best_eval.get("recall@10", -1.0)):
                        best_cfg = cfg
                        best_eval = {kk: vv for kk, vv in ev.items() if kk != "logs"}
    return best_cfg, best_eval


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_ranking_rows(model: TwiStarLite, samples: Sequence[Sample], candidate_k: int) -> List[dict]:
    rows: List[dict] = []
    for idx, s in enumerate(samples):
        fast = model.fast_rec(s.history, candidate_k)
        candidates = list(fast.recs)
        if s.target not in candidates:
            candidates = [s.target] + candidates[:-1]
        for rank, cand in enumerate(candidates, start=1):
            meta = model.items[cand]
            rows.append(
                {
                    "query_id": f"{s.split}:{idx}",
                    "split": s.split,
                    "user_id": s.user_id,
                    "history_json": json.dumps(list(s.history), ensure_ascii=False),
                    "target_item_id": s.target,
                    "candidate_item_id": cand,
                    "candidate_sid": meta.sid,
                    "candidate_title": meta.title,
                    "candidate_categories": list(meta.categories),
                    "candidate_rank_from_fast": rank if cand in fast.recs else 0,
                    "fast_score": float(fast.scores.get(cand, 0.0)),
                    "label": int(cand == s.target),
                    "injected_positive": bool(cand == s.target and s.target not in fast.recs),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1/6] Load items: {args.items}")
    items = load_items(Path(args.items).resolve())
    print(f"      items={len(items)}")

    print(f"[2/6] Load sequences: {args.seq}")
    sequences = load_sequences(Path(args.seq).resolve(), items, args.max_users)
    splits = make_splits(sequences)
    print(f"      users={len(sequences)}, splits={ {k: len(v) for k, v in splits.items()} }")

    print("[3/6] Fit fast/rank/slow proxy tools on train histories")
    model = TwiStarLite(items, max_pair_items=args.max_pair_items)
    # 严格避免 test target 泄漏：只用 train split 的 history + train target 之前的信息。
    train_histories = [s.history + (s.target,) for s in splits["train"]]
    model.fit(train_histories)

    print("[4/6] Tune lightweight router on validation set")
    val_fast = evaluate_tool(model, splits["val"], "fast", args.candidate_k, args.recall_k, keep_logs=False)
    val_rank = evaluate_tool(model, splits["val"], "rank", args.candidate_k, args.recall_k, keep_logs=False)
    val_slow = evaluate_tool(model, splits["val"], "slow", args.candidate_k, args.recall_k, keep_logs=False)
    best_cfg, val_router = tune_lite_router(model, splits["val"], args.candidate_k, args.recall_k)
    print(f"      best_cfg={best_cfg}")

    print("[5/6] Evaluate on test set")
    test_fast = evaluate_tool(model, splits["test"], "fast", args.candidate_k, args.recall_k)
    test_rank = evaluate_tool(model, splits["test"], "rank", args.candidate_k, args.recall_k)
    test_slow = evaluate_tool(model, splits["test"], "slow", args.candidate_k, args.recall_k)
    test_router = evaluate_tool(model, splits["test"], "lite_router", args.candidate_k, args.recall_k, best_cfg)

    print("[6/6] Write outputs")
    if args.write_ranking_data:
        write_jsonl(out_dir / "ranking_recall_data_train.jsonl", build_ranking_rows(model, splits["train"], args.candidate_k))
        write_jsonl(out_dir / "ranking_recall_data_val.jsonl", build_ranking_rows(model, splits["val"], args.candidate_k))
        write_jsonl(out_dir / "ranking_recall_data_test.jsonl", build_ranking_rows(model, splits["test"], args.candidate_k))
    write_jsonl(out_dir / "lite_router_test_routes.jsonl", test_router["logs"])

    def strip_logs(x: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in x.items() if k != "logs"}

    metrics = {
        "dataset": args.dataset,
        "num_items": len(items),
        "num_users": len(sequences),
        "splits": {k: len(v) for k, v in splits.items()},
        "recall_k": args.recall_k,
        "candidate_k": args.candidate_k,
        "validation": {
            "fast": strip_logs(val_fast),
            "rank": strip_logs(val_rank),
            "slow": strip_logs(val_slow),
            "lite_router": val_router,
        },
        "test": {
            "fast": strip_logs(test_fast),
            "rank": strip_logs(test_rank),
            "slow": strip_logs(test_slow),
            "lite_router": strip_logs(test_router),
        },
        "lite_router_cfg": best_cfg,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    write_json(out_dir / "metrics.json", metrics)

    print("\nFinal TwiSTAR-lite Test Result")
    print(f"Recall@10 = {test_router['recall@10']:.6f}")
    print(f"NDCG@10   = {test_router['ndcg@10']:.6f}")
    print(f"Routes    = {test_router['routes']}")
    print(f"Metrics written to: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
