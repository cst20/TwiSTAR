#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""分析 Agent 是否把深度资源分配给困难 case。

目标（面向论文写作）：
1) 定义 session 复杂度特征（历史长度、物品稀疏度、互补/多样性需求等）
2) 构造“是否需要调用 rank_candidates / think_and_rec”的监督信号（基于实际工具效果的增益）
3) 训练轻量 router（LogisticRegression），输出 P(rank_candidates)、P(think_and_rec)
4) 可视化 / 分桶统计：特征 vs 路由概率与真实触发率

注：这里的 slow logprob over shortlist 是分析用消融，不等同于论文主工具 think_and_rec(j)
直接生成；论文 planner 训练应采用 supervised warm-up + GRPO/PPO。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sid_pattern():
    import re

    return re.compile(r"<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>")


SID_PATTERN = _sid_pattern()


def extract_sids(text: str) -> List[str]:
    if not text:
        return []
    return SID_PATTERN.findall(text)


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


@dataclass
class SessionFeatures:
    hist_len: int
    cat_unique: int
    cat_diversity: float
    complementarity: float
    mean_popularity: float
    sparsity: float
    fast_margin: float
    din_margin: float
    has_gt_in_candidates: int

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.hist_len,
                self.cat_unique,
                self.cat_diversity,
                self.complementarity,
                self.mean_popularity,
                self.sparsity,
                self.fast_margin,
                self.din_margin,
                self.has_gt_in_candidates,
            ],
            dtype=np.float32,
        )


def compute_features(
    history_sids: Sequence[str],
    gt_sid: str,
    candidates: Sequence[str],
    fast_scores: Dict[str, float],
    din_scores: Dict[str, float],
    sid2meta: Dict[str, dict],
    popularity: Optional[Dict[str, float]] = None,
) -> SessionFeatures:
    # categories
    def parse_categories(cats) -> List[str]:
        if not cats:
            return []
        if isinstance(cats, list):
            return [str(x) for x in cats if x]
        return [c.strip() for c in str(cats).split(",") if c.strip()]

    hist = [s for s in history_sids if s]
    hist_len = len(hist)
    cats_per_item: List[List[str]] = []
    all_cats: List[str] = []
    for sid in hist:
        meta = sid2meta.get(sid, {}) or {}
        cats = parse_categories(meta.get("categories"))
        cats_per_item.append(cats)
        all_cats.extend(cats)
    unique_cats = sorted(set([c for c in all_cats if c]))
    cat_unique = len(unique_cats)
    cat_div = (cat_unique / float(hist_len)) if hist_len > 0 else 0.0

    # complementarity: 1 - avg jaccard(categories)
    if len(cats_per_item) >= 2:
        sims = []
        for i in range(len(cats_per_item)):
            for j in range(i + 1, len(cats_per_item)):
                sims.append(jaccard(cats_per_item[i], cats_per_item[j]))
        avg_sim = float(np.mean(sims)) if sims else 0.0
        comp = 1.0 - avg_sim
    else:
        comp = 0.0

    # popularity/sparsity proxy
    pop = popularity or {}
    pops = [safe_float(pop.get(sid, 0.0), 0.0) for sid in hist]
    mean_pop = float(np.mean(pops)) if pops else 0.0
    # sparsity = mean(-log(pop+eps))
    eps = 1e-9
    sparsity = float(np.mean([-math.log(p + eps) for p in pops])) if pops else 0.0

    # margins
    cand = [s for s in candidates if s]
    def top2_margin(scores: Dict[str, float]) -> float:
        vals = [safe_float(scores.get(s, 0.0), 0.0) for s in cand]
        if len(vals) < 2:
            return 0.0
        v = sorted(vals, reverse=True)
        return float(v[0] - v[1])

    fast_margin = top2_margin(fast_scores)
    din_margin = top2_margin(din_scores)
    has_gt = 1 if (gt_sid in set(cand)) else 0

    return SessionFeatures(
        hist_len=hist_len,
        cat_unique=cat_unique,
        cat_diversity=float(cat_div),
        complementarity=float(comp),
        mean_popularity=float(mean_pop),
        sparsity=float(sparsity),
        fast_margin=float(fast_margin),
        din_margin=float(din_margin),
        has_gt_in_candidates=int(has_gt),
    )


def rank_of(gt: str, ranked: Sequence[str]) -> int:
    """1-based rank; return large int if absent."""
    try:
        return int(list(ranked).index(gt) + 1)
    except ValueError:
        return 10**9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    tw_root = Path(__file__).resolve().parents[1]
    p.add_argument("--val_parquet", type=str, default=str(tw_root / "data" / "training_prediction_sid_data_val.parquet"))
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--trie_pkl", type=str, default=str(tw_root / "test" / "exact_trie.pkl"))
    p.add_argument("--din_model", type=str, default=str(tw_root / "train" / "results" / "din_ranking" / "din_ranking.pth"))
    p.add_argument("--din_sid2id", type=str, default=str(tw_root / "data" / "din_sid2id.json"))
    p.add_argument("--sid2text", type=str, default=str(tw_root / "data" / "sid2text.json"))
    p.add_argument("--beauty_items", type=str, default=str(tw_root / "data" / "Beauty.pretrain.json"))
    p.add_argument("--i2i_jsonl", type=str, default=str(tw_root / "data" / "i2i_swing_top5.jsonl"))
    p.add_argument(
        "--popularity_parquet",
        type=str,
        default=str(tw_root / "data" / "ranking_recall50_data_train.parquet"),
        help="用于估计 item popularity 的 parquet（默认用 recall50 训练集候选分布）",
    )

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--retrieve_k", type=int, default=10)
    p.add_argument("--rank_m", type=int, default=10, help="rank_candidates(m,n) / slow-logprob analysis shortlist size")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=6)
    p.add_argument("--slow_chunk", type=int, default=16)

    p.add_argument("--max_rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(int(args.seed))

    # import repo modules
    import sys

    tw_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(tw_root))
    from agent_workflow import (
        DINRankingModel,
        I2IKnowledge,
        MemoryQueryTool,
        TwiSTARFastRecTool,
        load_sid2text_or_beauty,
    )

    # output
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (tw_root / "test" / "route_analysis" / f"run_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df = pd.read_parquet(Path(args.val_parquet))
    if int(args.max_rows) > 0:
        df = df.iloc[: int(args.max_rows)].reset_index(drop=True)

    sid2meta = load_sid2text_or_beauty(Path(args.sid2text), beauty_items_path=Path(args.beauty_items))
    # popularity proxy（优先用 recall50 训练集候选分布；fallback 才用 i2i swing 的 global_pop）
    popularity: Dict[str, float] = {}
    pop_pq = Path(args.popularity_parquet)
    if pop_pq.exists():
        try:
            pop_df = pd.read_parquet(pop_pq, columns=["candidate_sid"])
            vc = pop_df["candidate_sid"].value_counts(dropna=True)
            total = float(vc.sum()) if float(vc.sum()) > 0 else 1.0
            popularity = {str(k): float(v) / total for k, v in vc.items() if k}
        except Exception:
            popularity = {}

    i2i = None
    if not popularity:
        try:
            p = Path(args.i2i_jsonl)
            if p.exists():
                i2i = I2IKnowledge(p)
                i2i.load()
                popularity = getattr(i2i, "_global_pop", {}) or {}
        except Exception:
            popularity = {}

    mem = MemoryQueryTool(i2i=i2i, sid2text_data=sid2meta)
    mem.load()

    # tools
    fast = TwiSTARFastRecTool(
        model_path=Path(args.model_path),
        memory_query=mem,
        trie_pkl_path=Path(args.trie_pkl) if str(args.trie_pkl).strip() else None,
        build_trie_from_sid2text=False,
        device=str(args.device),
        max_new_tokens=int(args.max_new_tokens),
        prompt_with_think=False,
    )
    fast._lazy_init()  # noqa: SLF001

    din = DINRankingModel(
        model_path=Path(args.din_model),
        sid2id_path=Path(args.din_sid2id),
        device=str(args.device),
        embedding_dim=64,
        max_seq_len=20,
    )
    din._lazy_init()  # noqa: SLF001

    # slow scorer: reuse fast tool to avoid double load
    slow_tool = fast

    # iterate
    rows = []
    batch_size = max(1, int(args.batch_size))
    retrieve_k = int(args.retrieve_k)
    rank_m = int(args.rank_m)
    slow_chunk = max(1, int(args.slow_chunk))

    t0 = time.perf_counter()
    processed = 0
    for start in range(0, len(df), batch_size):
        sub = df.iloc[start : start + batch_size]
        histories = [extract_sids(str(x)) for x in sub["description"].tolist()]
        gts = [str(x) for x in sub["groundtruth"].tolist()]

        batch_results = fast.retrieve_batch(histories, k=retrieve_k)

        for (cands, fast_scores), hist, gt in zip(batch_results, histories, gts):
            cand = [s for s in cands if s and s not in set(hist)]
            # ranks
            fast_ranked = cand
            # DIN scores (on same candidate set)
            din_scores = din.score(hist, cand, fast_scores=fast_scores)
            din_ranked = sorted(cand, key=lambda sid: float(din_scores.get(sid, fast_scores.get(sid, 0.0))), reverse=True)

            # Slow logprob analysis on top rank_m. This is an analysis ablation;
            # paper think_and_rec(j) directly generates recommendations.
            shortlist = din_ranked[: max(1, rank_m)]
            slow_scores: Dict[str, float] = {}
            for cs in range(0, len(shortlist), slow_chunk):
                chunk = shortlist[cs : cs + slow_chunk]
                hs_chunk = [hist for _ in chunk]
                vals = slow_tool.score_sid_batch(hs_chunk, chunk, normalize_by_length=True)
                for sid, sc in zip(chunk, vals):
                    slow_scores[sid] = float(sc)
            slow_ranked = sorted(shortlist, key=lambda sid: float(slow_scores.get(sid, float("-inf"))), reverse=True)

            r_fast = rank_of(gt, fast_ranked)
            r_din = rank_of(gt, din_ranked)
            r_slow = rank_of(gt, slow_ranked)

            need_rank = 1 if (r_din < r_fast and r_fast < 10**9) else 0
            need_slow = 1 if (r_slow < r_din and r_din < 10**9) else 0

            feats = compute_features(
                history_sids=hist,
                gt_sid=gt,
                candidates=cand,
                fast_scores=fast_scores,
                din_scores=din_scores,
                sid2meta=sid2meta,
                popularity=popularity,
            )

            rows.append(
                {
                    "hist_len": feats.hist_len,
                    "cat_unique": feats.cat_unique,
                    "cat_diversity": feats.cat_diversity,
                    "complementarity": feats.complementarity,
                    "mean_popularity": feats.mean_popularity,
                    "sparsity": feats.sparsity,
                    "fast_margin": feats.fast_margin,
                    "din_margin": feats.din_margin,
                    "has_gt": feats.has_gt_in_candidates,
                    "fast_rank": r_fast if r_fast < 10**9 else -1,
                    "din_rank": r_din if r_din < 10**9 else -1,
                    "slow_rank": r_slow if r_slow < 10**9 else -1,
                    "need_rank": need_rank,
                    "need_slow": need_slow,
                }
            )
            processed += 1

        if processed and processed % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"progress {processed}/{len(df)} elapsed={elapsed:.1f}s sps={processed/elapsed:.2f}", flush=True)

    data = pd.DataFrame(rows)
    data_path = out_dir / "routing_dataset.csv"
    data.to_csv(data_path, index=False)

    # Train routers (sklearn)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    feat_cols = [
        "hist_len",
        "cat_unique",
        "cat_diversity",
        "complementarity",
        "mean_popularity",
        "sparsity",
        "fast_margin",
        "din_margin",
        "has_gt",
    ]

    X = data[feat_cols].to_numpy(dtype=np.float32)
    y_rank = data["need_rank"].to_numpy(dtype=np.int64)
    y_slow = data["need_slow"].to_numpy(dtype=np.int64)

    X_train, X_test, y_rank_tr, y_rank_te, y_slow_tr, y_slow_te = train_test_split(
        X, y_rank, y_slow, test_size=0.3, random_state=int(args.seed), stratify=y_slow if (y_slow.sum() > 10) else None
    )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    def fit_lr(ytr, yte, name: str):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        p_te = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p_te) if (len(np.unique(yte)) > 1) else float("nan")
        return clf, float(auc)

    clf_rank, auc_rank = fit_lr(y_rank_tr, y_rank_te, "rank")
    clf_slow, auc_slow = fit_lr(y_slow_tr, y_slow_te, "slow")

    # predict on full
    X_all = scaler.transform(X)
    p_rank = clf_rank.predict_proba(X_all)[:, 1]
    p_slow = clf_slow.predict_proba(X_all)[:, 1]
    data["p_rank"] = p_rank
    data["p_slow"] = p_slow

    # bin analysis
    def bin_report(col: str, pcol: str, ycol: str, bins: int = 5):
        vals = data[col].to_numpy()
        qs = np.quantile(vals, np.linspace(0, 1, bins + 1))
        # make edges strictly increasing
        edges = [qs[0]]
        for v in qs[1:]:
            if v <= edges[-1]:
                v = edges[-1] + 1e-6
            edges.append(v)

        out = []
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            m = (vals >= lo) & (vals < hi if i < bins - 1 else vals <= hi)
            if not m.any():
                continue
            out.append(
                {
                    "bin": i,
                    "range": f"[{lo:.3g},{hi:.3g}]",
                    "n": int(m.sum()),
                    "p_mean": float(np.mean(data.loc[m, pcol])),
                    "y_rate": float(np.mean(data.loc[m, ycol])),
                }
            )
        return out

    reports = {
        "meta": {
            "rows": int(len(data)),
            "retrieve_k": retrieve_k,
            "rank_m": rank_m,
            "auc_rank": auc_rank,
            "auc_slow": auc_slow,
        },
        "p_rank": {
            "by_hist_len": bin_report("hist_len", "p_rank", "need_rank"),
            "by_sparsity": bin_report("sparsity", "p_rank", "need_rank"),
            "by_complementarity": bin_report("complementarity", "p_rank", "need_rank"),
        },
        "p_slow": {
            "by_hist_len": bin_report("hist_len", "p_slow", "need_slow"),
            "by_sparsity": bin_report("sparsity", "p_slow", "need_slow"),
            "by_complementarity": bin_report("complementarity", "p_slow", "need_slow"),
            "by_fast_margin": bin_report("fast_margin", "p_slow", "need_slow"),
            "by_din_margin": bin_report("din_margin", "p_slow", "need_slow"),
        },
    }

    (out_dir / "routing_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    # plots
    try:
        import matplotlib.pyplot as plt

        def plot_bins(title: str, items: List[dict], out_png: Path):
            xs = list(range(len(items)))
            pmean = [it["p_mean"] for it in items]
            yrate = [it["y_rate"] for it in items]
            labels = [it["range"] for it in items]
            plt.figure(figsize=(9, 4))
            plt.plot(xs, pmean, marker="o", label="Pred P")
            plt.plot(xs, yrate, marker="x", label="Empirical rate")
            plt.xticks(xs, labels, rotation=25, ha="right")
            plt.ylim(0, 1)
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()

        plot_bins("P(Rank) vs history length (binned)", reports["p_rank"]["by_hist_len"], out_dir / "p_rank_vs_hist_len.png")
        plot_bins("P(ThinkSlow) vs history length (binned)", reports["p_slow"]["by_hist_len"], out_dir / "p_slow_vs_hist_len.png")
        plot_bins("P(ThinkSlow) vs sparsity (binned)", reports["p_slow"]["by_sparsity"], out_dir / "p_slow_vs_sparsity.png")
        plot_bins("P(ThinkSlow) vs complementarity (binned)", reports["p_slow"]["by_complementarity"], out_dir / "p_slow_vs_complementarity.png")
        plot_bins("P(ThinkSlow) vs fast margin (binned)", reports["p_slow"]["by_fast_margin"], out_dir / "p_slow_vs_fast_margin.png")
        plot_bins("P(ThinkSlow) vs din margin (binned)", reports["p_slow"]["by_din_margin"], out_dir / "p_slow_vs_din_margin.png")
    except Exception:
        pass

    # console summary
    print("\n=== Routing analysis summary ===")
    print(f"out_dir={out_dir}")
    print(f"rows={len(data)} retrieve_k={retrieve_k} rank_m={rank_m}")
    print(f"AUC(rank_router)={auc_rank:.4f} AUC(slow_router)={auc_slow:.4f}")
    print("\nP(ThinkSlow) by hist_len bins:")
    for it in reports["p_slow"]["by_hist_len"]:
        print(it)
    print("\nP(ThinkSlow) by sparsity bins:")
    for it in reports["p_slow"]["by_sparsity"]:
        print(it)
    print("\nP(ThinkSlow) by complementarity bins:")
    for it in reports["p_slow"]["by_complementarity"]:
        print(it)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
