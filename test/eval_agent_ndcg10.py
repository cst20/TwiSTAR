#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""在 Beauty 验证集上评测 Agent 的 ndcg@10，并统计耗时。

评测模式：
  1) fast_only: 只调用 fast_rec(k)，直接取 beam 顺序 top10
  2) fast_din:  fast_rec(k) + rank_candidates(m,n)，取排序 top10
  3) fast_din_slow_logprob_ablation: Fast + DIN + slow logprob 候选打分，仅为消融；
     论文主工具 think_and_rec(j) 应直接生成推荐，不依赖候选 shortlist。

并行：支持 sample_offset/sample_num，用于多进程切分数据。
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import torch


TWISTAR_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    if parts and parts[0] == "TwiStar":
        parts = parts[1:]
    return TWISTAR_ROOT.joinpath(*parts)


def ndcg_at_10(pred: Sequence[str], gt: str) -> float:
    if not gt:
        return 0.0
    try:
        rank = list(pred[:10]).index(gt)
    except ValueError:
        return 0.0
    # 单一相关物品：DCG=1/log2(rank+2)，IDCG=1
    return 1.0 / math.log2(rank + 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate agent ndcg@10 on Beauty val (sharded)")

    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["fast_only", "fast_din", "fast_din_slow_logprob_ablation"],
        help="评测模式",
    )

    p.add_argument(
        "--val_parquet_file",
        type=str,
        default=str(repo_path("data", "training_prediction_sid_data_val.parquet")),
        help="验证集 parquet（需要 description + groundtruth）",
    )

    # shard
    p.add_argument("--sample_num", type=int, default=-1, help="本进程评测样本数，-1=全量")
    p.add_argument("--sample_offset", type=int, default=0, help="本进程起始 offset")
    p.add_argument("--gpu_id", type=int, default=0, help="日志用 GPU id")

    # models
    p.add_argument(
        "--onerec_model_path",
        type=str,
        required=True,
        help="fast_rec(k) 模型目录（HF 可加载）",
    )
    p.add_argument(
        "--slow_model_path",
        type=str,
        default="",
        help="Slow Model 候选 logprob 打分消融模型目录；为空则复用 --onerec_model_path。",
    )
    p.add_argument(
        "--sid2text_path",
        type=str,
        default=str(repo_path("data", "sid2text.json")),
        help="sid2text.json（构建 trie / meta）",
    )
    p.add_argument(
        "--beauty_items_path",
        type=str,
        default=str(repo_path("data", "Beauty.pretrain.json")),
        help="Beauty.pretrain.json（sid2text 缺失回退）",
    )
    p.add_argument(
        "--trie_pkl_path",
        type=str,
        default=str(repo_path("test", "exact_trie.pkl")),
        help="可选 trie pkl；不存在时自动从 sid2text 构建（更慢）",
    )
    p.add_argument("--onerec_device", type=str, default="cuda:0", help="fast_rec(k) 推理设备")
    p.add_argument("--onerec_max_new_tokens", type=int, default=20)
    p.add_argument(
        "--onerec_prompt_with_think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fast_rec prompt 是否预填空 <think> 块。关闭可显著提升 prefix constraint 吞吐。",
    )

    # fast
    p.add_argument("--retrieve_k", type=int, default=50)
    p.add_argument("--fast_batch_size", type=int, default=8, help="Fast 召回的 batch size（每批多少用户）")

    # DIN
    p.add_argument(
        "--din_model_path",
        type=str,
        default=str(repo_path("train", "results", "din_ranking", "din_ranking.pth")),
    )
    p.add_argument(
        "--din_sid2id_path",
        type=str,
        default=str(repo_path("data", "din_sid2id.json")),
    )
    p.add_argument("--din_device", type=str, default="cuda:0")
    p.add_argument("--din_embedding_dim", type=int, default=64)
    p.add_argument("--din_max_seq_len", type=int, default=20)
    p.add_argument("--rank_m", type=int, default=50, help="rank_candidates(m,n) 的 m / slow-logprob 消融 shortlist size")

    # slow
    p.add_argument("--slow_normalize_by_length", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--slow_chunk_size",
        type=int,
        default=16,
        help="Slow logprob 消融每次打分的候选 chunk 大小（防止一次 50 个 OOM）",
    )

    p.add_argument("--log_file", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    t_total0 = time.perf_counter()

    # logging
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        f = open(args.log_file, "w", encoding="utf-8")
    else:
        f = None

    def log(msg: str) -> None:
        print(msg, flush=True)
        if f is not None:
            f.write(msg + "\n")
            f.flush()

    torch.backends.cudnn.benchmark = True

    val_path = Path(args.val_parquet_file)
    df = pd.read_parquet(val_path)
    if args.sample_offset > 0:
        df = df.iloc[int(args.sample_offset) :].reset_index(drop=True)
    if args.sample_num and int(args.sample_num) > 0:
        df = df.iloc[: int(args.sample_num)].reset_index(drop=True)

    total = len(df)
    log(f"[GPU {args.gpu_id}] mode={args.mode} samples={total} parquet={val_path}")

    # 为了避免包导入路径问题：直接把 TwiStar 根目录加入 sys.path
    import sys

    sys.path.insert(0, str(TWISTAR_ROOT))
    from agent_workflow import (
        DINRankingModel,
        TwiSTARFastRecTool,
        extract_sids_from_text,
        load_sid2text_or_beauty,
        MemoryQueryTool,
    )

    t_init0 = time.perf_counter()
    sid2text = load_sid2text_or_beauty(Path(args.sid2text_path), beauty_items_path=Path(args.beauty_items_path))
    mem = MemoryQueryTool(i2i=None, sid2text_data=sid2text)
    mem.load()

    trie_pkl = Path(args.trie_pkl_path) if args.trie_pkl_path else None
    onerec = TwiSTARFastRecTool(
        model_path=Path(args.onerec_model_path),
        memory_query=mem,
        trie_pkl_path=trie_pkl if (trie_pkl and trie_pkl.exists()) else None,
        build_trie_from_sid2text=True if not (trie_pkl and trie_pkl.exists()) else False,
        device=str(args.onerec_device),
        max_new_tokens=int(args.onerec_max_new_tokens),
        prompt_with_think=bool(args.onerec_prompt_with_think),
    )

    # 强制初始化，避免把模型加载时间混入推理阶段
    onerec._lazy_init()  # noqa: SLF001

    slow_tool = None
    slow_logprob_ablation_mode = args.mode == "fast_din_slow_logprob_ablation"
    if slow_logprob_ablation_mode:
        # Slow logprob 消融默认复用 fast_rec 实例（避免在同一 GPU 上重复加载权重导致 OOM）。
        slow_path = Path(args.slow_model_path) if str(args.slow_model_path).strip() else Path(args.onerec_model_path)
        try:
            same_model = slow_path.resolve() == Path(args.onerec_model_path).resolve()
        except Exception:
            same_model = str(slow_path) == str(args.onerec_model_path)

        if same_model:
            slow_tool = onerec
        else:
            slow_tool = TwiSTARFastRecTool(
                model_path=slow_path,
                memory_query=mem,
                trie_pkl_path=None,
                build_trie_from_sid2text=False,
                device=str(args.onerec_device),
                max_new_tokens=int(args.onerec_max_new_tokens),
                prompt_with_think=bool(args.onerec_prompt_with_think),
            )
            slow_tool._lazy_init()  # noqa: SLF001

    din = None
    if args.mode in {"fast_din", "fast_din_slow_logprob_ablation"}:
        din = DINRankingModel(
            model_path=Path(args.din_model_path),
            sid2id_path=Path(args.din_sid2id_path),
            device=str(args.din_device),
            embedding_dim=int(args.din_embedding_dim),
            max_seq_len=int(args.din_max_seq_len),
        )

        din._lazy_init()  # noqa: SLF001

    init_time = time.perf_counter() - t_init0
    log(f"[GPU {args.gpu_id}] init_time_sec={init_time:.2f}")

    # iterate
    ndcg_sum = 0.0
    t0 = time.perf_counter()
    t_fast = 0.0
    t_rank = 0.0
    t_slow = 0.0
    processed = 0

    batch_size = max(1, int(args.fast_batch_size))
    retrieve_k = int(args.retrieve_k)
    rank_m = int(args.rank_m)
    slow_chunk = max(1, int(args.slow_chunk_size))

    # 预取所有 history / gt
    histories: List[List[str]] = []
    gts: List[str] = []
    for _, row in df.iterrows():
        desc = str(row.get("description", ""))
        gt = str(row.get("groundtruth", ""))
        hs = extract_sids_from_text(desc)
        histories.append(hs)
        gts.append(gt)

    for start in range(0, total, batch_size):
        batch_hist = histories[start : start + batch_size]
        batch_gt = gts[start : start + batch_size]

        # Fast retrieve
        t1 = time.perf_counter()
        batch_results = onerec.retrieve_batch(batch_hist, k=retrieve_k)
        t_fast += time.perf_counter() - t1

        for (cands, fast_scores), hist, gt in zip(batch_results, batch_hist, batch_gt):
            processed += 1
            # Fast only
            if args.mode == "fast_only":
                pred_top10 = [sid for sid in cands if sid and sid not in set(hist)][:10]
                ndcg_sum += ndcg_at_10(pred_top10, gt)
                continue

            # DIN ranking
            assert din is not None
            t2 = time.perf_counter()
            rank_scores = din.score(hist, cands, fast_scores=fast_scores)
            ranked0 = sorted(
                [sid for sid in cands if sid and sid not in set(hist)],
                key=lambda sid: float(rank_scores.get(sid, fast_scores.get(sid, 0.0))),
                reverse=True,
            )
            t_rank += time.perf_counter() - t2

            if args.mode == "fast_din":
                ndcg_sum += ndcg_at_10(ranked0[:10], gt)
                continue

            # Slow logprob ablation: teacher-forcing logprob over shortlist.
            # 注意：这不是论文主工具 think_and_rec(j)，论文主工具应直接生成 top-j SID。
            shortlist = ranked0[: max(rank_m, 10)]
            if not shortlist:
                ndcg_sum += 0.0
                continue

            t3 = time.perf_counter()
            # chunk 打分
            scores: Dict[str, float] = {}
            for cs in range(0, len(shortlist), slow_chunk):
                chunk = shortlist[cs : cs + slow_chunk]
                histories_chunk = [hist for _ in chunk]
                assert slow_tool is not None
                vals = slow_tool.score_sid_batch(
                    histories_chunk,
                    chunk,
                    normalize_by_length=bool(args.slow_normalize_by_length),
                )
                for sid, sc in zip(chunk, vals):
                    scores[sid] = float(sc)
            ranked = sorted(shortlist, key=lambda sid: float(scores.get(sid, float("-inf"))), reverse=True)
            t_slow += time.perf_counter() - t3
            ndcg_sum += ndcg_at_10(ranked[:10], gt)

        # progress
        if processed and (processed % 500 == 0 or processed >= total):
            elapsed = time.perf_counter() - t0
            sps = (processed / elapsed) if elapsed > 0 else 0.0
            log(f"[GPU {args.gpu_id}] progress {processed}/{total} elapsed={elapsed:.1f}s samples_per_s={sps:.2f}")

    wall = time.perf_counter() - t0
    ndcg10 = ndcg_sum / max(1, total)
    log(f"[GPU {args.gpu_id}] ndcg@10={ndcg10:.6f}")
    log(f"[GPU {args.gpu_id}] wall_time_sec={wall:.2f}")
    log(f"[GPU {args.gpu_id}] total_time_sec={time.perf_counter() - t_total0:.2f}")
    log(f"[GPU {args.gpu_id}] time_fast_sec={t_fast:.2f}")
    log(f"[GPU {args.gpu_id}] time_rank_sec={t_rank:.2f}")
    log(f"[GPU {args.gpu_id}] time_slow_sec={t_slow:.2f}")
    log(f"[GPU {args.gpu_id}] total_samples={total}")

    if f is not None:
        f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
