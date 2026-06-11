#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_workflow import MemoryQueryTool, TwiSTARFastRecTool, extract_sids_from_text  # noqa: E402


def load_sid_metadata(sid2text_path: Path | None, beauty_items_path: Path | None) -> Dict[str, dict]:
    if sid2text_path and sid2text_path.exists():
        with sid2text_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if obj and next(iter(obj.keys())).startswith("<|sid_begin|>"):
            return obj

    if beauty_items_path and beauty_items_path.exists():
        with beauty_items_path.open("r", encoding="utf-8") as f:
            beauty_items = json.load(f)

        sid2meta: Dict[str, dict] = {}
        for item_id, meta in beauty_items.items():
            sid = (meta or {}).get("sid")
            if not sid:
                continue
            sid2meta[sid] = {
                "title": (meta or {}).get("title", ""),
                "categories": (meta or {}).get("categories", ""),
                "item_id": item_id,
            }
        if sid2meta:
            return sid2meta

    raise FileNotFoundError("sid 元信息不存在：请提供 sid2text.json 或 Beauty.pretrain.json")


def build_expanded_train_queries(
    sequential_file: Path,
    beauty_items_path: Path,
    min_history_len: int,
) -> pd.DataFrame:
    if not sequential_file.exists():
        raise FileNotFoundError(f"训练序列文件不存在: {sequential_file}")
    if not beauty_items_path.exists():
        raise FileNotFoundError(f"Beauty 元信息文件不存在: {beauty_items_path}")

    with beauty_items_path.open("r", encoding="utf-8") as f:
        beauty_items = json.load(f)

    rows: List[dict] = []
    with sequential_file.open("r", encoding="utf-8") as f:
        for sequence_index, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) <= 1:
                continue

            user_id = parts[0]
            item_ids = parts[1:]
            sid_sequence: List[str] = []
            for item_id in item_ids:
                meta = beauty_items.get(item_id) or {}
                sid = meta.get("sid")
                if sid:
                    sid_sequence.append(sid)

            # 与现有 train split 对齐：保留最后两个物品给 val/test，不参与 train query 展开
            train_prefix_sequence = sid_sequence[:-2]
            if len(train_prefix_sequence) < (min_history_len + 1):
                continue

            for target_idx in range(min_history_len, len(train_prefix_sequence)):
                history_sids = train_prefix_sequence[:target_idx]
                groundtruth_sid = train_prefix_sequence[target_idx]
                rows.append(
                    {
                        "user_id": user_id,
                        "description": "The user has purchased the following items: " + "; ".join(history_sids) + ";",
                        "groundtruth": groundtruth_sid,
                        "sequence_index": sequence_index,
                        "prefix_index": target_idx - min_history_len,
                        "history_length": len(history_sids),
                        "source_sequence_length": len(sid_sequence),
                    }
                )

    return pd.DataFrame(rows)


def resolve_default_model_path() -> Path:
    candidates = [
        REPO_ROOT / "train" / "results" / "ReasoningActivation" / "epoch_2",
        REPO_ROOT / "train" / "results" / "beauty_sid_rec" / "checkpoint-8388",
        REPO_ROOT / "train" / "results" / "beauty_sid_rec",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rank_candidates(m,n) training data from TwiSTAR fast_rec(50) candidates")
    parser.add_argument("--input_parquet", type=str, default="", help="单个输入 parquet；为空时配合 --split 使用默认文件")
    parser.add_argument("--output_parquet", type=str, default="", help="单个输出 parquet；为空时配合 --split 使用默认文件")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="生成哪个 split；默认 all",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(REPO_ROOT / "data"),
        help="数据目录，默认 TwiSTAR/data",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="ranking_fast_rec50_data",
        help="all 模式下的输出前缀",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(resolve_default_model_path()),
        help="用于 fast_rec(50) 的 TwiSTAR fast model 目录",
    )
    parser.add_argument(
        "--sid2text_path",
        type=str,
        default=str(REPO_ROOT / "data" / "sid2text.json"),
        help="可选 sid2text.json；不存在时自动退化到 Beauty.pretrain.json",
    )
    parser.add_argument(
        "--beauty_items_path",
        type=str,
        default=str(REPO_ROOT / "data" / "Beauty.pretrain.json"),
        help="Beauty 元信息文件，用于构造 SID 元数据",
    )
    parser.add_argument(
        "--sequential_file",
        type=str,
        default=str(REPO_ROOT / "data" / "sequential_data_processed.txt"),
        help="原始用户购买序列文件，用于展开 train 多前缀样本",
    )
    parser.add_argument("--trie_pkl_path", type=str, default="", help="可选 exact trie pkl")
    parser.add_argument("--retrieve_k", type=int, default=50, help="Recall 候选数量，默认 50")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推理设备：auto/cpu/cuda/cuda:0/all/cuda:all，或逗号分隔如 cuda:0,1,2,3,4,5,6,7",
    )
    parser.add_argument("--max_new_tokens", type=int, default=20, help="fast_rec(k) 生成 SID 的最大 token 数")
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=0,
        help="每轮送入多卡推理的 query 数；0 表示自动按 GPU 副本数推断，通常比单纯按卡数更快",
    )
    parser.add_argument(
        "--write_chunk_size",
        type=int,
        default=50000,
        help="累计多少条 ranking 记录后分块写入 parquet，降低内存占用并减少末尾一次性落盘开销",
    )
    parser.add_argument("--min_history_len", type=int, default=5, help="train 前缀展开时的最短历史长度")
    parser.add_argument("--expand_train_prefixes", dest="expand_train_prefixes", action="store_true", help="对 train split 按前缀展开多个 query")
    parser.add_argument("--no_expand_train_prefixes", dest="expand_train_prefixes", action="store_false", help="关闭 train 前缀展开，退回单 query 模式")
    parser.add_argument("--max_rows", type=int, default=0, help="仅处理前 N 行，0 表示处理全部")
    parser.add_argument("--start_row", type=int, default=0, help="从第几行开始处理")
    parser.set_defaults(expand_train_prefixes=True)
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> List[tuple[str, Path, Path]]:
    if args.input_parquet:
        if not args.output_parquet:
            raise ValueError("传入 --input_parquet 时必须同时提供 --output_parquet")
        split = args.split if args.split != "all" else "custom"
        return [(split, Path(args.input_parquet), Path(args.output_parquet))]

    data_dir = Path(args.data_dir)
    splits: Iterable[str] = [args.split] if args.split != "all" else ["train", "val", "test"]
    jobs: List[tuple[str, Path, Path]] = []
    for split in splits:
        jobs.append(
            (
                split,
                data_dir / f"training_prediction_sid_data_{split}.parquet",
                data_dir / f"{args.output_prefix}_{split}.parquet",
            )
        )
    return jobs


def build_record(
    user_id: str,
    description: str,
    split: str,
    query_index: int,
    history_sids: List[str],
    groundtruth_sid: str,
    candidate_sid: str,
    label: int,
    candidate_rank: int,
    in_recall50: bool,
    is_injected_positive: bool,
    candidate_score: float,
    score_source: str,
    sid2meta: Dict[str, dict],
    retrieve_k: int,
) -> dict:
    candidate_meta = sid2meta.get(candidate_sid, {})
    groundtruth_meta = sid2meta.get(groundtruth_sid, {})
    return {
        "split": split,
        "query_index": int(query_index),
        "user_id": str(user_id),
        "description": description,
        "history_sids_json": json.dumps(history_sids, ensure_ascii=False),
        "history_length": len(history_sids),
        "groundtruth_sid": groundtruth_sid,
        "groundtruth_title": groundtruth_meta.get("title", ""),
        "groundtruth_categories": groundtruth_meta.get("categories", ""),
        "candidate_sid": candidate_sid,
        "candidate_title": candidate_meta.get("title", ""),
        "candidate_categories": candidate_meta.get("categories", ""),
        "label": int(label),
        "candidate_rank": int(candidate_rank),
        "in_recall50": bool(in_recall50),
        "is_injected_positive": bool(is_injected_positive),
        "candidate_score": float(candidate_score),
        "score_source": score_source,
        "retrieve_k": int(retrieve_k),
        "positive_missing_from_recall": bool(not in_recall50 and label == 1),
    }


def resolve_inference_batch_size(retriever: TwiSTARFastRecTool, requested_batch_size: int) -> int:
    parallel_size = max(1, int(getattr(retriever, "parallel_size", 1)))
    if requested_batch_size and int(requested_batch_size) > 0:
        return max(parallel_size, int(requested_batch_size))
    return max(parallel_size, parallel_size * 4)


def flush_records_to_parquet(
    output_parquet: Path,
    records_buffer: List[dict],
    writer,
) -> tuple[Optional[object], int]:
    if not records_buffer:
        return writer, 0

    import pyarrow as pa
    import pyarrow.parquet as pq

    chunk_df = pd.DataFrame(records_buffer)
    records_buffer.clear()
    table = pa.Table.from_pandas(chunk_df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(str(output_parquet), table.schema)
    writer.write_table(table)
    return writer, int(table.num_rows)


def generate_ranking_data_for_split(
    split: str,
    input_parquet: Path,
    output_parquet: Path,
    retriever: TwiSTARFastRecTool,
    sid2meta: Dict[str, dict],
    retrieve_k: int,
    start_row: int,
    max_rows: int,
    expand_train_prefixes: bool,
    sequential_file: Path,
    beauty_items_path: Path,
    min_history_len: int,
    inference_batch_size: int,
    write_chunk_size: int,
) -> None:
    if split == "train" and expand_train_prefixes:
        df = build_expanded_train_queries(
            sequential_file=sequential_file,
            beauty_items_path=beauty_items_path,
            min_history_len=min_history_len,
        )
    else:
        if not input_parquet.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_parquet}")
        df = pd.read_parquet(input_parquet)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    if start_row > 0:
        df = df.iloc[start_row:]
    if max_rows > 0:
        df = df.head(max_rows)

    records_buffer: List[dict] = []
    writer = None
    written_rows = 0
    # 防止验证/测试特征泄漏：只有 train 允许在未召回时补入 ground truth 并重新打分
    allow_injected_positive = split == "train"
    batch_size = resolve_inference_batch_size(retriever, inference_batch_size)
    print(f"[{split}] parallel_size={getattr(retriever, 'parallel_size', 1)}, inference_batch_size={batch_size}")
    total_batches = (len(df) + batch_size - 1) // batch_size if len(df) > 0 else 0
    iterator = tqdm(range(0, len(df), batch_size), total=total_batches, desc=f"Building ranking data ({split})")
    for batch_start in iterator:
        batch_df = df.iloc[batch_start : batch_start + batch_size]
        batch_payloads: List[dict] = []
        for local_offset, row in enumerate(batch_df.itertuples(index=False)):
            description = str(row.description)
            history_sids = extract_sids_from_text(description)
            groundtruth_sid = str(row.groundtruth)
            if not history_sids or not groundtruth_sid:
                continue
            batch_payloads.append(
                {
                    "query_index": start_row + batch_start + local_offset,
                    "user_id": str(row.user_id),
                    "description": description,
                    "history_sids": history_sids,
                    "groundtruth_sid": groundtruth_sid,
                }
            )

        if not batch_payloads:
            continue

        retrieve_outputs = retriever.retrieve_batch([payload["history_sids"] for payload in batch_payloads], k=retrieve_k)
        pending_positive_payloads: List[dict] = []

        for payload, (candidates, recall_scores) in zip(batch_payloads, retrieve_outputs):
            query_index = int(payload["query_index"])
            user_id = payload["user_id"]
            description = payload["description"]
            history_sids = payload["history_sids"]
            groundtruth_sid = payload["groundtruth_sid"]
            seen_candidates = set()
            positive_in_recall = False

            for rank, candidate_sid in enumerate(candidates[:retrieve_k]):
                if not candidate_sid or candidate_sid in seen_candidates:
                    continue
                seen_candidates.add(candidate_sid)
                label = int(candidate_sid == groundtruth_sid)
                if label == 1:
                    positive_in_recall = True

                records_buffer.append(
                    build_record(
                        user_id=user_id,
                        description=description,
                        split=split,
                        query_index=query_index,
                        history_sids=history_sids,
                        groundtruth_sid=groundtruth_sid,
                        candidate_sid=candidate_sid,
                        label=label,
                        candidate_rank=rank,
                        in_recall50=True,
                        is_injected_positive=False,
                        candidate_score=float(recall_scores.get(candidate_sid, -1000.0)),
                        score_source="recall50",
                        sid2meta=sid2meta,
                        retrieve_k=retrieve_k,
                    )
                )

            if allow_injected_positive and not positive_in_recall:
                pending_positive_payloads.append(payload)

        if pending_positive_payloads:
            positive_scores = retriever.score_sid_batch(
                [payload["history_sids"] for payload in pending_positive_payloads],
                [payload["groundtruth_sid"] for payload in pending_positive_payloads],
            )
            for payload, positive_score in zip(pending_positive_payloads, positive_scores):
                records_buffer.append(
                    build_record(
                        user_id=payload["user_id"],
                        description=payload["description"],
                        split=split,
                        query_index=int(payload["query_index"]),
                        history_sids=payload["history_sids"],
                        groundtruth_sid=payload["groundtruth_sid"],
                        candidate_sid=payload["groundtruth_sid"],
                        label=1,
                        candidate_rank=-1,
                        in_recall50=False,
                        is_injected_positive=True,
                        candidate_score=float(positive_score),
                        score_source="forced_positive",
                        sid2meta=sid2meta,
                        retrieve_k=retrieve_k,
                    )
                )

        if len(records_buffer) >= max(1, int(write_chunk_size)):
            writer, chunk_rows = flush_records_to_parquet(output_parquet, records_buffer, writer)
            written_rows += chunk_rows

    writer, chunk_rows = flush_records_to_parquet(output_parquet, records_buffer, writer)
    written_rows += chunk_rows
    if writer is not None:
        writer.close()
    else:
        pd.DataFrame().to_parquet(output_parquet, engine="pyarrow", index=False)
    print(f"Saved {written_rows} rows to {output_parquet}")


def main() -> None:
    args = parse_args()

    sid2meta = load_sid_metadata(
        sid2text_path=Path(args.sid2text_path) if args.sid2text_path else None,
        beauty_items_path=Path(args.beauty_items_path) if args.beauty_items_path else None,
    )
    memory_query = MemoryQueryTool(sid2text_data=sid2meta)
    memory_query.load()

    trie_pkl_path = Path(args.trie_pkl_path) if args.trie_pkl_path else None
    retriever = TwiSTARFastRecTool(
        model_path=Path(args.model_path),
        memory_query=memory_query,
        trie_pkl_path=trie_pkl_path,
        build_trie_from_sid2text=True,
        device=str(args.device),
        max_new_tokens=int(args.max_new_tokens),
    )

    jobs = build_jobs(args)
    for split, input_parquet, output_parquet in jobs:
        generate_ranking_data_for_split(
            split=split,
            input_parquet=input_parquet,
            output_parquet=output_parquet,
            retriever=retriever,
            sid2meta=sid2meta,
            retrieve_k=int(args.retrieve_k),
            start_row=int(args.start_row),
            max_rows=int(args.max_rows),
            expand_train_prefixes=bool(args.expand_train_prefixes),
            sequential_file=Path(args.sequential_file),
            beauty_items_path=Path(args.beauty_items_path),
            min_history_len=int(args.min_history_len),
            inference_batch_size=int(args.inference_batch_size),
            write_chunk_size=int(args.write_chunk_size),
        )


if __name__ == "__main__":
    main()
