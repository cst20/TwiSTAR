#!/usr/bin/env python3
"""Build item-to-item (I2I) table with Swing and export at SID level.

Inputs:
  - sequential_data_processed.txt: each line: user_id item_id1 item_id2 ...
  - Beauty.pretrain.json: item_id -> {sid, title, categories, ...}

Outputs:
  - i2i_swing_top5.jsonl: one JSON per line: {"sid": ..., "topk": [{"sid":...,"score":...}, ...]}
  - sid2text.json: flat sid -> {"title":...,"categories":...}
  - sid_hierarchy.json: nested dict keyed by s_a/s_b/s_c/s_d

Notes:
  - We compute Swing on item_id co-occurrence within each user sequence.
  - Then map item_id scores to SID scores by summing over contributing item_ids.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SID_RE = re.compile(r"<\|sid_begin\|>(.*?)<\|sid_end\|>")
TOKEN_RE = re.compile(r"<(s_[abcd])_(\d+)>")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="./sequential_data_processed.txt")
    p.add_argument("--items", default="./Beauty.pretrain.json")
    p.add_argument("--out_i2i", default="./i2i_swing_top5.jsonl")
    p.add_argument("--out_sid2text", default="./sid2text.json")
    p.add_argument("--out_hierarchy", default="./sid_hierarchy.json")

    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max_users", type=int, default=0, help="0 means all users")
    p.add_argument("--max_items_per_user", type=int, default=200)

    # Swing hyperparams
    p.add_argument("--alpha", type=float, default=1.0, help="user penalty: 1/(alpha+|I_u|)")
    p.add_argument("--min_common_users", type=int, default=2, help="min common users for an i-j pair")
    return p.parse_args()


def load_items(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_sid(raw_sid: str) -> str:
    # Keep exact string as key, but strip spaces if any
    return (raw_sid or "").strip()


def sid_tokens(raw_sid: str) -> Tuple[str, str, str, str] | None:
    """Parse <|sid_begin|><s_a_x><s_b_y><s_c_z><s_d_w><|sid_end|>."""
    raw_sid = normalize_sid(raw_sid)
    m = SID_RE.search(raw_sid)
    inner = m.group(1) if m else raw_sid
    toks = {k: None for k in ("s_a", "s_b", "s_c", "s_d")}
    for pref, idx in TOKEN_RE.findall(inner):
        toks[pref] = f"{pref}_{idx}"
    if all(toks[k] for k in toks):
        return toks["s_a"], toks["s_b"], toks["s_c"], toks["s_d"]
    return None


def build_sid_maps(items: Dict[str, dict]) -> Tuple[Dict[str, str], Dict[str, dict], dict]:
    """Return item_id->sid, sid2text(flat), sid_hierarchy(nested)."""
    item2sid: Dict[str, str] = {}
    sid2text: Dict[str, dict] = {}
    hierarchy: dict = {}

    def ensure_path(sa: str, sb: str, sc: str, sd: str) -> dict:
        node = hierarchy
        for k in (sa, sb, sc, sd):
            node = node.setdefault(k, {})
        return node

    for item_id, info in items.items():
        sid = normalize_sid(info.get("sid", ""))
        if not sid:
            continue
        item2sid[item_id] = sid
        # flat mapping
        if sid not in sid2text:
            sid2text[sid] = {
                "title": info.get("title", ""),
                "categories": info.get("categories", ""),
            }
        # hierarchy
        toks = sid_tokens(sid)
        if toks:
            sa, sb, sc, sd = toks
            leaf = ensure_path(sa, sb, sc, sd)
            # store a minimal payload at leaf
            leaf.setdefault("_sid", sid)
            leaf.setdefault("_title", info.get("title", ""))
            leaf.setdefault("_categories", info.get("categories", ""))

    return item2sid, sid2text, hierarchy


def iter_user_sequences(seq_path: Path, max_users: int, max_items_per_user: int) -> Iterable[List[str]]:
    with seq_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_users and idx >= max_users:
                break
            parts = line.strip().split()
            if len(parts) <= 2:
                continue
            items = parts[1:]
            if max_items_per_user and len(items) > max_items_per_user:
                items = items[:max_items_per_user]
            # de-dup within user (order not needed for co-occurrence)
            yield list(dict.fromkeys(items))


def swing_i2i(
    user_items: Iterable[List[str]],
    alpha: float,
    min_common_users: int,
) -> Dict[str, Dict[str, float]]:
    """Compute Swing scores on item_id.

    score(i,j) = sum_{u in U(i)∩U(j)} 1/(alpha + |I_u|)
    This is a simplified, common Swing-style user penalty.
    """
    # pair -> (count_common_users, score)
    pair_cnt: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_score: Dict[Tuple[str, str], float] = defaultdict(float)

    for items in user_items:
        n = len(items)
        if n < 2:
            continue
        w = 1.0 / (alpha + n)
        # O(n^2) within user; ok for moderate n (we cap max_items_per_user)
        for i in range(n):
            a = items[i]
            for j in range(i + 1, n):
                b = items[j]
                if a == b:
                    continue
                x, y = (a, b) if a < b else (b, a)
                pair_cnt[(x, y)] += 1
                pair_score[(x, y)] += w

    # build adjacency
    adj: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (x, y), c in pair_cnt.items():
        if c < min_common_users:
            continue
        s = pair_score[(x, y)]
        adj[x][y] = s
        adj[y][x] = s
    return adj


def aggregate_to_sid(
    item_adj: Dict[str, Dict[str, float]],
    item2sid: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    sid_adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for item_i, nbrs in item_adj.items():
        sid_i = item2sid.get(item_i)
        if not sid_i:
            continue
        for item_j, score in nbrs.items():
            sid_j = item2sid.get(item_j)
            if not sid_j or sid_j == sid_i:
                continue
            sid_adj[sid_i][sid_j] += float(score)
    # convert nested defaultdict to dict
    return {k: dict(v) for k, v in sid_adj.items()}


def topk_neighbors(sid_adj: Dict[str, Dict[str, float]], topk: int) -> Iterable[Tuple[str, List[Tuple[str, float]]]]:
    for sid, nbrs in sid_adj.items():
        items = sorted(nbrs.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        yield sid, items


def main() -> int:
    args = parse_args()
    seq_path = Path(args.seq)
    items_path = Path(args.items)
    out_i2i = Path(args.out_i2i)
    out_sid2text = Path(args.out_sid2text)
    out_hierarchy = Path(args.out_hierarchy)

    items = load_items(items_path)
    item2sid, sid2text, hierarchy = build_sid_maps(items)

    # Save sid maps first
    out_sid2text.write_text(json.dumps(sid2text, ensure_ascii=False), encoding="utf-8")
    out_hierarchy.write_text(json.dumps(hierarchy, ensure_ascii=False), encoding="utf-8")

    user_items = list(iter_user_sequences(seq_path, args.max_users, args.max_items_per_user))
    print(f"Loaded users: {len(user_items)}")

    item_adj = swing_i2i(user_items, alpha=args.alpha, min_common_users=args.min_common_users)
    print(f"Item nodes with neighbors: {len(item_adj)}")

    sid_adj = aggregate_to_sid(item_adj, item2sid)
    print(f"SID nodes with neighbors: {len(sid_adj)}")

    out_i2i.parent.mkdir(parents=True, exist_ok=True)
    with out_i2i.open("w", encoding="utf-8") as f:
        for sid, topk in topk_neighbors(sid_adj, args.topk):
            recs = [{"sid": s, "score": float(v)} for s, v in topk]
            f.write(json.dumps({"sid": sid, "topk": recs}, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_i2i}")
    print(f"Wrote: {out_sid2text}")
    print(f"Wrote: {out_hierarchy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

