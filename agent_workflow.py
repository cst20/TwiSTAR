#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TwiSTAR multi-tool recommendation agent workflow（Next-Item Prediction）。

This file keeps the inference-time tool interface aligned with the paper:

1) ``fast_rec(k)``: a fast non-reasoning SID generator retrieves top-k SIDs.
2) ``rank_candidates(m, n)``: a ranking model reranks the latest m candidates and
   returns the top-n candidates.
3) ``think_and_rec(j)``: a slow reasoning model directly recommends top-j SIDs.

The planner/controller is intentionally separated from the tools.  In the paper it
is trained by a two-stage recipe (supervised warm-up from tool-call labels, then
agentic RL such as GRPO/PPO with NDCG/latency/valid-tool rewards).  At inference
time this module can either use a trained controller that emits JSON tool-calls or
fall back to deterministic policies for debugging/reproduction.

TwiSTAR SID generation uses HF ``generate`` with an optional SID prefix trie
constraint; ranking can be DIN or an LLM scorer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
import re
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


SID_PATTERN = re.compile(r"<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>")


def _repo_rel_path(*parts: str) -> Path:
    # TwiStar/agent_workflow.py 所在目录
    here = Path(__file__).resolve().parent
    return here.joinpath(*parts)


def extract_sids_from_text(text: str) -> List[str]:
    if not text:
        return []
    return SID_PATTERN.findall(text)


def extract_first_sid_from_generation(text: str) -> Optional[str]:
    if not text:
        return None
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = SID_PATTERN.search(str(text).strip().replace(" ", ""))
    if m:
        return m.group(0)
    return None


def dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def parse_categories(cats) -> List[str]:
    """兼容sid2text里的categories字段（list/str/其它）。"""
    if not cats:
        return []
    if isinstance(cats, list):
        return [str(x) for x in cats if x]
    # 兼容字符串或其它格式
    return [c.strip() for c in str(cats).split(",") if c.strip()]


@dataclass
class ShortTermMemory:
    """短期记忆：对最近行为做时间衰减，返回每个历史SID的兴趣权重。"""

    max_len: int = 20
    decay: float = 0.85

    def update(self, history_sids: Sequence[str]) -> List[str]:
        # 取去重后的最后max_len个，避免重复点击/购买把权重无限放大
        history = dedup_keep_order([s for s in history_sids if s])
        if len(history) > self.max_len:
            history = history[-self.max_len :]
        return history

    def interest_weights(self, history_sids: Sequence[str]) -> Dict[str, float]:
        history = self.update(history_sids)
        # 最近的权重大；最后一个行为的权重为1.0
        weights: Dict[str, float] = {}
        # 从最旧到最新：w = decay^(distance_to_end)
        for idx, sid in enumerate(history):
            dist = (len(history) - 1) - idx
            weights[sid] = float(self.decay ** dist)
        return weights


class I2IKnowledge:
    """I2I长期记忆：SID -> 邻居SID列表（带分数）。"""

    def __init__(self, i2i_jsonl_path: Path):
        self.i2i_jsonl_path = Path(i2i_jsonl_path)
        self._neighbors: Dict[str, List[Tuple[str, float]]] = {}
        self._global_pop: Dict[str, float] = {}

    def load(self) -> None:
        if not self.i2i_jsonl_path.exists():
            raise FileNotFoundError(f"I2I文件不存在: {self.i2i_jsonl_path}")

        neighbors: Dict[str, List[Tuple[str, float]]] = {}
        global_pop: Dict[str, float] = {}

        with self.i2i_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sid = obj.get("sid")
                topk = obj.get("topk") or []
                if not sid:
                    continue
                lst: List[Tuple[str, float]] = []
                for t in topk:
                    tsid = t.get("sid")
                    score = t.get("score", 0.0)
                    if not tsid:
                        continue
                    s = float(score)
                    lst.append((tsid, s))
                    # 简单全局热度：记录某个sid在所有topk里出现的最高相似度
                    prev = global_pop.get(tsid)
                    if prev is None or s > prev:
                        global_pop[tsid] = s
                neighbors[sid] = lst

        self._neighbors = neighbors
        self._global_pop = global_pop

    def neighbors(self, sid: str) -> List[Tuple[str, float]]:
        return self._neighbors.get(sid, [])

    def global_popularity(self, sid: str) -> float:
        return float(self._global_pop.get(sid, 0.0))

    def global_popular_sids(self) -> List[str]:
        return [k for k, _ in sorted(self._global_pop.items(), key=lambda kv: kv[1], reverse=True)]


class MemoryQueryTool:
    """工具：查询记忆模块（I2I邻居 + SID元信息）。"""

    def __init__(
        self,
        i2i: Optional[I2IKnowledge] = None,
        sid2text_path: Optional[Path] = None,
        sid2text_data: Optional[Dict[str, dict]] = None,
    ):
        self.i2i = i2i
        self.sid2text_path = Path(sid2text_path) if sid2text_path else None
        self._sid2text_seed = sid2text_data
        self._sid2text: Optional[Dict[str, dict]] = None
        self._cat2sids: Dict[str, List[str]] = {}

    def load(self) -> None:
        if self._sid2text_seed is not None:
            self._sid2text = dict(self._sid2text_seed)
        elif self.sid2text_path and self.sid2text_path.exists():
            with self.sid2text_path.open("r", encoding="utf-8") as f:
                self._sid2text = json.load(f)
        else:
            self._sid2text = {}

        # 构建类别倒排：category -> [sid, ...]
        cat2sids: Dict[str, List[str]] = {}
        for sid, meta in (self._sid2text or {}).items():
            for c in parse_categories((meta or {}).get("categories")):
                cat2sids.setdefault(c, []).append(sid)
        self._cat2sids = cat2sids

    def all_sids(self) -> List[str]:
        if not self._sid2text:
            return []
        return list(self._sid2text.keys())

    def get_item_meta(self, sid: str) -> dict:
        if not self._sid2text:
            return {}
        return self._sid2text.get(sid, {}) or {}

    def get_i2i(self, sid: str, topk: int = 50) -> List[Tuple[str, float]]:
        if not self.i2i:
            return []
        nbrs = self.i2i.neighbors(sid)
        if not nbrs:
            return []
        return nbrs[:topk]

    def global_popularity(self, sid: str) -> float:
        if not self.i2i:
            return 0.0
        return self.i2i.global_popularity(sid)

    def global_popular_sids(self) -> List[str]:
        if not self.i2i:
            return []
        return self.i2i.global_popular_sids()

    def get_category_pool(self, sid: str, limit_per_cat: int = 200) -> List[str]:
        """返回与sid共享类别的候选池（不含打分）。"""
        meta = self.get_item_meta(sid)
        out: List[str] = []
        for c in parse_categories(meta.get("categories")):
            sids = self._cat2sids.get(c) or []
            if limit_per_cat > 0:
                sids = sids[:limit_per_cat]
            out.extend(sids)
        return dedup_keep_order(out)


class RetrieveTool:
    """工具：召回模型（基于I2I + 短期兴趣聚合），默认召回50个SID。"""

    def __init__(self, memory_query: MemoryQueryTool, short_term: ShortTermMemory):
        self.memory_query = memory_query
        self.short_term = short_term

    def retrieve(self, history_sids: Sequence[str], k: int = 50) -> Tuple[List[str], Dict[str, float]]:
        history_sids = [s for s in history_sids if s]
        seen = set(history_sids)
        weights = self.short_term.interest_weights(history_sids)

        agg: Dict[str, float] = {}
        for hs, w in weights.items():
            for cand, s in self.memory_query.get_i2i(hs, topk=max(k, 50)):
                if cand in seen:
                    continue
                agg[cand] = agg.get(cand, 0.0) + w * float(s)

        # I2I覆盖不足时，补一层“同类目候选”作为弱召回（分数较小）
        if len(agg) < k:
            cat_boost = 0.05
            for hs, w in weights.items():
                for cand in self.memory_query.get_category_pool(hs, limit_per_cat=200):
                    if cand in seen:
                        continue
                    # 避免盖过I2I：只做轻量增益
                    agg[cand] = agg.get(cand, 0.0) + float(w) * cat_boost

        # I2I覆盖不足时，用全局热门补齐
        ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        cands = [sid for sid, _ in ranked]

        if len(cands) < k:
            for sid in self.memory_query.global_popular_sids():
                if sid in seen or sid in agg:
                    continue
                agg[sid] = 0.0
                cands.append(sid)
                if len(cands) >= k:
                    break

        # 仍不足时，用全量sid池补齐（保证retrieve_k=50的契约）
        if len(cands) < k:
            for sid in self.memory_query.all_sids():
                if sid in seen or sid in agg:
                    continue
                agg[sid] = 0.0
                cands.append(sid)
                if len(cands) >= k:
                    break

        return cands[:k], agg


class LightweightRanker:
    """轻量ranker：用少量手工特征 + 可加载权重做打分。"""

    def __init__(
        self,
        memory_query: MemoryQueryTool,
        weights: Optional[Mapping[str, float]] = None,
    ):
        # 默认权重：更看重I2I聚合分，其次类别重合，再加一点全局热度
        self.w = {
            "i2i_sum": 1.0,
            "i2i_max": 0.3,
            "pop": 0.1,
            "cat": 0.2,
        }
        if weights:
            for k, v in dict(weights).items():
                try:
                    self.w[k] = float(v)
                except Exception:
                    continue
        self.memory_query = memory_query

    def _categories(self, sid: str) -> List[str]:
        meta = self.memory_query.get_item_meta(sid)
        return parse_categories(meta.get("categories"))

    def score(
        self,
        history_sids: Sequence[str],
        candidate_sid: str,
        retrieve_score: float,
    ) -> float:
        # i2i_max：候选与任一历史物品的最大I2I分
        i2i_max = 0.0
        for hs in history_sids[-20:]:
            for nbr, s in self.memory_query.get_i2i(hs, topk=200):
                if nbr == candidate_sid:
                    if float(s) > i2i_max:
                        i2i_max = float(s)
                    break

        # cat overlap：候选与最近几个物品类别是否重合
        cand_cats = set(self._categories(candidate_sid))
        if cand_cats:
            recent = history_sids[-5:]
            hit = 0
            for hs in recent:
                hc = set(self._categories(hs))
                if hc and (hc & cand_cats):
                    hit += 1
            cat_overlap = hit / max(1, len(recent))
        else:
            cat_overlap = 0.0

        # i2i可能未加载（onerec召回场景）
        pop = self.memory_query.global_popularity(candidate_sid)

        return (
            self.w["i2i_sum"] * float(retrieve_score)
            + self.w["i2i_max"] * float(i2i_max)
            + self.w["pop"] * float(pop)
            + self.w["cat"] * float(cat_overlap)
        )

    def rank(
        self,
        history_sids: Sequence[str],
        candidate_sids: Sequence[str],
        retrieve_scores: Mapping[str, float],
        topk: int = 20,
    ) -> List[Tuple[str, float]]:
        scored = []
        for sid in candidate_sids:
            rs = float(retrieve_scores.get(sid, 0.0))
            scored.append((sid, self.score(history_sids, sid, rs)))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:topk]


class NextItemAgent:
    """Agent：编排工具调用，完成 next-item 预测。"""

    def __init__(
        self,
        memory_query: MemoryQueryTool,
        retriever: RetrieveTool,
        ranker: LightweightRanker,
    ):
        self.memory_query = memory_query
        self.retriever = retriever
        self.ranker = ranker

    @staticmethod
    def _use_raw_beam_order(retriever: Any) -> bool:
        return isinstance(retriever, TwiSTARFastRecTool)

    def predict_next_items(self, history_sids: Sequence[str], retrieve_k: int = 50, topk: int = 20) -> List[str]:
        # 1) 更新短期记忆已在retrieve内部完成
        # 2) 召回
        candidates, retrieve_scores = self.retriever.retrieve(history_sids, k=retrieve_k)
        if self._use_raw_beam_order(self.retriever):
            return list(candidates[:topk])
        # 3) 排序
        ranked = self.ranker.rank(history_sids, candidates, retrieve_scores, topk=topk)
        return [sid for sid, _ in ranked]


class LLMRankingModel:
    """Ranking Model（LLM版）：给定 history + candidates，计算每个候选的偏好分。

    实现为可复用的 TwiSTAR ranking tool，并支持候选>10时分桶打分。

    打分方式：构造多选题（A/B/C...），取模型对“下一 token 是某个字母”的 logit 作为分数。
    注意：这不是生成，而是一次 forward 得到 next-token logits，因此速度更可控。
    """

    def __init__(
        self,
        model_path: Path,
        sid2text: Mapping[str, dict],
        device: str = "auto",
        max_candidates_per_call: int = 10,
    ):
        self.model_path = Path(model_path)
        self.sid2text = dict(sid2text or {})
        self.device = str(device or "auto")
        self.max_candidates_per_call = int(max(2, min(10, max_candidates_per_call)))
        self._tokenizer = None
        self._model = None
        self._letter_token_ids: Dict[str, int] = {}

    def _lazy_init(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.model_path.exists():
            raise FileNotFoundError(f"Ranking Model 路径不存在: {self.model_path}")

        tok = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        use_cuda = torch.cuda.is_available()
        if self.device.lower() == "auto":
            dev = "cuda" if use_cuda else "cpu"
        else:
            dev = self.device

        dtype = torch.bfloat16 if str(dev).startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(dev)
        model.eval()

        # 预计算字母 token id（与 `infer_ranking_tool.py` 一致）
        letter_token_ids: Dict[str, int] = {}
        for letter in list("ABCDEFGHIJ"):
            token_id = tok.encode(letter, add_special_tokens=False)[-1]
            letter_token_ids[letter] = int(token_id)

        self._tokenizer = tok
        self._model = model
        self._letter_token_ids = letter_token_ids

    def _meta_text(self, sid: str) -> Tuple[str, str]:
        meta = self.sid2text.get(sid, {}) or {}
        title = str(meta.get("title", "Unknown") or "Unknown")
        cat = meta.get("categories", "Unknown")
        if isinstance(cat, list):
            cat = ", ".join([str(x) for x in cat if x])
        cat = str(cat or "Unknown")
        return title, cat

    def _build_prompt(self, history_sids: Sequence[str], candidate_sids: Sequence[str]) -> Tuple[str, List[str]]:
        letters = list("ABCDEFGHIJ")
        candidate_sids = list(candidate_sids)[: len(letters)]
        valid_letters = letters[: len(candidate_sids)]

        history_lines = ["User purchase history:"]
        for i, sid in enumerate(history_sids):
            title, cat = self._meta_text(sid)
            history_lines.append(f"{i+1}. {sid} (Title: {title}, Category: {cat})")

        cand_lines = ["Candidates for next purchase:"]
        for i, sid in enumerate(candidate_sids):
            title, cat = self._meta_text(sid)
            cand_lines.append(f"{valid_letters[i]}. {sid} (Title: {title}, Category: {cat})")

        prompt = (
            "Based on the user's historical preferences, which of the following candidate items is the user most likely to purchase next?\n\n"
            + "\n".join(history_lines)
            + "\n\n"
            + "\n".join(cand_lines)
            + "\n\nPlease output the letter of the most likely candidate (e.g., A, B, C, D, E)."
        )
        return prompt, valid_letters

    def score(
        self,
        history_sids: Sequence[str],
        candidate_sids: Sequence[str],
        fast_scores: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, float]:
        self._lazy_init()
        assert self._tokenizer is not None and self._model is not None

        import torch

        scores: Dict[str, float] = {}
        history = [s for s in history_sids if s]
        candidates = [s for s in candidate_sids if s]
        if not history or not candidates:
            return scores

        # 分桶：每次最多10个候选（A..J）
        for start in range(0, len(candidates), self.max_candidates_per_call):
            chunk = candidates[start : start + self.max_candidates_per_call]
            prompt, valid_letters = self._build_prompt(history, chunk)
            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self._model(**inputs)
            next_token_logits = out.logits[0, -1, :]

            for i, sid in enumerate(chunk):
                letter = valid_letters[i]
                tid = self._letter_token_ids[letter]
                scores[sid] = float(next_token_logits[tid].item())

        return scores


class DINRankingModel:
    """Ranking Model（DIN版）：对候选进行粗排。

    该实现与 `train/scripts/train_din_ranking.py` 的模型结构保持一致（包含 candidate_score 特征），
    因此可直接加载训练得到的 `din_ranking.pth`。
    """

    class _DINAttention(torch.nn.Module):
        def __init__(self, embedding_size: int):
            super().__init__()
            self.fc1 = torch.nn.Linear(embedding_size * 4, 36)
            self.fc2 = torch.nn.Linear(36, 1)

        def forward(self, query, keys, keys_length):
            # query: [B, 1, D], keys: [B, L, D]
            batch_size, max_seq_len, _ = keys.size()
            queries = query.expand(-1, max_seq_len, -1)
            concat_input = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
            attention_score = torch.relu(self.fc1(concat_input))
            attention_score = self.fc2(attention_score)

            mask = torch.arange(max_seq_len, device=keys.device).expand(batch_size, max_seq_len) >= keys_length.unsqueeze(1)
            attention_score = attention_score.squeeze(-1)
            attention_score = attention_score.masked_fill(mask, -1e9)
            attention_weight = torch.softmax(attention_score, dim=-1)
            output = torch.bmm(attention_weight.unsqueeze(1), keys)
            return output.squeeze(1)

    class _DIN(torch.nn.Module):
        def __init__(self, num_items: int, embedding_dim: int = 64):
            super().__init__()
            self.item_emb = torch.nn.Embedding(num_items, embedding_dim, padding_idx=0)
            self.attention = DINRankingModel._DINAttention(embedding_dim)
            # 与 train_din_ranking.py 一致：拼接 candidate_score 1维特征
            self.fc1 = torch.nn.Linear(embedding_dim * 2 + 1, 64)
            self.fc2 = torch.nn.Linear(64, 16)
            self.out = torch.nn.Linear(16, 1)

        def forward(self, history, history_length, target_item, candidate_score=None):
            hist_emb = self.item_emb(history)
            target_emb = self.item_emb(target_item).unsqueeze(1)
            user_rep = self.attention(target_emb, hist_emb, history_length)
            target_emb = target_emb.squeeze(1)

            if candidate_score is None:
                candidate_score = torch.zeros(target_emb.size(0), device=target_emb.device, dtype=target_emb.dtype)
            candidate_score = candidate_score.view(-1, 1).to(target_emb.dtype)

            concat_features = torch.cat([user_rep, target_emb, candidate_score], dim=-1)
            x = torch.relu(self.fc1(concat_features))
            x = torch.relu(self.fc2(x))
            out = self.out(x)
            return out.squeeze(-1)

    def __init__(
        self,
        model_path: Path,
        sid2id_path: Path,
        device: str = "auto",
        embedding_dim: int = 64,
        max_seq_len: int = 20,
    ):
        self.model_path = Path(model_path)
        self.sid2id_path = Path(sid2id_path)
        self.device = str(device or "auto")
        self.embedding_dim = int(embedding_dim)
        self.max_seq_len = int(max_seq_len)

        self._sid2id: Optional[Dict[str, int]] = None
        self._model: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None

    def _lazy_init(self) -> None:
        if self._model is not None and self._sid2id is not None:
            return

        if not self.sid2id_path.exists():
            raise FileNotFoundError(f"DIN sid2id 映射不存在: {self.sid2id_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"DIN 模型权重不存在: {self.model_path}")

        with self.sid2id_path.open("r", encoding="utf-8") as f:
            sid2id = json.load(f)
        if not isinstance(sid2id, dict) or not sid2id:
            raise ValueError(f"DIN sid2id 映射非法或为空: {self.sid2id_path}")

        # device
        use_cuda = torch.cuda.is_available()
        if self.device.lower() == "auto":
            dev = torch.device("cuda" if use_cuda else "cpu")
        else:
            dev = torch.device(self.device)

        num_items = int(len(sid2id)) + 1  # 0 预留给 padding
        model = DINRankingModel._DIN(num_items=num_items, embedding_dim=self.embedding_dim)

        state = torch.load(str(self.model_path), map_location=dev)
        # 允许轻微不匹配（例如历史版本多/少了某些参数），但默认应完全匹配
        model.load_state_dict(state, strict=False)
        model.to(dev)
        model.eval()

        self._sid2id = {str(k): int(v) for k, v in sid2id.items()}
        self._model = model
        self._device = dev

    def _encode_history(self, history_sids: Sequence[str]) -> Tuple[List[int], int]:
        assert self._sid2id is not None
        history_ids = [self._sid2id.get(sid, 0) for sid in history_sids]
        if len(history_ids) > self.max_seq_len:
            history_ids = history_ids[-self.max_seq_len :]
            hist_len = self.max_seq_len
        else:
            hist_len = len(history_ids)
            history_ids = history_ids + [0] * (self.max_seq_len - len(history_ids))
        return history_ids, hist_len

    def score(
        self,
        history_sids: Sequence[str],
        candidate_sids: Sequence[str],
        fast_scores: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, float]:
        self._lazy_init()
        assert self._model is not None and self._sid2id is not None and self._device is not None

        history = [s for s in history_sids if s]
        candidates = [s for s in candidate_sids if s]
        if not history or not candidates:
            return {}

        history_ids, hist_len = self._encode_history(history)
        hist_tensor = torch.tensor(history_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        hist_len_tensor = torch.tensor([hist_len], dtype=torch.long, device=self._device)

        cand_ids = [self._sid2id.get(sid, 0) for sid in candidates]
        # batch: N candidates
        batch_size = len(cand_ids)
        hist_tensor_expanded = hist_tensor.repeat_interleave(batch_size, dim=0)
        hist_len_expanded = hist_len_tensor.repeat_interleave(batch_size, dim=0)
        targets_tensor = torch.tensor(cand_ids, dtype=torch.long, device=self._device)

        if fast_scores is None:
            cand_score_tensor = torch.zeros(batch_size, dtype=torch.float32, device=self._device)
        else:
            cand_score_tensor = torch.tensor(
                [float(fast_scores.get(sid, 0.0) or 0.0) for sid in candidates],
                dtype=torch.float32,
                device=self._device,
            )

        with torch.no_grad():
            logits = self._model(hist_tensor_expanded, hist_len_expanded, targets_tensor, candidate_score=cand_score_tensor)
            # 不做 sigmoid，直接用 logit 作为排序分数
            scores = logits.detach().float().cpu().tolist()

        return {sid: float(sc) for sid, sc in zip(candidates, scores)}


class SlowModelScorer:
    """Slow reasoning model wrapper.

    Paper-facing tool: ``think_and_rec(j)`` directly generates top-j SIDs with a
    reasoning prompt.  For ablation/evaluation compatibility we also keep
    ``candidate_logprob_ablation()`` for offline analysis only.
    """

    def __init__(self, scorer_tool: "TwiSTARFastRecTool", normalize_by_length: bool = True):
        self.scorer_tool = scorer_tool
        self.normalize_by_length = bool(normalize_by_length)

    def think_and_rec(self, history_sids: Sequence[str], j: int = 20) -> List[str]:
        """Implements the paper tool ``think_and_rec(j)``.

        The underlying slow model is loaded with ``prompt_style='reasoning'`` so
        generation can produce a non-empty ``<think>...</think>`` block before
        emitting legal SID tokens under the trie constraint.
        """

        candidates, _ = self.scorer_tool.retrieve(history_sids, k=int(j))
        seen = set([s for s in history_sids if s])
        return [sid for sid in candidates if sid and sid not in seen][: int(j)]

    def candidate_logprob_ablation(self, history_sids: Sequence[str], candidate_sids: Sequence[str], topk: int = 20) -> List[Tuple[str, float]]:
        history = [s for s in history_sids if s]
        candidates = [s for s in candidate_sids if s]
        if not history or not candidates:
            return []

        # 批量打分
        histories = [history for _ in candidates]
        scores = self.scorer_tool.score_sid_batch(histories, candidates, normalize_by_length=self.normalize_by_length)
        ranked = sorted(zip(candidates, scores), key=lambda kv: kv[1], reverse=True)
        return ranked[: int(topk)]


class MultiModelRecAgent:
    """TwiSTAR planner over the three paper tools.

    The exposed tool names and arguments intentionally match the manuscript:

    - ``fast_rec(k)``
    - ``rank_candidates(m, n)``
    - ``think_and_rec(j)``

    A trained planner should be obtained with the paper's two-stage procedure:
    first supervised warm-up on tool-call demonstrations (the planner may output
    either a single JSON tool-call or ``{"tool_calls": [...]}``), then agentic
    RL (GRPO/PPO) with recommendation quality, latency cost and valid-tool
    rewards.  This class executes those inference-time tool calls.  If no trained
    planner is provided, ``planner_policy`` selects a deterministic ablation
    policy.
    """

    def __init__(
        self,
        fast_model: TwiSTARFastRecTool,
        ranking_model: Optional[LLMRankingModel],
        slow_model: SlowModelScorer,
        controller_model: Optional[TwiSTARFastRecTool] = None,
        controller_prompt_prefix: str = "",
        planner_policy: str = "fast_rank",
    ):
        self.fast_model = fast_model
        self.ranking_model = ranking_model
        self.slow_model = slow_model
        self.controller_model = controller_model
        self.controller_prompt_prefix = str(controller_prompt_prefix or "").strip()
        self.planner_policy = str(planner_policy or "fast_rank").strip().lower()

    @staticmethod
    def _json_from_text(text: str) -> Optional[dict]:
        if not text:
            return None
        m = JSON_BLOCK_RE.search(text)
        if not m:
            return None
        blob = m.group(0)
        try:
            obj = json.loads(blob)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def fast_rec(self, history_sids: Sequence[str], k: int = 50) -> Tuple[List[str], Dict[str, float]]:
        """Paper tool ``fast_rec(k)``: fast SID retrieval/generation."""

        candidates, fast_scores = self.fast_model.retrieve(history_sids, k=int(k))
        candidates = [s for s in candidates if s and s not in set(history_sids)]
        return candidates[: int(k)], {sid: float(fast_scores.get(sid, 0.0)) for sid in candidates}

    def rank_candidates(
        self,
        history_sids: Sequence[str],
        candidate_sids: Sequence[str],
        fast_scores: Optional[Mapping[str, float]] = None,
        m: int = 50,
        n: int = 20,
    ) -> List[str]:
        """Paper tool ``rank_candidates(m, n)``: rerank m candidates to top-n."""

        candidates = [s for s in candidate_sids if s and s not in set(history_sids)][: int(m)]
        if not candidates:
            return []
        fast_scores = fast_scores or {}
        if self.ranking_model is not None:
            rank_scores = self.ranking_model.score(history_sids, candidates, fast_scores=fast_scores)
            ranked = sorted(
                candidates,
                key=lambda sid: float(rank_scores.get(sid, fast_scores.get(sid, 0.0))),
                reverse=True,
            )
        else:
            ranked = sorted(candidates, key=lambda sid: float(fast_scores.get(sid, 0.0)), reverse=True)
        return ranked[: int(n)]

    def think_and_rec(self, history_sids: Sequence[str], j: int = 20) -> List[str]:
        """Paper tool ``think_and_rec(j)``: slow reasoning recommendation."""

        return self.slow_model.think_and_rec(history_sids, j=int(j))

    def _run_fallback_policy(
        self,
        history_sids: Sequence[str],
        retrieve_k: int,
        rank_m: int,
        topk: int,
    ) -> List[str]:
        """Deterministic policy used when no trained planner is supplied."""

        policy = self.planner_policy
        if policy in {"slow", "fixed_slow", "think"}:
            return self.think_and_rec(history_sids, j=topk)

        candidates, fast_scores = self.fast_rec(history_sids, k=retrieve_k)
        if not candidates:
            return []
        if policy in {"fast", "fixed_fast", "fast_only"}:
            return candidates[: int(topk)]

        # Default paper-compatible light path: fast_rec(k) -> rank_candidates(m, n).
        return self.rank_candidates(
            history_sids,
            candidates,
            fast_scores=fast_scores,
            m=int(min(retrieve_k, rank_m)),
            n=int(topk),
        )

    def _controller_prompt(self, history_sids: Sequence[str], state: Mapping[str, Any]) -> str:
        # controller 模型只负责“决定调用什么工具”；为了稳健，我们要求它输出 JSON。
        tools_desc = (
            "You are the TwiSTAR planner. Output a single JSON object containing either one tool-call "
            "or a `tool_calls` list.\n"
            "Available tools:\n"
            "1) fast_rec: {\"tool\":\"fast_rec\", \"arguments\":{\"k\":50}}\n"
            "2) rank_candidates: {\"tool\":\"rank_candidates\", \"arguments\":{\"m\":50, \"n\":20}}\n"
            "3) think_and_rec: {\"tool\":\"think_and_rec\", \"arguments\":{\"j\":20}}\n"
            "Valid examples:\n"
            "- {\"tool_calls\":[{\"tool\":\"fast_rec\",\"arguments\":{\"k\":10}}]}\n"
            "- {\"tool_calls\":[{\"tool\":\"fast_rec\",\"arguments\":{\"k\":50}},"
            "{\"tool\":\"rank_candidates\",\"arguments\":{\"m\":50,\"n\":10}}]}\n"
            "- {\"tool_calls\":[{\"tool\":\"think_and_rec\",\"arguments\":{\"j\":10}}]}\n"
            "Rules: output JSON only; do not invent other tools."
        )
        hist = [s for s in history_sids if s]
        obs = json.dumps(dict(state), ensure_ascii=False)
        prefix = (self.controller_prompt_prefix + "\n\n") if self.controller_prompt_prefix else ""
        return prefix + (
            tools_desc
            + "\n\nUser purchase history SIDs:\n"
            + "\n".join(hist)
            + "\n\nCurrent state (observations):\n"
            + obs
            + "\n"
        )

    def _run_with_controller(
        self,
        history_sids: Sequence[str],
        retrieve_k: int,
        rank_m: int,
        topk: int,
        max_steps: int = 4,
    ) -> List[str]:
        # controller 失败时直接回退 deterministic policy
        if self.controller_model is None:
            return self._run_fallback_policy(history_sids, retrieve_k, rank_m, topk)

        state: Dict[str, Any] = {}
        candidates: List[str] = []
        fast_scores: Dict[str, float] = {}
        ranked0: List[str] = []

        for _ in range(int(max_steps)):
            prompt = self._controller_prompt(history_sids, state)
            # 用 controller_model 生成一个 SID 约束并不合适，这里直接走 unconstrained generate：
            # 我们复用 score_sid 的 tokenizer/model，但调用 generate。为了不入侵现有类，直接调用 fast_model 的 tokenizer/model。
            # 若 controller 输出无法解析 JSON，则降级。
            try:
                # 复用 fast_model 的 chat prompt 格式，让输出更稳定
                controller = self.controller_model
                controller._lazy_init()  # noqa: SLF001
                tok = controller._tokenizer  # noqa: SLF001
                model = controller._model  # noqa: SLF001
                assert tok is not None and model is not None

                inputs = tok(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                )
                text = tok.decode(out[0], skip_special_tokens=False)
            except Exception:
                return self._run_fallback_policy(history_sids, retrieve_k, rank_m, topk)

            obj = self._json_from_text(text)
            if not obj:
                return self._run_fallback_policy(history_sids, retrieve_k, rank_m, topk)

            raw_calls = obj.get("tool_calls") if isinstance(obj.get("tool_calls"), list) else [obj]
            calls = [c for c in raw_calls if isinstance(c, dict)]
            if not calls:
                return self._run_fallback_policy(history_sids, retrieve_k, rank_m, topk)

            for call in calls:
                tool = str(call.get("tool", "")).strip()
                args = call.get("arguments") if isinstance(call.get("arguments"), dict) else call
                if tool == "fast_rec":
                    rk = int(args.get("k", args.get("retrieve_k", retrieve_k)))
                    candidates, fast_scores = self.fast_rec(history_sids, k=rk)
                    state.update({"fast_rec": {"k": rk, "num_candidates": len(candidates)}})
                    ranked0 = []
                elif tool == "rank_candidates":
                    if not candidates:
                        state.update({"rank_candidates": {"error": "no_candidates"}})
                        return self._run_fallback_policy(history_sids, retrieve_k, rank_m, topk)
                    mm = int(args.get("m", rank_m))
                    nn = int(args.get("n", topk))
                    ranked0 = self.rank_candidates(history_sids, candidates, fast_scores=fast_scores, m=mm, n=nn)
                    state.update({"rank_candidates": {"m": mm, "n": nn, "num_shortlist": len(ranked0)}})
                elif tool == "think_and_rec":
                    jj = int(args.get("j", topk))
                    state.update({"think_and_rec": {"j": jj}})
                    return self.think_and_rec(history_sids, j=jj)
                else:
                    return self._run_fallback_policy(history_sids, retrieve_k, rank_m, topk)

            if ranked0:
                return ranked0[:topk]
            if candidates:
                return candidates[:topk]
            return self._run_fallback_policy(history_sids, retrieve_k, rank_m, topk)

        return self._run_fallback_policy(history_sids, retrieve_k, rank_m, topk)

    def recommend(
        self,
        history_sids: Sequence[str],
        retrieve_k: int = 50,
        rank_m: int = 50,
        topk: int = 20,
        use_controller: bool = False,
    ) -> List[str]:
        if use_controller:
            return self._run_with_controller(history_sids, retrieve_k, rank_m, topk)
        return self._run_fallback_policy(history_sids, retrieve_k, rank_m, topk)


class ExactSidTrie:
    """用于SID约束生成的exact trie（pos, prev_token)->allowed_next_tokens。"""

    def __init__(self, allowed_tokens: Dict[int, Dict[int, List[int]]], eos_token_id: int, vocab_size: int):
        self.allowed_tokens = allowed_tokens
        self.eos_token_id = int(eos_token_id)
        self.all_token_ids = list(range(int(vocab_size)))

    @classmethod
    def from_valid_sids(cls, tokenizer, valid_sids: Sequence[str]) -> "ExactSidTrie":
        # 延迟导入，避免在i2i模式下强依赖torch/transformers
        from collections import defaultdict

        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        exact_trie = defaultdict(lambda: defaultdict(set))
        for sid in valid_sids:
            toks = tokenizer.encode(sid, add_special_tokens=False)
            if not toks:
                continue
            for pos in range(len(toks)):
                cur = toks[pos]
                nxt = toks[pos + 1] if pos + 1 < len(toks) else eos_id
                exact_trie[pos][cur].add(nxt)

        allowed: Dict[int, Dict[int, List[int]]] = {}
        for pos in exact_trie:
            allowed[pos] = {tok: sorted(list(nxts)) for tok, nxts in exact_trie[pos].items()}
        return cls(allowed_tokens=allowed, eos_token_id=eos_id, vocab_size=len(tokenizer))

    @classmethod
    def from_pkl(cls, pkl_path: Path, tokenizer) -> "ExactSidTrie":
        import pickle

        with Path(pkl_path).open("rb") as f:
            data = pickle.load(f)
        trie = data.get("exact_trie")
        if not isinstance(trie, dict):
            raise ValueError("trie_pkl缺少 exact_trie")
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        return cls(allowed_tokens=trie, eos_token_id=eos_id, vocab_size=len(tokenizer))

    def prefix_allowed_tokens_fn_factory(self, prompt_len: int, tokenizer=None):
        allowed_tokens = self.allowed_tokens
        all_token_ids = self.all_token_ids

        if tokenizer is None:
            def prefix_allowed_tokens_fn(batch_id, sentence):
                # sentence: 1D LongTensor (prompt + generated)
                sid_pos = int(sentence.numel() - prompt_len)
                if sid_pos <= 0:
                    if 0 in allowed_tokens:
                        return list(allowed_tokens[0].keys())
                    return all_token_ids

                prev_pos = sid_pos - 1
                prev_token = int(sentence[prompt_len + prev_pos])
                if prev_pos in allowed_tokens and prev_token in allowed_tokens[prev_pos]:
                    return allowed_tokens[prev_pos][prev_token]
                return all_token_ids

            return prefix_allowed_tokens_fn

        sep = tokenizer("</think>", add_special_tokens=False)["input_ids"]
        newline_tokens = tokenizer.encode("\n", add_special_tokens=False)

        def find_last_sublist(lst, sub):
            if not sub:
                return None
            n, m = len(lst), len(sub)
            for start in range(n - m, -1, -1):
                if lst[start:start + m] == sub:
                    return start
            return None

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            pos = find_last_sublist(sentence, sep)
            if pos is None:
                return all_token_ids

            pos_after_sep = pos + len(sep)
            generated_after_sep = sentence[pos_after_sep:]
            current_pos = len(generated_after_sep)

            if current_pos == 0:
                return newline_tokens

            sid_pos = current_pos - 1
            if sid_pos == 0:
                if 0 in allowed_tokens:
                    return list(allowed_tokens[0].keys())
                return [self.eos_token_id]

            if len(generated_after_sep) > sid_pos:
                prev_token = generated_after_sep[sid_pos]
                prev_pos = sid_pos - 1
                if prev_pos in allowed_tokens and prev_token in allowed_tokens[prev_pos]:
                    return allowed_tokens[prev_pos][prev_token]

            return [self.eos_token_id]

        return prefix_allowed_tokens_fn


class TwiSTARFastRecTool:
    """SID generation backend used by TwiSTAR tools.

    - Fast ``fast_rec(k)`` uses ``prompt_style='prefill_empty_think'``: the prompt
      already contains an empty ``<think>...</think>`` block, so generation starts
      directly at the SID and can be trie-constrained immediately.
    - Slow ``think_and_rec(j)`` uses ``prompt_style='reasoning'``: generation
      starts from the assistant turn, lets the model produce
      ``<think>...</think>``, then constrains the SID after ``</think>``.
    """

    def __init__(
        self,
        model_path: Path,
        memory_query: MemoryQueryTool,
        trie_pkl_path: Optional[Path] = None,
        build_trie_from_sid2text: bool = True,
        device: str = "auto",
        max_new_tokens: int = 20,
        prompt_with_think: bool = True,
        prompt_style: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.memory_query = memory_query
        self.trie_pkl_path = Path(trie_pkl_path) if trie_pkl_path else None
        self.build_trie_from_sid2text = bool(build_trie_from_sid2text)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_with_think = bool(prompt_with_think)
        if prompt_style is None:
            prompt_style = "prefill_empty_think" if self.prompt_with_think else "plain"
        prompt_style = str(prompt_style).strip().lower()
        valid_prompt_styles = {"plain", "prefill_empty_think", "reasoning"}
        if prompt_style not in valid_prompt_styles:
            raise ValueError(f"prompt_style 必须是 {sorted(valid_prompt_styles)} 之一，当前为: {prompt_style}")
        self.prompt_style = prompt_style

        self._tokenizer = None
        self._model = None
        self._models: List[Any] = []
        self._model_devices: List[str] = []
        self._trie: Optional[ExactSidTrie] = None
        self._input_device = None
        self._all_sids_cache: Optional[List[str]] = None

    @property
    def parallel_size(self) -> int:
        return max(1, len(self._models))

    def _resolve_device_config(self, use_cuda: bool) -> List[str]:
        import torch

        raw_device = str(self.device or "auto").strip()
        normalized = raw_device.lower()

        if not use_cuda:
            return ["cpu"]

        if normalized == "auto":
            return ["cuda:0"]

        if normalized in {"all", "cuda:all", "cuda_all"}:
            device_count = torch.cuda.device_count()
            if device_count <= 0:
                return ["cpu"]
            return [f"cuda:{idx}" for idx in range(device_count)]

        if "," in raw_device:
            devices: List[str] = []
            seen_devices = set()
            for part in raw_device.split(","):
                token = part.strip().lower()
                if not token:
                    continue
                if token == "cuda":
                    normalized_device = "cuda:0"
                elif token.isdigit():
                    normalized_device = f"cuda:{token}"
                elif token.startswith("cuda:") and token[5:].isdigit():
                    normalized_device = f"cuda:{token[5:]}"
                else:
                    raise ValueError(f"无效的多卡设备字符串: {raw_device}")
                if normalized_device in seen_devices:
                    continue
                seen_devices.add(normalized_device)
                devices.append(normalized_device)
            if not devices:
                raise ValueError(f"无效的多卡设备字符串: {raw_device}")
            return devices

        if normalized == "cuda":
            return ["cuda:0"]

        return [raw_device]

    @staticmethod
    def _model_input_device_for_model(model) -> str:
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map:
            for module_device in hf_device_map.values():
                if isinstance(module_device, str) and module_device not in {"cpu", "disk"}:
                    return module_device

        try:
            return str(next(model.parameters()).device)
        except StopIteration:
            return "cpu"

    def _model_input_device(self):
        if self._input_device is not None:
            return self._input_device

        model = self._model
        if model is None:
            raise RuntimeError("模型尚未初始化")

        self._input_device = self._model_input_device_for_model(model)
        return self._input_device

    def _all_sids(self) -> List[str]:
        if self._all_sids_cache is None:
            self._all_sids_cache = list(self.memory_query.all_sids())
        return self._all_sids_cache

    @staticmethod
    def _split_indexed_items(items: Sequence[Tuple[int, Any]], shard_count: int) -> List[List[Tuple[int, Any]]]:
        if shard_count <= 1 or len(items) <= 1:
            return [list(items)] if items else []
        chunk_size = max(1, (len(items) + shard_count - 1) // shard_count)
        return [list(items[start : start + chunk_size]) for start in range(0, len(items), chunk_size)]

    def _parallel_map(self, items: Sequence[Tuple[int, Any]], worker_fn):
        if not items:
            return []

        if len(self._models) <= 1 or len(items) <= 1:
            model = self._models[0] if self._models else self._model
            device = self._model_devices[0] if self._model_devices else self._model_input_device()
            return worker_fn(list(items), model, device)

        shard_count = min(len(self._models), len(items))
        shards = self._split_indexed_items(items, shard_count)
        if len(shards) <= 1:
            return worker_fn(shards[0], self._models[0], self._model_devices[0])

        outputs = []
        with ThreadPoolExecutor(max_workers=len(shards)) as executor:
            futures = [
                executor.submit(worker_fn, shard, self._models[idx], self._model_devices[idx])
                for idx, shard in enumerate(shards)
            ]
            for future in futures:
                outputs.extend(future.result())
        outputs.sort(key=lambda item: item[0])
        return outputs

    @staticmethod
    def _format_user_content(history_sids: Sequence[str]) -> str:
        # 复用数据生成脚本的格式，便于模型对齐
        sids = [s for s in history_sids if s]
        return "The user has purchased the following items: " + "; ".join(sids) + ";"

    @staticmethod
    def _system_message() -> str:
        return (
            "You are a professional recommendation expert who needs to recommend the next possible purchase for users "
            "based on their purchase history. Please predict the most likely next product that the user will purchase "
            "based on the user's historical purchase information."
        )

    @classmethod
    def _format_chat_prompt(cls, user_content: str) -> str:
        """Fast/SFT-compatible prompt with an empty thinking block prefilled."""

        system_message = cls._system_message()
        chat_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
<think>

</think>
"""
        return chat_prompt

    @classmethod
    def _format_chat_prompt_reasoning(cls, user_content: str) -> str:
        """Slow reasoning prompt: model must generate <think>...</think> then SID."""

        system_message = cls._system_message()
        instruction = (
            f"{user_content}\n\n"
            "Strict output requirements:\n"
            "1) Put your reasoning ONLY inside <think>...</think>.\n"
            "2) After </think>, output EXACTLY ONE SID string and nothing else.\n"
        )
        chat_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
        return chat_prompt

    @classmethod
    def _format_chat_prompt_plain(cls, user_content: str) -> str:
        """更轻量的 prompt：不包含 <think> 块。

        主要用于高吞吐评测/线上推理，避免 prefix_allowed_tokens_fn 在每步扫描 "</think>" 带来的开销。
        """

        system_message = cls._system_message()
        chat_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""
        return chat_prompt

    def _format_prompt(self, history_sids: Sequence[str]) -> str:
        user_content = self._format_user_content(history_sids)
        if self.prompt_style == "reasoning":
            return self._format_chat_prompt_reasoning(user_content)
        if self.prompt_style == "plain":
            return self._format_chat_prompt_plain(user_content)
        return self._format_chat_prompt(user_content)

    @staticmethod
    def _legacy_format_chat_prompt(user_content: str) -> str:
        system_message = (
            "You are a professional recommendation expert who needs to recommend the next possible purchase for users "
            "based on their purchase history. Please predict the most likely next product that the user will purchase "
            "based on the user's historical purchase information."
        )
        chat_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
<think>

</think>
"""
        return chat_prompt

    def _lazy_init(self) -> None:
        if self._tokenizer is not None and self._models:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.model_path.exists():
            raise FileNotFoundError(f"TwiSTAR SID generation model path not found: {self.model_path}")

        tok = AutoTokenizer.from_pretrained(str(self.model_path))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        use_cuda = torch.cuda.is_available()
        devices = self._resolve_device_config(use_cuda)
        dtype = torch.float16 if any(str(device).startswith("cuda") for device in devices) else torch.float32
        load_kwargs = {"torch_dtype": dtype}

        models = []
        model_devices = []
        for device in devices:
            model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
            model = model.to(device)
            model.eval()
            models.append(model)
            model_devices.append(self._model_input_device_for_model(model))

        self._tokenizer = tok
        self._models = models
        self._model_devices = model_devices
        self._model = models[0]
        self._input_device = model_devices[0]

        # trie：优先加载pkl，否则从sid2text构建
        if self.trie_pkl_path and self.trie_pkl_path.exists():
            self._trie = ExactSidTrie.from_pkl(self.trie_pkl_path, tok)
        elif self.build_trie_from_sid2text:
            valid_sids = self._all_sids()
            if not valid_sids:
                raise RuntimeError("sid2text为空，无法从sid2text构建trie")
            self._trie = ExactSidTrie.from_valid_sids(tok, valid_sids)
        else:
            self._trie = None

    def _retrieve_batch_on_replica(
        self,
        indexed_histories: Sequence[Tuple[int, Sequence[str]]],
        model,
        input_device: str,
        k: int,
    ) -> List[Tuple[int, Tuple[List[str], Dict[str, float]]]]:
        import torch

        if not indexed_histories:
            return []

        tok = self._tokenizer
        assert tok is not None
        prompts = [self._format_prompt(history_sids) for _, history_sids in indexed_histories]
        enc = tok(prompts, return_tensors="pt", padding=True)
        enc = {kk: vv.to(input_device) for kk, vv in enc.items()}
        prompt_len = int(enc["input_ids"].shape[1])

        prefix_allowed_tokens_fn = None
        if self._trie is not None:
            # plain 模式时，约束位置可以直接用 prompt_len 计算；
            # 传 tokenizer=None 避免每步扫描 "</think>" 的额外开销。
            if self.prompt_style == "plain":
                prefix_allowed_tokens_fn = self._trie.prefix_allowed_tokens_fn_factory(prompt_len, None)
            else:
                prefix_allowed_tokens_fn = self._trie.prefix_allowed_tokens_fn_factory(prompt_len, tok)

        num_beams = max(1, int(k))
        gen_kwargs = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask", None),
            "max_new_tokens": int(self.max_new_tokens),
            "num_beams": num_beams,
            "num_return_sequences": num_beams,
            "output_scores": True,
            "return_dict_in_generate": True,
            "early_stopping": True,
            "use_cache": True,
        }
        if prefix_allowed_tokens_fn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

        device_ctx = torch.cuda.device(input_device) if str(input_device).startswith("cuda") else nullcontext()
        with device_ctx, torch.inference_mode():
            out = model.generate(**gen_kwargs)

        # 只解码生成部分，避免从 prompt 中误提取历史 SID（会导致候选全是 history 里的第一个 SID）。
        gen_seqs = out.sequences[:, prompt_len:]
        decoded = tok.batch_decode(gen_seqs, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        scores = out.get("sequences_scores", None)
        if scores is not None:
            scores_list = [float(s) for s in scores.detach().cpu().tolist()]
        else:
            scores_list = [0.0] * len(decoded)

        results: List[Tuple[int, Tuple[List[str], Dict[str, float]]]] = []
        for batch_offset, (original_index, history_sids) in enumerate(indexed_histories):
            sample_start = batch_offset * num_beams
            sample_end = sample_start + num_beams
            sample_texts = decoded[sample_start:sample_end]
            sample_scores = scores_list[sample_start:sample_end]
            sid_scores: Dict[str, float] = {}
            candidates: List[str] = []
            for text, sc in zip(sample_texts, sample_scores):
                sid = extract_first_sid_from_generation(text)
                if not sid:
                    continue
                if sid in sid_scores:
                    continue
                candidates.append(sid)
                sid_scores[sid] = float(sc)

            results.append((original_index, (candidates[:k], sid_scores)))
        return results

    def retrieve_batch(self, histories_batch: Sequence[Sequence[str]], k: int = 50) -> List[Tuple[List[str], Dict[str, float]]]:
        self._lazy_init()
        indexed_histories = [(idx, [s for s in history_sids if s]) for idx, history_sids in enumerate(histories_batch)]
        outputs = self._parallel_map(indexed_histories, lambda items, model, device: self._retrieve_batch_on_replica(items, model, device, k))
        return [result for _, result in outputs]

    def retrieve(self, history_sids: Sequence[str], k: int = 50) -> Tuple[List[str], Dict[str, float]]:
        batch_results = self.retrieve_batch([history_sids], k=k)
        if not batch_results:
            return [], {}
        return batch_results[0]

    def _score_batch_on_replica(
        self,
        indexed_pairs: Sequence[Tuple[int, Tuple[Sequence[str], str]]],
        model,
        input_device: str,
        normalize_by_length: bool,
    ) -> List[Tuple[int, float]]:
        import torch

        if not indexed_pairs:
            return []

        tok = self._tokenizer
        assert tok is not None
        prompts = []
        full_texts = []
        candidate_texts = []
        for _, (history_sids, candidate_sid) in indexed_pairs:
            history = [s for s in history_sids if s]
            prompt = self._format_chat_prompt(self._format_user_content(history))
            prompts.append(prompt)
            full_texts.append(prompt + candidate_sid)
            candidate_texts.append(candidate_sid)

        full_enc = tok(full_texts, return_tensors="pt", padding=True)
        input_ids = full_enc["input_ids"].to(input_device)
        attention_mask = full_enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_device)
        candidate_enc = tok(candidate_texts, return_tensors="pt", padding=True, add_special_tokens=False)
        candidate_attention_mask = candidate_enc.get("attention_mask", None)
        if candidate_attention_mask is not None:
            candidate_lengths = [int(v) for v in candidate_attention_mask.sum(dim=1).tolist()]
        else:
            candidate_lengths = [len(tok.encode(candidate_sid, add_special_tokens=False)) for candidate_sid in candidate_texts]

        device_ctx = torch.cuda.device(input_device) if str(input_device).startswith("cuda") else nullcontext()
        with device_ctx, torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)

        seq_len = int(input_ids.shape[1])
        scores: List[Tuple[int, float]] = []
        for batch_index, (original_index, (history_sids, candidate_sid)) in enumerate(indexed_pairs):
            history = [s for s in history_sids if s]
            cand_len = int(candidate_lengths[batch_index])
            if not history or not str(candidate_sid or "").strip() or cand_len <= 0:
                scores.append((original_index, float("-inf")))
                continue

            token_ids = input_ids[batch_index, seq_len - cand_len : seq_len]
            token_log_probs = log_probs[batch_index, seq_len - cand_len - 1 : seq_len - 1, :]
            gathered = token_log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            score_sum = float(gathered.sum().item())
            if normalize_by_length:
                scores.append((original_index, score_sum / float(cand_len)))
            else:
                scores.append((original_index, score_sum))
        return scores

    def score_sid_batch(
        self,
        histories_batch: Sequence[Sequence[str]],
        candidate_sids: Sequence[str],
        normalize_by_length: bool = True,
    ) -> List[float]:
        self._lazy_init()
        if len(histories_batch) != len(candidate_sids):
            raise ValueError("histories_batch 与 candidate_sids 长度不一致")

        indexed_pairs = [
            (idx, ([s for s in history_sids if s], str(candidate_sid or "").strip()))
            for idx, (history_sids, candidate_sid) in enumerate(zip(histories_batch, candidate_sids))
        ]
        outputs = self._parallel_map(
            indexed_pairs,
            lambda items, model, device: self._score_batch_on_replica(items, model, device, normalize_by_length),
        )
        return [score for _, score in outputs]

    def score_sid(self, history_sids: Sequence[str], candidate_sid: str, normalize_by_length: bool = True) -> float:
        scores = self.score_sid_batch([history_sids], [candidate_sid], normalize_by_length=normalize_by_length)
        if not scores:
            return float("-inf")
        return float(scores[0])


def load_ranker_weights(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ranker权重文件不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("ranker权重必须是JSON对象（dict）")
    return {str(k): float(v) for k, v in obj.items()}


def load_sid2text_or_beauty(sid2text_path: Path, beauty_items_path: Optional[Path] = None) -> Dict[str, dict]:
    """尽量加载 sid2text（sid->meta）；若不存在则从 Beauty.pretrain.json 回退。

    - sid2text.json 可能是两种格式：
      1) {sid: {title/categories/...}}
      2) {item_id: {sid/title/categories/...}}
    """

    def _from_itemid_meta(obj: Mapping[str, Any]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for item_id, meta in obj.items():
            if not isinstance(meta, dict):
                continue
            sid = meta.get("sid")
            if not sid:
                continue
            out[str(sid)] = {
                "title": meta.get("title", ""),
                "categories": meta.get("categories", ""),
                "item_id": item_id,
            }
        return out

    p = Path(sid2text_path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and obj:
            # 直接是 sid->meta
            first_key = next(iter(obj.keys()))
            if isinstance(first_key, str) and first_key.startswith("<|sid_begin|>"):
                return {str(k): (v or {}) for k, v in obj.items() if k}
            # 可能是 item_id->meta
            converted = _from_itemid_meta(obj)
            if converted:
                return converted

    if beauty_items_path is not None:
        bp = Path(beauty_items_path)
        if bp.exists():
            with bp.open("r", encoding="utf-8") as f:
                beauty = json.load(f)
            if isinstance(beauty, dict) and beauty:
                converted = _from_itemid_meta(beauty)
                if converted:
                    return converted

    return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TwiSTAR Agent workflow: fast_rec / rank_candidates / think_and_rec")

    p.add_argument(
        "--agent_mode",
        type=str,
        default="twistar",
        choices=["classic", "twistar"],
        help="twistar=论文三工具(agent/planner)链路；classic=旧版retrieve->rank 调试链路。",
    )
    p.add_argument(
        "--history_sids",
        type=str,
        default="",
        help="逗号分隔的SID列表，例如: <|sid_begin|>...<|sid_end|>,<|sid_begin|>...<|sid_end|>",
    )
    p.add_argument(
        "--history_text",
        type=str,
        default="",
        help="包含SID的原始文本（会用正则提取SID）。",
    )
    p.add_argument(
        "--i2i_path",
        type=str,
        default=str(_repo_rel_path("data", "i2i_swing_top5.jsonl")),
        help="I2I知识文件(jsonl)，默认使用仓库自带的Swing输出。",
    )
    p.add_argument(
        "--sid2text_path",
        type=str,
        default=str(_repo_rel_path("data", "sid2text.json")),
        help="SID元信息文件(json)，用于类别重合特征。",
    )

    p.add_argument(
        "--beauty_items_path",
        type=str,
        default=str(_repo_rel_path("data", "Beauty.pretrain.json")),
        help="回退用的商品元信息文件（item_id->meta，含sid/title/categories）。当 sid2text 缺失时使用。",
    )
    p.add_argument(
        "--retrieve_tool",
        type=str,
        default="onerec",
        choices=["onerec", "i2i"],
        help="召回工具：onerec=兼容旧参数名，实际表示 TwiSTAR fast_rec(k) 模型；i2i=基于I2I Swing召回。",
    )
    p.add_argument(
        "--onerec_model_path",
        type=str,
        default=str(_repo_rel_path("train", "results", "beauty_sid_rec")),
        help="TwiSTAR fast_rec(k) 模型目录（HF from_pretrained可加载的目录）。",
    )

    # TwiSTAR agent args
    p.add_argument(
        "--fast_model_path",
        type=str,
        default="",
        help="Fast Model (召回) 模型目录。为空则使用 --onerec_model_path。可指向一阶段对齐后的 merged 模型。",
    )
    p.add_argument(
        "--slow_model_path",
        type=str,
        default="",
        help="Slow Model (think_and_rec) 模型目录。为空则使用 --fast_model_path。",
    )
    p.add_argument(
        "--ranking_model_path",
        type=str,
        default="",
        help="Ranking Model 模型目录。ranking_mode=llm 时为 LLM；ranking_mode=din 时可忽略（使用 din_model_path）。",
    )
    p.add_argument(
        "--ranking_mode",
        type=str,
        default="din",
        choices=["din", "llm", "none"],
        help="Ranking Model 类型：din=DIN粗排；llm=LLM粗排；none=跳过粗排。",
    )
    p.add_argument(
        "--ranking_device",
        type=str,
        default="auto",
        help="Ranking Model 推理设备：auto/cpu/cuda/cuda:0 等。",
    )
    p.add_argument(
        "--din_model_path",
        type=str,
        default=str(_repo_rel_path("train", "results", "din_ranking", "din_ranking.pth")),
        help="DIN ranking 权重路径（.pth）。",
    )
    p.add_argument(
        "--din_sid2id_path",
        type=str,
        default=str(_repo_rel_path("data", "din_sid2id.json")),
        help="DIN ranking 的 sid2id 映射（json）。",
    )
    p.add_argument(
        "--din_embedding_dim",
        type=int,
        default=64,
        help="DIN embedding_dim（需与训练一致，默认64）。",
    )
    p.add_argument(
        "--rank_m",
        type=int,
        default=50,
        help="rank_candidates(m,n) 的 m：从 fast_rec 候选中取多少个进入 ranking tool。",
    )
    p.add_argument(
        "--slow_max_new_tokens",
        type=int,
        default=128,
        help="think_and_rec(j) 的最大生成长度；需要覆盖 <think> reasoning + SID。",
    )
    p.add_argument(
        "--planner_policy",
        type=str,
        default="fast_rank",
        choices=["fast_rank", "fast_only", "slow"],
        help="无训练planner/controller时的确定性回退策略：fast_rec->rank_candidates、仅fast_rec、或think_and_rec。",
    )
    p.add_argument(
        "--slow_normalize_by_length",
        action="store_true",
        default=True,
        help="Slow Model 打分是否按候选 token 长度归一化（默认开启）。",
    )
    p.add_argument(
        "--use_controller",
        action="store_true",
        help="启用 controller 模型做 JSON tool-call（失败自动降级为固定链路）。",
    )
    p.add_argument(
        "--controller_model_path",
        type=str,
        default="",
        help="Controller 模型目录；为空则复用 --fast_model_path。建议指向一阶段对齐后的模型。",
    )
    p.add_argument(
        "--controller_prompt_prefix",
        type=str,
        default="",
        help="可选：controller 额外提示词前缀（会拼在工具说明之前），用于强调策略/约束。",
    )
    p.add_argument(
        "--trie_pkl_path",
        type=str,
        default="",
        help="可选：预计算的exact trie pkl路径（如global_trie.pkl）。不填则可从sid2text构建。",
    )
    p.add_argument(
        "--onerec_build_trie_from_sid2text",
        action="store_true",
        default=True,
        help="fast_rec/think_and_rec 生成时从 sid2text 构建 SID 约束 trie（默认开启）。",
    )
    p.add_argument(
        "--onerec_device",
        type=str,
        default="auto",
        help="TwiSTAR SID 生成推理设备：auto/cpu/cuda/cuda:0 等。",
    )
    p.add_argument(
        "--onerec_max_new_tokens",
        type=int,
        default=20,
        help="fast_rec(k) 生成 SID 的 max_new_tokens。",
    )
    p.add_argument("--retrieve_k", type=int, default=50, help="召回数量，默认50")
    p.add_argument("--topk", type=int, default=20, help="最终输出数量，默认20")
    p.add_argument("--stm_len", type=int, default=20, help="短期记忆长度（最近N个行为）")
    p.add_argument("--stm_decay", type=float, default=0.85, help="短期记忆时间衰减系数")
    p.add_argument(
        "--ranker_weights",
        type=str,
        default="",
        help="可选：轻量ranker权重JSON路径（覆盖默认权重）。",
    )
    p.add_argument("--debug", action="store_true", help="输出更多中间信息")
    p.add_argument(
        "--self_check",
        action="store_true",
        help="运行最小自检：保证retrieve=50、输出top20，覆盖基础检索/排序契约。",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 先加载 sid2text（或从 Beauty.pretrain.json 回退），便于 trie 构建 / ranking prompt 等
    sid2text_data = load_sid2text_or_beauty(Path(args.sid2text_path), beauty_items_path=Path(args.beauty_items_path))
    mem_query = MemoryQueryTool(i2i=None, sid2text_data=sid2text_data)
    mem_query.load()

    # self_check：不强制加载大模型/大文件，保证契约正确即可
    if args.self_check:
        pool = mem_query.all_sids()
        if len(pool) < 60:
            raise RuntimeError("sid2text为空或过小，无法自检")

        history = pool[:3]

        class _DummyRetriever:
            def __init__(self, pool_sids: List[str]):
                self.pool_sids = pool_sids

            def retrieve(self, history_sids: Sequence[str], k: int = 50):
                seen = set(history_sids)
                out = [s for s in self.pool_sids if s not in seen][:k]
                return out, {s: 0.0 for s in out}

        dummy = _DummyRetriever(pool)
        w = load_ranker_weights(args.ranker_weights) if args.ranker_weights else None
        ranker = LightweightRanker(memory_query=mem_query, weights=w)
        agent = NextItemAgent(memory_query=mem_query, retriever=dummy, ranker=ranker)
        out = agent.predict_next_items(history, retrieve_k=50, topk=20)
        assert len(out) == 20, f"topk输出长度不为20: {len(out)}"
        assert len(set(out)) == 20, "topk输出存在重复"
        assert not (set(out) & set(history)), "topk输出包含历史物品"
        print("SELF_CHECK_OK")
        return 0

    # 选择召回工具
    is_twistar_mode = args.agent_mode == "twistar"

    if args.agent_mode == "classic" and args.retrieve_tool == "i2i":
        i2i = I2IKnowledge(Path(args.i2i_path))
        i2i.load()
        mem_query = MemoryQueryTool(i2i=i2i, sid2text_data=sid2text_data)
        mem_query.load()
        stm = ShortTermMemory(max_len=int(args.stm_len), decay=float(args.stm_decay))
        retriever = RetrieveTool(memory_query=mem_query, short_term=stm)
    elif args.agent_mode == "classic":
        trie_pkl = Path(args.trie_pkl_path) if args.trie_pkl_path else None
        retriever = TwiSTARFastRecTool(
            model_path=Path(args.onerec_model_path),
            memory_query=mem_query,
            trie_pkl_path=trie_pkl,
            build_trie_from_sid2text=bool(args.onerec_build_trie_from_sid2text),
            device=str(args.onerec_device),
            max_new_tokens=int(args.onerec_max_new_tokens),
            prompt_style="prefill_empty_think",
        )

    # TwiSTAR agent 初始化
    if is_twistar_mode:
        fast_model_path = Path(args.fast_model_path) if args.fast_model_path else Path(args.onerec_model_path)
        slow_model_path = Path(args.slow_model_path) if args.slow_model_path else fast_model_path
        controller_model_path = Path(args.controller_model_path) if args.controller_model_path else fast_model_path
        trie_pkl = Path(args.trie_pkl_path) if args.trie_pkl_path else None

        fast_model = TwiSTARFastRecTool(
            model_path=fast_model_path,
            memory_query=mem_query,
            trie_pkl_path=trie_pkl,
            build_trie_from_sid2text=bool(args.onerec_build_trie_from_sid2text),
            device=str(args.onerec_device),
            max_new_tokens=int(args.onerec_max_new_tokens),
            prompt_style="prefill_empty_think",
        )

        slow_tool = TwiSTARFastRecTool(
            model_path=slow_model_path,
            memory_query=mem_query,
            trie_pkl_path=trie_pkl,
            build_trie_from_sid2text=bool(args.onerec_build_trie_from_sid2text),
            device=str(args.onerec_device),
            max_new_tokens=int(max(args.slow_max_new_tokens, args.onerec_max_new_tokens)),
            prompt_style="reasoning",
        )

        ranking_model = None
        if str(args.ranking_mode).lower() == "llm":
            if not str(args.ranking_model_path).strip():
                raise ValueError("ranking_mode=llm 需要提供 --ranking_model_path")
            ranking_model = LLMRankingModel(
                model_path=Path(args.ranking_model_path),
                sid2text=sid2text_data,
                device=str(args.ranking_device),
            )
        elif str(args.ranking_mode).lower() == "din":
            ranking_model = DINRankingModel(
                model_path=Path(args.din_model_path),
                sid2id_path=Path(args.din_sid2id_path),
                device=str(args.ranking_device),
                embedding_dim=int(args.din_embedding_dim),
            )
        else:
            ranking_model = None

        slow_model = SlowModelScorer(scorer_tool=slow_tool, normalize_by_length=bool(args.slow_normalize_by_length))

        controller_tool = None
        if args.use_controller:
            controller_tool = TwiSTARFastRecTool(
                model_path=controller_model_path,
                memory_query=mem_query,
                trie_pkl_path=None,
                build_trie_from_sid2text=False,
                device=str(args.onerec_device),
                max_new_tokens=64,
                prompt_style="plain",
            )

        twistar_agent = MultiModelRecAgent(
            fast_model=fast_model,
            ranking_model=ranking_model,
            slow_model=slow_model,
            controller_model=controller_tool,
            controller_prompt_prefix=str(args.controller_prompt_prefix),
            planner_policy=str(args.planner_policy),
        )

    w = load_ranker_weights(args.ranker_weights) if args.ranker_weights else None
    ranker = LightweightRanker(memory_query=mem_query, weights=w)
    agent = None
    if args.agent_mode == "classic":
        agent = NextItemAgent(memory_query=mem_query, retriever=retriever, ranker=ranker)

    history: List[str] = []
    if args.history_text:
        history = extract_sids_from_text(args.history_text)
    if args.history_sids:
        # 允许与history_text合并
        parts = [x.strip() for x in args.history_sids.split(",") if x.strip()]
        history.extend(parts)
    history = [s for s in history if s]

    if not history:
        raise ValueError("history为空：请提供 --history_sids 或 --history_text，或使用 --self_check")

    if is_twistar_mode:
        top_items = twistar_agent.recommend(
            history,
            retrieve_k=int(args.retrieve_k),
            rank_m=int(args.rank_m),
            topk=int(args.topk),
            use_controller=bool(args.use_controller),
        )
    else:
        assert agent is not None
        top_items = agent.predict_next_items(history, retrieve_k=int(args.retrieve_k), topk=int(args.topk))

    if args.debug:
        # 同时输出召回集与rank分，便于对齐/调试
        if is_twistar_mode:
            cands, retrieve_scores = twistar_agent.fast_rec(history, k=int(args.retrieve_k))
        else:
            cands, retrieve_scores = retriever.retrieve(history, k=int(args.retrieve_k))
        out = {
            "history_len": len(history),
            "retrieve_k": int(args.retrieve_k),
            "topk": int(args.topk),
        }
        if is_twistar_mode:
            out.update(
                {
                    "agent_mode": "twistar",
                    "planner_policy": str(args.planner_policy),
                    "tool_interface": ["fast_rec(k)", "rank_candidates(m,n)", "think_and_rec(j)"],
                    "top20": top_items,
                    "fast_rec_debug": [
                        {"rank": idx + 1, "sid": sid, "score": float(retrieve_scores.get(sid, 0.0))}
                        for idx, sid in enumerate(cands[: int(args.retrieve_k)])
                    ],
                }
            )
        elif args.agent_mode == "classic" and agent is not None and agent._use_raw_beam_order(retriever):
            raw_topk = list(cands[: int(args.topk)])
            out.update(
                {
                    "top20": raw_topk,
                    "beam_debug": [
                        {"rank": idx + 1, "sid": sid, "retrieve": float(retrieve_scores.get(sid, 0.0))}
                        for idx, sid in enumerate(raw_topk)
                    ],
                }
            )
        else:
            ranked = ranker.rank(history, cands, retrieve_scores, topk=int(args.topk))
            out.update(
                {
                    "top20": [sid for sid, _ in ranked],
                    "rank_debug": [
                        {"sid": sid, "score": float(sc), "retrieve": float(retrieve_scores.get(sid, 0.0))}
                        for sid, sc in ranked
                    ],
                }
            )
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("\n".join(top_items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
