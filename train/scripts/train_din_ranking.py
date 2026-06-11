import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import re
import argparse
import os


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)

class DINAttention(nn.Module):
    def __init__(self, embedding_size):
        super(DINAttention, self).__init__()
        self.fc1 = nn.Linear(embedding_size * 4, 36)
        self.fc2 = nn.Linear(36, 1)

    def forward(self, query, keys, keys_length):
        """
        query: [batch_size, 1, embedding_size] (Candidate Item)
        keys: [batch_size, max_seq_len, embedding_size] (User History Sequence)
        keys_length: [batch_size] (Actual lengths of history sequences)
        """
        batch_size, max_seq_len, embedding_size = keys.size()
        
        # [batch_size, max_seq_len, embedding_size]
        queries = query.expand(-1, max_seq_len, -1)
        
        # [batch_size, max_seq_len, embedding_size * 4]
        concat_input = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
        
        # [batch_size, max_seq_len, 36]
        attention_score = F.relu(self.fc1(concat_input))
        
        # [batch_size, max_seq_len, 1]
        attention_score = self.fc2(attention_score)
        
        # Mask padded items
        # [batch_size, max_seq_len]
        mask = torch.arange(max_seq_len, device=keys.device).expand(batch_size, max_seq_len) >= keys_length.unsqueeze(1)
        attention_score = attention_score.squeeze(-1) # [batch_size, max_seq_len]
        
        # Padding values are very small so softmax ignores them
        attention_score = attention_score.masked_fill(mask, -1e9)
        
        # Softmax over sequence length
        attention_weight = F.softmax(attention_score, dim=-1) # [batch_size, max_seq_len]
        
        # [batch_size, 1, max_seq_len] * [batch_size, max_seq_len, embedding_size] -> [batch_size, 1, embedding_size]
        output = torch.bmm(attention_weight.unsqueeze(1), keys)
        return output.squeeze(1) # [batch_size, embedding_size]


class DIN(nn.Module):
    def __init__(self, num_items, embedding_dim=64):
        super(DIN, self).__init__()
        # Item Embeddings
        self.item_emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        
        self.attention = DINAttention(embedding_dim)
        
        # MLP for final prediction
        self.fc1 = nn.Linear(embedding_dim * 2 + 1, 64)
        self.fc2 = nn.Linear(64, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, history, history_length, target_item, candidate_score=None):
        """
        history: [batch_size, max_seq_len] (Tokenized History item IDs)
        history_length: [batch_size]
        target_item: [batch_size] (Tokenized Target item ID)
        candidate_score: [batch_size] (fast_rec score feature)
        """
        # Embeddings
        # [batch_size, max_seq_len, embedding_dim]
        hist_emb = self.item_emb(history)
        
        # [batch_size, 1, embedding_dim]
        target_emb = self.item_emb(target_item).unsqueeze(1)
        
        # User representation via target-aware attention
        # [batch_size, embedding_dim]
        user_rep = self.attention(target_emb, hist_emb, history_length)
        
        # Final output
        target_emb = target_emb.squeeze(1) # [batch_size, embedding_dim]
        if candidate_score is None:
            candidate_score = torch.zeros(target_emb.size(0), device=target_emb.device, dtype=target_emb.dtype)
        candidate_score = candidate_score.view(-1, 1).to(target_emb.dtype)
        
        # Concat User Rep, Target Item Rep and recall score feature
        concat_features = torch.cat([user_rep, target_emb, candidate_score], dim=-1)
        
        x = F.relu(self.fc1(concat_features))
        x = F.relu(self.fc2(x))
        out = self.out(x) # [batch_size, 1]
        
        return out.squeeze(-1) # [batch_size]

class DINDataset(Dataset):
    def __init__(self, parquet_path, sid2id, max_seq_len=20, num_negatives=4, max_rows=0):
        self.df = pd.read_parquet(parquet_path)
        if max_rows and int(max_rows) > 0:
            self.df = self.df.head(int(max_rows)).reset_index(drop=True)
        self.sid2id = sid2id
        self.max_seq_len = max_seq_len
        self.num_negatives = num_negatives
        self.all_sids = list(sid2id.keys())
        self.direct_ranking_mode = {'candidate_sid', 'label'}.issubset(self.df.columns)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        history_str = row['description']
        
        history_sids = re.findall(r'(<\|sid_begin\|>.*?<\|sid_end\|>)', history_str)
        history_ids = [self.sid2id.get(sid, 0) for sid in history_sids]
        
        # Truncate or Pad History
        if len(history_ids) > self.max_seq_len:
            history_ids = history_ids[-self.max_seq_len:]
            hist_len = self.max_seq_len
        else:
            hist_len = len(history_ids)
            history_ids = history_ids + [0] * (self.max_seq_len - len(history_ids))
            
        if self.direct_ranking_mode:
            target_sid = row['candidate_sid']
            target_id = self.sid2id.get(target_sid, 0)
            label = float(row['label'])
            candidate_score = float(row.get('candidate_score', row.get('onerec_score', 0.0)) or 0.0)
            return {
                'history': torch.tensor(history_ids, dtype=torch.long),
                'hist_len': torch.tensor(hist_len, dtype=torch.long),
                'targets': torch.tensor([target_id], dtype=torch.long),
                'labels': torch.tensor([label], dtype=torch.float32),
                'candidate_scores': torch.tensor([candidate_score], dtype=torch.float32)
            }

        groundtruth_sid = row['groundtruth']
        pos_id = self.sid2id.get(groundtruth_sid, 0)
        
        # Negative Sampling
        neg_ids = []
        while len(neg_ids) < self.num_negatives:
            neg_sid = random.choice(self.all_sids)
            if neg_sid not in history_sids and neg_sid != groundtruth_sid:
                neg_ids.append(self.sid2id.get(neg_sid, 0))
                
        targets = [pos_id] + neg_ids
        labels = [1.0] + [0.0] * self.num_negatives
        
        return {
            'history': torch.tensor(history_ids, dtype=torch.long),
            'hist_len': torch.tensor(hist_len, dtype=torch.long),
            'targets': torch.tensor(targets, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'candidate_scores': torch.tensor([0.0] * len(targets), dtype=torch.float32)
        }


def load_sid_mapping(meta_path: str, beauty_items_path: str) -> dict:
    primary = Path(meta_path)
    if primary.exists():
        with primary.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if data and next(iter(data.keys())).startswith('<|sid_begin|>'):
            return data
        sid2meta = {}
        for item_id, meta in data.items():
            sid = (meta or {}).get('sid')
            if sid:
                sid2meta[sid] = {
                    'title': (meta or {}).get('title', ''),
                    'categories': (meta or {}).get('categories', ''),
                    'item_id': item_id,
                }
        if sid2meta:
            return sid2meta

    beauty_path = Path(beauty_items_path)
    if beauty_path.exists():
        with beauty_path.open('r', encoding='utf-8') as f:
            beauty_items = json.load(f)
        sid2meta = {}
        for item_id, meta in beauty_items.items():
            sid = (meta or {}).get('sid')
            if sid:
                sid2meta[sid] = {
                    'title': (meta or {}).get('title', ''),
                    'categories': (meta or {}).get('categories', ''),
                    'item_id': item_id,
                }
        if sid2meta:
            return sid2meta

    raise FileNotFoundError(f"未找到可用的 sid 元信息文件: {meta_path} / {beauty_items_path}")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default=str(repo_path("data", "ranking_recall50_data_train.parquet")))
    parser.add_argument("--val_data", type=str, default=str(repo_path("data", "ranking_recall50_data_val.parquet")))
    parser.add_argument("--sid2text", type=str, default=str(repo_path("data", "sid2text.json")))
    parser.add_argument("--beauty_items", type=str, default=str(repo_path("data", "Beauty.pretrain.json")))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    parser.add_argument("--save_path", type=str, default=str(repo_path("train", "results", "din_ranking", "din_ranking.pth")))
    parser.add_argument("--save_sid2id_path", type=str, default=str(repo_path("data", "din_sid2id.json")))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data_path = Path(args.train_data).expanduser().resolve()
    if not train_data_path.exists():
        raise FileNotFoundError(f"训练数据不存在: {train_data_path}")

    val_data_path = Path(args.val_data).expanduser().resolve() if args.val_data else None
    if val_data_path is not None and not val_data_path.exists():
        print(f"Validation data not found, skip eval: {val_data_path}")
        val_data_path = None

    save_path = Path(args.save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_sid2id_path = Path(args.save_sid2id_path).expanduser().resolve()
    save_sid2id_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading sid mappings...")
    sid2text = load_sid_mapping(args.sid2text, args.beauty_items)
        
    sids = list(sid2text.keys())
    # Reserve 0 for padding
    sid2id = {sid: i+1 for i, sid in enumerate(sids)}
    num_items = len(sid2id) + 1
    
    # Save sid2id mapping for inference
    with save_sid2id_path.open('w', encoding='utf-8') as f:
        json.dump(sid2id, f)
        
    print(f"Total Unique Items: {num_items}")

    print("Loading datasets...")
    train_dataset = DINDataset(
        str(train_data_path),
        sid2id,
        max_seq_len=args.max_seq_len,
        max_rows=args.max_train_rows,
    )
    val_dataset = None
    if val_data_path is not None:
        val_dataset = DINDataset(
            str(val_data_path),
            sid2id,
            max_seq_len=args.max_seq_len,
            max_rows=args.max_val_rows,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = DIN(num_items=num_items, embedding_dim=args.embedding_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val samples: {len(val_dataset)}")
        print(f"Val direct ranking mode: {val_dataset.direct_ranking_mode}")
    print(f"Train direct ranking mode: {train_dataset.direct_ranking_mode}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # Train Loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            history = batch['history'].to(device)
            hist_len = batch['hist_len'].to(device)
            targets = batch['targets'].to(device) # [batch_size, 1+num_negatives]
            labels = batch['labels'].to(device) # [batch_size, 1+num_negatives]
            candidate_scores = batch['candidate_scores'].to(device) # [batch_size, 1+num_negatives]
            
            batch_size, num_targets = targets.shape
            
            # Expand history to match (batch_size * num_targets)
            history_expanded = history.repeat_interleave(num_targets, dim=0)
            hist_len_expanded = hist_len.repeat_interleave(num_targets, dim=0)
            targets_flat = targets.view(-1)
            labels_flat = labels.view(-1)
            candidate_scores_flat = candidate_scores.view(-1)
            
            optimizer.zero_grad()
            outputs = model(history_expanded, hist_len_expanded, targets_flat, candidate_scores_flat)
            
            loss = criterion(outputs, labels_flat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")
        
        if val_loader is None:
            continue

        # Eval Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Eval]"):
                history = batch['history'].to(device)
                hist_len = batch['hist_len'].to(device)
                targets = batch['targets'].to(device)
                labels = batch['labels'].to(device)
                candidate_scores = batch['candidate_scores'].to(device)
                
                batch_size, num_targets = targets.shape
                
                history_expanded = history.repeat_interleave(num_targets, dim=0)
                hist_len_expanded = hist_len.repeat_interleave(num_targets, dim=0)
                targets_flat = targets.view(-1)
                labels_flat = labels.view(-1)
                candidate_scores_flat = candidate_scores.view(-1)
                
                outputs = model(history_expanded, hist_len_expanded, targets_flat, candidate_scores_flat)
                loss = criterion(outputs, labels_flat)
                val_loss += loss.item()

                if val_dataset.direct_ranking_mode:
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    correct += (preds == labels_flat).sum().item()
                    total += labels_flat.numel()
                else:
                    outputs_reshaped = outputs.view(batch_size, num_targets)
                    preds = torch.argmax(outputs_reshaped, dim=1)
                    correct += (preds == 0).sum().item()
                    total += batch_size

        metric_name = "Accuracy" if val_dataset.direct_ranking_mode else "HR@1"
        metric_value = (correct / total) if total > 0 else 0.0
        print(f"Epoch {epoch+1} Val Loss: {val_loss/len(val_loader):.4f} | {metric_name}: {metric_value:.4f}")

    print(f"Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train()
