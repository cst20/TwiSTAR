import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class DINAttention(nn.Module):
    def __init__(self, embedding_size):
        super(DINAttention, self).__init__()
        self.fc1 = nn.Linear(embedding_size * 4, 36)
        self.fc2 = nn.Linear(36, 1)

    def forward(self, query, keys, keys_length):
        batch_size, max_seq_len, embedding_size = keys.size()
        queries = query.expand(-1, max_seq_len, -1)
        concat_input = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
        attention_score = F.relu(self.fc1(concat_input))
        attention_score = self.fc2(attention_score)
        
        mask = torch.arange(max_seq_len, device=keys.device).expand(batch_size, max_seq_len) >= keys_length.unsqueeze(1)
        attention_score = attention_score.squeeze(-1)
        attention_score = attention_score.masked_fill(mask, -1e9)
        attention_weight = F.softmax(attention_score, dim=-1)
        
        output = torch.bmm(attention_weight.unsqueeze(1), keys)
        return output.squeeze(1)

class DIN(nn.Module):
    def __init__(self, num_items, embedding_dim=64):
        super(DIN, self).__init__()
        self.item_emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.attention = DINAttention(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, history, history_length, target_item):
        hist_emb = self.item_emb(history)
        target_emb = self.item_emb(target_item).unsqueeze(1)
        
        user_rep = self.attention(target_emb, hist_emb, history_length)
        target_emb = target_emb.squeeze(1)
        
        concat_features = torch.cat([user_rep, target_emb], dim=-1)
        
        x = F.relu(self.fc1(concat_features))
        x = F.relu(self.fc2(x))
        out = self.out(x)
        return out.squeeze(-1)

class DINRankingTool:
    def __init__(self, model_path, sid2id_path, num_items, embedding_dim=64, device="cuda"):
        print(f"Loading DIN Model from {model_path}...")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        with open(sid2id_path, 'r') as f:
            self.sid2id = json.load(f)
            
        self.model = DIN(num_items=num_items, embedding_dim=embedding_dim)
        # Load weights, allow unstrict to ignore uninitialized parameter size
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print("Warning: Model checkpoint not found. Make sure you train it first.")
            print("Using random weights for testing purposes.")
        self.model.to(self.device)
        self.model.eval()
        
    def get_candidate_scores(self, history_sids, candidate_sids, max_seq_len=20):
        # Convert History
        history_ids = [self.sid2id.get(sid, 0) for sid in history_sids]
        if len(history_ids) > max_seq_len:
            history_ids = history_ids[-max_seq_len:]
            hist_len = max_seq_len
        else:
            hist_len = len(history_ids)
            history_ids = history_ids + [0] * (max_seq_len - len(history_ids))
            
        # Convert Candidates
        candidate_ids = [self.sid2id.get(sid, 0) for sid in candidate_sids]
        
        # Build Tensors
        hist_tensor = torch.tensor(history_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        hist_len_tensor = torch.tensor([hist_len], dtype=torch.long).to(self.device)
        
        # Broadcast history for batch inference of candidates
        batch_size = len(candidate_ids)
        hist_tensor_expanded = hist_tensor.repeat_interleave(batch_size, dim=0)
        hist_len_expanded = hist_len_tensor.repeat_interleave(batch_size, dim=0)
        targets_tensor = torch.tensor(candidate_ids, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(hist_tensor_expanded, hist_len_expanded, targets_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            
        scores = {sid: prob for sid, prob in zip(candidate_sids, probs)}
        return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../basemodel/din_ranking.pth")
    parser.add_argument("--sid2id_path", type=str, default="../data/din_sid2id.json")
    parser.add_argument("--num_items", type=int, default=12102) # Make sure this matches training
    args = parser.parse_args()

    # Load mapping to know exactly the number of items
    try:
        with open(args.sid2id_path, 'r') as f:
            sid2id = json.load(f)
        num_items = len(sid2id) + 1
    except FileNotFoundError:
        print("sid2id mapping not found, using default num_items")
        num_items = args.num_items

    # Create the ranking tool
    ranking_tool = DINRankingTool(args.model_path, args.sid2id_path, num_items=num_items)
    
    # Mock data
    test_history = ["<|sid_begin|><s_a_99><s_b_19><s_c_220><s_d_204><|sid_end|>", "<|sid_begin|><s_a_238><s_b_74><s_c_13><s_d_122><|sid_end|>"]
    test_candidates = ["<|sid_begin|><s_a_226><s_b_110><s_c_129><s_d_207><|sid_end|>", "<|sid_begin|><s_a_191><s_b_30><s_c_232><s_d_103><|sid_end|>", "<|sid_begin|><s_a_151><s_b_29><s_c_176><s_d_188><|sid_end|>"]
    
    print("Testing DIN Ranking Tool with sample input...")
    scores = ranking_tool.get_candidate_scores(test_history, test_candidates)
    
    print("\nDIN Ranking Scores:")
    for sid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"Item: {sid} | Score: {score:.4f}")
