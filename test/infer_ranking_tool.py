import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

class RankingTool:
    def __init__(self, model_path, sid2text_path, device="cuda"):
        print(f"Loading Ranking Model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).to(device)
        self.model.eval()
        self.device = device
        
        print(f"Loading item features from {sid2text_path}...")
        with open(sid2text_path, 'r') as f:
            self.sid2text = json.load(f)
            
        # Pre-compute token IDs for the candidate letters 'A', 'B', 'C', 'D', 'E', etc.
        # Ensure we capture cases where there's a preceding space or not.
        self.letter_tokens = {}
        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
            token_id = self.tokenizer.encode(letter, add_special_tokens=False)[-1]
            self.letter_tokens[letter] = token_id

    def _build_prompt(self, history_sids, candidate_sids):
        history_text = "User purchase history:\n"
        for i, sid in enumerate(history_sids):
            meta = self.sid2text.get(sid, {})
            title = meta.get("title", "Unknown")
            cat = meta.get("categories", "Unknown")
            history_text += f"{i+1}. {sid} (Title: {title}, Category: {cat})\n"

        candidates_text = "Candidates for next purchase:\n"
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        
        for i, sid in enumerate(candidate_sids):
            if i >= len(letters):
                break
            meta = self.sid2text.get(sid, {})
            title = meta.get("title", "Unknown")
            cat = meta.get("categories", "Unknown")
            candidates_text += f"{letters[i]}. {sid} (Title: {title}, Category: {cat})\n"
            
        prompt = (
            f"Based on the user's historical preferences, which of the following candidate items is the user most likely to purchase next?\n\n"
            f"{history_text}\n"
            f"{candidates_text}\n"
            f"Please output the letter of the most likely candidate (e.g., A, B, C, D, E)."
        )
        return prompt, letters[:len(candidate_sids)]

    def get_candidate_scores(self, history_sids, candidate_sids):
        """
        Input:
            history_sids: List of sid strings
            candidate_sids: List of sid strings
        Output:
            Dict mapping candidate sid to its ranking score
        """
        prompt, valid_letters = self._build_prompt(history_sids, candidate_sids)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # For evaluation, we encode up to the assistant generation point.
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get logits of the last token (which is the first token the assistant will generate)
        next_token_logits = outputs.logits[0, -1, :]
        
        # Extract the probability / logit for each valid letter token
        scores = {}
        for i, sid in enumerate(candidate_sids):
            if i >= len(valid_letters):
                break
            letter = valid_letters[i]
            token_id = self.letter_tokens[letter]
            # Use the logit as the score
            score = next_token_logits[token_id].item()
            scores[sid] = score
            
        return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../basemodel/Qwen3-1-7B-expand")
    parser.add_argument("--sid2text_path", type=str, default="../data/sid2text.json")
    args = parser.parse_args()

    # Create the ranking tool
    ranking_tool = RankingTool(args.model_path, args.sid2text_path)
    
    # Mock some data for demonstration
    test_history = ["<|sid_begin|><s_a_99><s_b_19><s_c_220><s_d_204><|sid_end|>", "<|sid_begin|><s_a_238><s_b_74><s_c_13><s_d_122><|sid_end|>"]
    test_candidates = ["<|sid_begin|><s_a_226><s_b_110><s_c_129><s_d_207><|sid_end|>", "<|sid_begin|><s_a_191><s_b_30><s_c_232><s_d_103><|sid_end|>", "<|sid_begin|><s_a_151><s_b_29><s_c_176><s_d_188><|sid_end|>"]
    
    print("Testing Ranking Tool with sample input...")
    scores = ranking_tool.get_candidate_scores(test_history, test_candidates)
    
    print("\nRanking Scores:")
    for sid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"Item: {sid} | Score: {score:.4f}")
