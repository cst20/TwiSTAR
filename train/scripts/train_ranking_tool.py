#!/usr/bin/env python3

import os
from dataclasses import dataclass, field
from typing import Optional, Dict

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    HfArgumentParser
)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="../../basemodel/Qwen3-1-7B-expand",
        metadata={"help": "Path to pretrained model"}
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA"})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA target modules"}
    )

@dataclass
class DataArguments:
    train_data_path: str = "../../data/ranking_data_train.parquet"
    val_data_path: str = "../../data/ranking_data_val.parquet"
    max_source_length: int = field(default=1024)
    max_target_length: int = field(default=10)

def prepare_chat_dataset(data_path, tokenizer, max_source_length, max_target_length, local_rank=0):
    if local_rank == 0:
        print(f"Loading parquet file: {data_path}")
    data_pq = pd.read_parquet(data_path)
    
    # Optional sample size constraint
    # data_pq = data_pq.sample(min(len(data_pq), 10000))
    
    def format_chat(example):
        # We format it as conversational prompt
        prompt = example['prompt']
        answer = example['groundtruth']
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}
    
    dataset = Dataset.from_pandas(data_pq)
    dataset = dataset.map(format_chat, num_proc=4)
    
    def tokenize_function(examples):
        outputs = tokenizer(examples['text'], padding=False, truncation=True, max_length=max_source_length + max_target_length)
        # In actual fine-tuning, you would mask out the prompt labels, but for simplicity here we let DataCollatorForLanguageModeling handle it or we use custom DataCollator.
        # But Qwen chat templates are handled well with DataCollatorForLanguageModeling if we set target as the same as input, shifting labels inside model.
        # However, to mask prompt:
        return outputs
    
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=4)
    return dataset

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    train_dataset = prepare_chat_dataset(
        data_args.train_data_path, tokenizer, data_args.max_source_length, data_args.max_target_length, local_rank
    )
    val_dataset = prepare_chat_dataset(
        data_args.val_data_path, tokenizer, data_args.max_source_length, data_args.max_target_length, local_rank
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if "LOCAL_RANK" not in os.environ else None,
        trust_remote_code=True
    )
    model.config.use_cache = False

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(","),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        if local_rank == 0:
            model.print_trainable_parameters()

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if local_rank == 0:
        print("Starting training the Ranking Tool model...")
        
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
