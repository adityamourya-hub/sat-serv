#!/usr/bin/env python3
"""
QLoRA-style fine-tuning scaffold for English SSA report generation.
Target models:
- Llama-3-8B (preferred)
- Sovereign 7B (drop-in alternative)

Outputs:
- ai/report/data/lora_adapter/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="ai/report/data/report_sft_dataset.jsonl")
    parser.add_argument("--output", type=str, default="ai/report/data/lora_adapter")
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    ds = load_dataset("json", data_files=args.dataset, split="train")

    def to_text(ex):
        return {
            "text": (
                f"Instruction: {ex['instruction']}\n"
                f"Input: {ex['input']}\n"
                f"Response: {ex['output']}"
            )
        }

    ds = ds.map(to_text)

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        dataset_text_field="text",
        tokenizer=tok,
        args=train_args,
        peft_config=peft_cfg,
        max_seq_length=1024,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    print(f"saved_adapter={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
