#!/usr/bin/env python3
"""
Run quantized local inference for English SSA summary generation.
Uses 4-bit base model + LoRA adapter and writes sample outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main() -> None:
    base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter = "ai/report/data/lora_adapter"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter)

    prompt = (
        "Instruction: Write a concise English SSA alert summary with action recommendation.\n"
        "Input: {'event_id':'EV-201','event_type':'CONJUNCTION_ALERT','severity':'high','asset_id':43286,'asset_name':'IRNSS-1I','threat_id':99999,'threat_name':'DEBRIS_UNKNOWN','miss_distance_km':0.021,'collision_probability':3.2e-05,'tca_minutes':9.1,'priority_score':126}\n"
        "Response:"
    )

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=140, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)

    out_dir = Path("ai/report/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "sample_report_output.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump({"output": text}, f, indent=2)

    print(f"saved={p}")


if __name__ == "__main__":
    main()
