# Natural-Language SSA Report Generator (Feature 7)

This module provides human-readable signed SSA alert summaries in English.

## Go runtime package

- [engine/pkg/ml/report/report.go](engine/pkg/ml/report/report.go)
- [engine/pkg/ml/report/model_stub.go](engine/pkg/ml/report/model_stub.go)
- [engine/pkg/ml/report/report_test.go](engine/pkg/ml/report/report_test.go)

Capabilities:
- structured SSA report input
- English narrative generation
- deterministic fallback summary generation
- cryptographic signing via existing `pkg/crypto` signer

## Fine-tuning and quantized inference scaffold (4-bit)

- [ai/report/prepare_report_dataset.py](ai/report/prepare_report_dataset.py)
- [ai/report/finetune_lora_4bit.py](ai/report/finetune_lora_4bit.py)
- [ai/report/run_quantized_inference.py](ai/report/run_quantized_inference.py)
- [ai/report/requirements.txt](ai/report/requirements.txt)

Supports:
- Llama-3-8B-Instruct (preferred)
- sovereign/Indian 7B model via `--base_model` override

## Quickstart

python3 -m venv .venv
source .venv/bin/activate
pip install -r ai/report/requirements.txt

# 1) prepare SFT data
python ai/report/prepare_report_dataset.py

# 2) run QLoRA-style 4-bit fine-tune
python ai/report/finetune_lora_4bit.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset ai/report/data/report_sft_dataset.jsonl \
  --output ai/report/data/lora_adapter

# 3) run sample quantized inference
python ai/report/run_quantized_inference.py

Artifact examples:
- `ai/report/data/lora_adapter/`
- `ai/report/data/sample_report_output.json`

## Security and auditability

The generated report can be signed with HMAC proof in Go runtime:
- `GenerateEnglishSignedReport(...)`
- `GenerateEnglishSignedReportJSON(...)`

This preserves your existing cryptographic alert chain while adding cognitive, human-readable summaries.
