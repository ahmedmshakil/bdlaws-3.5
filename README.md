# bdlaws-3.5

Hybrid RAG + QLoRA pipeline for Bangladesh law data with local Ollama publishing support.

## What This Project Does

- Canonicalizes Bangladesh law JSON files from `extracted_laws/bdlaws_json_extra/volume_*.json`
- Removes duplicate OCR-derived law entries created from title, year, and act-number variants
- Falls back to `extracted_laws/bdcode_txt/` when JSON pages are empty or unusable
- Builds citation-preserving law chunks for retrieval
- Generates bilingual SFT datasets for Bangladesh law assistance
- Fine-tunes a local LoRA adapter on top of `Qwen/Qwen2.5-3B-Instruct`
- Exports GGUF artifacts and an Ollama `Modelfile`
- Serves retrieval-grounded queries through CLI and FastAPI

## Local Data Requirement

This repository does not include the raw extracted law corpus, processed datasets, or model weights in Git. Keep these local on your machine:

- `extracted_laws/`
- `data/processed/`
- `data/benchmarks/`
- large training and export artifacts under `artifacts/`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[train,dev]
```

If you only need data preparation, retrieval, CLI, and API:

```bash
pip install -e .[dev]
```

## Core Commands

```bash
bdlaw prepare-data
bdlaw build-index
bdlaw make-sft
bdlaw train --config configs/train.yaml
bdlaw export-ollama
bdlaw query "বাংলাদেশ ব্যাংক অর্ডার, ১৯৭২ এর ৩ ধারা কী?"
bdlaw serve-api
```

## Configuration

- `configs/project.yaml`: project paths, models, language policy, export settings
- `configs/rag.yaml`: chunking, retrieval, and confidence thresholds
- `configs/train.yaml`: default training settings
- `configs/train.local-6gb.yaml`: safer local override for 6GB GPU workflows

## Repository Layout

- `src/bdlaw_ollama/`: package source code
- `configs/`: project, retrieval, and training configuration
- `tests/`: unit tests for data prep and retrieval behavior
- `ollama/Modelfile`: generated runtime config for Ollama import

## Output Layout

Typical local outputs are:

- `data/processed/laws.jsonl`
- `data/processed/chunks.jsonl`
- `data/processed/sft_{train,valid,test}.jsonl`
- `artifacts/retrieval/faiss.index`
- `artifacts/retrieval/bm25.pkl`
- `artifacts/training/lora/`
- `artifacts/training/merged-f16/`
- `artifacts/ollama/model-q4_k_m.gguf`

These outputs are intentionally kept out of Git.
