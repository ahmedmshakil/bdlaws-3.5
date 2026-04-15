---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
license: mit
---

# Model Card for bdlaws-3.5 LoRA Adapter

`bdlaws-3.5` is a PEFT LoRA adapter for Bangladesh law assistance. This artifact is intended for bilingual Bangla/English legal QA workflows, citation-aware answer formatting, cautious uncertainty handling, and downstream export to Ollama inside the Bangladesh Law hybrid RAG + QLoRA project.

## Model Details

### Model Description

This artifact is a saved LoRA adapter trained on top of `Qwen/Qwen2.5-3B-Instruct`. It is designed for Bangladesh-law-oriented question answering and works best when combined with the repository retrieval pipeline, citation-preserving chunking, and local deployment flow.

- **Developed by:** Md Shakil Ahmed
- **Funded by [optional]:** Not publicly specified
- **Shared by [optional]:** Md Shakil Ahmed
- **Model type:** PEFT LoRA adapter / causal language model adapter
- **Language(s) (NLP):** Bangla, English
- **License:** MIT for this repository and model-card content; use of the base model remains subject to the original `Qwen/Qwen2.5-3B-Instruct` terms
- **Finetuned from model [optional]:** `Qwen/Qwen2.5-3B-Instruct`

### Model Sources [optional]

- **Repository:** https://github.com/ahmedmshakil/bdlaws-3.5.git
- **Paper [optional]:** No separate paper has been published for this adapter
- **Demo [optional]:** No public hosted demo is currently provided; the intended use is local CLI, API, and Ollama deployment

## Uses

### Direct Use

This adapter can be loaded on top of `Qwen/Qwen2.5-3B-Instruct` to produce Bangladesh-law-oriented responses with bilingual support and citation-friendly style. It is appropriate for local experimentation, prototype legal assistants, and retrieval-grounded legal QA flows.

### Downstream Use [optional]

The main downstream use is hybrid RAG + Ollama deployment. In this repository, the adapter is paired with normalized Bangladesh law corpora, citation-preserving chunks, retrieval scoring, CLI/API query flows, and GGUF export for local serving.

### Out-of-Scope Use

This adapter is out of scope for formal legal advice, unsupported jurisdictions outside Bangladesh law, authoritative statutory quotation without verification, or autonomous legal decisions that affect rights, liability, or compliance.

## Bias, Risks, and Limitations

This adapter inherits limitations from the base model, OCR-extracted legal corpora, and the lightweight local fine-tuning procedure. Risks include OCR noise, incomplete statute coverage, hallucinated law references if retrieval is skipped, inconsistent bilingual quality across prompts, and overconfident answers in high-stakes legal scenarios.

### Recommendations

Use this adapter together with the repository retrieval pipeline, keep citations visible in final answers, and verify important legal claims against the original law text or another authoritative legal source. For practical deployment, treat it as an assistive research tool rather than a final legal authority.

## How to Get Started with the Model

Load the adapter on top of the base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "artifacts/training/lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, adapter_path)
```

For best results, use the adapter with the repository retrieval pipeline and downstream Ollama export flow instead of relying on generation alone.

## Training Details

### Training Data

Training data was generated from Bangladesh law sources stored locally in `extracted_laws/bdlaws_json_extra/volume_*.json`, with text fallback from `extracted_laws/bdcode_txt/` when JSON pages were empty or unusable. The project deduplicates OCR variants, normalizes mixed Bangla/English text, creates citation-preserving chunks, and builds chat-format SFT examples for bilingual legal assistance.

### Training Procedure

#### Preprocessing [optional]

The preprocessing pipeline canonicalizes duplicate law records, cleans OCR noise, normalizes digits, segments law text into retrieval-friendly chunks, and produces supervision examples focused on citations, bilingual behavior, and safe low-confidence responses.

#### Training Hyperparameters

- **Training regime:** 4-bit QLoRA with `float16` compute on a local 6GB GPU profile
- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **LoRA rank:** `8`
- **LoRA alpha:** `16`
- **LoRA dropout:** `0.05`
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Max sequence length:** `1024`
- **Per-device train batch size:** `1`
- **Gradient accumulation steps:** `32`
- **Learning rate:** `0.0002`
- **Epochs configured:** `1`
- **Max steps configured:** `100`
- **Gradient checkpointing:** enabled
- **Assistant-only loss:** enabled

#### Speeds, Sizes, Times [optional]

This saved adapter comes from a local smoke-style training workflow rather than a long, fully benchmarked training run. The adapter file `adapter_model.safetensors` is approximately 58 MB. Exact wall-clock training time and final held-out evaluation timing were not separately published in this model card.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

No separate public benchmark report is bundled with this adapter artifact. Evaluation should be interpreted as limited local smoke validation rather than a complete legal QA benchmark.

#### Factors

Relevant evaluation factors include Bangla vs English queries, citation presence, behavior under weak retrieval context, OCR-heavy legal text quality, and whether the adapter is used inside the intended hybrid RAG pipeline.

#### Metrics

Formal held-out benchmark metrics are not published in this adapter directory. The surrounding project uses retrieval quality, citation behavior, bilingual clarity, and low-confidence handling as practical acceptance dimensions.

### Results

This adapter is suitable for experimentation, downstream integration, and Ollama export, but it should not be described as a fully benchmarked production legal model.

#### Summary

The adapter captures useful Bangladesh-law-oriented response behavior, but dependable legal assistance still requires retrieval grounding and human verification.

## Model Examination [optional]

No separate interpretability or adversarial examination report has been published for this adapter.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA GeForce GTX 1660 SUPER 6GB
- **Hours used:** Exact training hours were not publicly reported
- **Cloud Provider:** Not applicable; this adapter was produced on a local machine
- **Compute Region:** Local workstation; geographic region not publicly specified
- **Carbon Emitted:** Not publicly reported

## Technical Specifications [optional]

### Model Architecture and Objective

The adapter uses PEFT LoRA on top of `Qwen/Qwen2.5-3B-Instruct`. Its objective is to shift the base model toward Bangladesh-law-oriented answering, bilingual response behavior, citation-aware formatting, and safer uncertainty handling for downstream RAG and Ollama use.

### Compute Infrastructure

This adapter was produced in a local QLoRA workflow designed around a 6GB GPU target and later used in a GGUF + Ollama publishing path.

#### Hardware

Local workstation with an NVIDIA GeForce GTX 1660 SUPER 6GB GPU.

#### Software

PEFT 0.14.0 plus the repository QLoRA/Ollama stack built around Transformers, TRL, bitsandbytes, and PyTorch.

## Citation [optional]

**BibTeX:**

```bibtex
@misc{ahmed2026bdlaws35adapter,
  author = {Ahmed, Md Shakil},
  title = {bdlaws-3.5 LoRA Adapter},
  year = {2026},
  howpublished = {\url{https://github.com/ahmedmshakil/bdlaws-3.5.git}},
  note = {Bangladesh law hybrid RAG, QLoRA, and Ollama adapter artifact}
}
```

**APA:**

Ahmed, M. S. (2026). *bdlaws-3.5 LoRA Adapter* [Computer software]. GitHub. https://github.com/ahmedmshakil/bdlaws-3.5.git

## Glossary [optional]

- **LoRA:** Low-Rank Adaptation, a parameter-efficient fine-tuning method.
- **QLoRA:** Quantized LoRA training for adapting large models on smaller GPUs.
- **RAG:** Retrieval-Augmented Generation, where legal context is retrieved before answer generation.

## More Information [optional]

This README documents the saved adapter directory in `artifacts/training/lora/`. It is separate from the root project README because it describes the model artifact itself rather than the whole repository workflow.

## Model Card Authors [optional]

Md Shakil Ahmed

## Model Card Contact

Repository issues: https://github.com/ahmedmshakil/bdlaws-3.5/issues

### Framework versions

- PEFT 0.14.0
