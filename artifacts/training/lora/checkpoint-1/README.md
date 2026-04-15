---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
license: mit
---

# Model Card for bdlaws-3.5 Checkpoint-1

`bdlaws-3.5` checkpoint-1 is an intermediate PEFT LoRA adapter built for Bangladesh law assistance. It is tuned for bilingual Bangla/English responses, citation-aware answer style, cautious uncertainty handling, and downstream Ollama publishing inside the Bangladesh Law hybrid RAG + QLoRA project.

## Model Details

### Model Description

This checkpoint is not a standalone base model. It is a local LoRA adapter checkpoint trained on top of `Qwen/Qwen2.5-3B-Instruct` for Bangladesh law workflows. The intended behavior is to support legal question answering with citation-friendly formatting and retrieval-grounded downstream deployment rather than raw memorization-only legal advice.

- **Developed by:** Md Shakil Ahmed
- **Model type:** PEFT LoRA adapter / causal language model checkpoint
- **Language(s) (NLP):** Bangla, English
- **License:** MIT for this repository and model-card content; use of the base model remains subject to the original `Qwen/Qwen2.5-3B-Instruct` terms
- **Finetuned from model [optional]:** `Qwen/Qwen2.5-3B-Instruct`

### Model Sources [optional]

- **Repository:** https://github.com/ahmedmshakil/bdlaws-3.5.git
- **Paper [optional]:** No separate paper has been published for this checkpoint
- **Demo [optional]:** No public hosted demo is currently provided; the project is intended for local CLI, API, and Ollama workflows

## Uses

### Direct Use

This checkpoint can be loaded as a LoRA adapter on top of `Qwen/Qwen2.5-3B-Instruct` for local experimentation with Bangladesh-law-oriented response style. It is best suited for bilingual legal assistance, section lookup prompts, explanation prompts, and citation-aware answer formatting.

### Downstream Use [optional]

The intended downstream use is a hybrid RAG + Ollama deployment. In this project, the adapter is combined with retrieval, chunk citations, CLI and FastAPI query paths, and GGUF export so the assistant can answer Bangladesh law questions more reliably than a standalone generation-only workflow.

### Out-of-Scope Use

This checkpoint is out of scope for formal legal advice, unsupported jurisdictions outside Bangladesh law, autonomous legal decision-making, or exact statutory quotation without retrieval support. It should not be treated as a substitute for the official law text, government gazette, or a qualified lawyer.

## Bias, Risks, and Limitations

The checkpoint inherits limitations from the base model, the OCR-extracted legal corpus, and the small local fine-tuning run. Risks include OCR noise, incomplete or outdated legal text coverage, hallucinated citations if used without retrieval, uneven Bangla/English quality across prompts, and overconfident responses in high-stakes legal contexts.

### Recommendations

Use this checkpoint with the project retrieval pipeline, preserve citations in the final answer, and verify any important legal claim against the original statute or an authoritative legal source. Human review is strongly recommended for legal research, compliance, litigation, contracts, or rights-sensitive decisions.

## How to Get Started with the Model

Load the checkpoint as a PEFT adapter on top of the base model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "artifacts/training/lora/checkpoint-1"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(model, adapter_path)
```

For practical Bangladesh law QA, pair the adapter with the repository retrieval pipeline and Ollama export flow instead of using it as a retrieval-free legal model.

## Training Details

### Training Data

Training data was derived from Bangladesh law corpora stored locally in `extracted_laws/bdlaws_json_extra/volume_*.json`, with fallback text from `extracted_laws/bdcode_txt/` when needed. The project canonicalizes duplicate OCR records, builds citation-preserving chunks, and generates chat-style SFT examples for bilingual legal assistance. A separate public dataset card is not provided for this checkpoint.

### Training Procedure

#### Preprocessing [optional]

The project normalizes OCR-heavy law records, removes duplicate title/year/act-number variants, normalizes Bangla and English digits, constructs citation-aware chunks, and generates instruction-style training examples that emphasize bilingual answers, legal citations, and low-confidence refusal behavior.

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

This `checkpoint-1` artifact comes from a local smoke-style training run rather than a long production training schedule. The checkpoint metadata shows `global_step: 1` and `epoch: 1.0`, and the saved adapter file is approximately 58 MB. Exact wall-clock training time was not separately documented in the checkpoint README.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

No separate public benchmark report is attached to this intermediate checkpoint. Evaluation for this artifact should be treated as limited local smoke validation rather than a formal legal benchmark.

#### Factors

Important evaluation factors include Bangla vs English prompting, citation preservation, low-confidence behavior, OCR-heavy legal text quality, and whether the checkpoint is used with retrieval.

#### Metrics

Only limited checkpoint metadata is available for this artifact. The recorded trainer state confirms that the saved checkpoint reached `epoch: 1.0` at `global_step: 1`, but no formal held-out legal QA metrics are published for this checkpoint.

### Results

This checkpoint should be considered an intermediate local adapter checkpoint with limited smoke-run evidence, not a fully benchmarked final release.

#### Summary

The checkpoint is suitable for continued experimentation, export, and downstream integration, but it is not sufficient by itself to claim production-grade legal reliability.

## Model Examination [optional]

No separate interpretability, adversarial robustness, or model examination report has been published for this checkpoint.

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA GeForce GTX 1660 SUPER 6GB
- **Hours used:** Exact training hours for this checkpoint were not publicly reported
- **Cloud Provider:** Not applicable; this checkpoint was produced on a local machine
- **Compute Region:** Local workstation; geographic region not publicly specified
- **Carbon Emitted:** Not publicly reported

## Technical Specifications [optional]

### Model Architecture and Objective

The checkpoint is a PEFT LoRA adapter for `Qwen/Qwen2.5-3B-Instruct`. Its objective is to adapt the base model toward Bangladesh-law-oriented answer style, bilingual response behavior, citation-friendly formatting, and safer low-confidence handling for downstream RAG and Ollama use.

### Compute Infrastructure

This checkpoint was produced in a local QLoRA training workflow designed for a 6GB GPU target and later integrated into an Ollama export pipeline.

#### Hardware

Local workstation with an NVIDIA GeForce GTX 1660 SUPER 6GB GPU.

#### Software

PEFT 0.14.0 plus the repository's QLoRA/Ollama toolchain built around Transformers, TRL, bitsandbytes, and PyTorch.

## Citation [optional]

**BibTeX:**

```bibtex
@misc{ahmed2026bdlaws35,
  author = {Ahmed, Md Shakil},
  title = {bdlaws-3.5},
  year = {2026},
  howpublished = {\url{https://github.com/ahmedmshakil/bdlaws-3.5.git}},
  note = {Bangladesh law hybrid RAG, QLoRA, and Ollama project}
}
```

**APA:**

Ahmed, M. S. (2026). *bdlaws-3.5* [Computer software]. GitHub. https://github.com/ahmedmshakil/bdlaws-3.5.git

## Glossary [optional]

- **LoRA:** Low-Rank Adaptation, a parameter-efficient fine-tuning method.
- **QLoRA:** Quantized LoRA training, typically used to fine-tune large models on smaller GPUs.
- **RAG:** Retrieval-Augmented Generation, where retrieved legal context is supplied before answer generation.

## More Information [optional]

This file describes `checkpoint-1`, which is an intermediate adapter checkpoint inside the training artifacts directory. It should be distinguished from a final fully validated release artifact.

## Model Card Authors [optional]

Md Shakil Ahmed

## Model Card Contact

Repository issues: https://github.com/ahmedmshakil/bdlaws-3.5/issues

### Framework versions

- PEFT 0.14.0
