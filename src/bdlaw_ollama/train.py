from __future__ import annotations

import json
import inspect
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import AppConfig, load_yaml
from .utils import ensure_dir, iter_jsonl, write_json


def preflight_training() -> dict[str, Any]:
    report = {"nvidia_smi": False, "torch_cuda": False, "cuda_device": None}
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        result = subprocess.run([nvidia], capture_output=True, text=True)
        report["nvidia_smi"] = result.returncode == 0
    try:
        import torch

        report["torch_cuda"] = bool(torch.cuda.is_available())
        if report["torch_cuda"]:
            report["cuda_device"] = torch.cuda.get_device_name(0)
            report["bf16_supported"] = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception as exc:  # pragma: no cover - defensive runtime path
        report["torch_error"] = str(exc)
    return report


def ensure_training_ready() -> dict[str, Any]:
    report = preflight_training()
    if not report.get("nvidia_smi") or not report.get("torch_cuda"):
        raise RuntimeError(
            "Training requires a working NVIDIA/CUDA setup. "
            f"Current preflight result: {json.dumps(report, ensure_ascii=False)}"
        )
    return report


def _patch_accelerate_unwrap_model() -> None:
    """Backfill newer unwrap_model kwargs when older accelerate is installed."""
    try:
        from accelerate import Accelerator
    except ModuleNotFoundError:  # pragma: no cover - dependency path
        return

    signature = inspect.signature(Accelerator.unwrap_model)
    if "keep_torch_compile" in signature.parameters:
        return

    original = Accelerator.unwrap_model

    def unwrap_model_compat(self, model, keep_fp32_wrapper: bool = True, keep_torch_compile: bool | None = None):
        return original(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

    Accelerator.unwrap_model = unwrap_model_compat


def _load_split(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def _render_chat_rows(rows: list[dict], tokenizer) -> list[dict]:
    rendered = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        rendered.append({"text": text, "messages": row["messages"]})
    return rendered


def _derive_response_template(tokenizer) -> str | None:
    probe_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": ""},
    ]
    rendered = tokenizer.apply_chat_template(
        probe_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    markers = [
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|im_start|>assistant\n",
        "assistant\n",
    ]
    for marker in markers:
        if marker in rendered:
            return marker
    return None


def run_training(
    config: AppConfig,
    train_config_path: Path | None = None,
    smoke: bool = False,
    max_train_samples: int | None = None,
) -> dict[str, Any]:
    ensure_training_ready()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency path
        raise RuntimeError(
            "Training dependencies are missing. Install with `pip install -e .[train]` before running `bdlaw train`."
        ) from exc

    train_cfg = config.train.copy()
    if train_config_path:
        train_cfg.update(load_yaml(train_config_path))

    _patch_accelerate_unwrap_model()

    dtype_name = train_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    use_bf16 = dtype_name == "bfloat16" and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model_name = train_cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(train_cfg["load_in_4bit"]),
        bnb_4bit_quant_type=str(train_cfg["bnb_4bit_quant_type"]),
        bnb_4bit_use_double_quant=bool(train_cfg["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=int(train_cfg["lora_r"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        target_modules=list(train_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    processed_dir = config.path("processed_dir")
    train_rows = _load_split(processed_dir / "sft_train.jsonl")
    valid_rows = _load_split(processed_dir / "sft_valid.jsonl")
    if smoke:
        train_rows = train_rows[: min(32, len(train_rows))]
        valid_rows = valid_rows[: min(8, len(valid_rows))]
    elif max_train_samples:
        train_rows = train_rows[:max_train_samples]

    train_rendered = _render_chat_rows(train_rows, tokenizer)
    valid_rendered = _render_chat_rows(valid_rows, tokenizer)

    response_template = _derive_response_template(tokenizer) if bool(train_cfg.get("assistant_only_loss", True)) else None
    collator = (
        DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
        if response_template
        else None
    )

    output_dir = ensure_dir(config.path("training_dir") / "lora")
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        max_seq_length=int(train_cfg["max_seq_length"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        optim=str(train_cfg["optim"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        eval_steps=int(train_cfg["eval_steps"]),
        gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]),
        report_to=[],
        do_eval=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=Dataset.from_list(train_rendered),
        eval_dataset=Dataset.from_list(valid_rendered),
        data_collator=collator,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = dict(getattr(train_result, "metrics", {}))
    if valid_rows and not bool(train_cfg.get("skip_final_eval", False)):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            metrics.update(trainer.evaluate())
        except RuntimeError as exc:
            message = str(exc)
            if "CUDA out of memory" not in message:
                raise
            metrics["eval_warning"] = message
    write_json(output_dir / "metrics.json", metrics)
    return {
        "output_dir": str(output_dir),
        "metrics": metrics,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "response_template": response_template,
    }
