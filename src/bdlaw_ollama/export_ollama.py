from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import AppConfig
from .utils import compact_whitespace, ensure_dir


def merge_lora_adapter(config: AppConfig) -> Path:
    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency path
        raise RuntimeError(
            "Export dependencies are missing. Install with `pip install -e .[train]` before running `bdlaw export-ollama`."
        ) from exc

    adapter_dir = config.path("training_dir") / "lora"
    if not adapter_dir.exists():
        raise RuntimeError("LoRA adapter not found. Run `bdlaw train` first.")

    temp_merged_dir = Path(config.project["paths"].get("temp_merged_dir", "/tmp/bdlaw-merged-f16")).resolve()
    merged_dir = ensure_dir(temp_merged_dir)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir),
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(merged_dir))
    return merged_dir


def _find_llama_cpp_dir(explicit: str | None = None) -> Path:
    candidates = [
        Path(explicit) if explicit else None,
        Path.cwd() / "llama.cpp",
        Path.cwd().parent / "llama.cpp",
        Path((Path.home() / "llama.cpp")),
        Path((Path.cwd() / ".." / ".." / "llama.cpp")).resolve(),
    ]
    env_dir = Path(Path.cwd().joinpath(".")).resolve()
    if "LLAMA_CPP_DIR" in __import__("os").environ:
        candidates.insert(0, Path(__import__("os").environ["LLAMA_CPP_DIR"]))
    for candidate in candidates:
        if candidate and (candidate / "convert_hf_to_gguf.py").exists():
            return candidate.resolve()
    raise RuntimeError("Could not find llama.cpp. Set LLAMA_CPP_DIR or pass --llama-cpp-dir.")


def _find_llama_quantize(llama_dir: Path) -> Path | None:
    candidates = [
        llama_dir / "build" / "bin" / "llama-quantize",
        llama_dir / "build" / "tools" / "quantize" / "llama-quantize",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _ensure_llama_quantize(llama_dir: Path) -> Path:
    existing = _find_llama_quantize(llama_dir)
    if existing:
        return existing

    build_dir = llama_dir / "build"
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(llama_dir),
            "-B",
            str(build_dir),
            "-DGGML_CUDA=OFF",
        ],
        capture_output=True,
        text=True,
    )
    if configure.returncode != 0:
        raise RuntimeError(
            f"llama.cpp configure failed:\nSTDOUT:\n{configure.stdout}\nSTDERR:\n{configure.stderr}"
        )

    build = subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "llama-quantize",
            "-j",
            str(max(1, min(os.cpu_count() or 1, 4))),
        ],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise RuntimeError(f"llama.cpp build failed:\nSTDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}")

    quantize = _find_llama_quantize(llama_dir)
    if not quantize:
        raise RuntimeError("llama-quantize binary was not found after building llama.cpp.")
    return quantize


def export_gguf(config: AppConfig, merged_dir: Path, llama_cpp_dir: str | None = None) -> Path:
    llama_dir = _find_llama_cpp_dir(llama_cpp_dir)
    output_dir = ensure_dir(config.path("ollama_dir"))
    quantization = config.project["export"]["gguf_quantization"]
    f16_path = output_dir / "model-f16.gguf"
    output_path = output_dir / f"model-{quantization}.gguf"

    convert_command = [
        "python3",
        str(llama_dir / "convert_hf_to_gguf.py"),
        str(merged_dir),
        "--outfile",
        str(f16_path),
        "--outtype",
        "f16",
    ]
    result = subprocess.run(convert_command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF export failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    if quantization != "f16":
        quantize_bin = _ensure_llama_quantize(llama_dir)
        quantize_result = subprocess.run(
            [str(quantize_bin), str(f16_path), str(output_path), quantization],
            capture_output=True,
            text=True,
        )
        if quantize_result.returncode != 0:
            raise RuntimeError(
                f"GGUF quantization failed:\nSTDOUT:\n{quantize_result.stdout}\nSTDERR:\n{quantize_result.stderr}"
            )
        f16_path.unlink(missing_ok=True)
    else:
        output_path = f16_path
    return output_path


def write_modelfile(config: AppConfig, gguf_path: Path) -> Path:
    modelfile_path = config.path("modelfile_path")
    system_prompt = config.project["assistant"]["system_prompt"].strip()
    export_cfg = config.project["export"]
    template = str(export_cfg["chat_template"]).strip()
    stop_tokens = [str(token) for token in export_cfg.get("stop_tokens", [])]
    seed_messages = list(export_cfg.get("seed_messages", []))
    parts = [
        f"FROM {gguf_path.resolve()}",
        "",
        f'TEMPLATE """{template}"""',
        "",
        f'SYSTEM """\n{system_prompt}\n"""',
        "",
        f"PARAMETER temperature {export_cfg['temperature']}",
        f"PARAMETER top_p {export_cfg['top_p']}",
        f"PARAMETER repeat_penalty {export_cfg['repeat_penalty']}",
        f"PARAMETER num_ctx {export_cfg['runtime_context']}",
        f"PARAMETER num_predict {export_cfg['num_predict']}",
    ]
    for token in stop_tokens:
        parts.append(f'PARAMETER stop "{token}"')
    if seed_messages:
        parts.append("")
        for message in seed_messages:
            role = str(message["role"]).strip()
            content = compact_whitespace(str(message["content"]))
            parts.append(f"MESSAGE {role} {content}")
    content = "\n".join(parts) + "\n"
    ensure_dir(modelfile_path.parent)
    modelfile_path.write_text(content, encoding="utf-8")
    return modelfile_path


def ollama_create(config: AppConfig, modelfile_path: Path) -> None:
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        raise RuntimeError("`ollama` command not found in PATH.")
    model_name = config.project["models"]["ollama_model_name"]
    result = subprocess.run(
        [ollama_bin, "create", model_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama create failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def ollama_push(config: AppConfig, target: str) -> None:
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        raise RuntimeError("`ollama` command not found in PATH.")
    local_name = config.project["models"]["ollama_model_name"]
    cp_result = subprocess.run([ollama_bin, "cp", local_name, target], capture_output=True, text=True)
    if cp_result.returncode != 0:
        raise RuntimeError(f"Ollama copy failed:\nSTDOUT:\n{cp_result.stdout}\nSTDERR:\n{cp_result.stderr}")
    push_result = subprocess.run([ollama_bin, "push", target], capture_output=True, text=True)
    if push_result.returncode != 0:
        raise RuntimeError(f"Ollama push failed:\nSTDOUT:\n{push_result.stdout}\nSTDERR:\n{push_result.stderr}")


def export_for_ollama(config: AppConfig, llama_cpp_dir: str | None = None, skip_create: bool = False, push_target: str | None = None) -> dict:
    merged_dir = merge_lora_adapter(config)
    gguf_path = export_gguf(config, merged_dir, llama_cpp_dir=llama_cpp_dir)
    modelfile_path = write_modelfile(config, gguf_path)
    if not skip_create:
        ollama_create(config, modelfile_path)
    if push_target:
        ollama_push(config, push_target)
    return {"merged_dir": str(merged_dir), "gguf_path": str(gguf_path), "modelfile_path": str(modelfile_path)}
