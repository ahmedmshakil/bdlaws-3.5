from pathlib import Path

from bdlaw_ollama.config import AppConfig
from bdlaw_ollama.export_ollama import write_modelfile


def test_write_modelfile_includes_template_stops_and_seed_messages(tmp_path: Path) -> None:
    config = AppConfig(
        root=tmp_path,
        project={
            "paths": {
                "modelfile_path": "ollama/Modelfile",
            },
            "assistant": {
                "system_prompt": "You are bdlaws-3.6.",
            },
            "export": {
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "runtime_context": 4096,
                "num_predict": 768,
                "chat_template": "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>",
                "stop_tokens": ["<|im_start|>", "<|im_end|>"],
                "seed_messages": [
                    {"role": "assistant", "content": "I am bdlaws-3.6."},
                ],
            },
        },
        rag={},
        train={},
    )

    modelfile_path = write_modelfile(config, tmp_path / "model-q4_k_m.gguf")
    content = modelfile_path.read_text(encoding="utf-8")

    assert "TEMPLATE" in content
    assert 'PARAMETER stop "<|im_start|>"' in content
    assert 'PARAMETER stop "<|im_end|>"' in content
    assert "MESSAGE assistant I am bdlaws-3.6." in content
