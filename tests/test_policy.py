from pathlib import Path

from bdlaw_ollama.config import AppConfig
from bdlaw_ollama.policy import build_guardrail_payload, route_user_message


def _make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        root=tmp_path,
        project={
            "paths": {
                "processed_dir": "data/processed",
                "benchmark_dir": "data/benchmarks",
                "modelfile_path": "ollama/Modelfile",
            },
            "assistant": {
                "default_response_language": "bangla",
                "intro_message_bn": "আমি bdlaws-3.6। আমাকে Md Shakil Ahmed তৈরি করেছেন।",
                "intro_message_en": "I am bdlaws-3.6. I was developed by Md Shakil Ahmed.",
                "offtopic_refusal_bn": "আমি মূলত বাংলাদেশি আইন সহায়তার জন্য তৈরি।",
                "offtopic_refusal_en": "I was built mainly for Bangladesh law assistance.",
                "clarification_message_bn": "নির্দিষ্ট আইন বা ধারা বলুন।",
                "clarification_message_en": "Please mention a specific law or section.",
            },
        },
        rag={},
        train={},
    )


def test_route_user_message_recognizes_greeting_intro() -> None:
    config = _make_config(Path("/tmp"))
    route = route_user_message("hi, introduce yourself", config)
    assert route.intent == "greeting"
    assert route.language == "english"


def test_route_user_message_blocks_offtopic_and_keeps_bangla() -> None:
    config = _make_config(Path("/tmp"))
    route = route_user_message("2+2 কত?", config)
    assert route.intent == "offtopic"
    assert route.language == "bangla"


def test_route_user_message_marks_generic_law_request_as_broad() -> None:
    config = _make_config(Path("/tmp"))
    route = route_user_message("amake bangladeshi law niye bolo", config)
    assert route.intent == "law_broad"
    assert route.language == "bangla"


def test_route_user_message_keeps_specific_legal_query_in_domain() -> None:
    config = _make_config(Path("/tmp"))
    route = route_user_message("Evidence Act 1872 er 3 dhara ki?", config)
    assert route.intent == "law"
    assert route.language == "bangla"


def test_build_guardrail_payload_for_intro_mentions_developer() -> None:
    config = _make_config(Path("/tmp"))
    payload = build_guardrail_payload("নিজের পরিচয় দাও", config)
    assert payload is not None
    assert payload["intent"] == "greeting"
    assert "Md Shakil Ahmed" in payload["answer"]
