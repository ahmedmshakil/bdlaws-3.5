from pathlib import Path

from bdlaw_ollama.config import AppConfig
from bdlaw_ollama.sft_data import make_sft_datasets
from bdlaw_ollama.utils import iter_jsonl, write_jsonl


def test_make_sft_datasets_includes_policy_examples(tmp_path: Path) -> None:
    config = AppConfig(
        root=tmp_path,
        project={
            "paths": {
                "processed_dir": "data/processed",
                "benchmark_dir": "data/benchmarks",
            },
            "assistant": {
                "intro_message_bn": "আমি bdlaws-3.6, বাংলাদেশি আইন সহায়তার জন্য তৈরি একটি AI মডেল। আমাকে Md Shakil Ahmed তৈরি করেছেন।",
                "intro_message_en": "I am bdlaws-3.6, an AI model built for Bangladesh law assistance. I was developed by Md Shakil Ahmed.",
                "offtopic_refusal_bn": "আমি মূলত বাংলাদেশি আইন সহায়তার জন্য তৈরি।",
                "offtopic_refusal_en": "I was built mainly for Bangladesh law assistance.",
                "clarification_message_bn": "নির্দিষ্ট আইন, ধারা, বা সাল উল্লেখ করুন।",
                "clarification_message_en": "Please mention a specific law, section, or year.",
                "low_confidence_message": "আমি নিশ্চিত নই। নির্দিষ্ট আইন বা ধারা উল্লেখ করুন।",
            },
        },
        rag={},
        train={},
    )
    chunks_path = tmp_path / "data" / "processed" / "chunks.jsonl"
    write_jsonl(
        chunks_path,
        [
            {
                "law_id": "law-1",
                "law_title": "Evidence Act, 1872",
                "section_id": "3",
                "citation_label": "Evidence Act, 1872, Section 3, pp. 1-2",
                "text": "Evidence means and includes all statements permitted by the Court.",
            },
            {
                "law_id": "law-1",
                "law_title": "Evidence Act, 1872",
                "section_id": "4",
                "citation_label": "Evidence Act, 1872, Section 4, pp. 2-3",
                "text": "May presume, shall presume, and conclusive proof are defined here.",
            },
        ],
    )

    make_sft_datasets(config)

    rows = []
    for split_name in ("train", "valid", "test"):
        rows.extend(iter_jsonl(tmp_path / "data" / "processed" / f"sft_{split_name}.jsonl"))

    policy_types = {row["type"] for row in rows}
    assert "identity_intro" in policy_types
    assert "offtopic_refusal" in policy_types
    assert "law_question_clarification" in policy_types
