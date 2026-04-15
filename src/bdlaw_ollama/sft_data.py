from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .config import AppConfig
from .utils import ensure_dir, iter_jsonl, normalize_digits, write_jsonl


def _brief(text: str, limit: int = 360) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    split_at = text.rfind(" ", 0, limit)
    split_at = split_at if split_at > 0 else limit
    return text[:split_at].strip() + "..."


def _seed_lookup_example(chunk: dict, bilingual: bool = False) -> dict:
    section_label = chunk.get("section_id") or "the cited provision"
    user_text = (
        f"{chunk['law_title']} এর {section_label} ধারা কী বলে?"
        if not bilingual
        else f"{chunk['law_title']} এর {section_label} section ta briefly explain koro."
    )
    answer = (
        f"সংক্ষেপে: {_brief(chunk['text'])}\n"
        f"উদ্ধৃতি: {chunk['citation_label']}\n"
        "এটি আনুষ্ঠানিক আইনগত পরামর্শ নয়।"
    )
    if bilingual:
        answer += "\nEnglish support: The cited section is summarized above from the retrieved legal text."
    return {"messages": [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}], "type": "lookup"}


def _seed_plain_language_example(chunk: dict) -> dict:
    user_text = f"{chunk['citation_label']} সহজ বাংলায় বুঝিয়ে বলো।"
    answer = (
        f"সহজ ভাষায়: {_brief(chunk['text'])}\n"
        f"উদ্ধৃতি: {chunk['citation_label']}\n"
        "আরও নিশ্চিত ব্যাখ্যার জন্য মূল আইনপাঠ মিলিয়ে নিন।"
    )
    return {"messages": [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}], "type": "plain_bangla"}


def _seed_comparison_example(chunk_a: dict, chunk_b: dict) -> dict:
    user_text = f"{chunk_a['law_title']} এর {chunk_a.get('section_id') or 'প্রথম'} আর {chunk_b.get('section_id') or 'দ্বিতীয়'} ধারার পার্থক্য কী?"
    answer = (
        f"প্রথম অংশ: {_brief(chunk_a['text'], 220)}\n"
        f"দ্বিতীয় অংশ: {_brief(chunk_b['text'], 220)}\n"
        f"উদ্ধৃতি: {chunk_a['citation_label']} | {chunk_b['citation_label']}\n"
        "এটি retrieved text-based comparison; প্রয়োজনে মূল আইনপাঠ মিলিয়ে নিন।"
    )
    return {"messages": [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}], "type": "comparison"}


def _seed_refusal_example(chunk: dict) -> dict:
    user_text = f"এই তথ্য দিয়ে নিশ্চিত করে বলো মামলায় শেষ রায় কী হবে? {chunk['law_title']} এর ভিত্তিতে final verdict দাও।"
    answer = (
        "আমি retrieved আইনপাঠ ছাড়া নিশ্চিত মামলার ফল ঘোষণা করতে পারি না। "
        f"প্রাসঙ্গিক রেফারেন্স হিসেবে {chunk['citation_label']} দেখা যেতে পারে, "
        "কিন্তু নির্দিষ্ট ঘটনা, প্রমাণ, এবং আদালতের মূল্যায়ন ছাড়া নিশ্চিত সিদ্ধান্ত বলা ঠিক হবে না।"
    )
    return {"messages": [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}], "type": "refusal"}


def _split_bucket(example: dict) -> str:
    question = example["messages"][0]["content"]
    bucket = sum(ord(ch) for ch in question) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "valid"
    return "train"


def make_sft_datasets(config: AppConfig) -> dict:
    chunks = list(iter_jsonl(config.path("processed_dir") / "chunks.jsonl"))
    if not chunks:
        raise RuntimeError("No chunks found. Run `bdlaw prepare-data` first.")

    examples: list[dict] = []
    by_law: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_law[chunk["law_id"]].append(chunk)
        examples.append(_seed_lookup_example(chunk))
        examples.append(_seed_plain_language_example(chunk))
        if len(examples) % 5 == 0:
            examples.append(_seed_lookup_example(chunk, bilingual=True))

    for law_chunks in by_law.values():
        if len(law_chunks) >= 2:
            examples.append(_seed_comparison_example(law_chunks[0], law_chunks[1]))
            examples.append(_seed_refusal_example(law_chunks[0]))

    splits = {"train": [], "valid": [], "test": []}
    for example in examples:
        split = _split_bucket(example)
        review_required = split in {"valid", "test"}
        example["review_required"] = review_required
        splits[split].append(example)

    processed_dir = ensure_dir(config.path("processed_dir"))
    for split_name, rows in splits.items():
        write_jsonl(processed_dir / f"sft_{split_name}.jsonl", rows)

    benchmark_rows = []
    for split_name in ("valid", "test"):
        for row in splits[split_name]:
            assistant_text = row["messages"][1]["content"]
            benchmark_rows.append(
                {
                    "question": row["messages"][0]["content"],
                    "expected_phrase": assistant_text.split("\n", 1)[0],
                    "review_required": True,
                    "split": split_name,
                }
            )
    write_jsonl(config.path("benchmark_dir") / "heldout_questions.jsonl", benchmark_rows)

    return {name: len(rows) for name, rows in splits.items()}
