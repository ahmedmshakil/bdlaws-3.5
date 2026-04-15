from __future__ import annotations

from typing import Iterable


def build_context_block(chunks: Iterable[dict]) -> str:
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        citation = chunk.get("citation_label", "Unknown citation")
        text = chunk.get("text", "").strip()
        blocks.append(f"[{idx}] {citation}\n{text}")
    return "\n\n".join(blocks)


def build_query_prompt(question: str, chunks: Iterable[dict], low_confidence_message: str) -> str:
    context_block = build_context_block(chunks)
    return (
        "You must answer using only the retrieved legal context.\n"
        "If the context is insufficient, say you are not certain and use this message:\n"
        f"{low_confidence_message}\n\n"
        "Retrieved context:\n"
        f"{context_block}\n\n"
        "User question:\n"
        f"{question}"
    )


def build_low_confidence_response(message: str) -> dict:
    return {
        "answer": message,
        "citations": [],
        "confidence": "low",
        "grounded": False,
    }
