from __future__ import annotations

from pathlib import Path

from .api import run_query_pipeline
from .config import AppConfig
from .utils import iter_jsonl


def run_benchmark(config: AppConfig, limit: int | None = None, model_name: str | None = None) -> dict:
    benchmark_path = config.path("benchmark_dir") / "heldout_questions.jsonl"
    if not benchmark_path.exists():
        raise RuntimeError("Benchmark file not found. Run `bdlaw make-sft` first.")

    rows = list(iter_jsonl(benchmark_path))
    if limit:
        rows = rows[:limit]

    results = []
    hits = 0
    for row in rows:
        response = run_query_pipeline(row["question"], root=config.root, model_name=model_name)
        matched = row["expected_phrase"] in response["answer"]
        hits += int(matched)
        results.append({"question": row["question"], "matched": matched, "confidence": response["confidence"]})

    return {"count": len(rows), "matched": hits, "accuracy": hits / len(rows) if rows else 0.0, "results": results}
