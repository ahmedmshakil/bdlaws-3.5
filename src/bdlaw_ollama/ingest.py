from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import AppConfig
from .normalize import (
    canonical_title,
    clean_page_text,
    derive_act_no,
    derive_year,
    find_best_txt_fallback,
    make_law_id,
    normalize_text_for_index,
)
from .utils import ensure_dir, iter_jsonl, read_json, write_jsonl


def discover_volume_files(config: AppConfig) -> list[Path]:
    raw_root = config.path("raw_root")
    raw_glob = config.project["paths"]["raw_glob"]
    ignore_globs = config.project["paths"].get("ignore_globs", [])
    ignored: set[Path] = set()
    for pattern in ignore_globs:
        ignored.update(raw_root.glob(pattern))
    return sorted(path for path in raw_root.glob(raw_glob) if path not in ignored)


def flatten_raw_laws(volume_file: Path) -> list[dict[str, Any]]:
    payload = read_json(volume_file)
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict):
            payload = payload[0]
        else:
            raise ValueError(f"Unexpected top-level list in {volume_file}")
    laws = payload.get("laws", [])
    flattened: list[dict[str, Any]] = []
    for law in laws:
        row = dict(law)
        row["_volume_name"] = payload.get("volume_name")
        row["_volume_type"] = payload.get("volume_type")
        row["_volume_number"] = payload.get("volume_number")
        flattened.append(row)
    return flattened


def group_laws_by_pdf(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        key = entry.get("pdf_path") or entry.get("filename") or entry.get("name")
        if key:
            grouped[key].append(entry)
    return grouped


def _merge_pages(entries: list[dict[str, Any]], title: str) -> list[dict[str, Any]]:
    by_page: dict[int, str] = {}
    for entry in entries:
        for page in entry.get("pages", []) or []:
            page_number = int(page.get("page_number") or 0)
            raw_text = page.get("text", "") or ""
            cleaned = clean_page_text(raw_text, title=title)
            if len(cleaned) > len(by_page.get(page_number, "")):
                by_page[page_number] = cleaned
    return [
        {"page_number": page_number, "text": text}
        for page_number, text in sorted(by_page.items())
        if text
    ]


def canonicalize_group(entries: list[dict[str, Any]], fallback_txt_root: Path) -> dict[str, Any]:
    names = [entry.get("name", "") for entry in entries]
    title = canonical_title(names)
    first = entries[0]
    pages = _merge_pages(entries, title=title)

    fallback_txt_path: Path | None = None
    if not pages:
        fallback_txt_path = find_best_txt_fallback(
            fallback_root=fallback_txt_root,
            volume_type=first.get("_volume_type", ""),
            volume_number=str(first.get("_volume_number", "")),
            title=title,
            filename=first.get("filename", ""),
        )
        if fallback_txt_path and fallback_txt_path.exists():
            pages = [{"page_number": 1, "text": clean_page_text(fallback_txt_path.read_text(encoding="utf-8"), title=title)}]

    joined_text = "\n\n".join(page["text"] for page in pages if page.get("text"))
    year = derive_year(title, joined_text, first.get("filename", ""))
    act_no = derive_act_no(title, joined_text, first.get("filename", ""))
    law_id = make_law_id(first.get("pdf_path", ""), title, year)
    language = first.get("ocr_language") or "unknown"
    normalized_text = normalize_text_for_index(joined_text)

    return {
        "law_id": law_id,
        "title": title,
        "year": year,
        "act_no": act_no,
        "language": language,
        "volume": {
            "name": first.get("_volume_name"),
            "type": first.get("_volume_type"),
            "number": first.get("_volume_number"),
        },
        "pdf_path": first.get("pdf_path"),
        "filename": first.get("filename"),
        "url": first.get("url"),
        "pages": pages,
        "normalized_text": normalized_text,
        "source_fallback_txt": str(fallback_txt_path) if fallback_txt_path else None,
    }


def load_canonical_laws(config: AppConfig) -> list[dict[str, Any]]:
    fallback_txt_root = config.path("fallback_txt_root")
    raw_entries: list[dict[str, Any]] = []
    for volume_file in discover_volume_files(config):
        raw_entries.extend(flatten_raw_laws(volume_file))
    grouped = group_laws_by_pdf(raw_entries)
    canonical = [canonicalize_group(entries, fallback_txt_root=fallback_txt_root) for entries in grouped.values()]
    canonical.sort(key=lambda item: (item["volume"]["type"] or "", str(item["volume"]["number"] or ""), item["title"]))
    return canonical


def write_canonical_laws(config: AppConfig, laws: list[dict[str, Any]]) -> Path:
    processed_dir = ensure_dir(config.path("processed_dir"))
    output_path = processed_dir / "laws.jsonl"
    write_jsonl(output_path, laws)
    return output_path


def load_processed_laws(config: AppConfig) -> list[dict[str, Any]]:
    return list(iter_jsonl(config.path("processed_dir") / "laws.jsonl"))
