from __future__ import annotations

import re
from bisect import bisect_right
from typing import Iterable

from .config import AppConfig
from .utils import compact_whitespace, normalize_digits, slugify, write_jsonl


PAGE_MARKER_RE = re.compile(r"\[\[PAGE=(\d+)\]\]")


def build_joined_text(pages: Iterable[dict]) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    markers: list[tuple[int, int]] = []
    cursor = 0
    for page in pages:
        marker = f"[[PAGE={page['page_number']}]]\n"
        markers.append((cursor, int(page["page_number"])))
        pieces.append(marker)
        cursor += len(marker)
        page_text = page.get("text", "").strip()
        pieces.append(page_text + "\n\n")
        cursor += len(page_text) + 2
    return "".join(pieces).strip(), markers


def page_for_offset(markers: list[tuple[int, int]], offset: int) -> int:
    starts = [position for position, _ in markers]
    index = bisect_right(starts, offset) - 1
    if index < 0:
        return markers[0][1] if markers else 1
    return markers[index][1]


def section_positions(text: str, patterns: list[str]) -> list[int]:
    starts: set[int] = set()
    for pattern in patterns:
        compiled = re.compile(pattern)
        starts.update(match.start() for match in compiled.finditer(text))
    filtered = sorted(position for position in starts if position < len(text))
    if filtered and filtered[0] != 0:
        filtered.insert(0, 0)
    return filtered or [0]


def strip_page_markers(text: str) -> str:
    return PAGE_MARKER_RE.sub("", text).strip()


def extract_section_id(text: str) -> tuple[str | None, str]:
    cleaned = strip_page_markers(text)
    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    patterns = [
        re.compile(r"(?i)^(?:section|article|chapter)\s+([A-Za-z0-9()./-]+)"),
        re.compile(r"^(?:ধারা|অনুচ্ছেদ|অধ্যায়)\s*([০-৯0-9A-Za-z()./-]+)?"),
        re.compile(r"^([০-৯0-9A-Za-z]+)\s*[।.)]"),
    ]
    for pattern in patterns:
        match = pattern.search(first_line)
        if match:
            return normalize_digits((match.group(1) or "").strip()) or None, first_line
    return None, first_line[:140]


def split_text_window(text: str, max_chars: int, overlap: int) -> list[str]:
    text = compact_whitespace(text)
    if len(text) <= max_chars:
        return [text]

    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start + max_chars // 2:
                end = split_at
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [window for window in windows if window]


def build_law_chunks(law: dict, rag_config: dict) -> list[dict]:
    joined_text, markers = build_joined_text(law.get("pages", []))
    if not joined_text.strip():
        return []

    chunk_cfg = rag_config["chunking"]
    positions = section_positions(joined_text, chunk_cfg["section_markers"])
    if positions[-1] != len(joined_text):
        positions.append(len(joined_text))

    chunks: list[dict] = []
    for index in range(len(positions) - 1):
        start = positions[index]
        end = positions[index + 1]
        raw_section = joined_text[start:end].strip()
        if not raw_section:
            continue

        section_id, section_title = extract_section_id(raw_section)
        cleaned_section = compact_whitespace(strip_page_markers(raw_section))
        if not cleaned_section:
            continue

        page_start = page_for_offset(markers, start)
        page_end = page_for_offset(markers, max(start, end - 1))
        windows = split_text_window(
            cleaned_section,
            max_chars=int(chunk_cfg["max_chars"]),
            overlap=int(chunk_cfg["overlap"]),
        )
        for window_index, window in enumerate(windows, start=1):
            citation = law["title"]
            if section_id:
                citation += f", Section {section_id}"
            elif section_title:
                citation += f", {section_title[:80]}"
            citation += f", p. {page_start}" if page_start == page_end else f", pp. {page_start}-{page_end}"
            chunks.append(
                {
                    "chunk_id": slugify(f"{law['law_id']}-{section_id or 'section'}-{index+1}-{window_index}"),
                    "law_id": law["law_id"],
                    "law_title": law["title"],
                    "year": law.get("year"),
                    "act_no": law.get("act_no"),
                    "language": law.get("language"),
                    "section_id": section_id,
                    "section_title": section_title,
                    "page_start": page_start,
                    "page_end": page_end,
                    "text": window,
                    "citation_label": citation,
                }
            )

    if chunks:
        return chunks

    fallback_windows = split_text_window(
        strip_page_markers(joined_text),
        max_chars=int(chunk_cfg["max_chars"]),
        overlap=int(chunk_cfg["overlap"]),
    )
    return [
        {
            "chunk_id": slugify(f"{law['law_id']}-fallback-{idx}"),
            "law_id": law["law_id"],
            "law_title": law["title"],
            "year": law.get("year"),
            "act_no": law.get("act_no"),
            "language": law.get("language"),
            "section_id": None,
            "section_title": "fallback chunk",
            "page_start": 1,
            "page_end": law.get("pages", [{}])[-1].get("page_number", 1),
            "text": text,
            "citation_label": f"{law['title']}, pp. 1-{law.get('pages', [{}])[-1].get('page_number', 1)}",
        }
        for idx, text in enumerate(fallback_windows, start=1)
    ]


def build_chunks(laws: list[dict], rag_config: dict) -> list[dict]:
    rows: list[dict] = []
    for law in laws:
        rows.extend(build_law_chunks(law, rag_config=rag_config))
    return rows


def write_chunks(config: AppConfig, chunks: list[dict]) -> None:
    output_path = config.path("processed_dir") / "chunks.jsonl"
    write_jsonl(output_path, chunks)
