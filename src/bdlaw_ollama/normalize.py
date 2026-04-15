from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Iterable

from .utils import choose_best_title, compact_whitespace, normalize_digits, slugify


ENGLISH_HEADER_RE = re.compile(r"(?i)\bthe bangladesh code\b|\bvolume[- ]?[ivxlcdm0-9]+\b")
BANGLA_HEADER_RE = re.compile(r"বাংলাদেশ কোড|ভলিউম[- ]?[০-৯0-9ivxlcdmIVXLCDM]+")
FOOTER_RE = re.compile(r"^\s*[0-9০-৯ivxlcdmIVXLCDM]+\s*$")
TITLE_WITH_PAGE_RE = re.compile(r"^(.+?)\s+[0-9০-৯]+$")

YEAR_RE = re.compile(r"((?:19|20)[0-9]{2}|(?:১৯|২০)[০-৯]{2})")
ACT_NO_PATTERNS = [
    re.compile(r"(?i)\b(?:Act|Order|Ordinance|Law)\s+No\.?\s*([A-Z0-9 -]+?)\s+(?:of|,)\s*(?:19|20)\d{2}"),
    re.compile(r"(?i)\bPresident'?s\s+Order\s+No\.?\s*([A-Z0-9 -]+)"),
    re.compile(r"(?i)\bAct\s+No\.?\s*([IVXLCDM0-9 -]+)"),
    re.compile(r"([০-৯0-9]+)\s*নং\s*আইন"),
]


def normalize_text_for_index(text: str) -> str:
    text = normalize_digits(text)
    text = compact_whitespace(text)
    return text


def is_ocr_noise_line(line: str, title: str = "") -> bool:
    raw = line.strip(" \t|[]")
    if not raw:
        return False
    lowered = raw.lower()
    if title and raw == title.strip():
        return True
    if title and raw.startswith(title.strip()) and FOOTER_RE.search(raw.split()[-1]):
        return True
    if ENGLISH_HEADER_RE.search(lowered) or BANGLA_HEADER_RE.search(raw):
        return True
    if FOOTER_RE.fullmatch(raw):
        return True
    if TITLE_WITH_PAGE_RE.match(raw) and title and title.lower() in lowered:
        return True
    if raw in {"|", "||", "[]", "[", "]"}:
        return True
    return False


def clean_page_text(text: str, title: str = "") -> str:
    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            kept_lines.append("")
            continue
        if is_ocr_noise_line(line, title=title):
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return compact_whitespace(cleaned)


def derive_year(title: str, *fallback_texts: str) -> str | None:
    candidates = (title, *fallback_texts)
    for candidate in candidates:
        match = YEAR_RE.search(candidate or "")
        if match:
            return normalize_digits(match.group(1))
    return None


def derive_act_no(*texts: str) -> str | None:
    for text in texts:
        if not text:
            continue
        for pattern in ACT_NO_PATTERNS:
            match = pattern.search(text)
            if match:
                return normalize_digits(match.group(1).strip())
    return None


def canonical_title(names: Iterable[str]) -> str:
    return choose_best_title(names)


def volume_to_txt_dir(volume_type: str, volume_number: str) -> str:
    if volume_type == "roman":
        return f"volume_{volume_number.lower()}"
    return f"volume_{volume_number}"


def normalize_match_key(text: str) -> str:
    text = normalize_digits(text).lower()
    text = re.sub(r"(?i)\b(the|bangladesh|law|act|order|ordinance|president|s)\b", " ", text)
    text = re.sub(r"বাংলাদেশ|আইন|অর্ডার|অধ্যাদেশ", " ", text)
    text = re.sub(r"[^a-z0-9\u0980-\u09ff]+", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def find_best_txt_fallback(
    fallback_root: Path,
    volume_type: str,
    volume_number: str,
    title: str,
    filename: str = "",
) -> Path | None:
    target_dir = fallback_root / volume_to_txt_dir(volume_type, volume_number)
    if not target_dir.exists():
        return None

    title_key = normalize_match_key(title)
    filename_key = normalize_match_key(Path(filename).stem)
    best_path: Path | None = None
    best_score = 0.0

    for candidate in target_dir.glob("*.txt"):
        stem_key = normalize_match_key(candidate.stem)
        if not stem_key:
            continue
        score = max(
            difflib.SequenceMatcher(None, title_key, stem_key).ratio(),
            difflib.SequenceMatcher(None, filename_key, stem_key).ratio() if filename_key else 0.0,
        )
        if title_key and title_key in stem_key:
            score += 0.25
        if filename_key and filename_key in stem_key:
            score += 0.2
        if score > best_score:
            best_score = score
            best_path = candidate

    return best_path if best_score >= 0.45 else None


def make_law_id(pdf_path: str, title: str, year: str | None) -> str:
    seed = f"{pdf_path}|{title}|{year or 'unknown'}"
    return slugify(seed)[:80]
